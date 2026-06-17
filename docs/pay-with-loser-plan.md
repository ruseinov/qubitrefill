# Plan: "Pay with your worst loser" Bitrefill skill

## Context

`demo.py` demonstrates the financial half of the idea: create a portfolio agent, optimize it,
then use the same μ (expected-return) estimator the solver uses to find the **worst-performing
asset** and retune the basket without it.

This turns that into a reusable **skill** that closes a different loop: instead of just dropping
the loser, **spend it** — pay for a real Bitrefill product (gift card / top-up / eSIM) using the
worst-performing crypto in the portfolio, then retune to drop that asset.

"Possible currencies" = the portfolio's holdings ∩ the crypto currencies Bitrefill accepts as
payment. Stocks (IONQ, GOOGL, …) can never pay an invoice; some held coins may not be supported
either.

### Decisions

- **Buy + retune** — after purchase, call `/optimize` without the spent asset (full demo.py loop).
- **New backend endpoint** — add `GET /agents/{id}/market` exposing per-asset μ + holdings, so the
  skill stays pure-HTTP (today the HTTP API has no market endpoint; demo.py reaches into backend
  internals, which a skill shouldn't have to do).
- **Confirm currency support at runtime** — a static ticker→payment-method map provides
  candidates; each candidate is probed against Bitrefill before it can be chosen.

## Part 1 — Backend: `GET /agents/{agent_id}/market`

Exposes the data demo.py computes by importing backend internals.

**`backend/src/backend/api/schemas.py`** — add two schemas (camelCase wire aliases, matching the
file's convention):

```python
class MarketAsset(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass = Field(alias="assetClass")   # "crypto" | "stock"
    mu: float                                             # annualised hourly expected return
    units: float                                          # held token units (0 if none yet)
    usd: float                                            # units × spot
    model_config = ConfigDict(populate_by_name=True)

class MarketResult(BaseModel):
    agent_id: str = Field(alias="agentId")
    assets: list[MarketAsset]
    model_config = ConfigDict(populate_by_name=True)
```
(`AssetClass` imported from `..financial.basket`.)

**`backend/src/backend/api/routes.py`** — add the route, reusing exactly the estimators/sources
demo.py uses (`expected_return`, `get_source`, `config.SIGMA_WINDOW_HOURS`,
`config.MU_WINDOW_HOURS`) plus `get_asset` for name/class and `spot_prices` for USD:

```python
@router.get("/agents/{agent_id}/market", response_model=MarketResult)
async def market(agent_id: str) -> MarketResult:
    record = get_agent_store().get(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return await asyncio.to_thread(_compute_market, record)
```
`_compute_market` (sync, off the event loop like `run_optimization`):
- `tickers = list(record.holdings_units) or validate_basket(record.assets)` — prefer the agent's
  *current* holdings; fall back to its basket if it hasn't solved yet.
- `returns = source.hourly_returns(tickers, config.SIGMA_WINDOW_HOURS)`;
  `mu = expected_return(returns, config.MU_WINDOW_HOURS)`.
- `spot = source.spot_prices(tickers)`; per ticker build `MarketAsset` with
  `units = record.holdings_units.get(t, 0.0)`, `usd = units * spot[t]`, name/class from
  `get_asset(t)`.

**`backend/tests/test_api.py`** — add a test: create agent → optimize → `GET …/market` returns
200 with one `MarketAsset` per held ticker, each with numeric `mu` and `assetClass` in
`{"crypto","stock"}`. Follow existing fixtures / `TestClient` patterns.

## Part 2 — New skill: `skills/pay-with-loser/SKILL.md`

A thin orchestration skill that **delegates purchase mechanics to the existing
[`bitrefill`](../bitrefill/SKILL.md) skill** (no duplicated MCP/CLI/API routing) and adds the
portfolio logic on top. Frontmatter `name`/`description`/`compatibility` in the same style as
`skills/bitrefill/SKILL.md`; triggers when the user asks to "pay/checkout with my worst performer
/ biggest loser / dump a losing coin on a gift card", etc.

Flow documented in the body:

1. **Pick the product** — bitrefill skill's search → `product-details` to settle on product +
   denomination (package_id) + price.
2. **Get the portfolio** — `GET {BASE}/agents/{id}/market` (BASE default
   `http://127.0.0.1:8000`). No agent yet → point user at `demo.py` / `POST /agents`.
3. **Build candidate currencies** — keep assets where `assetClass == "crypto"` **and**
   `units > 0`; map each ticker to a Bitrefill `payment_method` via a static table in the skill,
   e.g. `BTC→bitcoin`, `ETH→ethereum`, `USDC→usdc_base`, `USDT→usdt`, `SOL→solana`,
   `DOGE→dogecoin`, `BNB→bnb`, … Tickers absent from the table (HYPE, STRK, RENDER, …) are
   dropped as not-spendable.
4. **Confirm support at runtime** — for each remaining candidate, probe Bitrefill
   (`product-details` with that currency / accepted payment methods for the product); drop any the
   product can't be paid in. This is why the static map alone isn't trusted.
5. **Pick the worst loser** — among *confirmed* candidates, choose `min(mu)`. If none survive, tell
   the user no held coin can pay this product and stop (do not silently fall back to a stablecoin).
6. **Confirm + buy** — per bitrefill safeguards, present product, denomination, price, and chosen
   pay currency (with its μ, "your worst performer") and wait for explicit approval. Then
   `buy-products(cart_items=[…], payment_method=<loser method>, return_payment_link=true)` → pay
   via returned link → poll `get-invoice-by-id` to `complete` → `get-order-by-id` for the
   redemption code/QR. Log `invoice_id`, product, amount, method.
7. **Retune (close the loop)** — `POST {BASE}/agents/{id}/optimize` with
   `assets = currentBasket − spentTicker` (demo.py step 4). Report sold/kept (demo.py step 5).

Include a short worked example and a Safeguards note deferring to
`skills/bitrefill/references/safeguards.md` (real money, codes are cash, dedicated low-balance
account, log every purchase).

## Part 3 — Wire-up docs

- **`AGENTS.md`** — add a short "Pay with worst loser" subsection under Skills pointing at
  `skills/pay-with-loser/SKILL.md` and noting the new `GET /agents/{id}/market` endpoint.

## Files

| File | Change |
|------|--------|
| `backend/src/backend/api/schemas.py` | add `MarketAsset`, `MarketResult` |
| `backend/src/backend/api/routes.py` | add `GET /agents/{id}/market` + `_compute_market` helper |
| `backend/tests/test_api.py` | add endpoint test |
| `skills/pay-with-loser/SKILL.md` | **new** orchestration skill |
| `AGENTS.md` | mention new skill + endpoint |

## Verification

1. **Endpoint** — `cd backend && MARKET_DATA_SOURCE=synthetic uv run pytest tests/test_api.py`.
2. **Live** — start backend (`MARKET_DATA_SOURCE=synthetic uvicorn backend.api.app:app --reload
   --workers 1`), run `uv run python ../demo.py` to mint an agent, then
   `curl http://127.0.0.1:8000/agents/<id>/market`; cross-check `min(mu)` crypto matches demo.py's
   "worst loser" when restricted to crypto.
3. **Skill dry-run** — exercise bitrefill MCP search/`product-details` for a cheap product, walk
   steps 3–5 to confirm the chosen pay currency is the worst μ among Bitrefill-supported held
   coins. Stop before `buy-products` unless the user opts into a real purchase.