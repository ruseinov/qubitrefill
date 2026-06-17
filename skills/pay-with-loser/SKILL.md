---
name: pay-with-loser
description: "Pay for a Bitrefill product (gift card, top-up, eSIM) using the worst-performing crypto in the agent's portfolio — identified by lowest expected return μ — then retune the portfolio without it. Triggers when the user says 'pay with my worst performer', 'dump my biggest loser on a gift card', 'use my worst crypto to buy X', or similar."
compatibility: "Requires: (1) a running portfolio backend at http://127.0.0.1:8000 with an existing agent; (2) Bitrefill MCP (https://api.bitrefill.com/mcp) or CLI available. Delegates all purchase mechanics to the bitrefill skill."
metadata:
  author: hackathon
  version: "1.0.0"
---

# Pay with Your Worst Loser

Identify the worst-performing crypto in the portfolio (lowest annualised expected return μ), spend it on a Bitrefill product, then retune the portfolio without it.

Delegates all purchase mechanics to the [`bitrefill`](../bitrefill/SKILL.md) skill — read and invoke that skill for product search, pricing, buying, and payment polling. This skill adds the portfolio selection logic on top.

## Flow

### 1. Pick the product

Use the bitrefill skill's `search-products` → `product-details` to settle on:
- Product name + country
- Denomination (package_id)
- Price in USD

### 2. Get the portfolio

```
GET http://127.0.0.1:8000/agents/{agent_id}/market
```

Response shape:
```json
{
  "agentId": "abc12345",
  "assets": [
    {"ticker": "BTC", "name": "Bitcoin", "assetClass": "crypto", "mu": 0.0012, "units": 0.05, "usd": 3200.0},
    {"ticker": "ETH", "name": "Ethereum", "assetClass": "crypto", "mu": -0.0003, "units": 1.2, "usd": 3600.0},
    ...
  ]
}
```

If no agent exists yet, ask the user to create one:
```bash
uv run python demo.py           # from the repo root
# or POST /agents then POST /agents/{id}/optimize
```

### 3. Build candidate currencies

Keep only assets where **both**:
- `assetClass == "crypto"`
- `units > 0` (actually held)

Map each to a Bitrefill payment method via this static table:

| Ticker | Bitrefill payment_method |
|--------|--------------------------|
| BTC    | bitcoin                  |
| ETH    | ethereum                 |
| BNB    | bnb                      |
| SOL    | solana                   |
| XRP    | ripple                   |
| USDT   | usdt                     |
| USDC   | usdc_base                |
| DOGE   | dogecoin                 |
| ZEC    | zcash                    |
| ALGO   | algorand                 |
| FIL    | filecoin                 |

Tickers absent from this table (HYPE, STRK, RENDER, …) are not spendable on Bitrefill — drop them silently.

### 4. Confirm support at runtime

For each remaining candidate, probe Bitrefill by calling `product-details` with that `payment_method` for the chosen product. Drop any the product cannot be paid in. The static table is a cheap first filter; per-product restrictions must be confirmed live.

### 5. Pick the worst loser

Among confirmed candidates, choose `min(mu)`.

If none survive: tell the user no held coin can pay this product and stop — do **not** silently fall back to a stablecoin.

### 6. Confirm + buy

Present to the user and wait for explicit approval:
```
Product:   [name] — [denomination]
Price:     $[amount]
Pay with:  [TICKER] ([payment_method]) — your worst performer (μ = [mu])
```

After approval, use the bitrefill skill to:
1. `buy-products(cart_items=[{package_id, quantity: 1}], payment_method=<loser_method>, return_payment_link=true)`
2. Pay via the returned payment link
3. Poll `get-invoice-by-id` until status = `complete`
4. Call `get-order-by-id` for the redemption code / QR

Log: `invoice_id`, product, amount, payment method.

### 7. Retune (close the loop)

Remove the spent ticker from the agent's basket and re-optimize:

```
POST http://127.0.0.1:8000/agents/{agent_id}/optimize
{
  "assets": [<currentBasket minus spentTicker>]
}
```

Report the new allocation (what was kept, what was dropped).

## Worked example

> "Buy a $25 Amazon gift card and dump my worst crypto"

1. `search-products("Amazon gift card", country="US")` → find Amazon US package_id
2. `GET /agents/abc12345/market` → holdings: BTC (μ=0.0014), ETH (μ=−0.0008), DOGE (μ=−0.0021, units=150)
3. Candidates after static filter: BTC→bitcoin, ETH→ethereum, DOGE→dogecoin
4. Runtime check: Amazon US gift cards accept bitcoin, ethereum, dogecoin ✓
5. Worst loser: DOGE (μ=−0.0021)
6. Present: "Amazon US $25 · pay with DOGE (μ=−0.0021) — your worst performer · Approve?"
7. User approves → buy → pay → poll → redeem code
8. Retune: `POST /agents/abc12345/optimize {"assets": ["BTC", "ETH", ...]}`

## Safeguards

This skill executes real-money purchases. See [`skills/bitrefill/references/safeguards.md`](../bitrefill/references/safeguards.md) for the full spending policy:
- Confirm before every purchase — no autonomous buying without explicit opt-in
- Treat codes as cash — never log or paste redemption codes in public channels
- Use a dedicated low-balance account
- Log every purchase: invoice_id, product, amount, method

The retune in step 7 is irreversible — the spent asset is removed from the basket permanently until the user re-adds it manually.
