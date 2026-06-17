"""API surface via FastAPI TestClient: HTTP flow, 404s, and a WS push."""

from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app

requires_gurobi = pytest.mark.skipif(
    importlib.util.find_spec("gurobipy") is None, reason="gurobipy not installed"
)

_SLIDERS = {"rebalanceFrequency": 50, "riskPreference": 70, "maxPositionSize": 50}
_BASKET = ["BTC", "ETH", "IONQ", "QBTS"]


def _create(client: TestClient, name: str = "Neo") -> str:
    response = client.post(
        "/agents",
        json={
            "name": name,
            "email": f"{name.lower()}@example.com",
            "sliders": _SLIDERS,
            "assets": _BASKET,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["bankroll"] == 10_000.0
    assert body["qrUrl"].endswith(f"/p/{body['agentId']}")
    return body["agentId"]


def test_create_and_get_agent():
    with TestClient(create_app()) as client:
        agent_id = _create(client)
        got = client.get(f"/agents/{agent_id}")
        assert got.status_code == 200
        body = got.json()
        assert body["sliders"]["riskPreference"] == 70
        assert body["assets"] == _BASKET


def test_unknown_agent_is_404():
    with TestClient(create_app()) as client:
        assert client.get("/agents/nope").status_code == 404
        assert client.post("/agents/nope/optimize", json={}).status_code == 404


def test_leaderboard_lists_created_agents():
    with TestClient(create_app()) as client:
        agent_id = _create(client)
        board = client.get("/leaderboard").json()
        assert any(entry["agentId"] == agent_id for entry in board)


@requires_gurobi
def test_optimize_returns_routing_result():
    with TestClient(create_app()) as client:
        agent_id = _create(client)
        response = client.post(f"/agents/{agent_id}/optimize", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "first"
        assert body["providerType"] in ("QPU", "CPU")
        assert {entry["ticker"] for entry in body["portfolio"]} == set(_BASKET)
        assert sum(entry["pct"] for entry in body["portfolio"]) == pytest.approx(100.0, abs=1e-6)


@requires_gurobi
def test_websocket_streams_agent_update():
    with TestClient(create_app()) as client:
        agent_id = _create(client)
        client.post(f"/agents/{agent_id}/optimize", json={})
        with client.websocket_connect(f"/agents/{agent_id}") as socket:
            client.post(f"/agents/{agent_id}/optimize", json={})  # retune → push
            update = socket.receive_json()
            assert {"plUSD", "plPct", "total"} <= set(update)


def test_basket_below_minimum_is_rejected():
    with TestClient(create_app()) as client:
        response = client.post(
            "/agents",
            json={"name": "Tiny", "sliders": _SLIDERS, "assets": ["BTC"]},
        )
        assert response.status_code == 422
        assert "at least" in response.json()["detail"]


@requires_gurobi
def test_optimize_accepts_a_new_basket():
    with TestClient(create_app()) as client:
        agent_id = _create(client)
        client.post(f"/agents/{agent_id}/optimize", json={})
        response = client.post(
            f"/agents/{agent_id}/optimize", json={"assets": ["HON", "GOOGL", "IBM"]}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "retune"
        assert {e["ticker"] for e in body["portfolio"]} == {"HON", "GOOGL", "IBM"}

        too_small = client.post(f"/agents/{agent_id}/optimize", json={"assets": ["BTC"]})
        assert too_small.status_code == 422
