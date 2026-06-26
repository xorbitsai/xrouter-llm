"""CallStore integration tests, parameterized over DB backends via conftest.db_url."""
from __future__ import annotations


def test_record_and_recent(store) -> None:
    call_id = store.record(
        ts=1_000_000.0,
        config="all",
        prompt="hello",
        task=None,
        selected=["cheap"],
        candidates=[{"model_id": "cheap", "mu": 0.8, "sigma": 0.03, "cost": 0.001}],
        expected_quality=0.8,
        cost=0.001,
        latency=0.0,
    )
    assert isinstance(call_id, int)

    rows = store.recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == call_id
    assert rows[0]["selected"] == ["cheap"]
    assert rows[0]["candidates"][0]["model_id"] == "cheap"
    assert rows[0]["config"] == "all"


def test_recent_order_and_limit(store) -> None:
    for i in range(5):
        store.record(
            ts=float(i),
            config="all",
            prompt=f"prompt {i}",
            task=None,
            selected=["model-a"],
            candidates=[],
            expected_quality=0.8,
            cost=0.001,
            latency=0.0,
        )
    rows = store.recent(limit=3)
    assert len(rows) == 3
    # most-recent first
    assert rows[0]["ts"] > rows[1]["ts"]


def test_json_roundtrip(store) -> None:
    candidates = [
        {"model_id": "a", "mu": 0.9, "sigma": 0.02, "cost": 0.005},
        {"model_id": "b", "mu": 0.6, "sigma": 0.05, "cost": 0.001},
    ]
    store.record(
        ts=1.0, config="custom", prompt="p", task="coding",
        selected=["a"], candidates=candidates,
        expected_quality=0.9, cost=0.005, latency=0.0,
    )
    row = store.recent(1)[0]
    assert row["task"] == "coding"
    assert row["candidates"] == candidates
    assert row["selected"] == ["a"]


def test_model_counts(store) -> None:
    for model in ["cheap", "cheap", "strong"]:
        store.record(
            ts=1.0, config="all", prompt="p", task=None,
            selected=[model], candidates=[],
            expected_quality=0.8, cost=0.0, latency=0.0,
        )
    counts = store.model_counts()
    assert counts["cheap"] == 2
    assert counts["strong"] == 1
