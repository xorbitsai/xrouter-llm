import pytest
from starlette.testclient import TestClient

from xrouter_llm import (
    BenchmarkProfileCatalog,
    CallStore,
    ModelBenchmarkProfile,
    ModelPrediction,
    RoutingService,
    load_router_configs,
)


class _StubPredictor:
    """Routes by a fixed per-model completion probability; no training needed."""

    def __init__(self) -> None:
        self.mus = {"cheap": 0.80, "strong": 0.95}
        self.seen_tasks = []

    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None, task=None):
        self.seen_tasks.append(task)
        return [
            ModelPrediction(
                model_id=m,
                mu=self.mus.get(m, 0.5),
                sigma=0.03,
                cost=0.0 if costs is None else float(costs.get(m, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(m, 0.0)),
            )
            for m in tuple(model_ids)
        ]


class _LegacyPredictor(_StubPredictor):
    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None):
        return [
            ModelPrediction(
                model_id=m,
                mu=self.mus.get(m, 0.5),
                sigma=0.03,
                cost=0.0 if costs is None else float(costs.get(m, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(m, 0.0)),
            )
            for m in tuple(model_ids)
        ]


def _service(tmp_path):
    profiles = BenchmarkProfileCatalog(
        [
            ModelBenchmarkProfile("cheap", input_cost_per_1k=0.0001, output_cost_per_1k=0.0002),
            ModelBenchmarkProfile("strong", input_cost_per_1k=0.005, output_cost_per_1k=0.015),
        ]
    )
    routers = tmp_path / "routers"
    routers.mkdir()
    (routers / "auto.yaml").write_text(
        "name: auto\ncompletion_threshold: 0.7\nlambda_cost: 1.0\nmodels: [cheap, strong]\n",
        encoding="utf-8",
    )
    configs = load_router_configs(routers)
    store = CallStore(tmp_path / "calls.db")
    return RoutingService(_StubPredictor(), profiles=profiles, configs=configs, store=store)


def _legacy_service(tmp_path):
    profiles = BenchmarkProfileCatalog(
        [
            ModelBenchmarkProfile("cheap", input_cost_per_1k=0.0001, output_cost_per_1k=0.0002),
            ModelBenchmarkProfile("strong", input_cost_per_1k=0.005, output_cost_per_1k=0.015),
        ]
    )
    routers = tmp_path / "routers"
    routers.mkdir()
    (routers / "auto.yaml").write_text(
        "name: auto\ncompletion_threshold: 0.7\nlambda_cost: 1.0\nmodels: [cheap, strong]\n",
        encoding="utf-8",
    )
    configs = load_router_configs(routers)
    store = CallStore(tmp_path / "calls.db")
    return RoutingService(_LegacyPredictor(), profiles=profiles, configs=configs, store=store)


def test_router_config_parses_quality_pair(tmp_path) -> None:
    routers = tmp_path / "routers"
    routers.mkdir()
    (routers / "quality-pair.yaml").write_text(
        "\n".join(
            [
                "name: quality-pair",
                "completion_threshold: 0.8",
                "max_k: 2",
                "models: [cheap, strong]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_router_configs(routers)["quality-pair"]

    assert config.completion_threshold == 0.8
    assert config.max_k == 2


def test_route_picks_cheapest_capable_and_records(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.route("write a function", models=["cheap", "strong"], task="coding")

    # both clear the 0.7 threshold -> cheapest wins
    assert result["selected"] == ["cheap"]
    assert service.predictor.seen_tasks == ["coding"]
    assert result["cost"] > 0.0
    history = service.store.recent()
    assert len(history) == 1
    assert history[0]["selected"] == ["cheap"]
    assert history[0]["config"] == "custom"


def test_route_defaults_to_all_registered_models(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.route("write a function")

    assert result["selected"] == ["cheap"]
    history = service.store.recent()
    assert history[0]["config"] == "all"


def test_route_supports_predictors_without_task_parameter(tmp_path) -> None:
    service = _legacy_service(tmp_path)
    result = service.route("write a function", models=["cheap", "strong"], task="coding")

    assert result["selected"] == ["cheap"]


def test_route_via_config_name(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.route("write a function", config_name="auto")

    assert result["selected"] == ["cheap"]
    history = service.store.recent()
    assert history[0]["config"] == "auto"


def test_route_rejects_unknown_config_name(tmp_path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="unknown router config"):
        service.route("hello", config_name="nonexistent")


def test_route_rejects_empty_prompt(tmp_path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError):
        service.route("   ", models=["cheap", "strong"])


def test_http_endpoints_end_to_end(tmp_path) -> None:
    from xrouter_llm.server import create_app

    service = _service(tmp_path)
    client = TestClient(create_app(service))

    configs = client.get("/api/configs").json()
    assert configs["configs"][0]["name"] == "auto"

    routed = client.post(
        "/api/route",
        json={"prompt": "hello", "models": ["cheap", "strong"]},
    ).json()
    assert routed["selected"] == ["cheap"]

    history = client.get("/api/history").json()
    assert history["total"] == 1
    assert len(history["calls"]) == 1

    call_id = history["calls"][0]["id"]
    assert client.delete(f"/api/calls/{call_id}").json() == {"deleted": call_id}
    assert client.get("/api/history").json()["total"] == 0
    assert client.delete(f"/api/calls/{call_id}").status_code == 404


def test_feedback_endpoint(tmp_path) -> None:
    from xrouter_llm.server import create_app

    service = _service(tmp_path)
    client = TestClient(create_app(service))

    routed = client.post("/api/route", json={"prompt": "hello", "models": ["cheap", "strong"]}).json()
    call_id = routed["id"]

    # submit good feedback
    r = client.patch(f"/api/calls/{call_id}/feedback", json={"outcome": "good"})
    assert r.status_code == 200
    assert r.json()["feedback"]["outcome"] == "good"

    # verify it's persisted in history
    history = client.get("/api/history").json()
    assert history["calls"][0]["feedback"]["outcome"] == "good"

    # update to bad with correct_model
    r = client.patch(f"/api/calls/{call_id}/feedback",
                     json={"outcome": "bad", "correct_model": "strong"})
    assert r.status_code == 200
    assert r.json()["feedback"]["correct_model"] == "strong"

    # retract: feedback becomes None
    r = client.patch(f"/api/calls/{call_id}/feedback", json={"outcome": "retracted"})
    assert r.status_code == 200
    assert r.json()["feedback"] is None
    assert client.get("/api/history").json()["calls"][0]["feedback"] is None

    # 404 for unknown id
    assert client.patch("/api/calls/9999/feedback", json={"outcome": "good"}).status_code == 404

    # correct_model is only valid when outcome is 'bad'
    assert client.patch(f"/api/calls/{call_id}/feedback",
                        json={"outcome": "good", "correct_model": "strong"}).status_code == 422


def test_user_id_routing_and_history_filter(tmp_path) -> None:
    from xrouter_llm.server import create_app

    service = _service(tmp_path)
    client = TestClient(create_app(service))

    client.post("/api/route", json={"prompt": "hello", "models": ["cheap", "strong"], "user_id": "alice"})
    client.post("/api/route", json={"prompt": "world", "models": ["cheap", "strong"], "user_id": "bob"})
    client.post("/api/route", json={"prompt": "anon",  "models": ["cheap", "strong"]})

    assert client.get("/api/history").json()["total"] == 3
    assert client.get("/api/history?user_id=alice").json()["total"] == 1
    assert client.get("/api/history?user_id=bob").json()["total"] == 1

    alice_calls = client.get("/api/history?user_id=alice").json()["calls"]
    assert alice_calls[0]["user_id"] == "alice"


def test_history_pagination(tmp_path) -> None:
    from xrouter_llm.server import create_app

    service = _service(tmp_path)
    client = TestClient(create_app(service))

    for i in range(5):
        client.post("/api/route", json={"prompt": f"prompt {i}", "models": ["cheap", "strong"]})

    page1 = client.get("/api/history?limit=3&offset=0").json()
    assert page1["total"] == 5
    assert len(page1["calls"]) == 3

    page2 = client.get("/api/history?limit=3&offset=3").json()
    assert len(page2["calls"]) == 2
