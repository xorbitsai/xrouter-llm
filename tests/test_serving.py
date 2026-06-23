import json
import threading
import urllib.request

from xrouter_llm import (
    BenchmarkProfileCatalog,
    CallStore,
    ModelBenchmarkProfile,
    ModelPrediction,
    RoutingService,
    load_router_configs,
)
from xrouter_llm.server import run_server


class _StubPredictor:
    """Routes by a fixed per-model completion probability; no training needed."""

    def __init__(self) -> None:
        self.mus = {"cheap": 0.80, "strong": 0.95}
        self.seen_tasks = []

    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None, task=None):
        self.seen_tasks.append(task)
        ids = tuple(model_ids)
        return [
            ModelPrediction(
                model_id=m,
                mu=self.mus.get(m, 0.5),
                sigma=0.03,
                cost=0.0 if costs is None else float(costs.get(m, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(m, 0.0)),
            )
            for m in ids
        ]


class _LegacyPredictor(_StubPredictor):
    def predict(self, prompt, *, model_ids=None, costs=None, latencies=None):
        ids = tuple(model_ids)
        return [
            ModelPrediction(
                model_id=m,
                mu=self.mus.get(m, 0.5),
                sigma=0.03,
                cost=0.0 if costs is None else float(costs.get(m, 0.0)),
                latency=0.0 if latencies is None else float(latencies.get(m, 0.0)),
            )
            for m in ids
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
    result = service.route("write a function", config_name="auto", task="coding")

    # both clear the 0.7 threshold -> cheapest wins
    assert result["selected"] == ["cheap"]
    assert service.predictor.seen_tasks == ["coding"]
    assert result["cost"] > 0.0
    history = service.store.recent()
    assert len(history) == 1
    assert history[0]["selected"] == ["cheap"]
    assert history[0]["config"] == "auto"


def test_route_supports_predictors_without_task_parameter(tmp_path) -> None:
    service = _legacy_service(tmp_path)
    result = service.route("write a function", config_name="auto", task="coding")

    assert result["selected"] == ["cheap"]


def test_route_rejects_empty_prompt_and_unknown_config(tmp_path) -> None:
    service = _service(tmp_path)
    try:
        service.route("   ", config_name="auto")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        service.route("hi", config_name="nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_http_endpoints_end_to_end(tmp_path) -> None:
    service = _service(tmp_path)
    from xrouter_llm.server import create_handler
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        configs = json.loads(urllib.request.urlopen(base + "/api/configs").read())
        assert configs["configs"][0]["name"] == "auto"

        req = urllib.request.Request(
            base + "/api/route",
            data=json.dumps({"prompt": "hello", "config": "auto"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        routed = json.loads(urllib.request.urlopen(req).read())
        assert routed["selected"] == ["cheap"]

        history = json.loads(urllib.request.urlopen(base + "/api/history").read())
        assert len(history["calls"]) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
