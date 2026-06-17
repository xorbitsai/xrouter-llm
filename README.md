# xrouter-llm

`xrouter-llm` implements the routing invariant from the design doc:

```text
Do not train:
  prompt -> selected model

Train:
  prompt + model -> expected model quality

Then decide:
  predicted quality + uncertainty + cost + latency -> selected model set
```

The package has three separate layers:

- `ModelAwareRouterPredictor`: learns `P(model can complete prompt)` from `(prompt features, model benchmark profile)`.
- `RoutingPolicy`: converts quality, uncertainty, cost, and latency into a selected model set.
- `XRouter`: ties the predictor and policy together for top-1 routing or best-of-k fusion selection.

## Install

```bash
pip install -e ".[dev]"
```

## Training Data

Rows should keep the full score vector across models:

```json
{
  "prompt_id": "p1",
  "prompt": "Refactor this Python state machine",
  "model_id": "claude",
  "score": 0.92,
  "cost_usd": 0.012,
  "latency_s": 3.4,
  "task": "coding"
}
```

Dense rows are converted into completion labels during training:

```text
(prompt, model) -> score >= completion_score_threshold
```

Prediction returns completion probability as `mu`. The policy selects the
cheapest model or model set whose predicted completion probability reaches
`completion_threshold`; if none qualify, it falls back to the highest predicted
completion probability.

## Minimal Usage

```python
from xrouter_llm import (
    BenchmarkRow,
    ModelAwareRouterPredictor,
    ModelProfile,
    PolicyParams,
    XRouter,
)

rows = [
    BenchmarkRow("p1", "Refactor this Python state machine", "claude", 0.92),
    BenchmarkRow("p1", "Refactor this Python state machine", "gpt", 0.88),
    BenchmarkRow("p2", "Solve this calculus problem", "claude", 0.80),
    BenchmarkRow("p2", "Solve this calculus problem", "gpt", 0.91),
]

predictor = ModelAwareRouterPredictor(ensemble_size=8, random_state=7).fit(rows)
router = XRouter(
    predictor,
    model_profiles=[
        ModelProfile("claude", input_cost_per_1k=0.003, output_cost_per_1k=0.015, base_latency_s=2.0),
        ModelProfile("gpt", input_cost_per_1k=0.002, output_cost_per_1k=0.010, base_latency_s=1.6),
    ],
)

decision = router.route(
    "Write a clean Python parser for this log format",
    policy_params=PolicyParams(max_k=2, allow_fusion=True, lambda_cost=1.0),
)

print(decision.selected_model_ids)
print(decision.utility_breakdown)
```

`len(decision.selected_model_ids) == 1` is normal routing.
`len(decision.selected_model_ids) > 1` means call the selected models in parallel and fuse their answers.

The package intentionally does not call model vendors. For fusion, use the selected
model ids to call your providers, then construct a judge/synthesizer prompt:

```python
from xrouter_llm import build_fusion_prompt

fusion_prompt = build_fusion_prompt(
    prompt="Write a clean Python parser for this log format",
    answers={
        "claude": "...",
        "gpt": "...",
    },
)
```

## Offline Evaluation

```python
from xrouter_llm import PolicyParams, evaluate_offline, load_jsonl

rows = load_jsonl("examples/benchmark.jsonl")
result = evaluate_offline(
    rows,
    policy_params=PolicyParams(max_k=2, allow_fusion=True),
    test_size=0.25,
    random_state=13,
)

print(result.metrics)
print(result.route_distribution)
```

For fusion decisions, offline quality is reported as a **best-of-k upper bound**:

```text
actual_quality = max(actual_score(m) for m in selected_models)
```

This is not the same thing as measured fused-answer quality.

## Real RouterBench Training

The default real-data path uses
[`withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench),
which contains 30K+ prompts, responses from 11 LLMs, response costs, and correctness scores.

Download the 0-shot split:

```bash
xrouter-llm download-routerbench --split 0shot --output-dir data/raw
```

Train and save a model-aware predictor from RouterBench plus built-in public benchmark profiles:

```bash
xrouter-llm train-routerbench \
  --split 0shot \
  --output artifacts/models/routerbench_0shot.joblib \
  --metrics-output artifacts/models/routerbench_0shot.metrics.json \
  --ensemble-size 8 \
  --completion-score-threshold 0.75 \
  --completion-threshold 0.7 \
  --policy-max-k 2 \
  --allow-fusion
```

For a quick smoke test on real data:

```bash
xrouter-llm train-routerbench --split 0shot --max-prompts 1000 --ensemble-size 4
```

If `--max-prompts` is omitted, the command trains on the full selected RouterBench split.

Tune the capability threshold before using the router in production:

```bash
xrouter-llm sweep-routerbench \
  --input data/raw/routerbench_0shot.pkl \
  --max-prompts 1000 \
  --ensemble-size 4 \
  --completion-score-threshold 0.75 \
  --thresholds 0.5,0.6,0.7,0.8,0.9 \
  --output artifacts/reports/routerbench_0shot_threshold_sweep.json
```

The sweep trains once, then replays the held-out predictions for each
`completion_threshold`. The report shows completion rate, average cost,
completed-only average cost, oracle cheapest-success cost, route distribution,
and prediction calibration. Use it to choose the cheapest threshold that still
meets your target completion rate.

For full training without the extra held-out evaluation pass:

```bash
xrouter-llm train-routerbench --split 0shot --ensemble-size 4 --skip-eval
```

The built-in profile file is packaged at:

```text
src/xrouter_llm/resources/routerbench_public_benchmarks.json
```

Scores are intentionally sparse. Missing benchmarks are kept as missing values and exposed to the model through missingness features instead of being guessed.

To use a custom benchmark profile file:

```bash
xrouter-llm train-routerbench \
  --input data/raw/routerbench_0shot.pkl \
  --benchmark-profiles path/to/model_benchmark_profiles.json
```

The custom file must include the new model's published benchmark scores:

```json
[
  {
    "model_id": "qwen-max",
    "provider": "Alibaba Cloud",
    "source_quality": "official",
    "context_length": 32768,
    "input_cost_per_1k": 0.002,
    "output_cost_per_1k": 0.006,
    "benchmarks": {
      "mmlu": 86.0,
      "gsm8k": 92.0,
      "human_eval": 89.0,
      "mbpp": 82.0,
      "mt_bench": 8.4
    }
  }
]
```

For an already trained artifact, register a new model profile before routing:

```python
from xrouter_llm import (
    ModelAwareRouterPredictor,
    ModelBenchmarkProfile,
    ModelProfile,
    PolicyParams,
    XRouter,
)

predictor = ModelAwareRouterPredictor.load(
    "artifacts/models/routerbench_0shot_modelaware_full.joblib"
)
predictor.add_benchmark_profile(
    ModelBenchmarkProfile(
        model_id="qwen-max",
        provider="Alibaba Cloud",
        source_quality="official",
        context_length=32768,
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.006,
        benchmarks={
            "mmlu": 86.0,
            "gsm8k": 92.0,
            "human_eval": 89.0,
            "mbpp": 82.0,
            "mt_bench": 8.4,
        },
    )
)

router = XRouter(
    predictor,
    model_profiles=[
        ModelProfile("gpt-4-1106-preview", input_cost_per_1k=0.01, output_cost_per_1k=0.03),
        ModelProfile("qwen-max", input_cost_per_1k=0.002, output_cost_per_1k=0.006),
    ],
)

decision = router.route(
    "Write a robust JSONL parser in Python.",
    candidate_models=["gpt-4-1106-preview", "qwen-max"],
    policy_params=PolicyParams(max_k=1),
)
```

If the new profile uses benchmark names or provider categories not present during
training, retrain with `--benchmark-profiles` so those features enter the fitted
schema.
