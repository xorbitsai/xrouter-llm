# xrouter-llm Agent Notes

This repository builds an LLM router around one invariant:

```text
Do not train: prompt -> selected model
Train:        prompt + model -> probability the model can complete the prompt
Decide:       predicted completion + uncertainty + cost + latency -> selected model set
```

## Routing Objective

The production objective is not "pick the strongest model." It is:

```text
Among models predicted to complete the prompt, pick the cheapest one.
```

The completion label is currently:

```text
normalized_score >= 0.75
```

The policy uses `completion_threshold` as the predicted capability cutoff. If no
candidate reaches the cutoff, it falls back to the highest predicted completion
probability.

`best_fixed_*` metrics are diagnostic only. They mean "always route to one static
model" and should not be treated as the router target, because a strongest fixed
model will usually win raw score while losing the cost objective.

## Current Data Strategy

Use `NPULH/LLMRouterBench` as the current real-data source. RouterBench is kept
as a smaller legacy baseline.

Current local sample:

```text
data/raw/llmrouterbench_stream_sample_130k
```

Current sample characteristics:

```text
rows:    130000
prompts: 5382
models:  38
tasks:   arcc, arenahard_coding, gpqa, humaneval, livecodebench, medqa, winogrande
```

The sample is intentionally not committed. `data/` and `artifacts/` are local
working artifacts.

Multiple datasets are supported with repeated `--dataset kind:path` arguments.
Prompt ids are namespaced per dataset before training to avoid collisions.

## Benchmark Profiles

Model benchmark profiles are part of the model input, not metadata decoration.
They should come from published model-card, paper, official, or dataset-aggregate
benchmark scores when possible.

For LLMRouterBench samples, extract dataset aggregate profiles with:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli extract-llmrouterbench-profiles \
  --input data/raw/llmrouterbench_stream_sample_130k \
  --output artifacts/profiles/llmrouterbench_stream_sample_130k_profiles.json
```

Profiles include aggregate benchmark scores and fitted input/output cost when
token and cost fields are present.

The profile featurizer currently uses:

- normalized benchmark values
- benchmark missingness features
- benchmark coverage
- source quality
- context length and output-token limits
- parameter counts when present
- input/output cost
- provider one-hot features
- model-id one-hot features for observed trained models

For a new model, provide a `ModelBenchmarkProfile` with published benchmark
scores and cost. If the profile introduces new benchmark names, providers, or
feature categories, retrain so those features enter the fitted schema.

## Current Training Algorithm

The main predictor is `ModelAwareRouterPredictor`.

Feature construction:

```text
prompt text -> TF-IDF -> SVD dense prompt vector
model profile -> benchmark/cost/provider/model-id feature vector
training row -> [prompt features, profile features, prompt-profile interactions]
```

Classifier:

```text
ensemble of SGDClassifier(loss="log_loss")
target = score >= completion_score_threshold
```

Current training defaults:

```text
completion_score_threshold: 0.75
completion_epochs:          8
balance_classes:            true
max_tfidf_features:         20000
```

Keep class balancing enabled by default. A non-balanced experiment improved ECE
but worsened the completion/cost frontier, so it is only exposed through
`--no-balance-classes` for experiments.

## Reproduce Current Training

Train the current optimized artifact on the 130k LLMRouterBench sample:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli train \
  --input data/raw/llmrouterbench_stream_sample_130k \
  --format llmrouterbench \
  --benchmark-profiles artifacts/profiles/llmrouterbench_stream_sample_130k_profiles.json \
  --completion-score-threshold 0.75 \
  --completion-threshold 0.8 \
  --ensemble-size 4 \
  --completion-epochs 8 \
  --max-tfidf-features 20000 \
  --test-size 0.2 \
  --random-state 42 \
  --output artifacts/models/llmrouterbench_stream_sample_130k_optimized.joblib \
  --metrics-output artifacts/models/llmrouterbench_stream_sample_130k_optimized.metrics.json
```

Sweep thresholds before choosing a production threshold:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli sweep-thresholds \
  --input data/raw/llmrouterbench_stream_sample_130k \
  --format llmrouterbench \
  --benchmark-profiles artifacts/profiles/llmrouterbench_stream_sample_130k_profiles.json \
  --completion-score-threshold 0.75 \
  --ensemble-size 4 \
  --completion-epochs 8 \
  --max-tfidf-features 20000 \
  --test-size 0.2 \
  --random-state 42 \
  --thresholds 0.4,0.5,0.6,0.7,0.8,0.9 \
  --output artifacts/reports/llmrouterbench_stream_sample_130k_sweep_modelid_epochs8.json
```

## Evaluation Rules

Use held-out prompt splits, not row-random leakage:

```text
test_size:    0.2
random_state: 42
```

Sparse dataset coverage matters. During offline evaluation, only route among
models that have observed labels for that held-out prompt. Otherwise the router
could select a model with no ground-truth score and bias the reported result.

Report real effect with:

- `completion_rate`
- `average_score`
- `average_cost`
- `route_distribution`
- calibration as secondary context

Do not report smoke tests as final model quality. A smoke run only checks that
the pipeline executes on a small subset.

## Current 130k Sample Results

Corrected baseline before the latest optimization:

| threshold | completion_rate | average_score | average_cost |
| --- | ---: | ---: | ---: |
| 0.7 | 78.44% | 0.7862 | 0.005834 |
| 0.8 | 80.86% | 0.8099 | 0.011353 |

Optimized configuration: model-id features + 8 epochs + class balancing.

| threshold | completion_rate | average_score | average_cost |
| --- | ---: | ---: | ---: |
| 0.7 | 79.93% | 0.8011 | 0.005899 |
| 0.8 | 81.97% | 0.8211 | 0.010683 |
| 0.9 | 82.34% | 0.8248 | 0.011668 |

Recommendation:

```text
cost-sensitive:    completion_threshold = 0.7
quality-sensitive: completion_threshold = 0.8
```

The latest trained local artifact is:

```text
artifacts/models/llmrouterbench_stream_sample_130k_optimized.joblib
```

## Verification Before Commit

Run:

```bash
PYTHONPATH=src python3 -m compileall -q src tests
python3 -m pytest -q
```

Before committing, scan for accidentally pasted secrets:

```bash
rg -n "sk-[A-Za-z0-9]{16,}|[D]ASHSCOPE_API_KEY|dashscope[.]aliyuncs[.]com" -S . || true
```

Never commit local datasets, model artifacts, `.venv`, `.idea`, pytest caches,
or API keys.
