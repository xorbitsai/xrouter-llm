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

### Model registry (hand-authored YAML)

For real deployment models, profiles live as a per-model YAML registry under
`config/models/` (one file per model). `load_benchmark_profiles()` accepts a
JSON/YAML file, a single bare-mapping model file, or a whole directory.

```bash
--benchmark-profiles config/models            # load the registry
--benchmark-profiles builtin,config/models    # stack with builtin profiles
```

Conventions in the registry: benchmarks are stored as published percentages
(0-100; the featurizer normalizes >1 by /100). Do not put Elo/contest ratings
there. Only sourced/official values are active data; unverified third-party
numbers stay in YAML comments. Each model also carries `aa_intelligence_index`
(Artificial Analysis Index v4.1, a consistent cross-model capability scalar).

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

### Cold start / schema mismatch (important)

A deployment model that was not in training is reached only through its profile.
For that to actually help, the *training* models must be described with the
**same benchmark vocabulary** as the new model. The current optimized artifact
is fitted on `llmrouterbench_*` benchmark names; the `config/models/` registry
uses `mmlu / gpqa_diamond / aa_intelligence_index / ...` plus new providers.
Those names are **not in the fitted schema, so they are silently ignored** — for
the 8 registry models the predictor falls back to prompt text + generic numeric
features, which is why their predicted completions cluster tightly. To make the
registry profiles inform predictions, retrain on data whose models share this
benchmark vocabulary (e.g. the Artificial Analysis index as a common capability
axis). None of the registry models exist in the 130k sample.

## Current Training Algorithm

The main predictor is `ModelAwareRouterPredictor`.

Feature construction:

```text
prompt text -> prompt encoder -> SVD dense prompt vector
model profile -> benchmark/cost/provider/model-id feature vector
training row -> [prompt features, profile features, prompt-profile interactions]
```

The prompt encoder is pluggable (`--prompt-encoder`, see `encoders.py`):

- `tfidf_svd` (default): TF-IDF (1,2) + numeric features -> TruncatedSVD.
- `embedding`: a sentence-transformers backend (default `BAAI/bge-base-en-v1.5`)
  -> SVD to the same dim. Raw embeddings are cached per prompt under
  `artifacts/cache/embeddings/`. The backend is swappable (a Xinference backend
  can be added later). Optional `--task-features` adds a task one-hot.

Pre-encoder artifacts still load (predict falls back to `featurizer_/prompt_svd_`).

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

Cost is de-leaked: evaluation routes on *predicted* cost (profile per-1k pricing
x prompt length + assumed output budget), never the realized `cost_usd`. The
realized cost is reported as `average_cost`; the estimate is `average_decision_cost`.

### Generalization (model holdout)

Prompt-split metrics never test generalization to an unseen *model*, which is
the core invariant. Use `eval-model-holdout`: each model is removed from
training entirely and predicted only from its profile.

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli eval-model-holdout \
  --input data/raw/llmrouterbench_stream_sample_130k --format llmrouterbench \
  --benchmark-profiles artifacts/profiles/llmrouterbench_stream_sample_130k_profiles.json \
  --ensemble-size 4 --completion-epochs 8 \
  --output artifacts/reports/llmrouterbench_130k_model_holdout.json
```

Findings on the 130k sample (held-out models, macro AUC):

- TF-IDF encoder: **0.658** — barely above chance; strongest models worst
  (gpt-5-chat ~0.61).
- `bge-base` embedding encoder: **0.719** (38/38 models improved); biggest gains
  on the strong models TF-IDF couldn't place. The representation switch is the
  one lever that worked.
- model-id features: ~0 effect out-of-distribution (their in-distribution gain
  is memorization). Widening SVD dims (64->128) and `--task-features`: noise-level
  (+0.003 / +0.002; embedding already encodes task).

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

## Serving (routing decision API + web)

A zero-dependency serving layer exposes model selection over HTTP. It is
decision-only: it answers "which model should serve this prompt" and records the
choice. It does NOT proxy to the underlying LLMs.

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli serve \
  --model artifacts/models/llmrouterbench_stream_sample_130k_optimized.joblib \
  --models-dir config/models --routers-dir config/routers \
  --db artifacts/calls.db --port 8080
```

- A *router config* (`config/routers/*.yaml`, one per file) is the user's "auto
  config": a named candidate set (1 or N models) plus policy knobs
  (`completion_threshold`, `lambda_cost`, `max_k`). Ships `auto`, `cheap-pair`,
  `single-opus`.
- Endpoints: `GET /` (single-page UI), `GET /api/configs`, `POST /api/route`
  (`{prompt, config, task?}`), `GET /api/history?limit=N`.
- Every decision is logged to SQLite (`store.CallStore`): config, prompt,
  candidates, selected, predicted completion, cost. `*.db`/`*.sqlite` are
  gitignored (the log holds user prompts).
- See the cold-start note above: with the current artifact, registry-model
  predictions are weak until a profile-vocabulary-aligned retrain.

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
