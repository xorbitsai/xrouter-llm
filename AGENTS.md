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

The production difficulty model is trained on **multiple datasets combined**
(377,997 rows / 14,364 prompts / 283 subjects):

```text
NPULH/LLMRouterBench (350k stream sample)   37 models x ~13,775 prompts, single-turn
  data/raw/llmrouterbench_stream_sample_350k  (22 tasks: arcc, gpqa, humaneval,
  livecodebench, mmlupro, swe-bench, aime, medqa, winogrande, ...)
agent-psychometrics SWE-bench Verified      500 tasks x 134 agents, coding agent
agent-psychometrics Terminal-Bench 2.0      89 tasks x 112 agents, terminal agent
```

- The two agent-psychometrics matrices are loaded by `agentic.py` (SWE-bench
  Verified task text is joined from `princeton-nlp/SWE-bench_Verified`).
- All sources feed the difficulty axis; only the profiled llmrouterbench models
  feed the capability/combine logistic (agentic subjects have no profile).
- RouterBench (`withmartian/routerbench`) is kept as a smaller legacy baseline.
- `swebench_pro` (730x14) and `gso` (102x15) are available via `agentic.py` but
  not yet wired in (need an external task-text join).

The 130k sample was the earlier baseline; the 350k sample superseded it (cleaner,
more prompts/tasks). Datasets and artifacts are not committed (`data/`,
`artifacts/` are gitignored).

## Benchmark Profiles

Model benchmark profiles are part of the model input, not metadata decoration.
They should come from published model-card, paper, official, or dataset-aggregate
benchmark scores when possible.

For LLMRouterBench samples, extract dataset aggregate profiles with:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli extract-llmrouterbench-profiles \
  --input data/raw/llmrouterbench_stream_sample_350k \
  --output artifacts/profiles/llmrouterbench_350k_profiles.json
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

`IRTRouter` consumes profiles directly: a model's capability is the normalized
mean of its published `capability_benchmarks` (default `gpqa_diamond`,
`livecodebench`). There is no separate profile featurizer anymore (the old
`BenchmarkProfileFeaturizer` and its benchmark/provider/model-id feature vector
were removed with the old predictor).

For a new model, provide a `ModelBenchmarkProfile` with published benchmark
scores and cost. If the profile introduces new benchmark names, providers, or
feature categories, retrain so those features enter the fitted schema.

### Cold start / schema mismatch (important)

A deployment model that was not in training is reached only through its profile.
For that to help, the *training* models must be described with the **same
benchmark vocabulary** as the new model. The dataset task slugs are therefore
mapped to canonical benchmark names (`gpqa->gpqa_diamond`, `livecodebench`,
`humaneval`, ...; see `LLMROUTERBENCH_CANONICAL_BENCHMARKS`) so that
`gpqa_diamond` and `livecodebench` are shared features (37/37 on the 350k
training side, 8/8 on the registry side). Add a model's published benchmarks
under those canonical names so they enter the fitted schema.

### What we actually verified (don't re-derive from hunches)

Controlled, single-feature ablations on the trained predictors — these overturned
several plausible-sounding correlational hunches, so trust the ablation, not
cross-model correlations over the 8 registry models (n is tiny and confounded):

- **In-distribution it works.** On the 350k held-out prompt split (37 models,
  ~2755 prompts) the router is solid: completion_rate ~0.61–0.67 across
  thresholds, sensible cost frontier, ECE ~0.022.
- **Encoder.** `bge-base` beats TF-IDF on same-vocabulary leave-one-model-out
  AUC (0.66->0.72). But TF-IDF is near-constant w.r.t. profile features
  (doesn't differentiate); embedding actually responds to `gpqa_diamond`
  (controlled Δ+0.13). Use embedding.
- **Cost as a feature is neutral.** `--no-cost-feature` vs on: completion_rate,
  average_cost, average_score, ECE all within noise on the held-out frontier.
  Keep it or drop it; it neither helps nor hurts the cost objective measurably.
- **`benchmark_coverage` was a real bias** (an early run keyed mu on profile
  completeness, +0.73, mis-ranking sparse strong profiles). Disable it with
  `--no-coverage-feature` for OOD robustness.
- **`aa_intelligence_index` is fragile**: only ~9/37 training models have it,
  range 14–25 vs registry 7–56, and the learned sign is unstable
  (embedding gave it a negative slope). Useful as a consistent axis only if
  coverage/range improve.
- **Benchmark vs completion: marginally ~0, but strongly predictive once you
  control for difficulty.** The joint classifier gave gpqa ~0 weight, and the
  *marginal* corr(model pass-rate, gpqa) is only ~0.06 — but that is confounded
  by easy prompts (everyone passes, washing out capability). Within an
  informative prompt the passing models are the higher-benchmark ones 80% of the
  time, and a logistic `pass ~ [capability, difficulty]` gives capability coef
  **+3.9** (difficulty -1.2). So benchmarks ARE usable — but only in a
  *factored* model that separates the two axes, not the joint classifier (where
  difficulty, represented as noisy high-dim prompt features, drowns it).

## Factored router (IRTRouter) — the current approach

`IRTRouter` (`irt_router.py`) is the production router for unseen deployment
models. It models completion as two decoupled axes (the ZeroRouter idea):

```text
P(complete) = sigmoid(a*capability(model) + b*difficulty(prompt) + c)   # a~+3.9, b~-1.2
```

- **difficulty(prompt)**: Ridge on a multilingual embedding (`BAAI/bge-m3`,
  cross-lingual so Chinese transfers from English data), trained on each
  prompt's empirical pass-rate. Held-out Pearson ~0.6 on the 350k sample.
- **capability(model)**: the published benchmark composite (mean of available
  `gpqa_diamond`, `livecodebench`), used directly — so a brand-new model's
  benchmarks drive its ranking. NOT the dataset's per-model pass-rate (that is
  confounded and uncorrelated with benchmarks, corr ~0.1).
- A small logistic combines the two, fit on the per-(prompt, model) rows.

Result: strong models rank high on hard prompts, cheap models win easy prompts,
and it works in Chinese. Train with `train-irt`; `serve` loads any fitted
predictor (currently IRTRouter).

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli train-irt \
  --input data/raw/llmrouterbench_stream_sample_350k --format llmrouterbench \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_aa.json \
  --output artifacts/models/irt_router_350k.joblib
```

### Out-of-distribution ranking is still unsolved

Ranking the brand-new registry models by their profiles is NOT reliable yet: it
is governed by several profile features that misbehave out-of-distribution
(sparse profiles, range-mismatched `aa_index`), and no single feature toggle
fixes it. The system is trustworthy in-distribution; treat registry-model
rankings as provisional. None of the registry models exist in the training data.

## Training Algorithm

The only predictor is `IRTRouter` (see the "Factored router" section above):
difficulty(prompt) from a multilingual embedding + capability(model) from the
published benchmark composite, combined by a small logistic. The old joint
`ModelAwareRouterPredictor` (TF-IDF/embedding + benchmark/provider/model-id
features through an SGD ensemble) has been removed -- it could not rank unseen
models by their benchmarks (see the verified findings above).

Train / reproduce:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli train-irt \
  --input data/raw/llmrouterbench_stream_sample_350k --format llmrouterbench \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_aa.json \
  --output artifacts/models/irt_router_350k.joblib
```

`sweep-thresholds` and `eval-model-holdout` still exist for diagnostics and now
build an `IRTRouter` via their predictor factory.

### Agentic training data (difficulty axis)

The difficulty model is only reliable for prompt types it saw in training.
`agentic.py` loads the agent-psychometrics per-(task, agent) matrices
(SWE-bench Verified 500x134, Terminal-Bench 2.0 89x112; task text from
`tasks.jsonl` or an external map e.g. SWE-bench Verified `problem_statement`).
Training the difficulty model on llmrouterbench + these makes coding/terminal
agentic prompts sensible (a SWE-style prompt's difficulty went 0.21 -> -0.84).
The agentic subjects have no benchmark profiles, so they feed ONLY the
difficulty axis; the capability/combine logistic still fits on profiled models.

Limitation (verified): real xagent prompts (e.g. Chinese business + image-gen
agentic tasks) are NOT covered by SWE-bench/Terminal-Bench either, so they stay
out-of-distribution and get a near-max (clamped) difficulty. Difficulty is
clamped to the training range so P never collapses, but the only way to make it
accurate for a specific task mix is that deployment's own logged
prompts + outcomes.

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
