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
candidate reaches the cutoff, the completion objective has already failed, so it
does NOT pay for the single highest-completion (priciest) model — it takes the
**cheapest** candidate within `fallback_quality_margin` (default 0.05) of the
best predicted completion.

`best_fixed_*` metrics are diagnostic only. They mean "always route to one static
model" and should not be treated as the router target, because a strongest fixed
model will usually win raw score while losing the cost objective.

## Current Data Strategy

The production difficulty model is trained on **four datasets combined**
(378,397 rows / ~14,463 prompts / 287 subjects):

```text
NPULH/LLMRouterBench (350k stream sample)   37 models x ~13,775 prompts, single-turn
  data/raw/llmrouterbench_stream_sample_350k  (22 tasks: arcc, gpqa, humaneval,
  livecodebench, mmlupro, swe-bench, aime, medqa, winogrande, ...)
agent-psychometrics Terminal-Bench 2.0      89 tasks x 112 subjects, terminal agent
  data/agentic/terminalbench  (--dataset agentic:agentic/terminalbench)
agent-psychometrics SWE-bench Verified      500 tasks x 134 subjects, coding agent
  data/agentic/swebench_verified  (task text joined from princeton-nlp/SWE-bench_Verified)
Xorbits/xagent-xrouter-labels                100 prompts x 4 OpenRouter models, real xagent
  --dataset xagent-labels:Xorbits/xagent-xrouter-labels:full
```

Only models with benchmark profiles feed the capability/combine logistic: the
37 llmrouterbench models plus profiled xagent OpenRouter candidates. The
agent-psychometrics subjects (agent+scaffold combos) feed the difficulty axis
only. agent-psychometrics swebench_pro (730x14) and gso (102x15) are loadable
but NOT wired in (they ship no local task text and need an external join).

- agent-psychometrics matrices load via `agentic.py` and the CLI `agentic:`
  dataset kind. **terminalbench** (89x112, local `tasks.jsonl`) and
  **swebench_verified** (500x134, task text joined from
  `princeton-nlp/SWE-bench_Verified`) are wired in; see "Agentic training data".
- All sources feed the difficulty axis; only the profiled llmrouterbench models
  plus profiled xagent OpenRouter candidates feed the capability/combine
  logistic (agentic subjects have no profile).
- RouterBench (`withmartian/routerbench`) is kept as a smaller legacy baseline.
- `swebench_pro` (730x14), `gso` (102x15) ship no local task text and need an
  external join (not wired yet).

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

For real deployment models, profiles live as a per-model YAML registry bundled at
`src/xrouter_llm/resources/config/models/` (one file per model), shipped as
package data. `load_benchmark_profiles()` accepts a JSON/YAML file, a single
bare-mapping model file, or a whole directory. The bundled registry is the
default for `--benchmark-profiles` (resolved via `default_models_dir()`); pass a
path to override or stack:

```bash
# (default) the bundled registry is loaded automatically
--benchmark-profiles builtin,$(python -c 'import xrouter_llm;print(xrouter_llm.default_models_dir())')  # stack with builtin
```

The serve defaults — `--model`/`--models-dir`/`--routers-dir` — likewise resolve
to the bundled artifact/registry/configs (`default_model_path()`,
`default_models_dir()`, `default_routers_dir()`), so `xrouter-llm serve` runs
with no flags. Training still writes fresh artifacts to `artifacts/` and reads
profiles from collected JSON as before.

Conventions in the registry: benchmarks are stored as published percentages
(0-100; the featurizer normalizes >1 by /100). Do not put Elo/contest ratings
there. Only sourced/official values are active data; unverified third-party
numbers stay in YAML comments. (`aa_intelligence_index` was dropped from the
registry: it was never consumed by `IRTRouter` and was verified fragile -- see
the finding below. Recover it from git history if a future capability axis can
use it.)

`IRTRouter` consumes profiles directly: a model's capability is the mean of its
published `capability_benchmarks` (default **`gpqa_diamond` + `livecodebench`**
-- see "Capability benchmarks" below). There is no
separate profile featurizer anymore (the old `BenchmarkProfileFeaturizer` and
its benchmark/provider/model-id feature vector were removed with the old
predictor).

For a new model, provide a `ModelBenchmarkProfile` with published benchmark
scores and cost. If the profile introduces new benchmark names, providers, or
feature categories, retrain so those features enter the fitted schema.

### Cold start / schema mismatch (important)

A deployment model that was not in training is reached only through its profile.
For that to help, the *training* models must be described with the **same
benchmark vocabulary** as the new model. The dataset task slugs are therefore
mapped to canonical benchmark names (`gpqa->gpqa_diamond`, `livecodebench`,
`humaneval`, ...; see `LLMROUTERBENCH_CANONICAL_BENCHMARKS`) so that
`gpqa_diamond` and `livecodebench` are shared capability features on both sides
(37/37 on the 350k training side; registry gpqa 11/11, livecodebench 8/11; a
model missing one falls back to the mean of what it has). Always give a new
deployment model a published `gpqa_diamond` (and `livecodebench` if available)
so it enters the fitted schema.

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
  (controlled Δ+0.13). Use embedding. **Encoder probe (current):** within
  embeddings, `Qwen3-Embedding-0.6B` > `bge-m3` for the *difficulty* axis —
  same-data held-out Pearson 0.60 vs 0.55, and it fixes `bge-m3`'s trivial-
  prompt blowup (`bge-m3` rated "1+1=?"/"写一个快速排序" at the max 3.89; Qwen
  drops them to ~0.5–1.6 and lifts truly-hard prompts to the top). A frozen
  generative `Qwen3.5-0.8B` (mean-pooled, no fine-tuning) was worst (0.54,
  incoherent) — decoder hidden states aren't a difficulty axis untuned. The
  probe is `scripts/probe_qwen_difficulty.py`. `bge-m3` is no longer the
  default; `IRTRouter`/CLI default to `Qwen/Qwen3-Embedding-0.6B`.
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

- **difficulty(prompt)**: Ridge on a multilingual embedding
  (`Qwen/Qwen3-Embedding-0.6B`, cross-lingual so Chinese transfers from English
  data), trained on each prompt's empirical pass-rate. Held-out Pearson ~0.6 on
  the 350k sample. Chosen over `bge-m3` by a controlled probe (see "Encoder
  probe" below): same-data held-out Pearson 0.55 -> 0.60, and it stops the
  failure where `bge-m3` pinned *trivial* prompts to max difficulty (e.g.
  "1+1=?" 3.89 -> 0.76, "1加1等于几" 2.71 -> 0.53) while still ranking genuinely
  hard prompts highest ("prove the Riemann hypothesis" -> 3.89). A frozen
  generative LM (`Qwen3.5-0.8B`, mean-pooled last hidden, no fine-tuning) was
  *worse* (Pearson 0.54, erratic probes): raw decoder hidden states are not a
  clean difficulty axis without fine-tuning, which we are not doing.
- **capability(model)**: the mean of the published `gpqa_diamond` and
  `livecodebench`, used directly — so a brand-new model's benchmarks drive its
  ranking. NOT the dataset's per-model pass-rate (confounded, corr ~0.1).
- A small logistic combines the two, fit on the per-(prompt, model) rows.

### Long templated agent prompts: embedding view + xagent weighting

Two silent failure modes made agent-deployment labels (e.g. the public
`Xorbits/xagent-xrouter-labels` set) nearly useless to the difficulty axis:

1. **Tokenizer truncation ate the user's request.** The backend truncates at
   512 tokens, so an 11k-char templated prompt contributed only its template
   head. Skill-selection prompts are the worst case: the template fills both
   the head AND the tail, with the actual user task mid-prompt after a
   `<user>` marker — their embeddings were *identical* (within-category
   pairwise cosine exactly 1.0), so the bundled router assigned every skill
   prompt the same difficulty (std 0.000).
2. **Minority dilution.** ~100 deployment prompts next to ~14k benchmark
   prompts are 0.7% of the Ridge loss; unweighted they cannot move it.

Fixes (both on by default in `IRTRouter`, tunable via `train-irt` flags):

- `prompt_embedding_view` (`encoders.py`): embed `head + focus + tail` slices
  (600 chars each; focus starts at the LAST user marker, e.g. `<user>`,
  `## User Task`), shrunk to a ~460-token budget so CJK-heavy text also fits
  the window. Short prompts pass through unchanged, so existing embedding
  cache entries stay valid. Encoder-level the view is opt-in
  (`view_head_chars=0` default) — only `IRTRouter` turns it on.
- `xagent_weight` (default 8): sample weight for xagent-labeled prompts in
  the difficulty Ridge.

Verified effect (5-fold CV over the 99 public xagent prompts, out-of-fold
difficulty vs empirical pass-rate): Spearman 0.282 -> 0.368 (p=0.0002);
ablation shows both parts contribute (view only: 0.306, weight only: 0.329).
Skill-category within-class embedding cosine 1.000 -> 0.83, and the bundled
router's difficulty std on those prompts 0.000 -> 0.77. The bundled
`irt_router_350k.joblib` is retrained with these defaults (old artifacts are
replaced, not kept compatible).

### Capability benchmarks: gpqa+livecodebench — verified on the routing objective

The default is `gpqa_diamond` + `livecodebench` (mean of the two). Judge on the
**routing objective** (completion_rate / cost on the prompt split), NOT on
capability-ranking AUC: AUC measures whether capability ranks models within a
prompt, but per-model holdout AUC is mathematically invariant to capability and
even per-prompt cross-model AUC flips sign between prompt samples. The routing
metric is what production cares about.

Prompt-split routing (threshold 0.7, fuller *collected* profiles):

- `gpqa` only vs `gpqa+livecodebench`: within noise, sign flips with the prompt
  sample (full ~14k: gpqa-only 0.6363 vs 0.6356; Codex's 10k subset: gpqa+lcb
  0.6490 vs 0.6430). gpqa+lcb is chosen: at least as good, both 37/37 on the
  training side, no coverage downside, and the conventional two-axis composite.
- Going wider does NOT help. Flat mean over more benchmarks *dilutes* (a coding
  score is noise on a medicine prompt). A *learned* weighting over 7 benchmarks
  + missing-value imputation overfits: per-prompt cross-model AUC 0.715
  in-sample collapses to 0.683 leave-one-model-out — below the simple composite.
- Prompt-conditioned / domain-matched capability (`prompt_conditioned_irt.py`,
  experimental) edges gpqa+lcb on routing (pc 0.6575 vs irt 0.6490 at 10k) but
  needs a prompt-domain signal and is not in production yet.

Binding constraint: the *number of profiled models* (37), not benchmarks per
model — a learned multi-benchmark / prompt-conditioned capability only becomes a
robust win when that count is far larger. The collected wider benchmarks are
kept in the profiles as an archive for that future. Repro:
`scripts/ab_prompt_conditioned_capability.py` (capability configs × {irt, pc} ×
{routing, holdout}); collect/export with `scripts/collect_pricepertoken_benchmarks.py`
+ `scripts/export_collected_benchmark_profiles.py` into `scripts/benchmarks_to_collect.csv`.

Result: strong models rank high on hard prompts, cheap models win easy prompts,
and it works in Chinese. Train with `train-irt`; `serve` loads any fitted
predictor (currently IRTRouter).

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli train-irt \
  --dataset llmrouterbench:data/raw/llmrouterbench_stream_sample_350k \
  --dataset agentic:agentic/terminalbench \
  --dataset agentic:agentic/swebench_verified \
  --dataset xagent-labels:Xorbits/xagent-xrouter-labels:full \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_priority_collected.json,src/xrouter_llm/resources/config/models \
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
  --dataset llmrouterbench:data/raw/llmrouterbench_stream_sample_350k \
  --dataset agentic:agentic/terminalbench \
  --dataset agentic:agentic/swebench_verified \
  --dataset xagent-labels:Xorbits/xagent-xrouter-labels:full \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_priority_collected.json,src/xrouter_llm/resources/config/models \
  --output artifacts/models/irt_router_350k.joblib
```

`sweep-thresholds` and `eval-model-holdout` still exist for diagnostics and now
build an `IRTRouter` via their predictor factory.

### Agentic training data (difficulty axis)

The difficulty model is only reliable for prompt types it saw in training.
`agentic.py` loads the agent-psychometrics per-(task, agent) matrices
(github.com/dariakryvosheieva/agent-psychometrics, MIT) and is wired into the
CLI as the `agentic:` dataset kind: `--dataset agentic:agentic/terminalbench`
loads `data/agentic/<dataset>/responses.jsonl` + `tasks.jsonl`. Stack it with
llmrouterbench, e.g.:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli train-irt \
  --dataset llmrouterbench:data/raw/llmrouterbench_stream_sample_350k \
  --dataset agentic:agentic/terminalbench \
  --dataset agentic:agentic/swebench_verified \
  --dataset xagent-labels:Xorbits/xagent-xrouter-labels:full \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_priority_collected.json,src/xrouter_llm/resources/config/models \
  --output artifacts/models/irt_router_350k.joblib
```

- **terminalbench** (89 tasks x 112 subjects) is self-contained (local
  `tasks.jsonl`) -- the cheapest agentic source.
- **swebench_verified** (500 tasks x 134 subjects) is wired in; its task text is
  joined from `princeton-nlp/SWE-bench_Verified` `problem_statement` into a local
  `data/agentic/swebench_verified/tasks.jsonl` (regenerate on a fresh checkout
  via `datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")`).
- **xagent labels** are loaded from `Xorbits/xagent-xrouter-labels` via
  `xagent_labels.py` and the CLI `xagent-labels:` dataset kind.
- **swebench_pro** (730x14), **gso** (102x15) ship no local task text and need
  an external join -- not wired yet.

Adding the agentic sets calibrates agentic-prompt difficulty (sampled A/B: a
SWE-style prompt 0.89 -> 0.02 with swebench_verified, a bash terminal task
clamped-high -> mid-range with terminalbench) and even pulls trivial prompts
down (more diverse training distribution), without breaking hard prompts. The
agentic subjects have no
benchmark profiles, so they feed ONLY the difficulty axis; the capability/combine
logistic still fits on profiled models.

Limitation (verified): the public xagent labels are only a 100-prompt seed. They
slightly improve the public benchmark headline under a controlled base test set,
but real deployment accuracy for a specific task mix still needs that
deployment's own logged prompts + outcomes.

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

## Current 350k Results (agentic + xagent, embedding view + xagent weighting)

Full four-dataset口径 (378,397 rows / 287 subjects, prompt-grouped split,
random_state 42 -> 2,893 test prompts), IRTRouter defaults. "Before" is the
same data trained without the embedding view and xagent weighting:

| threshold | completion (before -> after) | average_cost (before -> after) |
| --- | --- | --- |
| 0.5 | — -> 56.45% | — -> 0.006571 |
| 0.6 | 56.27% -> 57.17% | 0.008185 -> 0.008456 |
| 0.7 | 56.24% -> 57.10% | 0.009923 -> 0.010447 |
| 0.8 | 57.07% -> 57.83% | 0.009873 -> 0.010517 |
| 0.9 | 57.17% -> 57.93% | 0.010064 -> 0.010648 |

Reading: +0.8 to +0.9 pts completion at every matched threshold, and the new
thr-0.5 point beats the old thr-0.6 completion at 19.7% lower cost. The
bundled `irt_router_350k.joblib` is trained with these defaults. Reproduce:

```bash
PYTHONPATH=src python3 -m xrouter_llm.cli sweep-thresholds \
  --dataset llmrouterbench:data/raw/llmrouterbench_stream_sample_350k \
  --dataset xagent-labels:Xorbits/xagent-xrouter-labels:full \
  --dataset agentic:agentic/terminalbench \
  --dataset agentic:agentic/swebench_verified \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_priority_collected.json,src/xrouter_llm/resources/config/models \
  --output artifacts/reports/irt_router_350k_agentic_xagent100_sweep.json
```

## Serving (routing decision API + web)

A zero-dependency serving layer exposes model selection over HTTP. It is
decision-only: it answers "which model should serve this prompt" and records the
choice. It does NOT proxy to the underlying LLMs.

```bash
# bundled router + registry + configs are the defaults:
xrouter-llm serve --port 8080
# or override any of them (e.g. a freshly trained artifact):
PYTHONPATH=src python3 -m xrouter_llm.cli serve \
  --model artifacts/models/irt_router_350k.joblib \
  --db artifacts/calls.db --port 8080
```

- A *router config* (`resources/config/routers/*.yaml`, one per file, bundled) is
  the user's "auto config": a named candidate set (1 or N models) plus policy
  knobs (`completion_threshold`, `lambda_cost`, `max_k`). Ships `auto`,
  `cheap-pair`, `single-opus`.
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
