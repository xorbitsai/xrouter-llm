# xrouter-llm

`xrouter-llm` is a prompt-aware LLM **routing-decision** service. It answers
"which model should serve this prompt?" and records the choice — it does NOT
call the underlying LLMs.

## Invariant

```text
Do not train:  prompt -> selected model
Train:         prompt + model -> probability the model completes the prompt
Decide:        predicted completion + cost -> cheapest model that can complete
```

Completion is factored into two decoupled axes (an IRT-style model):

```text
P(complete) = sigmoid(a * capability(model) + b * difficulty(prompt) + c)
```

- **capability(model)** = the model's published benchmark composite
  (`gpqa_diamond`, `livecodebench`). Used directly, so a brand-new model's
  benchmarks drive its ranking.
- **difficulty(prompt)** = a Ridge regressor on a multilingual embedding
  (`Qwen/Qwen3-Embedding-0.6B`), trained on each prompt's empirical pass-rate.
  Multilingual (Chinese transfers from English training data). Picked over
  `bge-m3` by a controlled probe (`scripts/probe_qwen_difficulty.py`): higher
  held-out Pearson and it no longer rates trivial prompts ("1+1=?") as maximally
  hard.

This factoring is the key lesson: a single joint classifier could not rank
unseen models by their benchmarks (on this data, model capability barely
explains completion *marginally* — but it does once difficulty is controlled,
which is exactly what the factored model exploits).

## Components

- `IRTRouter` (`irt_router.py`): the predictor (difficulty x capability).
- `RoutingPolicy` (`policy.py`): "cheapest model whose predicted completion
  clears `completion_threshold`; else the highest predicted completion".
- `serving.py` / `server.py`: HTTP routing-decision API + single-page web UI.
- `config/models/`: a per-model YAML registry of capability profiles.
- `config/routers/`: named "auto configs" (a candidate model set + policy).

## Install

```bash
pip install -e ".[dev]"
```

## Datasets

The production difficulty model is trained on **multiple datasets combined**
(all feed the difficulty axis; only profiled models feed the capability axis):

| Source | Type | Scale |
| --- | --- | --- |
| `NPULH/LLMRouterBench` (350k stream sample) | single-turn QA / code / math (22 tasks) | 37 models x ~13.8k prompts |
| agent-psychometrics — SWE-bench Verified | coding agent | 500 tasks x 134 agents |
| agent-psychometrics — Terminal-Bench 2.0 | terminal agent | 89 tasks x 112 agents |

The agentic matrices come from
[agent-psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics)
(MIT) and are loaded by `agentic.py`. RouterBench (`withmartian/routerbench`)
remains a smaller legacy baseline. Local datasets and trained artifacts are not
committed (`data/`, `artifacts/` are gitignored).

Adding more agentic prompt types (e.g. your own traffic) is the only way to make
difficulty accurate for task mixes outside coding/terminal — see AGENTS.md.

## Train

```bash
xrouter-llm train-irt \
  --input data/raw/llmrouterbench_stream_sample_350k --format llmrouterbench \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_aa.json \
  --output artifacts/models/irt_router_350k.joblib
```

Diagnostics: `sweep-thresholds` (cost/completion frontier + calibration) and
`eval-model-holdout` (leave-one-model-out generalization).

## Serve

```bash
xrouter-llm serve \
  --model artifacts/models/irt_router_350k.joblib \
  --models-dir config/models --routers-dir config/routers \
  --db artifacts/calls.db --port 8080
```

- `GET /` — single-page UI (prompt box, config picker, decision table, history)
- `GET /api/configs`, `POST /api/route` (`{prompt, config, task?}`),
  `GET /api/history?limit=N`
- Every decision is logged to SQLite (`*.db`/`*.sqlite` are gitignored — the log
  holds user prompts).

## Model registry

One YAML per supported model under `config/models/` (capability profile: provider,
costs, context, published benchmarks as 0-100 percentages). `model_id` is the
model's canonical OpenRouter slug (e.g. `anthropic/claude-opus-4.8`).
Load with `--benchmark-profiles config/models`. Add a model = add a file.

```python
from xrouter_llm import IRTRouter, load_benchmark_profiles

router = IRTRouter.load("artifacts/models/irt_router_350k.joblib")
for profile in load_benchmark_profiles("config/models").profiles():
    router.add_benchmark_profile(profile)

preds = router.predict("实现一个分布式一致性算法", model_ids=["claude-opus-4-8", "deepseek-v4-pro"])
print({p.model_id: round(p.mu, 3) for p in preds})
```
