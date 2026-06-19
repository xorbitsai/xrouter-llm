<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# xrouter-llm

</div>

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

- **capability(model)** = the mean of the model's published `gpqa_diamond` and
  `livecodebench` (both full-coverage on the training side). Going wider doesn't
  help at this data scale — a flat mean dilutes and learned weights overfit at
  37 profiled models; see AGENTS.md "Capability benchmarks". Used directly, so a
  brand-new model's benchmarks drive its ranking.
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
- `resources/config/models/`: a per-model YAML registry of capability profiles
  (bundled in the package; resolve with `default_models_dir()`).
- `resources/config/routers/`: named "auto configs" — a candidate model set +
  policy (bundled; `default_routers_dir()`).
- `resources/models/irt_router_350k.joblib`: the trained router shipped with the
  package (`default_model_path()`).

## Install

```bash
pip install xrouter-llm        # ships a trained router + model registry
# or, for development:
pip install -e ".[dev]"
```

The wheel bundles a trained router artifact, the model-profile registry, and the
router configs, so a fresh install can serve immediately with no extra files.

## Datasets

The production difficulty model is trained on **multiple datasets combined**
(all feed the difficulty axis; only profiled models feed the capability axis):

| Source | Type | Scale | In production train? |
| --- | --- | --- | --- |
| `NPULH/LLMRouterBench` (350k stream sample) | single-turn QA / code / math (22 tasks) | 37 models x ~13.8k prompts | ✅ |
| agent-psychometrics — Terminal-Bench 2.0 | terminal agent | 89 tasks x 112 subjects | ✅ `--dataset agentic:agentic/terminalbench` |
| agent-psychometrics — SWE-bench Verified | coding agent | 500 tasks x 134 subjects | ✅ task text joined from `princeton-nlp/SWE-bench_Verified` |
| agent-psychometrics — SWE-bench Pro / GSO | coding agent | 730x14 / 102x15 | ⛔ ship no local task text, external join needed |

The current artifact trains on LLMRouterBench 350k **+ Terminal-Bench +
SWE-bench Verified** (377,997 rows / ~14,364 prompts / 283 subjects). The
agentic matrices come from
[agent-psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics)
(MIT) via `agentic.py`. Only the 37 profiled llmrouterbench models feed the
capability axis; agentic subjects feed difficulty only. RouterBench
(`withmartian/routerbench`) remains a smaller legacy baseline. Local datasets and
trained artifacts are not committed (`data/`, `artifacts/` are gitignored).

Adding more agentic prompt types (e.g. your own traffic) is the only way to make
difficulty accurate for task mixes outside coding/terminal — see AGENTS.md.

## Train

```bash
xrouter-llm train-irt \
  --dataset llmrouterbench:data/raw/llmrouterbench_stream_sample_350k \
  --dataset agentic:agentic/terminalbench \
  --dataset agentic:agentic/swebench_verified \
  --benchmark-profiles artifacts/profiles/llmrouterbench_350k_profiles_priority_collected.json \
  --output artifacts/models/irt_router_350k.joblib
```

Diagnostics: `sweep-thresholds` (cost/completion frontier + calibration) and
`eval-model-holdout` (leave-one-model-out generalization).

## Serve

The bundled router, registry, and configs are the defaults, so a bare invocation
works out of the box:

```bash
xrouter-llm serve --port 8080
```

Override any of them to use your own trained model or registry:

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

One YAML per supported model, bundled under
`src/xrouter_llm/resources/config/models/` (capability profile: provider, costs,
context, published benchmarks as 0-100 percentages). `model_id` is the model's
canonical OpenRouter slug (e.g. `anthropic/claude-opus-4.8`). The bundled
registry is the default for `--benchmark-profiles`; point it at your own
directory or file to extend it. Add a model = add a file.

```python
from xrouter_llm import IRTRouter, default_model_path, default_models_dir, load_benchmark_profiles

router = IRTRouter.load(default_model_path())
for profile in load_benchmark_profiles(default_models_dir()).profiles():
    router.add_benchmark_profile(profile)

preds = router.predict("实现一个分布式一致性算法", model_ids=["claude-opus-4-8", "deepseek-v4-pro"])
print({p.model_id: round(p.mu, 3) for p in preds})
```

## License

`xrouter-llm` is released under the **Xagent Source License** (© Xorbits Inc.) —
see [LICENSE](LICENSE). It is source-available, **not** an OSI-approved open
source license.

The license text is shared verbatim with [Xagent](https://github.com/xorbitsai/xagent);
for this project the licensed "Software" is `xrouter-llm`, and the
"Restricted Functionality" / hosted-service and competitive-use clauses apply to
its routing-decision and model-selection capabilities. In short: use,
modification, and internal/single-tenant deployment are permitted; offering it as
a multi-tenant hosted/managed service, or a directly competing service, is not.
See [LICENSE](LICENSE) for the controlling terms.
