# xrouter-llm Core Algorithm

The project is built around one invariant:

```text
Predictor:
  prompt + model -> quality distribution

Policy:
  quality distribution + cost + latency -> selected model set

Executor:
  selected model set -> single call or fusion
```

It intentionally does not train a direct classifier like `prompt -> selected model`.

## Predictor

For each candidate model, prediction returns:

```text
mu     expected quality in [0, 1]
sigma  uncertainty in [0.03, 0.30]
cost   estimated request cost
latency estimated request latency
```

`ModelAwareRouterPredictor` trains one global bootstrap ensemble over completion
labels:

```text
prompt + model profile -> P(model can complete prompt)
```

Dense benchmark scores become binary completion labels:

```text
label = score >= completion_score_threshold
```

The completion classifier uses:

```text
prompt TF-IDF features
+ prompt numeric features
+ model benchmark profile features
+ prompt/profile interaction features
```

This keeps old and new models under the same contract:

```text
prompt + model -> quality distribution
```

For models with dense labels, RouterBench rows provide direct supervision. For a
new model without dense rows, published benchmark scores provide a cold-start
prior and the predictor increases uncertainty.

Prediction returns:

```text
mu = P(model can complete prompt)
sigma = uncertainty of that probability
```

The policy is constrained by capability first and cost second:

```text
1. Enumerate candidate model sets up to max_k.
2. Keep sets where expected_quality >= completion_threshold.
3. Select the cheapest capable set.
4. If no set is predicted capable, fall back to the highest expected quality.
```

The built-in benchmark profiles are sparse by design. Missing public scores are
encoded as missingness features instead of guessed.

## Policy

`RoutingPolicy` computes the expected completion probability for each candidate
subset, then applies a capability constraint before considering cost.

Single model:

```text
U({m}) = mu_m - lambda_cost * cost_m - lambda_latency * latency_m
```

Multiple models:

```text
U(S) =
  E[max(Q_i for i in S)]
  - lambda_cost * sum(cost_i)
  - lambda_latency * max(latency_i)
  - judge_cost
  - fusion_overhead
```

The current implementation estimates `E[max(Q_i)]` with deterministic Monte Carlo samples from the configured random seed.

## Selection

Selection enumerates candidate sets up to `max_k`:

```text
1. Build every allowed set of size 1..max_k.
2. Compute expected completion probability for each set.
3. Keep sets where expected_quality >= completion_threshold.
4. Pick the cheapest capable set.
5. If no set is predicted capable, pick the set with highest expected quality.
```

`k = 1` is normal routing. `k > 1` means the caller should execute selected models in parallel and fuse the answers.

## Evaluation

Offline fusion quality is reported as a best-of-k upper bound:

```text
actual_quality = max(actual_score(m) for m in selected_models)
```

This is useful for policy tuning, but it is not a measured fused-answer quality.

## Threshold Sweep

`evaluate_threshold_sweep` trains the predictor once, caches held-out prompt
predictions, and replays `RoutingPolicy` across multiple predicted completion
probability thresholds. The report is meant to choose:

```text
the cheapest threshold that satisfies the required completion rate
```

It also includes a calibration table:

```text
predicted P(complete) bucket -> observed completion rate
```

Poor calibration means the threshold should be treated as a policy knob rather
than an absolute probability until enough production labels are collected.
