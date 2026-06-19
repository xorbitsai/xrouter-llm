"""A/B: does expanding capability 2->4 dims improve model-holdout generalization?

Same prompts, same folds; only the capability composite changes:
  2-dim: gpqa_diamond, livecodebench
  4-dim: + hle, tau2_bench   (training profiles carry hle 14/37, tau2 12/37)
Reports the leave-one-model-out aggregate (macro AUC etc).
"""

from __future__ import annotations

import sys

from xrouter_llm.evaluation import evaluate_model_holdout
from xrouter_llm.irt_router import IRTRouter
from xrouter_llm.llmrouterbench import load_llmrouterbench
from xrouter_llm.profiles import load_benchmark_profiles

MAX_PROMPTS = 4000
SEED = 42
PROFILES = "artifacts/profiles/llmrouterbench_350k_profiles.json"
DATA = "data/raw/llmrouterbench_stream_sample_350k"

CONFIGS = {
    "2-dim (gpqa, livecodebench)": ("gpqa_diamond", "livecodebench"),
    "4-dim (+hle, +tau2_bench)": ("gpqa_diamond", "livecodebench", "hle", "tau2_bench"),
}


def main():
    print(f"loading rows (max_prompts={MAX_PROMPTS}) ...", flush=True)
    rows = load_llmrouterbench(DATA, max_prompts=MAX_PROMPTS, random_state=SEED)
    catalog = load_benchmark_profiles(PROFILES)

    only = sys.argv[1:] or list(CONFIGS)
    for tag in only:
        caps = CONFIGS[tag]
        print(f"\n=== {tag} ===", flush=True)
        report = evaluate_model_holdout(
            rows,
            predictor_factory=lambda caps=caps: IRTRouter(
                benchmark_profiles=catalog,
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_cache_dir="artifacts/cache/embeddings",
                capability_benchmarks=caps,
                random_state=SEED,
            ),
            test_size=0.2,
            random_state=SEED,
        )
        agg = report.aggregate
        for k in sorted(agg):
            print(f"  {k:32s} {agg[k]:.4f}")


if __name__ == "__main__":
    main()
