"""A/B on the production objective: does gpqa-only capability beat gpqa+lcb?

Per-prompt cross-model AUC said gpqa_diamond ALONE ranks models better than
gpqa+livecodebench (livecodebench dilutes on this mixed-task prompt set). This
confirms it on the full router (difficulty + combine) with the routing metrics.
"""

from __future__ import annotations

from xrouter_llm.evaluation import evaluate_offline
from xrouter_llm.irt_router import IRTRouter
from xrouter_llm.llmrouterbench import load_llmrouterbench
from xrouter_llm.policy import PolicyParams
from xrouter_llm.profiles import load_benchmark_profiles

SEED = 42
PROFILES = "artifacts/profiles/llmrouterbench_350k_profiles.json"
CONFIGS = {
    "gpqa+livecodebench (current)": ("gpqa_diamond", "livecodebench"),
    "gpqa_diamond only": ("gpqa_diamond",),
    "gpqa+mmlu_pro": ("gpqa_diamond", "mmlu_pro"),
}


def main():
    rows = load_llmrouterbench("data/raw/llmrouterbench_stream_sample_350k",
                               max_prompts=None, random_state=SEED)
    catalog = load_benchmark_profiles(PROFILES)
    for thr in (0.7, 0.8):
        print(f"\n##### completion_threshold = {thr} #####")
        for tag, caps in CONFIGS.items():
            res = evaluate_offline(
                rows,
                policy_params=PolicyParams(completion_threshold=thr, lambda_cost=1.0, max_k=1),
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
            m = res.metrics
            print(f"  {tag:30s} completion={m.get('completion_rate'):.4f} "
                  f"score={m.get('average_score'):.4f} cost={m.get('average_cost'):.6f}")


if __name__ == "__main__":
    main()
