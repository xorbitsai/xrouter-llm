"""A/B experiment for prompt-conditioned benchmark weighting.

This is intentionally a script, not a CLI command yet. It compares the current
fixed capability scalar against an experimental continuous demand(prompt) vector.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Sequence

from xrouter_llm.evaluation import evaluate_model_holdout, evaluate_offline
from xrouter_llm.irt_router import IRTRouter
from xrouter_llm.llmrouterbench import load_llmrouterbench
from xrouter_llm.policy import PolicyParams
from xrouter_llm.profiles import load_benchmark_profiles
from xrouter_llm.prompt_conditioned_irt import PromptConditionedIRTRouter


SEED = 42
DATA = "data/raw/llmrouterbench_stream_sample_350k"
PROFILES = "artifacts/profiles/llmrouterbench_350k_profiles.json"

CAPABILITY_CONFIGS: dict[str, tuple[str, ...]] = {
    "gpqa": ("gpqa_diamond",),
    "gpqa_lcb": ("gpqa_diamond", "livecodebench"),
    "gpqa_lcb_humaneval": ("gpqa_diamond", "livecodebench", "humaneval"),
    "priority_core": ("gpqa_diamond", "livecodebench", "hle", "tau2_bench"),
    "priority_reasoning": (
        "gpqa_diamond",
        "livecodebench",
        "mmlu_pro",
        "hle",
        "aime_2025",
        "tau2_bench",
    ),
    "priority_all": (
        "gpqa_diamond",
        "livecodebench",
        "mmlu_pro",
        "hle",
        "aime_2025",
        "tau2_bench",
        "ifbench",
        "scicode",
        "aa_lcr",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-prompts", type=int, default=2000)
    parser.add_argument("--mode", choices=("offline", "holdout", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--profiles", default=PROFILES)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=("gpqa", "gpqa_lcb", "gpqa_lcb_humaneval", "priority_core", "priority_reasoning", "priority_all"),
        choices=tuple(CAPABILITY_CONFIGS),
    )
    args = parser.parse_args()

    print(f"loading rows (max_prompts={args.max_prompts}) ...", flush=True)
    rows = load_llmrouterbench(DATA, max_prompts=args.max_prompts, random_state=SEED)
    catalog = load_benchmark_profiles(args.profiles)

    factories: dict[str, Callable[[], object]] = {}
    for config in args.configs:
        benchmarks = CAPABILITY_CONFIGS[config]
        factories[f"irt_{config}"] = _irt_factory(catalog, benchmarks)
        factories[f"pc_{config}"] = _pc_factory(catalog, benchmarks)

    if args.mode in {"offline", "both"}:
        print("\n# prompt-split routing metrics", flush=True)
        for name, factory in factories.items():
            report = evaluate_offline(
                rows,
                predictor_factory=factory,
                policy_params=PolicyParams(completion_threshold=args.threshold),
                test_size=0.2,
                random_state=SEED,
            )
            m = report.metrics
            print(
                f"{name:24s} completion={m['completion_rate']:.4f} "
                f"score={m['average_score']:.4f} cost={m['average_cost']:.6f} "
                f"cost_x_oracle={m['selected_cost_over_oracle_success_cost']:.3f}",
                flush=True,
            )

    if args.mode in {"holdout", "both"}:
        print("\n# leave-one-model-out completion prediction", flush=True)
        for name, factory in factories.items():
            report = evaluate_model_holdout(
                rows,
                predictor_factory=factory,
                test_size=0.2,
                random_state=SEED,
            )
            agg = report.aggregate
            print(
                f"{name:24s} auc={agg['macro_auc']:.4f} "
                f"brier={agg['macro_brier']:.4f} "
                f"ece={agg['macro_expected_calibration_error']:.4f}",
                flush=True,
            )


def _irt_factory(catalog: object, benchmarks: Sequence[str]) -> Callable[[], IRTRouter]:
    return lambda: IRTRouter(
        benchmark_profiles=catalog,
        capability_benchmarks=benchmarks,
        random_state=SEED,
    )


def _pc_factory(catalog: object, benchmarks: Sequence[str]) -> Callable[[], PromptConditionedIRTRouter]:
    return lambda: PromptConditionedIRTRouter(
        benchmark_profiles=catalog,
        capability_benchmarks=benchmarks,
        random_state=SEED,
    )


if __name__ == "__main__":
    main()
