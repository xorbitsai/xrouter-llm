from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from xrouter_llm.data import limit_rows_by_prompt, load_csv, load_jsonl
from xrouter_llm.evaluation import (
    evaluate_model_holdout,
    evaluate_offline,
    evaluate_threshold_sweep,
)
from xrouter_llm.llmrouterbench import (
    download_llmrouterbench,
    extract_llmrouterbench_profiles,
    load_llmrouterbench,
    sample_llmrouterbench,
)
from xrouter_llm.model_aware_predictor import ModelAwareRouterPredictor
from xrouter_llm.policy import PolicyParams
from xrouter_llm.profiles import (
    BenchmarkProfileCatalog,
    combine_benchmark_profile_catalogs,
    load_benchmark_profiles,
    load_builtin_benchmark_profiles,
)
from xrouter_llm.routerbench import (
    download_routerbench,
    load_routerbench_pickle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xrouter-llm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-routerbench")
    download_parser.add_argument("--split", choices=["0shot", "5shot", "raw"], default="0shot")
    download_parser.add_argument("--output-dir", default="data/raw")
    download_parser.set_defaults(func=_download_routerbench)

    download_llmrouterbench_parser = subparsers.add_parser("download-llmrouterbench")
    download_llmrouterbench_parser.add_argument("--output-dir", default="data/raw")
    download_llmrouterbench_parser.set_defaults(func=_download_llmrouterbench)

    sample_llmrouterbench_parser = subparsers.add_parser("sample-llmrouterbench")
    sample_llmrouterbench_parser.add_argument("--input", default=None)
    sample_llmrouterbench_parser.add_argument("--output-dir", default="data/raw/llmrouterbench_sample")
    sample_llmrouterbench_parser.add_argument("--max-records", type=int, default=5000)
    sample_llmrouterbench_parser.add_argument("--max-files", type=int, default=200)
    sample_llmrouterbench_parser.add_argument("--max-records-per-file", type=int, default=25)
    sample_llmrouterbench_parser.add_argument("--max-models", type=int, default=None)
    sample_llmrouterbench_parser.add_argument("--max-tasks", type=int, default=None)
    sample_llmrouterbench_parser.add_argument("--overwrite", action="store_true")
    sample_llmrouterbench_parser.set_defaults(func=_sample_llmrouterbench)

    extract_profiles_parser = subparsers.add_parser("extract-llmrouterbench-profiles")
    extract_profiles_parser.add_argument("--input", required=True)
    extract_profiles_parser.add_argument("--output", default="artifacts/profiles/llmrouterbench_profiles.json")
    extract_profiles_parser.set_defaults(func=_extract_llmrouterbench_profiles)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", action="append", default=[])
    train_parser.add_argument("--input", default=None)
    train_parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "routerbench-pkl", "llmrouterbench"],
        default="jsonl",
    )
    train_parser.add_argument("--output", default="artifacts/models/xrouter.joblib")
    train_parser.add_argument("--metrics-output", default=None)
    train_parser.add_argument("--max-prompts", type=int, default=None)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--ensemble-size", type=int, default=8)
    train_parser.add_argument("--max-tfidf-features", type=int, default=20_000)
    train_parser.add_argument("--completion-epochs", type=int, default=8)
    train_parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    train_parser.set_defaults(balance_classes=True)
    train_parser.add_argument("--benchmark-profiles", default="builtin")
    train_parser.add_argument("--completion-score-threshold", type=float, default=0.75)
    train_parser.add_argument("--completion-threshold", type=float, default=0.5)
    train_parser.add_argument("--policy-max-k", type=int, default=1)
    train_parser.add_argument("--allow-fusion", action="store_true")
    train_parser.add_argument("--lambda-cost", type=float, default=1.0)
    train_parser.add_argument("--lambda-latency", type=float, default=0.0)
    train_parser.add_argument("--min-fusion-gain", type=float, default=0.0)
    train_parser.add_argument("--skip-eval", action="store_true")
    _add_encoder_args(train_parser)
    train_parser.set_defaults(func=_train)

    train_routerbench_parser = subparsers.add_parser("train-routerbench")
    train_routerbench_parser.add_argument("--split", choices=["0shot", "5shot", "raw"], default="0shot")
    train_routerbench_parser.add_argument("--data-dir", default="data/raw")
    train_routerbench_parser.add_argument("--input", default=None)
    train_routerbench_parser.add_argument("--output", default="artifacts/models/routerbench_0shot.joblib")
    train_routerbench_parser.add_argument("--metrics-output", default="artifacts/models/routerbench_0shot.metrics.json")
    train_routerbench_parser.add_argument("--max-prompts", type=int, default=None)
    train_routerbench_parser.add_argument("--test-size", type=float, default=0.2)
    train_routerbench_parser.add_argument("--random-state", type=int, default=42)
    train_routerbench_parser.add_argument("--ensemble-size", type=int, default=8)
    train_routerbench_parser.add_argument("--max-tfidf-features", type=int, default=20_000)
    train_routerbench_parser.add_argument("--completion-epochs", type=int, default=8)
    train_routerbench_parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    train_routerbench_parser.set_defaults(balance_classes=True)
    train_routerbench_parser.add_argument("--benchmark-profiles", default="builtin")
    train_routerbench_parser.add_argument("--completion-score-threshold", type=float, default=0.75)
    train_routerbench_parser.add_argument("--completion-threshold", type=float, default=0.5)
    train_routerbench_parser.add_argument("--policy-max-k", type=int, default=1)
    train_routerbench_parser.add_argument("--allow-fusion", action="store_true")
    train_routerbench_parser.add_argument("--lambda-cost", type=float, default=1.0)
    train_routerbench_parser.add_argument("--lambda-latency", type=float, default=0.0)
    train_routerbench_parser.add_argument("--min-fusion-gain", type=float, default=0.0)
    train_routerbench_parser.add_argument("--skip-eval", action="store_true")
    _add_encoder_args(train_routerbench_parser)
    train_routerbench_parser.set_defaults(func=_train_routerbench)

    sweep_parser = subparsers.add_parser("sweep-thresholds")
    sweep_parser.add_argument("--dataset", action="append", default=[])
    sweep_parser.add_argument("--input", default=None)
    sweep_parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "routerbench-pkl", "llmrouterbench"],
        default="jsonl",
    )
    _add_sweep_args(sweep_parser)
    sweep_parser.set_defaults(func=_sweep_thresholds)

    sweep_routerbench_parser = subparsers.add_parser("sweep-routerbench")
    sweep_routerbench_parser.add_argument("--split", choices=["0shot", "5shot", "raw"], default="0shot")
    sweep_routerbench_parser.add_argument("--data-dir", default="data/raw")
    sweep_routerbench_parser.add_argument("--input", default=None)
    _add_sweep_args(sweep_routerbench_parser)
    sweep_routerbench_parser.set_defaults(func=_sweep_routerbench)

    holdout_parser = subparsers.add_parser("eval-model-holdout")
    holdout_parser.add_argument("--dataset", action="append", default=[])
    holdout_parser.add_argument("--input", default=None)
    holdout_parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "routerbench-pkl", "llmrouterbench"],
        default="jsonl",
    )
    holdout_parser.add_argument(
        "--holdout-models",
        default=None,
        help="Comma-separated model ids to hold out. Defaults to every model.",
    )
    holdout_parser.add_argument("--output", default="artifacts/reports/model_holdout.json")
    holdout_parser.add_argument("--max-prompts", type=int, default=None)
    holdout_parser.add_argument("--test-size", type=float, default=0.2)
    holdout_parser.add_argument("--random-state", type=int, default=42)
    holdout_parser.add_argument("--ensemble-size", type=int, default=8)
    holdout_parser.add_argument("--max-tfidf-features", type=int, default=20_000)
    holdout_parser.add_argument("--completion-epochs", type=int, default=8)
    holdout_parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    holdout_parser.set_defaults(balance_classes=True)
    holdout_parser.add_argument("--no-model-id-features", dest="model_id_features", action="store_false")
    holdout_parser.set_defaults(model_id_features=True)
    holdout_parser.add_argument("--benchmark-profiles", default="builtin")
    holdout_parser.add_argument("--completion-score-threshold", type=float, default=0.75)
    holdout_parser.add_argument("--calibration-bins", type=int, default=10)
    _add_encoder_args(holdout_parser)
    holdout_parser.set_defaults(func=_eval_model_holdout)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


def _download_routerbench(args: argparse.Namespace) -> None:
    path = download_routerbench(split=args.split, output_dir=args.output_dir)
    print(path)


def _download_llmrouterbench(args: argparse.Namespace) -> None:
    path = download_llmrouterbench(output_dir=args.output_dir)
    print(path)


def _sample_llmrouterbench(args: argparse.Namespace) -> None:
    result = sample_llmrouterbench(
        input_path=args.input,
        output_dir=args.output_dir,
        max_records=args.max_records,
        max_files=args.max_files,
        max_records_per_file=args.max_records_per_file,
        max_models=args.max_models,
        max_tasks=args.max_tasks,
        overwrite=args.overwrite,
    )
    print(_to_json(result.to_dict()))


def _extract_llmrouterbench_profiles(args: argparse.Namespace) -> None:
    profiles = extract_llmrouterbench_profiles(args.input)
    payload = {"models": [profile_to_json(profile) for profile in profiles]}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    print(_to_json({"output": str(output_path), "model_count": len(profiles)}))


def _train_routerbench(args: argparse.Namespace) -> None:
    input_path = Path(args.input) if args.input else download_routerbench(split=args.split, output_dir=args.data_dir)
    _train_from_rows(
        rows=load_routerbench_pickle(
            input_path,
            max_prompts=args.max_prompts,
            random_state=args.random_state,
        ),
        args=args,
    )


def _train(args: argparse.Namespace) -> None:
    rows = _load_rows_from_args(args)
    _train_from_rows(rows=rows, args=args)


def _sweep_routerbench(args: argparse.Namespace) -> None:
    input_path = Path(args.input) if args.input else download_routerbench(split=args.split, output_dir=args.data_dir)
    _sweep_from_rows(
        rows=load_routerbench_pickle(
            input_path,
            max_prompts=args.max_prompts,
            random_state=args.random_state,
        ),
        args=args,
    )


def _sweep_thresholds(args: argparse.Namespace) -> None:
    rows = _load_rows_from_args(args)
    _sweep_from_rows(rows=rows, args=args)


def _eval_model_holdout(args: argparse.Namespace) -> None:
    rows = _load_rows_from_args(args)
    profile_catalog = _load_profile_catalog(args.benchmark_profiles)
    holdout_models = (
        [part.strip() for part in args.holdout_models.split(",") if part.strip()]
        if args.holdout_models
        else None
    )
    report = evaluate_model_holdout(
        rows,
        holdout_models=holdout_models,
        predictor_factory=lambda: _build_predictor(args, profile_catalog),
        test_size=args.test_size,
        random_state=args.random_state,
        calibration_bins=args.calibration_bins,
    )
    payload = {
        **asdict(report),
        "row_count": len(rows),
        "benchmark_profile_count": len(profile_catalog),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, payload)

    print(_to_json(payload))


def _train_from_rows(rows: list[object], args: argparse.Namespace) -> None:
    profile_catalog = _load_profile_catalog(args.benchmark_profiles)
    predictor = _build_predictor(args, profile_catalog).fit(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(output_path)

    payload = {
        "model_path": str(output_path),
        "row_count": len(rows),
        "predictor_type": type(predictor).__name__,
        "model_count": len(predictor.model_ids_),
        "model_ids": list(predictor.model_ids_),
        "benchmark_profile_count": len(profile_catalog),
        "metrics": None,
        "route_distribution": None,
        "per_model_selection_rate": None,
    }

    if not args.skip_eval:
        policy_params = PolicyParams(
            max_k=args.policy_max_k,
            allow_fusion=args.allow_fusion,
            completion_threshold=args.completion_threshold,
            lambda_cost=args.lambda_cost,
            lambda_latency=args.lambda_latency,
            min_fusion_gain=args.min_fusion_gain,
        )
        evaluation = evaluate_offline(
            rows,
            policy_params=policy_params,
            test_size=args.test_size,
            random_state=args.random_state,
            predictor_factory=lambda: _build_predictor(args, profile_catalog),
        )
        payload["metrics"] = evaluation.metrics
        payload["route_distribution"] = evaluation.route_distribution
        payload["per_model_selection_rate"] = evaluation.per_model_selection_rate

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(metrics_path, payload)

    print(_to_json(payload))


def _sweep_from_rows(rows: list[object], args: argparse.Namespace) -> None:
    profile_catalog = _load_profile_catalog(args.benchmark_profiles)
    thresholds = _parse_float_list(args.thresholds)
    report = evaluate_threshold_sweep(
        rows,
        thresholds=thresholds,
        predictor_factory=lambda: _build_predictor(args, profile_catalog),
        test_size=args.test_size,
        random_state=args.random_state,
        calibration_bins=args.calibration_bins,
    )
    payload = {
        **asdict(report),
        "row_count": len(rows),
        "benchmark_profile_count": len(profile_catalog),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, payload)

    print(_to_json(payload))


def _load_profile_catalog(path: str) -> BenchmarkProfileCatalog:
    if path in {"none", ""}:
        return BenchmarkProfileCatalog()

    catalogs: list[BenchmarkProfileCatalog] = []
    for part in path.split(","):
        item = part.strip()
        if not item:
            continue
        if item == "builtin":
            catalogs.append(load_builtin_benchmark_profiles())
        else:
            catalogs.append(load_benchmark_profiles(item))
    return combine_benchmark_profile_catalogs(catalogs)


def _build_predictor(args: argparse.Namespace, profile_catalog: BenchmarkProfileCatalog) -> object:
    return ModelAwareRouterPredictor(
        benchmark_profiles=profile_catalog,
        ensemble_size=args.ensemble_size,
        max_tfidf_features=args.max_tfidf_features,
        completion_epochs=args.completion_epochs,
        balance_classes=args.balance_classes,
        completion_score_threshold=args.completion_score_threshold,
        include_model_id_features=getattr(args, "model_id_features", True),
        include_task_features=getattr(args, "task_features", False),
        prompt_encoder=getattr(args, "prompt_encoder", "tfidf_svd"),
        prompt_svd_components=getattr(args, "prompt_svd_components", 64),
        embedding_model=getattr(args, "embedding_model", "BAAI/bge-base-en-v1.5"),
        embedding_cache_dir=getattr(args, "embedding_cache_dir", "artifacts/cache/embeddings"),
        random_state=args.random_state,
    )


def _add_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-encoder",
        choices=["tfidf_svd", "embedding"],
        default="tfidf_svd",
        help="Prompt representation: bag-of-words SVD (default) or semantic embedding.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-cache-dir", default="artifacts/cache/embeddings")
    parser.add_argument("--prompt-svd-components", type=int, default=64)
    parser.add_argument("--task-features", dest="task_features", action="store_true")


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="artifacts/reports/threshold_sweep.json")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--max-tfidf-features", type=int, default=20_000)
    parser.add_argument("--completion-epochs", type=int, default=8)
    parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    parser.set_defaults(balance_classes=True)
    parser.add_argument("--benchmark-profiles", default="builtin")
    parser.add_argument("--completion-score-threshold", type=float, default=0.75)
    parser.add_argument(
        "--thresholds",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated predicted completion probability thresholds.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    _add_encoder_args(parser)


def _parse_float_list(value: str) -> list[float]:
    output = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not output:
        raise ValueError("Expected at least one float")
    return output


def _load_rows_from_args(args: argparse.Namespace) -> list[object]:
    dataset_specs = list(getattr(args, "dataset", []) or [])
    if dataset_specs:
        rows = []
        for index, spec in enumerate(dataset_specs):
            rows.extend(_load_dataset_spec(spec, args=args, namespace=f"dataset{index}"))
        return rows

    if not args.input:
        raise ValueError("Provide --input or at least one --dataset kind:path")
    return _load_dataset(args.format, args.input, args=args)


def _load_dataset_spec(spec: str, *, args: argparse.Namespace, namespace: str) -> list[object]:
    if ":" not in spec:
        raise ValueError("Dataset spec must use kind:path, for example llmrouterbench:data/raw/bench")
    kind, path = spec.split(":", 1)
    rows = _load_dataset(kind, path, args=args)
    return _namespace_prompt_ids(rows, namespace=namespace)


def _load_dataset(kind: str, path: str, *, args: argparse.Namespace) -> list[object]:
    if kind == "jsonl":
        return limit_rows_by_prompt(
            load_jsonl(path),
            max_prompts=args.max_prompts,
            random_state=args.random_state,
        )
    if kind == "csv":
        return limit_rows_by_prompt(
            load_csv(path),
            max_prompts=args.max_prompts,
            random_state=args.random_state,
        )
    if kind == "routerbench-pkl":
        return load_routerbench_pickle(
            path,
            max_prompts=args.max_prompts,
            random_state=args.random_state,
        )
    if kind == "llmrouterbench":
        return load_llmrouterbench(
            path,
            max_prompts=args.max_prompts,
            random_state=args.random_state,
        )
    raise ValueError(f"Unsupported dataset kind {kind!r}")


def _namespace_prompt_ids(rows: list[object], *, namespace: str) -> list[object]:
    output = []
    for row in rows:
        data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        data["prompt_id"] = f"{namespace}:{data['prompt_id']}"
        output.append(data)
    return output


def profile_to_json(profile: object) -> dict[str, object]:
    data = asdict(profile)
    return {key: value for key, value in data.items() if value not in (None, (), {})}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_to_json(payload), encoding="utf-8")


def _to_json(payload: object) -> str:
    return json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)


def _json_ready(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
