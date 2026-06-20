from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Define before local imports so submodules can read xrouter_llm.__version__
# during their own initialization without hitting a partially-initialized module.
try:
    __version__ = _pkg_version("xrouter-llm")
except PackageNotFoundError:  # not installed (e.g. running from a bare checkout)
    __version__ = "0.0.0+unknown"

from xrouter_llm.catalog import ModelCatalog
from xrouter_llm.encoders import (
    EmbeddingEncoder,
    SentenceTransformerBackend,
    TfidfSvdEncoder,
    build_prompt_encoder,
)
from xrouter_llm.data import limit_rows_by_prompt, load_csv, load_jsonl, split_by_prompt
from xrouter_llm.evaluation import (
    EvaluationResult,
    ModelHoldoutResult,
    ThresholdSweepResult,
    evaluate_model_holdout,
    evaluate_offline,
    evaluate_threshold_sweep,
)
from xrouter_llm.fusion import CandidateAnswer, build_fusion_prompt
from xrouter_llm.llmrouterbench import (
    LLMRouterBenchSampleResult,
    download_llmrouterbench,
    extract_llmrouterbench_profiles,
    load_llmrouterbench,
    sample_llmrouterbench,
)
from xrouter_llm.irt_router import IRTRouter
from xrouter_llm.paths import (
    default_model_path,
    default_models_dir,
    default_routers_dir,
)
from xrouter_llm.policy import PolicyParams, RoutingPolicy
from xrouter_llm.prompt_conditioned_irt import PromptConditionedIRTRouter
from xrouter_llm.profiles import (
    BenchmarkProfileCatalog,
    ModelBenchmarkProfile,
    load_builtin_benchmark_profiles,
    load_benchmark_profiles,
)
from xrouter_llm.routerbench import download_routerbench, load_routerbench_pickle
from xrouter_llm.router import XRouter
from xrouter_llm.serving import RouterConfig, RoutingService, load_router_configs
from xrouter_llm.store import CallStore
from xrouter_llm.types import (
    BenchmarkRow,
    ModelPrediction,
    ModelProfile,
    RouteDecision,
    UtilityBreakdown,
)

__all__ = [
    "BenchmarkRow",
    "CandidateAnswer",
    "EvaluationResult",
    "LLMRouterBenchSampleResult",
    "ModelHoldoutResult",
    "ThresholdSweepResult",
    "BenchmarkProfileCatalog",
    "CallStore",
    "EmbeddingEncoder",
    "IRTRouter",
    "default_model_path",
    "default_models_dir",
    "default_routers_dir",
    "RouterConfig",
    "RoutingService",
    "ModelBenchmarkProfile",
    "ModelCatalog",
    "ModelPrediction",
    "ModelProfile",
    "PolicyParams",
    "PromptConditionedIRTRouter",
    "RouteDecision",
    "RoutingPolicy",
    "SentenceTransformerBackend",
    "TfidfSvdEncoder",
    "UtilityBreakdown",
    "XRouter",
    "build_fusion_prompt",
    "build_prompt_encoder",
    "download_routerbench",
    "download_llmrouterbench",
    "evaluate_model_holdout",
    "evaluate_offline",
    "evaluate_threshold_sweep",
    "extract_llmrouterbench_profiles",
    "limit_rows_by_prompt",
    "load_benchmark_profiles",
    "load_router_configs",
    "load_builtin_benchmark_profiles",
    "load_csv",
    "load_llmrouterbench",
    "load_routerbench_pickle",
    "load_jsonl",
    "sample_llmrouterbench",
    "split_by_prompt",
]
