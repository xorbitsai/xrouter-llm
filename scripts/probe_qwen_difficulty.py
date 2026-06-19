"""Backbone probe: does a Qwen3 representation rate difficulty better than bge-m3?

No fine-tuning. For each frozen backbone we replicate IRTRouter's difficulty
axis exactly (per-prompt pass-rate -> b = -logit, Ridge on the embedding), on a
held-out prompt split, then report:
  - held-out Pearson (difficulty prediction quality, in-distribution)
  - difficulty estimates for a probe set (trivial vs hard, EN + ZH)
The probe is what answers the real question: trivial prompts should score LOW.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from xrouter_llm.data import coerce_benchmark_rows
from xrouter_llm.encoders import EmbeddingEncoder, SentenceTransformerBackend
from xrouter_llm.llmrouterbench import load_llmrouterbench
from xrouter_llm.score import ScoreNormalizer

MAX_PROMPTS = 5000
THRESHOLD = 0.75
PASSRATE_FLOOR = 0.02
MIN_MODELS = 3
SEED = 42

PROBES = [
    ("1+1=?", "trivial"),
    ("1加1等于几", "trivial"),
    ("What is 1+1?", "trivial"),
    ("法国的首都是哪里？", "easy-fact"),
    ("写一个快速排序", "easy-code"),
    ("用 Python 反转一个字符串", "easy-code"),
    ("证明黎曼猜想", "hard"),
    ("实现一个分布式一致性算法 (Raft)", "hard"),
    ("Prove that there are infinitely many primes.", "med-math"),
]


def _logit(p: float, floor: float = PASSRATE_FLOOR) -> float:
    p = min(max(p, floor), 1.0 - floor)
    return float(np.log(p / (1.0 - p)))


def build_labels(rows):
    rows = coerce_benchmark_rows(rows)
    norm = ScoreNormalizer().fit([r.score for r in rows])
    prompt_text, completed = {}, {}
    for r in rows:
        label = 1.0 if norm.transform(r.score) >= THRESHOLD else 0.0
        prompt_text.setdefault(r.prompt_id, r.prompt)
        completed.setdefault(r.prompt_id, []).append(label)
    pids = [p for p, l in completed.items() if len(l) >= MIN_MODELS]
    b = {p: -_logit(float(np.mean(completed[p]))) for p in pids}
    return pids, prompt_text, b


class HFHiddenStateBackend:
    """Frozen causal-LM as a feature extractor: mean-pool the last hidden layer.

    This is the "small LLM + regression head" idea without any fine-tuning --
    the generative backbone's representation is used as-is.
    """

    def __init__(self, model_name, *, max_seq_length=512, batch_size=16, device=None):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self._tok = None
        self._model = None

    @property
    def name(self):
        return f"hf-mean:{self.model_name}"

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=dtype,
            ).to(self.device).eval()
        return self._tok, self._model

    def encode(self, texts):
        import torch

        tok, model = self._ensure()
        out = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = tok(
                batch, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                res = model(**enc, output_hidden_states=True)
            hidden = res.hidden_states[-1]  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1)
            mean = (summed / counts).float().cpu().numpy()
            out.append(mean)
            print(f"      encoded {min(i + self.batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.vstack(out)


def run_backbone(tag, backend, pids, prompt_text, b, tr_idx, te_idx):
    enc = EmbeddingEncoder(
        backend, n_components=4096, include_numeric=False,
        cache_dir=f"artifacts/cache/embeddings", random_state=SEED,
    )
    train_prompts = [prompt_text[pids[i]] for i in tr_idx]
    test_prompts = [prompt_text[pids[i]] for i in te_idx]
    y_tr = np.array([b[pids[i]] for i in tr_idx])
    y_te = np.array([b[pids[i]] for i in te_idx])

    print(f"[{tag}] encoding {len(train_prompts)} train + {len(test_prompts)} test prompts ...", flush=True)
    x_tr = enc.fit_transform(train_prompts)
    x_te = enc.transform(test_prompts)
    model = Ridge(alpha=1.0).fit(x_tr, y_tr)

    pred = model.predict(x_te)
    r, _ = pearsonr(pred, y_te)
    dmin, dmax = float(y_tr.min()), float(y_tr.max())
    print(f"[{tag}] held-out Pearson = {r:.3f}   (difficulty range [{dmin:.2f}, {dmax:.2f}])")

    probe_x = enc.transform([p for p, _ in PROBES])
    probe_d = model.predict(probe_x)
    print(f"[{tag}] probe difficulties (clamped to train range; lower = easier):")
    for (text, kind), d in zip(PROBES, probe_d):
        dc = float(np.clip(d, dmin, dmax))
        print(f"    {dc:+6.2f}  [{kind:9s}] {text}")
    return r


def main():
    print(f"loading llmrouterbench (max_prompts={MAX_PROMPTS}) ...", flush=True)
    rows = load_llmrouterbench(
        "data/raw/llmrouterbench_stream_sample_350k",
        max_prompts=MAX_PROMPTS, random_state=SEED,
    )
    pids, prompt_text, b = build_labels(rows)
    print(f"usable prompts: {len(pids)}")

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(pids))
    cut = int(len(pids) * 0.8)
    tr_idx, te_idx = order[:cut], order[cut:]

    backbones = {
        "bge-m3": lambda: SentenceTransformerBackend("BAAI/bge-m3", max_seq_length=512),
        "qwen3-emb-0.6b": lambda: SentenceTransformerBackend(
            "Qwen/Qwen3-Embedding-0.6B", max_seq_length=512
        ),
        "qwen3.5-0.8b": lambda: HFHiddenStateBackend("Qwen/Qwen3.5-0.8B", max_seq_length=512),
    }
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(backbones)
    for tag in selected:
        run_backbone(tag, backbones[tag](), pids, prompt_text, b, tr_idx, te_idx)


if __name__ == "__main__":
    main()
