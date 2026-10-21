"""Reuse organizer baselines without importing ML packages until requested."""
import importlib.util
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"zero_shot": "01_zero_shot", "rag": "02_rag"}


def load_baseline(strategy):
    path = ROOT / BASELINES[strategy] / "main.py"
    spec = importlib.util.spec_from_file_location(f"baseline_{strategy}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_answer(value):
    # Nonempty is a local quality check; the code contract only specifies str.
    if not isinstance(value, str) or not value.strip():
        raise ValueError("answer() must return a nonempty string")
    if value != value.strip():
        raise ValueError("answer() returned surrounding whitespace")
    return value


def run(module, *, model_id, strategy, corpus_dir, limit=2, output="results/run.json"):
    """Measure cold first call and warm calls. Never pass reference answers to inference."""
    if strategy not in BASELINES or limit < 1:
        raise ValueError("Choose a known strategy and positive sample limit")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline inference requires a CUDA GPU")
    if strategy == "rag":
        if not list(Path(corpus_dir).glob("*.pdf")):
            raise ValueError("RAG requires user-supplied PDFs in corpus_dir")
        for name in ("embed_model", "docling_models"):
            if not (ROOT / BASELINES[strategy] / name).is_dir():
                raise ValueError(f"Missing RAG assets: {name}; see workflow documentation")
    if module._cache:
        raise ValueError("Restart the runtime or clear the baseline cache for a cold run")
    samples = json.loads((ROOT / "sample/sample_qa.json").read_text(encoding="utf-8"))[:limit]
    def command(args):
        return subprocess.check_output(args, text=True).strip()
    result = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(command(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "python": platform.python_version(),
        "packages": command(["uv", "pip", "freeze", "--python", __import__('sys').executable]).splitlines(),
        "model": model_id, "strategy": strategy, "corpus_dir": str(corpus_dir),
        "corpus_files": [
            {"name": p.name, "sha256": __import__('hashlib').sha256(p.read_bytes()).hexdigest()}
            for p in sorted(Path(corpus_dir).glob("*.pdf"))
        ] if strategy == "rag" else [],
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "generation": {"max_new_tokens": module.MAX_NEW_TOKENS, "do_sample": False, "dtype": "bfloat16"},
        "system_prompt": module.SYSTEM_PROMPT,
        "retrieval": {k: getattr(module, k) for k in
                      ("CHUNK_SIZE", "CHUNK_OVERLAP", "MAX_CONTEXT_TOKENS", "EMBED_MODEL_PATH")}
                     if strategy == "rag" else {},
        "answers": [],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    def save():
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for sample in samples:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        value = validate_answer(module.answer(sample["question"], str(corpus_dir), model_id))
        torch.cuda.synchronize()
        result["answers"].append({
            "id": sample["id"], "question": sample["question"], "answer": value,
            "reference_answer": sample["answer"], "cold_call": not result["answers"],
            "latency_seconds": time.perf_counter() - start,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        save()
    return result
