"""Modal launcher for EXP-003 (de-confounded re-establishment gate + scale-check).

Fans out, on L4 GPUs: the 4 LLM conditions (mute/haz/met/full) for BOTH models
(Qwen2.5-1.5B-Instruct workhorse + Qwen2.5-7B-Instruct scale-check) plus the 3
model-agnostic comparators (rand/canon/sanity) run once. Model is an explicit,
disclosed condition. Each run writes JSONL to a Modal Volume under
exp0_run2/<model_tag>/<cond>; pull with `modal volume get` (per-file on Windows).

Returns ONLY tiny status (large returns hit this Modal build's broken BlobGet).

Run:
    modal run modal_exp0.py                 # prewarm both models + full fan-out
    modal run modal_exp0.py --only 1.5B/full   # single (tag/cond) for debug
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath

import modal

LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = PurePosixPath("/root/llm-game-design")
VOL_ROOT = PurePosixPath("/vol")

APP_NAME = os.environ.get("MODAL_APP_NAME", "exp0-channel-regrounding")
GPU = os.environ.get("MODAL_GPU", "L4")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "7200"))
STARTUP = int(os.environ.get("MODAL_STARTUP", "1800"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "exp0-regrounding")
OUT_PREFIX = "exp0_run2"

MODELS = {
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "7B":   "Qwen/Qwen2.5-7B-Instruct",
}

VOL = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _secrets() -> list[modal.Secret]:
    vals = {k: os.environ[k] for k in ("HF_TOKEN",) if os.environ.get(k)}
    return [modal.Secret.from_dict(vals)] if vals else []


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers>=4.44", "accelerate>=0.30", "huggingface_hub",
                 "numpy", "pyyaml", "pandas", "tqdm", "matplotlib")
    .add_local_dir(
        str(LOCAL_ROOT), remote_path=str(REMOTE_ROOT), copy=True,
        ignore=["**/__pycache__", "**/*.pyc", ".git", "report", "report/figures",
                "models", "logs", "**/*.egg-info", "*.log", "**/*.log",
                "*.output", "modal_exp0_run.log"],
    )
    .run_commands(f"cd {shlex.quote(str(REMOTE_ROOT))} && pip install --no-deps -e .")
)

app = modal.App(APP_NAME)

LLM_CONDS = ["mute", "haz", "met", "full"]
COMPARATORS = [
    {"cond": "rand",   "boards": "all",     "n_seeds": 3, "n_games": 200, "K": 8},
    {"cond": "canon",  "boards": "all",     "n_seeds": 3, "n_games": 200, "K": 8},
    {"cond": "sanity", "boards": "default", "n_seeds": 1, "n_games": 400, "K": 8},
]


def _build_tasks():
    tasks = []
    for tag, model in MODELS.items():
        for cond in LLM_CONDS:
            tasks.append({"tag": tag, "model": model, "cond": cond,
                          "boards": "all", "n_seeds": 3, "n_games": 200, "K": 8})
    # comparators are model-agnostic (no LLM call) -> run once under 'shared'
    for c in COMPARATORS:
        tasks.append({"tag": "shared", "model": MODELS["1.5B"], **c})
    return tasks


def _hf_cache() -> str:
    return str(VOL_ROOT / "hf_cache")


@app.function(image=image, timeout=3600, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def download_models() -> str:
    """Pre-warm BOTH models into the shared Volume cache (files only, no load)."""
    from huggingface_hub import snapshot_download
    cache = _hf_cache()
    os.makedirs(cache, exist_ok=True)
    done = []
    for model in MODELS.values():
        snapshot_download(model, cache_dir=cache)
        done.append(model)
    VOL.commit()
    return "cached: " + ", ".join(done)


def _run_one(task: dict) -> dict:
    """Body shared by the GPU (LLM) and CPU (comparator) functions."""
    cond, tag, model = task["cond"], task["tag"], task["model"]
    out_dir = VOL_ROOT / OUT_PREFIX / tag / cond

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT)
    env["LLM_CACHE_DIR"] = _hf_cache()
    env["LLM_DTYPE"] = "bfloat16"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(_hf_cache(), exist_ok=True)

    cmd = [
        "python", "scripts/llm_design_loop.py",
        "--backend", "local", "--model", model,
        "--ablation-condition", cond,
        "--boards", task["boards"],
        "--n-seeds", str(task["n_seeds"]),
        "--K", str(task["K"]),
        "--n-games", str(task["n_games"]),
        "--out-dir", str(out_dir),
    ]
    print(f"[{tag}/{cond}] {shlex.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env,
                          capture_output=True, text=True)
    dt = time.monotonic() - t0
    VOL.commit()

    od = Path(str(out_dir))
    n_files = sum(1 for p in od.rglob("*")
                  if p.is_file() and p.suffix in (".jsonl", ".json", ".txt", ".md")) \
        if od.exists() else 0
    return {"tag": tag, "cond": cond, "returncode": proc.returncode, "seconds": dt,
            "n_files": n_files,
            "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]}


@app.function(image=image, gpu=GPU, cpu=8.0, timeout=TIMEOUT,
              startup_timeout=STARTUP, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def run_llm(task: dict) -> dict:
    """LLM conditions (mute/haz/met/full) -- need the GPU for model inference."""
    return _run_one(task)


@app.function(image=image, cpu=4.0, timeout=TIMEOUT,
              volumes={str(VOL_ROOT): VOL}, secrets=_secrets())
def run_comparator(task: dict) -> dict:
    """rand/canon/sanity make NO LLM call -- CPU only, so they don't consume a GPU
    slot (the run otherwise needs 8 GPUs for the LLM x {1.5B,7B} grid)."""
    return _run_one(task)


@app.local_entrypoint()
def main(only: str = "") -> None:
    tasks = _build_tasks()
    if only:
        tasks = [t for t in tasks if f"{t['tag']}/{t['cond']}" == only]
        if not tasks:
            raise SystemExit(f"no task matches {only!r}")

    llm_tasks = [t for t in tasks if t["cond"] in LLM_CONDS]      # 8 -> GPU
    cmp_tasks = [t for t in tasks if t["cond"] not in LLM_CONDS]  # 3 -> CPU
    print(f"== EXP-003 on Modal :: {len(llm_tasks)} GPU({GPU}) + {len(cmp_tasks)} CPU ==")
    print("pre-warming both models into the volume cache ...")
    print("  " + download_models.remote())

    # CPU comparators run concurrently (spawned), LLM grid on GPUs (<=8, under the
    # 10-GPU account limit). Then gather both.
    cmp_handles = [run_comparator.spawn(t) for t in cmp_tasks]
    results = list(run_llm.map(llm_tasks)) if llm_tasks else []
    results += [h.get() for h in cmp_handles]
    n_ok = 0
    for r in sorted(results, key=lambda x: (x["tag"], x["cond"])):
        status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        if r["returncode"] == 0:
            n_ok += 1
        print(f"[{r['tag']}/{r['cond']}] {status}  {r['seconds']:.0f}s  {r['n_files']} files")
        if r["returncode"] != 0:
            print("  " + "\n  ".join(r["stderr_tail"].splitlines()[-20:]))

    print(f"\n{n_ok}/{len(results)} OK. Artifacts on Volume '{VOLUME_NAME}' under "
          f"/{OUT_PREFIX} -- pull with:")
    print(f"  modal volume get {VOLUME_NAME} {OUT_PREFIX} report/figures/{OUT_PREFIX}")
