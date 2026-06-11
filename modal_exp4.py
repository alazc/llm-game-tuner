"""Modal launcher for EXP-004 (exploration-mechanism probe; 7B only).

Fans out, on L4 GPUs, the three probe conditions — all running
scripts/mech_probe_loop.py against Qwen2.5-7B-Instruct:

  peval       declarative recognition (24 board orderings; one forward pass each)
  phold_good  MET editing loop seeded FROM B* (rent x0.30)
  phold_bad   MET editing loop seeded FROM default (EXP-003 replication + control)

Each writes JSONL to a Modal Volume under exp4_run/<cond>; pull with
`modal volume get` (per-file on Windows). Returns ONLY tiny status (this Modal
build's BlobGet breaks on large returns).

Run:
    modal run modal_exp4.py                  # prewarm 7B + all 3 conditions
    modal run modal_exp4.py --only phold_good # single condition for debug
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

APP_NAME = os.environ.get("MODAL_APP_NAME", "exp4-mech-probe")
GPU = os.environ.get("MODAL_GPU", "L4")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "7200"))
STARTUP = int(os.environ.get("MODAL_STARTUP", "1800"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "exp0-regrounding")
OUT_PREFIX = "exp4_run"

MODEL = "Qwen/Qwen2.5-7B-Instruct"

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

# Each task = one probe mode. peval is cheap (24 forward passes); the phold_*
# loops are the long poles (5 seeds x K=8 x n_games=200, like EXP-003 7B).
TASKS = [
    {"mode": "peval",      "n_seeds": 5, "n_games": 200, "K": 8},
    {"mode": "phold_good", "n_seeds": 5, "n_games": 200, "K": 8},
    {"mode": "phold_bad",  "n_seeds": 5, "n_games": 200, "K": 8},
]


def _hf_cache() -> str:
    return str(VOL_ROOT / "hf_cache")


@app.function(image=image, timeout=3600, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def download_model() -> str:
    """Pre-warm the 7B into the shared Volume cache (files only, no load)."""
    from huggingface_hub import snapshot_download
    cache = _hf_cache()
    os.makedirs(cache, exist_ok=True)
    snapshot_download(MODEL, cache_dir=cache)
    VOL.commit()
    return "cached: " + MODEL


@app.function(image=image, gpu=GPU, cpu=8.0, timeout=TIMEOUT,
              startup_timeout=STARTUP, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def run_mode(task: dict) -> dict:
    mode = task["mode"]
    out_dir = VOL_ROOT / OUT_PREFIX / mode

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT)
    env["LLM_CACHE_DIR"] = _hf_cache()
    env["LLM_DTYPE"] = "bfloat16"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(_hf_cache(), exist_ok=True)

    cmd = [
        "python", "scripts/mech_probe_loop.py",
        "--mode", mode,
        "--backend", "local", "--model", MODEL,
        "--n-seeds", str(task["n_seeds"]),
        "--K", str(task["K"]),
        "--n-games", str(task["n_games"]),
        "--out-dir", str(out_dir),
    ]
    print(f"[{mode}] {shlex.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env,
                          capture_output=True, text=True)
    dt = time.monotonic() - t0
    VOL.commit()

    od = Path(str(out_dir))
    n_files = sum(1 for p in od.rglob("*") if p.is_file()) if od.exists() else 0
    return {"mode": mode, "returncode": proc.returncode, "seconds": dt,
            "n_files": n_files,
            "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


@app.local_entrypoint()
def main(only: str = "") -> None:
    tasks = TASKS
    if only:
        tasks = [t for t in tasks if t["mode"] == only]
        if not tasks:
            raise SystemExit(f"no task matches {only!r}")

    print(f"== EXP-004 on Modal :: {len(tasks)} GPU({GPU}) task(s) ==")
    print("pre-warming 7B into the volume cache ...")
    print("  " + download_model.remote())

    results = list(run_mode.map(tasks))
    n_ok = 0
    for r in sorted(results, key=lambda x: x["mode"]):
        status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        n_ok += int(r["returncode"] == 0)
        print(f"[{r['mode']}] {status}  {r['seconds']:.0f}s  {r['n_files']} files")
        print("  stdout: " + "\n  ".join(r["stdout_tail"].splitlines()[-8:]))
        if r["returncode"] != 0:
            print("  stderr: " + "\n  ".join(r["stderr_tail"].splitlines()[-20:]))

    print(f"\n{n_ok}/{len(results)} OK. Artifacts on Volume '{VOLUME_NAME}' under "
          f"/{OUT_PREFIX} -- pull with:")
    print(f"  modal volume get {VOLUME_NAME} {OUT_PREFIX} report/figures/{OUT_PREFIX}")
