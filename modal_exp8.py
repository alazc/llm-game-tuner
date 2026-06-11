"""Modal launcher for EXP-008 (proposal-distribution probe; 7B, A100).

Fans out the two probe states — both running scripts/proposal_probe.py against
Qwen2.5-7B-Instruct with batched sampling (N=50 x T in {0.8, 0.4} + greedy
reference), scoring every parsed candidate at the paired iteration-1 CRN seed:

  pfix    default board (the EXP-004 fix-it failure state)
  phold   B* = uniform rent x0.30 (in-band; "stay put" distribution)

Each writes JSONL + SUMMARY to the shared Volume under exp8_run/<state>; pull
per-file on Windows:
    modal volume get exp0-regrounding exp8_run report/figures/exp8_run

Run:
    modal run modal_exp8.py                 # both states
    modal run modal_exp8.py --only pfix     # single state
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

APP_NAME = os.environ.get("MODAL_APP_NAME", "exp8-proposal-probe")
GPU = os.environ.get("MODAL_GPU", "A100")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "7200"))
STARTUP = int(os.environ.get("MODAL_STARTUP", "1800"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "exp0-regrounding")
OUT_PREFIX = "exp8_run"

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
                "*.output", "pres", "pres2"],
    )
    .run_commands(f"cd {shlex.quote(str(REMOTE_ROOT))} && pip install --no-deps -e .")
)

app = modal.App(APP_NAME)

TASKS = [
    {"state": "pfix", "n_samples": 50, "temps": "0.8,0.4", "n_games": 200},
    {"state": "phold", "n_samples": 50, "temps": "0.8,0.4", "n_games": 200},
]


def _hf_cache() -> str:
    return str(VOL_ROOT / "hf_cache")


@app.function(image=image, timeout=3600, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def download_model() -> str:
    from huggingface_hub import snapshot_download
    cache = _hf_cache()
    os.makedirs(cache, exist_ok=True)
    snapshot_download(MODEL, cache_dir=cache)
    VOL.commit()
    return "cached: " + MODEL


@app.function(image=image, gpu=GPU, cpu=8.0, timeout=TIMEOUT,
              startup_timeout=STARTUP, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def run_state(task: dict) -> dict:
    state = task["state"]
    out_dir = VOL_ROOT / OUT_PREFIX / state

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT)
    env["LLM_CACHE_DIR"] = _hf_cache()
    env["LLM_DTYPE"] = "bfloat16"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(_hf_cache(), exist_ok=True)

    cmd = [
        "python", "scripts/proposal_probe.py",
        "--state", state,
        "--backend", "local", "--model", MODEL,
        "--n-samples", str(task["n_samples"]),
        "--temps", task["temps"],
        "--n-games", str(task["n_games"]),
        "--out-dir", str(out_dir),
    ]
    print(f"[{state}] {shlex.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env,
                          capture_output=True, text=True)
    dt = time.monotonic() - t0
    VOL.commit()

    od = Path(str(out_dir))
    n_files = sum(1 for p in od.rglob("*") if p.is_file()) if od.exists() else 0
    return {"state": state, "returncode": proc.returncode, "seconds": dt,
            "n_files": n_files,
            "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-2000:]}


@app.local_entrypoint()
def main(only: str = "") -> None:
    tasks = TASKS
    if only:
        tasks = [t for t in tasks if t["state"] == only]
        if not tasks:
            raise SystemExit(f"no task matches {only!r}")

    print(f"== EXP-008 on Modal :: {len(tasks)} GPU({GPU}) task(s) ==")
    print("pre-warming 7B into the volume cache ...")
    print("  " + download_model.remote())

    results = list(run_state.map(tasks))
    n_ok = 0
    for r in sorted(results, key=lambda x: x["state"]):
        status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        n_ok += int(r["returncode"] == 0)
        print(f"[{r['state']}] {status}  {r['seconds']:.0f}s  {r['n_files']} files")
        print("  stdout: " + "\n  ".join(r["stdout_tail"].splitlines()[-14:]))
        if r["returncode"] != 0:
            print("  stderr: " + "\n  ".join(r["stderr_tail"].splitlines()[-20:]))

    print(f"\n{n_ok}/{len(results)} OK. Pull:")
    print(f"  modal volume get {VOLUME_NAME} {OUT_PREFIX} report/figures/{OUT_PREFIX}")
