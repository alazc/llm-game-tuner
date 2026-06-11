"""Modal launcher for EXP-012 (choice-shaped actions; 7B, A100).

One task per board, both arms sequential inside (menu then lever). Pull:
    modal volume get exp0-regrounding exp12_run report/figures/exp12_run

Run:
    modal run modal_exp12.py
    modal run modal_exp12.py --only default --arm lever
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

APP_NAME = os.environ.get("MODAL_APP_NAME", "exp12-choice")
GPU = os.environ.get("MODAL_GPU", "A100")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "10800"))
STARTUP = int(os.environ.get("MODAL_STARTUP", "1800"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "exp0-regrounding")
OUT_PREFIX = "exp12_run"

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

BOARDS = ["default", "salary x2", "gut mid-tier"]


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
def run_board(task: dict) -> dict:
    board = task["board"]
    out_dir = VOL_ROOT / task.get("out_prefix", OUT_PREFIX)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT)
    env["LLM_CACHE_DIR"] = _hf_cache()
    env["LLM_DTYPE"] = "bfloat16"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(_hf_cache(), exist_ok=True)

    cmd = [
        "python", "scripts/choice_designer_loop.py",
        "--backend", "local", "--model", MODEL,
        "--arm", task.get("arm", "both"),
        "--boards", board, "--n-seeds", "3", "--K", "8",
        "--n-games", "200", "--workers", "4",
        "--out-dir", str(out_dir),
    ]
    print(f"[{board}] {shlex.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env,
                          capture_output=True, text=True)
    dt = time.monotonic() - t0
    VOL.commit()

    od = Path(str(out_dir))
    n_files = sum(1 for p in od.rglob("*.jsonl") if p.is_file()) if od.exists() else 0
    return {"board": board, "returncode": proc.returncode, "seconds": dt,
            "n_files": n_files,
            "stdout_tail": proc.stdout[-3500:], "stderr_tail": proc.stderr[-2500:]}


@app.local_entrypoint()
def main(only: str = "", arm: str = "both", out_prefix: str = "") -> None:
    boards = [b for b in BOARDS if not only or b == only]
    if not boards:
        raise SystemExit(f"no board matches {only!r}")
    prefix = out_prefix or OUT_PREFIX
    tasks = [{"board": b, "arm": arm, "out_prefix": prefix} for b in boards]

    print(f"== choice arms on Modal :: {len(boards)} GPU({GPU}) task(s) "
          f"arm={arm} out={prefix} ==")
    print("pre-warming 7B into the volume cache ...")
    print("  " + download_model.remote())

    results = list(run_board.map(tasks))
    n_ok = 0
    for r in sorted(results, key=lambda x: x["board"]):
        status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        n_ok += int(r["returncode"] == 0)
        print(f"[{r['board']}] {status}  {r['seconds']:.0f}s  {r['n_files']} jsonl")
        print("  stdout: " + "\n  ".join(r["stdout_tail"].splitlines()[-18:]))
        if r["returncode"] != 0:
            print("  stderr: " + "\n  ".join(r["stderr_tail"].splitlines()[-20:]))

    print(f"\n{n_ok}/{len(results)} OK. Pull:")
    print(f"  modal volume get {VOLUME_NAME} {OUT_PREFIX} report/figures/{OUT_PREFIX}")
