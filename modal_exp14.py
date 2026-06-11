"""Modal launcher for EXP-014 ranking elicitation (the only GPU step).

One A100 task runs scripts/screening_optimizer.py --elicit (3 Monopoly boards
+ 5 synthetic instances = 8 greedy ranking calls) and writes
exp14_run/rankings.json to the shared volume. Pull:
    modal volume get exp0-regrounding exp14_run/rankings.json \
        report/figures/exp14_run/rankings.json

All arms (llm / random / oracle) then run LOCALLY against the saved rankings.

Run:  modal run modal_exp14.py
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

APP_NAME = os.environ.get("MODAL_APP_NAME", "exp14-rankings")
GPU = os.environ.get("MODAL_GPU", "A100")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "exp0-regrounding")
OUT_PREFIX = "exp14_run"

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


@app.function(image=image, gpu=GPU, cpu=4.0, timeout=3600,
              volumes={str(VOL_ROOT): VOL}, secrets=_secrets())
def run_elicit(opts: dict) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT)
    env["LLM_CACHE_DIR"] = str(VOL_ROOT / "hf_cache")
    env["LLM_DTYPE"] = "bfloat16"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(str(VOL_ROOT / "hf_cache"), exist_ok=True)

    cmd = ["python", "scripts/screening_optimizer.py", "--elicit",
           "--backend", "local", "--model", MODEL,
           "--out-dir", str(VOL_ROOT / opts.get("out_prefix", OUT_PREFIX))]
    if opts.get("mask_values"):
        cmd.append("--mask-values")
    print(shlex.join(cmd))
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env,
                          capture_output=True, text=True)
    VOL.commit()
    return {"returncode": proc.returncode,
            "seconds": time.monotonic() - t0,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-1500:]}


@app.local_entrypoint()
def main(mask_values: bool = False, out_prefix: str = "") -> None:
    print(f"== EXP-014 ranking elicitation on Modal ({GPU}) "
          f"mask_values={mask_values} ==")
    r = run_elicit.remote({"mask_values": mask_values,
                           "out_prefix": out_prefix or OUT_PREFIX})
    status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
    print(f"[elicit] {status}  {r['seconds']:.0f}s")
    print("  " + "\n  ".join(r["stdout_tail"].splitlines()[-14:]))
    if r["returncode"] != 0:
        print("  stderr: " + "\n  ".join(r["stderr_tail"].splitlines()[-15:]))
    print(f"\nPull: modal volume get {VOLUME_NAME} {OUT_PREFIX}/rankings.json "
          f"report/figures/{OUT_PREFIX}/rankings.json")
