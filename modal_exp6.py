"""Modal launcher for EXP-006 (sign-aligned, discovery-fair optimum-sense isolation).

Fans out, on L4 GPUs, one task per (variant, model). Each task runs the synthetic
closed loop (scripts/synth_design_loop.py): the `synth` table feed + the `rand`
baseline always, plus the representation arm (`synth_curve`, `synth_verdict`) on the
PEAKED variant. The synthetic eval is instant (noise=0), so wall-time is dominated
by the LLM forward passes — high seed count is cheap.

  Qwen2.5-7B-Instruct  PRIMARY (the EXP-003/004 subject)
  Qwen2.5-1.5B-Instruct SECONDARY — instant evals + the synthetic task removes the
                       degeneracy escape hatch that made 1.5B uninterpretable on
                       Monopoly (it cannot reach the optimum by gutting; it must
                       actually optimise the dominant lever).

Each task writes JSONL to a Modal Volume under exp5_run/<variant>/<model_tag>; pull
with `modal volume get`. Returns ONLY tiny status (large returns break this build).

Run:
    modal run modal_exp5.py                       # prewarm both + all 6 tasks
    modal run modal_exp5.py --only peaked:7B      # one (variant:model) task
    modal run modal_exp5.py --models 7B           # 7B only (3 tasks)
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

APP_NAME = os.environ.get("MODAL_APP_NAME", "exp6-aligned-optimum-sense")
GPU = os.environ.get("MODAL_GPU", "A100")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "10800"))
STARTUP = int(os.environ.get("MODAL_STARTUP", "1800"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "exp0-regrounding")
OUT_PREFIX = "exp6_run"

MODELS = {
    "7B":   "Qwen/Qwen2.5-7B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}

# Sealed run config (prereg EXP-005).
N_INSTANCES = 5
N_SEEDS = 8
K = 8
FAMILY_BASE_SEED = 20260602
VARIANTS = ("peaked", "monotone", "separable")
# Fan-out keeps each GPU task to <=2 LLM conditions so it stays well under the
# timeout (7B ~= 1h per LLM condition at the canary's measured throughput). The
# PRIMARY phase runs the table feed + the rand baseline on every variant; the
# secondary representation ARM (curve/verdict, PEAKED only) runs as its own tasks.
PRIMARY_CONDS = ("synth", "rand")
ARM_CONDS = ("synth_curve", "synth_verdict")

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
                "*.output", "_exp0_dl"],
    )
    .run_commands(f"cd {shlex.quote(str(REMOTE_ROOT))} && pip install --no-deps -e .")
)

app = modal.App(APP_NAME)


def _hf_cache() -> str:
    return str(VOL_ROOT / "hf_cache")


@app.function(image=image, timeout=3600, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def download_model(model: str) -> str:
    from huggingface_hub import snapshot_download
    cache = _hf_cache()
    os.makedirs(cache, exist_ok=True)
    snapshot_download(model, cache_dir=cache)
    VOL.commit()
    return "cached: " + model


@app.function(image=image, gpu=GPU, cpu=8.0, timeout=TIMEOUT,
              startup_timeout=STARTUP, volumes={str(VOL_ROOT): VOL},
              secrets=_secrets())
def run_task(task: dict) -> dict:
    variant = task["variant"]
    model_tag = task["model_tag"]
    conds = task["conds"]
    model = MODELS[model_tag]
    out_dir = VOL_ROOT / OUT_PREFIX / variant / model_tag

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT)
    env["LLM_CACHE_DIR"] = _hf_cache()
    env["LLM_DTYPE"] = "bfloat16"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(_hf_cache(), exist_ok=True)

    logs = []
    rc_total = 0
    for cond in conds:
        cmd = [
            "python", "scripts/synth_design_loop.py",
            "--variant", variant, "--condition", cond,
            "--backend", ("heuristic" if cond == "rand" else "local"),
            "--n-instances", str(N_INSTANCES), "--n-seeds", str(N_SEEDS),
            "--K", str(K), "--family-base-seed", str(FAMILY_BASE_SEED),
            "--aligned",
            "--out-dir", str(out_dir),
        ]
        if cond != "rand":
            cmd += ["--model", model]
        print(f"[{variant}:{model_tag}:{cond}] {shlex.join(cmd)}")
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env,
                              capture_output=True, text=True)
        dt = time.monotonic() - t0
        rc_total |= proc.returncode
        logs.append({"cond": cond, "rc": proc.returncode, "s": round(dt),
                     "tail": proc.stdout[-600:] if proc.returncode == 0
                             else proc.stderr[-1500:]})
        VOL.commit()

    od = Path(str(out_dir))
    n_files = sum(1 for p in od.rglob("*") if p.is_file()) if od.exists() else 0
    return {"variant": variant, "model_tag": model_tag, "returncode": rc_total,
            "n_files": n_files, "logs": logs}


@app.local_entrypoint()
def main(phase: str = "all", models: str = "7B,1.5B",
         only: str = "", skip: str = "") -> None:
    """phase: primary | arm | all. only/skip: comma-sep 'variant:model_tag' specs
    (e.g. skip='separable:1.5B' for the already-run canary)."""
    model_tags = [m.strip() for m in models.split(",") if m.strip()]
    tasks: list[dict] = []
    if phase in ("primary", "all"):
        tasks += [{"variant": v, "model_tag": mt, "conds": list(PRIMARY_CONDS)}
                  for mt in model_tags for v in VARIANTS]
    if phase in ("arm", "all"):
        # one LLM condition per task so each stays ~1 GPU-condition under timeout
        tasks += [{"variant": "peaked", "model_tag": mt, "conds": [c]}
                  for mt in model_tags for c in ARM_CONDS]

    def key(t):
        return f'{t["variant"]}:{t["model_tag"]}'
    if only:
        want = {s.strip() for s in only.split(",")}
        tasks = [t for t in tasks if key(t) in want or t["variant"] in want
                 or t["model_tag"] in want]
    if skip:
        drop = {s.strip() for s in skip.split(",")}
        tasks = [t for t in tasks if key(t) not in drop]
    if not tasks:
        raise SystemExit("no task matches the filters")

    print(f"== EXP-006 (aligned) on Modal :: phase={phase} :: {len(tasks)} GPU({GPU}) task(s) "
          f"[{N_INSTANCES} inst x {N_SEEDS} seeds, K={K}] ==")
    for t in tasks:
        print(f"   - {key(t)}: {t['conds']}")
    for mt in sorted({t["model_tag"] for t in tasks}):
        print("pre-warming " + MODELS[mt] + " ...")
        print("  " + download_model.remote(MODELS[mt]))

    results = list(run_task.map(tasks))
    n_ok = 0
    for r in sorted(results, key=lambda x: (x["model_tag"], x["variant"])):
        status = "OK" if r["returncode"] == 0 else f"FAIL(rc={r['returncode']})"
        n_ok += int(r["returncode"] == 0)
        print(f"\n[{r['variant']}:{r['model_tag']}] {status}  {r['n_files']} files")
        for lg in r["logs"]:
            print(f"  - {lg['cond']}: rc={lg['rc']} {lg['s']}s")
            if lg["rc"] != 0:
                print("    stderr: " + "\n    ".join(lg["tail"].splitlines()[-12:]))

    print(f"\n{n_ok}/{len(results)} OK. Artifacts on Volume '{VOLUME_NAME}' under "
          f"/{OUT_PREFIX} -- pull with:")
    print(f"  modal volume get {VOLUME_NAME} {OUT_PREFIX} report/figures/{OUT_PREFIX}")
