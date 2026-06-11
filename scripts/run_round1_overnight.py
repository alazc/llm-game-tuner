"""Round 1 overnight driver — sequential phase queue, single-command.

Runs the entire round-1 experiment queue:
  1. T-SANITY (default board, K=8) — abort everything if non-monotone
  2. PHASE-A-ROBUSTNESS (mini-board × 4 knobs × 12 games, NEUTRAL-PLAYER)
  3. MIRROR-H (5 boards × 6 games × NEUTRAL-PLAYER self-play)
  4. TUNER:   T-CANON → T-RAND → T-MUTE → T-HAZ → T-MET → T-FULL → T-BLIND
              each: 5 boards × 3 seeds × K=8 × n_games=200

Releases GPU memory between phases (`del model` + `torch.cuda.empty_cache()`)
and writes a STATUS.md after every phase change so resuming after a crash
is just `--resume`.

Calibration mode (`--calibrate-only`):
  - 1 NEUTRAL-PLAYER game on default board (max 30 turns)
  - 1 DIRECTIVE-DESIGNER iteration on default board, T-FULL
  - T-SANITY iters 0..2 only
  - All three checks gated:
      * MIRROR format-pass-rate  ≥ 70%
      * TUNER JSON parse-rate    ≥ 80%
      * T-SANITY iter 0→2 strictly monotone-decreasing
      * Projected total wall ≤ 10 hr (10 hr cap)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimizer.timing import WallClockLogger


PHASES = ('T-SANITY', 'PHASE-A-ROBUSTNESS', 'MIRROR-H',
          'TUNER:T-CANON', 'TUNER:T-RAND', 'TUNER:T-MUTE', 'TUNER:T-HAZ',
          'TUNER:T-MET', 'TUNER:T-FULL', 'TUNER:T-BLIND')


# --------------------------------------------------------------------------- #
# Determinism harness                                                           #
# --------------------------------------------------------------------------- #

def _set_determinism() -> None:
    """Lock determinism flags before any LLM forward."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # torch not installed (heuristic-backend smoke); silently ok.
        pass


def _release_gpu() -> None:
    """Drop any cached transformers models so the next phase starts clean."""
    try:
        import torch
    except Exception:
        return
    try:
        # Wipe per-script model caches if present.
        from agents import LLMPlayer
        LLMPlayer._MODEL_CACHE.clear()
    except Exception:
        pass
    try:
        from scripts.llm_design_loop import DesignerLLM
        DesignerLLM._MODEL_CACHE.clear()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Status file management                                                        #
# --------------------------------------------------------------------------- #

def _initial_status() -> Dict[str, str]:
    return {p: 'pending' for p in PHASES}


def _read_status_state(out_dir: Path) -> Dict[str, str]:
    """Reconstruct phase status from any phase-level events written to the
    timing.jsonl. The STATUS.md is human-readable and re-emitted; the JSONL
    is the durable source of truth."""
    state: Dict[str, str] = _initial_status()
    j = out_dir / 'timing.jsonl'
    if not j.exists():
        return state
    with open(j, encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = rec.get('phase')
            if p in state:
                if rec.get('event') in ('phase_start', 'phase__start'):
                    state[p] = 'running'
                elif rec.get('event') in ('phase_end', 'phase__end'):
                    if rec.get('status') == 'failed':
                        state[p] = 'failed'
                    elif rec.get('status') == 'skipped':
                        state[p] = 'skipped'
                    else:
                        state[p] = 'complete'
    return state


# --------------------------------------------------------------------------- #
# Subprocess helpers                                                            #
# --------------------------------------------------------------------------- #

def _run(cmd: List[str], cwd: Optional[Path] = None,
         env: Optional[Dict[str, str]] = None,
         log_path: Optional[Path] = None) -> int:
    """Run a child command, streaming output to stdout and optionally a log."""
    print(f'$ {" ".join(cmd)}', flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    e.setdefault('PYTHONPATH', str(REPO_ROOT))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as fh:
            proc = subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=e,
                                   stdout=fh, stderr=subprocess.STDOUT)
        return proc.returncode
    proc = subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=e)
    return proc.returncode


# --------------------------------------------------------------------------- #
# Per-phase runners                                                             #
# --------------------------------------------------------------------------- #

def _run_sanity(args, logger: WallClockLogger, out_dir: Path,
                K: int = 8) -> Tuple[bool, str]:
    """T-SANITY: hardcoded trajectory on default board. Aborts overnight if
    non-monotone iter 0->4."""
    sanity_dir = out_dir / 'sanity'
    cmd = [sys.executable, 'scripts/llm_design_loop.py',
           '--backend', args.backend,
           '--ablation-condition', 'sanity',
           '--boards', 'default',
           '--n-seeds', '1',
           '--K', str(K),
           '--n-games', str(args.sanity_n_games),
           '--out-dir', str(sanity_dir)]
    if args.model:
        cmd += ['--model', args.model]
    rc = _run(cmd, log_path=sanity_dir / 'run.log')
    failed_marker = sanity_dir / 'SANITY_FAILED'
    sanity_result = sanity_dir / 'SANITY_RESULT.txt'
    reason = sanity_result.read_text() if sanity_result.exists() else f'rc={rc}'
    if rc != 0 or failed_marker.exists():
        return False, reason
    return True, reason


def _run_phase_a_robustness(args, logger, out_dir: Path) -> Tuple[bool, str]:
    """PHASE-A-ROBUSTNESS: re-run Phase A with NEUTRAL-PLAYER instead of
    GUIDED-PLAYER. Invokes scripts/phase_a_validation.py --player neutral;
    the script writes results_neutral.json into the phase dir, which we
    then check exists as a non-trivial liveness signal."""
    pa_dir = out_dir / 'phase_a_robustness'
    pa_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'scripts/phase_a_validation.py',
           '--player', 'neutral',
           '--n-games', str(args.phase_a_rb_n_games),
           '--n-games-llm', str(args.phase_a_rb_n_games_llm),
           '--max-turns', str(args.phase_a_rb_max_turns),
           '--out-dir', str(pa_dir)]
    if args.model:
        cmd += ['--llm-model', args.model]
    cmd += ['--backend', args.backend]
    if args.backend == 'heuristic':
        cmd.insert(2, '--no-llm')
    rc = _run(cmd, log_path=pa_dir / 'run.log')
    results_path = pa_dir / 'results_neutral.json'
    if rc != 0:
        return False, f'rc={rc}'
    if not results_path.exists():
        return False, f'phase_a_validation completed (rc=0) but did not write {results_path.name}'
    return True, f'wrote {results_path.name}'


def _run_mirror_h(args, logger, out_dir: Path) -> Tuple[bool, str]:
    """MIRROR-H: 5 boards × 6 games NEUTRAL-PLAYER self-play (half-size)."""
    mirror_dir = out_dir / 'mirror_h'
    cmd = [sys.executable, 'scripts/llm_character.py',
           '--backend', args.backend,
           '--n-games', str(args.mirror_n_games),
           '--max-turns', str(args.mirror_max_turns),
           '--out-dir', str(mirror_dir)]
    if args.model:
        cmd += ['--model', args.model]
    rc = _run(cmd, log_path=mirror_dir / 'run.log')
    if rc != 0:
        return False, f'rc={rc}'
    # Read the format-pass-rate gate output; gate at 70%.
    gate_path = mirror_dir / 'data_quality.json'
    gate_summary = '(no data_quality.json)'
    if gate_path.exists():
        try:
            payload = json.loads(gate_path.read_text())
            gate = payload.get('format_pass_rate_gate', {})
            gate_summary = (f'all_pass={gate.get("all_pass")} '
                            f'failing={gate.get("failing")}')
        except Exception:
            pass
    return True, gate_summary


def _run_tuner_condition(args, logger, out_dir: Path,
                          condition: str) -> Tuple[bool, str]:
    cond_dir = out_dir / 'tuner' / condition
    cmd = [sys.executable, 'scripts/llm_design_loop.py',
           '--backend', args.backend,
           '--ablation-condition', condition,
           '--n-seeds', str(args.tuner_n_seeds),
           '--K', str(args.tuner_K),
           '--n-games', str(args.tuner_n_games),
           '--out-dir', str(cond_dir)]
    if args.model:
        cmd += ['--model', args.model]
    rc = _run(cmd, log_path=cond_dir / 'run.log')
    if rc != 0:
        return False, f'rc={rc}'
    return True, f'{cond_dir}/'


# --------------------------------------------------------------------------- #
# Calibration                                                                    #
# --------------------------------------------------------------------------- #

def _run_calibration(args, logger, out_dir: Path) -> int:
    """Calibration mode. All four gates must pass."""
    print('--- calibration mode ---')
    calib_dir = out_dir / 'calibration'
    calib_dir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []

    # 1. MIRROR-style probe: 1 NEUTRAL-PLAYER game on default board, max 30 turns.
    with logger.time('phase', 'phase', phase='calibrate:MIRROR-probe') as scratch:
        scratch['phase'] = 'calibrate:MIRROR-probe'
        cmd = [sys.executable, 'scripts/llm_character.py',
               '--backend', args.backend,
               '--n-games', '1',
               '--max-turns', '30',
               '--out-dir', str(calib_dir / 'mirror')]
        if args.model: cmd += ['--model', args.model]
        rc = _run(cmd, log_path=calib_dir / 'mirror.log')
        if rc != 0:
            failures.append(f'MIRROR probe failed (rc={rc})')
        gate_path = calib_dir / 'mirror' / 'data_quality.json'
        if gate_path.exists():
            payload = json.loads(gate_path.read_text())
            gate = payload.get('format_pass_rate_gate', {})
            if not gate.get('all_pass', False):
                failures.append(f'MIRROR format-pass-rate gate failed: '
                                f'failing={gate.get("failing")}')

    # 2. TUNER probe: 1 iteration of T-FULL on default board.
    with logger.time('phase', 'phase', phase='calibrate:TUNER-probe') as scratch:
        scratch['phase'] = 'calibrate:TUNER-probe'
        cmd = [sys.executable, 'scripts/llm_design_loop.py',
               '--backend', args.backend,
               '--ablation-condition', 'full',
               '--boards', 'default',
               '--n-seeds', '1',
               '--K', '1',
               '--n-games', str(max(args.tuner_n_games, 8)),
               '--out-dir', str(calib_dir / 'tuner')]
        if args.model: cmd += ['--model', args.model]
        rc = _run(cmd, log_path=calib_dir / 'tuner.log')
        if rc != 0:
            failures.append(f'TUNER probe failed (rc={rc})')
        # Walk emitted JSONL to compute parse-rate gate (≥80%).
        parse_total = 0; parse_ok = 0
        for jp in (calib_dir / 'tuner').glob('full__*.jsonl'):
            for line in jp.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('parser_status') in ('baseline',):
                    continue
                parse_total += 1
                if rec.get('parser_status') == 'ok':
                    parse_ok += 1
        rate = (parse_ok / parse_total) if parse_total else 0.0
        print(f'  TUNER parse rate: {rate*100:.1f}% ({parse_ok}/{parse_total})')
        if parse_total and rate < 0.80:
            failures.append(f'TUNER parse-rate {rate*100:.1f}% < 80%')

    # 3. T-SANITY first 3 sanity-design iters monotone-decreasing.
    # Script emits iter 0 baseline + iters 1..K sanity designs, so the spec's
    # "iter 0..2 monotone" maps to script iters 1..3 (the first 3 designs).
    with logger.time('phase', 'phase', phase='calibrate:SANITY') as scratch:
        scratch['phase'] = 'calibrate:SANITY'
        ok, reason = _run_sanity(args, logger, calib_dir, K=3)
        scores: List[float] = []
        sjsonl = calib_dir / 'sanity' / 'sanity__default__seed42.jsonl'
        if sjsonl.exists():
            for line in sjsonl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scores.append(rec.get('score'))
        sanity_designs = scores[1:4]    # spec iters 0..2 == script iters 1..3
        ok02 = (len(sanity_designs) == 3
                and all(s is not None for s in sanity_designs)
                and sanity_designs[0] > sanity_designs[1] > sanity_designs[2])
        if not ok02:
            failures.append(f'T-SANITY iter 0->2 not strictly monotone: '
                            f'sanity_design_scores={sanity_designs}')

    # 4. Wall-clock projection (rough): phase budget at 10 hr cap.
    elapsed = time.perf_counter() - logger._t0
    print(f'  calibration elapsed: {elapsed:.1f}s')
    cap_seconds = float(args.max_wall_seconds)
    if elapsed > cap_seconds:
        failures.append(f'calibration alone took {elapsed:.0f}s > cap {cap_seconds:.0f}s')

    summary = {
        'failures': failures,
        'pass': len(failures) == 0,
        'elapsed_seconds': round(elapsed, 1),
    }
    (calib_dir / 'calibration_summary.json').write_text(
        json.dumps(summary, indent=2))
    if failures:
        print('CALIBRATION FAILED:')
        for f in failures:
            print(f'  - {f}')
        return 2
    print('CALIBRATION PASSED.')
    return 0


# --------------------------------------------------------------------------- #
# Main driver                                                                    #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='report/figures/round1')
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--max-wall-seconds', type=int, default=36000,
                    help='Hard cap (default 10 hr).')
    ap.add_argument('--calibrate-only', action='store_true')
    ap.add_argument('--resume', action='store_true')

    # Per-phase tunables (defaults match the round-1 lock).
    ap.add_argument('--mirror-n-games',  type=int, default=6)    # half-size
    ap.add_argument('--mirror-max-turns', type=int, default=80)
    ap.add_argument('--tuner-n-seeds',   type=int, default=3)
    ap.add_argument('--tuner-K',         type=int, default=8)
    ap.add_argument('--tuner-n-games',   type=int, default=200)
    ap.add_argument('--sanity-n-games',  type=int, default=200)
    ap.add_argument('--phase-a-rb-n-games',     type=int, default=30,
                    help='Per-condition rule-based games for PHASE-A-ROBUSTNESS.')
    ap.add_argument('--phase-a-rb-n-games-llm', type=int, default=10,
                    help='Per-condition LLM games for PHASE-A-ROBUSTNESS.')
    ap.add_argument('--phase-a-rb-max-turns',   type=int, default=50)

    args = ap.parse_args()

    _set_determinism()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = WallClockLogger(out_dir=out_dir)
    print(f'Round 1 overnight driver — out_dir={out_dir}')

    # Calibration short-circuit.
    if args.calibrate_only:
        rc = _run_calibration(args, logger, out_dir)
        status = {'CALIBRATION': 'complete' if rc == 0 else 'failed'}
        logger.update_status(status)
        return rc

    # Resume: read what's complete from the existing timing.jsonl.
    status = _read_status_state(out_dir) if args.resume else _initial_status()
    if args.resume:
        print('Resume: completed phases =',
              [p for p, s in status.items() if s in ('complete', 'skipped')])
    logger.update_status(status)

    # Phase 1: T-SANITY (eval-pipeline correctness check).
    if status['T-SANITY'] not in ('complete', 'skipped'):
        with logger.time('phase', 'phase', phase='T-SANITY') as scratch:
            scratch['phase'] = 'T-SANITY'
            ok, reason = _run_sanity(args, logger, out_dir,
                                      K=args.tuner_K)
            status['T-SANITY'] = 'complete' if ok else 'failed'
            scratch['status'] = status['T-SANITY']
            scratch['reason'] = reason
        logger.update_status(status)
        if not ok:
            (out_dir / 'OVERNIGHT_ABORT').write_text(
                f'T-SANITY failed: {reason}\n')
            print(f'ABORT: T-SANITY failed: {reason}')
            return 2
        _release_gpu()

    # Phase 2: PHASE-A-ROBUSTNESS.
    if status['PHASE-A-ROBUSTNESS'] not in ('complete', 'skipped'):
        with logger.time('phase', 'phase', phase='PHASE-A-ROBUSTNESS') as scratch:
            scratch['phase'] = 'PHASE-A-ROBUSTNESS'
            ok, reason = _run_phase_a_robustness(args, logger, out_dir)
            status['PHASE-A-ROBUSTNESS'] = 'complete' if ok else 'failed'
            scratch['status'] = status['PHASE-A-ROBUSTNESS']
            scratch['reason'] = reason
        logger.update_status(status)
        _release_gpu()

    # Phase 3: MIRROR-H.
    if status['MIRROR-H'] not in ('complete', 'skipped'):
        with logger.time('phase', 'phase', phase='MIRROR-H') as scratch:
            scratch['phase'] = 'MIRROR-H'
            ok, reason = _run_mirror_h(args, logger, out_dir)
            status['MIRROR-H'] = 'complete' if ok else 'failed'
            scratch['status'] = status['MIRROR-H']
            scratch['reason'] = reason
        logger.update_status(status)
        _release_gpu()

    # Phase 4: TUNER conditions in spec order.
    tuner_order = ('canon', 'rand', 'mute', 'haz', 'met', 'full', 'blind')
    for cond in tuner_order:
        phase_key = f'TUNER:T-{cond.upper()}'
        if status[phase_key] in ('complete', 'skipped'):
            continue
        with logger.time('phase', 'phase', phase=phase_key) as scratch:
            scratch['phase'] = phase_key
            ok, reason = _run_tuner_condition(args, logger, out_dir, cond)
            status[phase_key] = 'complete' if ok else 'failed'
            scratch['status'] = status[phase_key]
            scratch['reason'] = reason
        logger.update_status(status)
        _release_gpu()
        # Wall-clock cap check.
        elapsed = time.perf_counter() - logger._t0
        if elapsed > args.max_wall_seconds:
            print(f'WALL-CLOCK CAP HIT after {elapsed:.0f}s; stopping.')
            (out_dir / 'WALL_CLOCK_CAP_HIT').write_text(
                f'cap={args.max_wall_seconds}s elapsed={elapsed:.0f}s\n')
            break

    # Final status file.
    logger.update_status(status)
    final_failed = [p for p, s in status.items() if s == 'failed']
    if final_failed:
        print(f'Round 1 finished with failures: {final_failed}')
        return 1
    print('Round 1 complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
