#!/usr/bin/env python3
"""Round 1.5 driver — follow-up experiments motivated by round 1 findings.

Experiments (all use convergence-disabled protocol):
  T1: full_noconv   — T-FULL with convergence disabled, K=16
  T1: haz_noconv    — T-HAZ  with convergence disabled, K=16
  T3: full_unbounded — T-FULL with unbounded prompt + convergence disabled, K=16
  T3: haz_unbounded  — T-HAZ  with unbounded prompt + convergence disabled, K=16
  T5: haz_noconv (5 seeds) — HAZ sweep with more seeds for statistical power

T8 (exploit-resistance analysis) is a separate analysis script, not a run.

All conditions use K=16, convergence disabled, 3 boards x 3 seeds (T5: 5 seeds).
Estimated wall-clock: ~60 min on A10G with 7B model.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Tuple

from optimizer.timing import WallClockLogger


PHASES = (
    'T-SANITY',
    'TUNER:T-FULL_NOCONV',
    'TUNER:T-HAZ_NOCONV',
    'TUNER:T-FULL_UNBOUNDED',
    'TUNER:T-HAZ_UNBOUNDED',
    'TUNER:T-HAZ_NOCONV_5SEED',
)


def _set_determinism() -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _run(cmd, log_path=None) -> int:
    log_path = Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, 'w') if log_path else None
    try:
        print(f'  CMD: {" ".join(cmd)}')
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=7200)
        return p.returncode
    except subprocess.TimeoutExpired:
        return -1
    finally:
        if fh:
            fh.close()


def _release_gpu():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_sanity(args, logger, out_dir: Path) -> Tuple[bool, str]:
    """Quick sanity check — reuse round 1 sanity logic."""
    sanity_dir = out_dir / 'sanity'
    cmd = [sys.executable, 'scripts/llm_design_loop.py',
           '--backend', args.backend,
           '--ablation-condition', 'sanity',
           '--n-seeds', '1', '--K', '8',
           '--n-games', str(args.n_games),
           '--out-dir', str(sanity_dir)]
    if args.model:
        cmd += ['--model', args.model]
    rc = _run(cmd, log_path=sanity_dir / 'run.log')
    if rc != 0:
        return False, f'rc={rc}'
    import json
    summary = json.loads((sanity_dir / 'summary_sanity.json').read_text())
    scores = []
    for board, trajs in summary['trajectories_by_board'].items():
        for traj in trajs:
            scores = [it['score'] for it in traj]
    ok = all(scores[i] >= scores[i+1] for i in range(min(4, len(scores)-1)))
    return ok, f'sanity={ok}\nscores={scores}'


def _run_tuner_condition(args, out_dir: Path, condition: str,
                          K: int, n_seeds: int) -> Tuple[bool, str]:
    cond_dir = out_dir / 'tuner' / condition
    cmd = [sys.executable, 'scripts/llm_design_loop.py',
           '--backend', args.backend,
           '--ablation-condition', condition,
           '--n-seeds', str(n_seeds),
           '--K', str(K),
           '--n-games', str(args.n_games),
           '--out-dir', str(cond_dir)]
    if args.model:
        cmd += ['--model', args.model]
    rc = _run(cmd, log_path=cond_dir / 'run.log')
    if rc != 0:
        return False, f'rc={rc}'
    return True, f'{cond_dir}/'


def main():
    ap = argparse.ArgumentParser(description='Round 1.5 follow-up experiments')
    ap.add_argument('--out-dir', default='round1_5', help='output root')
    ap.add_argument('--backend', default='openai')
    ap.add_argument('--model', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--K', type=int, default=16,
                    help='Iterations per trajectory (default 16, up from round 1 K=8)')
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--haz-sweep-seeds', type=int, default=5,
                    help='Seeds for the HAZ sweep (T5)')
    ap.add_argument('--max-wall-seconds', type=int, default=7200,
                    help='Hard wall-clock cap (default 2 hours)')
    args = ap.parse_args()

    _set_determinism()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = WallClockLogger(out_dir=out_dir)
    print(f'Round 1.5 driver -- out_dir={out_dir}, K={args.K}, '
          f'n_seeds={args.n_seeds}, haz_sweep_seeds={args.haz_sweep_seeds}')

    status = {p: 'pending' for p in PHASES}
    if args.resume:
        try:
            import json
            existing = json.loads((out_dir / 'STATUS.md').read_text()
                                   .split('timing.jsonl`')[0])
        except Exception:
            pass

    # Phase 0: Sanity check
    phase = 'T-SANITY'
    if status[phase] != 'complete':
        with logger.time('phase', 'phase', phase=phase) as scratch:
            scratch['phase'] = phase
            ok, reason = _run_sanity(args, logger, out_dir)
            status[phase] = 'complete' if ok else 'failed'
            scratch['status'] = status[phase]
            scratch['reason'] = reason
            if not ok:
                print(f'SANITY FAILED: {reason}')
                logger.update_status(status)
                sys.exit(1)
        logger.update_status(status)
        _release_gpu()

    # Phase 1-4: TUNER conditions with K=16 and convergence disabled
    conditions_and_seeds = [
        ('full_noconv',    args.n_seeds),
        ('haz_noconv',     args.n_seeds),
        ('full_unbounded', args.n_seeds),
        ('haz_unbounded',  args.n_seeds),
    ]
    for cond, n_seeds in conditions_and_seeds:
        phase_key = f'TUNER:T-{cond.upper()}'
        if status.get(phase_key) in ('complete', 'skipped'):
            continue
        with logger.time('phase', 'phase', phase=phase_key) as scratch:
            scratch['phase'] = phase_key
            ok, reason = _run_tuner_condition(args, out_dir, cond,
                                               K=args.K, n_seeds=n_seeds)
            status[phase_key] = 'complete' if ok else 'failed'
            scratch['status'] = status[phase_key]
            scratch['reason'] = reason
        logger.update_status(status)
        _release_gpu()
        elapsed = time.perf_counter() - logger._t0
        if elapsed > args.max_wall_seconds:
            print(f'WALL-CLOCK CAP HIT after {elapsed:.0f}s; stopping.')
            break

    # Phase 5: HAZ sweep with 5 seeds for statistical power (T5)
    phase_key = 'TUNER:T-HAZ_NOCONV_5SEED'
    if status.get(phase_key) not in ('complete', 'skipped'):
        with logger.time('phase', 'phase', phase=phase_key) as scratch:
            scratch['phase'] = phase_key
            ok, reason = _run_tuner_condition(args, out_dir, 'haz_noconv',
                                               K=args.K,
                                               n_seeds=args.haz_sweep_seeds)
            status[phase_key] = 'complete' if ok else 'failed'
            scratch['status'] = status[phase_key]
            scratch['reason'] = reason
        logger.update_status(status)
        _release_gpu()

    print(f'\nRound 1.5 complete. Elapsed: {time.perf_counter() - logger._t0:.0f}s')
    for p, s in status.items():
        print(f'  {p}: {s}')


if __name__ == '__main__':
    main()
