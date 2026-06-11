"""EXP-005 post-hoc follow-up (EXPLORATORY — NOT a sealed test).

EXP-005's pre-registered PEAKED-vs-MONOTONE optimum-sense contrast was swamped
because the model rarely engages the dominant lever at all (lever-id failure) and
the controls were confounded. This script asks the narrower, conditional question
the sealed batteries could not: *given* the model DID engage the dominant dim, does
it SETTLE at the optimum or OVERSHOOT / ABANDON it?

The clean optimum-sense signal here is structure-agnostic (works for both interior
PEAKED and boundary MONOTONE optima):
  abandoned_better = the dominant dim got CLOSER to x*_dom at some iter than where
                     it ended -> the model found a better lever value and then left
                     it (the EXP-004 "abandons better mid-run designs" signature).
Plus PEAKED-only overshoot: crossed x*_dom and ended on the far side (a monotone
"keep pushing the lever" policy with no stopping point).

Caveat printed in the output: this is post-hoc on small engaged-subsamples; treat
as a LEAD for a sealed redesign, not a confirmatory result.

Run: python -m experiments.exp5_posthoc [ROOT]   (default report/figures/exp5_run)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean

import numpy as np

from synthetic.landscape import make_landscape, evaluate
from experiments.exp5_analyze import FNAME_RE, load, _dir_aligned

ENGAGED_MOVE = 0.10        # cumulative |delta| on the dominant dim to count as "engaged"
EPS = 1e-6


def _f_dom(spec, dom, val):
    """Deficit contributed by the dominant dim alone at value `val`."""
    from synthetic.landscape import _dim_deficit
    kind, p = spec.components[dom]
    return _dim_deficit(kind, p, float(val))


def traj_optimum_sense(recs, spec):
    dom = spec.dominant_dim
    if dom is None:
        return None
    xs = float(spec.x_star[dom])
    vals = [float(r['x'][dom]) for r in recs]
    start = vals[0]
    moved_total = sum(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
    if moved_total < ENGAGED_MOVE:
        return {'engaged': False}

    dist = [abs(v - xs) for v in vals]
    dist_min, dist_final = min(dist), dist[-1]
    fdoms = [_f_dom(spec, dom, v) for v in vals]
    fdom_min, fdom_final = min(fdoms), fdoms[-1]

    # signed edits on the dominant dim
    signs = [np.sign(vals[i] - vals[i - 1]) for i in range(1, len(vals))
             if abs(vals[i] - vals[i - 1]) > EPS]
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    flip_rate = flips / max(1, len(signs) - 1) if len(signs) >= 2 else 0.0

    crossed = any((vals[i - 1] - xs) * (vals[i] - xs) < 0 for i in range(1, len(vals)))
    # overshoot-and-persist: crossed the optimum and ended on the opposite side of start
    overshoot_persist = crossed and ((start - xs) * (vals[-1] - xs) < 0)

    return {
        'engaged': True,
        'abandoned_better': dist_min < dist_final - 0.03,    # got closer, then left
        'closest_dist': dist_min, 'final_dist': dist_final,
        'fdom_min': fdom_min, 'fdom_final': fdom_final,
        'fdom_left_on_table': fdom_final - fdom_min,
        'settled_near_opt': dist_final <= 0.10,              # ended at the optimum
        'crossed': crossed, 'overshoot_persist': overshoot_persist,
        'flip_rate': flip_rate,
    }


def collect(root):
    cells = defaultdict(lambda: defaultdict(list))   # (variant,model)->cond->list
    for p in glob.glob(os.path.join(root, '**', '*.jsonl'), recursive=True):
        m = FNAME_RE.search(os.path.basename(p))
        if not m or m.group('variant') not in ('peaked', 'monotone'):
            continue
        d = os.path.dirname(p)
        spec = make_landscape(m.group('variant'), seed=int(m.group('ls')),
                              aligned=_dir_aligned(d))
        model = os.path.basename(d)
        r = traj_optimum_sense(load(p), spec)
        if r is not None:
            cells[(m.group('variant'), model)][m.group('cond')].append(r)
    return cells


def _rate(rows, key):
    e = [r for r in rows if r.get('engaged')]
    return (sum(bool(r[key]) for r in e) / len(e)) if e else float('nan'), len(e)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'report/figures/exp5_run'
    print('=== EXP-005 post-hoc (exploratory) :: ' + root + ' ===')
    print('Conditional on ENGAGING the dominant dim (cumulative move >= '
          f'{ENGAGED_MOVE}); optimum-sense among engaged trajectories.\n')
    cells = collect(root)
    for model in sorted({m for (_v, m) in cells}):
        print(f'--- model={model} (synth condition) ---')
        for variant in ('peaked', 'monotone'):
            rows = cells.get((variant, model), {}).get('synth', [])
            n = len(rows)
            eng = [r for r in rows if r.get('engaged')]
            ne = len(eng)
            if ne == 0:
                print(f'  {variant:9s}: engaged 0/{n}  (cannot read optimum-sense)')
                continue
            ab, _ = _rate(rows, 'abandoned_better')
            settled, _ = _rate(rows, 'settled_near_opt')
            over, _ = _rate(rows, 'overshoot_persist')
            print(f'  {variant:9s}: engaged {ne}/{n}  |  settled-near-opt {settled:.2f}  '
                  f'abandoned-better {ab:.2f}  overshoot-persist {over:.2f}')
            print(f'             closest-dist {mean(r["closest_dist"] for r in eng):.3f} -> '
                  f'final-dist {mean(r["final_dist"] for r in eng):.3f}  |  '
                  f'f_dom min {mean(r["fdom_min"] for r in eng):.3f} -> '
                  f'final {mean(r["fdom_final"] for r in eng):.3f} '
                  f'(left-on-table {mean(r["fdom_left_on_table"] for r in eng):.3f})  '
                  f'flip-rate {mean(r["flip_rate"] for r in eng):.2f}')
        print()
    print('CAVEAT: post-hoc, small engaged subsamples (typically 10-30/60). A LEAD for a '
          'sealed sign-aligned/discovery-fair redesign, NOT a confirmatory optimum-sense result.')


if __name__ == '__main__':
    main()
