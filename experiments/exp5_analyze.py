"""EXP-005 analyzer - ground-truth steering battery (prior-free landscape).

Read-only over scripts/synth_design_loop.py artifacts. Because the landscape
structure is KNOWN (reconstructed from the landscape seed in each filename), the
batteries are FACTS, not inferences. For each variant x model:

  Battery 1  Lever identification  - fraction of edits landing on the DOMINANT dim
             vs decoy/inert; fraction of trajectories that EVER move the dominant
             dim substantially. Compared to the random-edit baseline (rand cond).
  Battery 2  Direction stability   - sign-consistency of dominant-dim edits (toward
             x*_dom), and the flip rate (mirrors the rent-direction flip).
  Battery 3  Optimum-sense/stopping- final distance-to-optimum, in-band rate,
             overshoot (crossed x*_dom), oscillation, convergence-declaration rate,
             stop-AT-optimum rate.

Headline proportions carry 95% CIs (Wilson) + the minimum detectable effect (MDE)
at the achieved n. Then the PEAKED/MONOTONE/SEPARABLE contrasts are classified
against sealed sub-predictions A (optimum-sense) and B (lever-id), plus the
SEPARABLE-success and representation-null controls.

Run:  python -m experiments.exp5_analyze [ROOT]
      (ROOT default report/figures/exp5_run; scans <variant>/<model>/ subdirs)
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional

import numpy as np

from synthetic.landscape import G_STAR, BAND, make_landscape, evaluate

FNAME_RE = re.compile(r'(?P<cond>[a-z_]+)__(?P<variant>peaked|monotone|separable)__'
                      r'inst(?P<inst>\d+)__ls(?P<ls>\d+)__seed(?P<seed>\d+)\.jsonl$')
SUBSTANTIAL = 0.10        # a "substantial" move on the dominant dim (>= 0.10 total)


def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a proportion -> (phat, lo, hi, half_width)."""
    if n == 0:
        return (float('nan'),) * 4
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, center - half, center + half, half


def mde_proportion(n: int, base: float = 0.5, z_a: float = 1.96, z_b: float = 0.84):
    """Minimum detectable absolute difference in a proportion (two-sided a=.05,
    power .80) at sample size n per arm, around `base`."""
    if n == 0:
        return float('nan')
    return (z_a + z_b) * math.sqrt(2 * base * (1 - base) / n)


# --------------------------------------------------------------------------- #
# Per-trajectory battery from a known spec                                      #
# --------------------------------------------------------------------------- #

def analyse_traj(recs: List[dict], spec) -> dict:
    dom = spec.dominant_dim
    names = spec.dim_names
    dom_name = names[dom] if dom is not None else None
    decoy = {names[j] for j in spec.decoy_dims}
    inert = {names[j] for j in spec.inert_dims}

    edits_dom = edits_decoy = edits_inert = edits_total = 0
    for r in recs[1:]:                          # skip baseline
        for d in (r.get('changed_dims') or []):
            edits_total += 1
            if d == dom_name:
                edits_dom += 1
            elif d in decoy:
                edits_decoy += 1
            elif d in inert:
                edits_inert += 1

    # dominant-dim trajectory of values + signed edits toward x*_dom
    dom_signs, correct_dir = [], []
    moved_dom_total = 0.0
    crossed = False
    if dom is not None:
        xs = spec.x_star[dom]
        prev_val = recs[0]['x'][dom]
        for r in recs[1:]:
            val = r['x'][dom]
            dv = val - prev_val
            if abs(dv) > 1e-6:
                dom_signs.append(math.copysign(1, dv))
                correct_dir.append(math.copysign(1, dv) == math.copysign(1, xs - prev_val))
                moved_dom_total += abs(dv)
                if (prev_val - xs) * (val - xs) < 0:    # straddled the optimum
                    crossed = True
            prev_val = val

    flips = sum(1 for i in range(1, len(dom_signs)) if dom_signs[i] != dom_signs[i - 1])
    flip_rate = flips / max(1, len(dom_signs) - 1) if len(dom_signs) >= 2 else 0.0
    dir_consistency = (mean(correct_dir) if correct_dir else float('nan'))

    final = recs[-1]
    x_final = np.array(final['x'])
    dist_opt = float(np.linalg.norm(x_final - spec.x_star))
    dom_dist = (abs(x_final[dom] - spec.x_star[dom]) if dom is not None else float('nan'))
    converged = any(r.get('converged_request') for r in recs)
    return {
        'edits_dom': edits_dom, 'edits_decoy': edits_decoy, 'edits_inert': edits_inert,
        'edits_total': edits_total,
        'moved_dom_substantial': moved_dom_total >= SUBSTANTIAL,
        'moved_dom_total': moved_dom_total,
        'dir_consistency': dir_consistency, 'flip_rate': flip_rate,
        'n_dom_edits': len(dom_signs), 'crossed_opt': crossed,
        'final_in_band': bool(final['in_band']), 'final_score': final['score'],
        'final_dist_opt': dist_opt, 'final_dom_dist': dom_dist,
        'converged': converged,
        'stop_at_opt': bool(final['in_band']) and converged,
    }


_ALIGNED_CACHE: Dict[str, bool] = {}


def _dir_aligned(d: str) -> bool:
    """Read the cell's FAMILY.json `aligned` flag (default False) so the
    reconstructed spec matches the geometry the run actually used."""
    if d not in _ALIGNED_CACHE:
        fam = os.path.join(d, 'FAMILY.json')
        try:
            meta = json.load(open(fam, encoding='utf-8'))
            _ALIGNED_CACHE[d] = bool(meta[0].get('aligned', False)) if meta else False
        except (FileNotFoundError, ValueError, IndexError, KeyError):
            _ALIGNED_CACHE[d] = False
    return _ALIGNED_CACHE[d]


def spec_from_fname(variant: str, ls_seed: int, aligned: bool = False):
    return make_landscape(variant, seed=ls_seed, aligned=aligned)


# --------------------------------------------------------------------------- #
# Aggregate one (variant, model) cell                                           #
# --------------------------------------------------------------------------- #

def collect(root: str):
    """Return cells[(variant, model)][cond] = list of per-traj battery dicts."""
    cells: Dict = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(root, '**', '*.jsonl'), recursive=True):
        m = FNAME_RE.search(os.path.basename(p))
        if not m:
            continue
        variant = m.group('variant')
        cond = m.group('cond')
        # model = the directory name just above the file (.../<variant>/<model>/file)
        d = os.path.dirname(p)
        model = os.path.basename(d)
        spec = spec_from_fname(variant, int(m.group('ls')), aligned=_dir_aligned(d))
        cells[(variant, model)][cond].append(analyse_traj(load(p), spec))
    return cells


def _agg(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None
          and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return mean(xs) if xs else float('nan')


def report_cell(variant, model, conds):
    print(f'\n{"="*70}\n=== {variant.upper()} :: {model} ===')
    rand = conds.get('rand', [])
    for cond in ('synth', 'synth_curve', 'synth_verdict'):
        rows = conds.get(cond, [])
        if not rows:
            continue
        n = len(rows)
        # Battery 1: lever-id
        tot_dom = sum(r['edits_dom'] for r in rows)
        tot_all = sum(r['edits_total'] for r in rows)
        frac_dom = tot_dom / tot_all if tot_all else float('nan')
        moved = sum(r['moved_dom_substantial'] for r in rows)
        p_moved, lo_m, hi_m, hw_m = wilson(moved, n)
        # Battery 3: optimum-sense
        inband = sum(r['final_in_band'] for r in rows)
        p_ib, lo_ib, hi_ib, hw_ib = wilson(inband, n)
        stop = sum(r['stop_at_opt'] for r in rows)
        conv = sum(r['converged'] for r in rows)
        crossed = sum(r['crossed_opt'] for r in rows)
        print(f'\n  [{cond}] n={n}   (MDE@n for a proportion ~ {mde_proportion(n):.2f})')
        print(f'   B1 lever-id   : edits on dominant = {tot_dom}/{tot_all} '
              f'(frac {frac_dom:.3f}); moved-dominant-substantially '
              f'{moved}/{n} = {p_moved:.2f} [{lo_m:.2f},{hi_m:.2f}]')
        print(f'      edits by role: dominant={tot_dom} decoy={sum(r["edits_decoy"] for r in rows)} '
              f'inert={sum(r["edits_inert"] for r in rows)}')
        print(f'   B2 direction  : dom-edit dir-consistency={_agg(rows,"dir_consistency"):.2f}  '
              f'flip-rate={_agg(rows,"flip_rate"):.2f}  (n dom-edits/traj={_agg(rows,"n_dom_edits"):.1f})')
        print(f'   B3 optimum    : final in-band {inband}/{n}={p_ib:.2f} [{lo_ib:.2f},{hi_ib:.2f}]  '
              f'stop-at-opt {stop}/{n}  converged {conv}/{n}  crossed-opt {crossed}/{n}')
        print(f'      final score mean={_agg(rows,"final_score"):.4f}  '
              f'dist-to-opt mean={_agg(rows,"final_dist_opt"):.3f}  '
              f'dom-dist mean={_agg(rows,"final_dom_dist"):.3f}')
    if rand:
        n = len(rand)
        tot_dom = sum(r['edits_dom'] for r in rand)
        tot_all = sum(r['edits_total'] for r in rand)
        frac = tot_dom / tot_all if tot_all else float('nan')
        ib = sum(r['final_in_band'] for r in rand)
        print(f'\n  [rand baseline] n={n}: dominant-edit frac={frac:.3f}  '
              f'final in-band {ib}/{n}  moved-dom-substantially '
              f'{sum(r["moved_dom_substantial"] for r in rand)}/{n}')
    return conds


# --------------------------------------------------------------------------- #
# Classification against sealed sub-predictions                                 #
# --------------------------------------------------------------------------- #

def _inband_rate(rows):
    return (sum(r['final_in_band'] for r in rows) / len(rows)) if rows else float('nan')


def _dom_frac(rows):
    td = sum(r['edits_dom'] for r in rows); ta = sum(r['edits_total'] for r in rows)
    return td / ta if ta else float('nan')


def classify(cells):
    print(f'\n{"#"*70}\n# CLASSIFICATION vs sealed sub-predictions (per model)\n{"#"*70}')
    models = sorted({m for (_v, m) in cells})
    for model in models:
        def cell(v, c='synth'):
            return cells.get((v, model), {}).get(c, [])
        peaked, mono, sep = cell('peaked'), cell('monotone'), cell('separable')
        if not peaked:
            continue
        ib_p, ib_m, ib_s = _inband_rate(peaked), _inband_rate(mono), _inband_rate(sep)
        stop_p = (sum(r['stop_at_opt'] for r in peaked) / len(peaked)) if peaked else float('nan')
        df_p = _dom_frac(peaked)
        rand_df = _dom_frac(cells.get(('peaked', model), {}).get('rand', []))
        moved_p = (sum(r['moved_dom_substantial'] for r in peaked) / len(peaked)) if peaked else float('nan')
        dir_p = _agg(peaked, 'dir_consistency')

        print(f'\n--- model={model} ---')
        print(f'  in-band rate: PEAKED={ib_p:.2f}  MONOTONE={ib_m:.2f}  SEPARABLE={ib_s:.2f}')
        print(f'  PEAKED stop-at-optimum={stop_p:.2f}; dominant-edit frac={df_p:.3f} '
              f'(random={rand_df:.3f}); moved-dominant={moved_p:.2f}; dir-consistency={dir_p:.2f}')

        # Sub-prediction A: optimum-sense failure reproduces; PEAKED << MONOTONE.
        a_repro = (stop_p < 0.34) and (not math.isnan(ib_m)) and (ib_m - ib_p > 0.15)
        a_cured = (ib_p > 0.66)
        print('  [A optimum-sense] ' + (
            'REPRODUCES - PEAKED fails to settle, MONOTONE markedly better'
            if a_repro else
            'CURED - PEAKED settles fine on the clean interior-optimum map' if a_cured else
            'MIXED - see numbers (neither clean reproduce nor clean cure)'))

        # Sub-prediction B: lever-id is the discriminating test.
        b_repro = (not math.isnan(rand_df)) and (df_p <= rand_df + 0.05)
        b_cured = (moved_p > 0.66) and (df_p > rand_df + 0.10)
        print('  [B lever-id] ' + (
            'REPRODUCES - dominant-dim edits at/below random; decoy/inert fixation'
            if b_repro else
            'CURED - reliably finds & moves the dominant lever on the clean map '
            '(failure was map-opacity, not steering)' if b_cured else
            'MIXED - see numbers'))

        # Controls.
        sep_ok = ib_s > 0.66
        print('  [control SEPARABLE] ' + (
            f'PASS - greedy-friendly map solved ({ib_s:.2f} in-band)' if sep_ok else
            f'FAIL - fails even on separable ({ib_s:.2f}); finding collapses to "weak optimiser"'))

        # Representation null (on PEAKED): synth vs curve vs verdict.
        reps = {c: _inband_rate(cells.get(('peaked', model), {}).get(c, []))
                for c in ('synth', 'synth_curve', 'synth_verdict')
                if cells.get(('peaked', model), {}).get(c)}
        if len(reps) >= 2:
            spread = max(reps.values()) - min(reps.values())
            print(f'  [control representation] in-band by render {reps} '
                  f'-> spread {spread:.2f} ' + ('(NULL holds)' if spread <= 0.15 else '(rendering MOVES outcome)'))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'report/figures/exp5_run'
    print(f'=== EXP-005 steering battery :: {root} '
          f'(g*={G_STAR}, band+/-{BAND}, substantial-move>={SUBSTANTIAL}) ===')
    cells = collect(root)
    if not cells:
        print('  (no trajectories found)'); return
    for (variant, model) in sorted(cells):
        report_cell(variant, model, cells[(variant, model)])
    classify(cells)


if __name__ == '__main__':
    main()
