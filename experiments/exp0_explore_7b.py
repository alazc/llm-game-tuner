"""Read-only exploration-vs-failure classification of the EXP-003 7B logs.

No reruns, no model calls. Reconstructs, per 7B condition/seed/iteration:
  (1) edit magnitude  = L1 delta of the cumulative design vector over VALID groups
      (salary, per-group rent/cost mults as |m-1|, drops) -- effective board movement;
  (2) current skill_share per iter (metrics.skill_share), as distance above the band;
  (3) cumulative distinct levers touched (coverage);
  (4) rent lever: most-aggressive valid-group rent_mult ever applied, + direction.
Plus convergence: when (if) the model declared converged (haz/met/full can; mute can't).

Reports, per condition, which failure SIGNATURE the trajectories carry:
  - "premature-convergence" : edits go quiet early and the board is left above band;
  - "keeps-editing / fails-to-commit" : the model edits to the end (or even abandons
    better mid-run designs) but never commits the productive move.
NOTE: this only measures the model's *behavior*. Whether the failure is the model's
or the task's (is the band reachable at all?) is settled EXTERNALLY — by the lever
probes / 1.5B uniform-sweep that show the band IS reachable — not by this script.
Do not read the printed signature as the "Branch 1 / Branch 2" verdict; those tokens
denote a different (model-vs-task) axis and collide with the labels
an earlier draft of this file printed.
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict
from statistics import mean

from config import GameConfig
from optimizer.exp_boards import build_exp0_boards
from optimizer.group_design import board_groups

ROOT = 'report/figures/exp0_run2/7B'
CONDS = ['mute', 'haz', 'met', 'full']
BANDC = (0.57, 0.63)
VALID = {label: set(board_groups(cfg)) for label, cfg in build_exp0_boards()}
BOARD_OF = {'default': 'default', 'salary_x2': 'salary x2', 'gut_mid_tier': 'gut mid-tier'}


def design_vec(d, valid):
    """Flat dict of lever -> deviation, restricted to valid groups (effective)."""
    v = {}
    if d.get('salary_mult', 1.0) != 1.0:
        v['salary'] = d['salary_mult'] - 1.0
    for g, m in (d.get('group_rent_mult') or {}).items():
        if g in valid and m != 1.0:
            v[f'rent:{g}'] = m - 1.0
    for g, m in (d.get('group_cost_mult') or {}).items():
        if g in valid and m != 1.0:
            v[f'cost:{g}'] = m - 1.0
    for g in (d.get('drop_groups') or []):
        if g in valid:
            v[f'drop:{g}'] = 1.0
    return v


def l1_delta(a, b):
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def board_from_fname(name):
    for tag in ('default', 'salary_x2', 'gut_mid_tier'):
        if f'__{tag}__' in name:
            return tag
    return None


def analyse_traj(recs, valid):
    vecs = [design_vec(r.get('design') or {}, valid) for r in recs]
    mags = [l1_delta(vecs[i], vecs[i - 1]) for i in range(1, len(vecs))]  # per iter 1..K
    shares = [(r.get('metrics') or {}).get('skill_share') for r in recs]
    coverage = [len(vecs[i]) for i in range(len(vecs))]                   # cumulative levers
    # rent lever: most-aggressive (lowest) valid-group rent mult ever, and any wrong-dir
    rent_vals = [m for r in recs for g, m in ((r.get('design') or {}).get('group_rent_mult') or {}).items()
                 if g in valid]
    min_rent = min(rent_vals) if rent_vals else None
    any_up = any(m > 1.05 for m in rent_vals)
    # convergence: first honored convergence (converged_request True & not padded yet)
    conv_iter = next((r['iter'] for r in recs if r.get('converged_request')), None)
    # "stopped moving": first iter from which all subsequent edit mags ~0
    stop_iter = None
    for i in range(len(mags)):
        if all(m < 1e-9 for m in mags[i:]):
            stop_iter = i + 1
            break
    return dict(mags=mags, shares=shares, coverage=coverage, min_rent=min_rent,
                any_up=any_up, conv_iter=conv_iter, stop_iter=stop_iter,
                final_share=next((s for s in reversed(shares) if s is not None), None))


def main():
    per_cond = defaultdict(list)
    for cond in CONDS:
        for p in sorted(glob.glob(f'{ROOT}/{cond}/*.jsonl')):
            board = board_from_fname(os.path.basename(p))
            if board is None:
                continue
            valid = VALID[BOARD_OF[board]]
            per_cond[cond].append((board, analyse_traj(load(p), valid)))

    K = 8
    print(f'=== 7B exploration classification (band {BANDC}, n=3 seeds x 3 boards = 9 traj/cond) ===\n')
    for cond in CONDS:
        rows = per_cond[cond]
        # mean edit-magnitude per iteration (1..K)
        magT = [mean([r['mags'][i] for _, r in rows if i < len(r['mags'])]) for i in range(K)]
        # mean distance ABOVE band per iteration (share-0.63; <=0 means in/below band)
        def dist(i):
            xs = [r['shares'][i] - BANDC[1] for _, r in rows if i < len(r['shares']) and r['shares'][i] is not None]
            return mean(xs) if xs else float('nan')
        distT = [dist(i) for i in range(K + 1)]
        covT = [mean([r['coverage'][i] for _, r in rows if i < len(r['coverage'])]) for i in range(K + 1)]
        conv = [r['conv_iter'] for _, r in rows]
        stop = [r['stop_iter'] for _, r in rows]
        min_rents = [r['min_rent'] for _, r in rows if r['min_rent'] is not None]
        aggressive = sum(1 for _, r in rows if r['min_rent'] is not None and r['min_rent'] <= 0.7)
        timid_or_up = sum(1 for _, r in rows if (r['min_rent'] is None) or (r['min_rent'] > 0.9))
        any_up = sum(1 for _, r in rows if r['any_up'])
        converged = sum(1 for c in conv if c is not None)
        stopped_above = sum(1 for (_, r) in rows
                            if r['stop_iter'] is not None and r['final_share'] is not None
                            and r['final_share'] > BANDC[1])

        print(f'--- {cond} ---')
        print('  edit-mag/iter : ' + ' '.join(f'{m:5.2f}' for m in magT))
        print('  dist>band/iter: ' + ' '.join(f'{d:+5.2f}' for d in distT))
        print('  coverage/iter : ' + ' '.join(f'{c:5.1f}' for c in covT))
        print(f'  converged-declared: {converged}/{len(rows)} seeds (iters {sorted(c for c in conv if c)})')
        print(f'  stopped-moving while ABOVE band: {stopped_above}/{len(rows)} seeds (stop iters {sorted(s for s in stop if s)})')
        print(f'  rent lever: min valid-group rent applied per traj = '
              f'{sorted(round(x,2) for x in min_rents)}')
        print(f'     aggressive reductions (<=0.7): {aggressive}/{len(rows)}  |  '
              f'timid/none (>0.9 or never): {timid_or_up}/{len(rows)}  |  pushed rent UP (wrong dir): {any_up}/{len(rows)}')
        # per-condition behavioral signature (NOT the model-vs-task verdict; see docstring)
        late_mag = mean(magT[3:]) if len(magT) > 3 else 0.0
        premature = (aggressive <= len(rows) // 3) and (late_mag < 0.25) and (distT[-1] > 0.02)
        sig = ('premature-convergence (edits go quiet early, left above band)' if premature
               else 'keeps-editing / fails-to-commit (aggressive to the end, never commits the move)')
        print(f'  >> SIGNATURE: {sig} '
              f'(late-iter mean mag={late_mag:.2f}, end dist>band={distT[-1]:+.2f}, aggressive={aggressive}/{len(rows)})\n')


if __name__ == '__main__':
    main()
