"""EXP: the two named metrics on the default board.

METRIC 1 -- SKILL SHARE (optimized): direct L3-vs-L0 2-player wealth share, with
error bars, starting at N=30 and bumping N if the standard error is too wide to
place the share confidently relative to g*=0.60. Compared to g* to fix the
optimization direction; scored by skill_score = max(0,|share-g*|-band).
METRIC 2 -- LADDER MONOTONICITY (diagnostic): full-ladder round-robin standings,
the validity check that competence climbs L0->L1->L2->L3 (rank correlation +
inversion count). Reported, never optimized.

Run: python -m experiments.exp_head_to_head
"""
from __future__ import annotations
import json
from pathlib import Path

from config import GameConfig
from optimizer.skill_expression import (
    skill_share, skill_score, optimization_direction, G_STAR, BAND,
    round_robin, standings_monotonicity, RUNG_ORDER)

OUT = Path('results/head_to_head.json')
# Bump N until the 95% half-width (~2*SE) clears this; keeps the share placed
# confidently on one side of g* (or honestly straddling it).
SE_TARGET = 0.03
N_LADDER = [30, 60, 120]
MAX_TURNS = 200


def run(base_seed: int = 0) -> dict:
    cfg = GameConfig.from_yaml('default_config.yaml')

    # --- METRIC 1 (optimized): skill share, escalate N on wide error bars ---
    m1 = None
    for n in N_LADDER:
        m1 = skill_share(cfg, n_seeds=n, base_seed=base_seed, max_turns=MAX_TURNS)
        if m1['se'] == m1['se'] and 2 * m1['se'] <= SE_TARGET:  # se==se filters NaN
            break

    share, se = m1['share'], m1['se']
    direction = optimization_direction(share)

    # --- METRIC 2 (diagnostic): full-ladder round-robin monotonicity ---
    standings = round_robin(cfg, n_seeds=m1['n'], base_seed=base_seed, max_turns=MAX_TURNS)
    mono = standings_monotonicity(standings)

    return {
        'metric1_skill_share': {
            'name': 'skill_share(L3 vs L0)',
            'share': share, 'se': se, 'n': m1['n'],
            'ci95': [share - 1.96 * se, share + 1.96 * se] if se == se else None,
            'scale': '0.5-centered wealth share',
            'score': skill_score(share), 'band': [G_STAR - BAND, G_STAR + BAND],
        },
        'g_star': G_STAR,
        'direction': direction,
        'metric2_ladder_monotonicity': {
            'standings': standings,
            'order': RUNG_ORDER,
            'monotone': mono['monotone'],
            'inversions': mono['inversions'],
            'rank_corr': mono['rank_corr'],
            'note': 'validity gate (is the ladder a real ruler) -- NEVER optimized',
        },
    }


if __name__ == '__main__':
    rep = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))

    o = rep['metric1_skill_share']
    print("\n=== METRIC 1 -- SKILL SHARE (optimized scalar): L3 vs L0, default board ===")
    print(f"  share = {o['share']:.4f}  +/- {o['se']:.4f} SE   (n={o['n']} CRN seeds)")
    if o['ci95']:
        print(f"  95% CI = [{o['ci95'][0]:.4f}, {o['ci95'][1]:.4f}]")
    print(f"  g* = {rep['g_star']:.2f}  score = {o['score']:.4f} "
          f"(dead-zone [{o['band'][0]:.2f}, {o['band'][1]:.2f}])")
    print(f"  ->  DIRECTION: {rep['direction'].upper()}")
    msg = {'reduce': 'share > g*  ->  REDUCE skill toward target',
           'increase': 'share < g*  ->  INCREASE skill toward target',
           'on_target': 'share in band  ->  on target'}[rep['direction']]
    print(f"  {msg}")

    d = rep['metric2_ladder_monotonicity']
    print("\n=== METRIC 2 -- LADDER MONOTONICITY (diagnostic, never optimized) ===")
    print("  standings: " + "  ".join(f"{k}={d['standings'][k]:.3f}" for k in d['order']))
    rc = d['rank_corr']
    print(f"  rank_corr = {rc:.3f}   monotone={d['monotone']} (inversions={d['inversions']})")
    print(f"\nwritten -> {OUT}")
