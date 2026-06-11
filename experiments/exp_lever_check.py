"""Prerequisite lever-check: does any design lever move the OPTIMIZED METRIC
(METRIC 1 -- skill share) on a PLAYABLE board? GO/NO-GO gate before optimizer
wiring (spec section 5).

Metric per board = skill_share(cfg)['share'] (L3-vs-L0 head-to-head, the thing
the optimizer actually scores). The full-ladder round-robin monotonicity
(METRIC 2) is reported alongside as the validity diagnostic, never the lever
target. (Earlier revisions of this file used the now-RETIRED round-robin gap;
re-run to refresh results/lever_check.json on the corrected metric.)

Run: python -m experiments.exp_lever_check
"""
from __future__ import annotations
import json
import statistics
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from config import GameConfig
from optimizer.skill_expression import (
    skill_share, round_robin, standings_monotonicity,
    LADDER, RAND_NAME)
from optimizer.simulate import run_matchup
from optimizer.group_design import apply_design, GroupDesign

OUT = Path('results/lever_check.json')
NOISE = 0.05          # min PLAYABLE skill-share range to call GO (judge vs CI, do not inflate)
MIN_ROUNDS = 15


def playable(records: list, min_rounds: int = MIN_ROUNDS) -> bool:
    """Median game length >= min_rounds (rejects degenerate-fast/broken boards)."""
    if not records:
        return False
    return statistics.median(r['rounds'] for r in records) >= min_rounds


def _salary_variant(base, mult):
    out = deepcopy(base)
    out.settings = replace(out.settings,
                           mechanics=replace(out.settings.mechanics,
                                             salary=int(out.settings.mechanics.salary * mult)))
    return out


def _rent_variant(base, mult):
    groups = {c.group for c in base.cells if getattr(c, 'group', None)}
    return apply_design(base, GroupDesign(group_rent_mult={g: mult for g in groups}),
                        strict_groups=False)


# Salary + rent multiplier are sufficient to settle the GO gate (rent already
# moves the skill share well beyond noise). Cost-multiplier / board-shrink /
# drop-group sweeps (spec §5) are out of scope here; add them if a fuller sweep
# is wanted once optimizer wiring begins.
LEVERS = {
    'salary':    (_salary_variant, [0.5, 0.75, 1.0, 1.5, 2.0]),
    'rent_mult': (_rent_variant,   [0.5, 0.75, 1.0, 1.5, 2.0]),
}


def _playability_records(cfg, n_seeds, base_seed, max_turns):
    # representative matchup (top rung vs random) just to judge game length
    top = next(t for t in LADDER if t[0] == 'L3_generalist')
    # random anchor settings/class from L0; label 'rand' is fine (playability only reads rounds)
    _, rs, rc = LADDER[0]
    return run_matchup(cfg, [(top[0], top[1], top[2]), (RAND_NAME, rs, rc)],
                       n_games=n_seeds, base_seed=base_seed, max_turns=max_turns)


def run(n_seeds: int = 30, base_seed: int = 0, max_turns: int = 150) -> dict:  # spec §5 start; the GO gate run used 20
    base = GameConfig.from_yaml('default_config.yaml')
    report = {}
    for lever, (make, values) in LEVERS.items():
        points = []
        for v in values:
            cfg = make(base, v)
            m1 = skill_share(cfg, n_seeds=n_seeds, base_seed=base_seed, max_turns=max_turns)  # CRN: same base_seed every value
            standings = round_robin(cfg, n_seeds=n_seeds, base_seed=base_seed, max_turns=max_turns)
            recs = _playability_records(cfg, n_seeds, base_seed, max_turns)
            points.append({
                'value': v,
                'skill_share': m1['share'],
                'skill_share_se': m1['se'],
                'standings': standings,
                'monotone': standings_monotonicity(standings)['monotone'],
                'playable': playable(recs),
            })
        playable_sh = [p['skill_share'] for p in points if p['playable']]
        rng = (max(playable_sh) - min(playable_sh)) if len(playable_sh) >= 2 else 0.0
        report[lever] = {'points': points, 'playable_range': rng, 'moves': rng > NOISE}
    report['GO'] = any(v['moves'] for k, v in report.items() if isinstance(v, dict))
    return report


if __name__ == '__main__':
    rep = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print("\n=== Lever-check (METRIC 1 -- skill share) ===")
    for lever, r in rep.items():
        if lever == 'GO':
            continue
        print(f"\n{lever}: playable skill-share range = {r['playable_range']:.3f}  moves={r['moves']}")
        for p in r['points']:
            flag = '' if p['playable'] else '  [UNPLAYABLE]'
            print(f"  {p['value']:>5}: share={p['skill_share']:.3f} monotone={p['monotone']}{flag}")
    print(f"\nDECISION: {'GO - skill is optimizable on the real board' if rep['GO'] else 'NO-GO - fall back to pacing as primary (skill lives in the synthetic landscape)'}")
    print(f"written -> {OUT}")
