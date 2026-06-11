"""EXP-002 starting boards: canonical anchor + named single-family transforms.

Purpose-built for the skill-expression re-establishment gate, REPLACING the
legacy build_five_boards/GA set (which depended on absent GA run-logs and imported
the old balance objective's structure). Each start is the canonical 40-cell board
plus ONE documented transform, validated so REDUCE is a real task (skill_share 95%
CI clears g*+band=0.63). No GA / balance-optimized boards.

Validated starting skill_share (n=200, CRN base_seed=0, 95% CI). The dominant
reduce-lever is RENT for all three (the engine's sole skill channel) -- so this
gate's "more info doesn't help / curve wins" claim is RENT-LEVER-SPECIFIC;
lever-generality is Experiment 4's job, NOT this gate's:

  default (anchor)   0.723 [0.665, 0.781]  familiar prior; baseline difficulty
  salary x2          0.800 [0.751, 0.850]  MISDIRECTION: perturbed lever = salary,
                                           but the fix is rent (decoy) -- reverting
                                           the visible knob is the WRONG move
  gut mid-tier       0.831 [0.786, 0.875]  drop Orange+Red+Yellow: unfamiliar gutted
                                           mid-board, far from g*, prior disrupted
"""
from __future__ import annotations

from typing import List, Tuple

from config import GameConfig
from optimizer.board_sources import modify_salary, remove_group

MID_TIER_GROUPS = ('Orange', 'Red', 'Yellow')

# label -> validated starting metrics (measured, for prereg/provenance).
VALIDATED_STARTS = {
    'default':      {'share': 0.723, 'ci': (0.665, 0.781), 'lever': 'rent',
                     'role': 'anchor / familiar prior'},
    'salary x2':    {'share': 0.800, 'ci': (0.751, 0.850), 'lever': 'rent',
                     'role': 'misdirection (decoy lever = salary; fix = rent)'},
    'gut mid-tier': {'share': 0.831, 'ci': (0.786, 0.875), 'lever': 'rent',
                     'role': 'prior-disruption (unfamiliar gutted mid-board)'},
}


def build_exp0_boards(canonical_config: str = 'default_config.yaml'
                      ) -> List[Tuple[str, GameConfig]]:
    """The 3 validated EXP-002 starts as (label, GameConfig). `default` is the
    untransformed anchor (kept under that label so the T-SANITY default-only
    filter still resolves it)."""
    canon = GameConfig.from_yaml(canonical_config)
    gutted = canon
    for g in MID_TIER_GROUPS:
        gutted = remove_group(gutted, g)
    return [
        ('default', canon),
        ('salary x2', modify_salary(canon, 2.0)),
        ('gut mid-tier', gutted),
    ]
