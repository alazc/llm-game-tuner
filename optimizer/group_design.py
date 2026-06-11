"""The parametric-edit vocabulary the LLM designer emits, plus the eval helper.

A GroupDesign is a small JSON-serialisable record describing a board edit in
exactly the operations the designer loop is allowed to make. Keeping the schema
here (and the audit set that exercises it) means the audit tests the same
vocabulary the loop iterates over.

Primitives:
    salary_mult       float                 scale mechanics.salary
    drop_groups       list[str]             colour groups -> inert Cell stubs
    group_cost_mult   dict[group, float]    scale Property.cost_base per group
    group_rent_mult   dict[group, float]    scale rent_base AND every rent_house entry
    prop_overrides    dict[name, dict]      per-property escape hatch (applied last)

Referencing a group not on the target board is an error (strict_groups), so a
"raise rent on Indigo" issued to the mini board fails loudly instead of no-oping.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _replace
from typing import Dict, List

from config import GameConfig
from monopoly.core.cell import Cell, Property


@dataclass
class GroupDesign:
    salary_mult: float = 1.0
    drop_groups: List[str] = field(default_factory=list)
    group_cost_mult: Dict[str, float] = field(default_factory=dict)
    group_rent_mult: Dict[str, float] = field(default_factory=dict)
    prop_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    label: str = ''
    rationale: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GroupDesign':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _board_groups(cfg: GameConfig) -> List[str]:
    seen: List[str] = []
    for c in cfg.cells:
        if isinstance(c, Property) and c.group not in seen:
            seen.append(c.group)
    return seen


def board_groups(cfg) -> list:
    """Public: the valid Property-group names on `cfg` (the names apply_design
    validates against). The single source of truth for the prompt vocabulary AND
    the invalid-edit detector -- they must never diverge."""
    return _board_groups(cfg)


def referenced_groups(design) -> set:
    """Groups a design touches (drop/cost/rent) -- the set checked against
    board_groups for invalid-edit detection."""
    return (set(design.drop_groups) | set(design.group_cost_mult)
            | set(design.group_rent_mult))


def apply_design(cfg: GameConfig, design: GroupDesign,
                 strict_groups: bool = True) -> GameConfig:
    """Return a deep-copied cfg with the design applied."""
    out = deepcopy(cfg)
    present = set(_board_groups(out))

    if design.salary_mult != 1.0:
        salary = int(round(out.settings.mechanics.salary * design.salary_mult))
        out.settings = _replace(
            out.settings,
            mechanics=_replace(out.settings.mechanics, salary=salary))

    referenced = (set(design.drop_groups)
                  | set(design.group_cost_mult)
                  | set(design.group_rent_mult))
    missing = referenced - present
    if missing and strict_groups:
        raise ValueError(f'design references groups not on board: {sorted(missing)}; '
                         f'present: {sorted(present)}')

    drop = set(design.drop_groups)
    if drop:
        out.cells = [Cell(c.name) if (isinstance(c, Property) and c.group in drop) else c
                     for c in out.cells]

    if design.group_cost_mult or design.group_rent_mult:
        cells = []
        for c in out.cells:
            if not isinstance(c, Property):
                cells.append(c)
                continue
            cm = design.group_cost_mult.get(c.group, 1.0)
            rm = design.group_rent_mult.get(c.group, 1.0)
            if cm == 1.0 and rm == 1.0:
                cells.append(c)
                continue
            cells.append(Property(
                name=c.name,
                cost_base=int(round(c.cost_base * cm)),
                rent_base=int(round(c.rent_base * rm)),
                cost_house=c.cost_house,
                rent_house=tuple(int(round(r * rm)) for r in c.rent_house),
                group=c.group,
            ))
        out.cells = cells

    for name, ov in design.prop_overrides.items():
        for i, c in enumerate(out.cells):
            if isinstance(c, Property) and c.name == name:
                out.cells[i] = Property(
                    name=c.name,
                    cost_base=int(ov.get('cost_base', c.cost_base)),
                    rent_base=int(ov.get('rent_base', c.rent_base)),
                    cost_house=int(ov.get('cost_house', c.cost_house)),
                    rent_house=tuple(ov.get('rent_house', c.rent_house)),
                    group=c.group,
                )
                break

    return out


# Audit set: 5 designs picked up-front (no cherry-picking), all expressible on
# both the mini board ({Brown, Lightblue, Pink, Orange}) and canonical.
AUDIT_DESIGNS: List[GroupDesign] = [
    GroupDesign(label='baseline', rationale='no-op control'),
    GroupDesign(label='salary x1.5', salary_mult=1.5, rationale='probe pacing knob'),
    GroupDesign(label='drop Brown', drop_groups=['Brown'], rationale='probe structural removal'),
    GroupDesign(label='Orange rent x1.5', group_rent_mult={'Orange': 1.5},
                rationale='probe single-group rent inflation'),
    GroupDesign(label='salary x0.75 + Lightblue cost x1.5', salary_mult=0.75,
                group_cost_mult={'Lightblue': 1.5}, rationale='probe two-knob interaction'),
]


def evaluate_config(cfg, pool=None, matchups=None, n_games: int = 120,
                    base_seed: int = 0, max_turns: int = 200,
                    record_trajectory: bool = False) -> dict:
    """Score `cfg` on METRIC 1 — skill share (L3 vs L0 head-to-head).

    `pool`/`matchups` are accepted but IGNORED (the ladder is fixed; the style
    pool is dropped) so the loop call sites need no signature change. `n_games`
    is the seed count for the single matchup (passed straight as `n_seeds`).
    Returns the score the loop MINIMISES plus the fields the loop logs.
    If `record_trajectory` is True, also computes and returns the skill_curve.
    """
    from optimizer.skill_expression import (
        skill_share, skill_score, optimization_direction, G_STAR, BAND,
        money_share_trajectory)

    m = skill_share(cfg, n_seeds=n_games, base_seed=base_seed, max_turns=max_turns)
    share = m['share']
    direction = optimization_direction(share)
    curve = (money_share_trajectory(cfg, n_seeds=n_games, base_seed=base_seed,
                                    max_turns=max_turns)
             if record_trajectory else None)
    return {
        'score': skill_score(share),            # minimised; 0 inside the dead-zone
        'skill_share': share,
        'se': m['se'],
        'direction': direction,
        'metrics': {'skill_share': share, 'se': m['se'], 'n': m['n'],
                    'g_star': G_STAR, 'band': BAND, 'direction': direction},
        'shares': m['shares'],                  # flat per-game shares -> CI input
        'per_game_records': m['records'],       # L3-vs-L0 games (pacing readouts)
        'n_games_total': m['n'],
        'skill_curve': curve,
    }


def bootstrap_score_ci(shares, n_resamples: int = 500, seed: int = 0) -> dict:
    """95% bootstrap CI on the skill SCORE, resampling the flat per-game share
    list (METRIC 1 head-to-head games are i.i.d.). Each resample recomputes
    `skill_score(mean(resample))`. `shares` is `evaluate_config(...)['shares']`."""
    import numpy as np
    from optimizer.skill_expression import skill_score

    n = len(shares)
    if n == 0:
        return {'mean': skill_score(0.5), 'ci_lo': 0.0, 'ci_hi': 0.0, 'n_resamples': 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(shares, dtype=float)
    scores = [skill_score(float(arr[rng.integers(0, n, size=n)].mean()))
              for _ in range(n_resamples)]
    s = np.asarray(scores, dtype=float)
    return {'mean': float(s.mean()),
            'ci_lo': float(np.percentile(s, 2.5)),
            'ci_hi': float(np.percentile(s, 97.5)),
            'n_resamples': int(n_resamples)}


def monotonicity_gate(cfg, n_seeds: int = 30, base_seed: int = 0,
                      max_turns: int = 200, min_rank_corr: float = 0.5) -> dict:
    """METRIC 2 validity gate (DIAGNOSTIC, never optimised). Run the full-ladder
    round-robin and check the standings climb L0->L1->L2->L3.

    Apply this to ACCEPTED / FINAL boards (not inside the scored loop — it costs
    ~7x the Metric-1 budget for a number that never enters `score`). `passes` is
    `rank_corr >= min_rank_corr`; a board can reach `skill_share ~ g*` while its
    ladder quietly stops being monotone (e.g. L2 overtakes L3) — that board
    passes the score but must FAIL this gate.
    """
    from optimizer.skill_expression import round_robin, standings_monotonicity

    standings = round_robin(cfg, n_seeds=n_seeds, base_seed=base_seed, max_turns=max_turns)
    mono = standings_monotonicity(standings)
    rc = mono['rank_corr']
    passes = (rc == rc) and rc >= min_rank_corr          # rc==rc filters NaN
    return {'standings': standings, 'monotone': mono['monotone'],
            'inversions': mono['inversions'], 'rank_corr': rc, 'passes': passes}
