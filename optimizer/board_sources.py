"""The shared "five boards" used by the exploration probes.

hazard_curves.py and llm_character.py both want the same fixed set of boards so
their results line up. Resolving them here (rather than duplicating per script)
keeps that set in one place:

    default     canonical board
    GA-2p       best vector from a 2-player GA run-log (skipped if log absent)
    GA-3p       best vector from a 3-player GA run-log (skipped if log absent)
    salary x2   mini board, salary doubled
    drop Brown  mini board, Brown group removed
"""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import GameConfig
from monopoly.core.cell import Cell, Property
from optimizer.design_space import DesignSpace


def modify_salary(cfg: GameConfig, multiplier: float) -> GameConfig:
    """Deep-copy cfg with mechanics.salary scaled by multiplier."""
    out = deepcopy(cfg)
    out.settings = replace(
        out.settings,
        mechanics=replace(out.settings.mechanics,
                          salary=int(out.settings.mechanics.salary * multiplier)),
    )
    return out


def remove_group(cfg: GameConfig, group_name: str) -> GameConfig:
    """Deep-copy cfg with every Property in group_name turned into an inert Cell stub.

    Note: this keeps board length unchanged (stubs occupy the slot). For an
    actual shorter board, use DesignSpace mask decoding instead.
    """
    out = deepcopy(cfg)
    out.cells = [Cell(c.name) if isinstance(c, Property) and c.group == group_name else c
                 for c in out.cells]
    return out


def best_vec_from_run(run_path: str) -> Optional[List[float]]:
    """Lowest-score vector from a JSONL optimiser log, or None if missing/empty."""
    p = Path(run_path)
    if not p.exists():
        return None
    best = None
    inf = float('inf')
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if best is None or e.get('score', inf) < best.get('score', inf):
            best = e
    return list(best['vec']) if best is not None else None


def build_five_boards(canonical_config: str,
                      mini_config: str,
                      ga_2p: Optional[str] = None,
                      ga_3p: Optional[str] = None,
                      removal_direction: str = 'cheapest',
                      ) -> List[Tuple[str, GameConfig]]:
    """Resolve the (label, GameConfig) set. GA entries are skipped with a notice
    when their run-logs are absent, so this works on a fresh checkout."""
    canonical = GameConfig.from_yaml(canonical_config)
    mini = GameConfig.from_yaml(mini_config)
    boards: List[Tuple[str, GameConfig]] = [('default', canonical)]

    space = DesignSpace(canonical, removal_direction=removal_direction)
    for label, run_path in (('GA-2p', ga_2p), ('GA-3p', ga_3p)):
        if run_path is None:
            print(f'  [skip] {label}: no run path provided')
            continue
        vec = best_vec_from_run(run_path)
        if vec is None:
            print(f'  [skip] {label}: run log not found at {run_path}')
            continue
        boards.append((label, space.decode(np.asarray(vec))))

    boards.append(('salary x2', modify_salary(mini, 2.0)))
    boards.append(('drop Brown', remove_group(mini, 'Brown')))
    return boards
