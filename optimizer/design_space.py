"""Board design space: encode a board as a numeric vector, decode back to a GameConfig.

Genotype layout (66 dims):
    [0 : 22)    cost multipliers, one per colour-group property, in [0.5, 2.0]
    [22 : 44)   rent multipliers, one per colour-group property, in [0.5, 2.0]
    [44 : 66)   keep mask, one bit per colour-group property, {0, 1}

A 0 bit drops the property from the board entirely, so the decoded board is
*shorter* than 40 cells (shorter laps -> more salary per game). The engine
copes with variable length because landmark/card lookups go through
board.cell_index_by_name / board.next_cell_of_group rather than hardcoded
indices (see monopoly/core/board.py patch layer).

Railroads and utilities are not part of the design space (fixed in v1).

A legacy 45-dim encoding (44 multipliers + an integer property count that drove
cost-ranked removal) is still decodable so old GA run-logs stay reproducible.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import List

import numpy as np

from monopoly.core.cell import FreeParking, Property

MULT_LO, MULT_HI = 0.5, 2.0
N_PROPS = 22          # colour-group properties on the canonical board
NPROPS_LO = 8         # legacy property-count floor
MIN_KEPT = 8          # mask must keep at least this many, or boards stop forming monopolies


def _colour_group_indices(cells) -> List[int]:
    """Board indices of colour-group properties, in board order (no rail/utility)."""
    from monopoly.core.constants import RAILROADS, UTILITIES
    return [i for i, c in enumerate(cells)
            if isinstance(c, Property) and c.group not in (RAILROADS, UTILITIES)]


def _cost_rank(cells, cg_indices, direction: str) -> List[int]:
    """Positions into cg_indices ordered for legacy removal (cheapest/expensive/middle)."""
    costs = [(pos, cells[bi].cost_base) for pos, bi in enumerate(cg_indices)]
    if direction == 'cheapest':
        costs.sort(key=lambda x: x[1])
    elif direction == 'expensive':
        costs.sort(key=lambda x: -x[1])
    elif direction == 'middle':
        median = sorted(c for _, c in costs)[len(costs) // 2]
        costs.sort(key=lambda x: abs(x[1] - median))
    else:
        raise ValueError(f'unknown removal direction: {direction}')
    return [pos for pos, _ in costs]


class DesignSpace:
    """Vector <-> GameConfig codec over a fixed base board."""

    # Structural splits exposed for search (continuous vs binary tail).
    N_CONT = 2 * N_PROPS   # 44
    N_BIN = N_PROPS        # 22
    MIN_KEPT = MIN_KEPT

    def __init__(self, base_cfg, removal_direction: str = 'cheapest'):
        self.base_cfg = base_cfg
        self.removal_direction = removal_direction
        self._cg = _colour_group_indices(base_cfg.cells)
        if len(self._cg) != N_PROPS:
            raise AssertionError(
                f'base board has {len(self._cg)} colour-group properties, expected {N_PROPS}')
        self._rank = _cost_rank(base_cfg.cells, self._cg, removal_direction)
        self._defaults = [{
            'name': base_cfg.cells[bi].name,
            'cost_base': base_cfg.cells[bi].cost_base,
            'rent_base': base_cfg.cells[bi].rent_base,
            'cost_house': base_cfg.cells[bi].cost_house,
            'rent_house': tuple(base_cfg.cells[bi].rent_house),
            'group': base_cfg.cells[bi].group,
        } for bi in self._cg]

    # ---- dims / bounds ---- #

    @property
    def n_dims(self) -> int:
        return self.N_CONT + self.N_BIN  # 66

    def bounds(self):
        return [(MULT_LO, MULT_HI)] * self.N_CONT + [(0.0, 1.0)] * self.N_BIN

    def identity_vec(self) -> np.ndarray:
        """All multipliers 1.0, all properties kept -> the unmodified base board."""
        return np.ones(self.n_dims, dtype=np.float64)

    # ---- sampling / clipping ---- #

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        v = np.empty(self.n_dims, dtype=np.float64)
        v[:self.N_CONT] = rng.uniform(MULT_LO, MULT_HI, self.N_CONT)
        v[self.N_CONT:] = self._enforce_floor(rng.integers(0, 2, self.N_BIN).astype(np.float64), rng)
        return v

    def _enforce_floor(self, mask: np.ndarray, rng) -> np.ndarray:
        kept = int(mask.sum())
        if kept >= MIN_KEPT:
            return mask
        off = np.where(mask == 0)[0]
        mask = mask.copy()
        mask[rng.choice(off, size=MIN_KEPT - kept, replace=False)] = 1.0
        return mask

    def clip(self, v: np.ndarray) -> np.ndarray:
        out = np.asarray(v, dtype=np.float64).copy()
        out[:self.N_CONT] = np.clip(out[:self.N_CONT], MULT_LO, MULT_HI)
        out[self.N_CONT:] = np.round(np.clip(out[self.N_CONT:], 0.0, 1.0))
        if out[self.N_CONT:].sum() < MIN_KEPT:
            # Deterministic repair: derive the RNG seed from the vector content
            # (SHA256, not built-in hash) so clip() is reproducible across runs
            # regardless of PYTHONHASHSEED.
            seed = int.from_bytes(hashlib.sha256(out.tobytes()).digest()[:4], 'big')
            out[self.N_CONT:] = self._enforce_floor(out[self.N_CONT:], np.random.default_rng(seed))
        return out

    # ---- decoding ---- #

    def decode(self, vec):
        """Decode to a simulatable GameConfig (shrinks the board for dropped props)."""
        vec = np.asarray(vec, dtype=np.float64)
        if len(vec) == self.N_CONT + 1:
            return self._decode_legacy(vec)
        if len(vec) == self.n_dims:
            return self._decode_mask(self.clip(vec))
        raise ValueError(f'bad vec length {len(vec)}; expected {self.N_CONT + 1} (legacy) '
                         f'or {self.n_dims}')

    def _scaled_property(self, pos: int, cost_m: float, rent_m: float) -> Property:
        d = self._defaults[pos]
        return Property(
            d['name'],
            max(1, int(round(d['cost_base'] * cost_m))),
            max(1, int(round(d['rent_base'] * rent_m))),
            max(1, int(round(d['cost_house'] * cost_m))),
            tuple(max(1, int(round(r * rent_m))) for r in d['rent_house']),
            d['group'],
        )

    def _decode_mask(self, vec: np.ndarray):
        cost_m = vec[:N_PROPS]
        rent_m = vec[N_PROPS:self.N_CONT]
        mask = vec[self.N_CONT:]
        drop = {bi for pos, bi in enumerate(self._cg) if mask[pos] < 0.5}
        keep_mult = {bi: (float(cost_m[pos]), float(rent_m[pos]))
                     for pos, bi in enumerate(self._cg) if bi not in drop}

        cfg = deepcopy(self.base_cfg)
        cells = []
        for bi, cell in enumerate(cfg.cells):
            if bi in drop:
                continue  # board genuinely gets shorter
            if bi not in keep_mult:
                cells.append(cell)
                continue
            cm, rm = keep_mult[bi]
            cells.append(self._scaled_property(self._cg.index(bi), cm, rm))
        cfg.cells = cells
        return cfg

    def decode_as_substituted(self, vec):
        """Render-only decode: keep the board at full length, replacing dropped
        properties with FreeParking cells in place (instead of removing them).

        Useful for side-by-side visualisation against the canonical layout; the
        result is NOT meant for simulation (use decode() for that).
        """
        vec = np.asarray(vec, dtype=np.float64)
        if len(vec) == self.N_CONT + 1:
            return self._decode_legacy(vec)
        if len(vec) != self.n_dims:
            raise ValueError(f'bad vec length {len(vec)}')
        vec = self.clip(vec)
        cost_m = vec[:N_PROPS]
        rent_m = vec[N_PROPS:self.N_CONT]
        mask = vec[self.N_CONT:]
        cfg = deepcopy(self.base_cfg)
        cells = list(cfg.cells)
        for pos, bi in enumerate(self._cg):
            if mask[pos] < 0.5:
                cells[bi] = FreeParking(self._defaults[pos]['name'])
            else:
                cells[bi] = self._scaled_property(pos, float(cost_m[pos]), float(rent_m[pos]))
        cfg.cells = cells
        return cfg

    def _decode_legacy(self, vec: np.ndarray):
        out = vec.copy()
        out[:self.N_CONT] = np.clip(out[:self.N_CONT], MULT_LO, MULT_HI)
        n_props = int(np.clip(round(float(out[-1])), NPROPS_LO, N_PROPS))
        cost_m = out[:N_PROPS]
        rent_m = out[N_PROPS:self.N_CONT]
        removed = set(self._rank[:N_PROPS - n_props])

        cfg = deepcopy(self.base_cfg)
        cells = list(cfg.cells)
        for pos, bi in enumerate(self._cg):
            if pos in removed:
                cells[bi] = FreeParking(self._defaults[pos]['name'])
            else:
                cells[bi] = self._scaled_property(pos, float(cost_m[pos]), float(rent_m[pos]))
        cfg.cells = cells
        return cfg
