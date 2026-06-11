"""Synthetic design landscape family for EXP-005 (prior-free mechanism test).

A board is replaced by a design vector x in [0,1]^d. The optimiser's job mirrors
the Monopoly harness exactly: drive a displayed QUALITY scalar into a target band
around g*. We expose quality = g* + f(x), where f(x) >= 0 is the total DEFICIT
(0 at the global optimum). So quality starts ABOVE g* (direction REDUCE) and the
band-score is the same skill_score the Monopoly loop minimises:
    score = max(0, |quality - g*| - band) = max(0, f(x) - band).

Per-dim component kinds (the structure the prereg pre-registers):
  dominant_peaked   Gaussian well, depth A_dom, centre x* INTERIOR. Far from x*
                    the gradient is ~0 (weak early signal); near x* it is steep
                    (strong). Over- AND under-shoot both raise f -> interior
                    optimum. The dominant lever HOLDS the gain (A_dom > band) and
                    is SUFFICIENT (decoys total <= band), so 'find & move the
                    dominant dim correctly' == success.
  dominant_monotone Linear ramp to a BOUNDARY optimum (more-is-better). Control:
                    a monotone value model matches it.
  decoy             One-sided ramp: a quick SHALLOW gain then a hard plateau (the
                    fixation trap). Small total amplitude; never sufficient.
  active (separable) Like a decoy but comparable-magnitude and several of them sum
                    to the whole deficit -> a per-dim greedy reaches the band.
  inert             f does not depend on it (analog of Monopoly's inert cost lever).

Noise is 0 by default (the point: the map is clean, so a failure cannot be blamed
on opacity). Build a FAMILY (varying which dim dominates, x*, well width, decoy
count) so the result is not a rigged single function.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Target + band on the displayed quality scale — IDENTICAL to the Monopoly metric
# (optimizer.skill_expression.G_STAR / BAND) so the feed and score carry over.
G_STAR = 0.60
BAND = 0.03

VARIANTS = ('peaked', 'monotone', 'separable')

# Fixed magnitudes (locked in the prereg). amp_dominant > BAND (the lever holds the
# gain and is necessary); decoy_total <= BAND (decoys are insufficient, the trap).
N_DIMS = 12
AMP_DOMINANT = 0.12
DECOY_TOTAL = 0.02
DECOY_RAMP = 0.15
WELL_W_RANGE = (0.08, 0.12)
XSTAR_RANGE = (0.40, 0.60)        # interior optimum location for PEAKED
DOM_START_OFFSET = 0.50           # start this far from x*_dom (on the flat plateau)
N_DECOYS = 3
N_ACTIVE_SEPARABLE = 4
SEP_RAMP = 0.30
SEP_START = 0.10
EPS = 1e-9

# EXP-006 sign-aligned ("aligned=True") parameters. Productive dims start HIGH and
# their optima sit BELOW the start, so "reduce quality" == "lower the dim". The
# PEAKED dominant uses a smooth quadratic well (informative gradient) with an
# INTERIOR optimum, so flooring the dim overshoots it.
A_XSTAR_RANGE = (0.30, 0.50)      # interior optimum for the aligned PEAKED dominant
A_WELL_W = 0.40                   # quadratic half-width (informative across the range)
A_DOM_START_OFFSET = 0.40         # aligned dominant starts this far ABOVE x*_dom
A_PROD_START = 0.90               # decoy / active / monotone-dominant start (high)


# --------------------------------------------------------------------------- #
# Per-dim deficit components                                                    #
# --------------------------------------------------------------------------- #

def _f_peaked(x: float, x_star: float, amp: float, w: float) -> float:
    return amp * (1.0 - math.exp(-((x - x_star) / w) ** 2))


def _f_monotone(x: float, boundary: float, amp: float) -> float:
    # optimum at `boundary` in {0,1}; linear, more-is-better toward the boundary.
    dist = abs(x - boundary)
    return amp * dist


def _f_ramp(x: float, x0: float, ramp: float, amp: float) -> float:
    # quick gain from x0 then hard plateau at 0 (one-sided, optimum HIGH: x >= x0+ramp).
    return amp * min(1.0, max(0.0, 1.0 - (x - x0) / ramp))


def _f_ramp_down(x: float, x0: float, ramp: float, amp: float) -> float:
    # EXP-006 sign-aligned ramp: starts high at x0, LOWERING x reduces the deficit;
    # optimum LOW (x <= x0 - ramp), then plateau at 0. So "reduce the objective" ==
    # "lower the dim".
    return amp * min(1.0, max(0.0, 1.0 - (x0 - x) / ramp))


def _f_peaked_smooth(x: float, x_star: float, amp: float, w: float) -> float:
    # EXP-006 sign-aligned PEAKED well: a SMOOTH quadratic basin (informative
    # gradient everywhere, no flat plateau), interior optimum x_star, capped at amp.
    # Over- AND under-shoot both raise f -> the model must STOP at x_star.
    return amp * min(1.0, ((x - x_star) / w) ** 2)


# --------------------------------------------------------------------------- #
# Spec                                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class LandscapeSpec:
    variant: str
    n_dims: int
    dim_names: List[str]
    dominant_dim: Optional[int]
    decoy_dims: List[int]
    inert_dims: List[int]
    active_dims: List[int]                 # all non-inert dims
    x_star: np.ndarray                     # known global optimum (f == 0)
    x_start: np.ndarray                    # out-of-band starting vector
    components: Dict[int, Tuple[str, dict]]  # dim -> (kind, params)
    amp_dominant: Optional[float] = None
    noise_std: float = 0.0
    seed: int = 0
    aligned: bool = False

    def amp_of(self, j: int) -> float:
        kind, p = self.components.get(j, ('inert', {}))
        return float(p.get('amp', 0.0))

    def name_to_dim(self) -> Dict[str, int]:
        return {n: i for i, n in enumerate(self.dim_names)}


def _dim_deficit(kind: str, p: dict, x: float) -> float:
    if kind == 'dominant_peaked':
        return _f_peaked(x, p['x_star'], p['amp'], p['w'])
    if kind == 'dominant_peaked_smooth':                 # EXP-006 aligned
        return _f_peaked_smooth(x, p['x_star'], p['amp'], p['w'])
    if kind == 'dominant_monotone':
        return _f_monotone(x, p['boundary'], p['amp'])
    if kind in ('decoy', 'active'):
        return _f_ramp(x, p['x0'], p['ramp'], p['amp'])
    if kind in ('decoy_down', 'active_down'):            # EXP-006 aligned
        return _f_ramp_down(x, p['x0'], p['ramp'], p['amp'])
    return 0.0                              # inert


# --------------------------------------------------------------------------- #
# evaluate                                                                      #
# --------------------------------------------------------------------------- #

def evaluate(spec: LandscapeSpec, x) -> dict:
    """f(x), the displayed quality, the band-score, direction, and per-dim deficit.
    Pure when noise_std == 0 (the default)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    per_dim: Dict[str, float] = {}
    f = 0.0
    for j, (kind, p) in spec.components.items():
        c = _dim_deficit(kind, p, float(x[j]))
        per_dim[spec.dim_names[j]] = c
        f += c
    if spec.noise_std > 0.0:                # off by default
        rng = np.random.default_rng(int(abs(hash((tuple(np.round(x, 6)),))) % (2 ** 32)))
        f = max(0.0, f + float(rng.normal(0.0, spec.noise_std)))
    quality = G_STAR + f
    score = max(0.0, abs(quality - G_STAR) - BAND)
    if quality > G_STAR + EPS:
        direction = 'reduce'
    elif quality < G_STAR - EPS:
        direction = 'increase'
    else:
        direction = 'on_target'
    return {'f': f, 'quality': quality, 'score': score,
            'direction': direction, 'in_band': abs(quality - G_STAR) <= BAND + EPS,
            'per_dim': per_dim}


# --------------------------------------------------------------------------- #
# Edit vocabulary                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class SyntheticDesign:
    """The parametric-edit vocabulary the designer LLM emits: absolute sets of
    named dimensions (mirrors GroupDesign's per-group multipliers). Unknown names
    are ignored (the analog of strict_groups=False)."""
    dim_set: Dict[str, float] = field(default_factory=dict)
    label: str = ''
    rationale: str = ''

    def to_dict(self) -> dict:
        return {'dim_set': dict(self.dim_set), 'label': self.label,
                'rationale': self.rationale}


def apply_design(spec: LandscapeSpec, design: SyntheticDesign,
                 base: Optional[np.ndarray] = None) -> np.ndarray:
    """Return base (default x_start) with the design's named dims set + clipped."""
    x = (spec.x_start if base is None else base).copy()
    name_to_dim = spec.name_to_dim()
    for name, val in design.dim_set.items():
        j = name_to_dim.get(str(name))
        if j is None:
            continue
        try:
            x[j] = float(val)
        except (TypeError, ValueError):
            continue
    return np.clip(x, 0.0, 1.0)


def referenced_dims(spec: LandscapeSpec, design: SyntheticDesign) -> set:
    names = set(spec.dim_names)
    return {str(k) for k in design.dim_set} - names      # invalid (off-board) names


# --------------------------------------------------------------------------- #
# Family construction                                                           #
# --------------------------------------------------------------------------- #

def _names(n: int) -> List[str]:
    return [f'd{i:02d}' for i in range(n)]


def make_landscape(variant: str, seed: int, n_dims: int = N_DIMS,
                   aligned: bool = False) -> LandscapeSpec:
    """One landscape instance. Placement of the dominant/decoy/inert dims is
    randomised by `seed` so position carries no information.

    aligned=False -> EXP-005 geometry (productive dims start LOW, optima above; the
                     PEAKED dominant is a flat-plateau Gaussian well).
    aligned=True  -> EXP-006 SIGN-ALIGNED geometry: productive dims start HIGH and
                     their optima sit BELOW, so "reduce quality" == "lower the dim";
                     the PEAKED dominant is a SMOOTH interior quadratic well, so the
                     naive 'floor everything' policy overshoots it (but solves the
                     monotone variants)."""
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}')
    rng = np.random.default_rng(seed)
    names = _names(n_dims)
    x_star = np.full(n_dims, 0.5)
    x_start = np.full(n_dims, 0.5)
    components: Dict[int, Tuple[str, dict]] = {}

    def _inert(j):
        components[j] = ('inert', {})
        x_start[j] = float(rng.uniform(0.3, 0.7))
        x_star[j] = x_start[j]

    if variant in ('peaked', 'monotone'):
        n_decoys = N_DECOYS
        roles = rng.permutation(n_dims)
        dominant = int(roles[0])
        decoys = sorted(int(j) for j in roles[1:1 + n_decoys])
        inert = sorted(int(j) for j in roles[1 + n_decoys:])

        if variant == 'peaked':
            xs = float(rng.uniform(*(A_XSTAR_RANGE if aligned else XSTAR_RANGE)))
            if aligned:
                start = xs + A_DOM_START_OFFSET             # start ABOVE x* (reduce-aligned)
                components[dominant] = ('dominant_peaked_smooth',
                                        {'x_star': xs, 'amp': AMP_DOMINANT, 'w': A_WELL_W})
            else:
                w = float(rng.uniform(*WELL_W_RANGE))
                start = xs + DOM_START_OFFSET if xs <= 0.5 else xs - DOM_START_OFFSET
                components[dominant] = ('dominant_peaked',
                                        {'x_star': xs, 'amp': AMP_DOMINANT, 'w': w})
            x_star[dominant] = xs
            x_start[dominant] = float(np.clip(start, 0.0, 1.0))
        else:  # monotone
            if aligned:
                boundary = 0.0                              # optimum LOW -> reduce-aligned
                components[dominant] = ('dominant_monotone',
                                        {'boundary': boundary, 'amp': AMP_DOMINANT})
                x_star[dominant] = boundary
                x_start[dominant] = A_PROD_START
            else:
                boundary = float(rng.integers(0, 2))        # 0.0 or 1.0
                components[dominant] = ('dominant_monotone',
                                        {'boundary': boundary, 'amp': AMP_DOMINANT})
                x_star[dominant] = boundary
                x_start[dominant] = 0.10 if boundary == 1.0 else 0.90

        amp_each = DECOY_TOTAL / max(1, n_decoys)
        for j in decoys:
            if aligned:
                x0 = A_PROD_START
                components[j] = ('decoy_down', {'x0': x0, 'ramp': DECOY_RAMP, 'amp': amp_each})
                x_start[j] = x0
                x_star[j] = float(np.clip(x0 - DECOY_RAMP, 0.0, 1.0))
            else:
                x0 = float(rng.uniform(0.05, 0.15))
                components[j] = ('decoy', {'x0': x0, 'ramp': DECOY_RAMP, 'amp': amp_each})
                x_start[j] = x0
                x_star[j] = float(np.clip(x0 + DECOY_RAMP, 0.0, 1.0))
        for j in inert:
            _inert(j)
        active = sorted([dominant] + decoys)
        return LandscapeSpec(variant=variant, n_dims=n_dims, dim_names=names,
                             dominant_dim=dominant, decoy_dims=decoys,
                             inert_dims=inert, active_dims=active,
                             x_star=x_star, x_start=x_start, components=components,
                             amp_dominant=AMP_DOMINANT, seed=seed, aligned=aligned)

    # separable: several comparable independent active dims, no dominant lever.
    n_active = N_ACTIVE_SEPARABLE
    roles = rng.permutation(n_dims)
    active = sorted(int(j) for j in roles[:n_active])
    inert = sorted(int(j) for j in roles[n_active:])
    amp_each = (AMP_DOMINANT + DECOY_TOTAL) / n_active     # same total deficit as PEAKED start
    for j in active:
        if aligned:
            x0 = A_PROD_START
            components[j] = ('active_down', {'x0': x0, 'ramp': SEP_RAMP, 'amp': amp_each})
            x_start[j] = x0
            x_star[j] = float(np.clip(x0 - SEP_RAMP, 0.0, 1.0))
        else:
            x0 = SEP_START
            components[j] = ('active', {'x0': x0, 'ramp': SEP_RAMP, 'amp': amp_each})
            x_start[j] = x0
            x_star[j] = float(np.clip(x0 + SEP_RAMP, 0.0, 1.0))
    for j in inert:
        _inert(j)
    return LandscapeSpec(variant='separable', n_dims=n_dims, dim_names=names,
                         dominant_dim=None, decoy_dims=[], inert_dims=inert,
                         active_dims=active, x_star=x_star, x_start=x_start,
                         components=components, amp_dominant=None, seed=seed,
                         aligned=aligned)


def build_family(variant: str, n_instances: int, base_seed: int,
                 n_dims: int = N_DIMS, aligned: bool = False) -> List[LandscapeSpec]:
    return [make_landscape(variant, seed=base_seed + 101 * i, n_dims=n_dims,
                           aligned=aligned)
            for i in range(n_instances)]


# --------------------------------------------------------------------------- #
# Reference myopic greedy — characterises the trap, budget-matched to the loop   #
# --------------------------------------------------------------------------- #

def greedy_optimize(spec: LandscapeSpec, step: float = 0.1, budget: int = 40,
                    tol: float = 1e-4, discover: bool = False) -> dict:
    """Best-improvement coordinate descent with a fixed step + a minimum-improvement
    threshold; STOPS when no single step improves f by more than `tol`.

    discover=False (EXP-005): iterate only spec.active_dims (HANDED the active set).
      On PEAKED it fixates on decoys then halts on the flat dominant plateau ->
      trapped, even though the optimum is in-band.
    discover=True (EXP-006 fair baseline): iterate ALL dims, so it must DISCOVER the
      active set by probing. On the aligned variants it reaches the band on all three
      (including PEAKED, because best-improvement HALTS at the interior optimum)."""
    dims = range(spec.n_dims) if discover else spec.active_dims
    x = spec.x_start.copy()
    f = evaluate(spec, x)['f']
    for _ in range(budget):
        best = (0.0, None, 0.0)            # (improvement, dim, new_val)
        for j in dims:
            for d in (step, -step):
                xv = float(np.clip(x[j] + d, 0.0, 1.0))
                if xv == x[j]:
                    continue
                trial = x.copy(); trial[j] = xv
                imp = f - evaluate(spec, trial)['f']
                if imp > best[0]:
                    best = (imp, j, xv)
        if best[1] is None or best[0] <= tol:
            break
        x[best[1]] = best[2]
        f = evaluate(spec, x)['f']
    ev = evaluate(spec, x)
    ev['x'] = x
    return ev
