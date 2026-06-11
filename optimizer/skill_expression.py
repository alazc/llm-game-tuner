"""Skill-expression objective: competence ladder + wealth-share metric.

share = nw_strong / (nw_strong + nw_rand) in [0,1]; 0.5 = no skill edge.
Game-end rules (engine records a winner ONLY on bankruptcy):
  - bankruptcy -> survivor share set EXPLICITLY (1 / 0); never via the ratio,
    which is unsafe when a net worth is near/below zero;
  - solvent truncation (no winner) -> graded net-worth ratio (floored at 0);
  - exact tie / both-zero -> 0.5.
See docs/superpowers/specs/2026-06-02-skill-expression-objective-design.md.

TWO NAMED METRICS WITH FIXED, SEPARATE JOBS
-------------------------------------------
METRIC 1 -- SKILL SHARE  (`skill_share(cfg)`): THE OPTIMIZED SCALAR.
   Head-to-head L3-vs-L0 wealth share, nw_L3/(nw_L3+nw_L0), 2-player, CRN,
   seat-balanced, in [0,1] with 0.5 = no edge. This is the ONLY thing the
   optimizer scores against. g* = 0.60, the direction (REDUCE/INCREASE), and
   `skill_score = max(0, |share - g*| - band)` all live on THIS scale. Its
   human analog (skilled vs novice, head-to-head) is the playtest bridge.

METRIC 2 -- LADDER MONOTONICITY (`round_robin(cfg)` -> `ladder_monotonicity`):
   THE DIAGNOSTIC. Full round-robin standings across L0->L1->L2->L3; its job is
   to confirm the rungs form a valid skill ruler (standings climb monotonically
   -- rank correlation + inversion count). Reported, NEVER optimized. This is
   the spec-section-3a validity gate.

STANDARDIZATION RULE (prevents re-conflation): every reference to "skill
expression", "SE", "the metric", "the score", "g*", or "direction" resolves to
exactly ONE of these on its stated scale. Optimizer / score / g* / direction
-> always METRIC 1 (skill share). Validity gate / monotonicity / "is the ladder
a real ruler" -> always METRIC 2. The old round-robin standings GAP
(standing[L3] - standing[L0], ~0.47) is RETIRED -- a tabulation artifact
contaminated by the middle rungs, on neither metric's scale; nothing references it.
"""
from __future__ import annotations

import itertools
import statistics
from typing import List

from optimizer.simulate import run_matchup   # patched in tests
from player_settings import RandomPlayerSettings, ParametricPlayerSettings

# METRIC 1 optimization target + score band, on the 0.5-centered share scale.
G_STAR = 0.60
BAND = 0.03


def game_share(record: dict, strong: str, rand: str) -> float:
    """Per-game wealth share of `strong` vs `rand` in [0, 1]."""
    winner = record['winner']
    if winner == strong:
        return 1.0
    if winner == rand:
        return 0.0
    s = max(0, record['net_worth'].get(strong, 0))
    r = max(0, record['net_worth'].get(rand, 0))
    if s + r == 0:
        return 0.5
    return s / (s + r)


def share_from_networth(nw: dict, strong: str, rand: str) -> float:
    """Per-round share from a net-worth snapshot, same clamps as game_share
    minus the winner rule (a live round has no winner). Both-zero -> 0.5."""
    s = max(0, nw.get(strong, 0))
    r = max(0, nw.get(rand, 0))
    if s + r == 0:
        return 0.5
    return s / (s + r)


def win_indicator(record: dict, name: str) -> float:
    """1.0 win / 0.0 loss / 0.5 truncation-or-draw."""
    winner = record['winner']
    if winner is None:
        return 0.5
    return 1.0 if winner == name else 0.0


def skill_gap(records: List[dict], strong: str, rand: str) -> float:
    """Mean per-game wealth share over records (per-game normalize, then average)."""
    if not records:
        return 0.5
    return sum(game_share(r, strong, rand) for r in records) / len(records)


def share_stats(records: List[dict], strong: str, rand: str) -> dict:
    """Mean per-game share AND its standard error across the games.

    SE = sample-stdev / sqrt(n): the error bar on the optimized scalar, on the
    same 0.5-centered share scale. With balance_seats the games already average
    over both seat orders, so each game is one CRN-paired observation.
    """
    shares = [game_share(r, strong, rand) for r in records]
    if not shares:
        return {'share': 0.5, 'se': float('nan'), 'n': 0, 'shares': []}
    mean = sum(shares) / len(shares)
    se = statistics.stdev(shares) / (len(shares) ** 0.5) if len(shares) > 1 else float('nan')
    return {'share': mean, 'se': se, 'n': len(shares), 'shares': shares}


def skill_share(cfg, n_seeds: int = 30, base_seed: int = 0,
                max_turns: int = 200,
                strong: str = 'L3_generalist',
                anchor: str = 'L0_random') -> dict:
    """METRIC 1 (the OPTIMIZED SCALAR): L3-vs-L0 head-to-head wealth share on `cfg`.

    Runs a 2-player matchup of the top competence rung against the random
    anchor, N seeds, common random numbers (every game = base_seed+i), seats
    balanced. Records per-game game_share = nw_L3/(nw_L3+nw_L0) under the
    bankruptcy/tie/solvent rules, then averages -> the share on the 0.5-centered
    scale g* lives on. Returns {share, se, n, shares, records}.
    """
    ts = next(t for t in LADDER if t[0] == strong)
    ta = next(t for t in LADDER if t[0] == anchor)
    matchup = [(strong, ts[1], ts[2]), (anchor, ta[1], ta[2])]
    records = run_matchup(cfg, matchup, n_games=n_seeds, base_seed=base_seed,
                          max_turns=max_turns, balance_seats=True)
    out = share_stats(records, strong, anchor)
    out['records'] = records
    return out


def _grid_frac(k, n_checkpoints):
    """Fraction in [0,1] of checkpoint k out of n (k/(n-1); 1.0 if n==1)."""
    return k / (n_checkpoints - 1) if n_checkpoints > 1 else 1.0


def _resample_to_grid(per_round, n_checkpoints):
    """Map an irregular [(round, share), ...] series onto a fixed [0,1] grid of
    n_checkpoints points by carrying the last observed share forward (step
    interpolation). Assumes `per_round` is sorted ascending by round (the
    simulator records in turn order); the early `break` relies on it.
    """
    if not per_round:
        return [0.5] * n_checkpoints
    last_round = per_round[-1][0]
    grid = []
    for k in range(n_checkpoints):
        frac = _grid_frac(k, n_checkpoints)
        target = frac * last_round
        val = per_round[0][1]
        for rnd, sh in per_round:
            if rnd <= target:
                val = sh
            else:
                break
        grid.append(val)
    return grid


def money_share_trajectory(cfg, n_seeds: int = 30, base_seed: int = 0,
                           max_turns: int = 200,
                           strong: str = 'L3_generalist',
                           anchor: str = 'L0_random',
                           n_checkpoints: int = 6) -> dict:
    """The within-game money-share trajectory (the skill CURVE object).

    Runs the SAME L3-vs-L0 matchup as skill_share (identical seeds/CRN/seat
    balancing) with per-round net-worth recording, converts each game to a
    per-round share series, and aggregates onto an n_checkpoints grid. The final
    checkpoint equals the mean skill_share endpoint, so the curve adds shape
    without adding information. Returns {checkpoints:[{frac,share}], per_game,
    n, endpoint}.
    """
    ts = next(t for t in LADDER if t[0] == strong)
    ta = next(t for t in LADDER if t[0] == anchor)
    matchup = [(strong, ts[1], ts[2]), (anchor, ta[1], ta[2])]
    records = run_matchup(cfg, matchup, n_games=n_seeds, base_seed=base_seed,
                          max_turns=max_turns, balance_seats=True,
                          record_trajectory=True)
    per_game_grids = []
    for rec in records:
        series = [(rnd, share_from_networth(nw, strong, anchor))
                  for rnd, nw in rec['trajectory']]
        grid = _resample_to_grid(series, n_checkpoints)
        grid[-1] = game_share(rec, strong, anchor)  # pin to the bankruptcy-aware endpoint -> keeps the curve's endpoint == skill_share
        per_game_grids.append(grid)
    n = len(per_game_grids)
    checkpoints = []
    for k in range(n_checkpoints):
        frac = _grid_frac(k, n_checkpoints)
        mean = sum(g[k] for g in per_game_grids) / n if n else 0.5
        checkpoints.append({'frac': frac, 'share': mean})
    return {'checkpoints': checkpoints, 'per_game': per_game_grids,
            'n': n, 'endpoint': checkpoints[-1]['share'] if checkpoints else 0.5}


def fmt_checkpoint(c: dict) -> str:
    """One checkpoint as 'rNN%:0.SS' — shared by the curve render and the EVAL
    table so both surface byte-identical share numbers (factoring guard)."""
    return f"r{int(c['frac']*100):02d}%:{c['share']:.2f}"


def render_goal_anchor(share: float, g_star: float = G_STAR, direction: str = '') -> str:
    """The shared in-feed goal anchor: current endpoint vs target + the direction
    word. Authored ONCE and injected byte-identically into haz/met/full so the
    target/REDUCE framing is not an asymmetry between channels. Not shown to mute
    (floor) or blind (goal-closed)."""
    d = (direction or '').lower()
    if d == 'reduce':
        tail = 'ABOVE target -> REDUCE (bring the share DOWN toward the target)'
    elif d == 'increase':
        tail = 'BELOW target -> INCREASE (bring the share UP toward the target)'
    else:
        tail = 'ON target'
    return (f'skilled-vs-random wealth share ends {share:.2f}; '
            f'target {g_star:.2f} -> {tail}')


def render_skill_curve(curve: dict) -> str:
    """Render the money-share trajectory as a CURVE (representation axis): a
    round-fraction checkpoint sequence plus a one-line shape descriptor naming
    where the skill edge accumulates. Carries no quantity the table form lacks."""
    cps = curve.get('checkpoints', [])
    if not cps:
        return '  (no trajectory)'
    seq = ' -> '.join(fmt_checkpoint(c) for c in cps)
    start, mid, end = cps[0]['share'], cps[len(cps)//2]['share'], cps[-1]['share']
    total = end - start
    if abs(total) < 0.02:            # < 2 share-points of total swing -> flat
        shape = 'stays near flat (little edge accumulates)'
    else:
        first_half = mid - start
        second_half = end - mid
        if abs(second_half) > abs(first_half) * 1.5:
            shape = 'the gap opens mostly in the later rounds'
        elif abs(first_half) > abs(second_half) * 1.5:
            shape = 'the gap opens mostly in the early rounds'
        else:
            shape = 'the gap opens steadily across the game'
    return f'  {seq}\n  shape: {shape}'


def skill_score(share: float, g_star: float = G_STAR, band: float = BAND) -> float:
    """METRIC 1 score the optimizer MINIMIZES: banded distance of the skill
    share from g*. 0 inside the dead-zone [g*-band, g*+band]; rises outside."""
    return max(0.0, abs(share - g_star) - band)


def optimization_direction(share: float, g_star: float = G_STAR) -> str:
    """Direction the optimizer should push the board, given the skill share
    (METRIC 1) vs the target g*. share>g* -> 'reduce' skill toward target;
    share<g* -> 'increase' it; within 1e-9 -> 'on_target'."""
    if share > g_star + 1e-9:
        return 'reduce'
    if share < g_star - 1e-9:
        return 'increase'
    return 'on_target'


def win_rate_gap(records: List[dict], strong: str, rand: str) -> float:
    """Reported human-bridge companion: strong win rate minus rand win rate.

    Truncations/draws count as half a win for BOTH players (spec game-end rule).
    Reported, not optimized.
    """
    if not records:
        return 0.0
    sw = sum(win_indicator(r, strong) for r in records) / len(records)
    rw = sum(win_indicator(r, rand) for r in records) / len(records)
    return sw - rw


def _greedy_settings() -> ParametricPlayerSettings:
    """L1: buy everything, NO trades (can only own what it lands on)."""
    return ParametricPlayerSettings(
        unspendable_cash=0, build_cash_floor=0,
        is_willing_to_make_trades=False,
        aggressive_build=True, buy_utilities=True, buy_railroads=True,
        ignore_property_groups=frozenset(),
    )


# (rung_name, settings_or_None, class_name). None -> class default settings.
LADDER = [
    ('L0_random',     RandomPlayerSettings(), 'RandomPlayer'),
    ('L1_greedy',     _greedy_settings(),     'ParametricPlayer'),
    ('L2_targeter',   None,                   'MonopolyTargeterPlayer'),
    ('L3_generalist', None,                   'GeneralistPlayer'),
]
RAND_NAME = 'rand'
HEADLINE_RUNG = 'L3_generalist'   # top rung; METRIC 1 (optimized) = skill_share() (L3 vs L0). round_robin -> ladder_monotonicity is METRIC 2 (diagnostic).
RUNG_ORDER = ['L0_random', 'L1_greedy', 'L2_targeter', 'L3_generalist']


def ladder_shares(cfg, n_seeds: int = 30, base_seed: int = 0, max_turns: int = 200) -> dict:
    """Wealth-share of each competent rung vs the random anchor on `cfg`.

    Every rung uses the SAME base_seed (common random numbers across rungs).
    Returns {rung_name: skill_gap}. Headline scalar = result[HEADLINE_RUNG].
    """
    _, rand_settings, rand_cls = LADDER[0]
    shares = {}
    for rung_name, settings, cls in LADDER[1:]:
        matchup = [(rung_name, settings, cls), (RAND_NAME, rand_settings, rand_cls)]
        records = run_matchup(cfg, matchup, n_games=n_seeds, base_seed=base_seed,
                              max_turns=max_turns, balance_seats=True)
        shares[rung_name] = skill_gap(records, rung_name, RAND_NAME)
    return shares

_COMPETENT_ORDER = ['L1_greedy', 'L2_targeter', 'L3_generalist']


def _spearman(vals: list) -> float:
    """Rank correlation of `vals` against their position index (the ideal
    ascending ladder). +1 = perfectly monotone-increasing, -1 = reversed.
    Average-rank tie handling; NaN if <2 points or all-equal (no variance)."""
    n = len(vals)
    if n < 2:
        return float('nan')
    order_idx = list(range(n))                      # ideal ranks: 0,1,...,n-1
    sorted_vals = sorted(vals)
    obs_rank = []
    for v in vals:                                  # average rank for ties
        idxs = [i for i, sv in enumerate(sorted_vals) if sv == v]
        obs_rank.append(sum(idxs) / len(idxs))
    mx = sum(order_idx) / n
    my = sum(obs_rank) / n
    cov = sum((order_idx[i] - mx) * (obs_rank[i] - my) for i in range(n))
    vx = sum((x - mx) ** 2 for x in order_idx)
    vy = sum((y - my) ** 2 for y in obs_rank)
    if vx == 0 or vy == 0:
        return float('nan')
    return cov / (vx * vy) ** 0.5


def ladder_monotonicity(shares_by_rung: dict, order: list = None) -> dict:
    """METRIC 2 readout: does wealth share climb monotonically along `order`?
    Returns values, inversion count, monotone flag, and Spearman rank
    correlation vs the ideal ascending ladder. Default order = the 3 competent
    rungs; pass RUNG_ORDER for the full round-robin standings."""
    order = order or _COMPETENT_ORDER
    vals = [shares_by_rung[k] for k in order if k in shares_by_rung]
    inversions = sum(1 for i in range(len(vals) - 1) if vals[i + 1] < vals[i])
    return {'values': vals, 'inversions': inversions, 'monotone': inversions == 0,
            'rank_corr': _spearman(vals)}


def validate_ladder(standings: dict) -> dict:
    """(3a) DEFAULT board: are the round-robin standings monotone L0<=...<=L3?
    Must pass before results are interpretable."""
    return ladder_monotonicity(standings, order=RUNG_ORDER)


def result_monotonicity(standings: dict) -> dict:
    """(3b) OPTIMIZED board: genuine graded skill or a gamed artifact? Same
    computation as validate_ladder, different meaning."""
    return ladder_monotonicity(standings, order=RUNG_ORDER)


def round_robin(cfg, n_seeds: int = 30, base_seed: int = 0, max_turns: int = 200) -> dict:
    """GRADIENT/MONOTONICITY DIAGNOSTIC (not the optimized scalar).

    Play every rung pair (CRN: same base_seed for every pair) and return
    standings = {rung: mean per-game wealth share across ALL its games}. Used
    only to check competence climbs L0->L1->L2->L3; the optimized scalar is
    skill_share() (METRIC 1)."""
    per_rung = {name: [] for name, _, _ in LADDER}
    for (na, sa, ca), (nb, sb, cb) in itertools.combinations(LADDER, 2):
        records = run_matchup(cfg, [(na, sa, ca), (nb, sb, cb)], n_games=n_seeds,
                              base_seed=base_seed, max_turns=max_turns, balance_seats=True)
        for rec in records:
            per_rung[na].append(game_share(rec, na, nb))
            per_rung[nb].append(game_share(rec, nb, na))
    return {name: (sum(v) / len(v) if v else 0.5) for name, v in per_rung.items()}


# NOTE: the old `skill_expression(standings) = standing[L3] - standing[L0]`
# round-robin GAP is RETIRED (see module docstring standardization rule). It was
# a tabulation artifact contaminated by the middle rungs, on neither metric's
# scale. METRIC 1 = skill_share(); METRIC 2 = standings_monotonicity().


def standings_monotonicity(standings: dict) -> dict:
    """METRIC 2: monotonicity of the 4 round-robin standings (L0<=L1<=L2<=L3),
    with inversion count + Spearman rank correlation. The validity gate."""
    return ladder_monotonicity(standings, order=RUNG_ORDER)
