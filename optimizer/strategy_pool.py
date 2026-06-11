"""Diverse strategy pool (10 named archetypes + 20 sampled) for board optimisation.

Each strategy is a ParametricPlayerSettings instance. The named archetypes are
hand-designed with explicit rationales for the report; the sampled strategies
draw from the 17-dim parameter space deterministically (given a seed) to fill
behavioural coverage gaps.

Use load_strategy_pool() to get the full 30-strategy list, and
load_eval_matchups(n_players) for the fixed evaluation matchups (10 pairs
for 2p, 10 triples for 3p).
"""
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from player_settings import ParametricPlayerSettings

# Colour groups in cost order (low → high) — matches the default board
COLOUR_GROUPS = ['Brown', 'Lightblue', 'Pink', 'Orange',
                 'Red', 'Yellow', 'Green', 'Indigo']


# --------------------------------------------------------------------------- #
# Named archetypes (10)                                                         #
# --------------------------------------------------------------------------- #

def _named_archetypes() -> List[Tuple[str, ParametricPlayerSettings]]:
    """Return (name, settings) pairs for the hand-designed archetypes."""
    return [
        ('AggressiveBuilder', ParametricPlayerSettings(
            unspendable_cash=0, build_cash_floor=0,
            is_willing_to_make_trades=False,
            aggressive_build=True, buy_utilities=True, buy_railroads=True,
            ignore_property_groups=frozenset(),
            jail_pay_threshold=0,
        )),
        ('CashHoarder', ParametricPlayerSettings(
            unspendable_cash=1000, build_cash_floor=1200,
            is_willing_to_make_trades=False,
            aggressive_build=False, buy_utilities=True, buy_railroads=True,
            ignore_property_groups=frozenset(),
            jail_pay_threshold=1500,
        )),
        ('Trader', ParametricPlayerSettings(
            unspendable_cash=300, build_cash_floor=400,
            is_willing_to_make_trades=True,
            trade_max_diff_absolute=400,
            trade_max_diff_relative=3.5,
            aggressive_build=True, buy_utilities=True, buy_railroads=True,
            ignore_property_groups=frozenset(),
            jail_pay_threshold=500,
        )),
        ('RailroadKing', ParametricPlayerSettings(
            unspendable_cash=100, build_cash_floor=100,
            is_willing_to_make_trades=False,
            aggressive_build=True,
            buy_utilities=False, buy_railroads=True,
            ignore_property_groups=frozenset(COLOUR_GROUPS),   # ignore every colour group
            jail_pay_threshold=300,
        )),
        ('LowCostOnly', ParametricPlayerSettings(
            unspendable_cash=200, build_cash_floor=250,
            is_willing_to_make_trades=True,
            aggressive_build=True, buy_utilities=True, buy_railroads=True,
            # ignore the 5 most expensive groups
            ignore_property_groups=frozenset({'Orange', 'Red', 'Yellow', 'Green', 'Indigo'}),
            jail_pay_threshold=200,
        )),
        ('HighCostOnly', ParametricPlayerSettings(
            unspendable_cash=400, build_cash_floor=500,
            is_willing_to_make_trades=True,
            aggressive_build=True, buy_utilities=False, buy_railroads=True,
            # ignore the 3 cheapest groups
            ignore_property_groups=frozenset({'Brown', 'Lightblue', 'Pink'}),
            jail_pay_threshold=400,
        )),
        ('Balanced', ParametricPlayerSettings(
            unspendable_cash=300, build_cash_floor=350,
            is_willing_to_make_trades=True,
            trade_max_diff_absolute=250,
            trade_max_diff_relative=2.5,
            aggressive_build=True, buy_utilities=True, buy_railroads=True,
            ignore_property_groups=frozenset(),
            jail_pay_threshold=250,
        )),
        ('Passive', ParametricPlayerSettings(
            unspendable_cash=500, build_cash_floor=700,
            is_willing_to_make_trades=False,
            aggressive_build=False,
            buy_utilities=False, buy_railroads=False,
            ignore_property_groups=frozenset(),
            jail_pay_threshold=1500,   # never exit early
        )),
        ('Bully', ParametricPlayerSettings(
            unspendable_cash=50, build_cash_floor=0,
            is_willing_to_make_trades=True,
            trade_max_diff_absolute=500,
            trade_max_diff_relative=5.0,
            aggressive_build=True, buy_utilities=True, buy_railroads=True,
            ignore_property_groups=frozenset(),
            jail_pay_threshold=100,
        )),
        ('RiskAverse', ParametricPlayerSettings(
            unspendable_cash=800, build_cash_floor=1000,
            is_willing_to_make_trades=False,
            aggressive_build=False,
            buy_utilities=False, buy_railroads=True,
            ignore_property_groups=frozenset({'Indigo', 'Green'}),   # avoid expensive dev
            jail_pay_threshold=1200,
        )),
    ]


# --------------------------------------------------------------------------- #
# Sampled strategies (20)                                                       #
# --------------------------------------------------------------------------- #

def _sample_strategies(n: int, rng: random.Random) -> List[Tuple[str, ParametricPlayerSettings]]:
    """Draw n strategies uniformly from the parameter space."""
    out = []
    for i in range(n):
        cash   = rng.randint(0, 1500)
        floor  = rng.randint(0, 1500)
        t_abs  = rng.randint(0, 500)
        t_rel  = round(rng.uniform(1.0, 5.0), 2)
        jail   = rng.randint(0, 1500)
        trades = rng.random() < 0.5
        aggr   = rng.random() < 0.5
        utils  = rng.random() < 0.5
        rails  = rng.random() < 0.5
        # Each colour group independently ignored with prob 0.25
        ignored = frozenset(g for g in COLOUR_GROUPS if rng.random() < 0.25)
        s = ParametricPlayerSettings(
            unspendable_cash=cash, build_cash_floor=floor,
            is_willing_to_make_trades=trades,
            trade_max_diff_absolute=t_abs,
            trade_max_diff_relative=t_rel,
            aggressive_build=aggr,
            buy_utilities=utils, buy_railroads=rails,
            ignore_property_groups=ignored,
            jail_pay_threshold=jail,
        )
        out.append((f'Sampled_{i:02d}', s))
    return out


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def build_pool(seed: int = 0, n_sampled: int = 20) -> List[Tuple[str, ParametricPlayerSettings]]:
    """Return the full 30-strategy pool: 10 named + n_sampled sampled."""
    named = _named_archetypes()
    rng = random.Random(seed)
    sampled = _sample_strategies(n_sampled, rng)
    return named + sampled


def save_pool(path: str, pool: List[Tuple[str, ParametricPlayerSettings]]):
    """Persist the pool to JSON (deterministic, readable)."""
    payload = []
    for name, s in pool:
        d = asdict(s)
        # frozenset isn't JSON-serialisable → convert to sorted list
        d['ignore_property_groups'] = sorted(d['ignore_property_groups'])
        payload.append({'name': name, 'settings': d})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_pool(path: str) -> List[Tuple[str, ParametricPlayerSettings]]:
    with open(path) as f:
        payload = json.load(f)
    out = []
    for entry in payload:
        d = dict(entry['settings'])
        d['ignore_property_groups'] = frozenset(d['ignore_property_groups'])
        out.append((entry['name'], ParametricPlayerSettings(**d)))
    return out


def load_strategy_pool(path: str = None, seed: int = 0) -> List[Tuple[str, ParametricPlayerSettings]]:
    """Load from `path` if it exists, else build fresh and return.

    Default path: monopoly/optimizer/strategy_pool.json.
    """
    if path is None:
        path = Path(__file__).parent / 'strategy_pool.json'
    p = Path(path)
    if p.exists():
        return load_pool(str(p))
    pool = build_pool(seed=seed)
    save_pool(str(p), pool)
    return pool


# --------------------------------------------------------------------------- #
# Evaluation matchups                                                           #
# --------------------------------------------------------------------------- #

def load_eval_matchups(n_players: int, pool_size: int = 30,
                       n_matchups: int = 10, seed: int = 1234
                       ) -> List[Tuple[int, ...]]:
    """Return a deterministic list of index-tuples into the strategy pool.

    Diversity constraints:
      - Every named archetype (indices 0..9) appears in at least one matchup.
      - At least 3 matchups include a sampled strategy (indices 10..).
      - No duplicate matchups.
    """
    rng = random.Random(seed)
    named_range = 10
    n_named_matchups = max(5, n_matchups - 3)

    matchups: List[Tuple[int, ...]] = []

    # Step 1: ensure every named archetype appears at least once by drawing
    # pairs / triples that all come from the named subset.
    named_used = set()
    attempts = 0
    while len(named_used) < named_range and len(matchups) < n_named_matchups and attempts < 200:
        attempts += 1
        idxs = tuple(sorted(rng.sample(range(named_range), n_players)))
        if idxs in matchups:
            continue
        # Prefer matchups that introduce an unseen archetype.
        if not (set(idxs) - named_used):
            continue
        matchups.append(idxs)
        named_used.update(idxs)

    # Step 2: fill remaining slots with random matchups that include at least
    # one sampled strategy.
    attempts = 0
    while len(matchups) < n_matchups and attempts < 500:
        attempts += 1
        idxs = tuple(sorted(rng.sample(range(pool_size), n_players)))
        if idxs in matchups:
            continue
        if not any(i >= named_range for i in idxs):
            continue
        matchups.append(idxs)

    # Step 3: top up with any non-duplicate matchups if we still don't have enough.
    while len(matchups) < n_matchups:
        idxs = tuple(sorted(rng.sample(range(pool_size), n_players)))
        if idxs not in matchups:
            matchups.append(idxs)

    return matchups
