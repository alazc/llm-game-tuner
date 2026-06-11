"""Player behavior settings.

All player strategy settings live here, separate from game/simulation settings
(GameMechanics, SimulationSettings) which live in settings.py.
"""
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class StandardPlayerSettings:
    unspendable_cash: int = 200  # Amount of money the player wants to keep unspent (money safety pillow)
    ignore_property_groups: FrozenSet[str] = frozenset()  # Group of properties do not buy, i.e.{"RED", "GREEN"}

    is_willing_to_make_trades: bool = True
    # agree to trades if the value difference is within these limits:
    trade_max_diff_absolute: int = 200  # More expensive - less expensive
    trade_max_diff_relative: float = 2.0  # More expensive / less expensive


@dataclass(frozen=True)
class HeroPlayerSettings(StandardPlayerSettings):
    """ here you can change the settings of the hero (the Experimental Player) """
    # ignore_property_groups: FrozenSet[str] = frozenset({"GREEN"})


@dataclass(frozen=True)
class RuleBasedPlayerSettings(StandardPlayerSettings):
    """Buy every affordable property, build houses immediately, keep no cash reserve."""
    unspendable_cash: int = 0
    is_willing_to_make_trades: bool = False


@dataclass(frozen=True)
class RandomPlayerSettings(StandardPlayerSettings):
    """Settings for RandomPlayer; actual buy decisions are randomised in the Player subclass."""
    unspendable_cash: int = 0
    is_willing_to_make_trades: bool = False


@dataclass(frozen=True)
class ParametricPlayerSettings(StandardPlayerSettings):
    """Full parametric ruleset for the strategy-pool optimiser.

    Used by agents.ParametricPlayer. Defines ~17 degrees of freedom the outer
    optimiser samples over to build a diverse strategy pool.
    """
    # --- continuous knobs (5) ---
    # unspendable_cash inherited from parent: [0, 1500]
    build_cash_floor: int = 200           # cash kept before building houses
    # trade_max_diff_absolute inherited: [0, 500]
    # trade_max_diff_relative inherited: [1.0, 5.0]
    jail_pay_threshold: int = 150         # pay jail fine early if cash above this

    # --- boolean knobs (4) ---
    # is_willing_to_make_trades inherited
    aggressive_build: bool = True         # True: build as many houses as affordable per turn; False: at most 1
    buy_utilities: bool = True            # skip utilities if False
    buy_railroads: bool = True            # skip railroads if False

    # --- bit-mask over 8 colour groups (ignore_property_groups inherited) ---
