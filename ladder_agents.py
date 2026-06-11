"""Skill-ladder competence rungs (L0/L1 reuse existing players).

L2 MonopolyTargeterPlayer: greedy buyer with TRADES ENABLED. The engine's base
Player already assembles monopolies via update_lists_of_properties_to_trade +
do_a_two_way_trade (gated by is_willing_to_make_trades); this rung turns that on
and adds a monopoly-aware buy preference. L1->L2 = trades off->on, the core skill.

L3 GeneralistPlayer = L2 + cash/build discipline (containment: L3 >= L2).
"""
from agents import ParametricPlayer
from player_settings import ParametricPlayerSettings


class MonopolyTargeterPlayer(ParametricPlayer):
    def __init__(self, name, settings=None):
        if settings is None:
            settings = ParametricPlayerSettings(is_willing_to_make_trades=True)
        super().__init__(name, settings)

    def _should_buy(self, property_to_buy) -> bool:
        # Prefer extending a group we already hold, but stay within the cash
        # reserve and honour ignored groups (don't defeat L3's discipline).
        owns_in_group = any(c.group == property_to_buy.group for c in self.owned)
        if (owns_in_group
                and self.money - property_to_buy.cost_base >= self.settings.unspendable_cash
                and property_to_buy.group not in self.settings.ignore_property_groups):
            return True
        return super()._should_buy(property_to_buy)


class GeneralistPlayer(MonopolyTargeterPlayer):
    def __init__(self, name, settings=None):
        if settings is None:
            settings = ParametricPlayerSettings(
                is_willing_to_make_trades=True,
                unspendable_cash=150,
                build_cash_floor=250,
                jail_pay_threshold=150,
                aggressive_build=True,
            )
        super().__init__(name, settings)
