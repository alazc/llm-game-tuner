"""Thin simulator wrapper that returns per-game stats (not just logs).

The core simulator (monopoly/core/game.py:monopoly_game_from_config) runs a
single game and only writes to files. For optimisation we need a function that
runs many games and returns a list of per-game dicts we can aggregate.

We also instrument player-to-player cash transfers here via a context manager
that monkey-patches Player.pay_money for the duration of a game. This keeps
the core game code untouched. Bank payments (taxes, fines, salary) are NOT
counted, only player→player transfers.
"""
from contextlib import contextmanager
from typing import List, Tuple, Optional
import hashlib
import random

from monopoly.core.game import setup_game_from_config
from monopoly.core.game_utils import _check_end_conditions
from monopoly.core.player import Player
from config import _PLAYER_CLASSES
from settings import SimulationSettings


def _stable_player_seed(game_seed: int, name: str) -> int:
    h = hashlib.sha256(f"{game_seed}:{name}".encode()).digest()
    return int.from_bytes(h[:4], 'big')


# --------------------------------------------------------------------------- #
# Money-transfer instrumentation                                                #
# --------------------------------------------------------------------------- #

@contextmanager
def _track_interplayer_transfers():
    """Monkey-patch Player.pay_money to sum up player→player cash flow.

    Measured as `payee.money_after − payee.money_before` when payee is a Player.
    Captures bankruptcy cascades correctly (line 813 of core/player.py pushes
    the bankrupt player's entire remaining cash to the creditor).
    """
    total = [0]
    original = Player.pay_money

    def wrapped(self, amount, payee, board, log):
        if isinstance(payee, Player):
            pre_payee = payee.money
            original(self, amount, payee, board, log)
            delta = payee.money - pre_payee
            if delta > 0:
                total[0] += delta
        else:
            original(self, amount, payee, board, log)

    Player.pay_money = wrapped
    try:
        yield total
    finally:
        Player.pay_money = original


@contextmanager
def _bounded_trade_loop(per_turn_counts: dict, max_per_turn: int = 5):
    """Cap do_a_two_way_trade calls per (player, turn) to prevent oscillating trade loops.

    Player.make_a_move runs `while self.do_a_two_way_trade(...)` without a bound.
    Aggressive-trader strategies on certain boards can enter oscillations where
    both players keep swapping properties of the same colour group back and
    forth. The cap forces False after N calls, which breaks the while-loop and
    lets the turn progress.

    per_turn_counts is a dict the caller clears at the start of each turn.
    """
    original = Player.do_a_two_way_trade

    def wrapped(self, players, board, log):
        c = per_turn_counts.get(id(self), 0)
        if c >= max_per_turn:
            return False
        per_turn_counts[id(self)] = c + 1
        return original(self, players, board, log)

    Player.do_a_two_way_trade = wrapped
    try:
        yield
    finally:
        Player.do_a_two_way_trade = original


# --------------------------------------------------------------------------- #
# Single-game runner                                                            #
# --------------------------------------------------------------------------- #

def run_single_game(cfg, players_spec, seed: int, max_turns: int = None,
                    on_turn=None, record_trajectory: bool = False) -> dict:
    """Run one game with the specified players; return a stat dict.

    Args:
      cfg: GameConfig with board + mechanics (players list is ignored)
      players_spec: list of (name, settings_instance, class_name_or_None)
                    of length n_players. Class None → use base Player.
      seed: integer seed for dice / deck shuffling.
      max_turns: hard cap on turns; defaults to SimulationSettings.n_moves.
      record_trajectory: if True, attach a list of (turn_n, net_worth_dict)
                         snapshots as result['trajectory'].

    Returns:
      {
        'seed': int, 'winner': str or None, 'rounds': int,
        'bankrupt': {name: bool, ...}, 'truncated': bool,
        'transfer_total': int, 'net_worth': {name: int, ...},
        'player_names': [name, ...],
      }
    """
    if max_turns is None:
        max_turns = SimulationSettings.n_moves

    starting_money_cfg = cfg.settings.starting_money
    if isinstance(starting_money_cfg, dict):
        default_starting = next(iter(starting_money_cfg.values()), 1500)
    else:
        default_starting = starting_money_cfg or 1500

    board, dice, elog, blog = setup_game_from_config(0, seed, cfg)
    elog.disabled = True
    blog.disabled = True

    players = []
    for name, settings, cls_name in players_spec:
        cls = _PLAYER_CLASSES.get(cls_name, Player) if cls_name else Player
        p = cls(name, settings)
        p.money = default_starting
        players.append(p)

    # CRN: any player with its own RNG (e.g. RandomPlayer) must be seeded
    # deterministically from the game seed so common random numbers hold.
    for p in players:
        if hasattr(p, '_rng'):
            p._rng = random.Random(_stable_player_seed(seed, p.name))

    # Deterministic turn order (no shuffle): seed discipline relies on identical
    # ordering when the design vector is unchanged.

    turn_n = 0
    per_turn_trade_counts: dict = {}
    trajectory: list = []
    with _track_interplayer_transfers() as total, \
            _bounded_trade_loop(per_turn_trade_counts, max_per_turn=5):
        for turn_n in range(1, max_turns + 1):
            per_turn_trade_counts.clear()   # reset budget at the start of each turn
            if _check_end_conditions(players, elog, 0, turn_n):
                break
            for p in players:
                if p.is_bankrupt:
                    continue
                p.make_a_move(board, players, dice, elog)
            if record_trajectory:
                trajectory.append((turn_n, {p.name: p.net_worth() for p in players}))
            if on_turn is not None:
                on_turn(turn_n, players, total[0])

    alive = [p for p in players if not p.is_bankrupt]
    winner = alive[0].name if len(alive) == 1 else None
    result = {
        'seed':           seed,
        'winner':         winner,
        'rounds':         turn_n,
        'bankrupt':       {p.name: bool(p.is_bankrupt) for p in players},
        'truncated':      winner is None,
        'transfer_total': total[0],
        'net_worth':      {p.name: p.net_worth() for p in players},
        'player_names':   [p.name for p in players],
    }
    if record_trajectory:
        final_nw = result['net_worth']
        if not trajectory or trajectory[-1][0] != turn_n:
            trajectory.append((turn_n, final_nw))
        result['trajectory'] = trajectory
    return result


# --------------------------------------------------------------------------- #
# Matchup runner (handles seat-order balancing)                                 #
# --------------------------------------------------------------------------- #

def run_matchup(cfg, matchup_strategies: List[Tuple[str, object, Optional[str]]],
                n_games: int, base_seed: int,
                max_turns: int = None, balance_seats: bool = True,
                record_trajectory: bool = False) -> List[dict]:
    """Run n_games between the specified strategies; optionally rotate seats.

    matchup_strategies: [(name, settings_instance, class_name_or_None), ...].
    When balance_seats=True, the games are split across the cyclic seat
    permutations (2p: A-B + B-A; 3p: 3 rotations). This removes first-mover
    advantage from strategy-level fairness comparisons.

    Per-game seeds are `base_seed + i` for i in [0, n_games), identical across
    design candidates — byte-identical metrics when the design vec is unchanged.

    Args:
      record_trajectory: forwarded to run_single_game; attaches trajectory list
                         to each per-game result dict when True.
    """
    n_players = len(matchup_strategies)

    if balance_seats:
        if n_players == 2:
            perms = [(0, 1), (1, 0)]
        elif n_players == 3:
            perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        else:
            perms = [tuple(range(n_players))]
    else:
        perms = [tuple(range(n_players))]

    games_per_perm = n_games // len(perms)
    extras = n_games - games_per_perm * len(perms)

    results = []
    game_idx = 0
    for pi, perm in enumerate(perms):
        k = games_per_perm + (1 if pi < extras else 0)
        for _ in range(k):
            seed = base_seed + game_idx
            ordered = [matchup_strategies[j] for j in perm]
            r = run_single_game(cfg, ordered, seed, max_turns,
                                record_trajectory=record_trajectory)
            r['seat_perm'] = list(perm)
            # Also tag the strategies that actually played, for aggregation.
            r['strategy_names'] = [matchup_strategies[j][0] for j in perm]
            results.append(r)
            game_idx += 1

    return results
