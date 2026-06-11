"""Survival-analysis hazard curves over the game-space exploration boards.

Three Isaksen-style hazard figures, each comparing five boards:
  (1a) game-end hazard       h(t) = P(game ends at turn t | still in progress at t)
  (1b) cash-level hazard     h(c) = P(player goes bankrupt next turn | cash bin c now)
  (1c) first-monopoly hazard h_m(t) = P(first colour-group monopoly forms at turn t
                                       | no monopoly through t-1)

Five board curves per figure (canonical default; GA-2p winner; GA-3p winner;
salary x2 mini; drop-Brown mini). Boards are different points in the design
space; the agent population is held fixed (two RuleBasedPlayer instances with
identical settings) so the only variable across curves is the board itself —
the Isaksen-style game-space probe.

Design intent (per plan_v2.md): the plot is a probe, not a deliverable. Flat
or bimodal hazards are findings, not failures. Curve confidence intervals are
recorded so a flat curve is honestly distinguishable from a noisy one.

Usage (from the repo root):
    set PYTHONPATH=. && python scripts/hazard_curves.py \
        --canonical-config default_config.yaml \
        --mini-config mini_config.yaml \
        --ga-2p logs/optimizer/ga_2p.jsonl \
        --ga-3p logs/optimizer/ga_3p.jsonl \
        --n-games 200 --base-seed 42 \
        --out-dir report/figures/hazards

If --ga-2p / --ga-3p paths are absent, the corresponding curves are
gracefully skipped and a warning is printed; the remaining three curves
still render. This makes the script runnable on a fresh checkout that has
not yet regenerated optimiser run logs.
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import List, Tuple

import numpy as np

# matplotlib is imported lazily so the script can do a --dry-run on a
# headless box without a display backend.

from config import GameConfig, _PLAYER_CLASSES
from monopoly.core.game import setup_game_from_config
from monopoly.core.game_utils import _check_end_conditions
from monopoly.core.player import Player
from optimizer import run_log
from optimizer.board_sources import build_five_boards
from optimizer.simulate import (_bounded_trade_loop,
                                 _track_interplayer_transfers)
from player_settings import RuleBasedPlayerSettings
from settings import SimulationSettings


# --------------------------------------------------------------------------- #
# Per-turn-instrumented game runner                                             #
# --------------------------------------------------------------------------- #

def _count_player_monopolies(player: Player, board) -> int:
    """Number of colour groups this player owns completely."""
    count = 0
    for cells in board.groups.values():
        if cells and all(c.owner is player for c in cells):
            count += 1
    return count


def run_game_with_snapshots(cfg, players_spec, seed: int, max_turns: int) -> dict:
    """Run one game; return outcome + per-turn snapshots needed for hazards.

    Snapshots are taken AFTER each turn (after every active player has moved),
    so an entry at index i corresponds to the state at the end of turn i+1.
    Each snapshot records, per player: cash, in-jail flag, bankrupt flag,
    monopoly-count. We also record the global "any monopoly held" boolean so
    the time-to-first-monopoly hazard can be derived without re-walking
    cells later.
    """
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

    snapshots: List[dict] = []
    turn_n = 0
    per_turn_trade_counts: dict = {}
    with ExitStack() as stack:
        total = stack.enter_context(_track_interplayer_transfers())
        stack.enter_context(_bounded_trade_loop(per_turn_trade_counts,
                                                 max_per_turn=5))
        for turn_n in range(1, max_turns + 1):
            per_turn_trade_counts.clear()
            if _check_end_conditions(players, elog, 0, turn_n):
                break
            for p in players:
                if p.is_bankrupt:
                    continue
                p.make_a_move(board, players, dice, elog)
            any_monopoly = any(_count_player_monopolies(p, board) > 0
                               for p in players)
            snapshots.append({
                'turn': turn_n,
                'players': [
                    {
                        'name': p.name,
                        'money': int(p.money),
                        'is_bankrupt': bool(p.is_bankrupt),
                        'monopolies': _count_player_monopolies(p, board),
                    }
                    for p in players
                ],
                'any_monopoly': bool(any_monopoly),
            })
        transfer_total = int(total[0]) if isinstance(total, list) else 0

    alive = [p for p in players if not p.is_bankrupt]
    winner = alive[0].name if len(alive) == 1 else None
    return {
        'seed':           seed,
        'rounds':         turn_n,
        'truncated':      winner is None and turn_n >= max_turns,
        'winner':         winner,
        'transfer_total': transfer_total,
        'snapshots':      snapshots,
    }


# Board sources (default / GA-2p / GA-3p / salary x2 / drop Brown) are
# resolved by `optimizer.board_sources.build_five_boards` so this script and
# `scripts/llm_character.py` always probe the same five points in the design
# space.


# --------------------------------------------------------------------------- #
# Hazard estimators                                                            #
# --------------------------------------------------------------------------- #

def _wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


def game_end_hazard(games: List[dict], max_t: int) -> dict:
    """h(t) = P(game ends at exactly turn t | still in progress at start of t).

    A game's "end turn" is `rounds` for both bankruptcy decisions and truncation
    (truncation is also a form of game-end and matters for the hazard shape;
    if you exclude truncated games the hazard is censored, not honest).
    """
    end_at = [g['rounds'] for g in games]
    n_games = len(games)
    h, lo, hi = [], [], []
    for t in range(1, max_t + 1):
        at_risk = sum(1 for e in end_at if e >= t)
        ended = sum(1 for e in end_at if e == t)
        p = ended / at_risk if at_risk > 0 else 0.0
        l, u = _wilson(p, at_risk)
        h.append(p); lo.append(l); hi.append(u)
    return {
        'turns': list(range(1, max_t + 1)),
        'hazard': h, 'ci_lo': lo, 'ci_hi': hi,
        'n_games': n_games,
    }


def cash_bankruptcy_hazard(games: List[dict], cash_edges: List[int]) -> dict:
    """h(c) = P(go bankrupt at t+1 | cash bin c at end of t, alive at end of t).

    Sweeps every (game, player, turn) triple where the player was alive at
    end-of-turn t AND we have an entry for end-of-turn t+1 to read its
    bankruptcy flag. Each such triple drops into one cash bin.
    """
    n_bins = len(cash_edges) - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    events = np.zeros(n_bins, dtype=np.int64)
    for g in games:
        snaps = g['snapshots']
        for i in range(len(snaps) - 1):
            cur = snaps[i]; nxt = snaps[i + 1]
            for ci, p_cur in enumerate(cur['players']):
                if p_cur['is_bankrupt']:
                    continue
                p_nxt = nxt['players'][ci]
                bin_idx = int(np.clip(np.digitize([p_cur['money']], cash_edges)[0] - 1,
                                      0, n_bins - 1))
                counts[bin_idx] += 1
                if p_nxt['is_bankrupt']:
                    events[bin_idx] += 1
    h = np.where(counts > 0, events / np.maximum(counts, 1), 0.0)
    lo = np.zeros(n_bins); hi = np.zeros(n_bins)
    for b in range(n_bins):
        l, u = _wilson(float(h[b]), int(counts[b]))
        lo[b] = l; hi[b] = u
    return {
        'cash_edges': list(cash_edges),
        'cash_centres': [(cash_edges[i] + cash_edges[i + 1]) / 2 for i in range(n_bins)],
        'counts': counts.tolist(),
        'events': events.tolist(),
        'hazard': h.tolist(), 'ci_lo': lo.tolist(), 'ci_hi': hi.tolist(),
    }


def first_monopoly_hazard(games: List[dict], max_t: int) -> dict:
    """h_m(t) = P(first colour-group monopoly forms at turn t | none through t-1).

    A game is "still no-monopoly" up to and including the last snapshot index
    j such that snapshots[0..j].any_monopoly is all False. The first-monopoly
    event lands on the smallest t with snapshots[t-1].any_monopoly == True.

    Games where no monopoly ever forms (truncated, or one player cleaned up
    properties before forming a full group) are right-censored: they
    contribute to "still at risk" at every turn but never to "events".
    """
    first_t: List[Optional[int]] = []
    last_at_risk: List[int] = []   # last turn the game contributes to denominator
    for g in games:
        snaps = g['snapshots']
        first = None
        for s in snaps:
            if s['any_monopoly']:
                first = s['turn']
                break
        first_t.append(first)
        last_at_risk.append(snaps[-1]['turn'] if snaps else 0)

    h, lo, hi = [], [], []
    for t in range(1, max_t + 1):
        at_risk = 0
        for first, last_t in zip(first_t, last_at_risk):
            # At risk at turn t iff (no monopoly yet by turn t-1) and (game still
            # generated a snapshot for turn t, i.e. last_at_risk >= t).
            if last_t < t:
                continue
            if first is None or first >= t:
                at_risk += 1
        events = sum(1 for f in first_t if f == t)
        p = events / at_risk if at_risk > 0 else 0.0
        l, u = _wilson(p, at_risk)
        h.append(p); lo.append(l); hi.append(u)
    return {
        'turns': list(range(1, max_t + 1)),
        'hazard': h, 'ci_lo': lo, 'ci_hi': hi,
        'n_first_events':   sum(1 for f in first_t if f is not None),
        'n_right_censored': sum(1 for f in first_t if f is None),
        'n_games':          len(games),
    }


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #

# Stable colour assignment so the same board has the same colour across all
# three figures. Keep order matched to the canonical board-source list.
_BOARD_COLOURS = {
    'default':    '#888888',
    'GA-2p':      '#1f77b4',
    'GA-3p':      '#2ca02c',
    'salary x2':  '#d62728',
    'drop Brown': '#9467bd',
}


def _plot_temporal_hazard(per_board, x_key: str, title: str, xlabel: str,
                          out_path: Path, smooth_window: int = 5):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, est in per_board.items():
        x = np.asarray(est[x_key], dtype=float)
        y = np.asarray(est['hazard'], dtype=float)
        if smooth_window > 1 and len(y) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            y_smooth = np.convolve(y, kernel, mode='same')
        else:
            y_smooth = y
        ax.plot(x, y_smooth,
                color=_BOARD_COLOURS.get(label, None),
                label=label, linewidth=1.6)
        # Light CI fill via the raw (unsmoothed) Wilson bounds.
        lo = np.asarray(est['ci_lo'], dtype=float)
        hi = np.asarray(est['ci_hi'], dtype=float)
        ax.fill_between(x, lo, hi, color=_BOARD_COLOURS.get(label, None),
                        alpha=0.10, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('hazard rate')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_cash_hazard(per_board, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, est in per_board.items():
        x = np.asarray(est['cash_centres'], dtype=float)
        y = np.asarray(est['hazard'], dtype=float)
        lo = np.asarray(est['ci_lo'], dtype=float)
        hi = np.asarray(est['ci_hi'], dtype=float)
        ax.plot(x, y, color=_BOARD_COLOURS.get(label, None),
                label=label, marker='o', markersize=3, linewidth=1.4)
        ax.fill_between(x, lo, hi, color=_BOARD_COLOURS.get(label, None),
                        alpha=0.12, linewidth=0)
    ax.set_xlabel('cash level at end of turn t (\\$)')
    ax.set_ylabel('P(bankrupt at t+1 | cash now)')
    ax.set_title('Cash-level bankruptcy hazard $h(c)$')
    ax.set_xscale('symlog', linthresh=50.0)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def run_one_board(label: str, cfg: GameConfig, n_games: int, base_seed: int,
                  max_turns: int) -> List[dict]:
    """Run n_games of (RuleBased vs RuleBased) on the given board."""
    specs = [
        ('A', RuleBasedPlayerSettings(), None),
        ('B', RuleBasedPlayerSettings(), None),
    ]
    out = []
    for i in range(n_games):
        out.append(run_game_with_snapshots(
            cfg, specs, seed=base_seed + i, max_turns=max_turns))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--mini-config',      default='mini_config.yaml')
    ap.add_argument('--ga-2p', default=None,
                    help='Path to ga_2p.jsonl optimiser log; lowest-score '
                         'entry is decoded against the canonical config.')
    ap.add_argument('--ga-3p', default=None)
    ap.add_argument('--n-games',   type=int, default=200)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--out-dir',   default='report/figures/hazards')
    ap.add_argument('--cash-edges', default='0,50,150,300,500,800,1200,2000,5000,20000',
                    help='Comma-separated edges defining cash bins for h(c). '
                         'Lowest edge must be 0; all values inclusive on left.')
    ap.add_argument('--smooth', type=int, default=5,
                    help='Moving-average window for temporal hazards (turns). '
                         'Set to 1 to disable.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cash_edges = [int(x) for x in args.cash_edges.split(',')]

    with run_log.track(experiment='hazards', script=__file__,
                       args=vars(args), out_dir=out_dir) as run:
        return _run_hazards(args, out_dir, cash_edges, run)


def _run_hazards(args, out_dir: Path, cash_edges, run) -> int:
    print('Resolving board sources...')
    sources = build_five_boards(
        canonical_config=args.canonical_config,
        mini_config=args.mini_config,
        ga_2p=args.ga_2p, ga_3p=args.ga_3p,
    )
    print(f'Boards under test: {[lbl for lbl, _ in sources]}')

    print(f'\nSimulating {args.n_games} games per board (max_turns={args.max_turns}, seed={args.base_seed})...')
    games_by_board = {}
    for lbl, cfg in sources:
        print(f'  {lbl}...', end='', flush=True)
        games_by_board[lbl] = run_one_board(
            lbl, cfg, args.n_games, args.base_seed, args.max_turns)
        n_trunc = sum(1 for g in games_by_board[lbl] if g['truncated'])
        mean_rounds = float(np.mean([g['rounds'] for g in games_by_board[lbl]]))
        print(f'  done. mean_rounds={mean_rounds:.1f}  truncated={n_trunc}/{args.n_games}')

    # ------------------------------------------------------------------ #
    # Estimators                                                          #
    # ------------------------------------------------------------------ #
    print('\nEstimating hazards...')
    h_end:   dict = {lbl: game_end_hazard(games_by_board[lbl], args.max_turns)
                     for lbl, _ in sources}
    h_cash:  dict = {lbl: cash_bankruptcy_hazard(games_by_board[lbl], cash_edges)
                     for lbl, _ in sources}
    h_first: dict = {lbl: first_monopoly_hazard(games_by_board[lbl], args.max_turns)
                     for lbl, _ in sources}

    # ------------------------------------------------------------------ #
    # Persist raw + estimates so figures are exactly regenerable.        #
    # ------------------------------------------------------------------ #
    raw_path = out_dir / 'hazards_raw.json'
    estimates_path = out_dir / 'hazards_estimates.json'
    with open(raw_path, 'w') as f:
        json.dump({
            'config': {
                'canonical_config': args.canonical_config,
                'mini_config':      args.mini_config,
                'ga_2p':            args.ga_2p,
                'ga_3p':            args.ga_3p,
                'n_games':          args.n_games,
                'max_turns':        args.max_turns,
                'base_seed':        args.base_seed,
                'cash_edges':       cash_edges,
            },
            'games_by_board': {
                lbl: [
                    {k: v for k, v in g.items() if k != 'snapshots'}
                    | {'rounds': g['rounds']}
                    for g in games_by_board[lbl]
                ]
                for lbl, _ in sources
            },
        }, f, indent=2)
    with open(estimates_path, 'w') as f:
        json.dump({'game_end': h_end, 'cash': h_cash, 'first_monopoly': h_first},
                  f, indent=2)
    print(f'  raw outcomes -> {raw_path}')
    print(f'  estimates    -> {estimates_path}')

    # ------------------------------------------------------------------ #
    # Plots                                                               #
    # ------------------------------------------------------------------ #
    print('\nRendering figures...')
    _plot_temporal_hazard(h_end, x_key='turns',
                           title='Game-end hazard $h(t)$',
                           xlabel='turn $t$',
                           out_path=out_dir / 'hazard_game_end.pdf',
                           smooth_window=args.smooth)
    _plot_cash_hazard(h_cash, out_path=out_dir / 'hazard_cash.pdf')
    _plot_temporal_hazard(h_first, x_key='turns',
                           title='Time-to-first-monopoly hazard $h_m(t)$',
                           xlabel='turn $t$',
                           out_path=out_dir / 'hazard_first_monopoly.pdf',
                           smooth_window=args.smooth)
    # Also write PNG copies for quick previewing.
    _plot_temporal_hazard(h_end, x_key='turns',
                           title='Game-end hazard $h(t)$',
                           xlabel='turn $t$',
                           out_path=out_dir / 'hazard_game_end.png',
                           smooth_window=args.smooth)
    _plot_cash_hazard(h_cash, out_path=out_dir / 'hazard_cash.png')
    _plot_temporal_hazard(h_first, x_key='turns',
                           title='Time-to-first-monopoly hazard $h_m(t)$',
                           xlabel='turn $t$',
                           out_path=out_dir / 'hazard_first_monopoly.png',
                           smooth_window=args.smooth)
    print(f'  hazard_game_end.pdf, hazard_cash.pdf, hazard_first_monopoly.pdf -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main() or 0)
