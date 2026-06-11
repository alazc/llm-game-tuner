"""Phase-A validation: pre-declared design-knob sweeps on the 4x4 mini board.

For each of three pre-declared design knobs, this script compares the
default mini board to a modified variant by running a batch of games
with both rule-based and LLM agents, and prints the agent-predicted
direction of each design effect.

The three knobs (matching report.tex §7.1):
  1. salary doubled            -> expect: shorter games, fewer draws
  2. Brown colour group removed -> expect: fewer monopolies, more draws
  3. Orange rent doubled        -> expect: higher win-rate spread, more bankruptcies

Output: a per-knob table comparing the two variants on the same set of
fixed seeds. Rule-based and LLM rows are reported separately so we can
see whether they agree on the direction of each effect.

The `--player {guided,neutral}` flag selects which LLM prompt to use:
  - guided  : 4-shot directive prompt (agents.LLMPlayer; the original
              Phase A run; in the report).
  - neutral : no-shot, neutral system prompt loaded from
              prompts/character_llm_prompt.txt (round-1
              PHASE-A-ROBUSTNESS re-run, used to test whether the
              cross-source agreement signal survives without GUIDED-
              PLAYER's strategic scaffolding and few-shot property-name
              overlap with the canonical default board).

Usage (from monopoly/):
    set PYTHONPATH=. && python scripts/phase_a_validation.py
    set PYTHONPATH=. && python scripts/phase_a_validation.py --no-llm
    set PYTHONPATH=. && python scripts/phase_a_validation.py --player neutral
    set PYTHONPATH=. && python scripts/phase_a_validation.py --n-games 20 --llm-model models/qwen2.5-1.5B
"""
import argparse
import os
import time
from copy import deepcopy
from dataclasses import replace

from tqdm import tqdm

from agents import LLMPlayer
from config import GameConfig, _PLAYER_CLASSES
from monopoly.core.cell import Cell, Property
from optimizer import run_log
from optimizer.simulate import run_single_game
from player_settings import RuleBasedPlayerSettings, StandardPlayerSettings
from prompts.loader import load_prompt


class NeutralLLMPlayer(LLMPlayer):
    """LLMPlayer subclass with no few-shot and the neutral system prompt
    loaded via the hash-locked prompt loader. Used by PHASE-A-ROBUSTNESS
    to re-run the Phase A matrix without GUIDED-PLAYER's strategic
    scaffolding (no shots, no archetypes) — same instrument as
    scripts/llm_character.py:CharacterLLMPlayer but plumbed through the
    `_PLAYER_CLASSES` registry that simulate.run_single_game uses."""

    _SYSTEM_PROMPT = load_prompt('character_llm_prompt.txt')
    _FEW_SHOT = []


# Register so simulate.py's class lookup can find it via class-name string.
_PLAYER_CLASSES['NeutralLLMPlayer'] = NeutralLLMPlayer


def _modify_salary(cfg, multiplier: float):
    out = deepcopy(cfg)
    out.settings = replace(
        out.settings,
        mechanics=replace(out.settings.mechanics,
                           salary=int(out.settings.mechanics.salary * multiplier)),
    )
    return out


def _remove_group(cfg, group_name: str):
    """Remove every Property in the given colour group, replacing each with
    a plain (no-effect) Cell at the same board index."""
    out = deepcopy(cfg)
    out.cells = [
        Cell(c.name) if isinstance(c, Property) and c.group == group_name else c
        for c in out.cells
    ]
    return out


def _scale_rent_in_group(cfg, group_name: str, multiplier: float):
    """Multiply rent_base and every rent_house tier by `multiplier` for all
    Properties in the given group."""
    out = deepcopy(cfg)
    new_cells = []
    for c in out.cells:
        if isinstance(c, Property) and c.group == group_name:
            new_cells.append(Property(
                c.name,
                c.cost_base,
                max(1, int(round(c.rent_base * multiplier))),
                c.cost_house,
                tuple(max(1, int(round(r * multiplier))) for r in c.rent_house),
                c.group,
            ))
        else:
            new_cells.append(c)
    out.cells = new_cells
    return out


def _summarise(results):
    """Given a list of per-game result dicts, return aggregate stats."""
    n = max(len(results), 1)
    rounds = sum(r['rounds'] for r in results) / n
    draws  = sum(1 for r in results if r['truncated']) / n
    xfer   = sum(r['transfer_total'] for r in results) / n
    bk_total = 0
    win_counts = {}
    for r in results:
        bk_total += sum(int(b) for b in r['bankrupt'].values())
        win_counts[r.get('winner')] = win_counts.get(r.get('winner'), 0) + 1
    bankruptcies = bk_total / n
    # Win-rate spread = max - min over named winners (excluding None/draw)
    named = [(k, v / n) for k, v in win_counts.items() if k is not None]
    if named:
        rates = [v for _, v in named]
        spread = max(rates) - min(rates) if len(rates) > 1 else max(rates)
    else:
        spread = 0.0
    return {
        'mean_rounds':   rounds,
        'draw_rate':     draws,
        'mean_xfer':     xfer,
        'bankruptcies':  bankruptcies,
        'wr_spread':     spread,
    }


def _run_batch(cfg, specs, n_games, base_seed, max_turns, progress_label=None,
               progress_every_turns: int = 5):
    out = []
    bar = None
    if progress_label is not None:
        bar = tqdm(total=n_games, desc=progress_label, unit='game',
                   leave=False, dynamic_ncols=True)
    for i in range(n_games):
        # Per-turn callback updates tqdm postfix every progress_every_turns
        # so the user sees liveness even when a single game takes 30-60s.
        def _cb(turn_n, players, transfer_so_far, _bar=bar):
            if _bar is None: return
            if turn_n % progress_every_turns == 0:
                _bar.set_postfix(turn=turn_n, xfer=int(transfer_so_far),
                                  refresh=True)
        out.append(run_single_game(
            cfg, specs, seed=base_seed + i, max_turns=max_turns,
            on_turn=_cb if bar is not None else None))
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return out


def _print_compare(label, default_stats, modified_stats):
    print(f'  {label:<14} '
          f'rounds={default_stats["mean_rounds"]:>6.1f} -> {modified_stats["mean_rounds"]:>6.1f}   '
          f'draws={100*default_stats["draw_rate"]:>5.1f}% -> {100*modified_stats["draw_rate"]:>5.1f}%   '
          f'xfer=${default_stats["mean_xfer"]:>6.0f} -> ${modified_stats["mean_xfer"]:>6.0f}   '
          f'bk={default_stats["bankruptcies"]:>4.2f} -> {modified_stats["bankruptcies"]:>4.2f}   '
          f'spread={default_stats["wr_spread"]:>4.2f} -> {modified_stats["wr_spread"]:>4.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mini-config', default='mini_config.yaml')
    ap.add_argument('--n-games',     type=int, default=30,
                    help='Games per condition for the rule-based phase. Cheap; default 30.')
    ap.add_argument('--n-games-llm', type=int, default=None,
                    help='Games per condition for the LLM phase. Default: n_games // 3 (LLM is slow).')
    ap.add_argument('--max-turns',   type=int, default=50,
                    help='Hard cap per game. Lower (50) for the LLM phase to '
                         'keep wall-clock manageable; bump for higher fidelity.')
    ap.add_argument('--base-seed',   type=int, default=42)
    ap.add_argument('--no-llm',      action='store_true',
                    help='Skip the LLM rows. Useful for fast iteration.')
    ap.add_argument('--llm-model',   default=None,
                    help='LLM_MODEL value (HF id or local path). Default: env var.')
    ap.add_argument('--backend',     choices=('local', 'openai', 'heuristic'),
                    default='local',
                    help='LLM backend. Use "openai" for vLLM / remote inference.')
    ap.add_argument('--player',      choices=('guided', 'neutral'), default='guided',
                    help="Which LLM prompt variant to use. 'guided' is the "
                         "original 4-shot directive (agents.LLMPlayer; in the "
                         "report). 'neutral' is the round-1 PHASE-A-ROBUSTNESS "
                         "re-run with no-shot prompt and no scaffolding.")
    ap.add_argument('--out-dir',     default=None,
                    help='Optional directory to write per-knob results JSON.')
    args = ap.parse_args()

    if args.llm_model:
        os.environ['LLM_MODEL'] = args.llm_model
    os.environ['LLM_BACKEND'] = args.backend

    out_dir = None
    if args.out_dir:
        from pathlib import Path as _PathTrack
        out_dir = _PathTrack(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    experiment = ('phase-a-robustness' if args.player == 'neutral'
                  else 'phase-a')
    with run_log.track(experiment=experiment, script=__file__,
                       args=vars(args), out_dir=out_dir,
                       condition=args.player) as run:
        return _run_phase_a(args, run)


def _run_phase_a(args, run) -> int:
    base_cfg = GameConfig.from_yaml(args.mini_config)
    print(f'Loaded mini config: {len(base_cfg.cells)} cells')

    # Sanity: confirm at least one of each colour group we plan to perturb is present.
    groups = {c.group for c in base_cfg.cells if isinstance(c, Property)}
    print(f'Property groups present: {sorted(groups)}')

    # Define the three knobs.
    knobs = [
        ('salary x2',     _modify_salary(base_cfg, 2.0)),
        ('drop Brown',    _remove_group(base_cfg, 'Brown')),
        ('Orange rent x2', _scale_rent_in_group(base_cfg, 'Orange', 2.0)),
    ]

    rb_specs = [
        ('A', RuleBasedPlayerSettings(), None),
        ('B', RuleBasedPlayerSettings(), None),
    ]
    llm_class_name = 'NeutralLLMPlayer' if args.player == 'neutral' else 'LLMPlayer'
    llm_specs = [
        ('A', StandardPlayerSettings(), llm_class_name),
        ('B', RuleBasedPlayerSettings(), None),
    ]
    print(f'LLM player variant: {args.player.upper()} ({llm_class_name})')

    results: dict = {'player_variant': args.player,
                     'n_games': args.n_games,
                     'rb': {}, 'llm': {}}

    print()
    print('=== Rule-based baseline (both players RuleBased) ===')
    print(f"  {'condition':<14} "
          f"{'rounds':>11}    {'draws':>11}    {'xfer':>13}    {'bk':>10}    {'spread':>10}")
    t0 = time.time()
    rb_default = _summarise(_run_batch(base_cfg, rb_specs, args.n_games, args.base_seed, args.max_turns))
    results['rb']['default'] = rb_default
    for label, mod_cfg in knobs:
        rb_mod = _summarise(_run_batch(mod_cfg, rb_specs, args.n_games, args.base_seed, args.max_turns))
        results['rb'][label] = rb_mod
        _print_compare(label, rb_default, rb_mod)
    print(f'  [{time.time()-t0:.1f} s for {(1+len(knobs))*args.n_games} games]')

    if not args.no_llm:
        print()
        print('=== LLM cross-check (P0 = LLMPlayer, P1 = RuleBased) ===')
        n_llm = args.n_games_llm if args.n_games_llm else max(1, args.n_games // 3)
        print(f"  {'condition':<14} "
              f"{'rounds':>11}    {'draws':>11}    {'xfer':>13}    {'bk':>10}    {'spread':>10}")
        t0 = time.time()
        llm_default = _summarise(_run_batch(
            base_cfg, llm_specs, n_llm, args.base_seed, args.max_turns,
            progress_label='default'))
        results['llm']['default'] = llm_default
        results['llm']['n_games'] = n_llm
        # Print the default result immediately so the user sees progress
        # before each subsequent knob's batch starts.
        print(f"  default        rounds={llm_default['mean_rounds']:>6.1f}            "
              f"draws={100*llm_default['draw_rate']:>5.1f}%               "
              f"xfer=${llm_default['mean_xfer']:>6.0f}              "
              f"bk={llm_default['bankruptcies']:>4.2f}           "
              f"spread={llm_default['wr_spread']:>4.2f}")
        for label, mod_cfg in knobs:
            llm_mod = _summarise(_run_batch(
                mod_cfg, llm_specs, n_llm, args.base_seed, args.max_turns,
                progress_label=label))
            results['llm'][label] = llm_mod
            _print_compare(label, llm_default, llm_mod)
        print(f'  [{time.time()-t0:.1f} s for {(1+len(knobs))*n_llm} games at n={n_llm}]')

    if args.out_dir:
        import json as _json
        from pathlib import Path as _Path
        out = _Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / f'results_{args.player}.json').write_text(
            _json.dumps(results, indent=2))
        print(f'  results -> {out / f"results_{args.player}.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main() or 0)
