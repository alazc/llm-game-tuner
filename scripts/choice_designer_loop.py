"""EXP-012 — choice-shaped actions: construction removed from the model.

Two arms, one question: once the model never CONSTRUCTS an edit, which of its
verified competences survives the loop?

  ARM A (menu)   the harness renders 7 candidate boards anchored on the
                 incumbent (3 rent-down rungs, a wrong-direction probe, two
                 decoys, hold); the model picks a LETTER (the EXP-004 P-EVAL
                 interface, looped). The pick applies unconditionally.
  ARM B (lever)  the model names only {lever, direction}; the harness
                 line-searches the magnitude (3 rungs + held incumbent, all at
                 the iteration's CRN seed) and applies the argmin iff it beats
                 the held read (EXP-010 paired acceptance). The model never
                 sees or emits a number.

Sealed in prereg EXP-012.

Usage:
    # heuristic IO smokes (no GPU)
    set PYTHONPATH=. && python scripts/choice_designer_loop.py --arm menu \
        --backend heuristic --boards default --n-seeds 1 --K 2 --n-games 20 \
        --out-dir report/figures/exp12_smoke
    set PYTHONPATH=. && python scripts/choice_designer_loop.py --arm lever \
        --backend heuristic --boards default --n-seeds 1 --K 2 --n-games 20 \
        --out-dir report/figures/exp12_smoke

    # analysis vs sealed thresholds
    set PYTHONPATH=. && python scripts/choice_designer_loop.py --analyze \
        --out-dir report/figures/exp12_run
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from optimizer.exp_boards import build_exp0_boards
from optimizer.group_design import (GroupDesign, apply_design,
                                    bootstrap_score_ci, monotonicity_gate)
from optimizer import run_log
from optimizer.skill_expression import G_STAR, BAND, render_goal_anchor
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool
from prompts.loader import load_prompt

from scripts.llm_design_loop import (DesignerLLM, Iteration, _RESPONSE_BLOCK,
                                     _design_diff_summary,
                                     _eval_with_aligned_seeds, eval_seed_for,
                                     _per_group_breakdown)
from scripts.mech_probe_loop import render_board
from scripts.scaffold_designer_loop import _worker_eval

ARMS = ('menu', 'lever', 'flat', 'free')
PEVAL_PROMPT = 'designer_llm_prompt_peval.txt'
LEVER_PROMPT = 'designer_llm_prompt_lever.txt'
FLAT_PROMPT = 'designer_llm_prompt_flat.txt'      # EXP-013: full raw vocabulary
FREE_PROMPT = 'designer_llm_prompt_freetext.txt'  # EXP-013: no vocabulary
ARM_PROMPT = {'menu': PEVAL_PROMPT, 'lever': LEVER_PROMPT,
              'flat': FLAT_PROMPT, 'free': FREE_PROMPT}

CLAMP = {'rent': (0.25, 4.0), 'cost': (0.25, 4.0), 'salary': (0.5, 3.0)}
# ARM A menu (sealed): (family, factor); None = hold.
MENU = [('rent', 0.5), ('rent', 0.7), ('rent', 0.85), ('rent', 1.15),
        ('cost', 0.7), ('salary', 1.5), (None, 1.0)]
LETTERS = 'ABCDEFG'
WRONG_DIRECTION = {('rent', 1.15), ('salary', 1.5)}
# ARM B rungs (sealed).
RUNGS = {'down': (0.5, 0.7, 0.85), 'up': (1.15, 1.4, 2.0)}


def _clamped(fam: str, v: float) -> float:
    lo, hi = CLAMP[fam]
    return max(lo, min(hi, v))


def _design_from_vals(groups: List[str], vals: Dict[str, float],
                      rationale: str) -> GroupDesign:
    return GroupDesign(
        salary_mult=round(vals['salary'], 6),
        group_rent_mult={g: round(vals['rent'], 6) for g in groups},
        group_cost_mult={g: round(vals['cost'], 6) for g in groups},
        label='cumulative', rationale=rationale)


def _apply_option(vals: Dict[str, float], fam: Optional[str],
                  factor: float) -> Dict[str, float]:
    out = dict(vals)
    if fam is not None:
        out[fam] = _clamped(fam, out[fam] * factor)
    return out


# --------------------------------------------------------------------------- #
# Parsers (letters A..G / lever schema), 1 retry each                           #
# --------------------------------------------------------------------------- #

def _parse_json_block(text: str) -> Optional[dict]:
    cands = [m.group(1) for m in _RESPONSE_BLOCK.finditer(text)] or [text.strip()]
    for c in cands:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def parse_pick(text: str, n_letters: int) -> Tuple[Optional[str], str, str]:
    d = _parse_json_block(text)
    if d is None:
        return None, '', 'parser_failure'
    pick = str(d.get('pick', '')).strip().upper()[:1]
    rat = str(d.get('rationale', '')).strip()
    if pick not in LETTERS[:n_letters]:
        return None, rat, 'invalid_pick'
    return pick, rat, 'ok'


# ------------------------- EXP-013: naming without curation ----------------- #

def knob_vocabulary(groups: List[str]) -> List[str]:
    """Full raw knob list, no family curation: rent:<G> x8, cost:<G> x8, salary."""
    return ([f'rent:{g}' for g in groups] + [f'cost:{g}' for g in groups]
            + ['salary'])


def parse_targets(text: str, vocab: List[str]
                  ) -> Tuple[Optional[List[str]], Optional[str], str, str]:
    """ARM FLAT reply: {"targets": [exact knob names], "direction"}."""
    d = _parse_json_block(text)
    if d is None:
        return None, None, '', 'parser_failure'
    rat = str(d.get('rationale', '')).strip()
    direction = str(d.get('direction', '')).strip().lower()
    raw_targets = d.get('targets')
    if direction not in RUNGS or not isinstance(raw_targets, list) or not raw_targets:
        return None, None, rat, 'invalid_pick'
    low = {v.lower(): v for v in vocab}
    targets = []
    for t in raw_targets:
        key = str(t).strip().lower()
        if key not in low:
            return None, None, rat, 'invalid_name'
        targets.append(low[key])
    return sorted(set(targets)), direction, rat, 'ok'


def ground_free(text: str, groups: List[str]) -> Optional[List[str]]:
    """SEALED grounder (prereg EXP-013): map a free-text phrase to knobs.
    Group names mentioned -> those groups (else ALL groups); /rent/ -> rent
    knobs of those groups; /cost|price|purchase/ -> cost knobs; /salary|go
    money|passing go|payout/ -> salary. Union; empty -> None."""
    low = text.lower()
    mentioned = [g for g in groups if g.lower() in low]
    target_groups = mentioned or groups
    knobs: List[str] = []
    if re.search(r'rent', low):
        knobs += [f'rent:{g}' for g in target_groups]
    if re.search(r'cost|price|purchas', low):
        knobs += [f'cost:{g}' for g in target_groups]
    if re.search(r'salar|go money|passing go|payout', low):
        knobs.append('salary')
    return sorted(set(knobs)) or None


def parse_free(text: str) -> Tuple[Optional[str], Optional[str], str, str]:
    """ARM FREE reply: {"change": "<phrase>", "direction"} (grounding separate)."""
    d = _parse_json_block(text)
    if d is None:
        return None, None, '', 'parser_failure'
    rat = str(d.get('rationale', '')).strip()
    change = str(d.get('change', '')).strip()
    direction = str(d.get('direction', '')).strip().lower()
    if not change or direction not in RUNGS:
        return None, None, rat, 'invalid_pick'
    return change, direction, rat, 'ok'


def _knob_design(knobvals: Dict[str, float], groups: List[str],
                 rationale: str) -> GroupDesign:
    """Per-knob values -> GroupDesign (knob keys: 'rent:<G>', 'cost:<G>', 'salary')."""
    return GroupDesign(
        salary_mult=round(knobvals.get('salary', 1.0), 6),
        group_rent_mult={g: round(knobvals.get(f'rent:{g}', 1.0), 6)
                         for g in groups},
        group_cost_mult={g: round(knobvals.get(f'cost:{g}', 1.0), 6)
                         for g in groups},
        label='cumulative', rationale=rationale)


def _apply_rung(knobvals: Dict[str, float], targets: List[str],
                factor: float) -> Dict[str, float]:
    out = dict(knobvals)
    for t in targets:
        fam = 'salary' if t == 'salary' else t.split(':', 1)[0]
        out[t] = _clamped(fam, out.get(t, 1.0) * factor)
    return out


def _lever_correct(targets: List[str], direction: str) -> bool:
    """Correct-direction move per the known levers: rent-down or salary-down."""
    fams = {('salary' if t == 'salary' else t.split(':', 1)[0]) for t in targets}
    return direction == 'down' and fams <= {'rent', 'salary'} and 'rent' in fams \
        or (fams == {'salary'} and direction == 'down')


def parse_lever(text: str) -> Tuple[Optional[str], Optional[str], str, str]:
    d = _parse_json_block(text)
    if d is None:
        return None, None, '', 'parser_failure'
    lever = str(d.get('lever', '')).strip().lower()
    direction = str(d.get('direction', '')).strip().lower()
    rat = str(d.get('rationale', '')).strip()
    if lever not in CLAMP or direction not in RUNGS:
        return None, None, rat, 'invalid_pick'
    return lever, direction, rat, 'ok'


# --------------------------------------------------------------------------- #
# Feeds                                                                          #
# --------------------------------------------------------------------------- #

def _context_lines(share: float, score: float,
                   prior_moves: List[str]) -> List[str]:
    parts = ['## GOAL', '  ' + render_goal_anchor(share, G_STAR, 'reduce'),
             '', '## CURRENT EVAL',
             f'  skill_share (endpoint): {share:.3f}',
             f'  score: {score:.4f}  (lower is better; 0 inside the dead-zone)']
    if prior_moves:
        parts += ['', '## YOUR PRIOR MOVES (most recent last)']
        parts += [f'  {m}' for m in prior_moves[-3:]]
    return parts


def build_menu_feed(option_cfgs, letters_order, share, score, prior_moves):
    parts = _context_lines(share, score, prior_moves)
    parts += ['', '## CANDIDATE BOARDS (structure only — pick the ONE whose '
                  'structure best reaches the GOAL)']
    for letter, cfg in zip(letters_order, option_cfgs):
        parts.append(render_board(cfg, letter))
    parts += ['', '## YOUR JOB',
              '  Pick the ONE board (by letter) that best moves the share '
              'toward g*=0.60. Reply with the JSON schema from the system '
              'prompt.']
    return '\n'.join(parts)


def build_lever_feed(cfg, share, score, prior_moves):
    parts = _context_lines(share, score, prior_moves)
    bd = _per_group_breakdown(cfg)
    parts += ['', '## PER-GROUP COST/RENT BREAKDOWN']
    parts += [f'  - {r["group"]:>10}: n={r["n"]} mean_cost=${r["mean_cost"]:.0f} '
              f'mean_rent=${r["mean_rent"]:.0f}' for r in bd]
    parts += ['', '## YOUR JOB',
              '  Name the lever family and direction for the next move. The '
              'harness will calibrate the amount by simulation. Reply with '
              'the JSON schema from the system prompt.']
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Trajectory runners                                                            #
# --------------------------------------------------------------------------- #

def _baseline_record(arm, starting_cfg, board_label, seed, pool, matchups,
                     args, gen_base, fh):
    t0 = time.perf_counter()
    ev = _eval_with_aligned_seeds(starting_cfg, pool, matchups, args.n_games,
                                  board_label, seed, 0, args.max_turns)
    ci = bootstrap_score_ci(ev['shares'], n_resamples=1500, seed=seed)
    vals = {'rent': 1.0, 'cost': 1.0, 'salary': 1.0}
    d0 = GroupDesign(label='cumulative')
    rec0 = Iteration(
        iter=0, design=d0.to_dict(), design_diff=_design_diff_summary(d0),
        rationale='[baseline]', parser_status='baseline', parser_error=None,
        converged_request=False, convergence_padded=False,
        convergence_violation=False, parser_retry=False,
        score=ev['score'], metrics=ev['metrics'], score_ci=ci,
        delta_vs_prev=None, improvement=None,
        n_games=ev['n_games_total'], token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='',
        condition=arm, gen_cfg=gen_base,
        eval_seed_base=eval_seed_for(board_label, seed, 0, 0),
        skill_curve=ev.get('skill_curve'))
    fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()
    return rec0, vals


def _pad_record(arm, k, vals, groups, last, gen_base, ev_seed, t0):
    d = _design_from_vals(groups, vals, '[harness-stop: carry-forward]')
    return Iteration(
        iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
        rationale='[harness-stop: carry-forward]', parser_status='ok',
        parser_error=None, converged_request=True, convergence_padded=True,
        convergence_violation=False, parser_retry=False,
        score=last.score, metrics=last.metrics, score_ci=last.score_ci,
        delta_vs_prev=0.0, improvement=False, n_games=0, token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='',
        condition=arm, gen_cfg=gen_base, eval_seed_base=ev_seed)


def _carry_record(arm, k, vals, groups, last, gen_cfg, ev_seed, t0,
                  status, err, raw):
    d = _design_from_vals(groups, vals, f'[{status}: carry-forward]')
    return Iteration(
        iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
        rationale=f'[{status}: carry-forward]', parser_status=status,
        parser_error=err, converged_request=False, convergence_padded=False,
        convergence_violation=False, parser_retry=True,
        score=last.score, metrics=last.metrics, score_ci=last.score_ci,
        delta_vs_prev=0.0, improvement=False, n_games=0, token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response=raw[:2000],
        condition=arm, gen_cfg=gen_cfg, eval_seed_base=ev_seed)


def run_menu_trajectory(starting_cfg, board_label, board_tag, seed, pool,
                        matchups, args, designer, out_path) -> List[Iteration]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    groups = [r['group'] for r in _per_group_breakdown(starting_cfg)]
    gen_base = {'engine': 'choice', 'arm': 'menu', 'model': designer.model_name,
                'prompt': PEVAL_PROMPT, 'note': 'prereg EXP-012 arm A'}
    rec0, vals = _baseline_record('menu', starting_cfg, board_label, seed,
                                  pool, matchups, args, gen_base, fh)
    iterations = [rec0]
    prior_moves: List[str] = []
    stopped = False

    for k in range(1, args.K + 1):
        t0 = time.perf_counter()
        ev_seed = eval_seed_for(board_label, seed, k, 0)
        if stopped:
            rec = _pad_record('menu', k, vals, groups, iterations[-1],
                              gen_base, ev_seed, t0)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush(); continue

        last = iterations[-1]
        share = (last.metrics or {}).get('skill_share')
        # candidate cfgs, shuffled letters (seeded; letter != identity)
        rng = np.random.default_rng(eval_seed_for(board_label, seed, k, 2))
        order = list(rng.permutation(len(MENU)))
        opt_vals = [_apply_option(vals, *MENU[j]) for j in order]
        opt_cfgs = [apply_design(starting_cfg,
                                 _design_from_vals(groups, v, ''),
                                 strict_groups=False) for v in opt_vals]
        feed = build_menu_feed(opt_cfgs, LETTERS, share, last.score,
                               prior_moves)

        if args.backend == 'heuristic':     # canned: pick the rent x0.7 letter
            j_target = order.index(MENU.index(('rent', 0.7)))
            raw = ('```json\n' + json.dumps(
                {'pick': LETTERS[j_target],
                 'rationale': '[heuristic] canned rent x0.7'}) + '\n```')
        else:
            raw, _ = designer.query(feed, iteration=k - 1)
        pick, rat, status = parse_pick(raw, len(MENU))
        if status != 'ok':                  # one format-reminder retry
            raw2, _ = (designer.query(
                feed + '\n\nReply with EXACTLY one fenced ```json``` object: '
                       '{"pick":"<letter>","rationale":"..."}.',
                iteration=k - 1) if args.backend != 'heuristic' else (raw, None))
            p2, r2, s2 = parse_pick(raw2, len(MENU))
            if s2 == 'ok':
                pick, rat, status, raw = p2, r2, s2, raw2
        if status != 'ok':
            rec = _carry_record('menu', k, vals, groups, last,
                                dict(gen_base, pick=None), ev_seed, t0,
                                status, 'unparseable pick', raw)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush(); continue

        j = order[LETTERS.index(pick)]
        fam, factor = MENU[j]
        vals = opt_vals[LETTERS.index(pick)]
        d = _design_from_vals(groups, vals,
                              f'[menu pick {pick}: {fam} x{factor}] ' + rat[:120])
        cfg2 = apply_design(starting_cfg, d, strict_groups=False)
        ev = _eval_with_aligned_seeds(cfg2, pool, matchups, args.n_games,
                                      board_label, seed, k, args.max_turns)
        ci = bootstrap_score_ci(ev['shares'], n_resamples=1500, seed=seed + k)
        prev = last.score
        gen_cfg = dict(gen_base, pick=pick, option=[fam, factor],
                       wrong_direction=(fam, factor) in WRONG_DIRECTION,
                       hold=fam is None)
        stop_now = ev['score'] == 0
        rec = Iteration(
            iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
            rationale=d.rationale, parser_status='ok', parser_error=None,
            converged_request=stop_now, convergence_padded=False,
            convergence_violation=False, parser_retry=False,
            score=ev['score'], metrics=ev['metrics'], score_ci=ci,
            delta_vs_prev=ev['score'] - prev,
            improvement=bool(ev['score'] < prev),
            n_games=ev['n_games_total'], token_count=None,
            wall_seconds=time.perf_counter() - t0, raw_response=raw[:1500],
            condition='menu', gen_cfg=gen_cfg, eval_seed_base=ev_seed,
            skill_curve=ev.get('skill_curve'))
        iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
        fh.flush()
        prior_moves.append(f'iter {k}: chose {fam or "hold"} x{factor} '
                           f'-> share {ev["skill_share"]:.3f}')
        if stop_now:
            stopped = True
    fh.close()
    return iterations


def run_lever_trajectory(starting_cfg, board_label, board_tag, seed, pool,
                         matchups, args, designer, executor,
                         out_path) -> List[Iteration]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    groups = [r['group'] for r in _per_group_breakdown(starting_cfg)]
    gen_base = {'engine': 'choice', 'arm': 'lever', 'model': designer.model_name,
                'prompt': LEVER_PROMPT, 'note': 'prereg EXP-012 arm B'}
    rec0, vals = _baseline_record('lever', starting_cfg, board_label, seed,
                                  pool, matchups, args, gen_base, fh)
    iterations = [rec0]
    prior_moves: List[str] = []
    stopped = False

    for k in range(1, args.K + 1):
        t0 = time.perf_counter()
        ev_seed = eval_seed_for(board_label, seed, k, 0)
        if stopped:
            rec = _pad_record('lever', k, vals, groups, iterations[-1],
                              gen_base, ev_seed, t0)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush(); continue

        last = iterations[-1]
        share = (last.metrics or {}).get('skill_share')
        cfg_now = apply_design(starting_cfg,
                               _design_from_vals(groups, vals, ''),
                               strict_groups=False)
        feed = build_lever_feed(cfg_now, share, last.score, prior_moves)

        if args.backend == 'heuristic':
            raw = ('```json\n' + json.dumps(
                {'lever': 'rent', 'direction': 'down',
                 'rationale': '[heuristic] canned rent down'}) + '\n```')
        else:
            raw, _ = designer.query(feed, iteration=k - 1)
        lever, direction, rat, status = parse_lever(raw)
        if status != 'ok':
            raw2, _ = (designer.query(
                feed + '\n\nReply with EXACTLY one fenced ```json``` object: '
                       '{"lever":"rent|cost|salary","direction":"down|up",'
                       '"rationale":"..."}.',
                iteration=k - 1) if args.backend != 'heuristic' else (raw, None))
            l2, d2, r2, s2 = parse_lever(raw2)
            if s2 == 'ok':
                lever, direction, rat, status, raw = l2, d2, r2, s2, raw2
        if status != 'ok':
            rec = _carry_record('lever', k, vals, groups, last,
                                dict(gen_base, lever=None), ev_seed, t0,
                                status, 'unparseable lever reply', raw)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush(); continue

        # Harness line-search: 3 rungs + held, all at this iteration's seed.
        rung_vals = [_apply_option(vals, lever, f) for f in RUNGS[direction]]
        tasks = [{'idx': j,
                  'design': _design_from_vals(groups, v, '').to_dict(),
                  'board_tag': board_tag, 'base_seed': ev_seed,
                  'n_games': args.n_games, 'max_turns': args.max_turns,
                  'canonical_config': args.canonical_config}
                 for j, v in enumerate(rung_vals)]
        tasks.append({'idx': len(rung_vals),
                      'design': _design_from_vals(groups, vals, '').to_dict(),
                      'board_tag': board_tag, 'base_seed': ev_seed,
                      'n_games': args.n_games, 'max_turns': args.max_turns,
                      'canonical_config': args.canonical_config})
        results = {r['idx']: r for r in executor.map(_worker_eval, tasks)}
        scores = [results[j]['score'] for j in range(len(rung_vals))]
        j_min = min(range(len(rung_vals)), key=lambda j: scores[j])
        held = results[len(rung_vals)]
        accepted = scores[j_min] < held['score']
        if accepted:
            vals = rung_vals[j_min]
            evj = results[j_min]
            chosen = RUNGS[direction][j_min]
            rationale = (f'[lever {lever} {direction}; harness rung x{chosen}; '
                         f'{scores[j_min]:.4f} < held {held["score"]:.4f}] '
                         + rat[:100])
        else:
            evj = held
            chosen = None
            rationale = (f'[lever {lever} {direction}; no rung beats held '
                         f'{held["score"]:.4f}; hold] ' + rat[:100])
        ci = bootstrap_score_ci(evj['shares'], n_resamples=1500, seed=seed + k)
        prev = last.score
        gen_cfg = dict(gen_base, lever=lever, direction=direction,
                       rung=chosen, held_score=held['score'],
                       rung_scores=[round(s, 5) for s in scores],
                       accepted=accepted)
        stop_now = evj['score'] == 0
        d = _design_from_vals(groups, vals, rationale)
        rec = Iteration(
            iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
            rationale=rationale, parser_status='ok', parser_error=None,
            converged_request=stop_now, convergence_padded=False,
            convergence_violation=False, parser_retry=False,
            score=evj['score'], metrics=evj['metrics'], score_ci=ci,
            delta_vs_prev=evj['score'] - prev,
            improvement=bool(evj['score'] < prev),
            n_games=args.n_games * (len(rung_vals) + 1), token_count=None,
            wall_seconds=time.perf_counter() - t0, raw_response=raw[:1500],
            condition='lever', gen_cfg=gen_cfg, eval_seed_base=ev_seed,
            skill_curve=evj.get('skill_curve'))
        iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
        fh.flush()
        prior_moves.append(f'iter {k}: {lever} {direction} '
                           f'{"x" + str(chosen) if chosen else "(held)"} '
                           f'-> share {(evj["metrics"] or {}).get("skill_share", float("nan")):.3f}')
        if stop_now:
            stopped = True
    fh.close()
    return iterations


def build_named_feed(arm: str, cfg, share, score, prior_moves,
                     vocab: List[str]) -> str:
    parts = _context_lines(share, score, prior_moves)
    bd = _per_group_breakdown(cfg)
    parts += ['', '## PER-GROUP COST/RENT BREAKDOWN']
    parts += [f'  - {r["group"]:>10}: n={r["n"]} mean_cost=${r["mean_cost"]:.0f} '
              f'mean_rent=${r["mean_rent"]:.0f}' for r in bd]
    if arm == 'flat':
        parts += ['', '## ADJUSTABLE KNOBS (use ONLY these exact names)',
                  '  ' + ', '.join(vocab)]
        parts += ['', '## YOUR JOB',
                  '  Name the knob(s) to adjust next and the direction. The '
                  'harness will calibrate the amount by simulation. Reply '
                  'with the JSON schema from the system prompt.']
    else:
        parts += ['', '## YOUR JOB',
                  '  Describe in one short phrase what to change next and the '
                  'direction. The harness will build and calibrate it by '
                  'simulation. Reply with the JSON schema from the system '
                  'prompt.']
    return '\n'.join(parts)


def run_named_trajectory(arm, starting_cfg, board_label, board_tag, seed,
                         pool, matchups, args, designer, executor,
                         out_path) -> List[Iteration]:
    """EXP-013 arms: FLAT (full raw vocabulary) and FREE (no vocabulary;
    sealed grounder). Downstream of naming = the EXP-012 ARM B machinery."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    groups = [r['group'] for r in _per_group_breakdown(starting_cfg)]
    vocab = knob_vocabulary(groups)
    gen_base = {'engine': 'choice', 'arm': arm, 'model': designer.model_name,
                'prompt': ARM_PROMPT[arm], 'note': 'prereg EXP-013'}
    rec0, _ = _baseline_record(arm, starting_cfg, board_label, seed,
                               pool, matchups, args, gen_base, fh)
    iterations = [rec0]
    knobvals: Dict[str, float] = {}
    prior_moves: List[str] = []
    stopped = False

    for k in range(1, args.K + 1):
        t0 = time.perf_counter()
        ev_seed = eval_seed_for(board_label, seed, k, 0)
        if stopped:
            d = _knob_design(knobvals, groups, '[harness-stop: carry-forward]')
            last = iterations[-1]
            rec = Iteration(
                iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
                rationale='[harness-stop: carry-forward]', parser_status='ok',
                parser_error=None, converged_request=True,
                convergence_padded=True, convergence_violation=False,
                parser_retry=False, score=last.score, metrics=last.metrics,
                score_ci=last.score_ci, delta_vs_prev=0.0, improvement=False,
                n_games=0, token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=arm, gen_cfg=gen_base, eval_seed_base=ev_seed)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush(); continue

        last = iterations[-1]
        share = (last.metrics or {}).get('skill_share')
        cfg_now = apply_design(starting_cfg, _knob_design(knobvals, groups, ''),
                               strict_groups=False)
        feed = build_named_feed(arm, cfg_now, share, last.score, prior_moves,
                                vocab)

        if args.backend == 'heuristic':
            raw = ('```json\n' + json.dumps(
                {'targets': [f'rent:{g}' for g in groups],
                 'direction': 'down', 'rationale': '[heuristic] all rents down'}
                if arm == 'flat' else
                {'change': 'lower all rents', 'direction': 'down',
                 'rationale': '[heuristic] lower all rents'}) + '\n```')
        else:
            raw, _ = designer.query(feed, iteration=k - 1)

        def _interpret(text):
            if arm == 'flat':
                tg, dr, rt, st = parse_targets(text, vocab)
                return tg, dr, rt, st, st
            ch, dr, rt, st = parse_free(text)
            if st != 'ok':
                return None, dr, rt, st, st
            tg = ground_free(ch, groups)
            return tg, dr, (ch + ' | ' + rt), st, \
                ('ok' if tg else 'grounding_failure')

        targets, direction, rat, pstat, gstat = _interpret(raw)
        if gstat != 'ok':
            retry_hint = ('\n\nReply with EXACTLY one fenced ```json``` object '
                          'matching the schema in the system prompt.')
            raw2, _ = (designer.query(feed + retry_hint, iteration=k - 1)
                       if args.backend != 'heuristic' else (raw, None))
            t2 = _interpret(raw2)
            if t2[4] == 'ok':
                targets, direction, rat, pstat, gstat = t2
                raw = raw2
        if gstat != 'ok':
            d = _knob_design(knobvals, groups, f'[{gstat}: carry-forward]')
            rec = Iteration(
                iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
                rationale=f'[{gstat}: carry-forward] ' + rat[:120],
                parser_status=gstat, parser_error=gstat,
                converged_request=False, convergence_padded=False,
                convergence_violation=False, parser_retry=True,
                score=last.score, metrics=last.metrics, score_ci=last.score_ci,
                delta_vs_prev=0.0, improvement=False, n_games=0,
                token_count=None, wall_seconds=time.perf_counter() - t0,
                raw_response=raw[:1500], condition=arm,
                gen_cfg=dict(gen_base, ground_status=gstat),
                eval_seed_base=ev_seed)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush(); continue

        rung_sets = [_apply_rung(knobvals, targets, f)
                     for f in RUNGS[direction]]
        tasks = [{'idx': j,
                  'design': _knob_design(v, groups, '').to_dict(),
                  'board_tag': board_tag, 'base_seed': ev_seed,
                  'n_games': args.n_games, 'max_turns': args.max_turns,
                  'canonical_config': args.canonical_config}
                 for j, v in enumerate(rung_sets)]
        tasks.append({'idx': len(rung_sets),
                      'design': _knob_design(knobvals, groups, '').to_dict(),
                      'board_tag': board_tag, 'base_seed': ev_seed,
                      'n_games': args.n_games, 'max_turns': args.max_turns,
                      'canonical_config': args.canonical_config})
        results = {r['idx']: r for r in executor.map(_worker_eval, tasks)}
        scores = [results[j]['score'] for j in range(len(rung_sets))]
        j_min = min(range(len(rung_sets)), key=lambda j: scores[j])
        held = results[len(rung_sets)]
        accepted = scores[j_min] < held['score']
        if accepted:
            knobvals = rung_sets[j_min]
            evj = results[j_min]
            chosen = RUNGS[direction][j_min]
            rationale = (f'[{arm}: {len(targets)} knobs {direction}; rung '
                         f'x{chosen}; {scores[j_min]:.4f} < held '
                         f'{held["score"]:.4f}] ' + rat[:90])
        else:
            evj = held
            chosen = None
            rationale = (f'[{arm}: {len(targets)} knobs {direction}; no rung '
                         f'beats held {held["score"]:.4f}; hold] ' + rat[:90])
        ci = bootstrap_score_ci(evj['shares'], n_resamples=1500, seed=seed + k)
        prev = last.score
        gen_cfg = dict(gen_base, targets=targets, direction=direction,
                       n_targets=len(targets), rung=chosen,
                       held_score=held['score'], accepted=accepted,
                       ground_status='ok',
                       lever_correct=_lever_correct(targets, direction),
                       raw_change=rat[:160])
        stop_now = evj['score'] == 0
        d = _knob_design(knobvals, groups, rationale)
        rec = Iteration(
            iter=k, design=d.to_dict(), design_diff=_design_diff_summary(d),
            rationale=rationale, parser_status='ok', parser_error=None,
            converged_request=stop_now, convergence_padded=False,
            convergence_violation=False, parser_retry=False,
            score=evj['score'], metrics=evj['metrics'], score_ci=ci,
            delta_vs_prev=evj['score'] - prev,
            improvement=bool(evj['score'] < prev),
            n_games=args.n_games * (len(rung_sets) + 1), token_count=None,
            wall_seconds=time.perf_counter() - t0, raw_response=raw[:1500],
            condition=arm, gen_cfg=gen_cfg, eval_seed_base=ev_seed,
            skill_curve=evj.get('skill_curve'))
        iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
        fh.flush()
        prior_moves.append(
            f'iter {k}: {direction} on {len(targets)} knob(s) '
            f'{"x" + str(chosen) if chosen else "(held)"} -> share '
            f'{(evj["metrics"] or {}).get("skill_share", float("nan")):.3f}')
        if stop_now:
            stopped = True
    fh.close()
    return iterations


# --------------------------------------------------------------------------- #
# Analysis vs sealed thresholds                                                  #
# --------------------------------------------------------------------------- #

def analyze(out_dir: Path, gate_n: int = 120) -> int:
    lines: List[str] = []

    def say(s=''):
        print(s); lines.append(s)

    labels = {re.sub(r'[^A-Za-z0-9]+', '_', l).strip('_'): c
              for l, c in build_exp0_boards()}
    say(f'=== EXP-012 choice-shaped actions :: {out_dir} ===')
    verdicts = {}
    for arm in ARMS:
        cells = {}
        for p in sorted(glob.glob(str(out_dir / arm / f'{arm}__*.jsonl'))):
            recs = [json.loads(l) for l in open(p, encoding='utf-8')]
            mm = re.match(rf'{arm}__(.+)__seed(\d+)', Path(p).stem)
            if not mm:
                continue
            board, seed = mm.group(1), int(mm.group(2))
            first = next((r['iter'] for r in recs
                          if r['iter'] >= 1 and r['score'] == 0), None)
            live = [r for r in recs
                    if r['iter'] >= 1 and not r['convergence_padded']]
            cells[(board, seed)] = dict(
                final_score=recs[-1]['score'],
                final_share=(recs[-1].get('metrics') or {}).get('skill_share'),
                final_inband=recs[-1]['score'] == 0, first_inband=first,
                parse_ok=sum(1 for r in live if r['parser_status'] == 'ok'),
                n_live=len(live), final_design=recs[-1]['design'], recs=recs)
        if not cells:
            say(f'\n--- {arm}: NO DATA'); continue
        n_in = sum(v['final_inband'] for v in cells.values())
        say(f'\n--- ARM {arm.upper()}: final in-band {n_in}/{len(cells)}')
        for (b, s), v in sorted(cells.items()):
            say(f'   {b:>14} seed={s}: final share={v["final_share"]:.3f} '
                f'score={v["final_score"]:.4f} '
                f'in-band={"Y" if v["final_inband"] else "n"} '
                f'first={v["first_inband"] if v["first_inband"] is not None else "-"} '
                f'parse {v["parse_ok"]}/{v["n_live"]}')
        med = sorted(v['first_inband'] for v in cells.values()
                     if v['first_inband'] is not None)
        med_txt = med[len(med) // 2] if med else None
        # ladder gates
        n_gate_fail, seen = 0, {}
        for (b, s), v in sorted(cells.items()):
            if not v['final_inband']:
                continue
            key = json.dumps(v['final_design'], sort_keys=True) + '|' + b
            if key not in seen:
                cfg = apply_design(labels[b],
                                   GroupDesign.from_dict(v['final_design']),
                                   strict_groups=False)
                seen[key] = monotonicity_gate(cfg, n_seeds=gate_n, base_seed=0)
            res = seen[key]
            n_gate_fail += int(not res['passes'])
            say(f'   gate {b:>14} seed={s}: rank_corr={res["rank_corr"]:+.2f} '
                f'{"PASS" if res["passes"] else "FAIL"}')
        n_valid = n_in - n_gate_fail
        verdicts[arm] = (n_valid, len(cells), med_txt)

        # arm-specific behavioral tables
        if arm == 'menu':
            picks, wrong_by_bin = {}, {}
            for v in cells.values():
                for r in v['recs']:
                    g = r.get('gen_cfg') or {}
                    if r['iter'] >= 1 and not r['convergence_padded'] \
                            and g.get('pick'):
                        fam, factor = g['option']
                        key = f'{fam or "hold"} x{factor}'
                        picks[key] = picks.get(key, 0) + 1
                        prev_sc = next((x['score'] for x in v['recs']
                                        if x['iter'] == r['iter'] - 1), None)
                        if prev_sc is not None:
                            bin_ = min(int(prev_sc / 0.05), 3)
                            wb = wrong_by_bin.setdefault(bin_, [0, 0])
                            wb[1] += 1
                            wb[0] += int(g.get('wrong_direction', False))
            say('   pick distribution: ' + '  '.join(
                f'{k}={n}' for k, n in sorted(picks.items(),
                                              key=lambda kv: -kv[1])))
            for bin_ in sorted(wrong_by_bin):
                w, n = wrong_by_bin[bin_]
                say(f'   wrong-direction rate @ incumbent score '
                    f'{bin_ * 0.05:.2f}-{(bin_ + 1) * 0.05:.2f}: '
                    f'{w}/{n} = {w / n:.2f}')
        elif arm == 'lever':
            levers, accepts = {}, [0, 0]
            for v in cells.values():
                for r in v['recs']:
                    g = r.get('gen_cfg') or {}
                    if r['iter'] >= 1 and not r['convergence_padded'] \
                            and g.get('lever'):
                        key = f'{g["lever"]}-{g["direction"]}'
                        levers[key] = levers.get(key, 0) + 1
                        accepts[1] += 1
                        accepts[0] += int(g.get('accepted', False))
            say('   lever distribution: ' + '  '.join(
                f'{k}={n}' for k, n in sorted(levers.items(),
                                              key=lambda kv: -kv[1])))
            say(f'   rung accepted: {accepts[0]}/{accepts[1]}')
        else:                       # EXP-013 flat / free
            n_live = n_ok = n_correct = 0
            fails = {}
            sizes = []
            fam_dirs = {}
            for v in cells.values():
                for r in v['recs']:
                    if r['iter'] < 1 or r['convergence_padded']:
                        continue
                    g = r.get('gen_cfg') or {}
                    n_live += 1
                    if r['parser_status'] != 'ok':
                        fails[r['parser_status']] = \
                            fails.get(r['parser_status'], 0) + 1
                        continue
                    n_ok += 1
                    n_correct += int(g.get('lever_correct', False))
                    sizes.append(g.get('n_targets', 0))
                    fams = sorted({('salary' if t == 'salary'
                                    else t.split(':', 1)[0])
                                   for t in (g.get('targets') or [])})
                    key = '+'.join(fams) + '-' + g.get('direction', '?')
                    fam_dirs[key] = fam_dirs.get(key, 0) + 1
            say(f'   grounded {n_ok}/{n_live} '
                f'(failures: {fails or "none"})  '
                f'lever-correct {n_correct}/{n_ok if n_ok else 1} '
                f'= {n_correct / n_ok if n_ok else 0:.2f}')
            say('   named-set sizes: ' + (
                f'median {sorted(sizes)[len(sizes) // 2]}, '
                f'min {min(sizes)}, max {max(sizes)}' if sizes else 'n/a'))
            say('   family-direction distribution: ' + '  '.join(
                f'{k}={n}' for k, n in sorted(fam_dirs.items(),
                                              key=lambda kv: -kv[1])))

    fl = verdicts.get('flat'); fr = verdicts.get('free')
    if fl or fr:
        say('\n--- verdicts vs sealed predictions (prereg EXP-013) ---')
        if fl:
            say(f'  ARM FLAT in-band(valid) {fl[0]}/{fl[1]} (sealed >=7/9): '
                f'{"MET" if fl[0] >= 7 else "NOT MET"}; median-to-band {fl[2]}')
        if fr:
            say(f'  ARM FREE in-band(valid) {fr[0]}/{fr[1]} (sealed >=6/9 + '
                f'grounding >=0.8): {"MET" if fr[0] >= 6 else "NOT MET"}')
        if fl and fr:
            if fl[0] >= 7 and fr[0] >= 6:
                say('  JOINT: naming is genuinely OPEN-ENDED — the EXP-012 '
                    'curated vocabulary was not load-bearing.')
            elif fl[0] >= 7:
                say('  JOINT: naming needs a VISIBLE SCHEMA (exact names '
                    'shown); free text does not ground reliably.')
            else:
                say('  JOINT: EXP-012-B was CURATION-DEPENDENT — the '
                    'restraint critique is confirmed quantitatively.')
    say('\n--- verdicts vs sealed predictions (prereg EXP-012) ---')
    a = verdicts.get('menu'); b = verdicts.get('lever')
    if b:
        say(f'  ARM B (lever)  in-band(valid) {b[0]}/{b[1]} (sealed >=7/9): '
            f'{"MET" if b[0] >= 7 else "NOT MET"}; median-to-band {b[2]} '
            f'(sealed <=6)')
    if a:
        say(f'  ARM A (menu)   in-band(valid) {a[0]}/{a[1]} (sealed >=5/9): '
            f'{"MET" if a[0] >= 5 else "NOT MET"}; median-to-band {a[2]}')
    if a and b:
        cls = {(True, True): '1: constructive deficit confirmed; choice '
                             'survives recursion (working designer)',
               (False, True): '2: durable contribution = NAMING the move; '
                              'magnitude fails even as choice',
               (False, False): '3: recursion deficit reaches lever choice '
                               '(check lever distribution)',
               (True, False): '4: interface-format anomaly'}
        say('  JOINT CLASSIFICATION ' +
            cls[(a[0] >= 5, b[0] >= 7)])
    (out_dir / 'EXP12_REPORT.txt').write_text('\n'.join(lines) + '\n',
                                              encoding='utf-8')
    say(f'\nwrote {out_dir / "EXP12_REPORT.txt"}')
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='both',
                    help="Arm name, comma list, 'both' (menu,lever) or "
                         "'named' (flat,free).")
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--boards', default='all')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--pool', default='optimizer/strategy_pool.json')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-dir', default='report/figures/exp12_run')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--gate-n', type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.analyze:
        return analyze(out_dir, gate_n=args.gate_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    alias = {'both': 'menu,lever', 'named': 'flat,free'}
    arms = [a.strip() for a in alias.get(args.arm, args.arm).split(',')]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f'unknown arm(s): {bad}; valid: {ARMS}')

    starting = list(build_exp0_boards(canonical_config=args.canonical_config))
    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        starting = [(l, c) for l, c in starting if l in wanted]

    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                  n_matchups=args.n_matchups,
                                  seed=args.matchup_seed)

    executor = ProcessPoolExecutor(max_workers=args.workers)  # pre-CUDA fork
    designers = {}
    for a in arms:
        if args.backend == 'heuristic':
            designers[a] = DesignerLLM(backend='heuristic')
            continue
        d = DesignerLLM(backend=args.backend, model_name=args.model,
                        goal_disclosure='open')
        d.system_prompt = load_prompt(ARM_PROMPT[a])
        d.gen_cfg['prompt_path'] = ARM_PROMPT[a]
        designers[a] = d

    with run_log.track(experiment='EXP-012-choice', script=__file__,
                       args=vars(args), out_dir=out_dir,
                       condition=','.join(arms)) as run:
        for arm in arms:
            for board_label, cfg in starting:
                tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
                for s in range(args.n_seeds):
                    seed = args.seed_offset + s * 1000 + 42
                    out_path = out_dir / arm / f'{arm}__{tag}__seed{seed}.jsonl'
                    print(f'\n[{board_label}|{arm}] seed={seed} K={args.K}')
                    run.note(f'{board_label}|{arm}|seed={seed}: starting')
                    kw = dict(starting_cfg=cfg, board_label=board_label,
                              board_tag=tag, seed=seed, pool=pool,
                              matchups=matchups, args=args,
                              designer=designers[arm], out_path=out_path)
                    if arm == 'menu':
                        iters = run_menu_trajectory(**kw)
                    elif arm == 'lever':
                        iters = run_lever_trajectory(executor=executor, **kw)
                    else:
                        iters = run_named_trajectory(arm, executor=executor,
                                                     **kw)
                    for it in iters:
                        sh = (it.metrics or {}).get('skill_share')
                        sh_txt = f'{sh:.3f}' if isinstance(sh, float) else 'NA'
                        pad = ' [PAD]' if it.convergence_padded else ''
                        print(f'    iter {it.iter}: share={sh_txt} '
                              f'score={it.score:.4f}{pad}  '
                              f'{it.rationale[:70]}')
                    run.note(f'{board_label}|{arm}|seed={seed}: done '
                             f'final={iters[-1].score}')
    executor.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
