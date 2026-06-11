"""LLM-as-parametric-designer closed loop — TUNER experiment, round 1.

Five-condition ablation over the same K=8 trajectory:

  T-MUTE  : no diagnostic feed (control)
  T-HAZ   : hazards channel only
  T-MET   : metrics + per-group breakdown + strategy-pool exploit-resistance
  T-FULL  : everything, GOAL-OPEN designer prompt (score function disclosed)
  T-BLIND : everything, GOAL-CLOSED designer prompt (score not disclosed)

Plus three structural comparators:

  T-RAND  : uniform random parametric edits (does feedback beat random?)
  T-CANON : no design intervention (variance floor)
  T-SANITY: hardcoded "obviously good" trajectory (eval check)

The LLM conditions all share the same per-iteration eval-game seed schedule
for fixed (board, seed, iter) so cross-condition comparisons are paired,
not independently noisy. The `## PRIOR ITERATIONS` block is redacted per
condition so even one iteration of full history wouldn't collapse T-HAZ
into T-FULL via the history channel (see ANALYSIS_PLAN §11).

Iteration board: per ANALYSIS_PLAN §7 (Spearman rho=0.30 < 0.4 cutoff),
iteration board is CANONICAL.

Fixed across the run (per round-1 lock):
  - 5 starting boards from optimizer.board_sources.build_five_boards
  - 3 seeds per board => 15 trajectories per condition
  - K=8 iterations max with LLM-declared {converged} action (T-FULL/HAZ/MET only)
  - n_games=200 per iteration; bootstrap n=1500
  - Strategy pool: optimizer/strategy_pool.json (10 named + 20 sampled, seed=0)

Usage (from the repo root):
    # Heuristic smoke (no LLM); validates the loop end-to-end in a few sec.
    set PYTHONPATH=. && python scripts/llm_design_loop.py --backend heuristic \\
        --ablation-condition full --boards default --n-seeds 1 --K 3 --n-games 20

    # Production (one condition):
    set PYTHONPATH=. && python scripts/llm_design_loop.py --backend local \\
        --ablation-condition full \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --n-seeds 3 --K 8 --n-games 200 \\
        --out-dir report/figures/round1/tuner/full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from config import GameConfig
from monopoly.core.cell import Property
from optimizer.exp_boards import build_exp0_boards
from optimizer.group_design import (GroupDesign, apply_design,
                                       bootstrap_score_ci, evaluate_config)
from optimizer import run_log
from optimizer.skill_expression import render_skill_curve, fmt_checkpoint, render_goal_anchor
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool
from prompts.loader import load_prompt


CONDITIONS = ('mute', 'haz', 'met', 'full', 'blind', 'rand', 'canon', 'sanity',
              'full_noconv', 'haz_noconv', 'full_unbounded', 'haz_unbounded')
LLM_CONDITIONS = ('mute', 'haz', 'met', 'full', 'blind',
                  'full_noconv', 'haz_noconv', 'full_unbounded', 'haz_unbounded')
NO_CONVERGE_CONDITIONS = ('mute', 'blind',
                          'full_noconv', 'haz_noconv', 'full_unbounded', 'haz_unbounded')
UNBOUNDED_CONDITIONS = ('full_unbounded', 'haz_unbounded')


# --------------------------------------------------------------------------- #
# Designer-LLM wrapper                                                          #
# --------------------------------------------------------------------------- #

_RESPONSE_BLOCK = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)


@dataclass
class DesignerResponse:
    raw_text:       str
    parsed:         Optional[dict]
    design:         Optional[GroupDesign]
    rationale:      str
    converged:      bool
    parser_status:  str
    error:          Optional[str] = None
    token_count:    Optional[int] = None


def _coerce_design(intervention: dict, rationale: str) -> Optional[GroupDesign]:
    try:
        return GroupDesign(
            salary_mult       = float(intervention.get('salary_mult', 1.0)),
            drop_groups       = list(intervention.get('drop_groups', [])),
            group_cost_mult   = {str(k): float(v) for k, v in
                                  (intervention.get('group_cost_mult', {}) or {}).items()},
            group_rent_mult   = {str(k): float(v) for k, v in
                                  (intervention.get('group_rent_mult', {}) or {}).items()},
            prop_overrides    = {str(k): dict(v) for k, v in
                                  (intervention.get('prop_overrides', {}) or {}).items()},
            label             = '',
            rationale         = rationale[:500],
        )
    except (TypeError, ValueError):
        return None


def parse_designer_response(text: str) -> DesignerResponse:
    """Extract a structured DesignerResponse from raw LLM text. Tries fenced
    JSON first, falls back to a bare object."""
    candidates: List[str] = [m.group(1) for m in _RESPONSE_BLOCK.finditer(text)]
    if not candidates:
        candidates.append(text.strip())

    last_err = 'no parseable JSON in response'
    parsed = None
    for c in candidates:
        try:
            parsed = json.loads(c)
            break
        except json.JSONDecodeError as ex:
            last_err = f'JSONDecodeError: {ex.msg}'
            parsed = None
    if parsed is None or not isinstance(parsed, dict):
        return DesignerResponse(raw_text=text, parsed=None, design=None,
                                rationale='', converged=False,
                                parser_status='parser_failure', error=last_err)

    rationale = str(parsed.get('rationale', '')).strip()
    converged = bool(parsed.get('converged', False))
    intervention = parsed.get('intervention', {}) or {}
    if not isinstance(intervention, dict):
        return DesignerResponse(raw_text=text, parsed=parsed, design=None,
                                rationale=rationale, converged=converged,
                                parser_status='invalid_intervention',
                                error='intervention must be an object')

    design = _coerce_design(intervention, rationale)
    if design is None:
        return DesignerResponse(raw_text=text, parsed=parsed, design=None,
                                rationale=rationale, converged=converged,
                                parser_status='invalid_intervention',
                                error='could not coerce intervention to GroupDesign')

    return DesignerResponse(raw_text=text, parsed=parsed, design=design,
                            rationale=rationale, converged=converged,
                            parser_status='ok')


def _designer_prompt_path(goal_disclosure: str, unbounded: bool = False) -> str:
    suffix = '_unbounded' if unbounded else ''
    if goal_disclosure == 'open':
        return f'designer_llm_prompt_open{suffix}.txt'
    if goal_disclosure == 'closed':
        return f'designer_llm_prompt_closed{suffix}.txt'
    raise ValueError(f'goal_disclosure must be open|closed, got {goal_disclosure!r}')


class DesignerLLM:
    """Wraps backend dispatch (heuristic / local / openai). The heuristic
    backend cycles a tiny canned library so the loop driver can be smoke-
    tested without any model. Goal-disclosure picks which hash-locked
    designer prompt is loaded at construction time (open vs closed)."""

    _MODEL_CACHE: dict = {}

    _HEURISTIC_CYCLE: List[dict] = [
        {'rationale': 'pacing too long; raise salary',
         'intervention': {'salary_mult': 1.25}, 'converged': False},
        {'rationale': 'orange rents look weak; bump',
         'intervention': {'group_rent_mult': {'Orange': 1.25}}, 'converged': False},
        {'rationale': 'no improvement margin remaining',
         'intervention': {}, 'converged': True},
    ]

    def __init__(self, backend: str = 'local', model_name: Optional[str] = None,
                 max_new_tokens: int = 320,
                 goal_disclosure: Literal['open', 'closed'] = 'open',
                 unbounded: bool = False):
        self.backend = backend
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.goal_disclosure = goal_disclosure
        self.unbounded = unbounded
        self.prompt_path = _designer_prompt_path(goal_disclosure, unbounded=unbounded)
        self.system_prompt = load_prompt(self.prompt_path)
        self.gen_cfg = {
            'model':           (model_name or 'Qwen/Qwen2.5-1.5B-Instruct'),
            'max_new_tokens':  int(max_new_tokens),
            'do_sample':       False,
            'temperature':     0.0,
            'goal_disclosure': goal_disclosure,
            'prompt_path':     self.prompt_path,
            'engine':          ('openai-compat' if backend == 'openai'
                                  else ('heuristic' if backend == 'heuristic'
                                          else 'transformers')),
            'dtype':           os.environ.get('LLM_DTYPE', 'float16'),
            'attn_impl':       os.environ.get('LLM_ATTN_IMPL', 'default'),
            'deterministic':   True,
        }

    def query(self, user_prompt: str, iteration: int = 0) -> Tuple[str, Optional[int]]:
        if self.backend == 'heuristic':
            payload = self._HEURISTIC_CYCLE[iteration % len(self._HEURISTIC_CYCLE)]
            return '```json\n' + json.dumps(payload) + '\n```', None
        if self.backend == 'openai':
            return self._query_openai(user_prompt), None
        return self._query_local(user_prompt)

    def _get_local(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        name = self.model_name or os.environ.get('LLM_MODEL',
                                                  'Qwen/Qwen2.5-1.5B-Instruct')
        if name in self._MODEL_CACHE:
            return self._MODEL_CACHE[name]
        cache_dir = os.environ.get('LLM_CACHE_DIR', 'models/hf_cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        is_local = Path(name).is_dir()
        kw = {} if is_local else {'cache_dir': cache_dir}
        tok = AutoTokenizer.from_pretrained(name, **kw)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            dtype_name = os.environ.get('LLM_DTYPE', 'float16')
            dtype = {'float16': torch.float16,
                      'bfloat16': torch.bfloat16,
                      'float32':  torch.float32}.get(dtype_name, torch.float16)
            kw['torch_dtype'] = dtype
            attn_impl = os.environ.get('LLM_ATTN_IMPL')
            if attn_impl:
                kw['attn_implementation'] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device)
        model.eval()
        self._MODEL_CACHE[name] = (tok, model, device)
        return self._MODEL_CACHE[name]

    def _query_local(self, prompt: str) -> Tuple[str, Optional[int]]:
        import torch
        tok, model, device = self._get_local()
        msgs = [{'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        gen_tokens = out[0, inputs['input_ids'].shape[1]:]
        gen = tok.decode(gen_tokens, skip_special_tokens=True)
        return gen.strip(), int(gen_tokens.shape[0])

    def _query_openai(self, prompt: str) -> str:
        import urllib.request
        base = os.environ.get('LLM_OPENAI_BASE_URL', 'http://localhost:11434/v1')
        key  = os.environ.get('LLM_OPENAI_API_KEY', 'no-key')
        model = self.model_name or os.environ.get('LLM_OPENAI_MODEL', 'qwen2.5:1.5b')
        body = {'model': model,
                'messages': [{'role': 'system', 'content': self.system_prompt},
                              {'role': 'user',   'content': prompt}],
                'max_tokens': self.max_new_tokens, 'temperature': 0.0}
        req = urllib.request.Request(f'{base}/chat/completions',
                                       data=json.dumps(body).encode('utf-8'),
                                       headers={'Content-Type': 'application/json',
                                                 'Authorization': f'Bearer {key}'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        return payload['choices'][0]['message']['content'].strip()


# --------------------------------------------------------------------------- #
# Diagnostic-feed components                                                    #
# --------------------------------------------------------------------------- #

def _per_group_breakdown(cfg: GameConfig) -> List[dict]:
    rows: Dict[str, dict] = {}
    for c in cfg.cells:
        if not isinstance(c, Property):
            continue
        r = rows.setdefault(c.group, {'group': c.group, 'n': 0,
                                        'mean_cost': 0.0, 'mean_rent': 0.0,
                                        'cost_sum': 0, 'rent_sum': 0})
        r['n'] += 1
        r['cost_sum'] += c.cost_base
        r['rent_sum'] += c.rent_base
    for r in rows.values():
        r['mean_cost'] = r['cost_sum'] / max(r['n'], 1)
        r['mean_rent'] = r['rent_sum'] / max(r['n'], 1)
    return [rows[g] for g in rows]


def _design_diff_summary(design: GroupDesign) -> str:
    parts = []
    if design.salary_mult != 1.0:
        parts.append(f'salary x{design.salary_mult:.2f}')
    for g in design.drop_groups:
        parts.append(f'dropped:{g}')
    for g, m in design.group_cost_mult.items():
        if m != 1.0: parts.append(f'cost {g} x{m:.2f}')
    for g, m in design.group_rent_mult.items():
        if m != 1.0: parts.append(f'rent {g} x{m:.2f}')
    for p, ov in design.prop_overrides.items():
        parts.append(f'override {p}={ov}')
    return '; '.join(parts) if parts else 'no diff from default'


def _strategy_pool_exploit_block(eval_out: dict) -> str:
    """Build the STRATEGY-POOL EXPLOITATION block.

    pairwise_winrate_dispersion : std of pairwise win-rates across matchup
                                  pairs (lower = more uniform = healthier).
    most_dominant_pair          : strat_a beats strat_b at <winrate>.
    archetype_entropy           : entropy of the winner-archetype distribution
                                  across all per-game records (higher = more
                                  diverse winners).
    """
    games = eval_out.get('per_game_records', []) or []
    if not games:
        return ('  pairwise_winrate_dispersion: n/a (no games)\n'
                '  most_dominant_pair: n/a\n'
                '  archetype_entropy: n/a')

    # Aggregate per-pair (winner_strat, loser_strat) wins.
    pair_counts: Dict[Tuple[str, str], List[int]] = {}
    winner_counts: Dict[str, int] = {}
    total = 0
    for g in games:
        # run_matchup writes per-game outputs as dicts with at least
        # 'winner_strategy' / 'players_strategies' fields. Be tolerant of
        # absent fields so this block never crashes the feed.
        ws = g.get('winner_strategy') or g.get('winner') or None
        strats = g.get('players_strategies') or g.get('strategies') or []
        if not ws or not strats:
            continue
        total += 1
        winner_counts[ws] = winner_counts.get(ws, 0) + 1
        for s in strats:
            if s == ws:
                continue
            key = (ws, s)
            slot = pair_counts.setdefault(key, [0, 0])
            slot[0] += 1                                              # ws beats s
            rev = pair_counts.setdefault((s, ws), [0, 0])
            rev[1] += 1                                                # s loses to ws

    if total == 0:
        return ('  pairwise_winrate_dispersion: n/a (no winner labels)\n'
                '  most_dominant_pair: n/a\n'
                '  archetype_entropy: n/a')

    rates = []
    most_dominant = ('', '', 0.0, 0)
    for (a, b), (wins, _losses) in pair_counts.items():
        # Sym-pair total appearances = wins + (other side's losses to this side)
        rev = pair_counts.get((b, a), [0, 0])
        n_pair = wins + rev[0]
        if n_pair < 4:        # too few games to read as a "pair winrate"
            continue
        wr = wins / n_pair
        rates.append(wr)
        if wr > most_dominant[2]:
            most_dominant = (a, b, wr, n_pair)

    if rates:
        dispersion = float(np.std(rates))
    else:
        dispersion = 0.0

    if winner_counts:
        ps = np.array(list(winner_counts.values()), dtype=float)
        ps = ps / ps.sum()
        # Shannon entropy in nats; convert to bits for readability.
        entropy = float(-np.sum(ps * np.log2(np.clip(ps, 1e-12, 1.0))))
    else:
        entropy = 0.0

    if most_dominant[2] > 0:
        a, b, wr, n_pair = most_dominant
        dom = f'{a} beats {b} at {wr*100:.1f}% (n={n_pair})'
    else:
        dom = 'no pair with sufficient sample size'

    return ('\n'.join([
        f'  pairwise_winrate_dispersion: {dispersion:.3f}    (lower = more uniform)',
        f'  most_dominant_pair: {dom}',
        f'  archetype_entropy: {entropy:.3f} bits          (higher = more diverse winners)',
    ]))


def _prior_iterations_block(prior_iters: List[dict], condition: str) -> str:
    """Format the last <=3 prior iterations in a per-condition redacted way.

    Per ANALYSIS_PLAN §11: even one iteration of full history would leak the
    hidden information channel and collapse T-HAZ / T-MET into T-FULL by
    iteration 3-4 of K=8. Redaction keeps the ablation single-variable.
    """
    if not prior_iters:
        return ''
    lines = []
    for it in prior_iters[-3:]:
        i = it.get('iter', '?')
        diff = it.get('design_diff', '') or 'no diff'
        rat  = (it.get('rationale') or '').strip()
        bits = [f'  iter {i}: diff={diff}  rationale={rat[:80]!r}']
        if condition in ('haz', 'full', 'blind',
                        'full_noconv', 'full_unbounded', 'haz_noconv', 'haz_unbounded'):
            hz = it.get('hazard_summary')
            if hz: bits.append(f'         hazards: {hz[:120]}')
        if condition in ('met', 'full', 'blind', 'full_noconv', 'full_unbounded'):
            sc = it.get('score'); m = it.get('metrics') or {}
            sk = m.get('skill_share'); dirn = m.get('direction')
            ex = it.get('exploit_summary')
            if sc is not None:
                sk_txt = f'{sk:.3f}' if isinstance(sk, (int, float)) else 'n/a'
                bits.append(f'         score={sc:.4f}  skill_share={sk_txt}  '
                            f'direction={dirn}')
            if ex: bits.append(f'         exploit: {ex[:120]}')
        lines.append('\n'.join(bits))
    return '\n'.join(lines)


def build_diagnostic_feed(cfg: GameConfig, design: GroupDesign,
                           eval_out: dict, prior_eval: Optional[dict],
                           condition: str,
                           prior_iters: Optional[List[dict]] = None,
                           hazard_summary: Optional[str] = None) -> str:
    """Assemble the user-prompt the designer LLM sees per iteration.

    Channels are gated per condition; in summary:

      - CURRENT EVAL          : met, full, blind
      - DESIGN DIFF           : all conditions (own action history)
      - PER-GROUP BREAKDOWN   : met, full, blind
      - STRATEGY-POOL EXPLOIT : met, full, blind
      - HAZARD SUMMARY        : haz, full, blind
      - PRIOR ITERATIONS      : all conditions, but redacted per condition
      - YOUR JOB              : all conditions; T-MUTE / T-BLIND get
                                "do not set converged=true"
    """
    if condition not in CONDITIONS:
        raise ValueError(f'condition must be one of {CONDITIONS}, got {condition!r}')

    show_metrics = condition in ('met', 'full', 'blind', 'full_noconv', 'full_unbounded')
    show_hazards = condition in ('haz', 'full', 'blind', 'full_noconv', 'full_unbounded',
                                 'haz_noconv', 'haz_unbounded')
    show_exploit = condition in ('met', 'full', 'blind', 'full_noconv', 'full_unbounded')
    show_groups  = condition in ('met', 'full', 'blind', 'full_noconv', 'full_unbounded')
    # The shared goal anchor goes to the OPEN informative conditions, byte-identical.
    # NOT mute (no-feedback floor) and NOT blind (goal-closed: would leak the target).
    show_anchor = condition in ('haz', 'met', 'full',
                                'full_noconv', 'full_unbounded',
                                'haz_noconv', 'haz_unbounded')

    parts: List[str] = []

    if show_anchor:
        m = eval_out.get('metrics', {}) or {}
        a_share = m.get('skill_share', eval_out.get('skill_share', float('nan')))
        a_gstar = m.get('g_star', 0.60)
        a_dir = m.get('direction', '')
        parts.append('## GOAL')
        parts.append('  ' + render_goal_anchor(a_share, a_gstar, a_dir))

    from optimizer.group_design import board_groups
    parts.append('## VALID GROUPS (use ONLY these exact names in your intervention)')
    parts.append('  ' + ', '.join(board_groups(cfg)))

    if show_metrics:
        m = eval_out.get('metrics', {}) or {}
        share = m.get('skill_share', eval_out.get('skill_share', float('nan')))
        g_star = m.get('g_star', 0.60)
        direction = (m.get('direction', '') or '').upper()
        parts.append('## CURRENT EVAL')
        share_txt = f'{share:.3f}' if isinstance(share, (int, float)) and math.isfinite(share) else 'n/a'
        parts.append(f'  skill_share (endpoint): {share_txt}')
        parts.append(f'  score: {eval_out.get("score", float("nan")):.4f}  '
                     f'(lower is better; 0 inside the dead-zone)')
        curve = eval_out.get('skill_curve')
        if curve and curve.get('checkpoints'):
            seq = '  '.join(fmt_checkpoint(c) for c in curve['checkpoints'])
            parts.append(f'  per-round checkpoints: {seq}')
        if prior_eval is not None and prior_eval.get('score') is not None:
            delta = eval_out['score'] - prior_eval['score']
            parts.append(f'  delta_score_vs_prev: {delta:+.4f}')

    parts.append('\n## CURRENT DESIGN DIFF FROM DEFAULT')
    parts.append('  ' + _design_diff_summary(design))

    if show_groups:
        bd = _per_group_breakdown(cfg)
        bd_lines = [f'  - {r["group"]:>10}: n={r["n"]} '
                    f'mean_cost=${r["mean_cost"]:.0f} '
                    f'mean_rent=${r["mean_rent"]:.0f}'
                    for r in bd]
        parts.append('\n## PER-GROUP COST/RENT BREAKDOWN')
        parts.extend(bd_lines)

    if show_exploit:
        parts.append('\n## STRATEGY-POOL EXPLOITATION')
        parts.append(_strategy_pool_exploit_block(eval_out))

    if show_hazards:
        curve = eval_out.get('skill_curve')
        if curve and curve.get('checkpoints'):
            parts.append('\n## SKILL-SHARE TRAJECTORY (this board)')
            parts.append(render_skill_curve(curve))
        elif hazard_summary:
            parts.append('\n## SKILL-SHARE TRAJECTORY (prior runs)')
            parts.append(hazard_summary)

    pi = _prior_iterations_block(prior_iters or [], condition)
    if pi:
        parts.append('\n## PRIOR ITERATIONS (redacted per condition)')
        parts.append(pi)

    parts.append('\n## YOUR JOB')
    if condition in NO_CONVERGE_CONDITIONS:
        parts.append('  Propose ONE small intervention. Reply only with the '
                     'JSON schema described in the system prompt. Do NOT set '
                     'converged=true (this run does not honor convergence in '
                     'this condition).')
    else:
        parts.append('  Propose ONE small intervention. Reply only with the '
                     'JSON schema described in the system prompt.')
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Cross-condition seed alignment                                                #
# --------------------------------------------------------------------------- #

def eval_seed_for(board: str, seed: int, iter_idx: int, game_idx: int) -> int:
    """Deterministic per-(board, seed, iter, game) eval seed.

    All 5 LLM conditions hit the same seed for fixed (board, seed, iter, game)
    so cross-condition score deltas are paired, not independently sampled.
    The hash-of-tuple choice lets us derive seeds without storing schedules
    on disk; reproducibility is purely a function of these four arguments.
    """
    s = f'{board}|{seed}|{iter_idx}|{game_idx}'.encode('utf-8')
    return int(hashlib.blake2s(s, digest_size=4).hexdigest(), 16) & 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# Trajectory runner                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Iteration:
    iter:                 int
    design:               dict
    design_diff:          str
    rationale:            str
    parser_status:        str
    parser_error:         Optional[str]
    converged_request:    bool
    convergence_padded:   bool                  # iter is a no-op pad after early {converged}
    convergence_violation: bool                 # T-MUTE/T-BLIND set converged=true (ignored)
    parser_retry:         bool                  # parser_failure on attempt #1, retry succeeded
    score:                Optional[float]
    metrics:              Optional[dict]
    score_ci:             Optional[dict]
    delta_vs_prev:        Optional[float]
    improvement:          Optional[bool]
    n_games:              int
    token_count:          Optional[int]
    wall_seconds:         float
    raw_response:         str
    condition:            str
    gen_cfg:              dict
    eval_seed_base:       int
    skill_curve:          Optional[dict] = None
    invalid_groups:       list = field(default_factory=list)


def _is_improvement(prev_score: float, cur_score: float, ci: dict,
                    rel_threshold: float = 0.03) -> bool:
    if prev_score <= 0:
        return False
    rel = (prev_score - cur_score) / prev_score
    return bool(rel >= rel_threshold and ci.get('ci_hi', cur_score) < prev_score)


def _merge_designs(prev: GroupDesign, new: GroupDesign) -> GroupDesign:
    return GroupDesign(
        salary_mult=(new.salary_mult if new.salary_mult != 1.0 else prev.salary_mult),
        drop_groups=list(set(prev.drop_groups) | set(new.drop_groups)),
        group_cost_mult={**prev.group_cost_mult, **new.group_cost_mult},
        group_rent_mult={**prev.group_rent_mult, **new.group_rent_mult},
        prop_overrides={**prev.prop_overrides, **new.prop_overrides},
        label='cumulative',
        rationale=new.rationale or prev.rationale,
    )


def _eval_with_aligned_seeds(cfg, pool, matchups, n_games, board, seed,
                              iter_idx, max_turns):
    """Wrap evaluate_config with cross-condition aligned seeds. The base seed
    used for each matchup is derived from (board, seed, iter, matchup_idx) so
    every condition that calls this with the same arguments gets exactly the
    same per-game schedule."""
    return evaluate_config(
        cfg, pool, matchups, n_games,
        base_seed=eval_seed_for(board, seed, iter_idx, 0),
        max_turns=max_turns,
        record_trajectory=True,
    )


# --------------------------------------------------------------------------- #
# T-RAND comparator                                                             #
# --------------------------------------------------------------------------- #

_RAND_FIELDS = ('salary_mult', 'group_cost_mult', 'group_rent_mult', 'drop_groups')


def sample_random_intervention(present_groups: List[str],
                                rng: np.random.Generator) -> GroupDesign:
    """Uniformly pick one of {salary_mult, group_cost_mult, group_rent_mult,
    drop_groups} and a uniform value in declared range.

    Bounds:
      salary_mult       : [0.5, 2.0]
      group_cost_mult   : [0.5, 2.0]  on a uniformly chosen present group
      group_rent_mult   : [0.5, 2.0]  on a uniformly chosen present group
      drop_groups       : a single uniformly chosen present group
    """
    field_choice = _RAND_FIELDS[rng.integers(0, len(_RAND_FIELDS))]
    if field_choice == 'salary_mult':
        return GroupDesign(salary_mult=float(rng.uniform(0.5, 2.0)),
                            label='rand', rationale='[T-RAND]')
    g = present_groups[rng.integers(0, len(present_groups))] if present_groups else 'Brown'
    val = float(rng.uniform(0.5, 2.0))
    if field_choice == 'group_cost_mult':
        return GroupDesign(group_cost_mult={g: val},
                            label='rand', rationale='[T-RAND]')
    if field_choice == 'group_rent_mult':
        return GroupDesign(group_rent_mult={g: val},
                            label='rand', rationale='[T-RAND]')
    return GroupDesign(drop_groups=[g], label='rand', rationale='[T-RAND]')


# --------------------------------------------------------------------------- #
# T-SANITY trajectory                                                           #
# --------------------------------------------------------------------------- #

# T-SANITY (skill objective): progressive UNIFORM rent cuts drive the L3-vs-L0
# share DOWN from the default (~0.69-0.73 depending on seed) toward g*=0.60,
# landing IN-BAND by design 3. Rent reduction is the empirically-verified
# skill-reducing lever (raising salary/rent INCREASES the skill edge, so the old
# balance-era trajectory moved the wrong way). The canary asserts share is
# non-increasing within tolerance across the out-of-band approach and that the
# final design lands in-band (skill_score == 0) -- see assert_sanity_share_monotone.
# Run at elevated n (>=400): the share floor sits ~0.62, so steps near the band
# are ~1 SE apart at n=400 and the tolerance, not raw strictness, carries them.
_SANITY_RENT_GROUPS = ['Brown', 'Lightblue', 'Pink', 'Orange',
                       'Red', 'Yellow', 'Green', 'Indigo']


def _sanity_rent_design(mult: float, label: str, note: str) -> GroupDesign:
    return GroupDesign(group_rent_mult={g: mult for g in _SANITY_RENT_GROUPS},
                       label=label, rationale=note)


SANITY_TRAJECTORY: List[GroupDesign] = [
    _sanity_rent_design(0.80, 'sanity-0', '[T-SANITY] all rents x0.80'),
    _sanity_rent_design(0.60, 'sanity-1', '[T-SANITY] all rents x0.60'),
    _sanity_rent_design(0.45, 'sanity-2', '[T-SANITY] all rents x0.45'),
    _sanity_rent_design(0.35, 'sanity-3', '[T-SANITY] all rents x0.35 (lands in-band)'),
    _sanity_rent_design(0.35, 'sanity-4', '[T-SANITY] hold in-band'),
    _sanity_rent_design(0.35, 'sanity-5', '[T-SANITY] hold in-band'),
    _sanity_rent_design(0.35, 'sanity-6', '[T-SANITY] hold in-band'),
    _sanity_rent_design(0.35, 'sanity-7', '[T-SANITY] hold in-band'),
]

# T-SANITY pins every iter's eval seed to iter=0 so the only thing varying
# across sanity iters is the design — not the per-game schedule. (Other
# conditions hash iter into the seed for cross-condition alignment; sanity
# isn't a TUNER condition and doesn't need that alignment, but DOES need
# apples-to-apples comparison across its own iters.)
SANITY_PINNED_ITER = 0


def assert_sanity_share_monotone(shares, tol: float = 0.03, min_drop: float = 0.04,
                                 g_star: float = 0.60, band: float = 0.03):
    """Skill-objective T-SANITY canary, asserted on SHARE (not skill_score, which
    floors at 0 inside the band and would swallow legitimate ties). Contract:
      1. across the out-of-band approach (baseline .. first in-band step) the
         share is non-increasing within `tol`;
      2. that approach drops at least `min_drop` from baseline (a real signal,
         not noise);
      3. the trajectory LANDS IN-BAND (|share - g*| <= band, i.e. score 0) --
         this explicitly canaries the score floor.
    `shares` = per-iter skill_share, iter 0 = baseline. Run at elevated n (>=400).
    Returns (ok, reason)."""
    if len(shares) < 5 or any(s is None for s in shares):
        return False, f'incomplete trajectory (need >=5 non-None shares): {shares}'
    def _inband(s):
        return abs(s - g_star) <= band
    landing = next((i for i in range(1, len(shares)) if _inband(shares[i])), None)
    if landing is None:
        return False, f'never reached the band: {shares}'
    approach = shares[:landing + 1]                 # baseline .. first in-band
    for i in range(len(approach) - 1):
        if approach[i + 1] > approach[i] + tol:
            return False, (f'share rose at approach step {i}->{i+1}: '
                           f'{approach[i]:.3f} -> {approach[i+1]:.3f} (tol {tol})')
    drop = shares[0] - approach[-1]
    if drop < min_drop:
        return False, f'approach drop {drop:.3f} < min_drop {min_drop} (no real signal)'
    after = shares[landing + 1:]
    if any(not _inband(s) for s in after):
        bad = next(i for i, s in enumerate(after) if not _inband(s))
        return False, (f'share left the band after landing at step '
                       f'{landing + 1 + bad}: {after[bad]:.3f}')
    return True, f'ok (landed in-band at step {landing}, approach drop {drop:.3f})'


# --------------------------------------------------------------------------- #
# Main per-condition trajectory runner                                          #
# --------------------------------------------------------------------------- #

def _present_groups(cfg) -> List[str]:
    seen: List[str] = []
    for c in cfg.cells:
        if isinstance(c, Property) and c.group not in seen:
            seen.append(c.group)
    return seen


def _design_to_iter_dict(it: 'Iteration') -> dict:
    """Slimmed-down view passed to the next iteration's prior_iters block."""
    return {
        'iter':            it.iter,
        'design_diff':     it.design_diff,
        'rationale':       it.rationale,
        'score':           it.score,
        'metrics':         it.metrics,
        # 'hazard_summary' / 'exploit_summary' are populated by the loop after
        # we know what hazard/exploit text the LLM saw this round.
    }


_RETRY_REMINDER = (
    "Your previous response did not parse. Please respond with exactly one "
    "JSON object inside a fenced ```json``` block matching the schema."
)


def run_trajectory(starting_cfg: GameConfig, board_label: str, seed: int,
                   pool, matchups, n_games: int, K: int,
                   condition: str,
                   designer: Optional[DesignerLLM],
                   max_turns: int,
                   out_path: Path,
                   hazard_summary: Optional[str] = None,
                   ) -> List[Iteration]:
    """Run one (board, seed) trajectory for one ablation condition. Writes
    one JSON line per iteration to `out_path`."""
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition {condition!r}')

    cfg = starting_cfg
    cumulative_design = GroupDesign(label='cumulative')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    iterations: List[Iteration] = []
    prev_score: Optional[float] = None
    rand_rng = np.random.default_rng(seed)

    gen_cfg = (designer.gen_cfg
               if designer is not None
               else {'condition': condition, 'note': 'no LLM call this condition'})
    base_eval_seed = eval_seed_for(board_label, seed, 0, 0)

    # Iteration 0: baseline eval (no LLM call).
    t0 = time.perf_counter()
    base_eval = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                           board_label, seed, 0, max_turns)
    base_ci = bootstrap_score_ci(base_eval['shares'],
                                   n_resamples=1500, seed=seed)
    rec0 = Iteration(
        iter=0, design=cumulative_design.to_dict(),
        design_diff=_design_diff_summary(cumulative_design),
        rationale='[baseline]', parser_status='baseline', parser_error=None,
        converged_request=False, convergence_padded=False,
        convergence_violation=False, parser_retry=False,
        score=base_eval['score'], metrics=base_eval['metrics'],
        score_ci=base_ci, delta_vs_prev=None, improvement=None,
        n_games=base_eval['n_games_total'], token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='',
        condition=condition, gen_cfg=gen_cfg, eval_seed_base=base_eval_seed,
        skill_curve=base_eval.get('skill_curve'))
    iterations.append(rec0)
    fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()
    prev_score = base_eval['score']

    converged_done = False        # True once we honor a {converged} request
    pad_design = cumulative_design

    for k in range(1, K + 1):
        t0 = time.perf_counter()
        ev_seed_base = eval_seed_for(board_label, seed, k, 0)
        prior_iters = [_design_to_iter_dict(it) for it in iterations]

        # ------------------------------------------------------------------ #
        # T-CANON: no design intervention; just re-eval the default config.
        # ------------------------------------------------------------------ #
        if condition == 'canon':
            ev = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                            board_label, seed, k, max_turns)
            ci = bootstrap_score_ci(ev['shares'],
                                      n_resamples=1500, seed=seed + k)
            delta = ev['score'] - prev_score
            improvement = _is_improvement(prev_score, ev['score'], ci)
            rec = Iteration(
                iter=k, design=cumulative_design.to_dict(),
                design_diff=_design_diff_summary(cumulative_design),
                rationale='[T-CANON: no intervention]', parser_status='ok',
                parser_error=None, converged_request=False,
                convergence_padded=False, convergence_violation=False,
                parser_retry=False,
                score=ev['score'], metrics=ev['metrics'], score_ci=ci,
                delta_vs_prev=delta, improvement=improvement,
                n_games=ev['n_games_total'], token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=condition, gen_cfg=gen_cfg,
                eval_seed_base=ev_seed_base,
                skill_curve=ev.get('skill_curve'))
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            prev_score = ev['score']
            continue

        # ------------------------------------------------------------------ #
        # T-RAND: uniform random parametric edit; no diagnostic feed.
        # ------------------------------------------------------------------ #
        if condition == 'rand':
            new = sample_random_intervention(_present_groups(cfg), rand_rng)
            cumulative_design = _merge_designs(cumulative_design, new)
            try:
                cfg = apply_design(starting_cfg, cumulative_design,
                                    strict_groups=False)
            except Exception as ex:
                rec = Iteration(
                    iter=k, design=cumulative_design.to_dict(),
                    design_diff=_design_diff_summary(cumulative_design),
                    rationale=new.rationale,
                    parser_status='apply_failure',
                    parser_error=f'{type(ex).__name__}: {ex}',
                    converged_request=False, convergence_padded=False,
                    convergence_violation=False, parser_retry=False,
                    score=None, metrics=None, score_ci=None,
                    delta_vs_prev=None, improvement=None,
                    n_games=0, token_count=None,
                    wall_seconds=time.perf_counter() - t0, raw_response='',
                    condition=condition, gen_cfg=gen_cfg,
                    eval_seed_base=ev_seed_base)
                iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
                fh.flush()
                continue
            ev = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                            board_label, seed, k, max_turns)
            ci = bootstrap_score_ci(ev['shares'],
                                      n_resamples=1500, seed=seed + k)
            delta = ev['score'] - prev_score
            improvement = _is_improvement(prev_score, ev['score'], ci)
            rec = Iteration(
                iter=k, design=cumulative_design.to_dict(),
                design_diff=_design_diff_summary(cumulative_design),
                rationale=new.rationale, parser_status='ok',
                parser_error=None, converged_request=False,
                convergence_padded=False, convergence_violation=False,
                parser_retry=False,
                score=ev['score'], metrics=ev['metrics'], score_ci=ci,
                delta_vs_prev=delta, improvement=improvement,
                n_games=ev['n_games_total'], token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=condition, gen_cfg=gen_cfg,
                eval_seed_base=ev_seed_base,
                skill_curve=ev.get('skill_curve'))
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            prev_score = ev['score']
            continue

        # ------------------------------------------------------------------ #
        # T-SANITY: hardcoded schedule.
        # ------------------------------------------------------------------ #
        if condition == 'sanity':
            schedule = SANITY_TRAJECTORY
            new = schedule[(k - 1) % len(schedule)]
            cumulative_design = GroupDesign(
                salary_mult=new.salary_mult,
                drop_groups=list(new.drop_groups),
                group_cost_mult=dict(new.group_cost_mult),
                group_rent_mult=dict(new.group_rent_mult),
                prop_overrides=dict(new.prop_overrides),
                label='cumulative', rationale=new.rationale,
            )
            try:
                cfg = apply_design(starting_cfg, cumulative_design,
                                    strict_groups=False)
            except Exception as ex:
                rec = Iteration(
                    iter=k, design=cumulative_design.to_dict(),
                    design_diff=_design_diff_summary(cumulative_design),
                    rationale=new.rationale,
                    parser_status='apply_failure',
                    parser_error=f'{type(ex).__name__}: {ex}',
                    converged_request=False, convergence_padded=False,
                    convergence_violation=False, parser_retry=False,
                    score=None, metrics=None, score_ci=None,
                    delta_vs_prev=None, improvement=None,
                    n_games=0, token_count=None,
                    wall_seconds=time.perf_counter() - t0, raw_response='',
                    condition=condition, gen_cfg=gen_cfg,
                    eval_seed_base=ev_seed_base)
                iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
                fh.flush()
                continue
            # Sanity pins every iter to one eval seed so the only thing
            # varying across iters is the design, not the per-game schedule.
            ev = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                            board_label, seed, SANITY_PINNED_ITER,
                                            max_turns)
            ci = bootstrap_score_ci(ev['shares'],
                                      n_resamples=1500, seed=seed + k)
            delta = ev['score'] - prev_score
            rec = Iteration(
                iter=k, design=cumulative_design.to_dict(),
                design_diff=_design_diff_summary(cumulative_design),
                rationale=new.rationale, parser_status='ok',
                parser_error=None, converged_request=False,
                convergence_padded=False, convergence_violation=False,
                parser_retry=False,
                score=ev['score'], metrics=ev['metrics'], score_ci=ci,
                delta_vs_prev=delta, improvement=None,
                n_games=ev['n_games_total'], token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=condition, gen_cfg=gen_cfg,
                eval_seed_base=eval_seed_for(board_label, seed,
                                              SANITY_PINNED_ITER, 0),
                skill_curve=ev.get('skill_curve'))
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            prev_score = ev['score']
            continue

        # ------------------------------------------------------------------ #
        # LLM conditions (mute, haz, met, full, blind).
        # ------------------------------------------------------------------ #
        # If we already honored convergence, pad remaining iters with
        # carry-forward design (no LLM call, no eval re-cost), recording
        # convergence_padded=true.
        if converged_done:
            rec = Iteration(
                iter=k, design=pad_design.to_dict(),
                design_diff=_design_diff_summary(pad_design),
                rationale='[converged: carry-forward]', parser_status='ok',
                parser_error=None, converged_request=True,
                convergence_padded=True, convergence_violation=False,
                parser_retry=False,
                score=iterations[-1].score, metrics=iterations[-1].metrics,
                score_ci=iterations[-1].score_ci,
                delta_vs_prev=0.0, improvement=False,
                n_games=0, token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=condition, gen_cfg=gen_cfg,
                eval_seed_base=ev_seed_base)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            continue

        feed = build_diagnostic_feed(
            cfg, cumulative_design,
            eval_out=asdict(iterations[-1]) if iterations else {},
            prior_eval=(asdict(iterations[-2]) if len(iterations) >= 2 else None),
            condition=condition, prior_iters=prior_iters,
            hazard_summary=hazard_summary)

        # Attempt #1
        retry_used = False
        try:
            raw, ntok = designer.query(feed, iteration=k - 1)
            resp = parse_designer_response(raw)
            resp.token_count = ntok
        except Exception as ex:
            raw, ntok = '', None
            resp = DesignerResponse(raw_text='', parsed=None, design=None,
                                     rationale='', converged=False,
                                     parser_status='backend_failure',
                                     error=f'{type(ex).__name__}: {ex}')

        # 1-retry on parser_failure with format reminder.
        if resp.parser_status == 'parser_failure':
            retry_used = True
            try:
                raw2, ntok2 = designer.query(feed + '\n\n' + _RETRY_REMINDER,
                                              iteration=k - 1)
                resp2 = parse_designer_response(raw2)
                resp2.token_count = ntok2
                if resp2.parser_status == 'ok':
                    resp = resp2
                    raw = raw2
            except Exception:
                pass

        # If still not parseable (or invalid), carry forward prior design.
        if resp.parser_status != 'ok' or resp.design is None:
            rec = Iteration(
                iter=k, design=cumulative_design.to_dict(),
                design_diff=_design_diff_summary(cumulative_design),
                rationale=resp.rationale or '[parser_failure carry-forward]',
                parser_status='parser_failure',
                parser_error=resp.error,
                converged_request=resp.converged,
                convergence_padded=False, convergence_violation=False,
                parser_retry=retry_used,
                score=iterations[-1].score, metrics=iterations[-1].metrics,
                score_ci=iterations[-1].score_ci,
                delta_vs_prev=0.0, improvement=False,
                n_games=0, token_count=resp.token_count,
                wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text,
                condition=condition, gen_cfg=gen_cfg,
                eval_seed_base=ev_seed_base)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            continue

        # Convergence handling: condition-aware.
        violation = False
        honored = False
        if resp.converged:
            if condition in NO_CONVERGE_CONDITIONS:
                violation = True            # ignore the request; force K iters
            else:
                honored = True

        # Apply intervention.
        cumulative_design = _merge_designs(cumulative_design, resp.design)
        try:
            new_cfg = apply_design(starting_cfg, cumulative_design,
                                    strict_groups=False)
        except Exception as ex:
            rec = Iteration(
                iter=k, design=cumulative_design.to_dict(),
                design_diff=_design_diff_summary(cumulative_design),
                rationale=resp.rationale,
                parser_status='apply_failure',
                parser_error=f'{type(ex).__name__}: {ex}',
                converged_request=resp.converged,
                convergence_padded=False,
                convergence_violation=violation,
                parser_retry=retry_used,
                score=None, metrics=None, score_ci=None,
                delta_vs_prev=None, improvement=None,
                n_games=0, token_count=resp.token_count,
                wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text,
                condition=condition, gen_cfg=gen_cfg,
                eval_seed_base=ev_seed_base)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            continue
        cfg = new_cfg

        from optimizer.group_design import board_groups, referenced_groups
        invalid = sorted(referenced_groups(resp.design) - set(board_groups(cfg)))

        ev = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                        board_label, seed, k, max_turns)
        ci = bootstrap_score_ci(ev['shares'],
                                  n_resamples=1500, seed=seed + k)
        delta = ev['score'] - prev_score
        improvement = _is_improvement(prev_score, ev['score'], ci)

        rec = Iteration(
            iter=k, design=cumulative_design.to_dict(),
            design_diff=_design_diff_summary(cumulative_design),
            rationale=resp.rationale, parser_status='ok',
            parser_error=None, converged_request=resp.converged,
            convergence_padded=False, convergence_violation=violation,
            parser_retry=retry_used,
            score=ev['score'], metrics=ev['metrics'], score_ci=ci,
            delta_vs_prev=delta, improvement=improvement,
            n_games=ev['n_games_total'], token_count=resp.token_count,
            wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text,
            condition=condition, gen_cfg=gen_cfg,
            eval_seed_base=ev_seed_base,
            skill_curve=ev.get('skill_curve'),
            invalid_groups=invalid)
        iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
        fh.flush()
        prev_score = ev['score']

        if honored:
            converged_done = True
            pad_design = cumulative_design

    fh.close()
    return iterations


# --------------------------------------------------------------------------- #
# Plot driver                                                                   #
# --------------------------------------------------------------------------- #

def plot_trajectories(trajectories_by_board: Dict[str, List[List[dict]]],
                      out_path: Path) -> None:
    import matplotlib.pyplot as plt
    boards = list(trajectories_by_board.keys())
    n = len(boards)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, boards):
        seeds = trajectories_by_board[b]
        max_iters = max((len(s) for s in seeds), default=0)
        if max_iters == 0:
            continue
        iters_axis = np.arange(max_iters)
        for s_idx, traj in enumerate(seeds):
            ys = [it['score'] if it['score'] is not None else np.nan for it in traj]
            ys = ys + [np.nan] * (max_iters - len(ys))
            ax.plot(iters_axis, ys, label=f'seed{s_idx}', linewidth=1.4, alpha=0.85)
        mat = np.full((len(seeds), max_iters), np.nan, dtype=float)
        for i, traj in enumerate(seeds):
            for j, it in enumerate(traj):
                if it['score'] is not None:
                    mat[i, j] = it['score']
        with np.errstate(all='ignore'):
            mean = np.nanmean(mat, axis=0); std = np.nanstd(mat, axis=0)
            ax.fill_between(iters_axis, mean - std, mean + std,
                             alpha=0.15, color='#444444')
        ax.set_title(b, fontsize=10); ax.set_xlabel('iteration')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(fontsize=7, loc='best', framealpha=0.85)
    axes[0].set_ylabel('score (lower is better)')
    fig.suptitle('LLM parametric closed-loop trajectories', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def write_rationales_table(trajectories_by_board: Dict[str, List[List[dict]]],
                            out_path: Path, k: int = 4) -> None:
    lines = ['| board | seed | iter | score_delta | rationale |',
             '|-------|------|------|-------------|-----------|']
    for board, seeds in trajectories_by_board.items():
        rows = 0
        for s_idx, traj in enumerate(seeds):
            for it in traj:
                if rows >= k: break
                if it.get('parser_status') != 'ok': continue
                d = it.get('delta_vs_prev')
                d_str = f'{d:+.3f}' if d is not None else '-'
                rat = (it.get('rationale') or '').replace('|', '/').strip()[:120]
                lines.append(f'| {board} | {s_idx} | {it["iter"]} | {d_str} | {rat} |')
                rows += 1
            if rows >= k: break
    out_path.write_text('\n'.join(lines))


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def _run_design_loop(args, out_dir: Path, run) -> int:
    """Body of main() — split out so main() can wrap it with run_log.track."""
    cond = args.ablation_condition

    print(f'TUNER condition: {cond}')
    print('Resolving EXP-002 starts (canonical anchor + validated transforms)...')
    starting_cfgs: List[Tuple[str, GameConfig]] = list(
        build_exp0_boards(canonical_config=args.canonical_config))

    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        starting_cfgs = [(l, c) for l, c in starting_cfgs if l in wanted]
    if cond == 'sanity':
        # T-SANITY runs on default board only (per spec).
        starting_cfgs = [(l, c) for l, c in starting_cfgs if l == 'default']
        if not starting_cfgs:
            raise SystemExit('T-SANITY requires the default board to be in the source set.')
    print(f'  starting boards: {[l for l, _ in starting_cfgs]}')

    print('Loading strategy pool + matchups...')
    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                    n_matchups=args.n_matchups,
                                    seed=args.matchup_seed)

    designer: Optional[DesignerLLM] = None
    if cond in LLM_CONDITIONS:
        goal = 'closed' if cond == 'blind' else 'open'
        unbounded = cond in UNBOUNDED_CONDITIONS
        designer = DesignerLLM(backend=args.backend, model_name=args.model,
                                goal_disclosure=goal, unbounded=unbounded)
        print(f'  designer prompt: {designer.prompt_path} '
              f'(sha-locked, goal_disclosure={goal}, unbounded={unbounded})')

    hazard_summary = (Path(args.hazard_summary_path).read_text()
                       if args.hazard_summary_path else None)

    trajectories_by_board: Dict[str, List[List[dict]]] = {}
    for board_label, cfg in starting_cfgs:
        trajectories_by_board[board_label] = []
        n_seeds_eff = 1 if cond == 'sanity' else args.n_seeds
        for s in range(n_seeds_eff):
            seed = args.seed_offset + s * 1000 + 42
            print(f'\n[{board_label}|cond={cond}] seed={seed} '
                  f'(s_idx={s})  K_max={args.K}')
            run.note(f'{board_label}|cond={cond}|seed={seed}: starting')
            tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
            out_path = out_dir / f'{cond}__{tag}__seed{seed}.jsonl'
            iters = run_trajectory(
                starting_cfg=cfg, board_label=board_label, seed=seed,
                pool=pool, matchups=matchups, n_games=args.n_games, K=args.K,
                condition=cond, designer=designer, max_turns=args.max_turns,
                out_path=out_path, hazard_summary=hazard_summary)
            for it in iters:
                ps = it.parser_status; sc = it.score
                cv = ' (converged)' if it.converged_request else ''
                pad = ' [PAD]' if it.convergence_padded else ''
                vio = ' [violation]' if it.convergence_violation else ''
                if sc is not None:
                    print(f'    iter {it.iter}: parser={ps}  score={sc:.4f}{cv}{pad}{vio}')
                else:
                    print(f'    iter {it.iter}: parser={ps}  score=N/A{cv}{pad}{vio}')
            trajectories_by_board[board_label].append([asdict(i) for i in iters])
            final_score = iters[-1].score if iters and iters[-1].score is not None else None
            run.note(f'{board_label}|cond={cond}|seed={seed}: done '
                     f'final_score={final_score}')

        # T-SANITY: assert share monotone approach + in-band landing.
        if cond == 'sanity':
            shares = [it.metrics.get('skill_share') if it.metrics else None
                      for it in iters]
            ok, reason = assert_sanity_share_monotone(shares)
            sanity_path = out_dir / 'SANITY_RESULT.txt'
            sanity_path.write_text(
                f'sanity={ok}\nreason={reason}\nshares={shares}\n')
            print(f'\n  T-SANITY assertion: {"PASS" if ok else "FAIL"}  ({reason})')
            if not ok:
                # Write SANITY_FAILED side-channel and abort with non-zero
                (out_dir / 'SANITY_FAILED').write_text(reason + '\n')
                # raise -> run_log marks the run as failed.
                raise SystemExit(2)

    plot_trajectories(trajectories_by_board, out_dir / f'trajectories_{cond}.pdf')
    plot_trajectories(trajectories_by_board, out_dir / f'trajectories_{cond}.png')
    write_rationales_table(trajectories_by_board,
                            out_dir / f'rationales_{cond}.md')
    summary_path = out_dir / f'summary_{cond}.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'condition': cond,
            'trajectories_by_board': trajectories_by_board,
        }, f, indent=2, default=str)
    print(f'\nDone. Trajectories -> {out_dir}/{{trajectories_{cond}.pdf,'
          f'rationales_{cond}.md,summary_{cond}.json}}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--mini-config',      default='mini_config.yaml')
    ap.add_argument('--ga-2p', default='logs/optimizer/ga_2p.jsonl')
    ap.add_argument('--ga-3p', default='logs/optimizer/ga_3p.jsonl')
    ap.add_argument('--pool', default='optimizer/strategy_pool.json')
    ap.add_argument('--boards', default='all',
                    help='Comma-separated board labels to run, or "all".')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-games', type=int, default=200,
                    help='Per-iteration game count for canonical eval.')
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--ablation-condition',
                    choices=CONDITIONS, default='full',
                    help='Which TUNER condition to run.')
    ap.add_argument('--out-dir', default='report/figures/llm_design_loop')
    ap.add_argument('--hazard-summary-path', default=None,
                    help='Optional path to a text file pasted into the feed.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_raw = args.ablation_condition
    cond_tag = (f'T-{cond_raw.upper()}' if cond_raw != 'sanity'
                else 'T-SANITY')
    if cond_raw == 'sanity':
        experiment = 'T-SANITY'
    elif args.backend == 'heuristic':
        experiment = 'TUNER-smoke'
    else:
        experiment = 'TUNER'

    with run_log.track(experiment=experiment, script=__file__,
                       args=vars(args), out_dir=out_dir,
                       condition=cond_tag) as run:
        return _run_design_loop(args, out_dir, run)


if __name__ == '__main__':
    raise SystemExit(main())
