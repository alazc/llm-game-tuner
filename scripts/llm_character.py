"""LLM character induction by environment (plan_v2 §5d).

Hold the LLM completely fixed (single Qwen2.5-1.5B-Instruct, single neutral
system prompt with no strategic scaffolding, same instance in every seat).
Vary only the board. Question: does the same instrument exhibit different
strategic character in different environments, and is that character
visible in its reasoning?

Two analyses, both pre-committed before looking at the corpora:
  (3a) per-board reasoning corpus characterization + concept-frequency
       bar chart (one panel per board, shared concept dictionary).
  (3b) cross-board character divergence (pairwise L1 over normalized
       concept-frequency distributions); identifies the maximum-
       divergence board pair.

Pipeline split into two cheap-then-expensive halves so an iteration cycle
on the analysis doesn't re-run the LLM:
  collect  -> writes per-decision JSONL (slow; LLM-bound)
  analyse  -> reads JSONL, emits figures + summaries (fast; pure Python)

Usage (from the repo root):
    # Smoke test using the heuristic backend (no LLM calls; sub-second).
    set PYTHONPATH=. && python scripts/llm_character.py \\
        --backend heuristic --n-games 1 --max-turns 30 \\
        --out-dir report/figures/llm_character_smoke

    # Production run on a local Qwen via transformers (GPU strongly recommended).
    set PYTHONPATH=. && python scripts/llm_character.py \\
        --backend local --model Qwen/Qwen2.5-1.5B-Instruct \\
        --n-games 12 --max-turns 80 \\
        --out-dir report/figures/llm_character

    # Same data, re-do the analysis pass without re-running games.
    set PYTHONPATH=. && python scripts/llm_character.py \\
        --analyse-only --decisions logs/llm_character/decisions.jsonl \\
        --out-dir report/figures/llm_character

Compute caveat: at 1.5B params on CPU expect ~50 sec per decision generation;
12 games x 5 boards x ~30 decisions = ~9000 generations = ~5 days. Plan
budget assumes GPU. Use --backend openai with a local Ollama server, or
fall back to --model Qwen/Qwen2.5-0.5B-Instruct, when no GPU is present.
"""
import argparse
import json
import re
import sys
import time
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# matplotlib lazy-imported in plotting helpers so --analyse-only stays fast
# on machines without a display.

from agents import LLMPlayer
from config import GameConfig
from monopoly.core.game import setup_game_from_config
from optimizer import run_log
from monopoly.core.game_utils import _check_end_conditions
from optimizer.board_sources import build_five_boards
from optimizer.simulate import (_bounded_trade_loop,
                                 _track_interplayer_transfers)
from player_settings import StandardPlayerSettings
from prompts.loader import load_prompt


# --------------------------------------------------------------------------- #
# Pre-committed concept dictionary (frozen before reading any corpora)         #
# --------------------------------------------------------------------------- #

# Each "concept" is a regex applied to the lowercased reasoning text. Multiple
# patterns per concept allow simple morphology / synonym handling without
# pulling in a stemmer. Frozen here for the experiment.
CONCEPT_PATTERNS: Dict[str, List[str]] = {
    'cash':       [r'\bcash\b', r'\bmoney\b', r'\bafford', r'\bliquid'],
    'risk':       [r'\brisk', r'\bdanger', r'\brisky\b', r'\bunsafe'],
    'monopoly':   [r'\bmonopoly\b', r'\bmonopolise', r'\bmonopolize',
                    r'\bcomplete (?:the |my )?(?:group|set|monopoly)'],
    'trade':      [r'\btrade', r'\bnegotiat', r'\bswap'],
    'rent':       [r'\brent\b', r'\brents\b', r'\brent income', r'\brent return'],
    'bankrupt':   [r'\bbankrupt'],
    'accumulate': [r'\baccumulate', r'\bstockpile', r'\bhoard', r'\bsave'],
    'defensive':  [r'\bdefensive', r'\bdefend', r'\bblock', r'\bdeny'],
    'opponent':   [r'\bopponent', r'\bother player', r'\brival'],
    'group':      [r'\bgroup\b', r'\bcolour group', r'\bcolor group', r'\bset\b'],
}
CONCEPTS = list(CONCEPT_PATTERNS.keys())

# A single neutral system prompt with no strategic scaffolding. The plan
# requires this be pinned in the script (not configurable per-board) so the
# environment is the only source of any reasoning shift. The text lives in
# prompts/character_llm_prompt.txt and is hash-locked via prompts.loader so a
# silent edit mid-run aborts the next experiment instead of contaminating it.
NEUTRAL_SYSTEM_PROMPT = load_prompt('character_llm_prompt.txt')


# --------------------------------------------------------------------------- #
# CharacterLLMPlayer: fixed instrument that logs every decision               #
# --------------------------------------------------------------------------- #

class CharacterLLMPlayer(LLMPlayer):
    """LLMPlayer with neutral system prompt, no few-shot, and per-decision logging.

    Each call to _should_buy emits a record onto the (class-level) sink list
    via the on_decision callback set at construction. The record carries the
    full (board_label, game_id, turn, player_name, prompt, response,
    parsed_decision, cash, owned_count, prop_name, prop_group) so the
    downstream analysis can slice however needed.
    """

    _SYSTEM_PROMPT = NEUTRAL_SYSTEM_PROMPT
    _FEW_SHOT: List[dict] = []   # explicitly empty so reasoning is uninstructed

    def __init__(self, name: str, settings=None, backend: str = 'local',
                 model_name: str = None, max_new_tokens: int = 160,
                 on_decision=None, board_label: str = '',
                 game_id: int = 0, turn_ref=None,
                 base_seed: Optional[int] = None):
        super().__init__(name, settings=settings, backend=backend,
                         model_name=model_name, max_new_tokens=max_new_tokens)
        self._on_decision = on_decision
        self._board_label = board_label
        self._game_id = game_id
        self._turn_ref = turn_ref or [0]
        # Snapshot of the locked generation parameters, written into every
        # decision record so post-run analysis can verify nothing drifted.
        # `engine`, `dtype`, `attn_impl` capture the cluster-vs-dev-host
        # delta so a reviewer can grep gen_cfg across decisions.jsonl and
        # confirm the entire run used one instrument config.
        import os as _os
        self._gen_cfg = {
            'model':           (model_name or 'Qwen/Qwen2.5-1.5B-Instruct'),
            'max_new_tokens':  int(max_new_tokens),
            'do_sample':       False,
            'temperature':     0.0,
            'seed':            int(base_seed) if base_seed is not None else None,
            'engine':          ('openai-compat' if backend == 'openai'
                                  else ('heuristic' if backend == 'heuristic'
                                          else 'transformers')),
            'dtype':           _os.environ.get('LLM_DTYPE', 'float16'),
            'attn_impl':       _os.environ.get('LLM_ATTN_IMPL', 'default'),
            'deterministic':   True,
        }

    def _should_buy(self, property_to_buy) -> bool:
        # Mirror parent affordability + ignore-group shortcuts so we don't
        # spam the LLM on impossible decisions (those wouldn't happen in
        # human play either).
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        if property_to_buy.group in self.settings.ignore_property_groups:
            return False
        backend_error: Optional[str] = None
        if self._backend == 'heuristic':
            decision = True
            response = 'REASON: heuristic backend always buys when affordable.\nANSWER: BUY'
        else:
            prompt_text = self._build_buy_prompt(property_to_buy)
            try:
                if self._backend == 'openai':
                    response = self._query_openai(prompt_text)
                elif self._backend == 'anthropic':
                    response = self._query_anthropic(prompt_text)
                else:
                    response = self._query_local(prompt_text)
            except Exception as ex:
                # Tag the decision so the analysis pass can drop it from the
                # corpus rather than treating "backend failure" as reasoning.
                backend_error = f'{type(ex).__name__}: {ex}'
                response = f'REASON: backend failure ({ex}); fallback BUY.\nANSWER: BUY'
            decision = self._parse_decision(response)
        reason_text, format_ok = _extract_reason(response)
        # Build prompt for logging (heuristic path skipped it for speed).
        prompt_for_log = self._build_buy_prompt(property_to_buy)
        if self._on_decision is not None:
            self._on_decision({
                'board':         self._board_label,
                'game':          self._game_id,
                'turn':          int(self._turn_ref[0]),
                'player':        self.name,
                'prop_name':     property_to_buy.name,
                'prop_group':    property_to_buy.group,
                'cash':          int(self.money),
                'owned_count':   len(self.owned),
                'prompt':        prompt_for_log,
                'response':      response,
                'reason':        reason_text,
                'decision':      'BUY' if decision else 'PASS',
                'format_ok':     bool(format_ok) and backend_error is None,
                'backend_error': backend_error,
                'gen_cfg':       dict(self._gen_cfg),
            })
        self._n_buy_decisions += 1
        if decision:
            self._n_buy_yes += 1
        return decision


def _extract_reason(response: str) -> Tuple[str, bool]:
    """Return (reason_text, format_ok).

    `format_ok` is True iff the response contained a parseable 'REASON:' tag.
    A False value means we fell back to the first-200-chars heuristic and the
    decision should be excluded from the corpus by the analysis pass (the
    "reasoning" otherwise pollutes concept frequencies with whatever prefix
    tokens or template fragments the LLM emitted)."""
    m = re.search(r'REASON\s*:\s*(.+?)(?:\n|ANSWER\s*:)',
                  response, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), True
    return response.strip()[:200], False


# --------------------------------------------------------------------------- #
# Game runner with LLM self-play                                               #
# --------------------------------------------------------------------------- #

def run_self_play(cfg, board_label: str, game_id: int, seed: int,
                  max_turns: int, backend: str, model_name: Optional[str],
                  on_decision) -> dict:
    """Run one game with two CharacterLLMPlayer instances (self-play)."""
    starting_money_cfg = cfg.settings.starting_money
    if isinstance(starting_money_cfg, dict):
        default_starting = next(iter(starting_money_cfg.values()), 1500)
    else:
        default_starting = starting_money_cfg or 1500

    board, dice, elog, blog = setup_game_from_config(0, seed, cfg)
    elog.disabled = True
    blog.disabled = True

    turn_ref = [0]   # mutable so the players see the live turn number
    players = []
    for nm in ('LLM-A', 'LLM-B'):
        p = CharacterLLMPlayer(
            nm, settings=StandardPlayerSettings(),
            backend=backend, model_name=model_name,
            on_decision=on_decision, board_label=board_label,
            game_id=game_id, turn_ref=turn_ref, base_seed=seed,
        )
        p.money = default_starting
        players.append(p)

    turn_n = 0
    per_turn_trade_counts: dict = {}
    with ExitStack() as stack:
        total = stack.enter_context(_track_interplayer_transfers())
        stack.enter_context(_bounded_trade_loop(per_turn_trade_counts,
                                                 max_per_turn=5))
        for turn_n in range(1, max_turns + 1):
            turn_ref[0] = turn_n
            per_turn_trade_counts.clear()
            if _check_end_conditions(players, elog, 0, turn_n):
                break
            for p in players:
                if p.is_bankrupt:
                    continue
                p.make_a_move(board, players, dice, elog)

    alive = [p for p in players if not p.is_bankrupt]
    return {
        'board':         board_label,
        'game':          game_id,
        'seed':          seed,
        'rounds':        turn_n,
        'truncated':     len(alive) != 1,
        'winner':        alive[0].name if len(alive) == 1 else None,
        'transfer':      int(total[0]) if isinstance(total, list) else 0,
    }


# Board sources are resolved by `optimizer.board_sources.build_five_boards`
# so this script and `scripts/hazard_curves.py` always probe the same five
# points in the design space.


# --------------------------------------------------------------------------- #
# Analysis (concept frequencies, divergence, plotting)                         #
# --------------------------------------------------------------------------- #

def concept_counts(reasonings: List[str]) -> Dict[str, int]:
    """Count concept hits across a list of reasoning strings.

    A reasoning hits a concept iff *any* of its patterns matches the lowercased
    text. We count one hit per (reasoning, concept) so a concept mentioned 3
    times in a single sentence does not dominate the corpus statistics.
    """
    counts: Dict[str, int] = {c: 0 for c in CONCEPTS}
    for raw in reasonings:
        if not raw:
            continue
        text = raw.lower()
        for c, pats in CONCEPT_PATTERNS.items():
            if any(re.search(p, text) for p in pats):
                counts[c] += 1
    return counts


def normalize_freq(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {c: 0.0 for c in counts}
    return {c: counts[c] / total for c in counts}


def l1_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    return 0.5 * sum(abs(p[c] - q[c]) for c in CONCEPTS)


def per_board_corpus(decisions: List[dict]) -> Dict[str, List[str]]:
    """Aggregate reasoning text per board, EXCLUDING decisions that failed
    format validation or hit a backend exception. The exclusion is the whole
    point of the format_ok / backend_error tags: malformed responses pollute
    concept frequencies, and synthetic 'backend failure' reasoning would
    bias every board it touches."""
    out: Dict[str, List[str]] = {}
    for d in decisions:
        if d.get('backend_error') is not None:
            continue
        # Decisions written by the pre-hardening smoke run lack format_ok;
        # treat absent as True so legacy JSONL still works.
        if not d.get('format_ok', True):
            continue
        out.setdefault(d['board'], []).append(d.get('reason', ''))
    return out


def per_board_quality(decisions: List[dict]) -> Dict[str, dict]:
    """Per-board format-compliance and backend-error rates. Reports the data
    quality of the LLM run BEFORE any analysis derives a finding from it."""
    out: Dict[str, dict] = {}
    for d in decisions:
        b = d['board']
        rec = out.setdefault(b, {'n_total': 0, 'n_format_ok': 0,
                                  'n_backend_error': 0})
        rec['n_total'] += 1
        if d.get('backend_error') is not None:
            rec['n_backend_error'] += 1
        elif d.get('format_ok', True):
            rec['n_format_ok'] += 1
    for b, rec in out.items():
        rec['format_ok_rate'] = (rec['n_format_ok'] / rec['n_total']
                                  if rec['n_total'] else 0.0)
        rec['backend_error_rate'] = (rec['n_backend_error'] / rec['n_total']
                                       if rec['n_total'] else 0.0)
    return out


def per_board_stats(corpora: Dict[str, List[str]]) -> Dict[str, dict]:
    out = {}
    for board, reasonings in corpora.items():
        counts = concept_counts(reasonings)
        out[board] = {
            'n_decisions':   len(reasonings),
            'avg_reason_chars': float(np.mean([len(r) for r in reasonings])) if reasonings else 0.0,
            'concept_counts':   counts,
            'concept_freq':     normalize_freq(counts),
        }
    return out


def divergence_matrix(stats: Dict[str, dict]) -> Tuple[List[str], np.ndarray]:
    boards = list(stats.keys())
    n = len(boards)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            M[i, j] = l1_divergence(stats[boards[i]]['concept_freq'],
                                    stats[boards[j]]['concept_freq'])
    return boards, M


def plot_concept_frequencies(stats: Dict[str, dict], out_path: Path):
    import matplotlib.pyplot as plt
    boards = list(stats.keys())
    n = len(boards)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    x = np.arange(len(CONCEPTS))
    for ax, board in zip(axes, boards):
        freq = stats[board]['concept_freq']
        ax.bar(x, [freq[c] for c in CONCEPTS], color='#1f77b4')
        ax.set_xticks(x); ax.set_xticklabels(CONCEPTS, rotation=60, ha='right',
                                              fontsize=8)
        ax.set_title(f'{board}\n(n={stats[board]["n_decisions"]})', fontsize=10)
        ax.set_ylim(0, max(0.05, max(freq.values()) * 1.15))
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    axes[0].set_ylabel('concept frequency (per reasoning hit)')
    fig.suptitle('LLM reasoning concept frequency by board', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_divergence_matrix(boards: List[str], M: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap='magma', vmin=0, vmax=max(0.05, float(M.max())))
    ax.set_xticks(range(len(boards))); ax.set_yticks(range(len(boards)))
    ax.set_xticklabels(boards, rotation=45, ha='right')
    ax.set_yticklabels(boards)
    for i in range(len(boards)):
        for j in range(len(boards)):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                    color='white' if M[i, j] > M.max() * 0.5 else 'black',
                    fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('L1 divergence over normalized concept freq')
    ax.set_title('Cross-board character divergence')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_corpus_samples(corpora: Dict[str, List[str]], out_path: Path,
                         k: int = 10):
    """Dump up to K reasoning blocks per board for human audit."""
    rng = np.random.default_rng(0)
    lines = []
    for board, reasonings in corpora.items():
        lines.append(f'## {board}  (n={len(reasonings)})')
        sample = (list(reasonings)
                  if len(reasonings) <= k
                  else [reasonings[i] for i in rng.choice(len(reasonings), k, replace=False)])
        for r in sample:
            lines.append(f'- {r}')
        lines.append('')
    out_path.write_text('\n'.join(lines))


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def run_collect(args, decisions_path: Path) -> List[dict]:
    print('Resolving board sources...')
    sources = build_five_boards(
        canonical_config=args.canonical_config,
        mini_config=args.mini_config,
        ga_2p=args.ga_2p, ga_3p=args.ga_3p,
    )
    print(f'Boards under test: {[lbl for lbl, _ in sources]}')

    decisions: List[dict] = []
    outcomes: List[dict] = []

    def on_decision(rec):
        decisions.append(rec)

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_path, 'w') as fh:
        for board_label, cfg in sources:
            print(f'  {board_label}: simulating {args.n_games} self-play game(s)...',
                  flush=True)
            for gi in range(args.n_games):
                seed = args.base_seed + gi
                t0 = time.perf_counter()
                outcome = run_self_play(
                    cfg=cfg, board_label=board_label, game_id=gi, seed=seed,
                    max_turns=args.max_turns, backend=args.backend,
                    model_name=args.model, on_decision=on_decision,
                )
                outcome['wall_seconds'] = float(time.perf_counter() - t0)
                outcomes.append(outcome)
                # Flush latest decisions to disk so a crash mid-overnight-run
                # preserves what we have.
                while decisions:
                    fh.write(json.dumps(decisions.pop(0)) + '\n')
                fh.flush()
                print(f'    game {gi+1}/{args.n_games}: '
                      f'rounds={outcome["rounds"]}  '
                      f'winner={outcome["winner"]}  '
                      f'truncated={outcome["truncated"]}  '
                      f'wall={outcome["wall_seconds"]:.1f}s',
                      flush=True)
    outcomes_path = decisions_path.with_name('outcomes.json')
    outcomes_path.write_text(json.dumps(outcomes, indent=2))
    return outcomes


_REQUIRED_DECISION_FIELDS = {'board', 'reason', 'decision'}


def _validate_decisions(decisions: List[dict], path: Path) -> None:
    """Fail fast if the JSONL is from a different experiment or a corrupted
    half-flushed run. Cheap insurance against the analysis pass producing a
    plausible-looking divergence matrix from junk."""
    if not decisions:
        raise SystemExit(f'{path}: no decisions read')
    missing = _REQUIRED_DECISION_FIELDS - set(decisions[0].keys())
    if missing:
        raise SystemExit(f'{path}: decision records missing fields {missing}')


def run_analyse(args, decisions_path: Path, out_dir: Path):
    print(f'Reading decisions from {decisions_path}...')
    decisions = []
    with open(decisions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    _validate_decisions(decisions, decisions_path)
    print(f'  {len(decisions)} decisions across boards: '
          f'{sorted(set(d["board"] for d in decisions))}')

    # Data-quality report BEFORE deriving any finding from the corpus.
    quality = per_board_quality(decisions)
    gate = format_pass_rate_gate(decisions, threshold=args.format_ok_threshold)
    print('  data quality:')
    failing_boards: List[str] = list(gate['failing'])
    for b, rec in quality.items():
        flag = ' OK' if rec['format_ok_rate'] >= args.format_ok_threshold else ' BELOW THRESHOLD'
        print(f'    {b:>12}: n={rec["n_total"]:5d}  '
              f'format_ok={rec["format_ok_rate"]*100:5.1f}%  '
              f'backend_err={rec["backend_error_rate"]*100:5.1f}%  {flag}')

    corpora = per_board_corpus(decisions)
    stats = per_board_stats(corpora)

    out_dir.mkdir(parents=True, exist_ok=True)
    quality_path = out_dir / 'data_quality.json'
    with open(quality_path, 'w') as f:
        json.dump({'per_board': quality,
                   'format_ok_threshold': args.format_ok_threshold,
                   'format_pass_rate_gate': gate},
                  f, indent=2)
    print(f'  quality   -> {quality_path}')

    stats_path = out_dir / 'concept_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  stats     -> {stats_path}')

    if failing_boards and not args.force_render:
        print(f'\n[REFUSE] {len(failing_boards)} board(s) below format_ok '
              f'threshold {args.format_ok_threshold}: {failing_boards}')
        print('         Refusing to render divergence matrix; the headline '
              'finding would be derived from contaminated corpora.')
        print('         Re-run with --force-render to override (e.g. for '
              'a calibration-finding write-up).')
        return

    boards, M = divergence_matrix(stats)

    # Statistical guards on the divergence matrix (A3 from eng review).
    print('Computing bootstrap CI + permutation p-value on divergence matrix...')
    boot = bootstrap_divergence(corpora, n_resamples=args.n_bootstrap,
                                 seed=args.stat_seed)
    perm = permutation_test_max_divergence(corpora,
                                            n_resamples=args.n_permutations,
                                            seed=args.stat_seed)

    # Within-board noise floor + grounded-rate precondition (Step 3 / LOCK §3,§4).
    print('Computing within-board L1 noise floor (100 random splits per board)...')
    noise = within_board_noise_floor(corpora, n_splits=args.n_within_splits,
                                      seed=args.stat_seed)
    print(f'  pooled noise floor: mean={noise["pooled_mean"]:.3f}  '
          f'sigma={noise["pooled_std"]:.3f}  '
          f'meaningful_threshold(mean+2sigma)={noise["meaningful_threshold"]:.3f}')

    print('Computing per-board grounded-reasoning rate...')
    grounded = per_board_grounded_rate(decisions)
    for b, rec in grounded.items():
        flag = ' OK' if rec.get('grounded_rate', 0) >= 0.5 else ' UNGROUNDED'
        print(f'    {b:>12}: grounded_rate={rec["grounded_rate"]*100:5.1f}%  '
              f'(prop={rec["has_prop_rate"]*100:4.1f}%  '
              f'money={rec["has_money_rate"]*100:4.1f}%  '
              f'owned={rec["has_owned_rate"]*100:4.1f}%){flag}')

    cross_max = _max_offdiag(M)
    cross_meaningful = bool(cross_max > noise['meaningful_threshold'])
    div_path = out_dir / 'divergence.json'
    with open(div_path, 'w') as f:
        json.dump({
            'boards':                boards,
            'matrix':                M.tolist(),
            'max_pair':              _argmax_offdiag(boards, M),
            'bootstrap_max_pair_ci': boot,
            'permutation_test':      perm,
            'noise_floor':           noise,
            'cross_board_meaningful':       cross_meaningful,
            'cross_board_meaningful_rule':  'max_cross_L1 > pooled_mean + 2*pooled_std',
            'grounded':              grounded,
        }, f, indent=2)
    print(f'  divergence -> {div_path}')
    print(f'  cross-board signal {"PASSES" if cross_meaningful else "FAILS"} '
          f'noise-floor threshold ({cross_max:.3f} vs {noise["meaningful_threshold"]:.3f})')

    print('Rendering figures + corpus samples...')
    plot_concept_frequencies(stats, out_dir / 'concept_frequencies.pdf')
    plot_concept_frequencies(stats, out_dir / 'concept_frequencies.png')
    plot_divergence_matrix(boards, M, out_dir / 'divergence_matrix.pdf')
    plot_divergence_matrix(boards, M, out_dir / 'divergence_matrix.png')
    plot_noise_floor(noise, M, out_dir / 'noise_floor.pdf')
    plot_noise_floor(noise, M, out_dir / 'noise_floor.png')
    plot_grounded_rate(grounded, out_dir / 'grounded_rate.pdf')
    plot_grounded_rate(grounded, out_dir / 'grounded_rate.png')
    write_corpus_samples(corpora, out_dir / 'corpus_samples.md', k=10)

    # Robustness checks (round 1 §1.3): primary 10-concept finding plus two
    # independent dictionaries — Empath's 200 categories and sentence-BERT
    # cosine. If all three agree, the divergence is dictionary-robust.
    print('Computing Empath-200 robustness...')
    empath = empath_robustness(decisions,
                                n_within_splits=args.n_within_splits,
                                seed=args.stat_seed)
    with open(out_dir / 'empath_divergence.json', 'w') as f:
        json.dump(empath, f, indent=2)
    if empath.get('matrix'):
        plot_empath_matrix(empath, out_dir / 'empath_divergence_matrix.pdf')
        plot_empath_matrix(empath, out_dir / 'empath_divergence_matrix.png')
        print(f'  empath cross_max={empath["cross_max"]:.3f} '
              f'(threshold={empath["noise_floor"]["meaningful_threshold"]:.3f}) '
              f'meaningful={empath["cross_board_meaningful"]}')
    else:
        print(f'  empath skipped: {empath.get("note", "no data")}')

    print('Computing sentence-BERT robustness...')
    sbert = sbert_robustness(decisions)
    with open(out_dir / 'sbert_cosine.json', 'w') as f:
        json.dump(sbert, f, indent=2)
    if sbert.get('cosine_matrix'):
        plot_sbert_matrix(sbert, out_dir / 'sbert_cosine_matrix.pdf')
        plot_sbert_matrix(sbert, out_dir / 'sbert_cosine_matrix.png')
        print(f'  sBERT model={sbert["model"]}; matrix written.')
    else:
        print(f'  sBERT skipped: {sbert.get("note", "no data")}')
    mp = _argmax_offdiag(boards, M)
    print(f'  max-divergence board pair: {mp["pair"]}  L1={mp["value"]:.3f}')
    print(f'    bootstrap 95% CI on max-pair L1: '
          f'[{boot["max_pair_ci_lo"]:.3f}, {boot["max_pair_ci_hi"]:.3f}]')
    print(f'    permutation p-value vs H0(no per-board signal): '
          f'p={perm["p_value"]:.4f}  (n_resamples={perm["n_resamples"]})')


def _argmax_offdiag(boards: List[str], M: np.ndarray) -> dict:
    n = len(boards)
    best = {'pair': (boards[0], boards[1] if n > 1 else boards[0]), 'value': 0.0}
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] > best['value']:
                best = {'pair': [boards[i], boards[j]], 'value': float(M[i, j])}
    return best


def _max_offdiag(M: np.ndarray) -> float:
    """Max strictly-upper-triangular entry of a square divergence matrix."""
    n = M.shape[0]
    out = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] > out:
                out = float(M[i, j])
    return out


def _divergence_max_from_corpora(corpora: Dict[str, List[str]]) -> float:
    """Recompute the max-pair L1 divergence from a (possibly resampled or
    relabelled) corpora dict. This is the single scalar that the bootstrap CI
    and the permutation test both characterise."""
    stats = per_board_stats(corpora)
    _, M = divergence_matrix(stats)
    return _max_offdiag(M)


def bootstrap_divergence(corpora: Dict[str, List[str]], n_resamples: int = 200,
                         seed: int = 0) -> dict:
    """Resample reasonings within each board with replacement; recompute the
    max-pair divergence each time; return mean + 95% percentile CI on it.

    This characterises the precision of our estimate of how different the
    most-divergent pair really is, conditional on each board's empirical
    reasoning distribution. It does NOT answer the null-hypothesis question
    of whether the boards differ at all (that's the permutation test below)."""
    rng = np.random.default_rng(seed)
    boards = list(corpora.keys())
    arrays = {b: np.asarray(corpora[b], dtype=object) for b in boards}
    sizes = {b: len(arrays[b]) for b in boards}
    samples: List[float] = []
    for _ in range(n_resamples):
        resampled = {
            b: list(arrays[b][rng.integers(0, sizes[b], size=sizes[b])])
                if sizes[b] > 0 else []
            for b in boards
        }
        samples.append(_divergence_max_from_corpora(resampled))
    samples_arr = np.asarray(samples, dtype=float)
    return {
        'n_resamples':       int(n_resamples),
        'mean':              float(samples_arr.mean()) if len(samples_arr) else 0.0,
        'max_pair_ci_lo':    float(np.percentile(samples_arr, 2.5)) if len(samples_arr) else 0.0,
        'max_pair_ci_hi':    float(np.percentile(samples_arr, 97.5)) if len(samples_arr) else 0.0,
    }


# --------------------------------------------------------------------------- #
# Within-board L1 noise floor (ANALYSIS_PLAN §3, Step 3 of build plan)         #
# --------------------------------------------------------------------------- #

def within_board_noise_floor(corpora: Dict[str, List[str]],
                             n_splits: int = 100, seed: int = 0) -> dict:
    """Estimate the within-board concept-frequency L1 distance.

    For each board: 100 random equal-size splits of its reasoning corpus into
    halves A and B; L1(freq(A), freq(B)). Pooled mean+sigma across all splits
    on all boards is the noise floor. A cross-board L1 is "signal" iff it
    exceeds mean + 2*sigma; otherwise it could be a within-board sampling
    artifact and the headline finding inverts (see ANALYSIS_PLAN §3).

    Returns a dict shaped:
      {
        'per_board': {board: {'mean': ..., 'std': ..., 'samples': [...]}},
        'pooled_mean': ..., 'pooled_std': ...,
        'meaningful_threshold': pooled_mean + 2*pooled_std,
        'n_splits': ..., 'seed': ...,
      }
    """
    rng = np.random.default_rng(seed)
    per_board: Dict[str, dict] = {}
    pooled: List[float] = []

    for board, reasonings in corpora.items():
        n = len(reasonings)
        if n < 4:
            # Need at least two reasonings per half to define a frequency.
            per_board[board] = {'mean': 0.0, 'std': 0.0,
                                 'samples': [], 'n_corpus': n,
                                 'note': 'corpus too small for split (<4)'}
            continue
        half = n // 2
        idx = np.arange(n)
        samples: List[float] = []
        for _ in range(n_splits):
            rng.shuffle(idx)
            a = [reasonings[i] for i in idx[:half]]
            b = [reasonings[i] for i in idx[half:2 * half]]
            fa = normalize_freq(concept_counts(a))
            fb = normalize_freq(concept_counts(b))
            samples.append(l1_divergence(fa, fb))
        arr = np.asarray(samples, dtype=float)
        per_board[board] = {
            'mean':       float(arr.mean()),
            'std':        float(arr.std(ddof=0)),
            'samples':    samples,
            'n_corpus':   n,
        }
        pooled.extend(samples)

    pooled_arr = np.asarray(pooled, dtype=float)
    if len(pooled_arr):
        pooled_mean = float(pooled_arr.mean())
        pooled_std  = float(pooled_arr.std(ddof=0))
    else:
        pooled_mean = pooled_std = 0.0
    return {
        'per_board':            per_board,
        'pooled_mean':          pooled_mean,
        'pooled_std':           pooled_std,
        'meaningful_threshold': pooled_mean + 2.0 * pooled_std,
        'n_splits':             int(n_splits),
        'seed':                 int(seed),
    }


# --------------------------------------------------------------------------- #
# Per-board board-state-reference precondition (Step 3, ANALYSIS_PLAN §4)     #
# --------------------------------------------------------------------------- #

# Money/rent integer marker: either a "$<digits>" anchor, or a standalone
# integer >= 20 within text reasonable bounds. The lower bound 20 cuts out
# small turn counters / dice values that are not rent-bearing.
_MONEY_PATTERN  = re.compile(r'\$\d{1,5}')
_INT_PATTERN    = re.compile(r'(?<!\d)([2-9]\d{1,3}|1\d{2,3})(?!\d)')

# Owned-list / portfolio keywords. Lowercased substring match.
_OWNED_KEYWORDS = (
    'owned', 'holdings', 'properties', 'portfolio',
    'i have', 'i own', 'we own', 'they own', "opponent's",
    'my cash', 'my money',
)


def _grounded_block(text: str, prop_names_lower: List[str]) -> dict:
    """Classify one reasoning block; return granular hits for the table."""
    lo = (text or '').lower()
    has_prop  = any(pn in lo for pn in prop_names_lower) if prop_names_lower else False
    has_money = bool(_MONEY_PATTERN.search(text or '')) or bool(_INT_PATTERN.search(text or ''))
    has_owned = any(k in lo for k in _OWNED_KEYWORDS)
    return {
        'grounded':  bool(has_prop or has_money or has_owned),
        'has_prop':  has_prop,
        'has_money': has_money,
        'has_owned': has_owned,
    }


def per_board_grounded_rate(decisions: List[dict]) -> Dict[str, dict]:
    """Per-board rate at which reasoning references observable game state.

    Property-name dictionary per board is derived from the decisions
    themselves: any prop_name the LLM was asked to decide on belongs to the
    board, so the union over decisions for that board is the board's property
    set as far as the LLM saw it. This matches what the precondition check
    actually wants: are reasoning blocks referring to props the LLM was
    *also asked about*, vs. emitting generic narration?
    """
    # Build per-board prop-name dictionary
    props_by_board: Dict[str, set] = {}
    for d in decisions:
        if d.get('format_ok', True) and d.get('backend_error') is None:
            props_by_board.setdefault(d['board'], set()).add(d.get('prop_name', ''))

    out: Dict[str, dict] = {}
    for d in decisions:
        if d.get('backend_error') is not None:
            continue
        if not d.get('format_ok', True):
            continue
        b = d['board']
        rec = out.setdefault(b, {
            'n_total': 0, 'n_grounded': 0,
            'n_has_prop': 0, 'n_has_money': 0, 'n_has_owned': 0,
        })
        rec['n_total'] += 1
        prop_names_lower = [p.lower() for p in props_by_board.get(b, set()) if p]
        cls = _grounded_block(d.get('reason', ''), prop_names_lower)
        if cls['grounded']:   rec['n_grounded']  += 1
        if cls['has_prop']:   rec['n_has_prop']  += 1
        if cls['has_money']:  rec['n_has_money'] += 1
        if cls['has_owned']:  rec['n_has_owned'] += 1

    for b, rec in out.items():
        n = max(rec['n_total'], 1)
        rec['grounded_rate']    = rec['n_grounded']  / n
        rec['has_prop_rate']    = rec['n_has_prop']  / n
        rec['has_money_rate']   = rec['n_has_money'] / n
        rec['has_owned_rate']   = rec['n_has_owned'] / n
    return out


def plot_grounded_rate(grounded: Dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    boards = list(grounded.keys())
    n = len(boards)
    if n == 0:
        return
    cats   = ('has_prop_rate', 'has_money_rate', 'has_owned_rate', 'grounded_rate')
    labels = ('property name', 'money/rent integer', 'owned-list keyword', 'any (grounded)')
    x = np.arange(n)
    width = 0.18
    fig, ax = plt.subplots(figsize=(0.9 + 1.2 * n, 4.0))
    for i, (cat, lab) in enumerate(zip(cats, labels)):
        vals = [grounded[b][cat] for b in boards]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(boards, rotation=20, ha='right')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='red', linestyle=':', linewidth=0.8,
               label='ungrounded threshold (50%)')
    ax.set_ylabel('rate')
    ax.set_title('Reasoning grounded in observable game state, per board')
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_noise_floor(noise: dict, divergence_M: np.ndarray, out_path: Path) -> None:
    """Box-plot of within-board L1 splits per board, with cross-board max
    annotated as a horizontal line and the meaningful-threshold also drawn.

    Lets a reviewer see at a glance whether cross-board signal exceeds the
    within-board sampling band.
    """
    import matplotlib.pyplot as plt
    per_board = noise.get('per_board', {})
    boards = list(per_board.keys())
    if not boards:
        return
    data = [per_board[b].get('samples', []) for b in boards]
    fig, ax = plt.subplots(figsize=(1.0 + 1.0 * len(boards), 4.0))
    # matplotlib renamed `labels` -> `tick_labels` in 3.9; support older envs.
    try:
        ax.boxplot(data, tick_labels=boards, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=boards, showfliers=False)
    threshold = noise.get('meaningful_threshold', 0.0)
    ax.axhline(threshold, color='red', linestyle='--',
               label=f'mean+2σ (threshold) = {threshold:.3f}')
    if divergence_M is not None and divergence_M.size > 0:
        cross_max = _max_offdiag(divergence_M)
        ax.axhline(cross_max, color='black', linestyle='-',
                   label=f'observed max cross-board L1 = {cross_max:.3f}')
    ax.set_ylabel('within-board L1 (random equal-size splits)')
    ax.set_title('Within-board noise floor vs cross-board signal')
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def permutation_test_max_divergence(corpora: Dict[str, List[str]],
                                     n_resamples: int = 500,
                                     seed: int = 0) -> dict:
    """Permutation test under H0: reasoning text is exchangeable across boards
    (i.e. the LLM has NO per-board character induction). Pool every reasoning,
    randomly redistribute back into per-board buckets at the original sizes,
    recompute the max-pair divergence; the p-value is the fraction of
    resamples whose statistic >= the observed statistic."""
    rng = np.random.default_rng(seed)
    boards = list(corpora.keys())
    sizes = [len(corpora[b]) for b in boards]
    pooled = []
    for b in boards:
        pooled.extend(corpora[b])
    pooled_arr = np.asarray(pooled, dtype=object)
    n_total = len(pooled_arr)

    observed = _divergence_max_from_corpora(corpora)
    if n_total == 0:
        return {'observed': 0.0, 'p_value': 1.0, 'n_resamples': 0,
                'null_mean': 0.0, 'null_ci_hi': 0.0}

    null_stats: List[float] = []
    for _ in range(n_resamples):
        idx = rng.permutation(n_total)
        cursor = 0
        relabelled: Dict[str, List[str]] = {}
        for b, sz in zip(boards, sizes):
            relabelled[b] = list(pooled_arr[idx[cursor:cursor + sz]])
            cursor += sz
        null_stats.append(_divergence_max_from_corpora(relabelled))
    null_arr = np.asarray(null_stats, dtype=float)
    p_value = float(((null_arr >= observed).sum() + 1) / (len(null_arr) + 1))
    return {
        'observed':     float(observed),
        'p_value':      p_value,
        'n_resamples':  int(n_resamples),
        'null_mean':    float(null_arr.mean()),
        'null_ci_hi':   float(np.percentile(null_arr, 95.0)),
    }


# --------------------------------------------------------------------------- #
# Format-pass-rate gate (round 1 §1.3)                                          #
# --------------------------------------------------------------------------- #

def format_pass_rate_gate(decisions: List[dict],
                          threshold: float = 0.70) -> dict:
    """Per-board format-compliance pass/fail.

    Returns a dict shaped:
      {
        'threshold': float,
        'per_board': {board: {'rate': float, 'pass': bool, 'n': int}},
        'all_pass':  bool,
        'failing':   [board_label, ...],
      }

    Backend errors count as format failures (the corpus excludes them anyway,
    but a high rate of backend errors is itself a data-quality flag and the
    gate is the right place to surface it). A board falling below `threshold`
    means its corpus is unreliable and downstream divergence claims should be
    flagged in the writeup."""
    quality = per_board_quality(decisions)
    per_board: Dict[str, dict] = {}
    failing: List[str] = []
    for board, rec in quality.items():
        rate = float(rec.get('format_ok_rate', 0.0))
        ok = rate >= threshold
        per_board[board] = {'rate': rate, 'pass': bool(ok),
                             'n': int(rec.get('n_total', 0))}
        if not ok:
            failing.append(board)
    return {
        'threshold':  float(threshold),
        'per_board':  per_board,
        'all_pass':   len(failing) == 0,
        'failing':    failing,
    }


# --------------------------------------------------------------------------- #
# Empath-200 robustness check (round 1 §1.3)                                    #
# --------------------------------------------------------------------------- #

def empath_robustness(decisions: List[dict],
                       n_within_splits: int = 100,
                       seed: int = 0) -> dict:
    """Cross-board L1 over Empath's 200 categories with the same noise-floor
    methodology as the primary 10-concept analysis.

    Empath gives us a much wider semantic dictionary; if the 10-concept
    headline replicates against 200 unrelated categories, the divergence
    finding is robust to the dictionary choice. This is the agreement
    check, not a replacement for the primary analysis.

    Returns a dict shaped:
      {
        'boards':                [...],
        'matrix':                [[float ...] ...],
        'cross_max':             float,
        'noise_floor':           {pooled_mean, pooled_std, meaningful_threshold, ...},
        'cross_board_meaningful': bool,
        'n_categories':          int,
        'note':                  str   (set if Empath unavailable)
      }
    """
    try:
        from empath import Empath
    except Exception as ex:
        return {'note': f'empath unavailable ({type(ex).__name__}: {ex})',
                'boards': [], 'matrix': [],
                'cross_max': 0.0, 'cross_board_meaningful': False,
                'n_categories': 0}

    lex = Empath()
    categories = sorted(lex.cats.keys())

    def _empath_freq(text: str) -> Dict[str, float]:
        # Empath returns category counts. normalize=False so we can pool.
        try:
            return lex.analyze(text or '', normalize=False) or {}
        except Exception:
            return {}

    def _l1(p: Dict[str, float], q: Dict[str, float]) -> float:
        s_p = sum(p.values()); s_q = sum(q.values())
        if s_p <= 0 or s_q <= 0:
            return 0.0
        return 0.5 * sum(abs(p.get(c, 0.0) / s_p - q.get(c, 0.0) / s_q)
                          for c in categories)

    corpora = per_board_corpus(decisions)
    boards = list(corpora.keys())
    if not boards:
        return {'boards': [], 'matrix': [], 'cross_max': 0.0,
                'cross_board_meaningful': False, 'n_categories': len(categories)}

    # Empath totals per board (single bag).
    totals: Dict[str, Dict[str, float]] = {}
    for b, rs in corpora.items():
        agg: Dict[str, float] = {c: 0.0 for c in categories}
        for r in rs:
            f = _empath_freq(r)
            for c, v in f.items():
                agg[c] = agg.get(c, 0.0) + float(v)
        totals[b] = agg

    n = len(boards)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            M[i, j] = _l1(totals[boards[i]], totals[boards[j]])

    # Within-board Empath noise floor: 100 random equal-size splits per board.
    rng = np.random.default_rng(seed)
    pooled: List[float] = []
    per_board_noise: Dict[str, dict] = {}
    for b, rs in corpora.items():
        if len(rs) < 4:
            per_board_noise[b] = {'mean': 0.0, 'std': 0.0,
                                   'note': 'corpus too small (<4)'}
            continue
        idx = np.arange(len(rs))
        half = len(rs) // 2
        samples: List[float] = []
        for _ in range(n_within_splits):
            rng.shuffle(idx)
            ai = idx[:half]; bi = idx[half:2 * half]
            agg_a: Dict[str, float] = {c: 0.0 for c in categories}
            agg_b: Dict[str, float] = {c: 0.0 for c in categories}
            for k in ai:
                f = _empath_freq(rs[k])
                for c, v in f.items(): agg_a[c] = agg_a.get(c, 0.0) + float(v)
            for k in bi:
                f = _empath_freq(rs[k])
                for c, v in f.items(): agg_b[c] = agg_b.get(c, 0.0) + float(v)
            samples.append(_l1(agg_a, agg_b))
        arr = np.asarray(samples, dtype=float)
        per_board_noise[b] = {'mean': float(arr.mean()),
                               'std':  float(arr.std(ddof=0)),
                               'samples': samples}
        pooled.extend(samples)

    pooled_arr = np.asarray(pooled, dtype=float)
    if len(pooled_arr):
        pmean = float(pooled_arr.mean()); pstd = float(pooled_arr.std(ddof=0))
    else:
        pmean = pstd = 0.0
    threshold = pmean + 2.0 * pstd
    cross_max = _max_offdiag(M)
    return {
        'boards':                boards,
        'matrix':                M.tolist(),
        'cross_max':             float(cross_max),
        'noise_floor':           {'per_board': per_board_noise,
                                   'pooled_mean': pmean, 'pooled_std': pstd,
                                   'meaningful_threshold': threshold,
                                   'n_within_splits': int(n_within_splits)},
        'cross_board_meaningful': bool(cross_max > threshold),
        'n_categories':          int(len(categories)),
    }


def plot_empath_matrix(robustness: dict, out_path: Path) -> None:
    if not robustness.get('boards') or not robustness.get('matrix'):
        return
    import matplotlib.pyplot as plt
    boards = robustness['boards']
    M = np.asarray(robustness['matrix'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap='magma', vmin=0, vmax=max(0.05, float(M.max())))
    ax.set_xticks(range(len(boards))); ax.set_yticks(range(len(boards)))
    ax.set_xticklabels(boards, rotation=45, ha='right')
    ax.set_yticklabels(boards)
    for i in range(len(boards)):
        for j in range(len(boards)):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                    color='white' if M[i, j] > M.max() * 0.5 else 'black',
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        f'L1 over Empath ({robustness.get("n_categories", 200)} categories)')
    ax.set_title('Cross-board character divergence (Empath robustness)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sentence-BERT robustness check (round 1 §1.3)                                 #
# --------------------------------------------------------------------------- #

def sbert_robustness(decisions: List[dict],
                      model_name: str = 'all-MiniLM-L6-v2') -> dict:
    """Per-decision sentence-BERT embeddings, mean-pooled per board, pairwise
    cosine. The complement of cosine ('1 - cos') is reported alongside the
    primary L1 matrix; agreement of (high primary L1) with (high 1-cos) is
    the agreement check.

    Returns a dict shaped:
      {
        'boards': [...],
        'cosine_matrix':  [[float ...] ...],   # pairwise cosine, diagonal=1
        'note':           str (set if model fails to load),
        'model':          str,
      }
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as ex:
        return {'note': f'sentence-transformers unavailable '
                        f'({type(ex).__name__}: {ex})',
                'boards': [], 'cosine_matrix': [], 'model': model_name}

    corpora = per_board_corpus(decisions)
    boards = list(corpora.keys())
    if not boards:
        return {'boards': [], 'cosine_matrix': [], 'model': model_name}

    try:
        model = SentenceTransformer(model_name)
    except Exception as ex:
        return {'note': f'failed to load {model_name} '
                        f'({type(ex).__name__}: {ex})',
                'boards': boards, 'cosine_matrix': [], 'model': model_name}

    centroids: List[np.ndarray] = []
    for b in boards:
        rs = [r for r in corpora[b] if (r or '').strip()]
        if not rs:
            centroids.append(np.zeros(model.get_sentence_embedding_dimension(),
                                       dtype=np.float32))
            continue
        embs = model.encode(rs, batch_size=32, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=False)
        c = np.asarray(embs, dtype=np.float32).mean(axis=0)
        n = float(np.linalg.norm(c))
        if n > 0:
            c = c / n
        centroids.append(c)

    n = len(centroids)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            ci = centroids[i]; cj = centroids[j]
            if np.linalg.norm(ci) == 0 or np.linalg.norm(cj) == 0:
                M[i, j] = 0.0 if i != j else 1.0
            else:
                M[i, j] = float(np.dot(ci, cj))
    return {
        'boards':         boards,
        'cosine_matrix':  M.tolist(),
        'model':          model_name,
    }


def plot_sbert_matrix(robustness: dict, out_path: Path) -> None:
    if not robustness.get('boards') or not robustness.get('cosine_matrix'):
        return
    import matplotlib.pyplot as plt
    boards = robustness['boards']
    M = np.asarray(robustness['cosine_matrix'], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap='viridis', vmin=float(np.nanmin(M)),
                   vmax=float(np.nanmax(M)))
    ax.set_xticks(range(len(boards))); ax.set_yticks(range(len(boards)))
    ax.set_xticklabels(boards, rotation=45, ha='right')
    ax.set_yticklabels(boards)
    for i in range(len(boards)):
        for j in range(len(boards)):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                    color='white' if M[i, j] < 0.6 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        f'pairwise cosine (sentence-BERT, {robustness.get("model", "")})')
    ax.set_title('Cross-board character divergence (sBERT robustness)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--mini-config',      default='mini_config.yaml')
    ap.add_argument('--ga-2p', default='logs/optimizer/ga_2p.jsonl')
    ap.add_argument('--ga-3p', default='logs/optimizer/ga_3p.jsonl')
    ap.add_argument('--n-games',  type=int, default=12)
    ap.add_argument('--max-turns', type=int, default=80,
                    help='Per-game cap. 80 covers most LLM games while keeping '
                         'overnight wall-clock bounded.')
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--backend',
                    choices=('local', 'openai', 'heuristic', 'anthropic'),
                    default='local')
    ap.add_argument('--model', default=None,
                    help='Override LLM_MODEL env var. Default: Qwen2.5-1.5B.')
    ap.add_argument('--out-dir',   default='report/figures/llm_character')
    ap.add_argument('--decisions', default=None,
                    help='Path to decisions.jsonl. If omitted, uses '
                         '<out-dir>/decisions.jsonl.')
    ap.add_argument('--analyse-only', action='store_true',
                    help='Skip game collection; re-run analysis on an existing '
                         'decisions.jsonl. Useful for fast iteration.')
    ap.add_argument('--format-ok-threshold', type=float, default=0.80,
                    help='Per-board minimum format-compliance rate. Below this, '
                         'the divergence matrix is NOT rendered (the headline '
                         'finding would derive from a contaminated corpus).')
    ap.add_argument('--force-render', action='store_true',
                    help='Render divergence matrix even if a board falls below '
                         '--format-ok-threshold. Useful for writing up a '
                         'calibration finding ("the LLM at this scale produces '
                         'context-invariant narration").')
    ap.add_argument('--n-bootstrap', type=int, default=200,
                    help='Bootstrap resamples for the max-pair divergence CI.')
    ap.add_argument('--n-permutations', type=int, default=500,
                    help='Permutations for the H0(no-per-board-signal) test.')
    ap.add_argument('--n-within-splits', type=int, default=100,
                    help='Random equal-size splits per board for the within-board '
                         'L1 noise floor (ANALYSIS_PLAN §3).')
    ap.add_argument('--stat-seed', type=int, default=0,
                    help='Seed for both the bootstrap and the permutation test.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = Path(args.decisions) if args.decisions else (out_dir / 'decisions.jsonl')

    experiment = 'MIRROR-smoke' if args.backend == 'heuristic' else 'MIRROR-H'
    if args.analyse_only:
        experiment += '-analyse'
    with run_log.track(experiment=experiment, script=__file__,
                       args=vars(args), out_dir=out_dir) as run:
        if not args.analyse_only:
            run.note('collect: starting')
            run_collect(args, decisions_path)
            run.note('collect: done')

        if not decisions_path.exists():
            print(f'No decisions file at {decisions_path}; nothing to analyse.',
                  file=sys.stderr)
            sys.exit(1)
        run.note('analyse: starting')
        run_analyse(args, decisions_path, out_dir)
        run.note('analyse: done')


if __name__ == '__main__':
    main()
