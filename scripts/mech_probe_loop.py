"""EXP-004 exploration-mechanism probe driver (Qwen2.5-7B-Instruct).

Separates the three mechanisms EXP-003's "exploration bottleneck" bundles —
GENERATION / CREDIT / PRIOR — with the diagnostic feed held constant at MET.

Three modes (one per probe condition):

  peval       Condition A — declarative recognition. Show 4 candidate boards by
              STRUCTURE ONLY (salary + per-group cost/rent), realized share
              WITHHELD, and ask which best reaches g*=0.60. One forward pass per
              candidate ORDERING (the "seeds"); greedy decoding. Signal = fraction
              of orderings that pick B* (by board identity, not displayed letter).

  phold_good  Condition B — seed the MET editing loop FROM B* (already in-band).
              Does it preserve B* or edit it back out?

  phold_bad   Condition C — seed FROM the default (out-of-band high). Matched
              control + EXP-003 generation-failure replication. The behavioral
              recognition signal is the phold_good - phold_bad DIFFERENCE.

phold_* reuse scripts/llm_design_loop.run_trajectory + build_diagnostic_feed
UNCHANGED (condition='met'); board_label='default' for BOTH so they are
CRN-paired (identical per-game eval schedule, only the starting board differs).

Usage:
    # smoke (no GPU)
    set PYTHONPATH=. && python scripts/mech_probe_loop.py --mode peval \
        --backend heuristic --out-dir report/figures/exp4_smoke/peval
    set PYTHONPATH=. && python scripts/mech_probe_loop.py --mode phold_good \
        --backend heuristic --n-seeds 1 --K 3 --n-games 20 \
        --out-dir report/figures/exp4_smoke/phold_good

    # production (one mode, GPU)
    set PYTHONPATH=. && python scripts/mech_probe_loop.py --mode phold_good \
        --backend local --model Qwen/Qwen2.5-7B-Instruct \
        --n-seeds 5 --K 8 --n-games 200 --out-dir report/figures/exp4_run/phold_good
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from config import GameConfig
from optimizer.group_design import board_groups, apply_design, GroupDesign
from optimizer.skill_expression import (skill_share, render_goal_anchor,
                                        G_STAR, BAND)
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool
from prompts.loader import load_prompt

# Reuse the EXACT loop + feed builder EXP-003 used (do not fork them).
from scripts.llm_design_loop import (DesignerLLM, run_trajectory,
                                     _per_group_breakdown, _RESPONSE_BLOCK)

PEVAL_PROMPT = 'designer_llm_prompt_peval.txt'

# Candidate boards for P-EVAL. `true_share` is what WE know (hidden from the
# model); B* is the unique in-band / closest-to-g* answer. Re-measured under
# this harness (n=200, 5-seed mean) in the Step-0 gate.
PEVAL_CANDIDATES = [
    ('default', 'no rent/salary change',          1.000, 'rent'),
    ('bstar',   'uniform rent x0.30 (the fix)',    0.300, 'rent'),
    ('overcut', 'uniform rent x0.10 (over-cut)',   0.100, 'rent'),
    ('salary2', 'salary x2.0 (wrong lever)',       2.000, 'salary'),
]
PEVAL_TRUE_SHARE = {'default': 0.728, 'bstar': 0.611, 'overcut': 0.530, 'salary2': 0.803}
CORRECT_ID = 'bstar'


# --------------------------------------------------------------------------- #
# Board construction                                                            #
# --------------------------------------------------------------------------- #

def build_bstar(canonical_config: str = 'default_config.yaml',
                m: float = 0.30) -> GameConfig:
    """B* = default with a uniform rent multiplier m on ALL groups (the
    `_rent_variant` operator from exp_lever_check). m=0.30 lands skill_share
    in-band (0.611) and keeps the ladder graded (Step-0 gate)."""
    canon = GameConfig.from_yaml(canonical_config)
    groups = board_groups(canon)
    return apply_design(canon, GroupDesign(group_rent_mult={g: m for g in groups}),
                        strict_groups=False)


def _peval_cfg(cand_id: str, canon: GameConfig, groups: List[str]) -> GameConfig:
    if cand_id == 'default':
        return canon
    if cand_id == 'bstar':
        return apply_design(canon, GroupDesign(group_rent_mult={g: 0.30 for g in groups}),
                            strict_groups=False)
    if cand_id == 'overcut':
        return apply_design(canon, GroupDesign(group_rent_mult={g: 0.10 for g in groups}),
                            strict_groups=False)
    if cand_id == 'salary2':
        return apply_design(canon, GroupDesign(salary_mult=2.0), strict_groups=False)
    raise ValueError(f'unknown candidate {cand_id!r}')


# --------------------------------------------------------------------------- #
# P-EVAL (Condition A)                                                          #
# --------------------------------------------------------------------------- #

def render_board(cfg: GameConfig, letter: str) -> str:
    """Board STRUCTURE only: salary + per-group cost/rent. No realized share."""
    sal = cfg.settings.mechanics.salary
    rows = _per_group_breakdown(cfg)
    lines = [f'### BOARD {letter}',
             f'  salary (collected on passing Go): ${sal}',
             '  per-group cost/rent:']
    for r in rows:
        lines.append(f'   - {r["group"]:>10}: mean_cost=${r["mean_cost"]:.0f} '
                     f'mean_rent=${r["mean_rent"]:.0f}')
    return '\n'.join(lines)


def build_peval_feed(ordering: List[Tuple[str, GameConfig]], default_share: float) -> str:
    parts = ['## GOAL',
             '  ' + render_goal_anchor(default_share, G_STAR, 'reduce'),
             '',
             '## CANDIDATE BOARDS (realized skilled-vs-random share is NOT shown '
             '— judge it from the structure)']
    for letter, (_bid, cfg) in zip('ABCD', ordering):
        parts.append(render_board(cfg, letter))
    parts += ['',
              '## YOUR JOB',
              '  Pick the ONE board whose structure best reaches the GOAL '
              '(skilled-vs-random share closest to g*=0.60). '
              'Reply with the JSON schema from the system prompt.']
    return '\n'.join(parts)


def parse_peval_response(text: str) -> Tuple[Optional[str], str, str]:
    """Return (pick_letter|None, rationale, parser_status)."""
    cands = [m.group(1) for m in _RESPONSE_BLOCK.finditer(text)] or [text.strip()]
    parsed = None
    for c in cands:
        try:
            parsed = json.loads(c)
            break
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        return None, '', 'parser_failure'
    pick = str(parsed.get('pick', '')).strip().upper()[:1]
    rationale = str(parsed.get('rationale', '')).strip()
    if pick not in ('A', 'B', 'C', 'D'):
        return None, rationale, 'invalid_pick'
    return pick, rationale, 'ok'


@dataclass
class PEvalRecord:
    ordering:      List[str]      # board ids in displayed A,B,C,D order
    pick_letter:   Optional[str]
    pick_id:       Optional[str]
    picked_bstar:  bool
    rationale:     str
    parser_status: str
    parser_retry:  bool
    raw_response:  str
    true_share:    Optional[float]


def _peval_query(gen: Optional[DesignerLLM], feed: str, backend: str,
                 ordering_ids: List[str]) -> Tuple[str, bool]:
    """Returns (raw_text, retry_used). Heuristic backend returns a canned pick
    of whichever displayed board is B* (so the smoke test exercises the
    identity-mapping, not model quality)."""
    if backend == 'heuristic':
        letter = 'ABCD'[ordering_ids.index('bstar')]
        return ('```json\n' + json.dumps(
            {'pick': letter, 'rationale': '[heuristic] canned pick of B*'}) + '\n```'), False
    raw, _ = gen.query(feed)
    retry = False
    pick, _r, status = parse_peval_response(raw)
    if status != 'ok':                       # one format-reminder retry
        retry = True
        raw2, _ = gen.query(feed + '\n\nReply with EXACTLY one fenced ```json``` '
                                   'object: {"pick":"<letter>","rationale":"..."}.')
        if parse_peval_response(raw2)[2] == 'ok':
            raw = raw2
    return raw, retry


def run_peval(args, out_dir: Path) -> int:
    canon = GameConfig.from_yaml(args.canonical_config)
    groups = board_groups(canon)
    cfgs = {cid: _peval_cfg(cid, canon, groups) for cid, *_ in PEVAL_CANDIDATES}

    # Anchor uses the default's share measured under THIS harness (one seed, n).
    import hashlib
    def esf(seed):  # mirror llm_design_loop.eval_seed_for(board='default',seed,0,0)
        s = f'default|{seed}|0|0'.encode('utf-8')
        return int(hashlib.blake2s(s, digest_size=4).hexdigest(), 16) & 0xFFFFFFFF
    default_share = skill_share(cfgs['default'], n_seeds=args.n_games,
                                base_seed=esf(42))['share']

    gen = None
    if args.backend != 'heuristic':
        gen = DesignerLLM(backend=args.backend, model_name=args.model,
                          goal_disclosure='open')
        gen.system_prompt = load_prompt(PEVAL_PROMPT)   # hash-checked
        gen.gen_cfg['prompt_path'] = PEVAL_PROMPT

    ids = [c[0] for c in PEVAL_CANDIDATES]
    orderings = list(itertools.permutations(ids))
    if args.peval_orderings and args.peval_orderings < len(orderings):
        orderings = orderings[:args.peval_orderings]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'peval.jsonl'
    n_bstar = 0
    with open(out_path, 'w', encoding='utf-8') as fh:
        for order in orderings:
            ordering = [(bid, cfgs[bid]) for bid in order]
            feed = build_peval_feed(ordering, default_share)
            raw, retry = _peval_query(gen, feed, args.backend, list(order))
            pick, rationale, status = parse_peval_response(raw)
            pick_id = list(order)['ABCD'.index(pick)] if pick else None
            picked_bstar = (pick_id == CORRECT_ID)
            n_bstar += int(picked_bstar)
            rec = PEvalRecord(
                ordering=list(order), pick_letter=pick, pick_id=pick_id,
                picked_bstar=picked_bstar, rationale=rationale,
                parser_status=status, parser_retry=retry, raw_response=raw[:2000],
                true_share=(PEVAL_TRUE_SHARE.get(pick_id) if pick_id else None))
            fh.write(json.dumps(asdict(rec)) + '\n')
    frac = n_bstar / len(orderings) if orderings else 0.0
    (out_dir / 'PEVAL_SUMMARY.txt').write_text(
        f'default_anchor_share={default_share:.3f}\n'
        f'orderings={len(orderings)}\npicked_bstar={n_bstar}\nfrac_bstar={frac:.3f}\n'
        f'chance=0.25\n', encoding='utf-8')
    print(f'[peval] default_anchor_share={default_share:.3f}  '
          f'picked_bstar={n_bstar}/{len(orderings)} (frac={frac:.3f}, chance=0.25) '
          f'-> {out_path}')
    return 0


# --------------------------------------------------------------------------- #
# P-HOLD (Conditions B & C)                                                     #
# --------------------------------------------------------------------------- #

def run_phold(mode: str, args, out_dir: Path) -> int:
    canon = GameConfig.from_yaml(args.canonical_config)
    start = build_bstar(args.canonical_config, args.bstar_m) if mode == 'phold_good' else canon

    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                  n_matchups=args.n_matchups, seed=args.matchup_seed)
    designer = (DesignerLLM(backend=args.backend, model_name=args.model,
                            goal_disclosure='open')
                if args.backend != 'heuristic'
                else DesignerLLM(backend='heuristic', goal_disclosure='open'))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Sanity: confirm the seeded board is what we think (B* in-band / default high).
    base_share = skill_share(start, n_seeds=min(args.n_games, 200), base_seed=0)['share']
    inband = (G_STAR - BAND) <= base_share <= (G_STAR + BAND)
    print(f'[{mode}] seeded board base skill_share={base_share:.3f} '
          f'({"IN-BAND" if inband else "out-of-band"})  K={args.K} '
          f'seeds={args.n_seeds} n_games={args.n_games}')

    for s in range(args.n_seeds):
        seed = args.seed_offset + s * 1000 + 42
        out_path = out_dir / f'{mode}__default__seed{seed}.jsonl'
        # board_label='default' for BOTH modes -> CRN-paired B vs C.
        iters = run_trajectory(
            starting_cfg=start, board_label='default', seed=seed,
            pool=pool, matchups=matchups, n_games=args.n_games, K=args.K,
            condition='met', designer=designer, max_turns=args.max_turns,
            out_path=out_path)
        shares = [(it.metrics or {}).get('skill_share') for it in iters]
        print(f'  seed={seed}: shares=' +
              ' '.join(f'{x:.3f}' if isinstance(x, float) else 'NA' for x in shares))
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('peval', 'phold_good', 'phold_bad'),
                    required=True)
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--bstar-m', type=float, default=0.30,
                    help='Uniform rent multiplier defining B* (Step-0 gate: 0.30).')
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--pool', default='optimizer/strategy_pool.json')
    ap.add_argument('--peval-orderings', type=int, default=0,
                    help='0 = all 24 permutations; else first N.')
    ap.add_argument('--out-dir', default='report/figures/exp4_run')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.mode == 'peval':
        return run_peval(args, out_dir)
    return run_phold(args.mode, args, out_dir)


if __name__ == '__main__':
    raise SystemExit(main())
