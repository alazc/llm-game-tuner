"""EXP-007 — scripted closed-loop baselines on the Monopoly task (no LLM).

Three feedback-driven scripted policies run the SAME loop the LLM ran in
EXP-003 — same boards, seeds, K=8 budget, evaluator, CRN seed stream, and
Iteration record schema — so "is the band closed-loop reachable under the
budget?" is answered with the optimizer, not the task, swapped out:

  bisect      1-D bisection on the uniform rent multiplier (prior-informed
              upper bound: knows rent is the lever AND share is monotone in m).
  prior_ctrl  proportional controller using ONLY the prior the 7B itself
              states in EXP-004 ("lower rent -> smaller skill gap"): fixed
              x0.8 / x1.15 steps on a uniform rent multiplier.
  blind_cd    prior-free ratcheted coordinate probe: both directions on every
              knob family, keep what improves, exploit the best.

Policies see exactly the feedback scalar the LLM saw (share/score); they are
sealed verbatim in prereg EXP-007 — no gain
tuning after results.

Usage:
    # IO smoke (record format only; scores at this n are noise)
    set PYTHONPATH=. && python scripts/scripted_optimizer_loop.py \
        --policy bisect --boards default --n-seeds 1 --K 2 --n-games 20 \
        --out-dir report/figures/exp7_smoke

    # production (all policies, all boards, sealed config)
    set PYTHONPATH=. && python scripts/scripted_optimizer_loop.py \
        --policy all --out-dir report/figures/exp7_run

    # analysis vs the sealed thresholds (+ ladder gate on in-band finals)
    set PYTHONPATH=. && python scripts/scripted_optimizer_loop.py \
        --analyze --out-dir report/figures/exp7_run
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from config import GameConfig
from optimizer.exp_boards import build_exp0_boards
from optimizer.group_design import (GroupDesign, apply_design,
                                    bootstrap_score_ci, monotonicity_gate)
from optimizer import run_log
from optimizer.skill_expression import G_STAR, BAND
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool

# Reuse the EXACT loop building blocks EXP-003 used (do not fork them).
from scripts.llm_design_loop import (Iteration, _design_diff_summary,
                                     _eval_with_aligned_seeds, _is_improvement,
                                     _present_groups, eval_seed_for)

G_LO, G_HI = G_STAR - BAND, G_STAR + BAND
POLICIES = ('bisect', 'prior_ctrl', 'blind_cd')


def _uniform_rent(groups: List[str], m: float, rationale: str) -> GroupDesign:
    return GroupDesign(group_rent_mult={g: round(m, 6) for g in groups},
                       label='cumulative', rationale=rationale)


# --------------------------------------------------------------------------- #
# Policies (sealed verbatim; propose() sees only the last share/score)          #
# --------------------------------------------------------------------------- #

class BisectPolicy:
    """1-D bisection on uniform rent m over [0.10, 1.00] (share monotone in m)."""
    name = 'bisect'

    def __init__(self, groups: List[str]):
        self.groups = groups
        self.lo, self.hi = 0.10, 1.00
        self.m: Optional[float] = None          # None = baseline (m implicitly 1.0)

    def propose(self, share: float, score: float) -> Tuple[GroupDesign, str, bool]:
        if score == 0:
            m = 1.0 if self.m is None else self.m
            d = (_uniform_rent(self.groups, m, '[T-BISECT] in-band; hold')
                 if self.m is not None else GroupDesign(label='cumulative',
                                                        rationale='[T-BISECT] in-band at baseline'))
            return d, d.rationale, True
        probed = 1.0 if self.m is None else self.m
        if share > G_HI:
            self.hi = min(self.hi, probed)
        elif share < G_LO:
            self.lo = max(self.lo, probed)
        if self.lo >= self.hi:                   # bracket degenerate (board below band)
            self.hi = min(4.0, max(2.0, self.lo * 2))
        self.m = (self.lo + self.hi) / 2.0
        rat = f'[T-BISECT] probe m={self.m:.3f} bracket=[{self.lo:.3f},{self.hi:.3f}]'
        return _uniform_rent(self.groups, self.m, rat), rat, False


class PriorCtrlPolicy:
    """Proportional controller with the 7B's own stated prior: rent down to
    reduce the skill edge. Fixed x0.8 / x1.15 steps; no magnitude knowledge."""
    name = 'prior_ctrl'

    def __init__(self, groups: List[str]):
        self.groups = groups
        self.m = 1.0

    def propose(self, share: float, score: float) -> Tuple[GroupDesign, str, bool]:
        if score == 0:
            d = (_uniform_rent(self.groups, self.m, '[T-PRIOR-CTRL] in-band; hold')
                 if self.m != 1.0 else GroupDesign(label='cumulative',
                                                   rationale='[T-PRIOR-CTRL] in-band at baseline'))
            return d, d.rationale, True
        if share > G_HI:
            self.m = max(0.25, self.m * 0.8)
        elif share < G_LO:
            self.m = min(2.0, self.m * 1.15)
        rat = f'[T-PRIOR-CTRL] rent_all m={self.m:.3f}'
        return _uniform_rent(self.groups, self.m, rat), rat, False


class BlindCDPolicy:
    """Prior-free ratcheted coordinate probe. Fixed order, both directions on
    every family; keep strict improvements; iterations 7-8 exploit the single
    most-improving accepted probe (compounding)."""
    name = 'blind_cd'
    PROBES = (('rent', 0.7), ('rent', 1.4), ('cost', 0.7), ('cost', 1.4),
              ('salary', 1.5), ('salary', 0.67))
    CLAMP = {'rent': (0.25, 4.0), 'cost': (0.25, 4.0), 'salary': (0.5, 3.0)}

    def __init__(self, groups: List[str]):
        self.groups = groups
        self.best = {'rent': 1.0, 'cost': 1.0, 'salary': 1.0}
        self.best_score: Optional[float] = None
        self.pending: Optional[dict] = None     # values just probed
        self.probe_i = 0
        self.best_probe: Optional[int] = None   # index of most-improving probe
        self.best_imp = 0.0

    def _design(self, vals: dict, rationale: str) -> GroupDesign:
        return GroupDesign(
            salary_mult=round(vals['salary'], 6),
            group_rent_mult={g: round(vals['rent'], 6) for g in self.groups},
            group_cost_mult={g: round(vals['cost'], 6) for g in self.groups},
            label='cumulative', rationale=rationale)

    def _clamped(self, fam: str, v: float) -> float:
        lo, hi = self.CLAMP[fam]
        return max(lo, min(hi, v))

    def propose(self, share: float, score: float) -> Tuple[GroupDesign, str, bool]:
        if self.best_score is None:              # first call: score is the baseline
            self.best_score = score
        elif self.pending is not None:           # settle the probe just evaluated
            if score < self.best_score - 1e-12:
                imp = self.best_score - score
                self.best = self.pending
                self.best_score = score
                last_idx = self.probe_i - 1 if self.probe_i <= len(self.PROBES) else self.best_probe
                if imp > self.best_imp and last_idx is not None:
                    self.best_imp, self.best_probe = imp, last_idx
            self.pending = None

        if self.best_score == 0:
            d = self._design(self.best, '[T-BLIND-CD] in-band; hold')
            return d, d.rationale, True

        if self.probe_i < len(self.PROBES):
            fam, f = self.PROBES[self.probe_i]
            self.probe_i += 1
        elif self.best_probe is not None:        # exploit phase
            fam, f = self.PROBES[self.best_probe]
            self.probe_i += 1
        else:
            d = self._design(self.best, '[T-BLIND-CD] no improving probe; hold best')
            return d, d.rationale, False

        vals = dict(self.best)
        vals[fam] = self._clamped(fam, vals[fam] * f)
        self.pending = vals
        rat = f'[T-BLIND-CD] probe {fam} x{f} -> {vals[fam]:.3f}'
        return self._design(vals, rat), rat, False


POLICY_CLS = {'bisect': BisectPolicy, 'prior_ctrl': PriorCtrlPolicy,
              'blind_cd': BlindCDPolicy}


# --------------------------------------------------------------------------- #
# Trajectory runner (mirrors run_trajectory's records; policy-driven)            #
# --------------------------------------------------------------------------- #

def run_scripted_trajectory(starting_cfg: GameConfig, board_label: str,
                            seed: int, pool, matchups, n_games: int, K: int,
                            policy, max_turns: int, out_path: Path) -> List[Iteration]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    iterations: List[Iteration] = []
    gen_cfg = {'engine': 'scripted', 'policy': policy.name,
               'note': 'no LLM call; sealed policy (prereg EXP-007)'}

    cumulative = GroupDesign(label='cumulative')
    cfg = starting_cfg

    t0 = time.perf_counter()
    ev = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                  board_label, seed, 0, max_turns)
    ci = bootstrap_score_ci(ev['shares'], n_resamples=1500, seed=seed)
    rec0 = Iteration(
        iter=0, design=cumulative.to_dict(),
        design_diff=_design_diff_summary(cumulative),
        rationale='[baseline]', parser_status='baseline', parser_error=None,
        converged_request=False, convergence_padded=False,
        convergence_violation=False, parser_retry=False,
        score=ev['score'], metrics=ev['metrics'], score_ci=ci,
        delta_vs_prev=None, improvement=None,
        n_games=ev['n_games_total'], token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='',
        condition=policy.name, gen_cfg=gen_cfg,
        eval_seed_base=eval_seed_for(board_label, seed, 0, 0),
        skill_curve=ev.get('skill_curve'))
    iterations.append(rec0)
    fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()
    prev_score = ev['score']

    converged_done = False
    pad_design = cumulative

    for k in range(1, K + 1):
        t0 = time.perf_counter()
        ev_seed = eval_seed_for(board_label, seed, k, 0)
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
                condition=policy.name, gen_cfg=gen_cfg, eval_seed_base=ev_seed)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            continue

        last = iterations[-1]
        share = (last.metrics or {}).get('skill_share')
        design, rationale, conv = policy.propose(share, last.score)
        cumulative = design
        cfg = apply_design(starting_cfg, cumulative, strict_groups=False)

        ev = _eval_with_aligned_seeds(cfg, pool, matchups, n_games,
                                      board_label, seed, k, max_turns)
        ci = bootstrap_score_ci(ev['shares'], n_resamples=1500, seed=seed + k)
        delta = ev['score'] - prev_score
        rec = Iteration(
            iter=k, design=cumulative.to_dict(),
            design_diff=_design_diff_summary(cumulative),
            rationale=rationale, parser_status='ok', parser_error=None,
            converged_request=conv, convergence_padded=False,
            convergence_violation=False, parser_retry=False,
            score=ev['score'], metrics=ev['metrics'], score_ci=ci,
            delta_vs_prev=delta,
            improvement=_is_improvement(prev_score, ev['score'], ci),
            n_games=ev['n_games_total'], token_count=None,
            wall_seconds=time.perf_counter() - t0, raw_response='',
            condition=policy.name, gen_cfg=gen_cfg, eval_seed_base=ev_seed,
            skill_curve=ev.get('skill_curve'))
        iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
        fh.flush()
        prev_score = ev['score']
        if conv:
            converged_done = True
            pad_design = cumulative

    fh.close()
    return iterations


# --------------------------------------------------------------------------- #
# Analysis vs the sealed thresholds                                              #
# --------------------------------------------------------------------------- #

def _load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def analyze(out_dir: Path, gate_n: int = 120) -> int:
    lines: List[str] = []

    def say(s=''):
        print(s); lines.append(s)

    cells = {}                                   # (policy, board, seed) -> stats
    for pol in POLICIES:
        for p in sorted(glob.glob(str(out_dir / pol / '*.jsonl'))):
            recs = _load(p)
            name = Path(p).stem                  # pol__tag__seedNNNN
            mm = re.match(rf'{pol}__(.+)__seed(\d+)', name)
            if not mm:
                continue
            board, seed = mm.group(1), int(mm.group(2))
            scores = [r['score'] for r in recs]
            shares = [(r.get('metrics') or {}).get('skill_share') for r in recs]
            first_inband = next((r['iter'] for r in recs
                                 if r['iter'] >= 1 and r['score'] == 0), None)
            n_evals = sum(1 for r in recs if r['n_games'] > 0)
            cells[(pol, board, seed)] = {
                'final_score': scores[-1], 'final_share': shares[-1],
                'final_inband': scores[-1] == 0, 'first_inband': first_inband,
                'n_evals': n_evals, 'final_design': recs[-1]['design'],
                'shares': shares,
            }

    say('=== EXP-007 scripted closed-loop baselines :: ' + str(out_dir) + ' ===')
    verdicts = {}
    for pol in POLICIES:
        sub = {k: v for k, v in cells.items() if k[0] == pol}
        if not sub:
            say(f'\n--- {pol}: NO DATA'); continue
        n_in = sum(v['final_inband'] for v in sub.values())
        say(f'\n--- {pol}:  final in-band {n_in}/{len(sub)}')
        for (p_, b, s), v in sorted(sub.items()):
            fi = v['first_inband']
            say(f'   {b:>14} seed={s}: final share={v["final_share"]:.3f} '
                f'score={v["final_score"]:.4f}  in-band={"Y" if v["final_inband"] else "n"}'
                f'  first-in-band-iter={fi if fi is not None else "-"}'
                f'  evals={v["n_evals"]}')
        med = sorted(v['first_inband'] for v in sub.values()
                     if v['first_inband'] is not None)
        med_txt = med[len(med) // 2] if med else None
        say(f'   median edits-to-band: {med_txt}')
        verdicts[pol] = (n_in, len(sub), med_txt)

    # Ladder gate on UNIQUE in-band final boards.
    say('\n--- METRIC-2 ladder gate on in-band finals (n_seeds=%d) ---' % gate_n)
    seen = {}
    gate_fail = []
    for (pol, board, seed), v in sorted(cells.items()):
        if not v['final_inband']:
            continue
        key = json.dumps(v['final_design'], sort_keys=True) + '|' + board
        if key in seen:
            res = seen[key]
        else:
            labels = {re.sub(r'[^A-Za-z0-9]+', '_', l).strip('_'): c
                      for l, c in build_exp0_boards()}
            cfg = apply_design(labels[board],
                               GroupDesign.from_dict(v['final_design']),
                               strict_groups=False)
            res = monotonicity_gate(cfg, n_seeds=gate_n, base_seed=0)
            seen[key] = res
        ok = res['passes']
        if not ok:
            gate_fail.append((pol, board, seed))
        say(f'   {pol:>10} {board:>14} seed={seed}: rank_corr={res["rank_corr"]:+.2f} '
            f'{"PASS" if ok else "FAIL"}  standings=' +
            ' '.join(f'{k}={x:.3f}' for k, x in res['standings'].items()))

    # Sealed-threshold verdicts (prereg EXP-007).
    say('\n--- verdicts vs sealed predictions ---')
    b = verdicts.get('bisect'); pc = verdicts.get('prior_ctrl'); bc = verdicts.get('blind_cd')
    if b:
        say(f'  T-BISECT     in-band {b[0]}/{b[1]} (sealed: >=8/9)  '
            f'{"MET" if b[0] >= 8 else "NOT MET"}; median-to-band {b[2]} (sealed <=4)')
    if pc:
        say(f'  T-PRIOR-CTRL in-band {pc[0]}/{pc[1]} (sealed: >=7/9)  '
            f'{"MET" if pc[0] >= 7 else "NOT MET"}; median-to-band {pc[2]} (sealed <=7)')
    if bc:
        say(f'  T-BLIND-CD   in-band {bc[0]}/{bc[1]} (sealed call: >=3/9; strongest-framing >=5/9)')
    if pc:
        if pc[0] >= 7:
            cls = ('SOLVABLE-BLIND (mechanical floor above the 7B)'
                   if bc and bc[0] >= 5 else 'SOLVABLE-WITH-PRIOR')
            say(f'  CLASSIFICATION: {cls}')
            say('  -> the band IS closed-loop reachable under K=8; the EXP-003 failure is the model\'s.')
        else:
            say('  CLASSIFICATION: NOT-SOLVABLE under K=8 at sealed gains -> headline narrows; EXP-009 BLOCKED.')
    if gate_fail:
        say(f'  NOTE: {len(gate_fail)} in-band final(s) FAILED the ladder gate -> excluded: {gate_fail}')

    (out_dir / 'EXP7_REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    say(f'\nwrote {out_dir / "EXP7_REPORT.txt"}')
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=POLICIES + ('all',), default='all')
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
    ap.add_argument('--out-dir', default='report/figures/exp7_run')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--gate-n', type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.analyze:
        return analyze(out_dir, gate_n=args.gate_n)

    out_dir.mkdir(parents=True, exist_ok=True)
    policies = list(POLICIES) if args.policy == 'all' else [args.policy]

    starting = list(build_exp0_boards(canonical_config=args.canonical_config))
    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        starting = [(l, c) for l, c in starting if l in wanted]

    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                  n_matchups=args.n_matchups,
                                  seed=args.matchup_seed)

    with run_log.track(experiment='EXP-007-scripted', script=__file__,
                       args=vars(args), out_dir=out_dir,
                       condition=','.join(policies)) as run:
        for pol_name in policies:
            for board_label, cfg in starting:
                tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
                for s in range(args.n_seeds):
                    seed = args.seed_offset + s * 1000 + 42
                    out_path = out_dir / pol_name / f'{pol_name}__{tag}__seed{seed}.jsonl'
                    policy = POLICY_CLS[pol_name](_present_groups(cfg))
                    print(f'\n[{board_label}|{pol_name}] seed={seed} K={args.K}')
                    run.note(f'{board_label}|{pol_name}|seed={seed}: starting')
                    iters = run_scripted_trajectory(
                        starting_cfg=cfg, board_label=board_label, seed=seed,
                        pool=pool, matchups=matchups, n_games=args.n_games,
                        K=args.K, policy=policy, max_turns=args.max_turns,
                        out_path=out_path)
                    for it in iters:
                        sh = (it.metrics or {}).get('skill_share')
                        sh_txt = f'{sh:.3f}' if isinstance(sh, float) else 'NA'
                        pad = ' [PAD]' if it.convergence_padded else ''
                        cv = ' (converged)' if it.converged_request else ''
                        print(f'    iter {it.iter}: share={sh_txt} '
                              f'score={it.score:.4f}{cv}{pad}  {it.rationale[:60]}')
                    run.note(f'{board_label}|{pol_name}|seed={seed}: done '
                             f'final={iters[-1].score}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
