"""EXP-009 — scaffolded designer: R1 (sample-N + simulator argmin + ratchet)
and R3 (harness stop) on the EXP-003 Monopoly task.

Role redistribution, model unchanged: per iteration the 7B PROPOSES N=22
candidate edits at tau=0.4 from the SAME met feed the plain loop shows; the
HARNESS evaluates every parsed candidate at the iteration's CRN seed, takes
the argmin, ratchets against best-so-far (reverting re-evaluates the held
design at this iteration's seed so the trajectory stays paired), and stops on
score == 0 (the model's `converged` flag is ignored). Parameters bound from
EXP-007/008 read-outs; sealed in prereg EXP-009.

Usage:
    # heuristic IO smoke (no GPU; record format + ratchet/stop plumbing)
    set PYTHONPATH=. && python scripts/scaffold_designer_loop.py \
        --backend heuristic --boards default --n-seeds 1 --K 2 \
        --n-candidates 3 --n-games 20 --workers 2 \
        --out-dir report/figures/exp9_smoke

    # production (one board; via modal_exp9.py)
    set PYTHONPATH=. && python scripts/scaffold_designer_loop.py \
        --backend local --model Qwen/Qwen2.5-7B-Instruct \
        --boards default --n-seeds 3 --out-dir report/figures/exp9_run

    # analysis vs sealed thresholds (after pulling all boards)
    set PYTHONPATH=. && python scripts/scaffold_designer_loop.py \
        --analyze --out-dir report/figures/exp9_run
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
from typing import List, Optional

from optimizer.exp_boards import build_exp0_boards
from optimizer.group_design import (GroupDesign, apply_design,
                                    bootstrap_score_ci, monotonicity_gate)
from optimizer import run_log
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool

# Reuse the EXACT EXP-003 pieces (do not fork).
from scripts.llm_design_loop import (DesignerLLM, Iteration,
                                     _design_diff_summary, _merge_designs,
                                     _eval_with_aligned_seeds, eval_seed_for,
                                     build_diagnostic_feed,
                                     parse_designer_response)
from scripts.proposal_probe import sample_batch

CONDITION = 'met'                      # the strongest LLM channel from EXP-003


# --------------------------------------------------------------------------- #
# Candidate evaluation worker (process pool; no torch in this path)              #
# --------------------------------------------------------------------------- #

_WORKER_BOARDS = None


def _worker_eval(task: dict) -> dict:
    """Evaluate one candidate design dict at the given CRN seed. Children
    rebuild boards once (module-global cache) and never touch torch."""
    global _WORKER_BOARDS
    if _WORKER_BOARDS is None:
        _WORKER_BOARDS = {re.sub(r'[^A-Za-z0-9]+', '_', l).strip('_'): c
                          for l, c in build_exp0_boards(
                              canonical_config=task['canonical_config'])}
    from optimizer.group_design import evaluate_config
    cfg = apply_design(_WORKER_BOARDS[task['board_tag']],
                       GroupDesign.from_dict(task['design']),
                       strict_groups=False)
    ev = evaluate_config(cfg, None, None, task['n_games'],
                         base_seed=task['base_seed'],
                         max_turns=task['max_turns'], record_trajectory=True)
    return {'idx': task['idx'], 'score': ev['score'],
            'skill_share': ev['skill_share'], 'metrics': ev['metrics'],
            'shares': ev['shares'], 'skill_curve': ev.get('skill_curve')}


# --------------------------------------------------------------------------- #
# Scaffolded trajectory                                                         #
# --------------------------------------------------------------------------- #

def run_scaffold_trajectory(starting_cfg, board_label: str, board_tag: str,
                            seed: int, pool, matchups, args, designer,
                            executor, out_path: Path) -> List[Iteration]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    iterations: List[Iteration] = []
    n_games, K, max_turns = args.n_games, args.K, args.max_turns
    gen_base = {'engine': 'scaffold', 'model': designer.model_name,
                'n_candidates': args.n_candidates,
                'temperature': args.temperature, 'condition': CONDITION,
                'accept_rule': args.accept_rule,
                'note': ('R1 sample-N+argmin+ratchet, R3 harness stop '
                         '(prereg EXP-009 historical / EXP-010 paired)')}

    t0 = time.perf_counter()
    ev = _eval_with_aligned_seeds(starting_cfg, pool, matchups, n_games,
                                  board_label, seed, 0, max_turns)
    ci = bootstrap_score_ci(ev['shares'], n_resamples=1500, seed=seed)
    best_design = GroupDesign(label='cumulative')
    best_score = ev['score']
    accepted_eval = {'metrics': ev['metrics'], 'score': ev['score'],
                     'skill_curve': ev.get('skill_curve')}
    rec0 = Iteration(
        iter=0, design=best_design.to_dict(),
        design_diff=_design_diff_summary(best_design),
        rationale='[baseline]', parser_status='baseline', parser_error=None,
        converged_request=False, convergence_padded=False,
        convergence_violation=False, parser_retry=False,
        score=ev['score'], metrics=ev['metrics'], score_ci=ci,
        delta_vs_prev=None, improvement=None,
        n_games=ev['n_games_total'], token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='',
        condition='scaffold', gen_cfg=gen_base,
        eval_seed_base=eval_seed_for(board_label, seed, 0, 0),
        skill_curve=ev.get('skill_curve'))
    iterations.append(rec0)
    fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()

    prior_iters = [{'iter': 0, 'design_diff': rec0.design_diff,
                    'rationale': rec0.rationale, 'score': rec0.score,
                    'metrics': rec0.metrics}]
    stopped = False

    for k in range(1, K + 1):
        t0 = time.perf_counter()
        ev_seed = eval_seed_for(board_label, seed, k, 0)
        if stopped:
            rec = Iteration(
                iter=k, design=best_design.to_dict(),
                design_diff=_design_diff_summary(best_design),
                rationale='[harness-stop: carry-forward]', parser_status='ok',
                parser_error=None, converged_request=True,
                convergence_padded=True, convergence_violation=False,
                parser_retry=False,
                score=iterations[-1].score, metrics=iterations[-1].metrics,
                score_ci=iterations[-1].score_ci,
                delta_vs_prev=0.0, improvement=False, n_games=0,
                token_count=None, wall_seconds=time.perf_counter() - t0,
                raw_response='', condition='scaffold', gen_cfg=gen_base,
                eval_seed_base=ev_seed)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            continue

        cfg_now = apply_design(starting_cfg, best_design, strict_groups=False)
        feed = build_diagnostic_feed(cfg_now, best_design,
                                     eval_out=accepted_eval, prior_eval=None,
                                     condition=CONDITION,
                                     prior_iters=prior_iters)
        batch = sample_batch(designer, feed, args.n_candidates,
                             args.temperature,
                             base_seed=eval_seed_for(board_label, seed, k, 1),
                             chunk=args.chunk)
        parsed = []
        for i, b in enumerate(batch):
            resp = parse_designer_response(b['raw'])
            if resp.parser_status == 'ok' and resp.design is not None:
                parsed.append((i, resp))
        n_parsed = len(parsed)

        gen_cfg = dict(gen_base, n_parsed=n_parsed)
        if not parsed:                              # nothing usable: carry forward
            rec = Iteration(
                iter=k, design=best_design.to_dict(),
                design_diff=_design_diff_summary(best_design),
                rationale='[no parsed candidates: carry-forward]',
                parser_status='parser_failure', parser_error='0 parsed of batch',
                converged_request=False, convergence_padded=False,
                convergence_violation=False, parser_retry=False,
                score=iterations[-1].score, metrics=iterations[-1].metrics,
                score_ci=iterations[-1].score_ci, delta_vs_prev=0.0,
                improvement=False, n_games=0, token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition='scaffold', gen_cfg=gen_cfg, eval_seed_base=ev_seed)
            iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            fh.flush()
            continue

        merged = [(i, resp, _merge_designs(best_design, resp.design))
                  for i, resp in parsed]
        tasks = [{'idx': j, 'design': m.to_dict(), 'board_tag': board_tag,
                  'base_seed': ev_seed, 'n_games': n_games,
                  'max_turns': max_turns,
                  'canonical_config': args.canonical_config}
                 for j, (_i, _r, m) in enumerate(merged)]
        if args.accept_rule == 'paired':
            # EXP-010 rule: the held design is re-evaluated at THIS iteration's
            # seed (always), and acceptance is the CRN-paired comparison
            # argmin < held@same-seed. No historical bar is kept.
            tasks.append({'idx': len(merged), 'design': best_design.to_dict(),
                          'board_tag': board_tag, 'base_seed': ev_seed,
                          'n_games': n_games, 'max_turns': max_turns,
                          'canonical_config': args.canonical_config})
        results = {r['idx']: r for r in executor.map(_worker_eval, tasks)}
        scores = [results[j]['score'] for j in range(len(merged))]
        j_min = min(range(len(merged)), key=lambda j: scores[j])
        argmin_score = scores[j_min]
        if args.accept_rule == 'paired':
            held = results[len(merged)]
            bar = held['score']
        else:                                       # EXP-009 historical rule
            held = None
            bar = best_score
        n_improving = sum(1 for s in scores if s < bar)

        if argmin_score < bar:                      # ACCEPT
            _i, resp, best_design = merged[j_min]
            best_score = argmin_score
            evj = results[j_min]
            accepted_eval = {'metrics': evj['metrics'], 'score': evj['score'],
                             'skill_curve': evj.get('skill_curve')}
            ci = bootstrap_score_ci(evj['shares'], n_resamples=1500,
                                    seed=seed + k)
            rationale = f'[scaffold-accept argmin {argmin_score:.4f}] ' + resp.rationale[:140]
            rec_score, rec_metrics = evj['score'], evj['metrics']
            rec_curve = evj.get('skill_curve')
            reverted = False
        else:                                       # RATCHET: revert to held/best
            if held is not None:                    # paired rule: already evaluated
                ev_hold = held
            else:                                   # historical rule: evaluate now
                ev_full = _eval_with_aligned_seeds(
                    apply_design(starting_cfg, best_design, strict_groups=False),
                    pool, matchups, n_games, board_label, seed, k, max_turns)
                ev_hold = {'score': ev_full['score'],
                           'skill_share': ev_full['skill_share'],
                           'metrics': ev_full['metrics'],
                           'shares': ev_full['shares'],
                           'skill_curve': ev_full.get('skill_curve')}
            ci = bootstrap_score_ci(ev_hold['shares'], n_resamples=1500,
                                    seed=seed + k)
            accepted_eval = {'metrics': ev_hold['metrics'],
                             'score': ev_hold['score'],
                             'skill_curve': ev_hold.get('skill_curve')}
            if args.accept_rule == 'paired':
                best_score = ev_hold['score']       # fresh bar; no historical min
            else:
                best_score = min(best_score, ev_hold['score'])
            rationale = (f'[ratchet: reverted; argmin {argmin_score:.4f} '
                         f'>= bar {bar:.4f}]')
            rec_score, rec_metrics = ev_hold['score'], ev_hold['metrics']
            rec_curve = ev_hold.get('skill_curve')
            reverted = True

        gen_cfg.update(argmin_score=argmin_score, n_improving=n_improving,
                       reverted=reverted, accept_bar=bar)
        stop_now = (rec_score == 0)
        prev_score = iterations[-1].score
        rec = Iteration(
            iter=k, design=best_design.to_dict(),
            design_diff=_design_diff_summary(best_design),
            rationale=rationale, parser_status='ok', parser_error=None,
            converged_request=stop_now, convergence_padded=False,
            convergence_violation=False, parser_retry=False,
            score=rec_score, metrics=rec_metrics, score_ci=ci,
            delta_vs_prev=(rec_score - prev_score if prev_score is not None else None),
            improvement=bool(rec_score < prev_score) if prev_score is not None else None,
            n_games=n_games * (len(merged) + (1 if reverted else 0)),
            token_count=None, wall_seconds=time.perf_counter() - t0,
            raw_response='', condition='scaffold', gen_cfg=gen_cfg,
            eval_seed_base=ev_seed, skill_curve=rec_curve)
        iterations.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
        fh.flush()
        prior_iters.append({'iter': k, 'design_diff': rec.design_diff,
                            'rationale': rec.rationale, 'score': rec.score,
                            'metrics': rec.metrics})
        if stop_now:
            stopped = True

    fh.close()
    return iterations


# --------------------------------------------------------------------------- #
# Analysis vs the sealed thresholds                                              #
# --------------------------------------------------------------------------- #

def analyze(out_dir: Path, gate_n: int = 120) -> int:
    lines: List[str] = []

    def say(s=''):
        print(s); lines.append(s)

    cells = {}
    for p in sorted(glob.glob(str(out_dir / 'scaffold__*.jsonl'))):
        recs = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        mm = re.match(r'scaffold__(.+)__seed(\d+)', Path(p).stem)
        if not mm:
            continue
        board, seed = mm.group(1), int(mm.group(2))
        first_inband = next((r['iter'] for r in recs
                             if r['iter'] >= 1 and r['score'] == 0), None)
        live = [r for r in recs if r['iter'] >= 1 and not r['convergence_padded']]
        cells[(board, seed)] = {
            'final_score': recs[-1]['score'],
            'final_share': (recs[-1].get('metrics') or {}).get('skill_share'),
            'final_inband': recs[-1]['score'] == 0,
            'first_inband': first_inband,
            'n_reverts': sum(1 for r in live if (r.get('gen_cfg') or {}).get('reverted')),
            'n_live': len(live),
            'total_cand_evals': sum(r['n_games'] for r in recs) // 200,
            'final_design': recs[-1]['design'],
        }

    say(f'=== EXP-009 scaffolded designer :: {out_dir} ===')
    n_in = sum(v['final_inband'] for v in cells.values())
    for (b, s), v in sorted(cells.items()):
        say(f'  {b:>14} seed={s}: final share={v["final_share"]:.3f} '
            f'score={v["final_score"]:.4f} in-band={"Y" if v["final_inband"] else "n"} '
            f'first-in-band={v["first_inband"] if v["first_inband"] is not None else "-"} '
            f'reverts={v["n_reverts"]}/{v["n_live"]} evals~{v["total_cand_evals"]}')
    med = sorted(v['first_inband'] for v in cells.values()
                 if v['first_inband'] is not None)
    med_txt = med[len(med) // 2] if med else None
    rev = sum(v['n_reverts'] for v in cells.values())
    liv = sum(v['n_live'] for v in cells.values())
    say(f'\n  final in-band {n_in}/{len(cells)}  median-edits-to-band {med_txt}  '
        f'revert-rate {rev}/{liv}')

    say(f'\n--- METRIC-2 ladder gate on in-band finals (n_seeds={gate_n}) ---')
    labels = {re.sub(r'[^A-Za-z0-9]+', '_', l).strip('_'): c
              for l, c in build_exp0_boards()}
    n_gate_fail, seen = 0, {}
    for (b, s), v in sorted(cells.items()):
        if not v['final_inband']:
            continue
        key = json.dumps(v['final_design'], sort_keys=True) + '|' + b
        if key not in seen:
            cfg = apply_design(labels[b], GroupDesign.from_dict(v['final_design']),
                               strict_groups=False)
            seen[key] = monotonicity_gate(cfg, n_seeds=gate_n, base_seed=0)
        res = seen[key]
        n_gate_fail += int(not res['passes'])
        say(f'  {b:>14} seed={s}: rank_corr={res["rank_corr"]:+.2f} '
            f'{"PASS" if res["passes"] else "FAIL"}')

    n_valid = n_in - n_gate_fail
    say('\n--- verdicts vs sealed predictions (prereg EXP-009) ---')
    say(f'  PRIMARY in-band(valid) {n_valid}/{len(cells)} (sealed >=5/9): '
        f'{"MET" if n_valid >= 5 else "NOT MET"}')
    say(f'  median edits-to-band {med_txt} (sealed <=7 among landers)')
    say(f'  revert rate {rev}/{liv} (sealed <25%): '
        f'{"MET" if liv and rev / liv < 0.25 else "NOT MET"}')
    if n_valid >= 5:
        say('  -> R1+R3 SUFFICE: the plain loop\'s 0/9 was selection/commitment + stopping,')
        say('     not proposal content. (vs EXP-003 met 0/9, CRN-paired.)')
    elif n_valid <= 1:
        say('  -> FALSIFIER: selection fix INSUFFICIENT at loop depth (generation-under-recursion).')
    else:
        say('  -> PARTIAL: report per-cell paired delta vs EXP-003 met (0/9); no full claim.')

    (out_dir / 'EXP9_REPORT.txt').write_text('\n'.join(lines) + '\n',
                                             encoding='utf-8')
    say(f'\nwrote {out_dir / "EXP9_REPORT.txt"}')
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--boards', default='all')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-candidates', type=int, default=22)
    ap.add_argument('--temperature', type=float, default=0.4)
    ap.add_argument('--accept-rule', choices=('historical', 'paired'),
                    default='historical',
                    help='historical = EXP-009 (best-so-far bar); '
                         'paired = EXP-010 (held re-eval at the same seed).')
    ap.add_argument('--chunk', type=int, default=22)
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--pool', default='optimizer/strategy_pool.json')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-dir', default='report/figures/exp9_run')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--gate-n', type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.analyze:
        return analyze(out_dir, gate_n=args.gate_n)
    out_dir.mkdir(parents=True, exist_ok=True)

    starting = list(build_exp0_boards(canonical_config=args.canonical_config))
    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        starting = [(l, c) for l, c in starting if l in wanted]

    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                  n_matchups=args.n_matchups,
                                  seed=args.matchup_seed)

    # Pool BEFORE any torch/CUDA init (fork safety on Linux).
    executor = ProcessPoolExecutor(max_workers=args.workers)
    designer = DesignerLLM(backend=args.backend, model_name=args.model,
                           goal_disclosure='open')

    with run_log.track(experiment='EXP-009-scaffold', script=__file__,
                       args=vars(args), out_dir=out_dir,
                       condition='scaffold') as run:
        for board_label, cfg in starting:
            tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
            for s in range(args.n_seeds):
                seed = args.seed_offset + s * 1000 + 42
                out_path = out_dir / f'scaffold__{tag}__seed{seed}.jsonl'
                print(f'\n[{board_label}|scaffold] seed={seed} K={args.K} '
                      f'N={args.n_candidates} T={args.temperature}')
                run.note(f'{board_label}|scaffold|seed={seed}: starting')
                iters = run_scaffold_trajectory(
                    starting_cfg=cfg, board_label=board_label, board_tag=tag,
                    seed=seed, pool=pool, matchups=matchups, args=args,
                    designer=designer, executor=executor, out_path=out_path)
                for it in iters:
                    sh = (it.metrics or {}).get('skill_share')
                    sh_txt = f'{sh:.3f}' if isinstance(sh, float) else 'NA'
                    g = it.gen_cfg or {}
                    extra = (f' parsed={g.get("n_parsed")} improving={g.get("n_improving")}'
                             f'{" REVERT" if g.get("reverted") else ""}'
                             if it.iter >= 1 and not it.convergence_padded else '')
                    pad = ' [PAD]' if it.convergence_padded else ''
                    print(f'    iter {it.iter}: share={sh_txt} '
                          f'score={it.score:.4f}{pad}{extra}')
                run.note(f'{board_label}|scaffold|seed={seed}: done '
                         f'final={iters[-1].score}')
    executor.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
