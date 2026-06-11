"""EXP-014 — prior-ranked screening optimizer: what is the LLM prior worth?

The generic loop: schema → the LLM
RANKS the parameter families (its only job: an ordering of measurements) →
the harness screens families in that order (both directions, CRN-paired) →
engages the first family whose probe improves the score by >= theta →
line-searches the winner → stops in-band. Arms differ ONLY in the ranking
source: llm | random | oracle. Headline = probes_to_engage by arm, on a bed
where the prior is right (Monopoly) and a bed where no prior can exist
(synthetic PEAKED, opaque dim names).

Rankings are elicited in ONE GPU pass (--elicit, via modal_exp14.py); all
trajectories then run locally with no LLM in the loop.

Usage:
    # heuristic elicitation + smoke (no GPU)
    set PYTHONPATH=. && python scripts/screening_optimizer.py --elicit \
        --backend heuristic --out-dir report/figures/exp14_smoke
    set PYTHONPATH=. && python scripts/screening_optimizer.py --run --arm oracle \
        --boards default --n-seeds 1 --n-games 20 --K 3 \
        --out-dir report/figures/exp14_smoke

    # production
    (GPU)   modal run modal_exp14.py          # writes exp14_run/rankings.json
    (local) python scripts/screening_optimizer.py --run --arm all \
                --out-dir report/figures/exp14_run
    (local) python scripts/screening_optimizer.py --analyze \
                --out-dir report/figures/exp14_run
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from optimizer.exp_boards import build_exp0_boards
from optimizer.group_design import GroupDesign, apply_design, monotonicity_gate
from optimizer import run_log
from prompts.loader import load_prompt

from scripts.llm_design_loop import (DesignerLLM, _per_group_breakdown,
                                     eval_seed_for)
from scripts.choice_designer_loop import (CLAMP, RUNGS, _parse_json_block)
from scripts.scaffold_designer_loop import _worker_eval

from synthetic.landscape import (make_landscape, evaluate as synth_eval)

MONO_FAMS = ('rent', 'cost', 'salary')
ORACLE_MONO = ['rent', 'salary', 'cost']
SYNTH_SEEDS = (20260602, 20260703, 20260804, 20260905, 20261006)
THETA_MONO, THETA_SYNTH = 0.015, 0.01
PROBE_MONO = (('down', 0.8), ('up', 1.25))
SYNTH_DELTA = 0.2
RANK_MONO_PROMPT = 'designer_llm_prompt_rank_mono.txt'
RANK_SYNTH_PROMPT = 'designer_llm_prompt_rank_synth.txt'
RANDOM_ORDER_SEEDS = (0, 1, 2)

FAM_DESC = {
    'rent': 'rent levels of all property groups, moved together',
    'cost': 'purchase costs of all property groups, moved together',
    'salary': 'the salary collected on passing Go',
}


def _fam_design(vals: Dict[str, float], groups: List[str],
                rationale: str = '') -> dict:
    return GroupDesign(
        salary_mult=round(vals.get('salary', 1.0), 6),
        group_rent_mult={g: round(vals.get('rent', 1.0), 6) for g in groups},
        group_cost_mult={g: round(vals.get('cost', 1.0), 6) for g in groups},
        label='cumulative', rationale=rationale).to_dict()


def _eval_mono(board_tag: str, design: dict, base_seed: int, args) -> dict:
    return _worker_eval({'idx': 0, 'design': design, 'board_tag': board_tag,
                         'base_seed': base_seed, 'n_games': args['n_games'],
                         'max_turns': args['max_turns'],
                         'canonical_config': args['canonical_config']})


# --------------------------------------------------------------------------- #
# Monopoly trajectory (screen -> exploit), runs in a pool worker                 #
# --------------------------------------------------------------------------- #

def run_mono_cell(job: dict) -> dict:
    board_label, tag, seed = job['board_label'], job['tag'], job['seed']
    order, args = job['order'], job['args']
    boards = {re.sub(r'[^A-Za-z0-9]+', '_', l).strip('_'): c
              for l, c in build_exp0_boards(
                  canonical_config=args['canonical_config'])}
    groups = [r['group'] for r in _per_group_breakdown(boards[tag])]
    base_vals = {'rent': 1.0, 'cost': 1.0, 'salary': 1.0}

    # ---- SCREEN (iteration 1; all probes at the same CRN seed -> paired) ----
    s1 = eval_seed_for(board_label, seed, 1, 0)
    held = _eval_mono(tag, _fam_design(base_vals, groups), s1, args)
    probes, screening = 1, []
    engaged: Optional[Tuple[str, str]] = None
    best_overall = (None, None, -1e9)
    probe_pairs = [tuple(p) for p in args.get('probe_mono', PROBE_MONO)]
    theta = args.get('theta_mono', THETA_MONO)
    for fam in order:
        fam_best = (None, -1e9)
        for direction, f in probe_pairs:
            vals = dict(base_vals); vals[fam] = f
            r = _eval_mono(tag, _fam_design(vals, groups), s1, args)
            probes += 1
            imp = held['score'] - r['score']
            screening.append({'family': fam, 'direction': direction,
                              'factor': f, 'score': r['score'],
                              'improvement': round(imp, 5)})
            if imp > fam_best[1]:
                fam_best = (direction, imp)
            if imp > best_overall[2]:
                best_overall = (fam, direction, imp)
        if fam_best[1] >= theta:
            engaged = (fam, fam_best[0])
            break
    fallback = engaged is None
    if fallback:
        engaged = (best_overall[0], best_overall[1])
    probes_to_engage = probes

    # ---- EXPLOIT (iterations 2..K; EXP-012-B rung machinery) ----
    fam, direction = engaged
    vals = dict(base_vals)
    exploit, in_band, first_inband = [], False, None
    last_score = held['score']
    for k in range(2, args['K'] + 1):
        sk = eval_seed_for(board_label, seed, k, 0)
        held_k = _eval_mono(tag, _fam_design(vals, groups), sk, args)
        cands = []
        for rung in RUNGS[direction]:
            v2 = dict(vals)
            lo, hi = CLAMP[fam]
            v2[fam] = max(lo, min(hi, v2[fam] * rung))
            cands.append((rung, v2,
                          _eval_mono(tag, _fam_design(v2, groups), sk, args)))
        probes += 1 + len(cands)
        rung, v2, best = min(cands, key=lambda c: c[2]['score'])
        accepted = best['score'] < held_k['score']
        if accepted:
            vals = v2
            rec = best
        else:
            rec = held_k
        exploit.append({'iter': k, 'rung': rung if accepted else None,
                        'accepted': accepted, 'score': rec['score'],
                        'share': rec['skill_share'],
                        'held_score': held_k['score']})
        last_score = rec['score']
        if rec['score'] == 0:
            in_band, first_inband = True, k
            break

    return {'bed': 'mono', 'board': tag, 'seed': seed, 'arm': job['arm'],
            'order_seed': job.get('order_seed'), 'order': list(order),
            'screening': screening, 'engaged_family': fam,
            'engaged_direction': direction, 'fallback': fallback,
            'probes_to_engage': probes_to_engage,
            'rank_of_rent': list(order).index('rent') + 1,
            'exploit': exploit, 'final_score': last_score,
            'in_band': in_band, 'first_inband': first_inband,
            'total_evals': probes,
            'final_design': _fam_design(vals, groups)}


# --------------------------------------------------------------------------- #
# Synthetic trajectory (deterministic; runs inline)                              #
# --------------------------------------------------------------------------- #

def run_synth_cell(inst_seed: int, order: List[str], arm: str,
                   order_seed=None) -> dict:
    spec = make_landscape('peaked', seed=inst_seed, aligned=True)
    n2d = spec.name_to_dim()
    x = spec.x_start.copy()
    held = synth_eval(spec, x)['score']
    probes, screening = 1, []
    engaged = None
    best_overall = (None, 0, -1e9)
    for name in order:
        j = n2d[name]
        dim_best = (0, -1e9)
        for sign in (-1, +1):
            x2 = x.copy()
            x2[j] = float(np.clip(x2[j] + sign * SYNTH_DELTA, 0.0, 1.0))
            s = synth_eval(spec, x2)['score']
            probes += 1
            imp = held - s
            screening.append({'dim': name, 'sign': sign,
                              'improvement': round(imp, 5)})
            if imp > dim_best[1]:
                dim_best = (sign, imp)
            if imp > best_overall[2]:
                best_overall = (name, sign, imp)
        if dim_best[1] >= THETA_SYNTH:
            engaged = (name, dim_best[0])
            break
    fallback = engaged is None
    if fallback:
        engaged = (best_overall[0], best_overall[1])
    probes_to_engage = probes

    name, sign = engaged
    j = n2d[name]
    # grid then refine on the engaged dim
    cands = []
    for k in range(1, 5):
        x2 = x.copy()
        x2[j] = float(np.clip(x2[j] + sign * 0.2 * k, 0.0, 1.0))
        cands.append((x2, synth_eval(spec, x2)))
        probes += 1
    bx, bev = min(cands, key=lambda c: c[1]['score'])
    for d in (-0.1, +0.1):
        x2 = bx.copy()
        x2[j] = float(np.clip(x2[j] + d, 0.0, 1.0))
        ev = synth_eval(spec, x2)
        probes += 1
        if ev['score'] < bev['score']:
            bx, bev = x2, ev
    dom_name = spec.dim_names[spec.dominant_dim]
    return {'bed': 'synth', 'instance': inst_seed, 'arm': arm,
            'order_seed': order_seed, 'order': list(order),
            'engaged_dim': name, 'engaged_is_dominant': name == dom_name,
            'fallback': fallback, 'probes_to_engage': probes_to_engage,
            'dominant_dim': dom_name,
            'rank_of_dominant': list(order).index(dom_name) + 1,
            'final_score': bev['score'], 'in_band': bool(bev['in_band']),
            'total_evals': probes}


# --------------------------------------------------------------------------- #
# Ranking elicitation (--elicit; the only LLM step)                              #
# --------------------------------------------------------------------------- #

def _parse_ranking(text: str, expected: List[str]
                   ) -> Tuple[Optional[List[str]], str]:
    d = _parse_json_block(text)
    if d is None or not isinstance(d.get('ranking'), list):
        return None, 'parser_failure'
    low = {e.lower(): e for e in expected}
    out = []
    for item in d['ranking']:
        key = str(item).strip().lower()
        if key not in low or low[key] in out:
            return None, 'invalid_ranking'
        out.append(low[key])
    if len(out) != len(expected):
        return None, 'invalid_ranking'
    return out, 'ok'


def _mono_feed(cfg) -> str:
    bd = _per_group_breakdown(cfg)
    parts = ['## FAMILIES']
    parts += [f'  - {f}: {FAM_DESC[f]}' for f in MONO_FAMS]
    parts += ['', '## PER-GROUP COST/RENT BREAKDOWN (current board)']
    parts += [f'  - {r["group"]:>10}: n={r["n"]} mean_cost=${r["mean_cost"]:.0f} '
              f'mean_rent=${r["mean_rent"]:.0f}' for r in bd]
    parts += ['', '## YOUR JOB',
              '  Rank ALL the families (most to least likely to move the '
              'skill share). Reply with the JSON schema from the system '
              'prompt.']
    return '\n'.join(parts)


MASK_VALUES = False


def _synth_feed(spec) -> str:
    if MASK_VALUES:                       # AMENDMENT b: names only
        parts = ['## DIMENSIONS', '  ' + ', '.join(spec.dim_names)]
    else:
        parts = ['## DIMENSIONS (name: current value)']
        parts += [f'  - {n}: {spec.x_start[i]:.2f}'
                  for i, n in enumerate(spec.dim_names)]
    parts += ['', '## YOUR JOB',
              '  Rank ALL the dimensions (most to least likely to affect the '
              'quality). Reply with the JSON schema from the system prompt.']
    return '\n'.join(parts)


def elicit(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings = {'mono': {}, 'synth': {}, 'gen_cfg': {
        'backend': args.backend, 'model': args.model,
        'prompts': [RANK_MONO_PROMPT, RANK_SYNTH_PROMPT]}}

    def ask(designer, prompt_file, feed, expected):
        if args.backend == 'heuristic':
            return list(expected), 'ok', '[heuristic identity order]'
        raw, _ = designer.query(feed)
        rank, status = _parse_ranking(raw, expected)
        if status != 'ok':
            raw2, _ = designer.query(
                feed + '\n\nReply with EXACTLY one fenced ```json``` object: '
                       '{"ranking": [...all names, each once...], '
                       '"rationale": "..."}.')
            r2, s2 = _parse_ranking(raw2, expected)
            if s2 == 'ok':
                rank, status, raw = r2, s2, raw2
        return rank, status, raw[:1500]

    d_mono = d_synth = None
    if args.backend != 'heuristic':
        d_mono = DesignerLLM(backend=args.backend, model_name=args.model,
                             goal_disclosure='open')
        d_mono.system_prompt = load_prompt(RANK_MONO_PROMPT)
        d_synth = DesignerLLM(backend=args.backend, model_name=args.model,
                              goal_disclosure='open')
        d_synth.system_prompt = load_prompt(RANK_SYNTH_PROMPT)

    for label, cfg in build_exp0_boards(
            canonical_config=args.canonical_config):
        tag = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')
        rank, status, raw = ask(d_mono, RANK_MONO_PROMPT, _mono_feed(cfg),
                                list(MONO_FAMS))
        rankings['mono'][tag] = {'ranking': rank, 'status': status,
                                 'raw': raw}
        print(f'[mono {tag}] {status}: {rank}')

    for s in SYNTH_SEEDS:
        spec = make_landscape('peaked', seed=s, aligned=True)
        rank, status, raw = ask(d_synth, RANK_SYNTH_PROMPT,
                                _synth_feed(spec), list(spec.dim_names))
        rankings['synth'][str(s)] = {'ranking': rank, 'status': status,
                                     'raw': raw,
                                     'dominant': spec.dim_names[spec.dominant_dim]}
        print(f'[synth {s}] {status}: top3={rank[:3] if rank else None} '
              f'dominant={spec.dim_names[spec.dominant_dim]}')

    (out_dir / 'rankings.json').write_text(json.dumps(rankings, indent=2),
                                           encoding='utf-8')
    print(f'wrote {out_dir / "rankings.json"}')
    return 0


# --------------------------------------------------------------------------- #
# Run arms                                                                       #
# --------------------------------------------------------------------------- #

def _orders_for(arm: str, cell_key: str, expected: List[str],
                llm_entry: Optional[dict]) -> List[Tuple[Optional[int], List[str], bool]]:
    """[(order_seed, order, llm_fallback)] for one cell."""
    if arm == 'oracle':
        if set(expected) == set(MONO_FAMS):
            return [(None, ORACLE_MONO, False)]
        raise ValueError('synth oracle handled separately')
    if arm == 'random':
        out = []
        for osd in RANDOM_ORDER_SEEDS:
            rng = np.random.default_rng(
                abs(hash((cell_key, osd))) % (2 ** 32))
            out.append((osd, list(rng.permutation(expected)), False))
        return out
    # llm
    if llm_entry and llm_entry.get('status') == 'ok':
        return [(None, list(llm_entry['ranking']), False)]
    rng = np.random.default_rng(abs(hash((cell_key, 0))) % (2 ** 32))
    return [(0, list(rng.permutation(expected)), True)]    # sealed fallback


def run_arms(args) -> int:
    out_dir = Path(args.out_dir)
    rk_path = out_dir / 'rankings.json'
    rankings = (json.loads(rk_path.read_text(encoding='utf-8'))
                if rk_path.exists() else {'mono': {}, 'synth': {}})
    arms = ['llm', 'random', 'oracle'] if args.arm == 'all' else [args.arm]

    starting = list(build_exp0_boards(canonical_config=args.canonical_config))
    if args.boards != 'all':
        wanted = set(s.strip() for s in args.boards.split(','))
        starting = [(l, c) for l, c in starting if l in wanted]
    margs = {'n_games': args.n_games, 'max_turns': args.max_turns,
             'canonical_config': args.canonical_config, 'K': args.K,
             'probe_mono': [list(p) for p in PROBE_MONO],
             'theta_mono': THETA_MONO}

    jobs = []
    for arm in arms:
        for board_label, _cfg in starting:
            tag = re.sub(r'[^A-Za-z0-9]+', '_', board_label).strip('_')
            for sidx in range(args.n_seeds):
                seed = sidx * 1000 + 42
                for osd, order, fb in _orders_for(
                        arm, f'mono|{tag}|{seed}', list(MONO_FAMS),
                        rankings['mono'].get(tag)):
                    jobs.append({'board_label': board_label, 'tag': tag,
                                 'seed': seed, 'order': order, 'arm': arm,
                                 'order_seed': osd, 'llm_fallback': fb,
                                 'args': margs})
    print(f'monopoly jobs: {len(jobs)} (workers={args.workers})')
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(run_mono_cell, jobs)):
            rec['llm_fallback'] = jobs[i]['llm_fallback']
            results.append(rec)
            print(f'  [{i + 1}/{len(jobs)}] {rec["arm"]:>6} {rec["board"]:>14} '
                  f'seed={rec["seed"]} probes={rec["probes_to_engage"]} '
                  f'engaged={rec["engaged_family"]}-{rec["engaged_direction"]} '
                  f'in-band={"Y" if rec["in_band"] else "n"}')
    print(f'monopoly done in {time.perf_counter() - t0:.0f}s')

    synth_results = []
    if args.synth:
        for arm in arms:
            for s in SYNTH_SEEDS:
                spec = make_landscape('peaked', seed=s, aligned=True)
                expected = list(spec.dim_names)
                if arm == 'oracle':
                    dom = spec.dim_names[spec.dominant_dim]
                    orders = [(None, [dom] + [n for n in expected if n != dom],
                               False)]
                else:
                    orders = _orders_for(arm, f'synth|{s}', expected,
                                         rankings['synth'].get(str(s)))
                for osd, order, fb in orders:
                    rec = run_synth_cell(s, order, arm, osd)
                    rec['llm_fallback'] = fb
                    synth_results.append(rec)
                    print(f'  synth {arm:>6} inst={s} '
                          f'probes={rec["probes_to_engage"]} '
                          f'dom-rank={rec["rank_of_dominant"]} '
                          f'in-band={"Y" if rec["in_band"] else "n"}')

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.arm
    with open(out_dir / f'results_{suffix}.jsonl', 'w', encoding='utf-8') as fh:
        for r in results + synth_results:
            fh.write(json.dumps(r) + '\n')
    print(f'wrote {out_dir / f"results_{suffix}.jsonl"}')
    return 0


# --------------------------------------------------------------------------- #
# Analysis vs sealed predictions                                                 #
# --------------------------------------------------------------------------- #

def analyze(args) -> int:
    out_dir = Path(args.out_dir)
    recs = []
    for p in out_dir.glob('results_*.jsonl'):
        recs += [json.loads(l) for l in open(p, encoding='utf-8')]
    lines: List[str] = []

    def say(s=''):
        print(s); lines.append(s)

    say(f'=== EXP-014 prior-ranked screening :: {out_dir} ===')
    labels = {re.sub(r'[^A-Za-z0-9]+', '_', l).strip('_'): c
              for l, c in build_exp0_boards()}
    prior_value = {}
    for bed in ('mono', 'synth'):
        sub = [r for r in recs if r['bed'] == bed]
        if not sub:
            continue
        say(f'\n--- bed: {bed} ---')
        means = {}
        for arm in ('oracle', 'llm', 'random'):
            a = [r for r in sub if r['arm'] == arm]
            if not a:
                continue
            probes = [r['probes_to_engage'] for r in a]
            means[arm] = float(np.mean(probes))
            inb = sum(r['in_band'] for r in a)
            fb = sum(r.get('fallback', False) for r in a)
            lfb = sum(r.get('llm_fallback', False) for r in a)
            extra = ''
            if bed == 'mono':
                pos = [r['rank_of_rent'] for r in a]
                eng = {}
                for r in a:
                    k = f'{r["engaged_family"]}-{r["engaged_direction"]}'
                    eng[k] = eng.get(k, 0) + 1
                extra = (f'  rent-rank median={int(np.median(pos))} '
                         f'(first {sum(p == 1 for p in pos)}/{len(pos)})  '
                         f'engaged={eng}')
            else:
                pos = [r['rank_of_dominant'] for r in a]
                extra = (f'  dominant-rank median={float(np.median(pos)):.1f} '
                         f'positions={sorted(pos)}  '
                         f'engaged-dominant '
                         f'{sum(r["engaged_is_dominant"] for r in a)}/{len(a)}')
            say(f'  {arm:>6}: n={len(a)}  probes_to_engage mean='
                f'{means[arm]:.2f}  in-band {inb}/{len(a)}'
                f'{"  screen-fallbacks=" + str(fb) if fb else ""}'
                f'{"  llm-parse-fallbacks=" + str(lfb) if lfb else ""}')
            say(f'        {extra}')
        if 'random' in means and 'llm' in means:
            prior_value[bed] = means['random'] - means['llm']
            say(f'  PRIOR VALUE ({bed}) = E[probes_random] - E[probes_llm] = '
                f'{prior_value[bed]:+.2f} evals')

    # gates on monopoly in-band finals
    say('\n--- ladder gates on Monopoly in-band finals (n=120, deduped) ---')
    seen, fails = {}, 0
    for r in recs:
        if r['bed'] != 'mono' or not r['in_band']:
            continue
        key = json.dumps(r['final_design'], sort_keys=True) + '|' + r['board']
        if key not in seen:
            cfg = apply_design(labels[r['board']],
                               GroupDesign.from_dict(r['final_design']),
                               strict_groups=False)
            seen[key] = monotonicity_gate(cfg, n_seeds=120, base_seed=0)
        if not seen[key]['passes']:
            fails += 1
    say(f'  unique in-band finals: {len(seen)}; gate failures: {fails}')

    say('\n--- verdicts vs sealed predictions (prereg EXP-014) ---')
    mono_llm = [r for r in recs if r['bed'] == 'mono' and r['arm'] == 'llm']
    if mono_llm:
        first = sum(r['rank_of_rent'] == 1 for r in mono_llm)
        say(f'  mono llm rent-first {first}/{len(mono_llm)} (sealed >=8/9): '
            f'{"MET" if first >= 8 else "NOT MET"}')
    if 'mono' in prior_value:
        say(f'  prior-value(mono) {prior_value["mono"]:+.2f} '
            f'(sealed ~ +2): {"CONSISTENT" if 1.0 <= prior_value["mono"] <= 3.5 else "OFF-CALL"}')
    synth_llm = [r for r in recs if r['bed'] == 'synth' and r['arm'] == 'llm']
    if synth_llm:
        med = float(np.median([r['rank_of_dominant'] for r in synth_llm]))
        say(f'  synth llm dominant-rank median {med:.1f} (sealed in [4,9]): '
            f'{"MET" if 4 <= med <= 9 else "NOT MET -> artifact check"}')
    if 'synth' in prior_value:
        say(f'  prior-value(synth) {prior_value["synth"]:+.2f} '
            f'(sealed ~ 0, within +-2): '
            f'{"CONSISTENT" if abs(prior_value["synth"]) <= 2.0 else "OFF-CALL"}')
    for bed, floor_n in (('mono', 7), ('synth', 5)):
        for arm in ('llm', 'random', 'oracle'):
            a = [r for r in recs if r['bed'] == bed and r['arm'] == arm]
            if not a:
                continue
            # random arm: count per-cell best? floor applies per arm on all runs
            inb = sum(r['in_band'] for r in a)
            need = floor_n if arm != 'random' else int(
                floor_n / 9 * len(a)) if bed == 'mono' else len(a)
            say(f'  solvability {bed}/{arm}: {inb}/{len(a)}')
    (out_dir / 'EXP14_REPORT.txt').write_text('\n'.join(lines) + '\n',
                                              encoding='utf-8')
    say(f'\nwrote {out_dir / "EXP14_REPORT.txt"}')
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--elicit', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--arm', choices=('llm', 'random', 'oracle', 'all'),
                    default='all')
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--boards', default='all')
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--synth', dest='synth', action='store_true', default=True)
    ap.add_argument('--no-synth', dest='synth', action='store_false')
    ap.add_argument('--out-dir', default='report/figures/exp14_run')
    # --- flagged-amendment knobs (sealed defaults; see prereg AMENDMENTS) ---
    ap.add_argument('--probe-factors', default=None,
                    help='AMENDMENT a: mono probe pair as "down,up" factors '
                         '(e.g. 0.5,2.0). Default = sealed 0.8,1.25.')
    ap.add_argument('--theta-mono', type=float, default=None,
                    help='AMENDMENT a: engagement threshold (default sealed 0.015).')
    ap.add_argument('--mask-values', action='store_true',
                    help='AMENDMENT b: synth elicitation feed shows dim NAMES '
                         'only (no current values).')
    args = ap.parse_args()
    global PROBE_MONO, THETA_MONO, MASK_VALUES
    if args.probe_factors:
        dn, up = (float(x) for x in args.probe_factors.split(','))
        PROBE_MONO = (('down', dn), ('up', up))
    if args.theta_mono is not None:
        THETA_MONO = args.theta_mono
    MASK_VALUES = bool(args.mask_values)

    if args.elicit:
        return elicit(args)
    if args.analyze:
        return analyze(args)
    if args.run:
        with run_log.track(experiment='EXP-014-screening', script=__file__,
                           args=vars(args), out_dir=Path(args.out_dir),
                           condition=args.arm) as _run:
            return run_arms(args)
    print('nothing to do: pass --elicit, --run, or --analyze')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
