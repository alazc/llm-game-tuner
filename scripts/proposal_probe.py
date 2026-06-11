"""EXP-008 — proposal-distribution probe: generation vs selection (7B).

No loop. Two fixed states; sample N candidate edits per temperature; score
every parsed candidate offline at the SAME iteration-1 CRN seed (paired with a
baseline-at-iter-1 reference), and read off:

  P(parsed), P(touches valid rent), P(direction down), P(productive: min valid
  rent mult <= 0.7), P(paired-improving), P(in-band), rent-cut magnitudes,
  the greedy (T=0) reference's rank within the sample, and the best-of-N curve.

States (prereg EXP-008):
  pfix   default board (== the EXP-004 P-HOLD-bad seed state; the 0/5 failure)
  phold  B* = uniform rent x0.30 (in-band; measures "stay put" proposals)

Feed/prompt/parser identical to the EXP-003 loop at iteration 1 (met channel,
designer_llm_prompt_open, 320-token cap). The ONLY changes: temperature > 0
with N return sequences, and no loop.

Usage:
    # model-independent gate (classifier checks + paired-seed determinism)
    set PYTHONPATH=. && python scripts/proposal_probe.py --selftest

    # heuristic IO smoke (no GPU)
    set PYTHONPATH=. && python scripts/proposal_probe.py --state pfix \
        --backend heuristic --n-samples 3 --n-games 20 \
        --out-dir report/figures/exp8_smoke/pfix

    # production (GPU; via modal_exp8.py)
    set PYTHONPATH=. && python scripts/proposal_probe.py --state pfix \
        --backend local --model Qwen/Qwen2.5-7B-Instruct \
        --n-samples 50 --temps 0.8,0.4 --n-games 200 \
        --out-dir report/figures/exp8_run/pfix
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import GameConfig
from optimizer.group_design import (GroupDesign, apply_design, board_groups,
                                    referenced_groups)
from optimizer.skill_expression import G_STAR, BAND

# Reuse the EXACT EXP-003 pieces (do not fork).
from scripts.llm_design_loop import (DesignerLLM, build_diagnostic_feed,
                                     _eval_with_aligned_seeds, eval_seed_for,
                                     parse_designer_response)
from scripts.mech_probe_loop import build_bstar
from optimizer.strategy_pool import load_eval_matchups, load_strategy_pool

STATES = ('pfix', 'phold')


# --------------------------------------------------------------------------- #
# Classification                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class SampleRecord:
    state:            str
    temperature:      float
    sample_idx:       int          # -1 = greedy reference
    chunk_seed:       int
    parser_status:    str
    rationale:        str
    invalid_groups:   list
    touches_valid_rent: bool
    direction_down:   Optional[bool]
    min_valid_rent:   Optional[float]
    productive:       bool
    n_knobs_touched:  int
    edit_l1:          float
    score:            Optional[float]
    share:            Optional[float]
    paired_improving: Optional[bool]
    in_band:          Optional[bool]
    design:           Optional[dict]
    raw_response:     str


def classify_design(design: GroupDesign, valid: set) -> dict:
    """Lever-level classification of one parsed design (no sims)."""
    rents = {g: m for g, m in design.group_rent_mult.items()
             if g in valid and m != 1.0}
    costs = {g: m for g, m in design.group_cost_mult.items()
             if g in valid and m != 1.0}
    drops = [g for g in design.drop_groups if g in valid]
    n_knobs = (len(rents) + len(costs) + len(drops)
               + (1 if design.salary_mult != 1.0 else 0)
               + len(design.prop_overrides))
    l1 = (sum(abs(m - 1.0) for m in rents.values())
          + sum(abs(m - 1.0) for m in costs.values())
          + abs(design.salary_mult - 1.0) + 1.0 * len(drops))
    min_rent = min(rents.values()) if rents else None
    return {
        'invalid_groups': sorted(referenced_groups(design) - valid),
        'touches_valid_rent': bool(rents),
        'direction_down': (all(m < 1.0 for m in rents.values()) if rents else None),
        'min_valid_rent': min_rent,
        'productive': bool(min_rent is not None and min_rent <= 0.7),
        'n_knobs_touched': n_knobs,
        'edit_l1': l1,
    }


# --------------------------------------------------------------------------- #
# Sampling                                                                       #
# --------------------------------------------------------------------------- #

def sample_batch(designer: DesignerLLM, feed: str, n: int, temperature: float,
                 base_seed: int, chunk: int = 25) -> List[dict]:
    """Batched sampling via the same chat template the loop uses. Returns
    [{'raw': str, 'chunk_seed': int}]. Heuristic backend: repeats the canned
    cycle (IO smoke only)."""
    if designer.backend == 'heuristic':
        return [{'raw': designer.query(feed, iteration=i)[0], 'chunk_seed': -1}
                for i in range(n)]
    import torch
    tok, model, device = designer._get_local()
    msgs = [{'role': 'system', 'content': designer.system_prompt},
            {'role': 'user', 'content': feed}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(device)
    out: List[dict] = []
    done = 0
    while done < n:
        take = min(chunk, n - done)
        seed = base_seed + done
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=designer.max_new_tokens,
                                 do_sample=True, temperature=temperature,
                                 top_p=1.0, num_return_sequences=take,
                                 pad_token_id=tok.eos_token_id)
        for row in gen[:, inputs['input_ids'].shape[1]:]:
            out.append({'raw': tok.decode(row, skip_special_tokens=True).strip(),
                        'chunk_seed': seed})
        done += take
    return out


# --------------------------------------------------------------------------- #
# Probe driver                                                                   #
# --------------------------------------------------------------------------- #

def build_state_cfg(state: str, canonical_config: str) -> GameConfig:
    if state == 'pfix':
        return GameConfig.from_yaml(canonical_config)
    if state == 'phold':
        return build_bstar(canonical_config, 0.30)
    raise ValueError(state)


def best_of_n_curve(scores: List[float], n_shuffles: int = 1000,
                    seed: int = 0) -> List[float]:
    """E[min score among the first N samples], N=1..len, MC over orderings."""
    if not scores:
        return []
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    acc = np.zeros(len(arr))
    for _ in range(n_shuffles):
        perm = rng.permutation(arr)
        acc += np.minimum.accumulate(perm)
    return list(acc / n_shuffles)


def run_probe(args, out_dir: Path) -> int:
    state = args.state
    start = build_state_cfg(state, args.canonical_config)
    valid = set(board_groups(start))
    board_label = 'default'                      # CRN convention from EXP-004
    seed = 42

    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(args.n_players, pool_size=len(pool),
                                  n_matchups=args.n_matchups,
                                  seed=args.matchup_seed)

    # Baseline evals: iter-0 (the feed's eval block) and iter-1 (paired ref).
    print(f'[{state}] baseline evals (n_games={args.n_games}) ...')
    ev0 = _eval_with_aligned_seeds(start, pool, matchups, args.n_games,
                                   board_label, seed, 0, args.max_turns)
    ev1 = _eval_with_aligned_seeds(start, pool, matchups, args.n_games,
                                   board_label, seed, 1, args.max_turns)
    base_score_iter1 = ev1['score']
    print(f'[{state}] iter0 share={ev0["skill_share"]:.3f} score={ev0["score"]:.4f} | '
          f'iter1-ref score={base_score_iter1:.4f}')

    # Iteration-1 feed, exactly as the loop builds it.
    design0 = GroupDesign(label='cumulative')
    prior_iters = [{'iter': 0, 'design_diff': 'no diff from default',
                    'rationale': '[baseline]', 'score': ev0['score'],
                    'metrics': ev0['metrics']}]
    feed = build_diagnostic_feed(start, design0,
                                 eval_out={'metrics': ev0['metrics'],
                                           'score': ev0['score'],
                                           'skill_curve': ev0.get('skill_curve')},
                                 prior_eval=None, condition='met',
                                 prior_iters=prior_iters)

    designer = DesignerLLM(backend=args.backend, model_name=args.model,
                           goal_disclosure='open')

    temps = [float(t) for t in args.temps.split(',') if t.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / 'samples.jsonl'
    fh = open(samples_path, 'w', encoding='utf-8')

    def eval_candidate(design: GroupDesign):
        cfg2 = apply_design(start, design, strict_groups=False)
        ev = _eval_with_aligned_seeds(cfg2, pool, matchups, args.n_games,
                                      board_label, seed, 1, args.max_turns)
        return ev['score'], ev['skill_share']

    def record(raw: str, chunk_seed: int, temp: float, idx: int) -> SampleRecord:
        resp = parse_designer_response(raw)
        base = dict(state=state, temperature=temp, sample_idx=idx,
                    chunk_seed=chunk_seed, parser_status=resp.parser_status,
                    rationale=resp.rationale[:200], raw_response=raw[:2000])
        if resp.parser_status != 'ok' or resp.design is None:
            return SampleRecord(**base, invalid_groups=[],
                                touches_valid_rent=False, direction_down=None,
                                min_valid_rent=None, productive=False,
                                n_knobs_touched=0, edit_l1=0.0, score=None,
                                share=None, paired_improving=None, in_band=None,
                                design=None)
        cls = classify_design(resp.design, valid)
        score, share = eval_candidate(resp.design)
        return SampleRecord(**base, **cls, score=score, share=share,
                            paired_improving=bool(score < base_score_iter1),
                            in_band=bool(score == 0),
                            design=resp.design.to_dict())

    all_recs: List[SampleRecord] = []
    # Greedy reference (the deployed policy's move).
    t0 = time.perf_counter()
    raw_greedy, _ = designer.query(feed, iteration=0)
    rec = record(raw_greedy, -1, 0.0, -1)
    all_recs.append(rec); fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
    print(f'[{state}] greedy ref: parser={rec.parser_status} score={rec.score} '
          f'({time.perf_counter() - t0:.0f}s)')

    for temp in temps:
        t0 = time.perf_counter()
        batch = sample_batch(designer, feed, args.n_samples, temp,
                             base_seed=args.sample_seed, chunk=args.chunk)
        print(f'[{state}] T={temp}: sampled {len(batch)} in '
              f'{time.perf_counter() - t0:.0f}s; scoring ...')
        for i, b in enumerate(batch):
            rec = record(b['raw'], b['chunk_seed'], temp, i)
            all_recs.append(rec); fh.write(json.dumps(asdict(rec)) + '\n')
            if (i + 1) % 10 == 0:
                fh.flush()
                print(f'    scored {i + 1}/{len(batch)}')
        fh.flush()
    fh.close()

    # ---------------- summary + sealed decision rule ----------------
    summary = {'state': state, 'n_samples': args.n_samples, 'temps': temps,
               'base_score_iter1': base_score_iter1,
               'baseline_share_iter0': ev0['skill_share'],
               'greedy': asdict(all_recs[0]), 'per_temp': {}}
    txt = [f'=== EXP-008 probe :: state={state} ===',
           f'baseline iter0 share={ev0["skill_share"]:.3f}  '
           f'iter1-ref score={base_score_iter1:.4f}',
           f'greedy ref: parser={all_recs[0].parser_status} '
           f'score={all_recs[0].score} productive={all_recs[0].productive}']
    for temp in temps:
        recs = [r for r in all_recs if r.temperature == temp and r.sample_idx >= 0]
        parsed = [r for r in recs if r.parser_status == 'ok']
        n, np_ = len(recs), len(parsed)
        if not n:
            continue
        scores = [r.score for r in parsed]
        improving = [r for r in parsed if r.paired_improving]
        curve = best_of_n_curve(scores) if scores else []
        target = None
        if curve:
            gain50 = curve[0] - curve[-1]
            if gain50 > 1e-9:
                target = next((i + 1 for i, v in enumerate(curve)
                               if (curve[0] - v) >= 0.8 * gain50), None)
        d = {
            'n': n, 'parse_rate': np_ / n,
            'p_touch_rent': (sum(r.touches_valid_rent for r in parsed) / np_) if np_ else 0,
            'p_direction_down': (sum(bool(r.direction_down) for r in parsed) / np_) if np_ else 0,
            'p_productive': (sum(r.productive for r in parsed) / np_) if np_ else 0,
            'p_improving': (len(improving) / np_) if np_ else 0,
            'p_in_band': (sum(bool(r.in_band) for r in parsed) / np_) if np_ else 0,
            'mean_edit_l1': (float(np.mean([r.edit_l1 for r in parsed])) if np_ else 0),
            'min_valid_rents': sorted(round(r.min_valid_rent, 3) for r in parsed
                                      if r.min_valid_rent is not None),
            'greedy_rank_pct': (float(np.mean([s < all_recs[0].score for s in scores]))
                                if scores and all_recs[0].score is not None else None),
            'best_of_n_curve': [round(v, 5) for v in curve],
            'n_star_80pct': target,
        }
        summary['per_temp'][str(temp)] = d
        txt += [f'--- T={temp}: parse {np_}/{n}',
                f'    P(touch valid rent)={d["p_touch_rent"]:.2f}  '
                f'P(down)={d["p_direction_down"]:.2f}  '
                f'P(productive<=0.7)={d["p_productive"]:.2f}',
                f'    P(paired-improving)={d["p_improving"]:.2f}  '
                f'P(in-band)={d["p_in_band"]:.2f}  N*(80%)={d["n_star_80pct"]}',
                f'    greedy-better-than {100 * (d["greedy_rank_pct"] or 0):.0f}% of samples']
    # Sealed branch call on pfix @ T=0.8.
    if state == 'pfix' and '0.8' in summary['per_temp']:
        p = summary['per_temp']['0.8']['p_improving']
        branch = ('SELECTION' if p >= 0.20 else
                  'GENERATION' if p <= 0.05 else 'MIXED')
        summary['branch'] = branch
        txt.append(f'SEALED DECISION RULE (pfix, T=0.8): P(improving|parsed)={p:.2f} '
                   f'-> {branch}')
    (out_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2),
                                          encoding='utf-8')
    (out_dir / 'PROBE_SUMMARY.txt').write_text('\n'.join(txt) + '\n',
                                               encoding='utf-8')
    print('\n'.join(txt))
    return 0


# --------------------------------------------------------------------------- #
# Self-test (model-independent gate; pre-seal)                                   #
# --------------------------------------------------------------------------- #

def selftest(args) -> int:
    # (iii) classifier unit checks on hand-written designs.
    canon = GameConfig.from_yaml(args.canonical_config)
    valid = set(board_groups(canon))
    d_rent = GroupDesign(group_rent_mult={'Orange': 0.5, 'Red': 0.6})
    c = classify_design(d_rent, valid)
    assert c['touches_valid_rent'] and c['direction_down'] and c['productive'], c
    assert c['min_valid_rent'] == 0.5 and c['n_knobs_touched'] == 2, c
    d_cost = GroupDesign(group_cost_mult={'Orange': 0.5})
    c = classify_design(d_cost, valid)
    assert not c['touches_valid_rent'] and not c['productive'], c
    d_bad = GroupDesign(group_rent_mult={'NotAGroup': 0.5})
    c = classify_design(d_bad, valid)
    assert c['invalid_groups'] == ['NotAGroup'] and not c['touches_valid_rent'], c
    d_sal = GroupDesign(salary_mult=2.0)
    c = classify_design(d_sal, valid)
    assert c['n_knobs_touched'] == 1 and c['min_valid_rent'] is None, c
    print('classifier checks: OK')

    # (ii) paired-seed determinism: same design scored twice -> identical.
    pool = load_strategy_pool(args.pool)
    matchups = load_eval_matchups(2, pool_size=len(pool), n_matchups=10, seed=1234)
    cfg2 = apply_design(canon, d_rent, strict_groups=False)
    e1 = _eval_with_aligned_seeds(cfg2, pool, matchups, 20, 'default', 42, 1, 200)
    e2 = _eval_with_aligned_seeds(cfg2, pool, matchups, 20, 'default', 42, 1, 200)
    assert e1['score'] == e2['score'] and e1['skill_share'] == e2['skill_share']
    print(f'paired-seed determinism: OK (score={e1["score"]:.4f} twice)')

    # best-of-N curve sanity: monotone non-increasing, endpoints exact.
    cur = best_of_n_curve([0.3, 0.1, 0.2, 0.0])
    assert all(b <= a + 1e-12 for a, b in zip(cur, cur[1:]))
    assert abs(cur[-1] - 0.0) < 1e-12 and abs(cur[0] - 0.15) < 0.02
    print('best-of-N curve: OK')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', choices=STATES, default='pfix')
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'),
                    default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--canonical-config', default='default_config.yaml')
    ap.add_argument('--n-samples', type=int, default=50)
    ap.add_argument('--temps', default='0.8,0.4')
    ap.add_argument('--sample-seed', type=int, default=12345)
    ap.add_argument('--chunk', type=int, default=25)
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--n-matchups', type=int, default=10)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--n-players', type=int, default=2, choices=(2, 3))
    ap.add_argument('--matchup-seed', type=int, default=1234)
    ap.add_argument('--pool', default='optimizer/strategy_pool.json')
    ap.add_argument('--out-dir', default='report/figures/exp8_run/pfix')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        return selftest(args)
    return run_probe(args, Path(args.out_dir))


if __name__ == '__main__':
    raise SystemExit(main())
