"""EXP-004 analyzer + classifier - GENERATION vs CREDIT vs PRIOR.

Read-only over the probe artifacts (report/figures/exp4_run/{peval,phold_good,
phold_bad}). Produces the three condition summaries, the B-vs-C contrast, and the
classification against the sealed 3-mode table, with the implied Exp-4 prediction.

  peval       -> fraction of orderings picking B* (chance 0.25) + prior/score-talk
  phold_good  -> per-iter edit-mag / share / in-band; preserve vs edit-out;
                 first score-worsening -> revert-or-persist (+ rationale)
  phold_bad   -> EXP-003 generation-failure replication (flat high, 0/N band)

Run: python -m experiments.exp4_analyze [ROOT]   (default report/figures/exp4_run)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from statistics import mean

from config import GameConfig
from optimizer.group_design import board_groups
from optimizer.skill_expression import G_STAR, BAND
# reuse the EXP-003 classifier's effective-board vector + L1 delta
from experiments.exp0_explore_7b import design_vec, l1_delta

BAND_LO, BAND_HI = G_STAR - BAND, G_STAR + BAND          # 0.57, 0.63
VALID = set(board_groups(GameConfig.from_yaml('default_config.yaml')))

# rationale keyword lexicons (convenience tag; verbatim text is printed too)
PRIOR_WORDS = ('high', 'higher', 'competitive', 'valuable', 'premium', 'realistic',
               'expensive', 'lucrative', 'should remain', 'keep', 'maintain', 'strong')
SCORE_WORDS = ('0.6', '.60', 'target', 'closest', 'reduce', 'lower', 'edge', 'luck',
               'random', 'share', 'contestable', 'down', 'closer')


def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def _tag(rat: str) -> str:
    r = rat.lower()
    p = sum(w in r for w in PRIOR_WORDS)
    s = sum(w in r for w in SCORE_WORDS)
    if p > s:
        return 'prior'
    if s > p:
        return 'score'
    return 'mixed/none'


# --------------------------------------------------------------------------- #
# P-EVAL                                                                        #
# --------------------------------------------------------------------------- #

def analyse_peval(root):
    p = os.path.join(root, 'peval', 'peval.jsonl')
    if not os.path.exists(p):
        print('  (no peval.jsonl)'); return None
    recs = load(p)
    n = len(recs)
    n_bstar = sum(r['picked_bstar'] for r in recs)
    frac = n_bstar / n if n else 0.0
    dist = {}
    for r in recs:
        dist[r['pick_id']] = dist.get(r['pick_id'], 0) + 1
    tags = {'prior': 0, 'score': 0, 'mixed/none': 0}
    for r in recs:
        tags[_tag(r.get('rationale', ''))] += 1
    print(f'--- P-EVAL (Condition A: declarative recognition) ---')
    print(f'  orderings: {n}   picked B*: {n_bstar}  (frac={frac:.3f}, chance=0.25)')
    print(f'  pick distribution: ' +
          '  '.join(f'{k}={v}' for k, v in sorted(dist.items(), key=lambda x: -x[1])))
    print(f'  rationale tags: prior-talk={tags["prior"]} score-talk={tags["score"]} '
          f'mixed/none={tags["mixed/none"]}')
    # a few verbatim rationales for B*-picks and for non-B*-picks
    print('  rationale excerpts:')
    shown = 0
    for r in recs:
        if shown >= 6:
            break
        tag = _tag(r.get('rationale', ''))
        print(f'    [{r["pick_id"]:>7} {tag:>10}] {r.get("rationale","")[:140]!r}')
        shown += 1
    above_chance = frac >= 0.50          # clearly above 0.25 chance
    return {'n': n, 'frac_bstar': frac, 'dist': dist, 'tags': tags,
            'above_chance': above_chance}


# --------------------------------------------------------------------------- #
# P-HOLD                                                                        #
# --------------------------------------------------------------------------- #

def _traj(recs):
    vecs = [design_vec(r.get('design') or {}, VALID) for r in recs]
    mags = [l1_delta(vecs[i], vecs[i - 1]) for i in range(1, len(vecs))]
    shares = [(r.get('metrics') or {}).get('skill_share') for r in recs]
    dist = [abs(s - G_STAR) if s is not None else None for s in shares]   # dist from g*
    inband = [(s is not None and BAND_LO <= s <= BAND_HI) for s in shares]
    rats = [r.get('rationale', '') for r in recs]
    conv = next((r['iter'] for r in recs if r.get('converged_request')), None)
    # first iter (k>=1) whose dist-from-g* worsens vs prev; then does k+1 revert?
    first_worsen = None
    revert = None
    decision_rat = ''
    for k in range(1, len(dist)):
        if dist[k] is None or dist[k - 1] is None:
            continue
        if dist[k] > dist[k - 1] + 1e-6:           # score worsened at k
            first_worsen = k
            if k + 1 < len(dist) and dist[k + 1] is not None:
                made_edit = (k + 1 - 1) < len(mags) and mags[k] > 1e-9
                revert = dist[k + 1] < dist[k] - 1e-6      # moved back toward g*
                decision_rat = rats[k + 1] if k + 1 < len(rats) else ''
            break
    coverage = [len(v) for v in vecs]
    # min rent multiplier ever applied (aggressive cut?) and any up (wrong dir)
    rent_vals = [m for r in recs for g, m in ((r.get('design') or {}).get('group_rent_mult') or {}).items()
                 if g in VALID]
    return dict(mags=mags, shares=shares, dist=dist, inband=inband, conv=conv,
                first_worsen=first_worsen, revert=revert, decision_rat=decision_rat,
                coverage=coverage, final_share=shares[-1], final_inband=inband[-1],
                min_rent=(min(rent_vals) if rent_vals else None),
                any_up=any(m > 1.05 for m in rent_vals))


def analyse_phold(root, mode):
    paths = sorted(glob.glob(os.path.join(root, mode, f'{mode}__*.jsonl')))
    if not paths:
        print(f'  (no {mode} trajectories)'); return None
    rows = [_traj(load(p)) for p in paths]
    K = max(len(r['shares']) for r in rows) - 1
    def col(key, i):
        xs = [r[key][i] for r in rows if i < len(r[key]) and r[key][i] is not None]
        return mean(xs) if xs else float('nan')
    print(f'--- {mode} ({"B: seed FROM B*" if mode=="phold_good" else "C: seed FROM default"}) ---')
    print('  share/iter   : ' + ' '.join(f'{col("shares",i):.3f}' for i in range(K + 1)))
    print('  edit-mag/iter: ' + ' '.join(f'{col("mags",i):5.2f}' for i in range(K)))
    in_iter = [sum(1 for r in rows if i < len(r['inband']) and r['inband'][i]) for i in range(K + 1)]
    print('  in-band/iter : ' + ' '.join(f'{x}/{len(rows)}' for x in in_iter))
    final_in = sum(1 for r in rows if r['final_inband'])
    mean_mag = mean([m for r in rows for m in r['mags']]) if any(r['mags'] for r in rows) else 0.0
    print(f'  final in-band: {final_in}/{len(rows)}   mean edit-mag(all iters)={mean_mag:.2f}')
    print(f'  converged-declared: {sum(1 for r in rows if r["conv"] is not None)}/{len(rows)}'
          f'  | pushed rent UP (wrong dir): {sum(1 for r in rows if r["any_up"])}/{len(rows)}'
          f'  | min rent applied: {sorted(round(r["min_rent"],2) for r in rows if r["min_rent"] is not None)}')
    # first-worsening -> revert/persist (phold_good is the critical one)
    fw = [(i, r) for i, r in enumerate(rows) if r['first_worsen'] is not None]
    rev = sum(1 for _, r in fw if r['revert'])
    per = sum(1 for _, r in fw if r['revert'] is False)
    print(f'  first score-worsening seen in {len(fw)}/{len(rows)} seeds: '
          f'reverted(uses feedback)={rev}  persisted(overrides feedback)={per}')
    for i, r in fw[:3]:
        print(f'    seed{i} worsen@iter{r["first_worsen"]} -> '
              f'{"REVERT" if r["revert"] else "PERSIST"}  decision-rationale: {r["decision_rat"][:120]!r}')
    return {'rows': rows, 'final_in': final_in, 'n': len(rows), 'mean_mag': mean_mag,
            'reverted': rev, 'persisted': per, 'n_worsen': len(fw)}


# --------------------------------------------------------------------------- #
# Contrast + classification                                                     #
# --------------------------------------------------------------------------- #

def classify(peval, good, bad):
    print('\n=== B vs C CONTRAST ===')
    if good and bad:
        print(f'  mean edit-mag  good(B)={good["mean_mag"]:.2f}  bad(C)={bad["mean_mag"]:.2f}  '
              f'(similar => quality does NOT change behavior => any holding is passivity)')
        print(f'  final in-band  good(B)={good["final_in"]}/{good["n"]}  '
              f'bad(C)={bad["final_in"]}/{bad["n"]}')

    print('\n=== CLASSIFICATION (sealed 3-mode table) ===')
    if peval is None or good is None or bad is None:
        print('  incomplete artifacts - cannot classify'); return

    recognizes = peval['above_chance']
    # "preserve" = small edits AND ends in-band on a majority of seeds
    preserves = (good['final_in'] >= (good['n'] + 1) // 2) and (good['mean_mag'] < 0.25)
    degrades_persists = (good['final_in'] < (good['n'] + 1) // 2) and \
                        (good['persisted'] >= good['reverted'] and good['n_worsen'] > 0)
    bad_fails = bad['final_in'] == 0

    print(f'  P-EVAL recognizes B* (frac>=0.50): {recognizes} (frac={peval["frac_bstar"]:.2f})')
    print(f'  P-HOLD-good preserves B*:          {preserves} '
          f'(final in-band {good["final_in"]}/{good["n"]}, mean-mag {good["mean_mag"]:.2f})')
    print(f'  P-HOLD-good degrades + persists:   {degrades_persists} '
          f'(persist {good["persisted"]} vs revert {good["reverted"]})')
    print(f'  P-HOLD-bad fails to reach band:    {bad_fails} (final in-band {bad["final_in"]}/{bad["n"]})')

    if not recognizes:
        verdict = ('CREDIT / EVALUATION bottleneck - no usable value signal. '
                   'Implied Exp-4: better feedback about WHAT IS GOOD could matter.')
    elif recognizes and preserves and bad_fails:
        verdict = ('GENERATION bottleneck - recognizes & preserves the good policy, '
                   'cannot invent it. Implied Exp-4 (prior-free): REPRODUCES the failure.')
    elif recognizes and degrades_persists:
        verdict = ('PRIOR bottleneck - knows better, edits against it. '
                   'Implied Exp-4 (prior-free): does NOT reproduce the failure.')
    else:
        verdict = ('INCONCLUSIVE - signals do not cleanly match one row '
                   '(report the raw numbers; do not force a row).')
    print(f'\n  >> VERDICT: {verdict}')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'report/figures/exp4_run'
    print(f'=== EXP-004 mechanism probe :: {root} (band [{BAND_LO:.2f},{BAND_HI:.2f}]) ===\n')
    peval = analyse_peval(root)
    print()
    good = analyse_phold(root, 'phold_good')
    print()
    bad = analyse_phold(root, 'phold_bad')
    classify(peval, good, bad)


if __name__ == '__main__':
    main()
