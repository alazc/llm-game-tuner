"""EXP-002 analysis: per-board (NOT pooled) info-gradient (H1) + curve-vs-table (H2).

Reads report/figures/exp0/<cond>/<cond>__<board>__seed<seed>.jsonl (Iteration
records) and reports, WITHIN each board:
  - final skill_score per condition (mean +/- SE over seeds), the H1 gradient;
  - H2 contrast haz (curve) vs met (table);
  - rand/canon anchors; iterations-to-dead-zone;
  - salary x2 DECOY readout: did the designer chase salary (wrong) or find rent?
Per the prereg, scores are NEVER pooled across boards (different lever profiles).

Run: python -m experiments.exp0_analyze
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import os
ROOT = Path(os.environ.get('EXP0_ROOT', 'report/figures/exp0'))
G_STAR, BAND = 0.60, 0.03
CONDS = ['mute', 'haz', 'met', 'full', 'rand', 'canon']     # sanity handled separately
BOARDS = ['default', 'salary_x2', 'gut_mid_tier']
FNAME = re.compile(r'(?P<cond>\w+?)__(?P<board>.+?)__seed(?P<seed>\d+)\.jsonl$')


def load_traj(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def final_score(records):
    scored = [r for r in records if r.get('score') is not None]
    return scored[-1]['score'] if scored else None


def final_share(records):
    for r in reversed(records):
        m = r.get('metrics') or {}
        if m.get('skill_share') is not None:
            return m['skill_share']
    return None


def iters_to_band(records):
    for r in records:
        s = r.get('score')
        if s is not None and s <= 1e-9:
            return r.get('iter')
    return None


def se(xs):
    xs = [x for x in xs if x is not None]
    return (stdev(xs) / len(xs) ** 0.5) if len(xs) > 1 else float('nan')


def collect():
    # cell[(cond, board)] = list of (seed, records)
    cell = defaultdict(list)
    for cond in CONDS + ['sanity']:
        d = ROOT / cond
        if not d.exists():
            continue
        for p in sorted(d.glob('*.jsonl')):
            m = FNAME.search(p.name)
            if not m:
                continue
            cell[(m['cond'], m['board'])].append((int(m['seed']), load_traj(p)))
    return cell


def salary_decoy_readout(records):
    """On the salary x2 board: track whether the cumulative design moved salary
    (chasing the decoy) vs rents (the real lever). Returns (touched_salary,
    touched_rent) over the trajectory."""
    touched_salary = touched_rent = False
    for r in records:
        d = r.get('design') or {}
        # starting board has salary doubled already; the designer's OWN edits are
        # in the cumulative design diff. salary_mult != 1.0 from the designer, or
        # group_rent_mult entries, signal which lever it reached for.
        if float(d.get('salary_mult', 1.0)) != 1.0:
            touched_salary = True
        if d.get('group_rent_mult'):
            touched_rent = True
    return touched_salary, touched_rent


def invalid_edit_rate(records):
    """Fraction of LLM iterations (parser ok, has a design) that named >=1 invalid
    group. Step-5.2 dependent-variable repair readout."""
    llm = [r for r in records if r.get('parser_status') == 'ok' and r.get('iter', 0) > 0]
    if not llm:
        return float('nan'), 0
    bad = sum(1 for r in llm if r.get('invalid_groups'))
    return bad / len(llm), len(llm)


def applied_boards_distinct(cell, board):
    """Step 5.1 gate: do mute/haz/met/full produce different EFFECTIVE boards?
    Compared via CRN-aligned final SCORES (identical score <=> identical effective
    board, since eval seeds are condition-independent). This is the signal that
    exposed the run-1 mute==haz collapse -- design DICTS can differ while no-op
    invalid-group edits leave the scored board identical, so dict-equality misses it."""
    finals = {}
    for cond in ('mute', 'haz', 'met', 'full'):
        runs = cell.get((cond, board), [])
        if runs:
            finals[cond] = tuple(round(final_score(rec), 6) if final_score(rec) is not None else None
                                 for _, rec in sorted(runs))
    out = {}
    for a, b in (('mute', 'haz'), ('met', 'full'), ('mute', 'met')):
        if a in finals and b in finals:
            out[f'{a}=={b}'] = finals[a] == finals[b]
    return out


def main():
    cell = collect()
    if not cell:
        print('no artifacts under', ROOT, '- run the Modal pull first.')
        return

    for board in BOARDS:
        print(f'\n================ BOARD: {board} ================')
        print(f'{"cond":6} {"n":>2}  {"final_score":>18}  {"final_share":>11}  {"iters->band":>11}')
        finals = {}
        for cond in CONDS:
            runs = cell.get((cond, board), [])
            if not runs:
                continue
            fs = [final_score(rec) for _, rec in runs]
            sh = [final_share(rec) for _, rec in runs]
            itb = [iters_to_band(rec) for _, rec in runs]
            finals[cond] = fs
            in_band = sum(1 for x in fs if x is not None and x <= 1e-9)
            ms = mean([x for x in fs if x is not None]) if any(x is not None for x in fs) else float('nan')
            print(f'{cond:6} {len(runs):>2}  {ms:>8.4f} +/-{se(fs):6.4f}  '
                  f'{mean([x for x in sh if x is not None]):>11.3f}  '
                  f'{str([x for x in itb]):>11}   (in-band {in_band}/{len(fs)})')
        # H1 / H2 verdicts (within board)
        if all(c in finals for c in ('mute', 'haz', 'met', 'full')):
            g = {c: mean([x for x in finals[c] if x is not None]) for c in ('mute', 'haz', 'met', 'full')}
            print(f'  H1 info-gradient (final_score, lower=better): '
                  f"mute={g['mute']:.4f}  haz={g['haz']:.4f}  met={g['met']:.4f}  full={g['full']:.4f}")
            print(f"     -> full beats haz? {'YES' if g['full'] < g['haz'] else 'NO'} "
                  f"(more info {'helped' if g['full'] < g['haz'] else 'did NOT help'})")
            print(f"  H2 curve-vs-table: haz(curve)={g['haz']:.4f}  met(table)={g['met']:.4f}  "
                  f"-> curve better? {'YES' if g['haz'] < g['met'] else 'NO'}")
        # Step 5.1 gate: applied boards must now differ (esp. mute vs haz)
        dist = applied_boards_distinct(cell, board)
        if dist:
            flags = '  '.join(f"{k}:{'SAME' if v else 'diff'}" for k, v in dist.items())
            print(f'  [5.1] applied-board identity -> {flags}'
                  + ('   <-- mute==haz STILL: affordance fix did NOT take' if dist.get('mute==haz') else ''))
        # Step 5.2: per-condition invalid-edit rate (should drop toward 0)
        rates = []
        for cond in ('mute', 'haz', 'met', 'full'):
            runs = cell.get((cond, board), [])
            if runs:
                rs = [invalid_edit_rate(rec)[0] for _, rec in runs]
                rs = [r for r in rs if r == r]
                rates.append(f"{cond}={mean(rs):.2f}" if rs else f"{cond}=na")
        if rates:
            print(f'  [5.2] invalid-edit rate (frac LLM iters naming a bad group): ' + '  '.join(rates))

    # salary decoy readout
    print('\n================ salary x2 DECOY readout ================')
    for cond in ('met', 'full', 'haz', 'mute'):
        runs = cell.get((cond, 'salary_x2'), [])
        if not runs:
            continue
        sal = rent = 0
        for _, rec in runs:
            ts, tr = salary_decoy_readout(rec)
            sal += ts
            rent += tr
        print(f'  {cond:5}: of {len(runs)} seeds -> touched salary(decoy)={sal}, touched rent(real)={rent}')

    # sanity
    sp = ROOT / 'sanity' / 'SANITY_RESULT.txt'
    if sp.exists():
        print('\n================ sanity canary ================')
        print('  ' + sp.read_text().strip().replace('\n', '\n  '))


if __name__ == '__main__':
    main()
