"""Novelty search demo: pick K=3 maximally distant non-degenerate boards.

Direct analog of Isaksen 2018 §VIII-D ("find k unique games"): given a pool
of candidate game variants, select K of them that are as far apart in
parameter space as possible, subject to a quality (non-degeneracy) gate.

Pipeline:
  1. Build a candidate pool by reading optimizer JSONL run logs (one entry
     per evaluation; each carries `vec`, `score`, and a `metrics` dict).
  2. Filter entries with combined `score < score_threshold` (default 1.2).
     Default-board score is ~1.46 at n=1000, so 1.2 keeps non-degenerate
     designs without being too tight.
  3. Normalize each design vector to [0, 1]^66 via DesignSpace.bounds() and
     compute the full N x N pairwise Euclidean distance matrix.
  4. Brute-force enumerate every C(N, K) triple, score it as
     `min(pairwise distances within the triple)`, and pick the argmax.
     For K=3 and N a few hundred this is sub-second.
  5. Persist the chosen triple + its profile to JSON; render each board in
     two styles (shrunk + legacy) on a single combined PDF figure.

Usage (from the repo root):
    set PYTHONPATH=. && python scripts/novelty_search.py \\
        --runs logs/optimizer/ga_2p.jsonl logs/optimizer/ga_3p.jsonl \\
        --score-threshold 1.2 --k 3 \\
        --out-dir report/figures/novelty
"""
import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import GameConfig
from optimizer import run_log
from optimizer.design_space import DesignSpace
# matplotlib + render_board are imported lazily inside render_triple() so this
# module's pure helpers (max_min_tuple, normalize_vecs, ...) import without a
# matplotlib/render_board dependency.


# --------------------------------------------------------------------------- #
# Pool collection + normalization                                              #
# --------------------------------------------------------------------------- #

def load_pool(run_paths: List[str], score_threshold: float) -> List[dict]:
    """Return [{run, idx, score, metrics, vec}, ...] across all run logs,
    keeping only entries with score < threshold."""
    out: List[dict] = []
    for run_path in run_paths:
        p = Path(run_path)
        if not p.exists():
            print(f'  [warn] {run_path} not found, skipping')
            continue
        with open(p) as f:
            for li, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get('score', float('inf')) >= score_threshold:
                    continue
                out.append({
                    'run':     p.stem,
                    'idx':     li,
                    'score':   float(e['score']),
                    'metrics': dict(e.get('metrics', {})),
                    'vec':     list(e['vec']),
                })
    return out


def normalize_vecs(vecs: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Map each dim from [lo, hi] to [0, 1]. Binary dims are already 0/1 so
    the transform is identity for them given their (0, 1) bounds."""
    lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (vecs - lo) / span


# --------------------------------------------------------------------------- #
# K-tuple max-min selection                                                    #
# --------------------------------------------------------------------------- #

def max_min_tuple(dist_matrix: np.ndarray, K: int) -> Tuple[List[int], float]:
    """Brute-force enumerate C(N, K) tuples and return (indices, min_pairwise_dist)
    of the tuple with the largest min-pairwise-distance.

    For K=3 this is O(N^3) which is fine at N up to ~1000. For larger K we
    would switch to a greedy farthest-first heuristic.
    """
    N = dist_matrix.shape[0]
    if N < K:
        raise ValueError(f'Pool has only {N} candidates; K={K} requires >= K.')
    best_d = -1.0
    best_tuple: Tuple[int, ...] = tuple(range(K))
    for tup in combinations(range(N), K):
        # min over all C(K,2) pairwise distances within the tuple.
        min_d = float('inf')
        for i_a, i_b in combinations(tup, 2):
            d = dist_matrix[i_a, i_b]
            if d < min_d:
                min_d = d
                if min_d <= best_d:
                    break   # cannot beat best with this tuple
        if min_d > best_d:
            best_d = float(min_d)
            best_tuple = tup
    return list(best_tuple), best_d


# --------------------------------------------------------------------------- #
# Figure rendering                                                             #
# --------------------------------------------------------------------------- #

def render_triple(canonical_cfg: GameConfig, vecs: List[np.ndarray],
                  labels: List[str], out_path: Path,
                  removal_direction: str = 'cheapest'):
    """Render the K boards in two styles on a single 2-row figure.
    Top row: shrunk layout (true post-shrinkage cell count).
    Bottom row: legacy layout (canonical 40-cell perimeter, removed cells
    shown as FreeParking placeholders so the reader can see WHICH props
    each board dropped)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.render_board import draw_board as draw_board_shrunk
    from scripts.render_board_legacy import draw_board as draw_board_legacy

    space = DesignSpace(canonical_cfg, removal_direction=removal_direction)
    K = len(vecs)
    fig, axes = plt.subplots(2, K, figsize=(5.0 * K, 9.4))
    default_decoded_legacy = space.decode_as_substituted(space.identity_vec())
    for k in range(K):
        v = np.asarray(vecs[k], dtype=np.float64)
        cfg_shrunk = space.decode(v)
        cfg_legacy = space.decode_as_substituted(v)
        draw_board_shrunk(axes[0, k], cfg_shrunk, default_cfg=canonical_cfg,
                          title=f'{labels[k]}  (shrunk)', annotate_changes=False)
        draw_board_legacy(axes[1, k], cfg_legacy,
                          default_cfg=default_decoded_legacy,
                          title=f'{labels[k]}  (legacy)', annotate_changes=True)
    fig.suptitle('Novelty search: K=3 maximally distant non-degenerate boards',
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def write_profile_table(rows: List[dict], out_path: Path):
    """Emit a markdown-readable per-metric profile table for the K boards."""
    cols = ['label', 'source', 'score', 'mean_fairness', 'max_fairness',
            'mean_rounds', 'mean_draw_rate', 'mean_transfer_rate',
            'kept_props']
    lines = []
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
    for r in rows:
        m = r['metrics']
        lines.append('| ' + ' | '.join([
            r['label'],
            r['source'],
            f"{r['score']:.3f}",
            f"{m.get('mean_fairness', float('nan')):.3f}",
            f"{m.get('max_fairness', float('nan')):.3f}",
            f"{m.get('mean_rounds', float('nan')):.1f}",
            f"{m.get('mean_draw_rate', float('nan')):.3f}",
            f"{m.get('mean_transfer_rate', float('nan')):.1f}",
            str(r['kept_props']),
        ]) + ' |')
    out_path.write_text('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+',
                    default=['logs/optimizer/ga_2p.jsonl',
                             'logs/optimizer/ga_3p.jsonl'],
                    help='Optimizer JSONL run logs to combine into the candidate pool.')
    ap.add_argument('--score-threshold', type=float, default=1.2,
                    help='Drop entries with combined score >= this. The default '
                         'board scores ~1.46 at n=1000 so 1.2 is a soft '
                         'non-degeneracy gate.')
    ap.add_argument('--k', type=int, default=3,
                    help='Number of maximally-distant boards to select.')
    ap.add_argument('--config', default='default_config.yaml',
                    help='Canonical config used to build DesignSpace bounds + render layouts.')
    ap.add_argument('--removal-direction', default='cheapest',
                    choices=('cheapest', 'expensive', 'middle'))
    ap.add_argument('--out-dir', default='report/figures/novelty')
    ap.add_argument('--seed', type=int, default=0,
                    help='Tie-breaker seed (currently unused but recorded for reproducibility).')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with run_log.track(experiment='novelty', script=__file__,
                       args=vars(args), out_dir=out_dir) as run:
        return _run_novelty(args, out_dir, run)


def _run_novelty(args, out_dir: Path, run) -> int:
    canonical_cfg = GameConfig.from_yaml(args.config)
    space = DesignSpace(canonical_cfg, removal_direction=args.removal_direction)
    bounds = space.bounds()

    print(f'Loading candidate pool from {len(args.runs)} run log(s) '
          f'(score < {args.score_threshold})...')
    pool = load_pool(args.runs, args.score_threshold)
    print(f'  {len(pool)} non-degenerate candidates')

    if len(pool) < args.k:
        raise SystemExit(f'Pool has only {len(pool)} entries; need >= K={args.k}. '
                         'Lower --score-threshold or include more --runs.')

    # Pad legacy 45-dim vectors up to 66 dims would be needed if any are present;
    # this script assumes mask-based 66-dim vecs (current encoding) and rejects
    # legacy entries to avoid a silent dimension-mismatch in distance comp.
    pool = [e for e in pool if len(e['vec']) == space.n_dims]
    if not pool:
        raise SystemExit('No 66-dim entries in pool. Re-run optimisation with the '
                         'current mask encoding to populate logs/optimizer/.')

    vecs = np.asarray([e['vec'] for e in pool], dtype=np.float64)
    norm = normalize_vecs(vecs, bounds)

    # Pairwise distance matrix
    diff = norm[:, None, :] - norm[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(D, np.inf)   # never pair with self

    print(f'Searching for max-min K={args.k} tuple over {len(pool)} '
          f'candidates (C({len(pool)}, {args.k}) = {_n_choose_k(len(pool), args.k):,} tuples)...')
    chosen, min_d = max_min_tuple(D, args.k)
    print(f'  best min-pairwise distance = {min_d:.4f} (in normalized 66-dim space)')

    # Build output rows
    rows = []
    chosen_vecs: List[np.ndarray] = []
    for ci, idx in enumerate(chosen):
        e = pool[idx]
        v = np.asarray(e['vec'], dtype=np.float64)
        kept = int(round(np.sum(v[space.N_CONT:] >= 0.5)))
        rows.append({
            'label':       f'novelty_{ci+1}',
            'source':      f"{e['run']}#{e['idx']}",
            'score':       e['score'],
            'metrics':     e['metrics'],
            'kept_props':  kept,
            'vec':         e['vec'],
        })
        chosen_vecs.append(v)

    # Persist + render (out_dir already created in main())
    json_out = out_dir / 'novelty_triple.json'
    with open(json_out, 'w') as f:
        json.dump({
            'config': vars(args),
            'pool_size_total':       len(pool),
            'min_pairwise_distance': float(min_d),
            'chosen':                rows,
        }, f, indent=2)
    print(f'  triple summary -> {json_out}')

    table_out = out_dir / 'novelty_profile.md'
    write_profile_table(rows, table_out)
    print(f'  profile table  -> {table_out}')

    print('Rendering combined figure...')
    pdf_out = out_dir / 'novelty_triple.pdf'
    png_out = out_dir / 'novelty_triple.png'
    render_triple(canonical_cfg, chosen_vecs,
                  [r['label'] for r in rows], pdf_out,
                  removal_direction=args.removal_direction)
    render_triple(canonical_cfg, chosen_vecs,
                  [r['label'] for r in rows], png_out,
                  removal_direction=args.removal_direction)
    print(f'  figure         -> {pdf_out}')

    # Console echo of the profile table
    print()
    print(table_out.read_text())
    return 0


def _n_choose_k(n: int, k: int) -> int:
    if k > n or k < 0:
        return 0
    num = 1
    den = 1
    for i in range(k):
        num *= (n - i)
        den *= (i + 1)
    return num // den


if __name__ == '__main__':
    raise SystemExit(main() or 0)
