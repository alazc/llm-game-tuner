"""EXP-005 synthetic-landscape closed loop (prior-free mechanism test).

The Monopoly board + game-eval are replaced by a known function f(x) over a design
vector x in [0,1]^d (synthetic/landscape.py). Everything ELSE is the EXP-003/004
harness: same designer-LLM wrapper + backend dispatch, greedy decoding, K=8, a
goal anchor + REDUCE framing, a JSON edit schema, and per-iteration logging. Because
the structure is KNOWN, "did the model ever move the dominant dimension / did it
settle at the interior optimum" are FACTS, not inferences.

Conditions:
  synth          primary feed (MET-analog TABLE): per-dim current values + quality +
                 score + delta + the goal anchor. Per-dim CONTRIBUTIONS are hidden —
                 the model must discover which lever matters from how the score moves
                 (the clean-map analog of Monopoly MET).
  synth_curve    representation arm: identical content, the quality history rendered
                 as a CURVE (shape descriptor).
  synth_verdict  representation arm: identical content, rendered as a one-line VERDICT.
  rand           random-edit baseline (no LLM): each iter set one uniformly-chosen dim
                 to a uniform value. The lever-identification / stopping reference.

Usage:
    # heuristic smoke (no GPU)
    set PYTHONPATH=. && python scripts/synth_design_loop.py --variant peaked \
        --backend heuristic --n-instances 2 --n-seeds 1 --K 4 \
        --out-dir report/figures/exp5_smoke/peaked

    # production (one variant, one model, GPU)
    set PYTHONPATH=. && python scripts/synth_design_loop.py --variant peaked \
        --backend local --model Qwen/Qwen2.5-7B-Instruct \
        --n-instances 5 --n-seeds 10 --K 8 --out-dir report/figures/exp5_run/peaked/7B
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from prompts.loader import load_prompt
from scripts.llm_design_loop import DesignerLLM, _RESPONSE_BLOCK
from synthetic.landscape import (
    G_STAR, BAND, VARIANTS, LandscapeSpec, build_family,
    evaluate, SyntheticDesign, apply_design, referenced_dims,
)

SYNTH_PROMPT = 'designer_llm_prompt_synth.txt'
CONDITIONS = ('synth', 'synth_curve', 'synth_verdict', 'rand')
LLM_CONDITIONS = ('synth', 'synth_curve', 'synth_verdict')
RENDER_OF = {'synth': 'table', 'synth_curve': 'curve', 'synth_verdict': 'verdict'}


# --------------------------------------------------------------------------- #
# Response parsing (synthetic edit schema)                                      #
# --------------------------------------------------------------------------- #

@dataclass
class SynthResponse:
    raw_text:      str
    design:        Optional[SyntheticDesign]
    rationale:     str
    converged:     bool
    parser_status: str
    error:         Optional[str] = None
    token_count:   Optional[int] = None


def parse_synth_response(text: str) -> SynthResponse:
    cands = [m.group(1) for m in _RESPONSE_BLOCK.finditer(text)] or [text.strip()]
    parsed, last_err = None, 'no parseable JSON in response'
    for c in cands:
        try:
            parsed = json.loads(c); break
        except json.JSONDecodeError as ex:
            last_err = f'JSONDecodeError: {ex.msg}'; parsed = None
    if not isinstance(parsed, dict):
        return SynthResponse(text, None, '', False, 'parser_failure', last_err)
    rationale = str(parsed.get('rationale', '')).strip()
    converged = bool(parsed.get('converged', False))
    interv = parsed.get('intervention', {}) or {}
    dim_set_raw = (interv.get('dim_set', {}) if isinstance(interv, dict) else {}) or {}
    if not isinstance(dim_set_raw, dict):
        return SynthResponse(text, None, rationale, converged,
                             'invalid_intervention', 'dim_set must be an object')
    dim_set: Dict[str, float] = {}
    for k, v in dim_set_raw.items():
        try:
            dim_set[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    design = SyntheticDesign(dim_set=dim_set, rationale=rationale[:500])
    return SynthResponse(text, design, rationale, converged, 'ok')


# --------------------------------------------------------------------------- #
# Diagnostic feed                                                               #
# --------------------------------------------------------------------------- #

def _goal_anchor(quality: float) -> str:
    """Mirror render_goal_anchor on the QUALITY scale; the design starts ABOVE g*."""
    if quality > G_STAR + 1e-9:
        tail = 'ABOVE target -> REDUCE (bring quality DOWN toward the target)'
    elif quality < G_STAR - 1e-9:
        tail = 'BELOW target -> INCREASE (bring quality UP toward the target)'
    else:
        tail = 'ON target'
    return f'design quality ends {quality:.3f}; target {G_STAR:.2f} -> {tail}'


def _quality_curve(quality_hist: List[float]) -> str:
    """Representation arm: the model's own quality trajectory as a checkpoint
    sequence + a one-line shape descriptor (info-matched to the table history)."""
    if not quality_hist:
        return '  (no trajectory yet)'
    seq = ' -> '.join(f'i{ix}:{q:.3f}' for ix, q in enumerate(quality_hist))
    swing = quality_hist[-1] - quality_hist[0]
    if abs(swing) < 0.01:
        shape = 'quality has barely moved (edits not finding the lever)'
    elif swing < 0:
        shape = 'quality is trending DOWN toward the target'
    else:
        shape = 'quality is trending UP, away from the target'
    return f'  {seq}\n  shape: {shape}'


def _quality_verdict(quality_hist: List[float], last_change: Optional[str]) -> str:
    """Representation arm: one-line natural-language verdict (same content)."""
    if not quality_hist:
        return '  no edits yet; quality is above target and must come down.'
    q = quality_hist[-1]
    if len(quality_hist) >= 2:
        d = quality_hist[-1] - quality_hist[-2]
        moved = ('the last edit moved quality toward the target'
                 if d < -1e-4 else
                 'the last edit moved quality away from the target'
                 if d > 1e-4 else 'the last edit did not change quality')
    else:
        moved = 'no prior edit to compare'
    return f'  quality is {q:.3f} ({"above" if q > G_STAR else "near/below"} the {G_STAR:.2f} target); {moved}.'


def build_synth_feed(spec: LandscapeSpec, x: np.ndarray, ev: dict,
                     prev_ev: Optional[dict], prior_iters: List[dict],
                     render: str) -> str:
    parts = ['## GOAL', '  ' + _goal_anchor(ev['quality'])]
    parts.append('## VALID DIMENSIONS (use ONLY these exact names in dim_set)')
    parts.append('  ' + ', '.join(spec.dim_names))

    parts.append('\n## CURRENT EVAL')
    parts.append(f'  quality: {ev["quality"]:.3f}')
    parts.append(f'  score: {ev["score"]:.4f}  (lower is better; 0 inside the dead-zone)')
    if prev_ev is not None:
        parts.append(f'  delta_score_vs_prev: {ev["score"] - prev_ev["score"]:+.4f}')

    # The map: current dimension VALUES only (contributions are hidden — the model
    # must discover which lever moves quality from how the score responds).
    parts.append('\n## CURRENT DESIGN (dimension values)')
    parts.append('  ' + '  '.join(f'{n}={x[i]:.2f}' for i, n in enumerate(spec.dim_names)))

    quality_hist = [it['quality'] for it in prior_iters if it.get('quality') is not None]
    quality_hist = quality_hist + [ev['quality']]
    if render == 'curve':
        parts.append('\n## QUALITY TRAJECTORY (your edits so far)')
        parts.append(_quality_curve(quality_hist))
    elif render == 'verdict':
        parts.append('\n## QUALITY VERDICT')
        parts.append(_quality_verdict(quality_hist, None))

    if prior_iters:
        parts.append('\n## PRIOR ITERATIONS')
        for it in prior_iters[-3:]:
            chg = it.get('changed_dims') or []
            parts.append(f'  iter {it.get("iter","?")}: set={chg}  '
                         f'quality={it.get("quality"):.3f}  '
                         f'rationale={ (it.get("rationale") or "")[:80]!r}')

    parts.append('\n## YOUR JOB')
    parts.append('  Propose ONE small intervention (set a few dimensions). Reply only '
                 'with the JSON schema described in the system prompt.')
    return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Iteration record                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class SynthIteration:
    iter:              int
    x:                 List[float]
    changed_dims:      List[str]
    edit_mag:          float            # L1 of x - x_prev over all dims
    dominant_edit_mag: float            # |delta| on the dominant dim (0 if none)
    quality:           Optional[float]
    score:             Optional[float]
    f:                 Optional[float]
    in_band:           Optional[bool]
    direction:         Optional[str]
    rationale:         str
    converged_request: bool
    convergence_padded: bool
    parser_status:     str
    parser_error:      Optional[str]
    parser_retry:      bool
    invalid_dims:      List[str]
    token_count:       Optional[int]
    wall_seconds:      float
    raw_response:      str
    condition:         str
    gen_cfg:           dict


def _changed(x_new: np.ndarray, x_prev: np.ndarray, names: List[str],
             tol: float = 1e-6) -> List[str]:
    return [names[i] for i in range(len(names)) if abs(x_new[i] - x_prev[i]) > tol]


_RETRY_REMINDER = ("Your previous response did not parse. Reply with exactly one "
                   "fenced ```json``` object: "
                   '{"rationale":"...","intervention":{"dim_set":{...}},"converged":false}.')


# --------------------------------------------------------------------------- #
# Trajectory runner                                                             #
# --------------------------------------------------------------------------- #

def run_synth_trajectory(spec: LandscapeSpec, seed: int, K: int, condition: str,
                         designer: Optional[DesignerLLM], out_path: Path,
                         max_new_token_retry: bool = True) -> List[SynthIteration]:
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition {condition!r}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w', encoding='utf-8')
    iters: List[SynthIteration] = []
    rng = np.random.default_rng(seed)
    render = RENDER_OF.get(condition, 'table')

    x = spec.x_start.copy()
    dom = spec.dominant_dim

    # Iteration 0: baseline (no edit).
    t0 = time.perf_counter()
    ev = evaluate(spec, x)
    rec0 = SynthIteration(
        iter=0, x=x.tolist(), changed_dims=[], edit_mag=0.0, dominant_edit_mag=0.0,
        quality=ev['quality'], score=ev['score'], f=ev['f'], in_band=ev['in_band'],
        direction=ev['direction'], rationale='[baseline]', converged_request=False,
        convergence_padded=False, parser_status='baseline', parser_error=None,
        parser_retry=False, invalid_dims=[], token_count=None,
        wall_seconds=time.perf_counter() - t0, raw_response='', condition=condition,
        gen_cfg=(designer.gen_cfg if designer else {'condition': condition}))
    iters.append(rec0); fh.write(json.dumps(asdict(rec0)) + '\n'); fh.flush()
    prev_ev = ev
    converged_done = False

    for k in range(1, K + 1):
        t0 = time.perf_counter()
        x_prev = x.copy()

        # ---------------- rand baseline (no LLM) ---------------- #
        if condition == 'rand':
            j = int(rng.integers(0, spec.n_dims))
            design = SyntheticDesign(dim_set={spec.dim_names[j]: float(rng.uniform(0.0, 1.0))},
                                     rationale='[rand]')
            x = apply_design(spec, design, base=x)
            ev = evaluate(spec, x)
            changed = _changed(x, x_prev, spec.dim_names)
            rec = SynthIteration(
                iter=k, x=x.tolist(), changed_dims=changed,
                edit_mag=float(np.abs(x - x_prev).sum()),
                dominant_edit_mag=(abs(x[dom] - x_prev[dom]) if dom is not None else 0.0),
                quality=ev['quality'], score=ev['score'], f=ev['f'], in_band=ev['in_band'],
                direction=ev['direction'], rationale='[rand]', converged_request=False,
                convergence_padded=False, parser_status='ok', parser_error=None,
                parser_retry=False, invalid_dims=[], token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=condition, gen_cfg={'condition': 'rand'})
            iters.append(rec); fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            prev_ev = ev
            continue

        # ---------------- converged: carry forward ---------------- #
        if converged_done:
            ev = evaluate(spec, x)
            rec = SynthIteration(
                iter=k, x=x.tolist(), changed_dims=[], edit_mag=0.0, dominant_edit_mag=0.0,
                quality=ev['quality'], score=ev['score'], f=ev['f'], in_band=ev['in_band'],
                direction=ev['direction'], rationale='[converged: carry-forward]',
                converged_request=True, convergence_padded=True, parser_status='ok',
                parser_error=None, parser_retry=False, invalid_dims=[], token_count=None,
                wall_seconds=time.perf_counter() - t0, raw_response='',
                condition=condition, gen_cfg=designer.gen_cfg)
            iters.append(rec); fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            continue

        # ---------------- LLM conditions ---------------- #
        prior = [asdict(it) for it in iters]
        feed = build_synth_feed(spec, x, prev_ev, prev_ev, prior, render)

        retry_used = False
        try:
            raw, ntok = designer.query(feed, iteration=k - 1)
            resp = parse_synth_response(raw); resp.token_count = ntok
        except Exception as ex:
            raw, resp = '', SynthResponse('', None, '', False, 'backend_failure',
                                          f'{type(ex).__name__}: {ex}')
        if resp.parser_status == 'parser_failure' and max_new_token_retry:
            retry_used = True
            try:
                raw2, ntok2 = designer.query(feed + '\n\n' + _RETRY_REMINDER, iteration=k - 1)
                resp2 = parse_synth_response(raw2); resp2.token_count = ntok2
                if resp2.parser_status == 'ok':
                    resp, raw = resp2, raw2
            except Exception:
                pass

        if resp.parser_status != 'ok' or resp.design is None:
            ev = evaluate(spec, x)
            rec = SynthIteration(
                iter=k, x=x.tolist(), changed_dims=[], edit_mag=0.0, dominant_edit_mag=0.0,
                quality=ev['quality'], score=ev['score'], f=ev['f'], in_band=ev['in_band'],
                direction=ev['direction'],
                rationale=resp.rationale or '[parser_failure carry-forward]',
                converged_request=resp.converged, convergence_padded=False,
                parser_status=resp.parser_status, parser_error=resp.error,
                parser_retry=retry_used, invalid_dims=[], token_count=resp.token_count,
                wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text[:2000],
                condition=condition, gen_cfg=designer.gen_cfg)
            iters.append(rec); fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
            continue

        invalid = sorted(referenced_dims(spec, resp.design))
        x = apply_design(spec, resp.design, base=x)
        ev = evaluate(spec, x)
        changed = _changed(x, x_prev, spec.dim_names)
        rec = SynthIteration(
            iter=k, x=x.tolist(), changed_dims=changed,
            edit_mag=float(np.abs(x - x_prev).sum()),
            dominant_edit_mag=(abs(x[dom] - x_prev[dom]) if dom is not None else 0.0),
            quality=ev['quality'], score=ev['score'], f=ev['f'], in_band=ev['in_band'],
            direction=ev['direction'], rationale=resp.rationale,
            converged_request=resp.converged, convergence_padded=False,
            parser_status='ok', parser_error=None, parser_retry=retry_used,
            invalid_dims=invalid, token_count=resp.token_count,
            wall_seconds=time.perf_counter() - t0, raw_response=resp.raw_text[:2000],
            condition=condition, gen_cfg=designer.gen_cfg)
        iters.append(rec); fh.write(json.dumps(asdict(rec)) + '\n'); fh.flush()
        prev_ev = ev
        if resp.converged:
            converged_done = True

    fh.close()
    return iters


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #

def _install_synth_heuristic(designer: DesignerLLM) -> None:
    """Replace the (Monopoly-shaped) heuristic cycle with synth-shaped canned
    responses so the no-GPU smoke exercises edit -> parse -> converge plumbing."""
    cycle = [
        {'rationale': 'try moving d00 down', 'intervention': {'dim_set': {'d00': 0.30}},
         'converged': False},
        {'rationale': 'nudge d01', 'intervention': {'dim_set': {'d01': 0.50}},
         'converged': False},
        {'rationale': 'no further gain', 'intervention': {'dim_set': {}}, 'converged': True},
    ]

    def _q(user_prompt, iteration: int = 0):
        return '```json\n' + json.dumps(cycle[iteration % len(cycle)]) + '\n```', None

    designer.query = _q


def _spec_meta(spec: LandscapeSpec) -> dict:
    return {'variant': spec.variant, 'landscape_seed': spec.seed,
            'aligned': spec.aligned,
            'n_dims': spec.n_dims, 'dim_names': spec.dim_names,
            'dominant_dim': spec.dominant_dim, 'decoy_dims': spec.decoy_dims,
            'inert_dims': spec.inert_dims, 'active_dims': spec.active_dims,
            'x_star': spec.x_star.tolist(), 'x_start': spec.x_start.tolist(),
            'amp_dominant': spec.amp_dominant}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=VARIANTS, required=True)
    ap.add_argument('--condition', choices=CONDITIONS, default='synth')
    ap.add_argument('--backend', choices=('local', 'openai', 'heuristic'), default='local')
    ap.add_argument('--model', default=None)
    ap.add_argument('--n-instances', type=int, default=5)
    ap.add_argument('--n-seeds', type=int, default=10)
    ap.add_argument('--family-base-seed', type=int, default=20260602)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--aligned', action='store_true',
                    help='EXP-006 sign-aligned geometry (reduce==lower; smooth '
                         'interior PEAKED well).')
    ap.add_argument('--out-dir', default='report/figures/exp5_run')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    family = build_family(args.variant, n_instances=args.n_instances,
                          base_seed=args.family_base_seed, aligned=args.aligned)

    designer = None
    if args.condition in LLM_CONDITIONS:
        if args.backend == 'heuristic':
            designer = DesignerLLM(backend='heuristic', goal_disclosure='open')
            _install_synth_heuristic(designer)                    # synth-shaped canned cycle
        else:
            designer = DesignerLLM(backend=args.backend, model_name=args.model,
                                   goal_disclosure='open')
        designer.system_prompt = load_prompt(SYNTH_PROMPT)        # hash-checked
        designer.gen_cfg['prompt_path'] = SYNTH_PROMPT
        designer.gen_cfg['experiment'] = 'EXP-005'
        designer.gen_cfg['variant'] = args.variant
        print(f'  designer prompt: {SYNTH_PROMPT} (sha-locked)  '
              f'backend={args.backend} model={args.model}')

    (out_dir / 'FAMILY.json').write_text(
        json.dumps([_spec_meta(s) for s in family], indent=2), encoding='utf-8')

    print(f'== EXP-005 {args.variant} :: cond={args.condition} '
          f'instances={args.n_instances} seeds={args.n_seeds} K={args.K} ==')
    for inst_idx, spec in enumerate(family):
        for s in range(args.n_seeds):
            seed = args.seed_offset + s * 1000 + 42
            tag = f'{args.variant}__inst{inst_idx}__ls{spec.seed}__seed{seed}'
            out_path = out_dir / f'{args.condition}__{tag}.jsonl'
            iters = run_synth_trajectory(spec, seed=seed, K=args.K,
                                         condition=args.condition, designer=designer,
                                         out_path=out_path)
            final = iters[-1]
            dom_touched = any(it.dominant_edit_mag > 1e-6 for it in iters)
            print(f'  [inst{inst_idx} seed{seed}] final score={final.score:.4f} '
                  f'q={final.quality:.3f} in_band={final.in_band} '
                  f'dom_moved={dom_touched} conv={any(it.converged_request for it in iters)}')
    print(f'Done -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
