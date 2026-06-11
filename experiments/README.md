# Experiments

**Read-only** analysis runners over logged run artifacts — they reconstruct
metrics, with no model calls or reruns. The designer *loops* that produced the
artifacts live in [`../scripts/`](../scripts) (e.g. `llm_design_loop`,
`scaffold_designer_loop`, `synth_design_loop`); these modules read their
`.jsonl` output under `report/figures/` and `results/`/`logs/`. Run with
`python -m experiments.<name>` after `pip install -e .`. The narrative they
back is in the accompanying project report. The `modal_*.py` files at the repo
root launch the loops on Modal.

| Module | What it does |
|---|---|
| `exp_lever_check` | GO/NO-GO gate: does any design lever move the optimized metric (skill share) on a playable board? |
| `exp_head_to_head` | The two named metrics on the default board: skill share (optimized) + ladder monotonicity (validity diagnostic). |
| `exp0_analyze` / `exp0_explore_7b` | The matched four-report comparison (Findings 1–2): info-gradient + chart-vs-table contrast, exploration-vs-failure classification. |
| `exp4_analyze` | Belief-vs-action probes (Finding 3): generation vs credit vs prior classification (pick-best / hold-good / fix-bad). |
| `exp5_analyze` / `exp5_posthoc` | Game-removed synthetic landscape (Finding 4): lever identification + the post-hoc optimum-sense follow-up. |

Shared dependencies:
- Objective: `optimizer/skill_expression.py` — the skill-edge score + validity check.
- Encoder/eval: `optimizer/{design_space,group_design,simulate,strategy_pool,exp_boards}.py`.
