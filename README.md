# llm-game-tuner

**Feedback Representation Is Not the Bottleneck: A Small LLM Game Designer Picks
Right, Edits Wrong.**

A small open model (Qwen2.5-Instruct, 1.5B/7B) acts as a game designer: it
edits a Monopoly board, exposed as 21 continuous knobs, to make the game reward
skill. The target is skill expression — the wealth share an expert wins
over a random opponent — and the job is to move it from 0.73 into a 0.57–0.63
zone across an eight-edit closed loop. Is the bottleneck the information the
designer gets, or its representation?

See the project report (`report.pdf`) for the full write-up.

## Quickstart

```bash
pip install -e .          # editable install; puts all packages on sys.path
python -m experiments.<name>   # run / analyze an experiment
```

No `PYTHONPATH` needed after the editable install.

## Layout

| Path | What |
|---|---|
| `monopoly/` | The game engine — public gamescomputersplay simulator (unlicensed; vendored @ `7c4df31`) + local patches (config-injection, variable-length boards). Patched files carry a `# MODIFIED from upstream` header. |
| `config.py` `settings.py` `player_settings.py` `agents.py` | Glue: `GameConfig` serialization, engine settings, player agents. Kept as flat modules because the engine imports them by bare name. |
| `optimizer/` | Core library: `skill_expression` (the optimized objective — skill-edge score + validity check), `design_space` (board↔vector), `board_sources`/`exp_boards` (starting boards + traps), `group_design` (edit vocabulary + eval), `simulate` (bounded trade loop), `strategy_pool` (the skill ladder), `timing`/`run_log`. |
| `scripts/` | The designer loops behind the findings: `llm_design_loop` (model writes the edit — the failing baseline), `scaffold_designer_loop` (name-and-measure — the 9/9 repair), `choice_designer_loop` (menu choice), `scripted_optimizer_loop` (no-model scripted rule), `proposal_probe`/`screening_optimizer` (sample-and-select), `mech_probe_loop` (belief vs. action probes), `synth_design_loop` (game-free synthetic problem). Plus `hazard_curves`, `novelty_search`. |
| `experiments/` | Analysis runners for the report (`exp0_*`, `exp4_*`, `exp5_*`, `exp_lever_check`, `exp_head_to_head`). See `experiments/README.md`. |
| `synthetic/` | Ground-truth f(x) landscape (game-removed control): one payoff knob, three dead ends, eight inert. |
| `prompts/` | Hash-locked LLM prompts (`loader.py` verifies SHA256 sidecars). |
| `results/` `logs/` `report/figures/` | Run artifacts (gitignored). |

## Licensing

The original work in this repository is released under the **MIT License** (see
`LICENSE`).

This does not include the vendored game engine under `monopoly/`, which is a
copy of [`gamescomputersplay/monopoly`](https://github.com/gamescomputersplay/monopoly)
@ `7c4df31`. That upstream repository carried no license at the vendored commit,
so its author retains all rights; the copy here is included for research
reproducibility, with attribution, and is not covered by the MIT license
above. Full details and a list of local patches are in `monopoly/NOTICE`.
