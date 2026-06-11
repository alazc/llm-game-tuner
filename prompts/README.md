# LLM prompts used by `LLMPlayer`

This folder is the canonical record of how the LLM-based player in
`agents.LLMPlayer` is prompted. The files here are documentation /
artefacts, not code: changing them does **not** change runtime
behaviour. The runtime values live in
[`agents.py`](../agents.py)
(`LLMPlayer._SYSTEM_PROMPT`, `LLMPlayer._FEW_SHOT`,
`LLMPlayer._build_buy_prompt`).

## Files

| File | Format | Audience |
|---|---|---|
| `llm_player_prompts.txt` | Human-readable transcript laid out as a chat | Reviewers, paper readers, anyone wanting to see "what the LLM sees" at a glance |
| `llm_player_prompts.json` | Structured (system / few-shot list / template / settings) | Re-implementations, downstream tooling, automated extraction |

Both files describe the **same** prompt; the JSON is more machine-friendly,
the TXT is laid out the way a chat-template renderer would produce it.

## What the LLM is asked

A single binary BUY/PASS decision when the player lands on an unowned
property. The prompt includes:

- the player's cash
- the property's name, colour group, cost, base rent
- how many properties the player and opponent own (total + in this group)
- the size of the colour group on the current board

The LLM is asked to reply in a fixed two-line format
(`REASON: <one sentence>` then `ANSWER: BUY` or `ANSWER: PASS`).
A four-shot prompt covers the obvious cases (BUY when cheap and
unopposed, PASS when cash-tight or opponent-dominated, BUY to complete
a monopoly, PASS when liquidity matters).

## What the LLM is **not** asked

- House / hotel building decisions
- Trade decisions
- Jail-exit decisions

These keep the base `Player` logic. Routing every micro-decision through
the LLM would make games take minutes each rather than seconds, and the
buy decision is the strategically dominant one anyway.

## Why this format

We tried two model sizes:

- **Qwen2.5-0.5B-Instruct**: saturates to BUY on 100% of decisions
  even with the rich prompt. Insufficient capacity for the cash /
  opponent / monopoly-completion reasoning the prompt asks for.
- **Qwen2.5-1.5B-Instruct** (current default): differentiates correctly
  on the three diagnostic axes (cash level, opponent ownership in
  group, monopoly-completion opportunity). Roughly $1$s per call on
  a single-GPU machine; ~3.4s on CPU.

## Hash-locked prompt artefacts (round 1)

The round-1 experiments load every system prompt through
`prompts/loader.py:load_prompt`, which verifies the `.txt` against the
`.json` sidecar's `sha256` before returning the text. A mid-experiment
edit to a prompt without rehashing the sidecar aborts the run on the
next LLM call, instead of silently invalidating every trajectory.

| File | Used by | Notes |
|---|---|---|
| `character_llm_prompt.{txt,json}`     | `scripts/llm_character.py:CharacterLLMPlayer` (NEUTRAL-PLAYER) | MIRROR-H, PHASE-A-ROBUSTNESS |
| `designer_llm_prompt_open.{txt,json}` | `scripts/llm_design_loop.py:DesignerLLM` GOAL-OPEN | T-MUTE / T-HAZ / T-MET / T-FULL |
| `designer_llm_prompt_closed.{txt,json}` | `scripts/llm_design_loop.py:DesignerLLM` GOAL-CLOSED | T-BLIND only |
| `guided_player_system_prompt.{txt,json}` | `agents.py:LLMPlayer` (GUIDED-PLAYER) | Phase A original (already done) |

Update procedure when a prompt legitimately changes:
1. Edit the `.txt`.
2. `python -m prompts.loader --rehash <path>` — refreshes the `.json` sha.
3. Commit both files together so the diff is auditable.

## Known limitations

**Few-shot property-name overlap with default board (test-set
contamination risk).** The 4-shot examples in `LLMPlayer._FEW_SHOT`
reference four real properties present in `default_config.yaml`:
B1 Oriental Avenue, D1 St. James Place, C2 States Avenue, G1 Pacific
Avenue. When GUIDED-PLAYER plays on the canonical default board, the
model has been pre-coached on those exact properties.

Why the names were chosen: convenience during early development.
Layer-1 specificity (template matching) and layer-2 specificity (one
example per strategic axis) are deliberate and empirically necessary
at 1.5B parameters; layer-3 specificity (real names) is incidental
and could be replaced with synthetic placeholders without affecting
the prompt's pedagogical function.

When this matters: any future study using GUIDED-PLAYER on the
canonical default board.

When this does NOT matter: Phase A (mini-board, no overlap), MIRROR
and PHASE-A-ROBUSTNESS (NEUTRAL-PLAYER, no shots), TUNER and
ARCHITECT (no LLM player at all).

Mitigation if needed: replace `_FEW_SHOT` with synthetic property
names that don't appear in any test-bench config.
