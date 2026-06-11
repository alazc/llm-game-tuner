"""Game-design optimisation package.

Components:
  strategy_pool    -- curated + sampled ParametricPlayer strategies
  simulate         -- runs games, collects per-game stats; bounds the engine's
                      trade loop so small boards terminate
  timing, run_log  -- wall-clock + cross-run instrumentation
  design_space     -- board-parameter vector <-> GameConfig encoding
  board_sources    -- the shared "five boards" probe set
  group_design     -- LLM parametric-edit vocabulary + eval helper
  skill_expression -- the optimized objective: skill-edge score + validity check
"""
