"""Synthetic landscape — ground-truth testbed for the search mechanism (experiment 4).

Replaces the Monopoly simulator + objective with a known function f(x) over a
small, interpretable design vector, keeping the designer task identical. Because
the structure is known, "did the model ever move the dominant dimension" is a
fact, not an inference.

The landscape family + edit vocabulary live in landscape.py; the closed-loop
driver is scripts/synth_design_loop.py (EXP-005).
"""
from synthetic.landscape import (
    G_STAR, BAND, VARIANTS,
    LandscapeSpec, make_landscape, build_family,
    evaluate, SyntheticDesign, apply_design, referenced_dims, greedy_optimize,
)

__all__ = [
    'G_STAR', 'BAND', 'VARIANTS',
    'LandscapeSpec', 'make_landscape', 'build_family',
    'evaluate', 'SyntheticDesign', 'apply_design', 'referenced_dims',
    'greedy_optimize',
]
