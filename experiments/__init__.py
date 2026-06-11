"""Experiment runners and analyzers.

Each module is a standalone entry point (python -m experiments.<name>).
Runners orchestrate the designers and the objective against the simulator or
the synthetic landscape, and write results under results/ + logs/ via
optimizer.run_log; analyzers are read-only over the produced artifacts.
"""
