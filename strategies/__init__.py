"""
Strategies package for Lean-Lite.

This package contains example trading strategies and strategy templates
for the Lean-Lite runtime.
"""

from .example_strategy import ExampleStrategy
from .moving_average_crossover import MovingAverageCrossover
from .qc_example_strategy import QCExampleStrategy

__all__ = [
    "ExampleStrategy", 
    "MovingAverageCrossover",
    "QCExampleStrategy"
] 