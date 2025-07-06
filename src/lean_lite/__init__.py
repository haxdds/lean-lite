"""
Lean-Lite: A lightweight QuantConnect LEAN runtime.

This package provides a simplified, containerized version of the QuantConnect LEAN engine
for running algorithmic trading strategies with minimal overhead.
"""

__version__ = "0.1.0"
__author__ = "Lean-Lite Team"

from .engine import LeanEngine
from .config import Config

__all__ = ["LeanEngine", "Config"] 