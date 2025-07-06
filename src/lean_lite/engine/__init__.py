"""
Engine package for Lean-Lite.

This package contains the core engine components that orchestrate
the algorithmic trading system.
"""

from .lean_engine import LeanEngine
# from .engine_config import EngineConfig

__all__ = ["LeanEngine", "EngineConfig"] 