"""
Data package for Lean-Lite.

This package handles data feeds, market data processing,
and data management for algorithmic trading.
"""

from .data_manager import DataManager
from .simple_data_engine import (
    SimpleDataEngine, SimpleDataFeed, SimpleAlpacaFeed, SimpleCache, validate_basic_data
)
from .alpaca_data import AlpacaData, AlpacaCredentials, AlpacaTimeframe

__all__ = [
    "DataManager",
    "SimpleDataEngine",
    "SimpleDataFeed", 
    "SimpleAlpacaFeed",
    "SimpleCache",
    "validate_basic_data",
    "AlpacaData",
    "AlpacaCredentials",
    "AlpacaTimeframe"
]
