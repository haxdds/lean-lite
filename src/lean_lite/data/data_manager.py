"""
Data manager for Lean-Lite.

This module provides an abstract base class for handling data feeds and market data processing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DataManager(ABC):
    """Abstract base class for managing data feeds and market data processing."""
    
    def __init__(self, broker, config):
        """Initialize the data manager."""
        self.broker = broker
        self.config = config
        self.subscribed_symbols: List[str] = []
        self.latest_data: Dict[str, Any] = {}
        
    @abstractmethod
    def initialize(self):
        """Initialize the data manager."""
        pass
        
    @abstractmethod
    def subscribe_symbol(self, symbol: str):
        """Subscribe to real-time data for a symbol."""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str = "1Min", limit: int = 100):
        """Get historical market data."""
        pass
    
    def _on_bar_update(self, bar: Dict[str, Any]):
        """Handle real-time bar updates."""
        symbol = bar["symbol"]
        self.latest_data[symbol] = {
            "symbol": symbol,
            "open": float(bar["open"]),
            "high": float(bar["high"]), 
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": int(bar["volume"]),
            "timestamp": bar["timestamp"],
            "type": "bar"
        }
        logger.debug(f"Bar update for {symbol}: ${bar['close']}")
    
    def _on_trade_update(self, trade: Dict[str, Any]):
        """Handle real-time trade updates."""
        symbol = trade["symbol"]
        self.latest_data[symbol] = {
            "symbol": symbol,
            "price": float(trade["price"]),
            "size": int(trade["size"]), 
            "timestamp": trade["timestamp"],
            "type": "trade"
        }
        logger.debug(f"Trade update for {symbol}: ${trade['price']}")
        
    @abstractmethod
    def process_data(self) -> Dict[str, Any]:
        """Process incoming market data."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from data streams."""
        pass