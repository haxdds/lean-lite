"""
Base algorithm class for Lean-Lite.

This module provides the base class that all trading strategies
must inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """Base class for all trading algorithms."""
    
    def __init__(self):
        """Initialize the base algorithm."""
        self.symbol = None
        self.position_size = 0
        self.broker = None
        self.data_manager = None
        
    def set_symbol(self, symbol: str):
        """Set the trading symbol."""
        self.symbol = symbol
        
    def set_position_size(self, size: int):
        """Set the position size."""
        self.position_size = size
        
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in the symbol."""
        if self.broker and hasattr(self.broker, 'get_positions'):
            try:
                positions = self.broker.get_positions()
                return symbol in positions
            except Exception as e:
                logger.error(f"Error checking position for {symbol}: {e}")
                return False
        return False
        
    def buy(self, symbol: str, quantity: int):
        """Place a buy order."""
        if self.broker:
            try:
                order = self.broker.buy(symbol, quantity)
                logger.info(f"Buy order placed: {order.id} for {quantity} shares of {symbol}")
                return order
            except Exception as e:
                logger.error(f"Error placing buy order: {e}")
                raise
        else:
            raise RuntimeError("No broker available")
            
    def sell(self, symbol: str, quantity: int):
        """Place a sell order."""
        if self.broker:
            try:
                order = self.broker.sell(symbol, quantity)
                logger.info(f"Sell order placed: {order.id} for {quantity} shares of {symbol}")
                return order
            except Exception as e:
                logger.error(f"Error placing sell order: {e}")
                raise
        else:
            raise RuntimeError("No broker available")
    
    @abstractmethod
    def initialize(self):
        """Initialize the algorithm."""
        pass
        
    @abstractmethod
    def on_data(self, data: Dict[str, Any]):
        """Handle incoming market data."""
        pass
        
    def on_order_filled(self, order):
        """Handle order fill events."""
        pass
        
    def on_error(self, error):
        """Handle errors."""
        logger.error(f"Algorithm error: {error}") 