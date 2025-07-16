"""
Base broker interface for Lean-Lite.

This module provides the base interface that all broker implementations
must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from decimal import Decimal


class BaseBroker(ABC):
    """Base interface for all broker implementations."""
    
    @abstractmethod
    def connect(self):
        """Connect to the broker."""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker."""
        pass
        
    @abstractmethod
    def market_order(self, symbol: str, quantity: int, side: str):
        """Place a market order.
        
        Args:
            symbol (str): The symbol to trade
            quantity (int): Number of shares/contracts
            side (str): 'buy' or 'sell'
        """
        pass

    @abstractmethod
    def limit_order(self, symbol: str, quantity: int, limit_price: Decimal, side: str):
        """Place a limit order.
        
        Args:
            symbol (str): The symbol to trade
            quantity (int): Number of shares/contracts 
            limit_price (Decimal): Limit price
            side (str): 'buy' or 'sell'
        """
        pass

    @abstractmethod
    def stop_order(self, symbol: str, quantity: int, stop_price: Decimal, side: str):
        """Place a stop order.
        
        Args:
            symbol (str): The symbol to trade
            quantity (int): Number of shares/contracts
            stop_price (Decimal): Stop trigger price
            side (str): 'buy' or 'sell'
        """
        pass

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        pass