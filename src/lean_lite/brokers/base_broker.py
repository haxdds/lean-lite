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
    def market_buy(self, symbol: str, quantity: int):
        """Place a market buy order."""
        pass
        
    @abstractmethod 
    def market_sell(self, symbol: str, quantity: int):
        """Place a market sell order."""
        pass
        
    @abstractmethod
    def limit_buy(self, symbol: str, quantity: int, limit_price: Decimal):
        """Place a limit buy order."""
        pass
        
    @abstractmethod
    def limit_sell(self, symbol: str, quantity: int, limit_price: Decimal):
        """Place a limit sell order."""
        pass
        
    @abstractmethod
    def stop_buy(self, symbol: str, quantity: int, stop_price: Decimal):
        """Place a stop buy order."""
        pass
        
    @abstractmethod
    def stop_sell(self, symbol: str, quantity: int, stop_price: Decimal):
        """Place a stop sell order."""
        pass
        
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        pass