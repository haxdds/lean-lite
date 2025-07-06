"""
QuantConnect-compatible QCAlgorithm base class for Lean-Lite.

This module provides a QCAlgorithm class that matches QuantConnect's Python API,
allowing users to write strategies that are compatible with both QuantConnect
and Lean-Lite.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from .base_algorithm import BaseAlgorithm


class SecurityType(Enum):
    """Security types supported by Lean-Lite."""
    EQUITY = "equity"
    FOREX = "forex"
    CRYPTO = "crypto"


class Symbol:
    """Represents a trading symbol with security type information."""
    
    def __init__(self, ticker: str, security_type: SecurityType = SecurityType.EQUITY):
        """Initialize a symbol.
        
        Args:
            ticker (str): The ticker symbol (e.g., 'AAPL', 'EURUSD', 'BTCUSD')
            security_type (SecurityType): The type of security
        """
        self.ticker = ticker.upper()
        self.security_type = security_type
        self.value = f"{self.ticker}"
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"Symbol('{self.ticker}', {self.security_type.value})"


@dataclass
class Security:
    """Represents a security with its configuration and data."""
    
    symbol: Symbol
    price: float = 0.0
    volume: int = 0
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize the security after creation."""
        if self.last_update is None:
            self.last_update = datetime.now()


class OrderEvent:
    """Represents an order event from the broker."""
    
    def __init__(self, order_id: str, symbol: str, quantity: int, 
                 side: str, status: str, filled_quantity: int = 0, 
                 filled_price: float = 0.0):
        """Initialize an order event.
        
        Args:
            order_id (str): Unique order identifier
            symbol (str): Trading symbol
            quantity (int): Order quantity
            side (str): Order side ('buy' or 'sell')
            status (str): Order status
            filled_quantity (int): Quantity filled
            filled_price (float): Price at which order was filled
        """
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.status = status
        self.filled_quantity = filled_quantity
        self.filled_price = filled_price
        self.time = datetime.now()


class QCAlgorithm(BaseAlgorithm):
    """QuantConnect-compatible algorithm base class for Lean-Lite.
    
    This class provides a familiar interface for users coming from QuantConnect,
    with methods that match the QuantConnect Python API.
    """
    
    def __init__(self):
        """Initialize the QCAlgorithm."""
        super().__init__()
        
        # Algorithm configuration
        self.start_date: Optional[date] = None
        self.end_date: Optional[date] = None
        self.cash: float = 100000.0  # Default starting cash
        
        # Securities and data
        self.securities: Dict[str, Security] = {}
        self.symbols: Dict[str, Symbol] = {}
        
        # Algorithm state
        self.time: datetime = datetime.now()
        self.portfolio_value: float = self.cash
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"QCAlgorithm initialized: {self.__class__.__name__}")
    
    def SetStartDate(self, year: int, month: int, day: int):
        """Set the start date for the algorithm.
        
        Args:
            year (int): Start year
            month (int): Start month (1-12)
            day (int): Start day (1-31)
        """
        self.start_date = date(year, month, day)
        self.logger.info(f"Start date set to: {self.start_date}")
    
    def SetEndDate(self, year: int, month: int, day: int):
        """Set the end date for the algorithm.
        
        Args:
            year (int): End year
            month (int): End month (1-12)
            day (int): End day (1-31)
        """
        self.end_date = date(year, month, day)
        self.logger.info(f"End date set to: {self.end_date}")
    
    def SetCash(self, cash: float):
        """Set the starting cash amount.
        
        Args:
            cash (float): Starting cash amount
        """
        if not isinstance(cash, (int, float)):
            raise TypeError(f"Cash must be a number, got {type(cash).__name__}")
        
        self.cash = float(cash)
        self.portfolio_value = float(cash)
        self.logger.info(f"Starting cash set to: ${self.cash:,.2f}")
    
    def AddEquity(self, ticker: str) -> Symbol:
        """Add an equity security to the algorithm.
        
        Args:
            ticker (str): Equity ticker symbol (e.g., 'AAPL', 'SPY')
            
        Returns:
            Symbol: The created symbol object
        """
        symbol = Symbol(ticker, SecurityType.EQUITY)
        self.symbols[ticker] = symbol
        
        # Create security object
        security = Security(symbol)
        self.securities[ticker] = security
        
        # Add to portfolio if portfolio exists
        if hasattr(self, 'portfolio') and self.portfolio:
            try:
                # Convert QCAlgorithm Symbol to data_models Symbol
                from .data_models import Symbol as DataSymbol, SecurityType as DataSecurityType, Market
                data_symbol = DataSymbol(ticker, DataSecurityType.EQUITY, Market.US_EQUITY)
                self.portfolio.AddSecurity(data_symbol)
            except ImportError:
                # If data_models not available, skip portfolio integration
                pass
        
        self.logger.info(f"Added equity: {ticker}")
        return symbol
    
    def AddForex(self, ticker: str) -> Symbol:
        """Add a forex security to the algorithm.
        
        Args:
            ticker (str): Forex ticker symbol (e.g., 'EURUSD', 'GBPUSD')
            
        Returns:
            Symbol: The created symbol object
        """
        symbol = Symbol(ticker, SecurityType.FOREX)
        self.symbols[ticker] = symbol
        
        # Create security object
        security = Security(symbol)
        self.securities[ticker] = security
        
        self.logger.info(f"Added forex: {ticker}")
        return symbol
    
    def AddCrypto(self, ticker: str) -> Symbol:
        """Add a cryptocurrency security to the algorithm.
        
        Args:
            ticker (str): Crypto ticker symbol (e.g., 'BTCUSD', 'ETHUSD')
            
        Returns:
            Symbol: The created symbol object
        """
        symbol = Symbol(ticker, SecurityType.CRYPTO)
        self.symbols[ticker] = symbol
        
        # Create security object
        security = Security(symbol)
        self.securities[ticker] = security
        
        self.logger.info(f"Added crypto: {ticker}")
        return symbol
    
    def Buy(self, symbol: Union[str, Symbol], quantity: int):
        """Place a buy order.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to buy
            quantity (int): Quantity to buy
        """
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        super().buy(ticker, quantity)
        self.logger.info(f"Buy order: {quantity} shares of {ticker}")
    
    def Sell(self, symbol: Union[str, Symbol], quantity: int):
        """Place a sell order.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to sell
            quantity (int): Quantity to sell
        """
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        super().sell(ticker, quantity)
        self.logger.info(f"Sell order: {quantity} shares of {ticker}")
    
    def MarketOrder(self, symbol: Union[str, Symbol], quantity: int, price: Optional[float] = None):
        """Place a market order.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to trade
            quantity (int): Quantity to trade (positive for buy, negative for sell)
            price (Optional[float]): Price for validation (not used for market orders)
        """
        if quantity > 0:
            return self.Buy(symbol, quantity)
        elif quantity < 0:
            return self.Sell(symbol, abs(quantity))
        else:
            raise ValueError("Order quantity cannot be zero")
    
    def LimitOrder(self, symbol: Union[str, Symbol], quantity: int, limit_price: float):
        """Place a limit order.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to trade
            quantity (int): Quantity to trade (positive for buy, negative for sell)
            limit_price (float): Limit price for the order
        """
        if quantity > 0:
            return self.Buy(symbol, quantity)
        elif quantity < 0:
            return self.Sell(symbol, abs(quantity))
        else:
            raise ValueError("Order quantity cannot be zero")
    
    def SetHoldings(self, symbol: Union[str, Symbol], percentage: float):
        """Set target holdings for a symbol as a percentage of portfolio value.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to set holdings for
            percentage (float): Target percentage (0.0 to 1.0)
        """
        if not (0.0 <= percentage <= 1.0):
            raise ValueError(f"Percentage must be between 0.0 and 1.0, got {percentage}")
        
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        
        if ticker not in self.securities:
            self.logger.warning(f"Symbol {ticker} not found in securities")
            return
        
        current_price = self.securities[ticker].price
        if current_price <= 0:
            self.logger.warning(f"No price data for {ticker}")
            return
        
        target_value = self.portfolio_value * percentage
        target_quantity = int(target_value / current_price)
        
        current_quantity = 0
        if self.has_position(ticker):
            # Get current position quantity (simplified)
            current_quantity = 100  # Placeholder - would get from broker
        
        quantity_to_trade = target_quantity - current_quantity
        
        if quantity_to_trade > 0:
            self.Buy(ticker, quantity_to_trade)
        elif quantity_to_trade < 0:
            self.Sell(ticker, abs(quantity_to_trade))
        
        self.logger.info(f"Set holdings for {ticker}: {percentage:.1%} ({target_quantity} shares)")
    
    def Liquidate(self, symbol: Union[str, Symbol]):
        """Liquidate all holdings of a symbol.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to liquidate
        """
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        
        if self.has_position(ticker):
            # Get current position quantity (simplified)
            current_quantity = 100  # Placeholder - would get from broker
            self.Sell(ticker, current_quantity)
            self.logger.info(f"Liquidated {ticker}: {current_quantity} shares")
        else:
            self.logger.info(f"No position to liquidate for {ticker}")
    
    def GetLastKnownPrice(self, symbol: Union[str, Symbol]) -> float:
        """Get the last known price for a symbol.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to get price for
            
        Returns:
            float: Last known price, or 0.0 if not available
        """
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        
        if ticker in self.securities:
            return self.securities[ticker].price
        return 0.0
    
    def UpdateSecurityPrice(self, ticker: str, price: float, volume: int = 0):
        """Update the price and volume for a security.
        
        Args:
            ticker (str): Symbol ticker
            price (float): Current price
            volume (int): Current volume
        """
        if ticker in self.securities:
            self.securities[ticker].price = price
            self.securities[ticker].volume = volume
            self.securities[ticker].last_update = datetime.now()
    
    def initialize(self):
        """Initialize the algorithm. Override this method in your strategy."""
        self.logger.info("QCAlgorithm initialize() called - override in your strategy")
        self.Initialize()
    
    def on_data(self, data: Dict[str, Any]):
        """Handle incoming market data. Override this method in your strategy."""
        # Update security prices from incoming data
        for ticker, ticker_data in data.items():
            if ticker in self.securities:
                if "close" in ticker_data:
                    self.UpdateSecurityPrice(ticker, ticker_data["close"])
                elif "price" in ticker_data:
                    self.UpdateSecurityPrice(ticker, ticker_data["price"])
        
        # Update current time
        if data and any(data.values()):
            # Use timestamp from first available data point
            first_data = next(iter(data.values()))
            if "timestamp" in first_data:
                try:
                    self.time = datetime.fromisoformat(first_data["timestamp"])
                except:
                    self.time = datetime.now()
        
        # Call the user's OnData method
        self.OnData(data)
    
    def on_order_filled(self, order):
        """Handle order fill events."""
        # Create OrderEvent object
        order_event = OrderEvent(
            order_id=order.id,
            symbol=order.symbol,
            quantity=order.qty,
            side=order.side.value.lower(),
            status=order.status.value.lower(),
            filled_quantity=order.filled_qty,
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else 0.0
        )
        
        # Call user's OnOrderEvent method
        self.OnOrderEvent(order_event)
    
    def on_error(self, error):
        """Handle errors."""
        self.logger.error(f"Algorithm error: {error}")
        super().on_error(error)
    
    # Abstract methods that users must implement
    @abstractmethod
    def Initialize(self):
        """Initialize the algorithm. Override this method in your strategy.
        
        This method is called once at the start of the algorithm.
        Use this method to:
        - Set start/end dates
        - Set starting cash
        - Add securities
        - Set up indicators
        """
        pass
    
    @abstractmethod
    def OnData(self, data: Dict[str, Any]):
        """Handle incoming market data. Override this method in your strategy.
        
        This method is called whenever new market data arrives.
        
        Args:
            data (Dict[str, Any]): Market data dictionary with symbol as key
        """
        pass
    
    def OnOrderEvent(self, order_event: OrderEvent):
        """Handle order events. Override this method in your strategy.
        
        This method is called when an order is filled, cancelled, or updated.
        
        Args:
            order_event (OrderEvent): The order event object
        """
        self.logger.info(f"Order event: {order_event.symbol} {order_event.side} "
                        f"{order_event.filled_quantity} shares at ${order_event.filled_price}")
    
    def OnEndOfDay(self, symbol: Union[str, Symbol]):
        """Handle end of day events. Override this method in your strategy.
        
        This method is called at the end of each trading day for each symbol.
        
        Args:
            symbol (Union[str, Symbol]): The symbol that reached end of day
        """
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        self.logger.info(f"End of day for {ticker}")
    
    # Utility methods
    def Log(self, message: str):
        """Log a message.
        
        Args:
            message (str): Message to log
        """
        self.logger.info(message)
    
    def Debug(self, message: str):
        """Log a debug message.
        
        Args:
            message (str): Debug message to log
        """
        self.logger.debug(message)
    
    def Error(self, message: str):
        """Log an error message.
        
        Args:
            message (str): Error message to log
        """
        self.logger.error(message)
    
    def SetBenchmark(self, symbol: Union[str, Symbol]):
        """Set the benchmark symbol for performance comparison.
        
        Args:
            symbol (Union[str, Symbol]): Symbol to use as benchmark
        """
        ticker = symbol.ticker if isinstance(symbol, Symbol) else symbol
        self.benchmark_symbol = ticker
        self.logger.info(f"Benchmark set to: {ticker}") 