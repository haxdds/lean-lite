"""
Alpaca Data Integration for Lean-Lite.

This module provides comprehensive Alpaca market data integration using alpaca-py SDK including:
- Real-time market data streaming via WebSocket
- Historical data retrieval
- Data normalization to QuantConnect format
- Connection management and reconnection logic
- Error handling for API rate limits
- Async methods for non-blocking data retrieval
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum

import websocket
from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream
import pandas as pd

from ..config import Config
from ..algorithm.data_models import (
    Symbol, TradeBar, QuoteBar, Resolution, SecurityType, Market
)
from .data_manager import DataManager

logger = logging.getLogger(__name__)


class AlpacaTimeframe(Enum):
    """Alpaca timeframe mappings."""
    MINUTE = TimeFrame.Minute
    HOUR = TimeFrame.Hour
    DAILY = TimeFrame.Day


@dataclass
class AlpacaCredentials:
    """Alpaca API credentials."""
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    stream_url: str = "wss://stream.data.alpaca.markets/v2/iex"


class AlpacaData(DataManager):
    """
    Alpaca data integration for Lean-Lite using alpaca-py SDK.
    
    Provides real-time market data streaming and historical data retrieval
    with automatic data normalization to QuantConnect format.
    """
    
    def __init__(self, config: Config, credentials: Optional[AlpacaCredentials] = None):
        """Initialize Alpaca data integration."""
        super().__init__(None, config)
        
        # Initialize credentials
        if credentials:
            self.credentials = credentials
        else:
            self.credentials = AlpacaCredentials(
                api_key=config.alpaca_api_key or "",
                secret_key=config.alpaca_secret_key or "",
                base_url=config.alpaca_base_url
            )
        
        # Initialize API clients
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self.stream_client: Optional[StockDataStream] = None
        
        # Connection state
        self.is_connected = False
        self.is_streaming = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        
        # Data storage
        self.subscribed_symbols: List[str] = []
        self.latest_bars: Dict[str, TradeBar] = {}
        self.latest_quotes: Dict[str, QuoteBar] = {}
        self.latest_trades: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self.on_bar_update: Optional[Callable[[TradeBar], None]] = None
        self.on_quote_update: Optional[Callable[[QuoteBar], None]] = None
        self.on_trade_update: Optional[Callable[[Dict[str, Any]], None]] = None
        
        logger.info("AlpacaData initialized")
    
    def initialize(self):
        """Initialize the Alpaca data connection."""
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.credentials.api_key,
                secret_key=self.credentials.secret_key,
                paper=True if "paper" in self.credentials.base_url else False
            )
            
            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.credentials.api_key,
                secret_key=self.credentials.secret_key
            )
            
            # Test connection
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca: {account.status}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca connection: {e}")
            self.is_connected = False
            return False
    
    async def initialize_async(self):
        """Initialize the Alpaca data connection asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.initialize)
    

    
    def _normalize_symbol(self, symbol: Union[str, Symbol]) -> str:
        """Normalize symbol to Alpaca format."""
        if isinstance(symbol, Symbol):
            return symbol.Value
        return str(symbol).upper()
    
    def _create_symbol(self, symbol_str: str) -> Symbol:
        """Create Symbol object from string."""
        return Symbol(
            Value=symbol_str,
            SecurityType=SecurityType.EQUITY,
            Market=Market.US_EQUITY
        )
    
    def _alpaca_to_trade_bar(self, alpaca_bar, symbol: str) -> TradeBar:
        """Convert Alpaca bar data to TradeBar."""
        symbol_obj = self._create_symbol(symbol)
        
        return TradeBar(
            Symbol=symbol_obj,
            Time=alpaca_bar.timestamp,
            Open=float(alpaca_bar.open),
            High=float(alpaca_bar.high),
            Low=float(alpaca_bar.low),
            Close=float(alpaca_bar.close),
            Volume=int(alpaca_bar.volume),
            Period=Resolution.MINUTE
        )
    
    def _alpaca_to_quote_bar(self, alpaca_quote, symbol: str) -> QuoteBar:
        """Convert Alpaca quote data to QuoteBar."""
        symbol_obj = self._create_symbol(symbol)
        timestamp = alpaca_quote.timestamp
        
        # Create bid TradeBar
        bid_bar = None
        if alpaca_quote.bid_price and alpaca_quote.bid_size:
            bid_bar = TradeBar(
                Symbol=symbol_obj,
                Time=timestamp,
                Open=float(alpaca_quote.bid_price),
                High=float(alpaca_quote.bid_price),
                Low=float(alpaca_quote.bid_price),
                Close=float(alpaca_quote.bid_price),
                Volume=int(alpaca_quote.bid_size),
                Period=Resolution.MINUTE
            )
        
        # Create ask TradeBar
        ask_bar = None
        if alpaca_quote.ask_price and alpaca_quote.ask_size:
            ask_bar = TradeBar(
                Symbol=symbol_obj,
                Time=timestamp,
                Open=float(alpaca_quote.ask_price),
                High=float(alpaca_quote.ask_price),
                Low=float(alpaca_quote.ask_price),
                Close=float(alpaca_quote.ask_price),
                Volume=int(alpaca_quote.ask_size),
                Period=Resolution.MINUTE
            )
        
        return QuoteBar(
            Symbol=symbol_obj,
            Time=timestamp,
            Bid=bid_bar,
            Ask=ask_bar,
            Period=Resolution.MINUTE
        )
    
    def _alpaca_to_trade(self, alpaca_trade, symbol: str) -> Dict[str, Any]:
        """Convert Alpaca trade data to internal format."""
        return {
            'symbol': symbol,
            'price': float(alpaca_trade.price),
            'size': int(alpaca_trade.size),
            'timestamp': alpaca_trade.timestamp,
            'exchange': alpaca_trade.exchange,
            'conditions': alpaca_trade.conditions if hasattr(alpaca_trade, 'conditions') else []
        }
    
    def subscribe_symbol(self, symbol: Union[str, Symbol]):
        """Subscribe to real-time data for a symbol."""
        symbol_str = self._normalize_symbol(symbol)
        
        if symbol_str not in self.subscribed_symbols:
            self.subscribed_symbols.append(symbol_str)
            logger.info(f"Subscribed to symbol: {symbol_str}")
            
            # Initialize data storage
            self.latest_bars[symbol_str] = None
            self.latest_quotes[symbol_str] = None
            self.latest_trades[symbol_str] = None
    
    def unsubscribe_symbol(self, symbol: Union[str, Symbol]):
        """Unsubscribe from real-time data for a symbol."""
        symbol_str = self._normalize_symbol(symbol)
        
        if symbol_str in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol_str)
            
            # Clean up data storage
            self.latest_bars.pop(symbol_str, None)
            self.latest_quotes.pop(symbol_str, None)
            self.latest_trades.pop(symbol_str, None)
            
            logger.info(f"Unsubscribed from symbol: {symbol_str}")
    
    def start_streaming(self):
        """Start real-time data streaming."""
        if not self.is_connected:
            logger.error("Not connected to Alpaca")
            return False
        
        try:
            # Initialize stream client
            self.stream_client = StockDataStream(
                api_key=self.credentials.api_key,
                secret_key=self.credentials.secret_key
            )
            
            # Set up handlers
            self.stream_client.subscribe_bars(self._on_bar_update)
            self.stream_client.subscribe_quotes(self._on_quote_update)
            self.stream_client.subscribe_trades(self._on_trade_update)
            
            # Start streaming
            self.stream_client.run()
            self.is_streaming = True
            
            logger.info("Started Alpaca data streaming")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    async def start_streaming_async(self):
        """Start real-time data streaming asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.start_streaming)
    
    def _on_bar_update(self, bar_data):
        """Handle real-time bar updates."""
        try:
            symbol = bar_data.symbol
            trade_bar = self._alpaca_to_trade_bar(bar_data, symbol)
            
            self.latest_bars[symbol] = trade_bar
            if self.on_bar_update:
                self.on_bar_update(trade_bar)
            
            logger.debug(f"Bar update: {symbol} @ ${trade_bar.Close:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing bar update: {e}")
    
    def _on_quote_update(self, quote_data):
        """Handle real-time quote updates."""
        try:
            symbol = quote_data.symbol
            quote_bar = self._alpaca_to_quote_bar(quote_data, symbol)
            
            self.latest_quotes[symbol] = quote_bar
            if self.on_quote_update:
                self.on_quote_update(quote_bar)
            
            logger.debug(f"Quote update: {symbol} Bid:${quote_bar.Bid.Close:.2f} Ask:${quote_bar.Ask.Close:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing quote update: {e}")
    
    def _on_trade_update(self, trade_data):
        """Handle real-time trade updates."""
        try:
            symbol = trade_data.symbol
            trade = self._alpaca_to_trade(trade_data, symbol)
            
            self.latest_trades[symbol] = trade
            if self.on_trade_update:
                self.on_trade_update(trade)
            
            logger.debug(f"Trade update: {symbol} @ ${trade['price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing trade update: {e}")
    
    def get_historical_data(self, symbol: Union[str, Symbol], 
                          timeframe: str = "1Min", 
                          limit: int = 100,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[TradeBar]:
        """Get historical market data."""
        if not self.is_connected:
            logger.error("Not connected to Alpaca")
            return []
        
        symbol_str = self._normalize_symbol(symbol)
        
        try:
            # Convert timeframe to Alpaca format
            alpaca_timeframe = self._get_alpaca_timeframe(timeframe)
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol_str,
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date
            )
            
            # Get historical bars
            bars = self.data_client.get_stock_bars(request)
            
            # Convert to TradeBar objects
            trade_bars = []
            for bar in bars.data.get(symbol_str, []):
                trade_bar = self._alpaca_to_trade_bar(bar, symbol_str)
                trade_bars.append(trade_bar)
            
            logger.info(f"Retrieved {len(trade_bars)} historical bars for {symbol_str}")
            return trade_bars
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol_str}: {e}")
            return []
    
    async def get_historical_data_async(self, symbol: Union[str, Symbol],
                                      timeframe: str = "1Min",
                                      limit: int = 100,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> List[TradeBar]:
        """Get historical market data asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.get_historical_data, 
            symbol, timeframe, limit, start_date, end_date
        )
    
    def get_historical_quotes(self, symbol: Union[str, Symbol],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: int = 100) -> List[QuoteBar]:
        """Get historical quote data."""
        if not self.is_connected:
            logger.error("Not connected to Alpaca")
            return []
        
        symbol_str = self._normalize_symbol(symbol)
        
        try:
            # Create request
            request = StockQuotesRequest(
                symbol_or_symbols=symbol_str,
                start=start_date,
                end=end_date
            )
            
            # Get historical quotes
            quotes = self.data_client.get_stock_quotes(request)
            
            # Convert to QuoteBar objects
            quote_bars = []
            for quote in quotes.data.get(symbol_str, []):
                quote_bar = self._alpaca_to_quote_bar(quote, symbol_str)
                quote_bars.append(quote_bar)
            
            logger.info(f"Retrieved {len(quote_bars)} historical quotes for {symbol_str}")
            return quote_bars
            
        except Exception as e:
            logger.error(f"Error retrieving historical quotes for {symbol_str}: {e}")
            return []
    
    def get_historical_trades(self, symbol: Union[str, Symbol],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical trade data."""
        if not self.is_connected:
            logger.error("Not connected to Alpaca")
            return []
        
        symbol_str = self._normalize_symbol(symbol)
        
        try:
            # Create request
            request = StockTradesRequest(
                symbol_or_symbols=symbol_str,
                start=start_date,
                end=end_date
            )
            
            # Get historical trades
            trades = self.data_client.get_stock_trades(request)
            
            # Convert to internal format
            trade_data = []
            for trade in trades.data.get(symbol_str, []):
                trade_dict = self._alpaca_to_trade(trade, symbol_str)
                trade_data.append(trade_dict)
            
            logger.info(f"Retrieved {len(trade_data)} historical trades for {symbol_str}")
            return trade_data
            
        except Exception as e:
            logger.error(f"Error retrieving historical trades for {symbol_str}: {e}")
            return []
    
    def _get_alpaca_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert timeframe string to Alpaca format."""
        timeframe_map = {
            "1Min": TimeFrame.Minute, 
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day
        }
        return timeframe_map.get(timeframe, TimeFrame.Minute)
    
    def get_latest_bar(self, symbol: Union[str, Symbol]) -> Optional[TradeBar]:
        """Get the latest bar for a symbol."""
        symbol_str = self._normalize_symbol(symbol)
        return self.latest_bars.get(symbol_str)
    
    def get_latest_quote(self, symbol: Union[str, Symbol]) -> Optional[QuoteBar]:
        """Get the latest quote for a symbol."""
        symbol_str = self._normalize_symbol(symbol)
        return self.latest_quotes.get(symbol_str)
    
    def get_latest_trade(self, symbol: Union[str, Symbol]) -> Optional[Dict[str, Any]]:
        """Get the latest trade for a symbol."""
        symbol_str = self._normalize_symbol(symbol)
        return self.latest_trades.get(symbol_str)
    
    def get_current_price(self, symbol: Union[str, Symbol]) -> Optional[float]:
        """Get current price for a symbol."""
        # Try to get from latest trade first
        latest_trade = self.get_latest_trade(symbol)
        if latest_trade:
            return latest_trade['price']
        
        # Fall back to latest bar
        latest_bar = self.get_latest_bar(symbol)
        if latest_bar:
            return latest_bar.Close
        
        # Fall back to latest quote
        latest_quote = self.get_latest_quote(symbol)
        if latest_quote:
            return latest_quote.Price
        
        return None
    
    def process_data(self) -> Dict[str, Any]:
        """Process incoming market data."""
        processed_data = {}
        
        for symbol in self.subscribed_symbols:
            symbol_data = {
                'symbol': symbol,
                'price': self.get_current_price(symbol),
                'bar': self.latest_bars.get(symbol),
                'quote': self.latest_quotes.get(symbol),
                'trade': self.latest_trades.get(symbol),
                'timestamp': datetime.now()
            }
            processed_data[symbol] = symbol_data
        
        return processed_data
    
    def _reconnect(self):
        """Attempt to reconnect to Alpaca."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
        
        self.reconnect_attempts += 1
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        try:
            # Stop current connections
            self.disconnect()
            
            # Wait before reconnecting
            time.sleep(self.reconnect_delay)
            
            # Reinitialize
            if self.initialize():
                self.reconnect_attempts = 0
                self.reconnect_delay = min(self.reconnect_delay * 2, 30.0)  # Exponential backoff
                logger.info("Successfully reconnected to Alpaca")
                return True
            else:
                self.reconnect_delay = min(self.reconnect_delay * 2, 30.0)
                return False
                
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    async def _reconnect_async(self):
        """Attempt to reconnect to Alpaca asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._reconnect)
    
    def disconnect(self):
        """Disconnect from Alpaca data streams."""
        try:
            if self.stream_client:
                self.stream_client.close()
                self.stream_client = None
            
            self.is_streaming = False
            self.is_connected = False
            
            logger.info("Disconnected from Alpaca")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def disconnect_async(self):
        """Disconnect from Alpaca data streams asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.disconnect)
    
    def set_callbacks(self, 
                     on_bar_update: Optional[Callable[[TradeBar], None]] = None,
                     on_quote_update: Optional[Callable[[QuoteBar], None]] = None,
                     on_trade_update: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Set callback functions for data updates."""
        self.on_bar_update = on_bar_update
        self.on_quote_update = on_quote_update
        self.on_trade_update = on_trade_update
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        if not self.is_connected or not self.trading_client:
            return None
        
        try:
            account = self.trading_client.get_account()
            return {
                'id': account.id,
                'status': account.status,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_asset_info(self, symbol: Union[str, Symbol]) -> Optional[Dict[str, Any]]:
        """Get asset information."""
        if not self.is_connected or not self.trading_client:
            return None
        
        symbol_str = self._normalize_symbol(symbol)
        
        try:
            asset = self.trading_client.get_asset(symbol_str)
            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'class': asset.asset_class,
                'status': asset.status,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable
            }
        except Exception as e:
            logger.error(f"Error getting asset info for {symbol_str}: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_async() 