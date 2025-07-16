"""
Alpaca Data Feed Implementation for Lean-Lite.

This module provides a concrete implementation of DataFeed that integrates
with the existing AlpacaData functionality for real-time and historical data.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from lean_lite.algorithm.data_models import TradeBar, Resolution, Symbol, SecurityType, Market
from lean_lite.config import Config
from lean_lite.data.alpaca_data import AlpacaData, AlpacaCredentials
from lean_lite.data.data_engine import DataFeed, DataQualityLevel

logger = logging.getLogger(__name__)


class AlpacaDataFeed(DataFeed):
    """Alpaca data feed implementation."""
    
    def __init__(self, name: str, config: Config, credentials: Optional[AlpacaCredentials] = None):
        super().__init__(name, config)
        self.alpaca_data = AlpacaData(config, credentials)
        self._connection_task: Optional[asyncio.Task] = None
        self._streaming_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Connect to Alpaca data source."""
        try:
            # Initialize Alpaca connection
            success = self.alpaca_data.initialize()
            if not success:
                logger.error(f"Failed to initialize Alpaca connection for {self.name}")
                return False
            
            self.is_connected = True
            logger.info(f"Successfully connected to Alpaca for {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Alpaca for {self.name}: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca data source."""
        try:
            # Stop streaming if active
            if self.alpaca_data.is_streaming:
                self.alpaca_data.stop_streaming()
            
            # Disconnect Alpaca
            self.alpaca_data.disconnect()
            self.is_connected = False
            
            # Cancel any running tasks
            if self._connection_task:
                self._connection_task.cancel()
            if self._streaming_task:
                self._streaming_task.cancel()
            
            logger.info(f"Disconnected from Alpaca for {self.name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca for {self.name}: {e}")
    
    async def subscribe(self, symbol: str) -> bool:
        """Subscribe to a symbol on Alpaca."""
        try:
            if not self.is_connected:
                logger.error(f"Cannot subscribe to {symbol}: not connected")
                return False
            
            # Subscribe to symbol in AlpacaData
            self.alpaca_data.subscribe_symbol(symbol)
            self.subscribed_symbols.add(symbol)
            
            # Start streaming if not already started
            if not self.alpaca_data.is_streaming:
                success = self.alpaca_data.start_streaming()
                if not success:
                    logger.error(f"Failed to start streaming for {self.name}")
                    return False
            
            logger.info(f"Successfully subscribed to {symbol} on {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol} on {self.name}: {e}")
            return False
    
    async def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from a symbol on Alpaca."""
        try:
            if symbol not in self.subscribed_symbols:
                logger.warning(f"Symbol {symbol} not subscribed to {self.name}")
                return True
            
            # Unsubscribe from symbol in AlpacaData
            self.alpaca_data.unsubscribe_symbol(symbol)
            self.subscribed_symbols.discard(symbol)
            
            logger.info(f"Successfully unsubscribed from {symbol} on {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol} on {self.name}: {e}")
            return False
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, resolution: Resolution) -> List[TradeBar]:
        """Get historical data from Alpaca."""
        try:
            if not self.is_connected:
                logger.error(f"Cannot get historical data: not connected")
                return []
            
            # Convert Resolution to Alpaca timeframe
            timeframe_map = {
                Resolution.MINUTE: "1Min",
                Resolution.HOUR: "1Hour", 
                Resolution.DAILY: "1Day"
            }
            
            alpaca_timeframe = timeframe_map.get(resolution, "1Min")
            
            # Get historical data from AlpacaData
            bars = self.alpaca_data.get_historical_data(
                symbol=symbol,
                timeframe=alpaca_timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            logger.debug(f"Retrieved {len(bars)} historical bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def _on_bar_update(self, bar: TradeBar):
        """Handle bar updates from Alpaca."""
        try:
            # Process and validate the data
            is_valid, quality, message = self.process_data(bar)
            
            if is_valid:
                logger.debug(f"Processed bar update for {bar.Symbol.Value}: {bar.Close}")
            else:
                logger.warning(f"Invalid bar data for {bar.Symbol.Value}: {message}")
                
        except Exception as e:
            logger.error(f"Error processing bar update for {bar.Symbol.Value}: {e}")
            self.metrics.increment_errors()
    
    def _on_quote_update(self, quote):
        """Handle quote updates from Alpaca."""
        try:
            # Process and validate the data
            is_valid, quality, message = self.process_data(quote)
            
            if is_valid:
                logger.debug(f"Processed quote update for {quote.Symbol.Value}")
            else:
                logger.warning(f"Invalid quote data for {quote.Symbol.Value}: {message}")
                
        except Exception as e:
            logger.error(f"Error processing quote update for {quote.Symbol.Value}: {e}")
            self.metrics.increment_errors()
    
    def _on_trade_update(self, trade):
        """Handle trade updates from Alpaca."""
        try:
            # Process and validate the data
            is_valid, quality, message = self.process_data(trade)
            
            if is_valid:
                logger.debug(f"Processed trade update for {trade['symbol']}")
            else:
                logger.warning(f"Invalid trade data for {trade['symbol']}: {message}")
                
        except Exception as e:
            logger.error(f"Error processing trade update: {e}")
            self.metrics.increment_errors()
    
    def setup_callbacks(self):
        """Setup callbacks for Alpaca data updates."""
        # Set up callbacks in AlpacaData to route data through our processing
        self.alpaca_data.on_bar_update = self._on_bar_update
        self.alpaca_data.on_quote_update = self._on_quote_update
        self.alpaca_data.on_trade_update = self._on_trade_update
    
    async def start_streaming(self) -> bool:
        """Start real-time data streaming."""
        try:
            if not self.is_connected:
                logger.error("Cannot start streaming: not connected")
                return False
            
            # Setup callbacks
            self.setup_callbacks()
            
            # Start streaming
            success = self.alpaca_data.start_streaming()
            if success:
                logger.info(f"Started streaming for {self.name}")
            else:
                logger.error(f"Failed to start streaming for {self.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting streaming for {self.name}: {e}")
            return False
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        try:
            if self.alpaca_data.is_streaming:
                self.alpaca_data.stop_streaming()
                logger.info(f"Stopped streaming for {self.name}")
        except Exception as e:
            logger.error(f"Error stopping streaming for {self.name}: {e}")
    
    def get_latest_bar(self, symbol: str) -> Optional[TradeBar]:
        """Get the latest bar for a symbol."""
        try:
            return self.alpaca_data.latest_bars.get(symbol)
        except Exception as e:
            logger.error(f"Error getting latest bar for {symbol}: {e}")
            return None
    
    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Get the latest quote for a symbol."""
        try:
            return self.alpaca_data.latest_quotes.get(symbol)
        except Exception as e:
            logger.error(f"Error getting latest quote for {symbol}: {e}")
            return None
    
    def get_latest_trade(self, symbol: str) -> Optional[dict]:
        """Get the latest trade for a symbol."""
        try:
            return self.alpaca_data.latest_trades.get(symbol)
        except Exception as e:
            logger.error(f"Error getting latest trade for {symbol}: {e}")
            return None
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self.subscribed_symbols)
    
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if a symbol is currently subscribed."""
        return symbol in self.subscribed_symbols
    
    def get_connection_status(self) -> dict:
        """Get detailed connection status."""
        return {
            'feed_name': self.name,
            'is_connected': self.is_connected,
            'is_streaming': self.alpaca_data.is_streaming,
            'subscribed_symbols': list(self.subscribed_symbols),
            'alpaca_connected': self.alpaca_data.is_connected,
            'reconnect_attempts': self.alpaca_data.reconnect_attempts
        } 