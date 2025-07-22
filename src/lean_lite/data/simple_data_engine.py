"""
Simplified Data Pipeline System for Lean-Lite.

A lean, practical data pipeline focused on what algorithmic traders actually need:
- Basic data feed coordination
- Simple caching
- Historical data retrieval
- Real-time subscriptions
- Basic validation
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from lean_lite.algorithm.data_models import TradeBar, Resolution, Symbol, SecurityType, Market
from lean_lite.config import Config

# Conditional import for AlpacaData to avoid dependency issues
try:
    from .alpaca_data import AlpacaData
except ImportError:
    AlpacaData = None


class SimpleCache:
    """Simple in-memory cache with expiry."""
    
    def __init__(self):
        self.data = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.data:
            return None
        
        # Check if expired
        if time.time() - self.timestamps[key] > self.data.get(f"{key}_ttl", 3600):
            del self.data[key]
            del self.timestamps[key]
            if f"{key}_ttl" in self.data:
                del self.data[f"{key}_ttl"]
            return None
        
        return self.data[key]
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """Store value in cache with expiry."""
        self.data[key] = value
        self.timestamps[key] = time.time()
        self.data[f"{key}_ttl"] = ttl
    
    def clear(self):
        """Clear all cached data."""
        self.data.clear()
        self.timestamps.clear()


def validate_basic_data(data) -> bool:
    """Basic data validation - just check for obvious issues."""
    if data is None:
        return False
    
    # Check for negative prices
    if hasattr(data, 'Close') and data.Close <= 0:
        return False
    
    if hasattr(data, 'Open') and data.Open <= 0:
        return False
    
    if hasattr(data, 'High') and data.High <= 0:
        return False
    
    if hasattr(data, 'Low') and data.Low <= 0:
        return False
    
    # Check for negative volume
    if hasattr(data, 'Volume') and data.Volume < 0:
        return False
    
    # Check that Low <= min(Open, Close) and High >= max(Open, Close)
    if hasattr(data, 'Open') and hasattr(data, 'Close') and hasattr(data, 'Low') and hasattr(data, 'High'):
        min_price = min(data.Open, data.Close)
        max_price = max(data.Open, data.Close)
        
        if data.Low > min_price:
            raise ValueError("Low must be <= min(Open, Close)")
        
        if data.High < max_price:
            raise ValueError("High must be >= max(Open, Close)")
    
    return True


class SimpleDataFeed(ABC):
    """Simple abstract data feed."""
    
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
        self.is_connected = False
        self.subscribed_symbols = set()
        self.cache = SimpleCache()
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    def subscribe(self, symbol: str) -> bool:
        """Subscribe to symbol."""
        pass
    
    @abstractmethod
    def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from symbol."""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, resolution: Resolution) -> List[TradeBar]:
        """Get historical data."""
        pass
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        # Default implementation - override in subclasses
        return None


class SimpleAlpacaFeed(SimpleDataFeed):
    """Simple Alpaca data feed."""
    
    def __init__(self, name: str, config: Config):
        super().__init__(name, config)
        if AlpacaData is None:
            raise ImportError("AlpacaData is not available. Please install required dependencies.")
        self.alpaca_data = AlpacaData(config)
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            success = self.alpaca_data.initialize()
            self.is_connected = success
            return success
        except Exception:
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Alpaca."""
        try:
            if self.alpaca_data.is_streaming:
                self.alpaca_data.stop_streaming()
            self.alpaca_data.disconnect()
            self.is_connected = False
        except Exception:
            pass
    
    def subscribe(self, symbol: str) -> bool:
        """Subscribe to symbol."""
        try:
            if not self.is_connected:
                return False
            
            self.alpaca_data.subscribe_symbol(symbol)
            self.subscribed_symbols.add(symbol)
            
            if not self.alpaca_data.is_streaming:
                self.alpaca_data.start_streaming()
            
            return True
        except Exception:
            return False
    
    def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from symbol."""
        try:
            self.alpaca_data.unsubscribe_symbol(symbol)
            self.subscribed_symbols.discard(symbol)
            return True
        except Exception:
            return False
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, resolution: Resolution) -> List[TradeBar]:
        """Get historical data from Alpaca."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}_{resolution.value}"
            cached_data = self.cache.get(cache_key)  # Uses default TTL
            if cached_data:
                return cached_data
            
            # Convert resolution to Alpaca timeframe
            timeframe_map = {
                Resolution.MINUTE: "1Min",
                Resolution.HOUR: "1Hour",
                Resolution.DAILY: "1Day"
            }
            alpaca_timeframe = timeframe_map.get(resolution, "1Min")
            
            # Get data from Alpaca
            bars = self.alpaca_data.get_historical_data(
                symbol=symbol,
                timeframe=alpaca_timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            # Cache the result
            if bars:
                self.cache.put(cache_key, bars, ttl=300)
            
            return bars
        except Exception:
            return []
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from latest bar."""
        try:
            latest_bar = self.alpaca_data.latest_bars.get(symbol)
            if latest_bar and validate_basic_data(latest_bar):
                return latest_bar.Close
            return None
        except Exception:
            return None


class SimpleDataEngine:
    """Simple data engine for coordinating feeds."""
    
    def __init__(self, config: Config):
        self.config = config
        self.feeds = {}
        self.symbol_feeds = {}  # symbol -> feed_name
        self.cache = SimpleCache()
    
    def add_feed(self, feed: SimpleDataFeed):
        """Add a data feed."""
        self.feeds[feed.name] = feed
    
    def remove_feed(self, feed_name: str):
        """Remove a data feed."""
        if feed_name in self.feeds:
            del self.feeds[feed_name]
            # Remove associated symbols
            symbols_to_remove = [s for s, f in self.symbol_feeds.items() if f == feed_name]
            for symbol in symbols_to_remove:
                del self.symbol_feeds[symbol]
    
    def start(self):
        """Start all feeds."""
        for feed in self.feeds.values():
            try:
                feed.connect()
            except Exception:
                pass
    
    def stop(self):
        """Stop all feeds."""
        for feed in self.feeds.values():
            try:
                feed.disconnect()
            except Exception:
                pass
    
    def subscribe(self, symbol: str, feed_name: str = None) -> bool:
        """Subscribe to symbol on specified feed or auto-select."""
        if symbol in self.symbol_feeds:
            return True  # Already subscribed
        
        # Auto-select feed if not specified
        if feed_name is None:
            feed_name = self._select_feed_for_symbol(symbol)
        
        if feed_name not in self.feeds:
            return False
        
        feed = self.feeds[feed_name]
        success = feed.subscribe(symbol)
        
        if success:
            self.symbol_feeds[symbol] = feed_name
        
        return success
    
    def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from symbol."""
        if symbol not in self.symbol_feeds:
            return True
        
        feed_name = self.symbol_feeds[symbol]
        feed = self.feeds[feed_name]
        success = feed.unsubscribe(symbol)
        
        if success:
            del self.symbol_feeds[symbol]
        
        return success
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, resolution: Resolution = Resolution.MINUTE,
                          feed_name: str = None) -> List[TradeBar]:
        """Get historical data for symbol."""
        # Check cache first
        cache_key = f"hist_{symbol}_{start_date}_{end_date}_{resolution.value}"
        cached_data = self.cache.get(cache_key)  # Uses default TTL
        if cached_data:
            return cached_data
        
        # Auto-select feed if not specified
        if feed_name is None:
            feed_name = self._select_feed_for_symbol(symbol)
        
        if feed_name not in self.feeds:
            return []
        
        feed = self.feeds[feed_name]
        data = feed.get_historical_data(symbol, start_date, end_date, resolution)
        
        # Cache the result
        if data:
            self.cache.put(cache_key, data, ttl=600)
        
        return data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        if symbol not in self.symbol_feeds:
            return None
        
        feed_name = self.symbol_feeds[symbol]
        feed = self.feeds[feed_name]
        return feed.get_current_price(symbol)
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols."""
        return list(self.symbol_feeds.keys())
    
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if symbol is subscribed."""
        return symbol in self.symbol_feeds
    
    def _select_feed_for_symbol(self, symbol: str) -> str:
        """Select appropriate feed for symbol."""
        symbol_upper = symbol.upper()
        
        # Simple heuristics
        if symbol_upper.endswith('USD') or symbol_upper.endswith('EUR') or '/' in symbol_upper:
            # Forex pairs
            for feed_name in self.feeds:
                if 'forex' in feed_name.lower():
                    return feed_name
        elif symbol_upper in ['BTC', 'ETH', 'ADA', 'DOT'] or symbol_upper.endswith('USDT'):
            # Crypto
            for feed_name in self.feeds:
                if 'crypto' in feed_name.lower():
                    return feed_name
        
        # Default to first available feed
        return next(iter(self.feeds.keys())) if self.feeds else None 