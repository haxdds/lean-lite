"""
Simple tests for the Simplified Data Pipeline System.

Basic tests covering core functionality without enterprise complexity.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from lean_lite.config import Config
from lean_lite.data.simple_data_engine import (
    SimpleDataEngine, SimpleDataFeed, SimpleAlpacaFeed, SimpleCache, validate_basic_data
)
from lean_lite.algorithm.data_models import (
    TradeBar, Symbol, Resolution, SecurityType, Market
)


class MockSimpleFeed(SimpleDataFeed):
    """Mock simple data feed for testing."""
    
    def connect(self) -> bool:
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def subscribe(self, symbol: str) -> bool:
        self.subscribed_symbols.add(symbol)
        return True
    
    def unsubscribe(self, symbol: str) -> bool:
        self.subscribed_symbols.discard(symbol)
        return True
    
    def get_historical_data(self, symbol: str, start_date, end_date, resolution):
        # Return mock data
        symbol_obj = Symbol(symbol, SecurityType.EQUITY, Market.US_EQUITY)
        bars = []
        current_time = start_date
        while current_time <= end_date:
            bar = TradeBar(
                Symbol=symbol_obj,
                Time=current_time,
                Open=100.0,
                High=101.0,
                Low=99.0,
                Close=100.5,
                Volume=1000,
                Period=resolution
            )
            bars.append(bar)
            current_time += timedelta(minutes=1)
        return bars


class TestSimpleCache:
    """Test SimpleCache functionality."""
    
    def setup_method(self):
        self.cache = SimpleCache()
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        self.cache.put("test_key", "test_value")
        value = self.cache.get("test_key")
        assert value == "test_value"
    
    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        value = self.cache.get("nonexistent")
        assert value is None
    
    def test_cache_expiry(self):
        """Test cache expiry."""
        self.cache.put("test_key", "test_value", ttl=1)  # 1 second expiry
        
        # Should be available immediately
        value = self.cache.get("test_key")
        assert value == "test_value"
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired
        value = self.cache.get("test_key")
        assert value is None
    
    def test_clear(self):
        """Test clearing cache."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("key2") == "value2"
        
        self.cache.clear()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None


class TestValidateBasicData:
    """Test basic data validation."""
    
    def setup_method(self):
        self.symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
    
    def test_valid_data(self):
        """Test validation of valid data."""
        bar = TradeBar(
            Symbol=self.symbol,
            Time=datetime.now(),
            Open=100.0,
            High=101.0,
            Low=99.0,
            Close=100.5,
            Volume=1000,
            Period=Resolution.MINUTE
        )
        
        assert validate_basic_data(bar) is True
    
    def test_none_data(self):
        """Test validation of None data."""
        assert validate_basic_data(None) is False
    
    def test_negative_prices(self):
        """Test validation with negative prices."""
        # Create a mock object that simulates invalid data
        class MockInvalidBar:
            def __init__(self):
                self.Open = -100.0  # Negative price
                self.High = 101.0
                self.Low = 99.0
                self.Close = 100.5
                self.Volume = 1000
        
        bar = MockInvalidBar()
        assert validate_basic_data(bar) is False
    
    def test_negative_volume(self):
        """Test validation with negative volume."""
        # Create a mock object that simulates invalid data
        class MockInvalidBar:
            def __init__(self):
                self.Open = 100.0
                self.High = 101.0
                self.Low = 99.0
                self.Close = 100.5
                self.Volume = -1000  # Negative volume
        
        bar = MockInvalidBar()
        assert validate_basic_data(bar) is False


class TestSimpleDataFeed:
    """Test SimpleDataFeed abstract class."""
    
    def setup_method(self):
        self.config = Config()
        self.feed = MockSimpleFeed("test_feed", self.config)
    
    def test_initialization(self):
        """Test feed initialization."""
        assert self.feed.name == "test_feed"
        assert not self.feed.is_connected
        assert len(self.feed.subscribed_symbols) == 0
    
    def test_connect_and_disconnect(self):
        """Test connect and disconnect."""
        assert self.feed.connect() is True
        assert self.feed.is_connected is True
        
        self.feed.disconnect()
        assert self.feed.is_connected is False
    
    def test_subscribe_and_unsubscribe(self):
        """Test subscribe and unsubscribe."""
        self.feed.connect()
        
        # Subscribe
        assert self.feed.subscribe("AAPL") is True
        assert "AAPL" in self.feed.subscribed_symbols
        
        # Unsubscribe
        assert self.feed.unsubscribe("AAPL") is True
        assert "AAPL" not in self.feed.subscribed_symbols
    
    def test_get_historical_data(self):
        """Test historical data retrieval."""
        self.feed.connect()
        
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)
        
        bars = self.feed.get_historical_data("AAPL", start_date, end_date, Resolution.MINUTE)
        
        assert len(bars) > 0
        assert all(isinstance(bar, TradeBar) for bar in bars)
        assert all(bar.Symbol.Value == "AAPL" for bar in bars)


class TestSimpleDataEngine:
    """Test SimpleDataEngine functionality."""
    
    def setup_method(self):
        self.config = Config()
        self.engine = SimpleDataEngine(self.config)
        self.feed1 = MockSimpleFeed("feed1", self.config)
        self.feed2 = MockSimpleFeed("feed2", self.config)
    
    def test_add_and_remove_feed(self):
        """Test adding and removing feeds."""
        # Add feeds
        self.engine.add_feed(self.feed1)
        self.engine.add_feed(self.feed2)
        
        assert len(self.engine.feeds) == 2
        assert "feed1" in self.engine.feeds
        assert "feed2" in self.engine.feeds
        
        # Remove feed
        self.engine.remove_feed("feed1")
        assert len(self.engine.feeds) == 1
        assert "feed1" not in self.engine.feeds
        assert "feed2" in self.engine.feeds
    
    def test_start_and_stop(self):
        """Test starting and stopping engine."""
        self.engine.add_feed(self.feed1)
        
        # Start engine
        self.engine.start()
        assert self.feed1.is_connected is True
        
        # Stop engine
        self.engine.stop()
        assert self.feed1.is_connected is False
    
    def test_subscribe_and_unsubscribe(self):
        """Test subscribing and unsubscribing to symbols."""
        self.engine.add_feed(self.feed1)
        self.engine.start()
        
        # Subscribe
        success = self.engine.subscribe("AAPL", "feed1")
        assert success is True
        assert "AAPL" in self.engine.symbol_feeds
        assert self.engine.symbol_feeds["AAPL"] == "feed1"
        assert "AAPL" in self.feed1.subscribed_symbols
        
        # Unsubscribe
        success = self.engine.unsubscribe("AAPL")
        assert success is True
        assert "AAPL" not in self.engine.symbol_feeds
        assert "AAPL" not in self.feed1.subscribed_symbols
    
    def test_get_historical_data(self):
        """Test historical data retrieval."""
        self.engine.add_feed(self.feed1)
        self.engine.start()
        
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)
        
        bars = self.engine.get_historical_data(
            "AAPL", start_date, end_date, Resolution.MINUTE, "feed1"
        )
        
        assert len(bars) > 0
        assert all(isinstance(bar, TradeBar) for bar in bars)
    
    def test_get_historical_data_with_cache(self):
        """Test historical data retrieval with caching."""
        self.engine.add_feed(self.feed1)
        self.engine.start()
        
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)
        
        # First call - should hit the feed
        bars1 = self.engine.get_historical_data(
            "AAPL", start_date, end_date, Resolution.MINUTE, "feed1"
        )
        
        # Second call - should hit the cache
        bars2 = self.engine.get_historical_data(
            "AAPL", start_date, end_date, Resolution.MINUTE, "feed1"
        )
        
        assert len(bars1) == len(bars2)
        assert bars1 == bars2
    
    def test_get_subscribed_symbols(self):
        """Test getting subscribed symbols."""
        self.engine.add_feed(self.feed1)
        self.engine.start()
        
        self.engine.subscribe("AAPL", "feed1")
        self.engine.subscribe("MSFT", "feed1")
        
        symbols = self.engine.get_subscribed_symbols()
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert len(symbols) == 2
    
    def test_is_symbol_subscribed(self):
        """Test checking if symbol is subscribed."""
        self.engine.add_feed(self.feed1)
        self.engine.start()
        
        assert self.engine.is_symbol_subscribed("AAPL") is False
        
        self.engine.subscribe("AAPL", "feed1")
        assert self.engine.is_symbol_subscribed("AAPL") is True
    
    def test_feed_selection(self):
        """Test automatic feed selection."""
        self.engine.add_feed(self.feed1)
        self.engine.add_feed(self.feed2)
        
        # Should select first available feed
        selected_feed = self.engine._select_feed_for_symbol("AAPL")
        assert selected_feed in ["feed1", "feed2"]


class TestSimpleAlpacaFeed:
    """Test SimpleAlpacaFeed with mocked AlpacaData."""
    
    def setup_method(self):
        self.config = Config()
    
    @patch('lean_lite.data.simple_data_engine.AlpacaData')
    def test_connect_success(self, mock_alpaca_data):
        """Test successful connection."""
        mock_alpaca = Mock()
        mock_alpaca.initialize.return_value = True
        mock_alpaca_data.return_value = mock_alpaca
        
        feed = SimpleAlpacaFeed("alpaca", self.config)
        success = feed.connect()
        
        assert success is True
        assert feed.is_connected is True
    
    @patch('lean_lite.data.simple_data_engine.AlpacaData')
    def test_connect_failure(self, mock_alpaca_data):
        """Test connection failure."""
        mock_alpaca = Mock()
        mock_alpaca.initialize.return_value = False
        mock_alpaca_data.return_value = mock_alpaca
        
        feed = SimpleAlpacaFeed("alpaca", self.config)
        success = feed.connect()
        
        assert success is False
        assert feed.is_connected is False
    
    @patch('lean_lite.data.simple_data_engine.AlpacaData')
    def test_subscribe(self, mock_alpaca_data):
        """Test subscription."""
        mock_alpaca = Mock()
        mock_alpaca.initialize.return_value = True
        mock_alpaca.is_streaming = False
        mock_alpaca_data.return_value = mock_alpaca
        
        feed = SimpleAlpacaFeed("alpaca", self.config)
        feed.connect()
        
        success = feed.subscribe("AAPL")
        
        assert success is True
        assert "AAPL" in feed.subscribed_symbols
        mock_alpaca.subscribe_symbol.assert_called_with("AAPL")
        mock_alpaca.start_streaming.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__]) 