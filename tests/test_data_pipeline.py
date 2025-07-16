"""
Comprehensive tests for the Data Pipeline System.

This module tests the complete data pipeline functionality including:
- DataEngine coordination and management
- DataFeed implementations and integrations
- Data buffering and caching mechanisms
- Data synchronization across timeframes
- Data transformation utilities
- Data quality validation
- Data persistence and metrics
- Error handling and recovery
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from lean_lite.config import Config
from lean_lite.data.data_engine import (
    DataEngine, DataFeed, DataBuffer, DataCache, DataValidator, 
    DataTransformer, DataPersistence, DataMetrics, DataQualityLevel
)
from lean_lite.algorithm.data_models import (
    TradeBar, QuoteBar, Symbol, Resolution, SecurityType, Market
)


class MockDataFeed(DataFeed):
    """Mock data feed for testing."""
    
    def __init__(self, name: str, config: Config):
        super().__init__(name, config)
        self.connect_called = False
        self.disconnect_called = False
        self.subscribe_called = False
        self.unsubscribe_called = False
        self.historical_data_called = False
    
    async def connect(self) -> bool:
        """Mock connect method."""
        self.connect_called = True
        self.is_connected = True
        return True
    
    async def disconnect(self):
        """Mock disconnect method."""
        self.disconnect_called = True
        self.is_connected = False
    
    async def subscribe(self, symbol: str) -> bool:
        """Mock subscribe method."""
        self.subscribe_called = True
        self.subscribed_symbols.add(symbol)
        return True
    
    async def unsubscribe(self, symbol: str) -> bool:
        """Mock unsubscribe method."""
        self.unsubscribe_called = True
        self.subscribed_symbols.discard(symbol)
        return True
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, resolution: Resolution):
        """Mock historical data method."""
        self.historical_data_called = True
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


class TestDataBuffer:
    """Test DataBuffer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.buffer = DataBuffer(max_size=5)
    
    def test_add_and_get_latest(self):
        """Test adding items and getting latest."""
        # Add items
        for i in range(3):
            self.buffer.add(f"item_{i}")
        
        # Get latest
        latest = self.buffer.get_latest(2)
        assert len(latest) == 2
        assert latest == ["item_1", "item_2"]
    
    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        # Add more items than max_size
        for i in range(7):
            self.buffer.add(f"item_{i}")
        
        # Should only keep the last 5 items
        assert self.buffer.size() == 5
        latest = self.buffer.get_latest(5)
        assert latest == ["item_2", "item_3", "item_4", "item_5", "item_6"]
    
    def test_clear_buffer(self):
        """Test clearing buffer."""
        self.buffer.add("item_1")
        self.buffer.add("item_2")
        assert self.buffer.size() == 2
        
        self.buffer.clear()
        assert self.buffer.size() == 0
    
    def test_utilization(self):
        """Test buffer utilization calculation."""
        assert self.buffer.utilization == 0.0
        
        self.buffer.add("item_1")
        assert self.buffer.utilization == 20.0  # 1/5 * 100
        
        self.buffer.add("item_2")
        assert self.buffer.utilization == 40.0  # 2/5 * 100


class TestDataCache:
    """Test DataCache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DataCache(max_size=3, cache_dir=self.temp_dir)
        self.start_date = datetime(2024, 1, 1, 10, 0, 0)
        self.end_date = datetime(2024, 1, 1, 11, 0, 0)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_put_and_get(self):
        """Test putting and getting data from cache."""
        test_data = [{"test": "data"}]
        
        # Put data in cache
        self.cache.put("AAPL", "1Min", self.start_date, self.end_date, test_data)
        
        # Get data from cache
        retrieved_data = self.cache.get("AAPL", "1Min", self.start_date, self.end_date)
        assert retrieved_data == test_data
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(4):
            self.cache.put(f"SYMBOL_{i}", "1Min", self.start_date, self.end_date, [f"data_{i}"])
        
        # First item should be evicted
        assert self.cache.get("SYMBOL_0", "1Min", self.start_date, self.end_date) is None
        assert self.cache.get("SYMBOL_3", "1Min", self.start_date, self.end_date) == ["data_3"]
    
    def test_cache_clear(self):
        """Test clearing cache."""
        self.cache.put("AAPL", "1Min", self.start_date, self.end_date, ["data"])
        assert len(self.cache.cache) == 1
        
        self.cache.clear()
        assert len(self.cache.cache) == 0


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
    
    def test_validate_valid_trade_bar(self):
        """Test validation of valid trade bar."""
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
        
        is_valid, quality, message = DataValidator.validate_trade_bar(bar)
        assert is_valid is True
        assert quality == DataQualityLevel.EXCELLENT
    
    def test_validate_invalid_trade_bar(self):
        """Test validation of invalid trade bar."""
        bar = TradeBar(
            Symbol=self.symbol,
            Time=datetime.now(),
            Open=100.0,
            High=99.0,  # High < Close
            Low=99.0,
            Close=100.5,
            Volume=1000,
            Period=Resolution.MINUTE
        )
        
        is_valid, quality, message = DataValidator.validate_trade_bar(bar)
        assert is_valid is False
        assert quality == DataQualityLevel.INVALID
        assert "inconsistent" in message.lower()
    
    def test_validate_negative_prices(self):
        """Test validation with negative prices."""
        bar = TradeBar(
            Symbol=self.symbol,
            Time=datetime.now(),
            Open=-100.0,  # Negative price
            High=101.0,
            Low=99.0,
            Close=100.5,
            Volume=1000,
            Period=Resolution.MINUTE
        )
        
        is_valid, quality, message = DataValidator.validate_trade_bar(bar)
        assert is_valid is False
        assert "negative" in message.lower()
    
    def test_validate_quote_bar(self):
        """Test validation of quote bar."""
        bid_bar = TradeBar(
            Symbol=self.symbol,
            Time=datetime.now(),
            Open=100.0,
            High=100.1,
            Low=99.9,
            Close=100.0,
            Volume=500,
            Period=Resolution.MINUTE
        )
        
        ask_bar = TradeBar(
            Symbol=self.symbol,
            Time=datetime.now(),
            Open=100.1,
            High=100.2,
            Low=100.0,
            Close=100.1,
            Volume=500,
            Period=Resolution.MINUTE
        )
        
        quote = QuoteBar(
            Symbol=self.symbol,
            Time=datetime.now(),
            Bid=bid_bar,
            Ask=ask_bar,
            Period=Resolution.MINUTE
        )
        
        is_valid, quality, message = DataValidator.validate_quote_bar(quote)
        assert is_valid is True
        assert quality == DataQualityLevel.EXCELLENT


class TestDataTransformer:
    """Test DataTransformer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        self.bars = []
        
        # Create sample bars
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(60):  # 60 minutes
            bar = TradeBar(
                Symbol=self.symbol,
                Time=base_time + timedelta(minutes=i),
                Open=100.0 + i * 0.1,
                High=100.0 + i * 0.1 + 0.5,
                Low=100.0 + i * 0.1 - 0.5,
                Close=100.0 + i * 0.1 + 0.2,
                Volume=1000 + i * 10,
                Period=Resolution.MINUTE
            )
            self.bars.append(bar)
    
    def test_resample_bars_to_hourly(self):
        """Test resampling bars to hourly resolution."""
        hourly_bars = DataTransformer.resample_bars(self.bars, Resolution.HOUR)
        
        # Should have fewer bars (60 minutes -> 1 hour)
        assert len(hourly_bars) == 1
        assert hourly_bars[0].Period == Resolution.HOUR
        
        # Check OHLC values
        assert hourly_bars[0].Open == self.bars[0].Open
        assert hourly_bars[0].High == max(bar.High for bar in self.bars)
        assert hourly_bars[0].Low == min(bar.Low for bar in self.bars)
        assert hourly_bars[0].Close == self.bars[-1].Close
    
    def test_normalize_data_minmax(self):
        """Test data normalization using minmax method."""
        normalized = DataTransformer.normalize_data(self.bars, method="minmax")
        
        assert len(normalized) == len(self.bars)
        assert all(0 <= item['normalized_price'] <= 1 for item in normalized)
    
    def test_normalize_data_zscore(self):
        """Test data normalization using zscore method."""
        normalized = DataTransformer.normalize_data(self.bars, method="zscore")
        
        assert len(normalized) == len(self.bars)
        # Z-scores should be around 0 with some variance
        prices = [item['normalized_price'] for item in normalized]
        assert abs(sum(prices) / len(prices)) < 1.0  # Mean close to 0
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        indicators = DataTransformer.calculate_technical_indicators(self.bars)
        
        # Should have some indicators
        assert len(indicators) > 0
        
        # Check SMA calculation
        if 'sma_20' in indicators:
            sma_values = indicators['sma_20']
            assert len(sma_values) == len(self.bars)
            # SMA should be reasonable
            assert all(95 < sma < 110 for sma in sma_values if sma is not None)


class TestDataPersistence:
    """Test DataPersistence functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.persistence = DataPersistence(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)
    
    def test_save_and_retrieve_market_data(self):
        """Test saving and retrieving market data."""
        symbol = "AAPL"
        timestamp = datetime.now()
        data = {"price": 150.0, "volume": 1000}
        
        # Save data
        self.persistence.save_market_data(symbol, timestamp, "trade", data, 0.95)
        
        # Retrieve data
        recent_data = self.persistence.get_recent_data(symbol, hours=1)
        assert len(recent_data) == 1
        assert recent_data[0]['symbol'] == symbol
        assert recent_data[0]['data'] == data
        assert recent_data[0]['quality_score'] == 0.95
    
    def test_save_metrics(self):
        """Test saving metrics."""
        metrics = DataMetrics()
        metrics.total_messages = 100
        metrics.average_latency_ms = 50.0
        
        self.persistence.save_metrics(metrics)
        
        # Retrieve metrics
        metrics_history = self.persistence.get_metrics_history(hours=1)
        assert len(metrics_history) == 1
        assert metrics_history[0].total_messages == 100
        assert metrics_history[0].average_latency_ms == 50.0
    
    def test_save_error(self):
        """Test saving error information."""
        error_type = "connection_error"
        error_message = "Failed to connect"
        symbol = "AAPL"
        
        self.persistence.save_error(error_type, error_message, symbol)
        
        # Verify error was saved (we can't easily retrieve it without adding a method)
        # This test mainly ensures the method doesn't raise exceptions


class TestDataFeed:
    """Test DataFeed abstract class and concrete implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.feed = MockDataFeed("test_feed", self.config)
    
    def test_feed_initialization(self):
        """Test feed initialization."""
        assert self.feed.name == "test_feed"
        assert not self.feed.is_connected
        assert len(self.feed.subscribed_symbols) == 0
    
    def test_process_data(self):
        """Test data processing."""
        symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        bar = TradeBar(
            Symbol=symbol,
            Time=datetime.now(),
            Open=100.0,
            High=101.0,
            Low=99.0,
            Close=100.5,
            Volume=1000,
            Period=Resolution.MINUTE
        )
        
        is_valid, quality, message = self.feed.process_data(bar)
        assert is_valid is True
        assert quality == DataQualityLevel.EXCELLENT
        
        # Check that data was added to buffer
        latest_data = self.feed.get_latest_data(1)
        assert len(latest_data) == 1
        assert latest_data[0] == bar
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        # Process some data first
        symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
        bar = TradeBar(
            Symbol=symbol,
            Time=datetime.now(),
            Open=100.0,
            High=101.0,
            Low=99.0,
            Close=100.5,
            Volume=1000,
            Period=Resolution.MINUTE
        )
        self.feed.process_data(bar)
        
        metrics = self.feed.get_metrics()
        assert metrics.total_messages == 1
        assert metrics.data_quality_score == 1.0


class TestDataEngine:
    """Test DataEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.engine = DataEngine(self.config)
        self.feed1 = MockDataFeed("feed1", self.config)
        self.feed2 = MockDataFeed("feed2", self.config)
    
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
    
    @pytest.mark.asyncio
    async def test_start_and_stop_engine(self):
        """Test starting and stopping the engine."""
        self.engine.add_feed(self.feed1)
        
        # Start engine
        await self.engine.start()
        assert self.engine.is_running is True
        assert self.feed1.connect_called is True
        
        # Stop engine
        await self.engine.stop()
        assert self.engine.is_running is False
        assert self.feed1.disconnect_called is True
    
    @pytest.mark.asyncio
    async def test_subscribe_and_unsubscribe_symbol(self):
        """Test subscribing and unsubscribing to symbols."""
        self.engine.add_feed(self.feed1)
        await self.engine.start()
        
        # Subscribe to symbol
        success = await self.engine.subscribe_symbol("AAPL", "feed1")
        assert success is True
        assert "AAPL" in self.engine.symbol_feeds
        assert self.engine.symbol_feeds["AAPL"] == "feed1"
        assert self.feed1.subscribe_called is True
        
        # Unsubscribe from symbol
        success = await self.engine.unsubscribe_symbol("AAPL")
        assert success is True
        assert "AAPL" not in self.engine.symbol_feeds
        assert self.feed1.unsubscribe_called is True
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self):
        """Test historical data retrieval."""
        self.engine.add_feed(self.feed1)
        await self.engine.start()
        
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)
        
        bars = await self.engine.get_historical_data(
            "AAPL", start_date, end_date, Resolution.MINUTE, "feed1"
        )
        
        assert len(bars) > 0
        assert self.feed1.historical_data_called is True
        assert all(isinstance(bar, TradeBar) for bar in bars)
    
    def test_add_callbacks(self):
        """Test adding callbacks."""
        def data_callback(symbol, data, quality):
            pass
        
        def error_callback(error_type, error_message, symbol=None):
            pass
        
        self.engine.add_data_callback(data_callback)
        self.engine.add_error_callback(error_callback)
        
        assert len(self.engine.data_callbacks) == 1
        assert len(self.engine.error_callbacks) == 1
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        self.engine.add_feed(self.feed1)
        
        metrics = self.engine.get_metrics()
        assert 'engine' in metrics
        assert 'feeds' in metrics
        assert 'cache' in metrics
        assert 'symbols' in metrics
        assert 'feed1' in metrics['feeds']
    
    def test_data_quality_report(self):
        """Test data quality report generation."""
        # This test would require setting up some data in persistence
        # For now, just test that the method doesn't raise exceptions
        report = self.engine.get_data_quality_report("AAPL", hours=1)
        assert isinstance(report, dict)
        assert 'error' in report  # Should return error when no data exists


class TestDataMetrics:
    """Test DataMetrics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = DataMetrics()
    
    def test_initialization(self):
        """Test metrics initialization."""
        assert self.metrics.total_messages == 0
        assert self.metrics.error_count == 0
        assert self.metrics.data_quality_score == 1.0
    
    def test_increment_messages(self):
        """Test message increment."""
        self.metrics.increment_messages()
        assert self.metrics.total_messages == 1
        
        self.metrics.increment_messages()
        assert self.metrics.total_messages == 2
    
    def test_increment_errors(self):
        """Test error increment."""
        self.metrics.increment_errors()
        assert self.metrics.error_count == 1
        
        self.metrics.increment_errors()
        assert self.metrics.error_count == 2
    
    def test_update_latency(self):
        """Test latency update."""
        self.metrics.update_latency(100.0)
        assert self.metrics.average_latency_ms == 100.0
        
        self.metrics.update_latency(200.0)
        # Should be exponential moving average
        assert 100.0 < self.metrics.average_latency_ms < 200.0
    
    def test_serialization(self):
        """Test metrics serialization."""
        self.metrics.total_messages = 100
        self.metrics.average_latency_ms = 50.0
        
        # Test to_dict
        data = self.metrics.to_dict()
        assert data['total_messages'] == 100
        assert data['average_latency_ms'] == 50.0
        
        # Test from_dict
        new_metrics = DataMetrics.from_dict(data)
        assert new_metrics.total_messages == 100
        assert new_metrics.average_latency_ms == 50.0
        
        # Test to_json
        json_str = self.metrics.to_json()
        assert isinstance(json_str, str)
        assert "total_messages" in json_str


if __name__ == "__main__":
    pytest.main([__file__]) 