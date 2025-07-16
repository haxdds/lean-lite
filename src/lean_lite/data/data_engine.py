"""
Data Pipeline System for Lean-Lite.

This module provides a comprehensive data pipeline system including:
- DataEngine: Coordinates data flow and manages multiple data sources
- DataFeed: Manages individual data sources and subscriptions
- Data buffering and caching mechanisms
- Data synchronization across different timeframes
- Data transformation utilities (resample, normalize)
- Data quality checks and validation
- Data persistence for debugging and analysis
- Metrics and monitoring for data pipeline health
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from lean_lite.algorithm.data_models import (
    Symbol, TradeBar, QuoteBar, Resolution, SecurityType, Market
)
from lean_lite.config import Config

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality levels for validation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class AssetClass(Enum):
    """Supported asset classes."""
    STOCKS = "stocks"
    FOREX = "forex"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"
    INDICES = "indices"


@dataclass
class DataMetrics:
    """Metrics for monitoring data pipeline health."""
    total_messages: int = 0
    messages_per_second: float = 0.0
    average_latency_ms: float = 0.0
    error_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 1.0
    cache_hit_rate: float = 0.0
    buffer_utilization: float = 0.0
    
    def update_latency(self, latency_ms: float):
        """Update average latency."""
        if self.total_messages == 0:
            self.average_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_latency_ms = (alpha * latency_ms + 
                                     (1 - alpha) * self.average_latency_ms)
    
    def increment_messages(self):
        """Increment message count."""
        self.total_messages += 1
        self.last_update = datetime.now()
    
    def increment_errors(self):
        """Increment error count."""
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_messages': self.total_messages,
            'messages_per_second': self.messages_per_second,
            'average_latency_ms': self.average_latency_ms,
            'error_count': self.error_count,
            'last_update': self.last_update.isoformat(),
            'data_quality_score': self.data_quality_score,
            'cache_hit_rate': self.cache_hit_rate,
            'buffer_utilization': self.buffer_utilization
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataMetrics':
        """Create DataMetrics from dictionary."""
        return cls(
            total_messages=data['total_messages'],
            messages_per_second=data['messages_per_second'],
            average_latency_ms=data['average_latency_ms'],
            error_count=data['error_count'],
            last_update=datetime.fromisoformat(data['last_update']),
            data_quality_score=data['data_quality_score'],
            cache_hit_rate=data['cache_hit_rate'],
            buffer_utilization=data['buffer_utilization']
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class DataBuffer:
    """Thread-safe data buffer for storing market data."""
    max_size: int = 10000
    data: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add(self, item: Any):
        """Add item to buffer."""
        with self.lock:
            self.data.append(item)
            if len(self.data) > self.max_size:
                self.data.popleft()
    
    def get_latest(self, count: int = 1) -> List[Any]:
        """Get latest items from buffer."""
        with self.lock:
            return list(self.data)[-count:]
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.data.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.data)
    
    @property
    def utilization(self) -> float:
        """Get buffer utilization percentage."""
        return (self.size() / self.max_size) * 100


class DataCache:
    """LRU cache for historical data with persistence."""
    
    def __init__(self, max_size: int = 1000, cache_dir: str = "cache"):
        self.max_size = max_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
        
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: datetime, 
                      end_date: datetime) -> str:
        """Generate cache key for data request."""
        return f"{symbol}_{timeframe}_{start_date.isoformat()}_{end_date.isoformat()}"
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, symbol: str, timeframe: str, start_date: datetime, 
            end_date: datetime) -> Optional[List[Any]]:
        """Get data from cache."""
        key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        
        with self.lock:
            # Check memory cache first
            if key in self.cache:
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            
            # Check disk cache
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    # Add to memory cache
                    self._add_to_cache(key, data)
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def put(self, symbol: str, timeframe: str, start_date: datetime, 
            end_date: datetime, data: List[Any]):
        """Store data in cache."""
        key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        
        with self.lock:
            # Store in memory
            self._add_to_cache(key, data)
            
            # Store on disk
            cache_file = self._get_cache_file(key)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def _add_to_cache(self, key: str, data: Any):
        """Add item to memory cache with LRU eviction."""
        if key in self.cache:
            self.access_order.remove(key)
        else:
            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_size:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
        
        self.cache[key] = data
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cache data."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")


class DataValidator:
    """Data quality validation and checks."""
    
    @staticmethod
    def validate_trade_bar(bar: TradeBar) -> Tuple[bool, DataQualityLevel, str]:
        """Validate trade bar data quality."""
        issues = []
        
        # Check for negative prices
        if bar.Open <= 0 or bar.High <= 0 or bar.Low <= 0 or bar.Close <= 0:
            issues.append("Negative or zero prices")
        
        # Check OHLC consistency
        if bar.High < max(bar.Open, bar.Close):
            issues.append("High price inconsistent with OHLC")
        if bar.Low > min(bar.Open, bar.Close):
            issues.append("Low price inconsistent with OHLC")
        
        # Check for extreme price movements (>50% in one bar)
        if bar.Open > 0:
            price_change = abs(bar.Close - bar.Open) / bar.Open
            if price_change > 0.5:
                issues.append("Extreme price movement detected")
        
        # Check volume
        if bar.Volume < 0:
            issues.append("Negative volume")
        
        # Check timestamp
        if bar.Time > datetime.now() + timedelta(minutes=5):
            issues.append("Future timestamp")
        
        if not issues:
            return True, DataQualityLevel.EXCELLENT, "Data is valid"
        elif len(issues) == 1 and "Extreme price movement" in issues[0]:
            return True, DataQualityLevel.GOOD, "; ".join(issues)
        elif len(issues) <= 2:
            return True, DataQualityLevel.FAIR, "; ".join(issues)
        else:
            return False, DataQualityLevel.INVALID, "; ".join(issues)
    
    @staticmethod
    def validate_quote_bar(quote: QuoteBar) -> Tuple[bool, DataQualityLevel, str]:
        """Validate quote bar data quality."""
        issues = []
        
        if quote.Bid and quote.Ask:
            # Check bid-ask spread
            spread = quote.Ask.Close - quote.Bid.Close
            if spread < 0:
                issues.append("Bid price higher than ask price")
            elif spread > quote.Bid.Close * 0.1:  # >10% spread
                issues.append("Unusually wide bid-ask spread")
        
        # Validate individual bid/ask bars
        if quote.Bid:
            valid, _, issue = DataValidator.validate_trade_bar(quote.Bid)
            if not valid:
                issues.append(f"Bid bar: {issue}")
        
        if quote.Ask:
            valid, _, issue = DataValidator.validate_trade_bar(quote.Ask)
            if not valid:
                issues.append(f"Ask bar: {issue}")
        
        if not issues:
            return True, DataQualityLevel.EXCELLENT, "Data is valid"
        elif len(issues) <= 1:
            return True, DataQualityLevel.GOOD, "; ".join(issues)
        else:
            return False, DataQualityLevel.INVALID, "; ".join(issues)


class DataTransformer:
    """Data transformation utilities."""
    
    @staticmethod
    def resample_bars(bars: List[TradeBar], target_resolution: Resolution) -> List[TradeBar]:
        """Resample bars to a different resolution."""
        if not bars:
            return []
        
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame([
            {
                'time': bar.Time,
                'open': bar.Open,
                'high': bar.High,
                'low': bar.Low,
                'close': bar.Close,
                'volume': bar.Volume
            }
            for bar in bars
        ])
        
        df.set_index('time', inplace=True)
        
        # Resample based on target resolution
        if target_resolution == Resolution.MINUTE:
            resampled = df.resample('1T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_resolution == Resolution.HOUR:
            resampled = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_resolution == Resolution.DAILY:
            resampled = df.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            return bars  # No resampling needed
        
        # Convert back to TradeBar objects
        resampled_bars = []
        for time, row in resampled.iterrows():
            if pd.notna(row['open']):
                bar = TradeBar(
                    Symbol=bars[0].Symbol,
                    Time=time.to_pydatetime(),
                    Open=row['open'],
                    High=row['high'],
                    Low=row['low'],
                    Close=row['close'],
                    Volume=int(row['volume']),
                    Period=target_resolution
                )
                resampled_bars.append(bar)
        
        return resampled_bars
    
    @staticmethod
    def normalize_data(data: List[TradeBar], method: str = "minmax") -> List[Dict[str, float]]:
        """Normalize price data using various methods."""
        if not data:
            return []
        
        prices = [bar.Close for bar in data]
        
        if method == "minmax":
            min_price = min(prices)
            max_price = max(prices)
            if max_price == min_price:
                normalized = [0.5] * len(prices)
            else:
                normalized = [(p - min_price) / (max_price - min_price) for p in prices]
        elif method == "zscore":
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            if std_price == 0:
                normalized = [0.0] * len(prices)
            else:
                normalized = [(p - mean_price) / std_price for p in prices]
        elif method == "log":
            normalized = [np.log(p) if p > 0 else 0 for p in prices]
        else:
            normalized = prices
        
        return [
            {
                'time': bar.Time,
                'original_price': bar.Close,
                'normalized_price': norm_price
            }
            for bar, norm_price in zip(data, normalized)
        ]
    
    @staticmethod
    def calculate_technical_indicators(bars: List[TradeBar]) -> Dict[str, List[float]]:
        """Calculate common technical indicators."""
        if len(bars) < 2:
            return {}
        
        closes = [bar.Close for bar in bars]
        highs = [bar.High for bar in bars]
        lows = [bar.Low for bar in bars]
        volumes = [bar.Volume for bar in bars]
        
        indicators = {}
        
        # Simple Moving Averages
        if len(closes) >= 20:
            indicators['sma_20'] = pd.Series(closes).rolling(20).mean().tolist()
        
        if len(closes) >= 50:
            indicators['sma_50'] = pd.Series(closes).rolling(50).mean().tolist()
        
        # Exponential Moving Averages
        if len(closes) >= 12:
            indicators['ema_12'] = pd.Series(closes).ewm(span=12).mean().tolist()
        
        if len(closes) >= 26:
            indicators['ema_26'] = pd.Series(closes).ewm(span=26).mean().tolist()
        
        # RSI
        if len(closes) >= 14:
            delta = pd.Series(closes).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.tolist()
        
        # Bollinger Bands
        if len(closes) >= 20:
            sma_20 = pd.Series(closes).rolling(20).mean()
            std_20 = pd.Series(closes).rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).tolist()
            indicators['bb_middle'] = sma_20.tolist()
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).tolist()
        
        return indicators


class DataPersistence:
    """Data persistence for debugging and analysis."""
    
    def __init__(self, db_path: str = "data_pipeline.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    data_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    quality_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    symbol TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_market_data(self, symbol: str, timestamp: datetime, data_type: str, 
                        data: Any, quality_score: float = 1.0):
        """Save market data to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO market_data (symbol, timestamp, data_type, data_json, quality_score)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, timestamp.isoformat(), data_type, json.dumps(data), quality_score))
            conn.commit()
    
    def save_metrics(self, metrics: DataMetrics):
        """Save data metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_metrics (timestamp, metrics_json)
                VALUES (?, ?)
            """, (metrics.last_update.isoformat(), metrics.to_json()))
            conn.commit()
    
    def save_error(self, error_type: str, error_message: str, symbol: str = None):
        """Save error information to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_errors (timestamp, error_type, error_message, symbol)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), error_type, error_message, symbol))
            conn.commit()
    
    def get_recent_data(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent market data for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM market_data 
                WHERE symbol = ? AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours), (symbol,))
            
            return [
                {
                    'symbol': row[1],
                    'timestamp': row[2],
                    'data_type': row[3],
                    'data': json.loads(row[4]),
                    'quality_score': row[5]
                }
                for row in cursor.fetchall()
            ]
    
    def get_metrics_history(self, hours: int = 24) -> List[DataMetrics]:
        """Get metrics history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metrics_json FROM data_metrics 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours))
            
            return [DataMetrics.from_dict(json.loads(row[0])) for row in cursor.fetchall()]


class DataFeed(ABC):
    """Abstract base class for data feeds."""
    
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
        self.is_connected = False
        self.subscribed_symbols: Set[str] = set()
        self.data_buffer = DataBuffer()
        self.metrics = DataMetrics()
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbol: str) -> bool:
        """Subscribe to a symbol."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from a symbol."""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, resolution: Resolution) -> List[TradeBar]:
        """Get historical data."""
        pass
    
    def process_data(self, data: Any) -> Tuple[bool, DataQualityLevel, str]:
        """Process and validate incoming data."""
        start_time = time.time()
        
        try:
            # Validate data quality
            if isinstance(data, TradeBar):
                is_valid, quality, message = self.validator.validate_trade_bar(data)
            elif isinstance(data, QuoteBar):
                is_valid, quality, message = self.validator.validate_quote_bar(data)
            else:
                is_valid, quality, message = True, DataQualityLevel.GOOD, "Unknown data type"
            
            if is_valid:
                # Add to buffer
                self.data_buffer.add(data)
                
                # Update metrics
                self.metrics.increment_messages()
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update_latency(latency_ms)
                self.metrics.data_quality_score = quality.value
            
            return is_valid, quality, message
            
        except Exception as e:
            self.metrics.increment_errors()
            logger.error(f"Error processing data in {self.name}: {e}")
            return False, DataQualityLevel.INVALID, str(e)
    
    def get_latest_data(self, count: int = 1) -> List[Any]:
        """Get latest data from buffer."""
        return self.data_buffer.get_latest(count)
    
    def get_metrics(self) -> DataMetrics:
        """Get current metrics."""
        self.metrics.buffer_utilization = self.data_buffer.utilization
        return self.metrics


class DataEngine:
    """Main data engine that coordinates multiple data feeds."""
    
    def __init__(self, config: Config):
        self.config = config
        self.feeds: Dict[str, DataFeed] = {}
        self.symbol_feeds: Dict[str, str] = {}  # symbol -> feed_name
        self.data_cache = DataCache()
        self.persistence = DataPersistence()
        self.metrics = DataMetrics()
        self.is_running = False
        self.lock = threading.Lock()
        
        # Data synchronization
        self.sync_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.sync_workers: Dict[str, asyncio.Task] = {}
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
    def add_feed(self, feed: DataFeed):
        """Add a data feed to the engine."""
        with self.lock:
            self.feeds[feed.name] = feed
            logger.info(f"Added data feed: {feed.name}")
    
    def remove_feed(self, feed_name: str):
        """Remove a data feed from the engine."""
        with self.lock:
            if feed_name in self.feeds:
                del self.feeds[feed_name]
                # Remove symbols associated with this feed
                symbols_to_remove = [
                    symbol for symbol, feed in self.symbol_feeds.items() 
                    if feed == feed_name
                ]
                for symbol in symbols_to_remove:
                    del self.symbol_feeds[symbol]
                logger.info(f"Removed data feed: {feed_name}")
    
    async def start(self):
        """Start the data engine."""
        if self.is_running:
            logger.warning("Data engine is already running")
            return
        
        self.is_running = True
        logger.info("Starting data engine...")
        
        # Connect all feeds
        for feed in self.feeds.values():
            try:
                success = await feed.connect()
                if not success:
                    logger.error(f"Failed to connect feed: {feed.name}")
            except Exception as e:
                logger.error(f"Error connecting feed {feed.name}: {e}")
        
        # Start synchronization workers
        await self._start_sync_workers()
        
        logger.info("Data engine started successfully")
    
    async def stop(self):
        """Stop the data engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping data engine...")
        
        # Stop synchronization workers
        await self._stop_sync_workers()
        
        # Disconnect all feeds
        for feed in self.feeds.values():
            try:
                await feed.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting feed {feed.name}: {e}")
        
        logger.info("Data engine stopped")
    
    async def subscribe_symbol(self, symbol: str, feed_name: str = None) -> bool:
        """Subscribe to a symbol on a specific feed or auto-select."""
        if not self.is_running:
            logger.error("Data engine is not running")
            return False
        
        with self.lock:
            if symbol in self.symbol_feeds:
                logger.warning(f"Symbol {symbol} is already subscribed")
                return True
            
            # Auto-select feed if not specified
            if feed_name is None:
                feed_name = self._select_feed_for_symbol(symbol)
            
            if feed_name not in self.feeds:
                logger.error(f"Feed {feed_name} not found")
                return False
            
            feed = self.feeds[feed_name]
            success = await feed.subscribe(symbol)
            
            if success:
                self.symbol_feeds[symbol] = feed_name
                logger.info(f"Subscribed to {symbol} on {feed_name}")
                return True
            else:
                logger.error(f"Failed to subscribe to {symbol} on {feed_name}")
                return False
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from a symbol."""
        with self.lock:
            if symbol not in self.symbol_feeds:
                logger.warning(f"Symbol {symbol} is not subscribed")
                return True
            
            feed_name = self.symbol_feeds[symbol]
            feed = self.feeds[feed_name]
            success = await feed.unsubscribe(symbol)
            
            if success:
                del self.symbol_feeds[symbol]
                logger.info(f"Unsubscribed from {symbol}")
                return True
            else:
                logger.error(f"Failed to unsubscribe from {symbol}")
                return False
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, resolution: Resolution,
                                feed_name: str = None) -> List[TradeBar]:
        """Get historical data for a symbol."""
        # Check cache first
        cached_data = self.data_cache.get(symbol, resolution.value, start_date, end_date)
        if cached_data:
            logger.debug(f"Retrieved {len(cached_data)} bars from cache for {symbol}")
            return cached_data
        
        # Auto-select feed if not specified
        if feed_name is None:
            feed_name = self._select_feed_for_symbol(symbol)
        
        if feed_name not in self.feeds:
            logger.error(f"Feed {feed_name} not found")
            return []
        
        feed = self.feeds[feed_name]
        data = await feed.get_historical_data(symbol, start_date, end_date, resolution)
        
        # Cache the data
        if data:
            self.data_cache.put(symbol, resolution.value, start_date, end_date, data)
        
        return data
    
    def add_data_callback(self, callback: Callable):
        """Add a callback for data updates."""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add a callback for error handling."""
        self.error_callbacks.append(callback)
    
    def _select_feed_for_symbol(self, symbol: str) -> str:
        """Select the appropriate feed for a symbol based on asset class."""
        # Simple heuristic based on symbol characteristics
        symbol_upper = symbol.upper()
        
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
        else:
            # Default to first available feed
            return next(iter(self.feeds.keys()))
        
        # Fallback to first feed
        return next(iter(self.feeds.keys()))
    
    async def _start_sync_workers(self):
        """Start data synchronization workers."""
        for symbol in self.symbol_feeds:
            await self._start_sync_worker(symbol)
    
    async def _stop_sync_workers(self):
        """Stop data synchronization workers."""
        for task in self.sync_workers.values():
            task.cancel()
        await asyncio.gather(*self.sync_workers.values(), return_exceptions=True)
        self.sync_workers.clear()
    
    async def _start_sync_worker(self, symbol: str):
        """Start synchronization worker for a symbol."""
        if symbol in self.sync_workers:
            return
        
        async def sync_worker():
            queue = self.sync_queues[symbol]
            while self.is_running:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Process data through all feeds
                    for feed in self.feeds.values():
                        if symbol in feed.subscribed_symbols:
                            is_valid, quality, message = feed.process_data(data)
                            
                            if not is_valid:
                                self.persistence.save_error(
                                    "data_validation", 
                                    f"Invalid data for {symbol}: {message}",
                                    symbol
                                )
                            
                            # Call data callbacks
                            for callback in self.data_callbacks:
                                try:
                                    callback(symbol, data, quality)
                                except Exception as e:
                                    logger.error(f"Error in data callback: {e}")
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in sync worker for {symbol}: {e}")
                    self.persistence.save_error("sync_worker", str(e), symbol)
        
        self.sync_workers[symbol] = asyncio.create_task(sync_worker())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all feeds."""
        metrics = {
            'engine': self.metrics.to_dict(),
            'feeds': {},
            'cache': {
                'hit_rate': 0.0,  # TODO: Implement cache hit rate calculation
                'size': len(self.data_cache.cache)
            },
            'symbols': len(self.symbol_feeds)
        }
        
        for feed_name, feed in self.feeds.items():
            metrics['feeds'][feed_name] = feed.get_metrics().to_dict()
        
        return metrics
    
    def save_metrics(self):
        """Save current metrics to persistence."""
        self.persistence.save_metrics(self.metrics)
        for feed in self.feeds.values():
            self.persistence.save_metrics(feed.get_metrics())
    
    def get_data_quality_report(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get data quality report for a symbol."""
        recent_data = self.persistence.get_recent_data(symbol, hours)
        
        if not recent_data:
            return {"error": "No data found"}
        
        quality_scores = [item['quality_score'] for item in recent_data]
        
        return {
            'symbol': symbol,
            'period_hours': hours,
            'total_records': len(recent_data),
            'average_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'quality_distribution': {
                'excellent': len([s for s in quality_scores if s >= 0.9]),
                'good': len([s for s in quality_scores if 0.7 <= s < 0.9]),
                'fair': len([s for s in quality_scores if 0.5 <= s < 0.7]),
                'poor': len([s for s in quality_scores if s < 0.5])
            }
        } 