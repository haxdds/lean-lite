# Data Pipeline System

The Lean-Lite Data Pipeline System provides a comprehensive solution for managing market data feeds, processing, validation, and analysis. It's designed to be scalable, reliable, and easy to integrate with algorithmic trading strategies.

## Overview

The data pipeline system consists of several key components:

- **DataEngine**: Central coordinator for managing multiple data feeds
- **DataFeed**: Abstract interface for data sources (with AlpacaDataFeed implementation)
- **DataBuffer**: Thread-safe buffering for real-time data
- **DataCache**: LRU caching with disk persistence for historical data
- **DataValidator**: Quality validation and data integrity checks
- **DataTransformer**: Utilities for data resampling, normalization, and technical indicators
- **DataPersistence**: SQLite-based storage for debugging and analysis
- **DataMetrics**: Comprehensive monitoring and health metrics

## Quick Start

```python
import asyncio
from lean_lite.config import Config
from lean_lite.data import DataEngine, AlpacaDataFeed

async def main():
    # Initialize configuration
    config = Config()
    
    # Create data engine
    engine = DataEngine(config)
    
    # Add Alpaca data feed
    alpaca_feed = AlpacaDataFeed("alpaca_main", config)
    engine.add_feed(alpaca_feed)
    
    # Start the engine
    await engine.start()
    
    # Subscribe to symbols
    await engine.subscribe_symbol("AAPL")
    await engine.subscribe_symbol("MSFT")
    
    # Get historical data
    from datetime import datetime, timedelta
    from lean_lite.algorithm.data_models import Resolution
    
    bars = await engine.get_historical_data(
        symbol="AAPL",
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        resolution=Resolution.MINUTE
    )
    
    print(f"Retrieved {len(bars)} historical bars")
    
    # Stop the engine
    await engine.stop()

asyncio.run(main())
```

## Core Components

### DataEngine

The `DataEngine` is the central coordinator that manages multiple data feeds, handles subscriptions, and provides a unified interface for data access.

**Key Features:**
- Multi-feed coordination
- Automatic feed selection based on symbol characteristics
- Data synchronization across feeds
- Callback system for real-time updates
- Comprehensive metrics and monitoring

**Usage:**
```python
from lean_lite.data import DataEngine

engine = DataEngine(config)

# Add data feeds
engine.add_feed(alpaca_feed)
engine.add_feed(other_feed)

# Start the engine
await engine.start()

# Subscribe to symbols
await engine.subscribe_symbol("AAPL")
await engine.subscribe_symbol("EUR/USD", feed_name="forex_feed")

# Get historical data
bars = await engine.get_historical_data("AAPL", start_date, end_date, Resolution.MINUTE)

# Add callbacks
def on_data_update(symbol, data, quality):
    print(f"New data for {symbol}: {data}")

engine.add_data_callback(on_data_update)
```

### DataFeed

`DataFeed` is an abstract base class that defines the interface for data sources. The system includes a concrete implementation for Alpaca.

**Key Features:**
- Abstract interface for data sources
- Built-in data validation and processing
- Metrics collection
- Buffer management

**Creating Custom Feeds:**
```python
from lean_lite.data import DataFeed

class CustomDataFeed(DataFeed):
    async def connect(self) -> bool:
        # Implement connection logic
        return True
    
    async def disconnect(self):
        # Implement disconnection logic
        pass
    
    async def subscribe(self, symbol: str) -> bool:
        # Implement subscription logic
        return True
    
    async def unsubscribe(self, symbol: str) -> bool:
        # Implement unsubscription logic
        return True
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, resolution: Resolution):
        # Implement historical data retrieval
        return []
```

### DataBuffer

`DataBuffer` provides thread-safe buffering for real-time market data with configurable size limits.

**Features:**
- Thread-safe operations
- Configurable maximum size
- LRU eviction policy
- Utilization monitoring

**Usage:**
```python
from lean_lite.data import DataBuffer

buffer = DataBuffer(max_size=10000)

# Add data
buffer.add(trade_bar)

# Get latest data
latest = buffer.get_latest(10)

# Check utilization
utilization = buffer.utilization  # Percentage
```

### DataCache

`DataCache` provides LRU caching with disk persistence for historical data to improve performance and reduce API calls.

**Features:**
- Memory and disk caching
- LRU eviction policy
- Automatic cache key generation
- Thread-safe operations

**Usage:**
```python
from lean_lite.data import DataCache

cache = DataCache(max_size=1000, cache_dir="cache")

# Store data
cache.put("AAPL", "1Min", start_date, end_date, bars)

# Retrieve data
bars = cache.get("AAPL", "1Min", start_date, end_date)

# Clear cache
cache.clear()
```

### DataValidator

`DataValidator` provides comprehensive data quality validation for market data.

**Validation Checks:**
- OHLC consistency (High >= max(Open, Close), Low <= min(Open, Close))
- Negative price detection
- Extreme price movement detection
- Future timestamp detection
- Bid-ask spread validation

**Usage:**
```python
from lean_lite.data import DataValidator

# Validate trade bar
is_valid, quality, message = DataValidator.validate_trade_bar(bar)

# Validate quote bar
is_valid, quality, message = DataValidator.validate_quote_bar(quote)

# Quality levels: EXCELLENT, GOOD, FAIR, POOR, INVALID
```

### DataTransformer

`DataTransformer` provides utilities for data manipulation and analysis.

**Features:**
- Timeframe resampling (minute to hour, hour to day, etc.)
- Data normalization (minmax, zscore, log)
- Technical indicator calculation (SMA, EMA, RSI, Bollinger Bands)

**Usage:**
```python
from lean_lite.data import DataTransformer
from lean_lite.algorithm.data_models import Resolution

# Resample bars
hourly_bars = DataTransformer.resample_bars(minute_bars, Resolution.HOUR)

# Normalize data
normalized = DataTransformer.normalize_data(bars, method="minmax")

# Calculate indicators
indicators = DataTransformer.calculate_technical_indicators(bars)
sma_20 = indicators['sma_20']
rsi = indicators['rsi']
```

### DataPersistence

`DataPersistence` provides SQLite-based storage for market data, metrics, and error tracking.

**Features:**
- Market data storage with quality scores
- Metrics history tracking
- Error logging and analysis
- Configurable retention periods

**Usage:**
```python
from lean_lite.data import DataPersistence

persistence = DataPersistence("data_pipeline.db")

# Save market data
persistence.save_market_data("AAPL", timestamp, "trade", data, quality_score=0.95)

# Save metrics
persistence.save_metrics(metrics)

# Save error
persistence.save_error("connection_error", "Failed to connect", "AAPL")

# Retrieve recent data
recent_data = persistence.get_recent_data("AAPL", hours=24)

# Get metrics history
metrics_history = persistence.get_metrics_history(hours=24)
```

### DataMetrics

`DataMetrics` provides comprehensive monitoring and health metrics for the data pipeline.

**Metrics Tracked:**
- Total messages processed
- Messages per second
- Average latency
- Error count
- Data quality scores
- Cache hit rates
- Buffer utilization

**Usage:**
```python
from lean_lite.data import DataMetrics

metrics = DataMetrics()

# Update metrics
metrics.increment_messages()
metrics.update_latency(50.0)  # milliseconds
metrics.increment_errors()

# Get metrics as dictionary
metrics_dict = metrics.to_dict()

# Serialize to JSON
json_str = metrics.to_json()
```

## Advanced Usage

### Multi-Feed Configuration

```python
# Create multiple feeds for different asset classes
alpaca_stocks = AlpacaDataFeed("alpaca_stocks", config)
alpaca_crypto = AlpacaDataFeed("alpaca_crypto", config)
forex_feed = CustomForexFeed("forex_feed", config)

# Add to engine
engine.add_feed(alpaca_stocks)
engine.add_feed(alpaca_crypto)
engine.add_feed(forex_feed)

# Subscribe to different asset types
await engine.subscribe_symbol("AAPL")  # Auto-selects alpaca_stocks
await engine.subscribe_symbol("BTC/USD")  # Auto-selects alpaca_crypto
await engine.subscribe_symbol("EUR/USD", feed_name="forex_feed")
```

### Data Quality Monitoring

```python
# Set up quality monitoring
def quality_callback(symbol, data, quality):
    if quality == DataQualityLevel.INVALID:
        logger.error(f"Invalid data for {symbol}")
    elif quality == DataQualityLevel.POOR:
        logger.warning(f"Poor quality data for {symbol}")

engine.add_data_callback(quality_callback)

# Generate quality reports
quality_report = engine.get_data_quality_report("AAPL", hours=24)
print(f"Quality score: {quality_report['average_quality_score']}")
```

### Custom Data Processing

```python
# Custom data processing pipeline
def custom_processor(symbol, data, quality):
    # Validate data
    if quality == DataQualityLevel.INVALID:
        return
    
    # Transform data
    if isinstance(data, TradeBar):
        # Calculate custom indicators
        sma = calculate_sma(data)
        rsi = calculate_rsi(data)
        
        # Store processed data
        store_processed_data(symbol, sma, rsi)

engine.add_data_callback(custom_processor)
```

### Performance Optimization

```python
# Configure cache for better performance
cache = DataCache(max_size=5000, cache_dir="cache")

# Configure buffer for real-time data
buffer = DataBuffer(max_size=50000)

# Use async processing for heavy operations
async def process_historical_data(symbol):
    bars = await engine.get_historical_data(symbol, start_date, end_date, Resolution.MINUTE)
    
    # Process in background
    asyncio.create_task(heavy_processing(bars))

async def heavy_processing(bars):
    # CPU-intensive operations
    indicators = DataTransformer.calculate_technical_indicators(bars)
    # Store results
```

## Best Practices

### 1. Error Handling

```python
try:
    await engine.start()
    await engine.subscribe_symbol("AAPL")
except Exception as e:
    logger.error(f"Failed to start data pipeline: {e}")
    # Implement retry logic or fallback
```

### 2. Resource Management

```python
async with asyncio.timeout(30):  # 30 second timeout
    await engine.start()

# Always clean up
try:
    # Your code here
    pass
finally:
    await engine.stop()
```

### 3. Monitoring and Alerting

```python
# Set up periodic metrics collection
async def monitor_pipeline():
    while True:
        metrics = engine.get_metrics()
        
        # Check for issues
        if metrics['engine']['error_count'] > 10:
            send_alert("High error rate detected")
        
        if metrics['engine']['average_latency_ms'] > 1000:
            send_alert("High latency detected")
        
        await asyncio.sleep(60)  # Check every minute

asyncio.create_task(monitor_pipeline())
```

### 4. Data Quality

```python
# Implement quality thresholds
def quality_filter(symbol, data, quality):
    if quality.value < 0.7:  # Poor quality
        logger.warning(f"Low quality data for {symbol}: {quality.value}")
        return False
    return True

engine.add_data_callback(quality_filter)
```

### 5. Caching Strategy

```python
# Use appropriate cache sizes
cache = DataCache(max_size=1000)  # For frequently accessed data

# Clear cache periodically
async def clear_cache_periodically():
    while True:
        await asyncio.sleep(3600)  # Every hour
        cache.clear()

asyncio.create_task(clear_cache_periodically())
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check API credentials
   - Verify network connectivity
   - Check rate limits

2. **High Latency**
   - Monitor buffer utilization
   - Check data feed performance
   - Optimize processing callbacks

3. **Data Quality Issues**
   - Review validation rules
   - Check data source quality
   - Monitor error logs

4. **Memory Usage**
   - Adjust buffer sizes
   - Monitor cache usage
   - Implement data cleanup

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('lean_lite.data').setLevel(logging.DEBUG)

# Get detailed metrics
metrics = engine.get_metrics()
print(json.dumps(metrics, indent=2))

# Check connection status
for feed_name, feed in engine.feeds.items():
    status = feed.get_connection_status()
    print(f"{feed_name}: {status}")
```

## API Reference

For detailed API documentation, see the individual module docstrings:

- `DataEngine`: Main coordination class
- `DataFeed`: Abstract data feed interface
- `AlpacaDataFeed`: Alpaca-specific implementation
- `DataBuffer`: Thread-safe data buffering
- `DataCache`: LRU caching with persistence
- `DataValidator`: Data quality validation
- `DataTransformer`: Data manipulation utilities
- `DataPersistence`: SQLite-based storage
- `DataMetrics`: Monitoring and metrics 