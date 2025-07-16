# Simple Data Pipeline System

A lean, practical data pipeline system for algorithmic traders who want to get started quickly without enterprise complexity.

## Overview

The Simple Data Pipeline provides just what you need:
- **SimpleDataEngine**: Coordinate multiple data feeds
- **SimpleDataFeed**: Abstract interface for data sources
- **SimpleCache**: Basic in-memory caching with expiry
- **Basic validation**: Simple data quality checks
- **Historical data**: Get past market data
- **Real-time subscriptions**: Subscribe to live updates

## Quick Start

```python
from lean_lite.config import Config
from lean_lite.data import SimpleDataEngine, SimpleAlpacaFeed
from lean_lite.algorithm.data_models import Resolution
from datetime import datetime, timedelta

# Setup
config = Config()
engine = SimpleDataEngine(config)

# Add Alpaca feed
alpaca_feed = SimpleAlpacaFeed("alpaca", config)
engine.add_feed(alpaca_feed)

# Start and subscribe
engine.start()
engine.subscribe("AAPL")
engine.subscribe("MSFT")

# Get historical data
bars = engine.get_historical_data(
    symbol="AAPL",
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    resolution=Resolution.MINUTE
)

print(f"Got {len(bars)} bars")

# Get current price
price = engine.get_current_price("AAPL")
print(f"Current price: ${price}")

# Cleanup
engine.stop()
```

## Core Components

### SimpleDataEngine

The main coordinator for data feeds.

```python
engine = SimpleDataEngine(config)

# Add feeds
engine.add_feed(alpaca_feed)

# Start/stop
engine.start()
engine.stop()

# Subscribe to symbols
engine.subscribe("AAPL")
engine.subscribe("MSFT", feed_name="alpaca")

# Get data
bars = engine.get_historical_data("AAPL", start_date, end_date, Resolution.MINUTE)
price = engine.get_current_price("AAPL")

# Check subscriptions
symbols = engine.get_subscribed_symbols()
is_subscribed = engine.is_symbol_subscribed("AAPL")
```

### SimpleDataFeed

Abstract base class for data sources.

```python
from lean_lite.data import SimpleDataFeed

class MyCustomFeed(SimpleDataFeed):
    def connect(self) -> bool:
        # Connect to your data source
        return True
    
    def disconnect(self):
        # Disconnect from your data source
        pass
    
    def subscribe(self, symbol: str) -> bool:
        # Subscribe to symbol
        return True
    
    def unsubscribe(self, symbol: str) -> bool:
        # Unsubscribe from symbol
        return True
    
    def get_historical_data(self, symbol: str, start_date, end_date, resolution):
        # Return historical data as list of TradeBar objects
        return []
```

### SimpleCache

Basic in-memory caching with expiry.

```python
from lean_lite.data import SimpleCache

cache = SimpleCache()

# Store data
cache.put("key", "value", ttl=3600)  # 1 hour expiry

# Retrieve data
value = cache.get("key")
value = cache.get("key", ttl=1800)  # Override default TTL

# Clear cache
cache.clear()
```

### Basic Validation

Simple data quality checks.

```python
from lean_lite.data import validate_basic_data

# Check if data is valid
is_valid = validate_basic_data(trade_bar)

# Returns False for:
# - None data
# - Negative prices
# - Negative volume
```

## Usage Examples

### Multiple Data Sources

```python
# Create feeds for different asset classes
alpaca_stocks = SimpleAlpacaFeed("alpaca_stocks", config)
alpaca_crypto = SimpleAlpacaFeed("alpaca_crypto", config)

# Add to engine
engine.add_feed(alpaca_stocks)
engine.add_feed(alpaca_crypto)

# Subscribe to different assets
engine.subscribe("AAPL")  # Auto-selects alpaca_stocks
engine.subscribe("BTC/USD")  # Auto-selects alpaca_crypto
```

### Custom Data Processing

```python
# Get historical data
bars = engine.get_historical_data("AAPL", start_date, end_date, Resolution.MINUTE)

# Process the data
for bar in bars:
    if validate_basic_data(bar):
        # Your trading logic here
        if bar.Close > bar.Open:
            print("Price went up")
        else:
            print("Price went down")
```

### Error Handling

```python
try:
    engine.start()
    engine.subscribe("AAPL")
    
    bars = engine.get_historical_data("AAPL", start_date, end_date, Resolution.MINUTE)
    if not bars:
        print("No data received")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    engine.stop()
```

## Best Practices

### 1. Always Clean Up

```python
try:
    engine.start()
    # Your code here
finally:
    engine.stop()
```

### 2. Check Data Quality

```python
bars = engine.get_historical_data("AAPL", start_date, end_date, Resolution.MINUTE)
valid_bars = [bar for bar in bars if validate_basic_data(bar)]
```

### 3. Use Caching Wisely

```python
# Cache frequently accessed data
cache = SimpleCache()
cache.put("frequent_data", data, ttl=300)  # 5 minutes
```

### 4. Handle Missing Data

```python
price = engine.get_current_price("AAPL")
if price is None:
    print("No current price available")
else:
    print(f"Current price: ${price}")
```

## API Reference

### SimpleDataEngine

- `add_feed(feed)`: Add a data feed
- `remove_feed(feed_name)`: Remove a data feed
- `start()`: Start all feeds
- `stop()`: Stop all feeds
- `subscribe(symbol, feed_name=None)`: Subscribe to symbol
- `unsubscribe(symbol)`: Unsubscribe from symbol
- `get_historical_data(symbol, start_date, end_date, resolution)`: Get historical data
- `get_current_price(symbol)`: Get current price
- `get_subscribed_symbols()`: Get list of subscribed symbols
- `is_symbol_subscribed(symbol)`: Check if symbol is subscribed

### SimpleDataFeed

- `connect()`: Connect to data source
- `disconnect()`: Disconnect from data source
- `subscribe(symbol)`: Subscribe to symbol
- `unsubscribe(symbol)`: Unsubscribe from symbol
- `get_historical_data(symbol, start_date, end_date, resolution)`: Get historical data
- `get_current_price(symbol)`: Get current price

### SimpleCache

- `put(key, value, ttl=3600)`: Store value with expiry
- `get(key, ttl=3600)`: Retrieve value if not expired
- `clear()`: Clear all cached data

### validate_basic_data(data)

Returns `True` if data is valid, `False` otherwise.

## What's Not Included

This simplified system intentionally excludes:
- Complex data quality scoring
- SQLite persistence
- Thread-safe buffers
- Technical indicators
- Data transformation utilities
- Comprehensive metrics
- Multiple callback systems
- Complex error logging

For these features, use the full `DataEngine` system instead. 