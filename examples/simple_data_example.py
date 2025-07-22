"""
Simple Data Pipeline Example for Lean-Lite.

This example demonstrates the simplified data pipeline system
with basic usage patterns for algorithmic trading.
"""

from datetime import datetime, timedelta

from lean_lite.config import Config
from lean_lite.data.simple_data_engine import SimpleDataEngine, SimpleAlpacaFeed
from lean_lite.algorithm.data_models import Resolution


def main():
    """Simple data pipeline example."""
    print("Starting Simple Data Pipeline Example")
    
    # Initialize configuration
    config = Config()
    
    # Create simple data engine
    engine = SimpleDataEngine(config)
    
    # Add Alpaca feed
    alpaca_feed = SimpleAlpacaFeed("alpaca", config)
    engine.add_feed(alpaca_feed)
    
    try:
        # Start the engine
        print("Starting data engine...")
        engine.start()
        
        # Subscribe to symbols
        symbols = ["AAPL", "MSFT"]
        for symbol in symbols:
            success = engine.subscribe(symbol)
            if success:
                print(f"Subscribed to {symbol}")
            else:
                print(f"Failed to subscribe to {symbol}")
        
        # Get historical data
        print("Getting historical data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        for symbol in symbols:
            bars = engine.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                resolution=Resolution.MINUTE
            )
            print(f"Retrieved {len(bars)} bars for {symbol}")
            
            if bars:
                latest_bar = bars[-1]
                print(f"  Latest price: ${latest_bar.Close:.2f}")
        
        # Get current prices
        print("Getting current prices...")
        for symbol in symbols:
            price = engine.get_current_price(symbol)
            if price:
                print(f"Current price for {symbol}: ${price:.2f}")
            else:
                print(f"No current price for {symbol}")
        
        # Check subscriptions
        subscribed = engine.get_subscribed_symbols()
        print(f"Subscribed symbols: {subscribed}")
        
        # Unsubscribe from one symbol
        engine.unsubscribe("MSFT")
        print("Unsubscribed from MSFT")
        
        subscribed = engine.get_subscribed_symbols()
        print(f"Subscribed symbols after unsubscribe: {subscribed}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop the engine
        print("Stopping data engine...")
        engine.stop()
        print("Example completed")


def demonstrate_cache():
    """Demonstrate simple caching."""
    print("\nDemonstrating Simple Cache")
    
    from lean_lite.data.simple_data_engine import SimpleCache
    
    cache = SimpleCache()
    
    # Store data
    cache.put("test_key", "test_value", ttl=5)  # 5 second expiry
    
    # Retrieve data
    value = cache.get("test_key")
    print(f"Cached value: {value}")
    
    # Try to get non-existent key
    value = cache.get("non_existent")
    print(f"Non-existent key: {value}")
    
    # Clear cache
    cache.clear()
    value = cache.get("test_key")
    print(f"After clear: {value}")


def demonstrate_validation():
    """Demonstrate basic data validation."""
    print("\nDemonstrating Basic Data Validation")
    
    from lean_lite.data.simple_data_engine import validate_basic_data
    from lean_lite.algorithm.data_models import TradeBar, Symbol, SecurityType, Market
    
    # Create valid data
    symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
    valid_bar = TradeBar(
        Symbol=symbol,
        Time=datetime.now(),
        Open=100.0,
        High=101.0,
        Low=99.0,
        Close=100.5,
        Volume=1000,
        Period=Resolution.MINUTE
    )
    
    is_valid = validate_basic_data(valid_bar)
    print(f"Valid bar: {is_valid}")
    
    # Create invalid data (Low > min(Open, Close))
    try:
        invalid_bar = TradeBar(
            Symbol=symbol,
            Time=datetime.now(),
            Open=100.0,
            High=101.0,
            Low=102.0,  # Low > min(Open, Close) = 100.0 (invalid)
            Close=100.5,
            Volume=1000,
            Period=Resolution.MINUTE
        )
        is_valid = validate_basic_data(invalid_bar)
        print(f"Invalid bar (Low > min(Open, Close)): {is_valid}")
    except ValueError as e:
        print(f"Invalid bar (Low > min(Open, Close)): Validation failed - {e}")
    
    # Test None data
    is_valid = validate_basic_data(None)
    print(f"None data: {is_valid}")


if __name__ == "__main__":
    demonstrate_cache()
    demonstrate_validation()
    main() 