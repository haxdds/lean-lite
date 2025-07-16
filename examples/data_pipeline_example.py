"""
Data Pipeline Example for Lean-Lite.

This example demonstrates how to use the comprehensive data pipeline system
including DataEngine, AlpacaDataFeed, data validation, transformation, and monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

from lean_lite.config import Config
from lean_lite.data.data_engine import DataEngine, DataQualityLevel
from lean_lite.data.alpaca_feed import AlpacaDataFeed
from lean_lite.algorithm.data_models import Resolution, Symbol, SecurityType, Market

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def data_callback(symbol: str, data, quality: DataQualityLevel):
    """Callback function for data updates."""
    logger.info(f"Data update for {symbol}: Quality={quality.value}")
    if hasattr(data, 'Close'):
        logger.info(f"  Price: ${data.Close:.2f}")
    elif hasattr(data, 'Price'):
        logger.info(f"  Price: ${data.Price:.2f}")


def error_callback(error_type: str, error_message: str, symbol: str = None):
    """Callback function for error handling."""
    logger.error(f"Error [{error_type}]: {error_message}" + (f" (Symbol: {symbol})" if symbol else ""))


async def main():
    """Main example function."""
    logger.info("Starting Data Pipeline Example")
    
    # Initialize configuration
    config = Config()
    
    # Create data engine
    data_engine = DataEngine(config)
    
    # Add data callbacks
    data_engine.add_data_callback(data_callback)
    data_engine.add_error_callback(error_callback)
    
    # Create Alpaca data feed
    alpaca_feed = AlpacaDataFeed("alpaca_main", config)
    data_engine.add_feed(alpaca_feed)
    
    try:
        # Start the data engine
        logger.info("Starting data engine...")
        await data_engine.start()
        
        # Subscribe to symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            success = await data_engine.subscribe_symbol(symbol)
            if success:
                logger.info(f"Successfully subscribed to {symbol}")
            else:
                logger.error(f"Failed to subscribe to {symbol}")
        
        # Wait for some data to arrive
        logger.info("Waiting for data updates...")
        await asyncio.sleep(10)
        
        # Get historical data
        logger.info("Retrieving historical data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        for symbol in symbols:
            bars = await data_engine.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                resolution=Resolution.MINUTE
            )
            logger.info(f"Retrieved {len(bars)} historical bars for {symbol}")
            
            if bars:
                # Demonstrate data transformation
                from lean_lite.data.data_engine import DataTransformer
                
                # Resample to hourly data
                hourly_bars = DataTransformer.resample_bars(bars, Resolution.HOUR)
                logger.info(f"Resampled to {len(hourly_bars)} hourly bars for {symbol}")
                
                # Calculate technical indicators
                indicators = DataTransformer.calculate_technical_indicators(bars)
                if 'sma_20' in indicators:
                    latest_sma = indicators['sma_20'][-1]
                    logger.info(f"Latest 20-period SMA for {symbol}: {latest_sma:.2f}")
        
        # Get metrics
        logger.info("Retrieving pipeline metrics...")
        metrics = data_engine.get_metrics()
        logger.info(f"Engine metrics: {metrics['engine']}")
        
        for feed_name, feed_metrics in metrics['feeds'].items():
            logger.info(f"Feed {feed_name} metrics: {feed_metrics}")
        
        # Get data quality report
        logger.info("Generating data quality report...")
        for symbol in symbols:
            quality_report = data_engine.get_data_quality_report(symbol, hours=1)
            logger.info(f"Quality report for {symbol}: {quality_report}")
        
        # Demonstrate data validation
        logger.info("Demonstrating data validation...")
        from lean_lite.data.data_engine import DataValidator
        
        # Get latest data from feed
        for symbol in symbols:
            latest_bar = alpaca_feed.get_latest_bar(symbol)
            if latest_bar:
                is_valid, quality, message = DataValidator.validate_trade_bar(latest_bar)
                logger.info(f"Validation for {symbol}: Valid={is_valid}, Quality={quality.value}, Message={message}")
        
        # Wait for more real-time data
        logger.info("Waiting for more real-time data...")
        await asyncio.sleep(5)
        
        # Save metrics to persistence
        logger.info("Saving metrics to persistence...")
        data_engine.save_metrics()
        
        # Get connection status
        logger.info("Checking connection status...")
        status = alpaca_feed.get_connection_status()
        logger.info(f"Connection status: {status}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        await data_engine.stop()
        logger.info("Data pipeline example completed")


async def demonstrate_data_transformation():
    """Demonstrate data transformation capabilities."""
    logger.info("Demonstrating Data Transformation Capabilities")
    
    # Create sample data
    from lean_lite.algorithm.data_models import TradeBar, Symbol, SecurityType, Market
    
    symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
    bars = []
    
    # Create sample bars
    base_time = datetime.now() - timedelta(hours=24)
    base_price = 150.0
    
    for i in range(1440):  # 24 hours of minute data
        time = base_time + timedelta(minutes=i)
        price_change = (i % 60 - 30) * 0.1  # Simulate price movement
        price = base_price + price_change
        
        bar = TradeBar(
            Symbol=symbol,
            Time=time,
            Open=price - 0.05,
            High=price + 0.1,
            Low=price - 0.1,
            Close=price,
            Volume=1000 + (i % 100) * 10,
            Period=Resolution.MINUTE
        )
        bars.append(bar)
    
    logger.info(f"Created {len(bars)} sample bars")
    
    # Demonstrate resampling
    from lean_lite.data.data_engine import DataTransformer
    
    # Resample to hourly
    hourly_bars = DataTransformer.resample_bars(bars, Resolution.HOUR)
    logger.info(f"Resampled to {len(hourly_bars)} hourly bars")
    
    # Resample to daily
    daily_bars = DataTransformer.resample_bars(bars, Resolution.DAILY)
    logger.info(f"Resampled to {len(daily_bars)} daily bars")
    
    # Demonstrate normalization
    normalized_data = DataTransformer.normalize_data(bars, method="minmax")
    logger.info(f"Normalized {len(normalized_data)} data points using minmax method")
    
    # Demonstrate technical indicators
    indicators = DataTransformer.calculate_technical_indicators(bars)
    logger.info(f"Calculated {len(indicators)} technical indicators")
    
    for indicator_name, values in indicators.items():
        if values:
            logger.info(f"  {indicator_name}: {values[-1]:.2f}")


async def demonstrate_data_quality():
    """Demonstrate data quality validation."""
    logger.info("Demonstrating Data Quality Validation")
    
    from lean_lite.data.data_engine import DataValidator, DataQualityLevel
    from lean_lite.algorithm.data_models import TradeBar, Symbol, SecurityType, Market
    
    symbol = Symbol("AAPL", SecurityType.EQUITY, Market.US_EQUITY)
    
    # Test valid data
    valid_bar = TradeBar(
        Symbol=symbol,
        Time=datetime.now(),
        Open=150.0,
        High=151.0,
        Low=149.0,
        Close=150.5,
        Volume=1000,
        Period=Resolution.MINUTE
    )
    
    is_valid, quality, message = DataValidator.validate_trade_bar(valid_bar)
    logger.info(f"Valid bar: Valid={is_valid}, Quality={quality.value}, Message={message}")
    
    # Test invalid data (high < close)
    invalid_bar = TradeBar(
        Symbol=symbol,
        Time=datetime.now(),
        Open=150.0,
        High=149.0,  # High < Close
        Low=149.0,
        Close=150.5,
        Volume=1000,
        Period=Resolution.MINUTE
    )
    
    is_valid, quality, message = DataValidator.validate_trade_bar(invalid_bar)
    logger.info(f"Invalid bar: Valid={is_valid}, Quality={quality.value}, Message={message}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(demonstrate_data_transformation())
    asyncio.run(demonstrate_data_quality())
    asyncio.run(main()) 