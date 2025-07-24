"""
Test script to check compatibility of QuantConnect strategy with Lean-Lite.
This will help identify what needs to be implemented or adapted.
"""

import sys
import traceback
from datetime import datetime, timedelta

# Import lean-lite components
from lean_lite.algorithm.qc_algorithm import QCAlgorithm
from lean_lite.algorithm.data_models import Resolution, Symbol, SecurityType, Market
from lean_lite.config import Config

def test_strategy_compatibility():
    """Test if the strategy code compiles and works with lean-lite."""
    
    print("Testing MultidimensionalTransdimensionalPrism Strategy Compatibility")
    print("=" * 70)
    
    # Test 1: Basic class inheritance
    print("\n1. Testing class inheritance...")
    try:
        class MultidimensionalTransdimensionalPrism(QCAlgorithm):
            pass
        print("✅ Class inheritance works")
    except Exception as e:
        print(f"❌ Class inheritance failed: {e}")
        return False
    
    # Test 2: Check if required methods exist
    print("\n2. Testing required methods...")
    required_methods = ['Initialize', 'OnData', 'SetStartDate', 'SetEndDate', 'SetCash', 
                       'AddEquity', 'Schedule', 'SetHoldings', 'liquidate']
    
    for method in required_methods:
        if hasattr(QCAlgorithm, method):
            print(f"✅ {method} method exists")
        else:
            print(f"❌ {method} method missing")
    
    # Test 3: Test basic initialization
    print("\n3. Testing basic initialization...")
    try:
        class TestStrategy(QCAlgorithm):
            def Initialize(self):
                self.SetStartDate(2010, 2, 1)
                self.SetEndDate(2025, 12, 27)
                self.SetCash(100000)
                self.AddEquity("TQQQ", Resolution.HOUR)
        
        strategy = TestStrategy()
        print("✅ Basic initialization works")
    except Exception as e:
        print(f"❌ Basic initialization failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    # Test 4: Test data models
    print("\n4. Testing data models...")
    try:
        # Test Resolution enum
        assert Resolution.HOUR is not None
        assert Resolution.DAILY is not None
        print("✅ Resolution enum works")
        
        # Test Symbol creation
        symbol = Symbol("TQQQ", SecurityType.EQUITY, Market.US_EQUITY)
        print("✅ Symbol creation works")
        
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
    
    # Test 5: Test the actual strategy (with modifications)
    print("\n5. Testing actual strategy (simplified)...")
    try:
        class MultidimensionalTransdimensionalPrism(QCAlgorithm):
            def Initialize(self):
                self.SetStartDate(2010, 2, 1)
                self.SetEndDate(2025, 12, 27)
                self.SetCash(100000)
                self.AddEquity("TQQQ", Resolution.HOUR)
                self.AddEquity("UBT", Resolution.HOUR)
                self.AddEquity("UST", Resolution.HOUR)
                self.tkr = ["TQQQ", "UBT", "UST"]
                
                self.rebal = 2
                self.rebalTimer = self.rebal - 1
                self.flag1 = 0
                
                # Note: Schedule functionality might need implementation
                # self.Schedule.On(self.DateRules.WeekStart("UST"), 
                #                  self.TimeRules.AfterMarketOpen("UST", 150), 
                #                  self.Rebalance)
                
                # Note: VIX data and indicators need implementation
                # self.vix = self.add_data(CBOE, 'VIX', Resolution.DAILY).symbol
                # self.vxv = self.add_data(CBOE, 'VIX3M', Resolution.DAILY).symbol
                # self.vix_50ema = self.ema(self.vix, 50, Resolution.Daily)
                # self.set_warm_up(timedelta(days=50))
            
            def OnData(self, data):
                # Simplified version without VIX logic
                if self.flag1 == 1:
                    for stock in self.tkr:
                        self.SetHoldings(stock, 0.33)
                    self.rebalTimer = 0
                self.flag1 = 0
            
            def Rebalance(self):
                self.rebalTimer += 1
                if self.rebalTimer == self.rebal:
                    self.flag1 = 1
        
        strategy = MultidimensionalTransdimensionalPrism()
        print("✅ Strategy class creation works (simplified)")
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    # Test 6: Check what's missing
    print("\n6. Identifying missing components...")
    missing_components = []
    
    # Check for CBOE data source
    try:
        from lean_lite.data import CBOE
        print("✅ CBOE data source available")
    except ImportError:
        print("❌ CBOE data source missing")
        missing_components.append("CBOE data source")
    
    # Check for indicators
    try:
        from lean_lite.indicators import ema
        print("✅ EMA indicator available")
    except ImportError:
        print("❌ EMA indicator missing")
        missing_components.append("EMA indicator")
    
    # Check for scheduling
    if hasattr(QCAlgorithm, 'Schedule'):
        print("✅ Schedule functionality available")
    else:
        print("❌ Schedule functionality missing")
        missing_components.append("Schedule functionality")
    
    # Check for warm-up
    if hasattr(QCAlgorithm, 'set_warm_up'):
        print("✅ Warm-up functionality available")
    else:
        print("❌ Warm-up functionality missing")
        missing_components.append("Warm-up functionality")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPATIBILITY SUMMARY")
    print("=" * 70)
    
    if missing_components:
        print(f"\n❌ Missing components that need implementation:")
        for component in missing_components:
            print(f"   - {component}")
        
        print(f"\n📋 To make this strategy work, you'll need to implement:")
        print(f"   1. CBOE data source for VIX data")
        print(f"   2. EMA indicator (50-period)")
        print(f"   3. Schedule functionality for weekly rebalancing")
        print(f"   4. Warm-up functionality")
        print(f"   5. VIX ratio calculation (3mo/1mo)")
    else:
        print("\n✅ All components are available! The strategy should work.")
    
    print(f"\n💡 The core algorithm structure is compatible with lean-lite.")
    print(f"   You can start with a simplified version and add features incrementally.")

if __name__ == "__main__":
    test_strategy_compatibility() 