#!/usr/bin/env python3
"""
Test script for Phase G - AI-Driven Autonomous Trading System

This script tests the autonomous trading system including:
- Signal generation
- Risk validation
- Trade execution
- Portfolio management
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the bondx directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bondx'))

from ai_risk_engine.autonomous_trading_system import (
    AutonomousTradingSystem, TradingSignal, TradingAction, StrategyType
)

async def test_autonomous_trading_system():
    """Test the autonomous trading system"""
    print("🚀 Testing Phase G - AI-Driven Autonomous Trading System")
    print("=" * 60)
    
    # Initialize the system
    trading_system = AutonomousTradingSystem()
    
    try:
        # Test 1: System initialization
        print("\n✅ Test 1: System Initialization")
        status = trading_system.get_system_status()
        print(f"   Status: {status['status']}")
        print(f"   Portfolio Value: ${status['trading_stats']['portfolio_value']:,.2f}")
        
        # Test 2: Start the system
        print("\n✅ Test 2: Starting System")
        await trading_system.start()
        await asyncio.sleep(2)  # Wait for system to start
        
        status = trading_system.get_system_status()
        print(f"   Status: {status['status']}")
        
        # Test 3: Check initial state
        print("\n✅ Test 3: Initial State")
        portfolio = trading_system.get_portfolio_state()
        if portfolio:
            print(f"   Cash: ${portfolio['cash']:,.2f}")
            print(f"   Total Value: ${portfolio['total_value']:,.2f}")
            print(f"   Positions: {len(portfolio['positions'])}")
        
        # Test 4: Wait for signal generation
        print("\n✅ Test 4: Signal Generation")
        print("   Waiting for AI signals...")
        await asyncio.sleep(5)  # Wait for signals
        
        signals = trading_system.get_trading_signals()
        print(f"   Generated Signals: {len(signals)}")
        
        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
            print(f"     Signal {i+1}: {signal['symbol']} {signal['action']} "
                  f"({signal['confidence']:.2f} confidence)")
            print(f"       Strategy: {signal['strategy']}")
            print(f"       Reasoning: {signal['reasoning']}")
        
        # Test 5: Wait for trade execution
        print("\n✅ Test 5: Trade Execution")
        print("   Waiting for trade execution...")
        await asyncio.sleep(10)  # Wait for execution
        
        # Test 6: Check execution results
        print("\n✅ Test 6: Execution Results")
        portfolio = trading_system.get_portfolio_state()
        if portfolio:
            print(f"   Cash: ${portfolio['cash']:,.2f}")
            print(f"   Total Value: ${portfolio['total_value']:,.2f}")
            print(f"   Positions: {len(portfolio['positions'])}")
            
            if portfolio['positions']:
                print("   Position Details:")
                for symbol, quantity in portfolio['positions'].items():
                    print(f"     {symbol}: {quantity:,.0f}")
        
        # Test 7: System performance
        print("\n✅ Test 7: System Performance")
        status = trading_system.get_system_status()
        print(f"   Total Signals: {status['trading_stats']['total_signals']}")
        print(f"   Total Trades: {status['trading_stats']['total_trades']}")
        print(f"   Portfolio Value: ${status['trading_stats']['portfolio_value']:,.2f}")
        
        # Test 8: Stop the system
        print("\n✅ Test 8: Stopping System")
        await trading_system.stop()
        
        status = trading_system.get_system_status()
        print(f"   Final Status: {status['status']}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure system is stopped
        try:
            await trading_system.stop()
        except:
            pass

def test_manual_signal_creation():
    """Test manual signal creation"""
    print("\n🔧 Testing Manual Signal Creation")
    print("-" * 40)
    
    try:
        # Create a manual signal
        signal = TradingSignal(
            timestamp=datetime.utcnow(),
            symbol="TEST_BOND",
            action=TradingAction.BUY,
            confidence=0.9,
            quantity=50000.0,
            price=100.0,
            strategy=StrategyType.MACHINE_LEARNING,
            reasoning="Manual test signal",
            risk_score=0.2,
            expected_return=0.015
        )
        
        print(f"✅ Created signal: {signal.symbol} {signal.action.value}")
        print(f"   Confidence: {signal.confidence}")
        print(f"   Quantity: {signal.quantity:,.0f}")
        print(f"   Strategy: {signal.strategy.value}")
        print(f"   Reasoning: {signal.reasoning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual signal creation failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 BondX Phase G - Autonomous Trading System Test Suite")
    print("=" * 70)
    
    # Test manual signal creation
    test_manual_signal_creation()
    
    # Test the full system
    await test_autonomous_trading_system()
    
    print("\n🏁 Test suite completed!")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
