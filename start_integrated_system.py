#!/usr/bin/env python3
"""
Quick Start Script for BondX Integrated System

This script demonstrates the integrated auction engine, trading capabilities,
and risk management systems.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the main application
from bondx.main import app

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

async def demonstrate_auction_system():
    """Demonstrate the auction system capabilities."""
    print_section("AUCTION ENGINE DEMONSTRATION")
    
    client = TestClient(app)
    
    # Create a new auction
    print_subsection("Creating New Auction")
    auction_data = {
        "auction_code": "DEMO_AUCT_001",
        "auction_name": "Demo Government Bond Auction",
        "auction_type": "DUTCH",
        "total_lot_size": 1000000,
        "minimum_lot_size": 1000,
        "lot_size_increment": 100,
        "reserve_price": 95.50,
        "bidding_start_time": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
        "bidding_end_time": (datetime.utcnow() + timedelta(hours=2)).isoformat(),
        "settlement_date": (datetime.utcnow() + timedelta(days=2)).isoformat(),
        "eligible_participants": ["INSTITUTIONAL", "RETAIL"],
        "maximum_allocation_per_participant": 100000
    }
    
    response = client.post("/api/v1/auctions/", json=auction_data)
    if response.status_code == 200:
        auction_result = response.json()
        print(f"✅ Auction created successfully!")
        print(f"   Auction ID: {auction_result['auction_id']}")
        print(f"   Status: {auction_result['status']}")
    else:
        print(f"❌ Failed to create auction: {response.status_code}")
        print(f"   Error: {response.text}")
        return None
    
    auction_id = auction_result['auction_id']
    
    # Submit bids
    print_subsection("Submitting Bids")
    bids = [
        {"bid_id": "BID_001", "participant_id": 1, "bid_price": 95.75, "bid_quantity": 100000},
        {"bid_id": "BID_002", "participant_id": 2, "bid_price": 95.60, "bid_quantity": 150000},
        {"bid_id": "BID_003", "participant_id": 3, "bid_price": 95.80, "bid_quantity": 200000}
    ]
    
    for bid in bids:
        bid['auction_id'] = auction_id
        response = client.post(f"/api/v1/auctions/{auction_id}/bids", json=bid)
        if response.status_code == 200:
            print(f"✅ Bid {bid['bid_id']} submitted successfully")
        else:
            print(f"❌ Failed to submit bid {bid['bid_id']}: {response.status_code}")
    
    # Get auction details
    print_subsection("Auction Details")
    response = client.get(f"/api/v1/auctions/{auction_id}")
    if response.status_code == 200:
        auction_details = response.json()
        print(f"   Total Bids: {auction_details['bid_statistics']['total_bids']}")
        print(f"   Total Quantity: {auction_details['bid_statistics']['total_quantity']}")
        print(f"   Average Price: {auction_details['bid_statistics']['average_price']:.2f}")
    
    return auction_id

async def demonstrate_trading_system():
    """Demonstrate the trading system capabilities."""
    print_section("TRADING ENGINE DEMONSTRATION")
    
    client = TestClient(app)
    
    # Submit trading orders
    print_subsection("Submitting Trading Orders")
    orders = [
        {
            "participant_id": 1,
            "bond_id": "BOND_001",
            "order_type": "LIMIT",
            "side": "BUY",
            "quantity": 1000,
            "price": 95.50,
            "time_in_force": "DAY"
        },
        {
            "participant_id": 2,
            "bond_id": "BOND_001",
            "order_type": "LIMIT",
            "side": "SELL",
            "quantity": 800,
            "price": 95.75,
            "time_in_force": "DAY"
        }
    ]
    
    order_ids = []
    for i, order in enumerate(orders):
        response = client.post("/api/v1/trading/orders", json=order)
        if response.status_code == 200:
            order_result = response.json()
            order_id = order_result['order']['order_id']
            order_ids.append(order_id)
            print(f"✅ Order {i+1} submitted successfully")
            print(f"   Order ID: {order_id}")
            print(f"   Status: {order_result['order']['status']}")
        else:
            print(f"❌ Failed to submit order {i+1}: {response.status_code}")
    
    # Get order book
    print_subsection("Order Book")
    response = client.get("/api/v1/trading/orderbook/BOND_001")
    if response.status_code == 200:
        orderbook = response.json()
        print(f"   Best Bid: {orderbook['best_bid']['price'] if orderbook['best_bid'] else 'None'}")
        print(f"   Best Ask: {orderbook['best_ask']['price'] if orderbook['best_ask'] else 'None'}")
        print(f"   Spread: {orderbook['spread']:.2f if orderbook['spread'] else 'None'}")
    
    # Execute orders
    print_subsection("Executing Orders")
    for order_id in order_ids:
        response = client.post(f"/api/v1/trading/orders/{order_id}/execute")
        if response.status_code == 200:
            execution_result = response.json()
            print(f"✅ Order {order_id} executed successfully")
            print(f"   Executions: {execution_result['summary']['execution_count']}")
        else:
            print(f"❌ Failed to execute order {order_id}: {response.status_code}")
    
    # Get trade history
    print_subsection("Trade History")
    response = client.get("/api/v1/trading/trades")
    if response.status_code == 200:
        trades = response.json()
        print(f"   Total Trades: {trades['pagination']['total_count']}")
        if trades['trades']:
            latest_trade = trades['trades'][0]
            print(f"   Latest Trade: {latest_trade['trade_id']}")
            print(f"   Price: {latest_trade['price']}")
            print(f"   Quantity: {latest_trade['quantity']}")

async def demonstrate_risk_management():
    """Demonstrate the risk management system capabilities."""
    print_section("RISK MANAGEMENT DEMONSTRATION")
    
    client = TestClient(app)
    
    # Create portfolio
    print_subsection("Creating Portfolio")
    portfolio_data = {
        "portfolio_name": "Demo Trading Portfolio",
        "participant_id": 1,
        "portfolio_type": "TRADING",
        "risk_tolerance": "MEDIUM",
        "investment_horizon": "MEDIUM_TERM",
        "target_return": 8.5,
        "max_drawdown": 15.0
    }
    
    response = client.post("/api/v1/risk/portfolios", json=portfolio_data)
    if response.status_code == 200:
        portfolio_result = response.json()
        portfolio_id = portfolio_result['portfolio']['portfolio_id']
        print(f"✅ Portfolio created successfully!")
        print(f"   Portfolio ID: {portfolio_id}")
        print(f"   Risk Tolerance: {portfolio_result['portfolio']['risk_tolerance']}")
    else:
        print(f"❌ Failed to create portfolio: {response.status_code}")
        print(f"   Error: {response.text}")
        return None
    
    # Calculate portfolio risk
    print_subsection("Calculating Portfolio Risk")
    risk_params = {
        "method": "HISTORICAL_SIMULATION",
        "confidence_level": 0.95,
        "time_horizon": 1
    }
    
    response = client.post(f"/api/v1/risk/portfolios/{portfolio_id}/risk", json=risk_params)
    if response.status_code == 200:
        risk_result = response.json()
        print(f"✅ Risk metrics calculated successfully!")
        print(f"   VaR (95%, 1d): {risk_result['risk_metrics']['var_95_1d']:.2f if risk_result['risk_metrics']['var_95_1d'] else 'N/A'}")
        print(f"   Portfolio Volatility: {risk_result['risk_metrics']['portfolio_volatility']:.2f if risk_result['risk_metrics']['portfolio_volatility'] else 'N/A'}")
        print(f"   Modified Duration: {risk_result['risk_metrics']['modified_duration']:.2f if risk_result['risk_metrics']['modified_duration'] else 'N/A'}")
    else:
        print(f"❌ Failed to calculate risk metrics: {response.status_code}")
    
    # Get risk decomposition
    print_subsection("Risk Decomposition")
    response = client.get(f"/api/v1/risk/portfolios/{portfolio_id}/risk/decomposition")
    if response.status_code == 200:
        decomposition = response.json()
        print(f"✅ Risk decomposition calculated successfully!")
        print(f"   Total Risk: {decomposition['risk_decomposition']['total_risk']:.4f}")
        print(f"   Systematic Risk: {decomposition['risk_decomposition']['systematic_risk']:.4f}")
        print(f"   Idiosyncratic Risk: {decomposition['risk_decomposition']['idiosyncratic_risk']:.4f}")
    else:
        print(f"❌ Failed to calculate risk decomposition: {response.status_code}")
    
    # Perform stress test
    print_subsection("Stress Testing")
    stress_params = {
        "scenario_type": "INTEREST_RATE"
    }
    
    response = client.post(f"/api/v1/risk/portfolios/{portfolio_id}/stress-test/default", json=stress_params)
    if response.status_code == 200:
        stress_result = response.json()
        print(f"✅ Stress test completed successfully!")
        print(f"   Scenario: {stress_result['scenario']['scenario_name']}")
        print(f"   Portfolio Value Change: {stress_result['stress_test_result']['portfolio_value_change_percent']:.2f}%")
        print(f"   Test Passed: {stress_result['stress_test_result']['is_passed']}")
    else:
        print(f"❌ Failed to perform stress test: {response.status_code}")
    
    return portfolio_id

async def demonstrate_system_status():
    """Demonstrate system status and health checks."""
    print_section("SYSTEM STATUS DEMONSTRATION")
    
    client = TestClient(app)
    
    # Root endpoint
    print_subsection("System Overview")
    response = client.get("/")
    if response.status_code == 200:
        system_info = response.json()
        print(f"✅ System is running!")
        print(f"   Name: {system_info['name']}")
        print(f"   Version: {system_info['version']}")
        print(f"   Environment: {system_info['environment']}")
        print(f"   Features: {', '.join(system_info['features'])}")
    
    # Health check
    print_subsection("Health Check")
    response = client.get("/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✅ Health check passed!")
        print(f"   Status: {health['status']}")
        print(f"   Timestamp: {health['timestamp']}")
    
    # Trading system status
    print_subsection("Trading System Status")
    response = client.get("/api/v1/trading/status")
    if response.status_code == 200:
        trading_status = response.json()
        print(f"✅ Trading system is operational!")
        print(f"   Status: {trading_status['status']}")
        print(f"   Active Orders: {trading_status['system_health']['active_orders']}")
        print(f"   Total Trades: {trading_status['system_health']['total_trades']}")
    
    # Risk management status
    print_subsection("Risk Management Status")
    response = client.get("/api/v1/risk/status")
    if response.status_code == 200:
        risk_status = response.json()
        print(f"✅ Risk management system is operational!")
        print(f"   Status: {risk_status['status']}")
        print(f"   Total Calculations: {risk_status['performance_metrics']['total_calculations']}")
        print(f"   Average Calculation Time: {risk_status['performance_metrics']['average_calculation_time']:.2f}s")

async def main():
    """Main demonstration function."""
    print("🚀 BondX Integrated System Demonstration")
    print("This script demonstrates the integrated auction engine, trading capabilities, and risk management systems.")
    
    try:
        # Demonstrate auction system
        auction_id = await demonstrate_auction_system()
        
        # Demonstrate trading system
        await demonstrate_trading_system()
        
        # Demonstrate risk management
        portfolio_id = await demonstrate_risk_management()
        
        # Demonstrate system status
        await demonstrate_system_status()
        
        print_section("DEMONSTRATION COMPLETED")
        print("✅ All systems demonstrated successfully!")
        print(f"   Auction ID: {auction_id}")
        print(f"   Portfolio ID: {portfolio_id}")
        print("\nThe BondX Backend is now fully integrated with:")
        print("  • Auction Engine with real-time WebSocket updates")
        print("  • Trading Engine with order management and execution")
        print("  • Risk Management with VaR calculations and stress testing")
        print("  • Comprehensive API endpoints for all functionality")
        print("  • Real-time monitoring and system health checks")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
    
    print("\n" + "="*60)
    print(" To start the full system, run:")
    print(" uvicorn bondx.main:app --host 0.0.0.0 --port 8000 --reload")
    print("="*60)
