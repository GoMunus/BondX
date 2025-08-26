"""
Comprehensive Testing Suite for BondX Phase 3.

This module tests all Phase 3 components:
- Real-time trading engine
- Risk management
- WebSocket infrastructure
- Mobile APIs
- System integration
"""

import asyncio
import pytest
import json
import websockets
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

from bondx.trading_engine.order_book import OrderBook, OrderBookManager
from bondx.trading_engine.matching_engine import MatchingEngine, Order, OrderSide, OrderType
from bondx.trading_engine.market_maker import MarketMaker, MarketMakerManager
from bondx.trading_engine.order_router import OrderRouter, RoutingStrategy
from bondx.risk_management.real_time_risk import RealTimeRiskEngine, PortfolioPosition
from bondx.websocket.websocket_manager import WebSocketManager


class TestOrderBook:
    """Test order book functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.order_book = OrderBook("TEST_BOND")
    
    def test_add_bid_order(self):
        """Test adding bid orders."""
        order = Order(
            order_id="TEST_ORDER_1",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        
        success = self.order_book.add_order(order)
        assert success is True
        
        # Check order book state
        assert self.order_book.get_best_bid() == Decimal('100.50')
        assert self.order_book.total_bid_volume == Decimal('1000')
        assert len(self.order_book.orders_by_id) == 1
    
    def test_add_ask_order(self):
        """Test adding ask orders."""
        order = Order(
            order_id="TEST_ORDER_2",
            participant_id=2,
            bond_id="TEST_BOND",
            side=OrderSide.SELL,
            quantity=Decimal('500'),
            price=Decimal('101.00'),
            order_type=OrderType.LIMIT
        )
        
        success = self.order_book.add_order(order)
        assert success is True
        
        # Check order book state
        assert self.order_book.get_best_ask() == Decimal('101.00')
        assert self.order_book.total_ask_volume == Decimal('500')
    
    def test_order_matching(self):
        """Test order matching."""
        # Add bid order
        bid_order = Order(
            order_id="BID_1",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        self.order_book.add_order(bid_order)
        
        # Add ask order that should match
        ask_order = Order(
            order_id="ASK_1",
            participant_id=2,
            bond_id="TEST_BOND",
            side=OrderSide.SELL,
            quantity=Decimal('500'),
            price=Decimal('100.00'),  # Lower price, should match
            order_type=OrderType.LIMIT
        )
        self.order_book.add_order(ask_order)
        
        # Check that orders are matched
        # In practice, this would be handled by the matching engine
        assert self.order_book.get_best_ask() is None  # Ask order should be filled
    
    def test_order_removal(self):
        """Test order removal."""
        order = Order(
            order_id="TEST_ORDER_3",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        
        self.order_book.add_order(order)
        assert len(self.order_book.orders_by_id) == 1
        
        success = self.order_book.remove_order("TEST_ORDER_3")
        assert success is True
        assert len(self.order_book.orders_by_id) == 0
        assert self.order_book.total_bid_volume == Decimal('0')


class TestMatchingEngine:
    """Test matching engine functionality."""
    
    @pytest.fixture
    async def matching_engine(self):
        """Create matching engine fixture."""
        order_book_manager = OrderBookManager()
        engine = MatchingEngine(order_book_manager)
        yield engine
        await engine.stop()
    
    async def test_order_submission(self, matching_engine):
        """Test order submission."""
        order = Order(
            order_id="TEST_ORDER_1",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        
        event_id = await matching_engine.submit_order(order)
        assert event_id is not None
        
        # Check that order was added to queue
        assert len(matching_engine.event_queue) == 1
    
    async def test_order_cancellation(self, matching_engine):
        """Test order cancellation."""
        # First submit an order
        order = Order(
            order_id="TEST_ORDER_2",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        
        await matching_engine.submit_order(order)
        
        # Then cancel it
        success = await matching_engine.cancel_order("TEST_ORDER_2", 1)
        assert success is True
        
        # Check that cancellation event was added to queue
        assert len(matching_engine.event_queue) == 2  # Submit + cancel


class TestMarketMaker:
    """Test market maker functionality."""
    
    @pytest.fixture
    async def market_maker(self):
        """Create market maker fixture."""
        order_book_manager = OrderBookManager()
        maker = MarketMaker(
            instrument_id="TEST_BOND",
            order_book_manager=order_book_manager
        )
        yield maker
        await maker.stop()
    
    async def test_market_maker_start_stop(self, market_maker):
        """Test market maker start/stop."""
        assert market_maker.is_running is False
        
        await market_maker.start()
        assert market_maker.is_running is True
        assert market_maker.status.value == "ACTIVE"
        
        await market_maker.stop()
        assert market_maker.is_running is False
        assert market_maker.status.value == "INACTIVE"
    
    async def test_fair_value_calculation(self, market_maker):
        """Test fair value calculation."""
        # Start market maker to enable fair value updates
        await market_maker.start()
        
        # Wait for fair value calculation
        await asyncio.sleep(1)
        
        fair_value = market_maker.get_fair_value()
        assert fair_value is not None
        assert fair_value.fair_value > 0
        
        await market_maker.stop()
    
    async def test_quote_generation(self, market_maker):
        """Test quote generation."""
        await market_maker.start()
        
        # Wait for quotes to be generated
        await asyncio.sleep(2)
        
        # Check that quotes were created
        assert len(market_maker.active_quotes) > 0
        
        await market_maker.stop()


class TestOrderRouter:
    """Test order router functionality."""
    
    @pytest.fixture
    async def order_router(self):
        """Create order router fixture."""
        order_book_manager = OrderBookManager()
        matching_engine = MatchingEngine(order_book_manager)
        auction_engine = None  # Mock for testing
        
        router = OrderRouter(
            order_book_manager=order_book_manager,
            matching_engine=matching_engine,
            auction_engine=auction_engine
        )
        yield router
    
    async def test_venue_initialization(self, order_router):
        """Test venue initialization."""
        # Check that internal venues are initialized
        assert "INTERNAL_CONTINUOUS" in order_router.venues
        assert "INTERNAL_AUCTION" in order_router.venues
        
        # Check venue info
        continuous_venue = order_router.venues["INTERNAL_CONTINUOUS"]
        assert continuous_venue.venue_type.value == "INTERNAL_CONTINUOUS"
        assert continuous_venue.is_active is True
    
    async def test_routing_decision(self, order_router):
        """Test routing decision generation."""
        order = Order(
            order_id="TEST_ORDER_1",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        
        # Route order
        routing_decision = await order_router.route_order(
            order, 
            strategy=RoutingStrategy.SMART_ROUTING
        )
        
        assert routing_decision is not None
        assert routing_decision.order_id == "TEST_ORDER_1"
        assert routing_decision.strategy == RoutingStrategy.SMART_ROUTING
        assert routing_decision.primary_venue is not None


class TestRiskEngine:
    """Test risk management functionality."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create risk engine fixture."""
        return RealTimeRiskEngine()
    
    def test_portfolio_position_management(self, risk_engine):
        """Test portfolio position management."""
        position = PortfolioPosition(
            instrument_id="TEST_BOND",
            quantity=Decimal('10000'),
            market_value=Decimal('1000000'),
            clean_price=Decimal('100.00'),
            dirty_price=Decimal('100.50'),
            duration=Decimal('5.0'),
            convexity=Decimal('25.0'),
            dv01=Decimal('500'),
            yield_to_maturity=Decimal('6.5'),
            credit_rating="AA",
            sector="GOVERNMENT",
            issuer="TEST_ISSUER",
            maturity_date=datetime.utcnow(),
            last_update=datetime.utcnow()
        )
        
        # Add position to portfolio
        risk_engine.add_portfolio_position("TEST_PORTFOLIO", position)
        
        # Check that position was added
        assert "TEST_PORTFOLIO" in risk_engine.portfolios
        assert "TEST_BOND" in risk_engine.portfolios["TEST_PORTFOLIO"]
    
    def test_risk_limit_management(self, risk_engine):
        """Test risk limit management."""
        from bondx.risk_management.risk_models import RiskLimit
        
        limit = RiskLimit(
            limit_id="TEST_LIMIT",
            portfolio_id="TEST_PORTFOLIO",
            participant_id=1,
            limit_type="MARKET_VALUE",
            limit_value=Decimal('10000000'),
            current_value=Decimal('5000000'),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Add risk limit
        risk_engine.add_risk_limit(limit)
        
        # Check that limit was added
        limit_key = f"TEST_PORTFOLIO_MARKET_VALUE"
        assert limit_key in risk_engine.risk_limits


class TestWebSocketManager:
    """Test WebSocket functionality."""
    
    @pytest.fixture
    async def websocket_manager(self):
        """Create WebSocket manager fixture."""
        manager = WebSocketManager()
        yield manager
        await manager.stop()
    
    async def test_websocket_start_stop(self, websocket_manager):
        """Test WebSocket manager start/stop."""
        assert websocket_manager.is_running is False
        
        await websocket_manager.start()
        assert websocket_manager.is_running is True
        
        await websocket_manager.stop()
        assert websocket_manager.is_running is False
    
    async def test_market_data_broadcasting(self, websocket_manager):
        """Test market data broadcasting."""
        await websocket_manager.start()
        
        # Broadcast market data
        market_data = {
            "best_bid": 100.50,
            "best_ask": 100.75,
            "last_price": 100.60,
            "volume": 1000000
        }
        
        await websocket_manager.broadcast_market_data("TEST_BOND", market_data)
        
        # Check that message was queued
        assert len(websocket_manager.message_queue) > 0
        
        await websocket_manager.stop()


class TestSystemIntegration:
    """Test system integration."""
    
    async def test_complete_trading_flow(self):
        """Test complete trading flow."""
        # Initialize components
        order_book_manager = OrderBookManager()
        matching_engine = MatchingEngine(order_book_manager)
        risk_engine = RealTimeRiskEngine()
        
        # Start matching engine
        await matching_engine.start()
        
        # Create and submit order
        order = Order(
            order_id="INTEGRATION_TEST_ORDER",
            participant_id=1,
            bond_id="TEST_BOND",
            side=OrderSide.BUY,
            quantity=Decimal('1000'),
            price=Decimal('100.50'),
            order_type=OrderType.LIMIT
        )
        
        event_id = await matching_engine.submit_order(order)
        assert event_id is not None
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check order book
        order_book = order_book_manager.get_order_book("TEST_BOND")
        assert order_book is not None
        
        # Check that order was added
        assert len(order_book.orders_by_id) == 1
        
        # Stop components
        await matching_engine.stop()
    
    async def test_risk_integration(self):
        """Test risk management integration."""
        # Initialize risk engine
        risk_engine = RealTimeRiskEngine()
        
        # Add portfolio position
        position = PortfolioPosition(
            instrument_id="TEST_BOND",
            quantity=Decimal('10000'),
            market_value=Decimal('1000000'),
            clean_price=Decimal('100.00'),
            dirty_price=Decimal('100.50'),
            duration=Decimal('5.0'),
            convexity=Decimal('25.0'),
            dv01=Decimal('500'),
            yield_to_maturity=Decimal('6.5'),
            credit_rating="AA",
            sector="GOVERNMENT",
            issuer="TEST_ISSUER",
            maturity_date=datetime.utcnow(),
            last_update=datetime.utcnow()
        )
        
        risk_engine.add_portfolio_position("TEST_PORTFOLIO", position)
        
        # Calculate portfolio risk
        risk_snapshot = await risk_engine.calculate_portfolio_risk("TEST_PORTFOLIO")
        
        assert risk_snapshot is not None
        assert risk_snapshot.portfolio_id == "TEST_PORTFOLIO"
        assert risk_snapshot.total_market_value == Decimal('1000000')


# Performance tests
class TestPerformance:
    """Performance tests for Phase 3 components."""
    
    async def test_order_book_performance(self):
        """Test order book performance under load."""
        order_book = OrderBook("PERF_TEST_BOND")
        
        # Add many orders
        start_time = datetime.utcnow()
        
        for i in range(1000):
            order = Order(
                order_id=f"PERF_ORDER_{i}",
                participant_id=i % 10,
                bond_id="PERF_TEST_BOND",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=Decimal('1000'),
                price=Decimal('100.00') + Decimal(str(i % 100)),
                order_type=OrderType.LIMIT
            )
            order_book.add_order(order)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert duration < 1.0  # Less than 1 second
        
        # Check final state
        assert len(order_book.orders_by_id) == 1000
    
    async def test_matching_engine_performance(self):
        """Test matching engine performance."""
        order_book_manager = OrderBookManager()
        matching_engine = MatchingEngine(order_book_manager)
        
        await matching_engine.start()
        
        # Submit many orders
        start_time = datetime.utcnow()
        
        for i in range(100):
            order = Order(
                order_id=f"PERF_MATCH_ORDER_{i}",
                participant_id=i % 10,
                bond_id="PERF_TEST_BOND",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=Decimal('1000'),
                price=Decimal('100.00') + Decimal(str(i % 50)),
                order_type=OrderType.LIMIT
            )
            await matching_engine.submit_order(order)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert duration < 2.0  # Less than 2 seconds
        
        await matching_engine.stop()


# Load testing
class TestLoadTesting:
    """Load testing for system components."""
    
    async def test_websocket_load(self):
        """Test WebSocket manager under load."""
        websocket_manager = WebSocketManager()
        await websocket_manager.start()
        
        # Simulate many market data updates
        start_time = datetime.utcnow()
        
        for i in range(1000):
            market_data = {
                "best_bid": 100.00 + (i % 100) * 0.01,
                "best_ask": 100.50 + (i % 100) * 0.01,
                "last_price": 100.25 + (i % 100) * 0.01,
                "volume": 1000000 + i * 1000
            }
            
            await websocket_manager.broadcast_market_data(f"BOND_{i % 10}", market_data)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should handle load efficiently
        assert duration < 5.0  # Less than 5 seconds
        
        await websocket_manager.stop()


# Main test runner
async def run_all_tests():
    """Run all Phase 3 tests."""
    print("Starting BondX Phase 3 Integration Tests...")
    
    # Test categories
    test_classes = [
        TestOrderBook,
        TestMatchingEngine,
        TestMarketMaker,
        TestOrderRouter,
        TestRiskEngine,
        TestWebSocketManager,
        TestSystemIntegration,
        TestPerformance,
        TestLoadTesting
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_') and callable(getattr(test_class, method))]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Run test
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run async test
                if asyncio.iscoroutinefunction(getattr(test_instance, test_method)):
                    await getattr(test_instance, test_method)()
                else:
                    getattr(test_instance, test_method)()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
                failed_tests += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"{'='*50}")
    
    return failed_tests == 0


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🎉 All Phase 3 tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        exit(1)
