"""
Smart Order Routing for BondX Trading Engine.

This module implements intelligent order routing with:
- Venue selection (internal auction, continuous book, external connectors)
- Slippage minimization using depth and latency estimates
- Order splitting across venues with time slicing
- Adapter pattern for external venues
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
from uuid import uuid4

from ..core.logging import get_logger
from .trading_models import Order, OrderSide, OrderType, TimeInForce
from .order_book import OrderBook, OrderBookManager
from .matching_engine import MatchingEngine
from ..auction_engine.auction_engine import AuctionEngine

logger = get_logger(__name__)


class VenueType(Enum):
    """Trading venue types."""
    INTERNAL_AUCTION = "INTERNAL_AUCTION"
    INTERNAL_CONTINUOUS = "INTERNAL_CONTINUOUS"
    EXTERNAL_PRIMARY = "EXTERNAL_PRIMARY"
    EXTERNAL_SECONDARY = "EXTERNAL_SECONDARY"
    EXTERNAL_DARK_POOL = "EXTERNAL_DARK_POOL"


class RoutingStrategy(Enum):
    """Order routing strategies."""
    BEST_EXECUTION = "BEST_EXECUTION"
    LOWEST_COST = "LOWEST_COST"
    FASTEST_FILL = "FASTEST_FILL"
    LIQUIDITY_SEEKING = "LIQUIDITY_SEEKING"
    SMART_ROUTING = "SMART_ROUTING"


@dataclass
class VenueInfo:
    """Information about a trading venue."""
    venue_id: str
    venue_type: VenueType
    name: str
    is_active: bool
    latency_ms: float
    fees_bps: Decimal
    min_order_size: Decimal
    max_order_size: Decimal
    supported_instruments: List[str]
    last_health_check: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionEstimate:
    """Execution cost and probability estimate."""
    venue_id: str
    venue_type: VenueType
    estimated_price: Decimal
    estimated_quantity: Decimal
    execution_probability: float
    estimated_latency_ms: float
    estimated_fees: Decimal
    total_cost: Decimal
    confidence_score: float


@dataclass
class RoutingDecision:
    """Routing decision for an order."""
    order_id: str
    strategy: RoutingStrategy
    primary_venue: str
    fallback_venues: List[str]
    split_orders: List[Dict[str, Any]]
    estimated_total_cost: Decimal
    estimated_fill_probability: float
    routing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VenueAdapter:
    """Abstract adapter for external venues."""
    venue_id: str
    venue_type: VenueType
    is_connected: bool
    connection_metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def connect(self) -> bool:
        """Connect to the venue."""
        raise NotImplementedError
    
    async def disconnect(self) -> bool:
        """Disconnect from the venue."""
        raise NotImplementedError
    
    async def submit_order(self, order: Order) -> str:
        """Submit order to venue."""
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order at venue."""
        raise NotImplementedError
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from venue."""
        raise NotImplementedError
    
    async def get_market_data(self, instrument_id: str) -> Dict[str, Any]:
        """Get market data from venue."""
        raise NotImplementedError


class MockExternalVenueAdapter(VenueAdapter):
    """Mock adapter for external venues (for development/testing)."""
    
    def __init__(self, venue_id: str, venue_type: VenueType):
        super().__init__(venue_id, venue_type)
        self.orders: Dict[str, Order] = {}
        self.order_statuses: Dict[str, str] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self) -> bool:
        """Mock connection."""
        self.is_connected = True
        logger.info(f"Mock connected to {self.venue_id}")
        return True
    
    async def disconnect(self) -> bool:
        """Mock disconnection."""
        self.is_connected = False
        logger.info(f"Mock disconnected from {self.venue_id}")
        return True
    
    async def submit_order(self, order: Order) -> str:
        """Mock order submission."""
        if not self.is_connected:
            raise Exception("Not connected to venue")
        
        self.orders[order.order_id] = order
        self.order_statuses[order.order_id] = "SUBMITTED"
        
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.01, 0.1))
        
        # Randomly fill or reject
        if random.random() > 0.3:  # 70% fill rate
            self.order_statuses[order.order_id] = "FILLED"
        else:
            self.order_statuses[order.order_id] = "REJECTED"
        
        return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        if order_id in self.order_statuses:
            self.order_statuses[order_id] = "CANCELLED"
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Mock order status."""
        if order_id in self.order_statuses:
            return {
                "order_id": order_id,
                "status": self.order_statuses[order_id],
                "venue_id": self.venue_id
            }
        return {"order_id": order_id, "status": "NOT_FOUND", "venue_id": self.venue_id}
    
    async def get_market_data(self, instrument_id: str) -> Dict[str, Any]:
        """Mock market data."""
        if instrument_id not in self.market_data:
            # Generate mock data
            self.market_data[instrument_id] = {
                "bid": Decimal(str(random.uniform(95, 105))),
                "ask": Decimal(str(random.uniform(95, 105))),
                "last": Decimal(str(random.uniform(95, 105))),
                "volume": Decimal(str(random.uniform(1000000, 10000000))),
                "timestamp": datetime.utcnow()
            }
        
        return self.market_data[instrument_id]


class OrderRouter:
    """
    Smart order router for optimal execution.
    
    Features:
    - Multi-venue routing
    - Slippage minimization
    - Order splitting and time slicing
    - Cost optimization
    - Health monitoring
    """
    
    def __init__(self, 
                 order_book_manager: OrderBookManager,
                 matching_engine: MatchingEngine,
                 auction_engine: AuctionEngine):
        """Initialize the order router."""
        self.order_book_manager = order_book_manager
        self.matching_engine = matching_engine
        self.auction_engine = auction_engine
        
        # Venue management
        self.venues: Dict[str, VenueInfo] = {}
        self.venue_adapters: Dict[str, VenueAdapter] = {}
        
        # Routing configuration
        self.default_strategy = RoutingStrategy.SMART_ROUTING
        self.max_venue_latency_ms = 1000  # 1 second
        self.min_execution_probability = 0.3
        self.max_order_splits = 5
        self.split_size_threshold = Decimal('1000000')  # 1M notional
        
        # Performance tracking
        self.orders_routed = 0
        self.routing_decisions = 0
        self.start_time = datetime.utcnow()
        
        # Initialize internal venues
        self._initialize_internal_venues()
        
        logger.info("Order Router initialized successfully")
    
    def _initialize_internal_venues(self) -> None:
        """Initialize internal trading venues."""
        # Internal continuous trading
        internal_continuous = VenueInfo(
            venue_id="INTERNAL_CONTINUOUS",
            venue_type=VenueType.INTERNAL_CONTINUOUS,
            name="BondX Continuous Trading",
            is_active=True,
            latency_ms=1.0,  # 1ms latency
            fees_bps=Decimal('0.5'),  # 0.5 basis points
            min_order_size=Decimal('10000'),  # 10k minimum
            max_order_size=Decimal('100000000'),  # 100M maximum
            supported_instruments=[],  # All instruments
            last_health_check=datetime.utcnow()
        )
        
        # Internal auction
        internal_auction = VenueInfo(
            venue_id="INTERNAL_AUCTION",
            venue_type=VenueType.INTERNAL_AUCTION,
            name="BondX Auction",
            is_active=True,
            latency_ms=5.0,  # 5ms latency
            fees_bps=Decimal('0.3'),  # 0.3 basis points
            min_order_size=Decimal('100000'),  # 100k minimum
            max_order_size=Decimal('1000000000'),  # 1B maximum
            supported_instruments=[],  # All instruments
            last_health_check=datetime.utcnow()
        )
        
        self.venues["INTERNAL_CONTINUOUS"] = internal_continuous
        self.venues["INTERNAL_AUCTION"] = internal_auction
        
        logger.info("Internal venues initialized")
    
    def add_external_venue(self, venue_info: VenueInfo, adapter: VenueAdapter) -> bool:
        """Add an external venue with adapter."""
        try:
            self.venues[venue_info.venue_id] = venue_info
            self.venue_adapters[venue_info.venue_id] = adapter
            logger.info(f"Added external venue: {venue_info.venue_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding external venue {venue_info.venue_id}: {e}")
            return False
    
    def remove_venue(self, venue_id: str) -> bool:
        """Remove a venue."""
        try:
            if venue_id in self.venues:
                del self.venues[venue_id]
            
            if venue_id in self.venue_adapters:
                adapter = self.venue_adapters[venue_id]
                asyncio.create_task(adapter.disconnect())
                del self.venue_adapters[venue_id]
            
            logger.info(f"Removed venue: {venue_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing venue {venue_id}: {e}")
            return False
    
    async def route_order(self, order: Order, strategy: Optional[RoutingStrategy] = None) -> RoutingDecision:
        """
        Route an order to the best venue(s).
        
        Args:
            order: Order to route
            strategy: Routing strategy (uses default if None)
            
        Returns:
            Routing decision
        """
        if strategy is None:
            strategy = self.default_strategy
        
        try:
            # Get available venues for the instrument
            available_venues = self._get_available_venues(order.bond_id)
            
            if not available_venues:
                raise Exception(f"No available venues for instrument {order.bond_id}")
            
            # Get execution estimates for each venue
            execution_estimates = await self._get_execution_estimates(order, available_venues)
            
            # Apply routing strategy
            routing_decision = self._apply_routing_strategy(order, strategy, execution_estimates)
            
            # Update statistics
            self.orders_routed += 1
            self.routing_decisions += 1
            
            logger.info(f"Order {order.order_id} routed using {strategy.value}")
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error routing order {order.order_id}: {e}")
            raise
    
    def _get_available_venues(self, instrument_id: str) -> List[VenueInfo]:
        """Get available venues for an instrument."""
        available_venues = []
        
        for venue in self.venues.values():
            if not venue.is_active:
                continue
            
            # Check if venue supports the instrument
            if (venue.supported_instruments == [] or  # All instruments
                instrument_id in venue.supported_instruments):
                available_venues.append(venue)
        
        return available_venues
    
    async def _get_execution_estimates(self, order: Order, venues: List[VenueInfo]) -> List[ExecutionEstimate]:
        """Get execution estimates for each venue."""
        estimates = []
        
        for venue in venues:
            try:
                estimate = await self._estimate_execution(order, venue)
                if estimate:
                    estimates.append(estimate)
            except Exception as e:
                logger.warning(f"Error estimating execution for venue {venue.venue_id}: {e}")
                continue
        
        return estimates
    
    async def _estimate_execution(self, order: Order, venue: VenueInfo) -> Optional[ExecutionEstimate]:
        """Estimate execution at a specific venue."""
        try:
            if venue.venue_type == VenueType.INTERNAL_CONTINUOUS:
                return await self._estimate_internal_continuous(order, venue)
            elif venue.venue_type == VenueType.INTERNAL_AUCTION:
                return await self._estimate_internal_auction(order, venue)
            elif venue.venue_type in [VenueType.EXTERNAL_PRIMARY, VenueType.EXTERNAL_SECONDARY, VenueType.EXTERNAL_DARK_POOL]:
                return await self._estimate_external_venue(order, venue)
            else:
                logger.warning(f"Unknown venue type: {venue.venue_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error estimating execution for venue {venue.venue_id}: {e}")
            return None
    
    async def _estimate_internal_continuous(self, order: Order, venue: VenueInfo) -> ExecutionEstimate:
        """Estimate execution on internal continuous venue."""
        try:
            # Get order book data
            order_book = self.order_book_manager.get_order_book(order.bond_id)
            market_data = order_book.get_level_1_snapshot()
            
            # Estimate price based on order side and market data
            if order.side == OrderSide.BUY:
                # For buy orders, estimate based on ask side
                estimated_price = market_data.get('best_ask')
                if estimated_price is None:
                    estimated_price = order.price or Decimal('100')
            else:
                # For sell orders, estimate based on bid side
                estimated_price = market_data.get('best_bid')
                if estimated_price is None:
                    estimated_price = order.price or Decimal('100')
            
            # Estimate execution probability based on market depth
            execution_probability = self._estimate_execution_probability(order, market_data)
            
            # Calculate fees
            estimated_fees = (order.quantity * estimated_price * venue.fees_bps) / Decimal('10000')
            
            # Total cost
            total_cost = (order.quantity * estimated_price) + estimated_fees
            
            return ExecutionEstimate(
                venue_id=venue.venue_id,
                venue_type=venue.venue_type,
                estimated_price=estimated_price,
                estimated_quantity=order.quantity,
                execution_probability=execution_probability,
                estimated_latency_ms=venue.latency_ms,
                estimated_fees=estimated_fees,
                total_cost=total_cost,
                confidence_score=0.9  # High confidence for internal venue
            )
            
        except Exception as e:
            logger.error(f"Error estimating internal continuous execution: {e}")
            raise
    
    async def _estimate_internal_auction(self, order: Order, venue: VenueInfo) -> ExecutionEstimate:
        """Estimate execution on internal auction venue."""
        try:
            # For auctions, we need to estimate based on historical data or current auction state
            # This is a simplified implementation
            
            # Use order price as estimated price
            estimated_price = order.price or Decimal('100')
            
            # Auctions typically have higher execution probability for large orders
            if order.quantity > self.split_size_threshold:
                execution_probability = 0.8
            else:
                execution_probability = 0.6
            
            # Calculate fees
            estimated_fees = (order.quantity * estimated_price * venue.fees_bps) / Decimal('10000')
            
            # Total cost
            total_cost = (order.quantity * estimated_price) + estimated_fees
            
            return ExecutionEstimate(
                venue_id=venue.venue_id,
                venue_type=venue.venue_type,
                estimated_price=estimated_price,
                estimated_quantity=order.quantity,
                execution_probability=execution_probability,
                estimated_latency_ms=venue.latency_ms,
                estimated_fees=estimated_fees,
                total_cost=total_cost,
                confidence_score=0.8  # Good confidence for internal auction
            )
            
        except Exception as e:
            logger.error(f"Error estimating internal auction execution: {e}")
            raise
    
    async def _estimate_external_venue(self, order: Order, venue: VenueInfo) -> ExecutionEstimate:
        """Estimate execution on external venue."""
        try:
            # Get market data from external venue
            adapter = self.venue_adapters.get(venue.venue_id)
            if not adapter or not adapter.is_connected:
                return None
            
            market_data = await adapter.get_market_data(order.bond_id)
            
            # Estimate price from external market data
            if order.side == OrderSide.BUY:
                estimated_price = market_data.get('ask', order.price or Decimal('100'))
            else:
                estimated_price = market_data.get('bid', order.price or Decimal('100'))
            
            # External venues typically have lower execution probability
            execution_probability = 0.5
            
            # Calculate fees (external venues may have higher fees)
            estimated_fees = (order.quantity * estimated_price * venue.fees_bps) / Decimal('10000')
            
            # Total cost
            total_cost = (order.quantity * estimated_price) + estimated_fees
            
            return ExecutionEstimate(
                venue_id=venue.venue_id,
                venue_type=venue.venue_type,
                estimated_price=estimated_price,
                estimated_quantity=order.quantity,
                execution_probability=execution_probability,
                estimated_latency_ms=venue.latency_ms,
                estimated_fees=estimated_fees,
                total_cost=total_cost,
                confidence_score=0.6  # Lower confidence for external venues
            )
            
        except Exception as e:
            logger.error(f"Error estimating external venue execution: {e}")
            return None
    
    def _estimate_execution_probability(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Estimate execution probability based on market data."""
        try:
            # Base probability
            base_probability = 0.7
            
            # Adjust based on order size relative to market depth
            if 'best_bid_quantity' in market_data and 'best_ask_quantity' in market_data:
                bid_quantity = market_data['best_bid_quantity'] or Decimal('0')
                ask_quantity = market_data['best_ask_quantity'] or Decimal('0')
                
                if order.side == OrderSide.BUY:
                    # For buy orders, compare with ask quantity
                    if ask_quantity > 0:
                        size_ratio = min(1.0, float(order.quantity / ask_quantity))
                        base_probability *= (1.0 - size_ratio * 0.3)
                else:
                    # For sell orders, compare with bid quantity
                    if bid_quantity > 0:
                        size_ratio = min(1.0, float(order.quantity / bid_quantity))
                        base_probability *= (1.0 - size_ratio * 0.3)
            
            # Adjust based on spread
            if 'spread' in market_data and market_data['spread']:
                spread = market_data['spread']
                if spread > Decimal('0.01'):  # Wide spread
                    base_probability *= 0.8
                elif spread < Decimal('0.001'):  # Tight spread
                    base_probability *= 1.1
            
            # Ensure probability is within bounds
            return max(0.1, min(0.95, base_probability))
            
        except Exception as e:
            logger.error(f"Error estimating execution probability: {e}")
            return 0.5
    
    def _apply_routing_strategy(self, order: Order, strategy: RoutingStrategy, 
                               estimates: List[ExecutionEstimate]) -> RoutingDecision:
        """Apply routing strategy to select venues."""
        try:
            if not estimates:
                raise Exception("No execution estimates available")
            
            # Sort estimates based on strategy
            if strategy == RoutingStrategy.BEST_EXECUTION:
                # Best execution: minimize total cost
                sorted_estimates = sorted(estimates, key=lambda x: x.total_cost)
            elif strategy == RoutingStrategy.LOWEST_COST:
                # Lowest cost: minimize fees
                sorted_estimates = sorted(estimates, key=lambda x: x.estimated_fees)
            elif strategy == RoutingStrategy.FASTEST_FILL:
                # Fastest fill: minimize latency
                sorted_estimates = sorted(estimates, key=lambda x: x.estimated_latency_ms)
            elif strategy == RoutingStrategy.LIQUIDITY_SEEKING:
                # Liquidity seeking: maximize execution probability
                sorted_estimates = sorted(estimates, key=lambda x: x.execution_probability, reverse=True)
            else:  # SMART_ROUTING
                # Smart routing: weighted combination of factors
                sorted_estimates = self._smart_routing_sort(estimates)
            
            # Select primary venue
            primary_venue = sorted_estimates[0].venue_id
            
            # Select fallback venues
            fallback_venues = [est.venue_id for est in sorted_estimates[1:3]]
            
            # Determine if order splitting is needed
            split_orders = self._determine_order_splits(order, sorted_estimates)
            
            # Calculate total estimated cost
            total_cost = sum(split['estimated_cost'] for split in split_orders)
            
            # Calculate overall fill probability
            overall_probability = 1.0 - (1.0 - sorted_estimates[0].execution_probability)
            for est in sorted_estimates[1:3]:
                overall_probability = 1.0 - (1.0 - overall_probability) * (1.0 - est.execution_probability)
            
            return RoutingDecision(
                order_id=order.order_id,
                strategy=strategy,
                primary_venue=primary_venue,
                fallback_venues=fallback_venues,
                split_orders=split_orders,
                estimated_total_cost=total_cost,
                estimated_fill_probability=overall_probability
            )
            
        except Exception as e:
            logger.error(f"Error applying routing strategy: {e}")
            raise
    
    def _smart_routing_sort(self, estimates: List[ExecutionEstimate]) -> List[ExecutionEstimate]:
        """Smart routing: weighted combination of factors."""
        try:
            # Calculate composite score for each estimate
            for estimate in estimates:
                # Normalize factors to 0-1 scale
                cost_score = 1.0 / (1.0 + float(estimate.total_cost / Decimal('1000000')))  # Normalize to 1M
                latency_score = 1.0 / (1.0 + estimate.estimated_latency_ms / 100.0)  # Normalize to 100ms
                probability_score = estimate.execution_probability
                confidence_score = estimate.confidence_score
                
                # Weighted combination
                composite_score = (
                    cost_score * 0.3 +
                    latency_score * 0.2 +
                    probability_score * 0.3 +
                    confidence_score * 0.2
                )
                
                estimate.metadata = getattr(estimate, 'metadata', {})
                estimate.metadata['composite_score'] = composite_score
            
            # Sort by composite score
            return sorted(estimates, key=lambda x: x.metadata.get('composite_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in smart routing sort: {e}")
            # Fall back to cost-based sorting
            return sorted(estimates, key=lambda x: x.total_cost)
    
    def _determine_order_splits(self, order: Order, estimates: List[ExecutionEstimate]) -> List[Dict[str, Any]]:
        """Determine if and how to split the order."""
        splits = []
        
        # Check if order splitting is needed
        if order.quantity <= self.split_size_threshold or len(estimates) < 2:
            # No splitting needed
            splits.append({
                'venue_id': estimates[0].venue_id,
                'quantity': order.quantity,
                'estimated_cost': estimates[0].total_cost,
                'estimated_price': estimates[0].estimated_price
            })
        else:
            # Split order across venues
            remaining_quantity = order.quantity
            max_splits = min(self.max_order_splits, len(estimates))
            
            for i in range(max_splits):
                if remaining_quantity <= 0:
                    break
                
                estimate = estimates[i]
                split_quantity = min(remaining_quantity, order.quantity / max_splits)
                
                splits.append({
                    'venue_id': estimate.venue_id,
                    'quantity': split_quantity,
                    'estimated_cost': estimate.total_cost * (split_quantity / order.quantity),
                    'estimated_price': estimate.estimated_price
                })
                
                remaining_quantity -= split_quantity
        
        return splits
    
    async def execute_routing_decision(self, routing_decision: RoutingDecision, order: Order) -> List[str]:
        """
        Execute the routing decision by submitting orders to venues.
        
        Args:
            routing_decision: Routing decision to execute
            order: Original order
            
        Returns:
            List of order IDs submitted to venues
        """
        submitted_orders = []
        
        try:
            for split in routing_decision.split_orders:
                venue_id = split['venue_id']
                quantity = split['quantity']
                
                # Create split order
                split_order = Order(
                    order_id=str(uuid4()),
                    participant_id=order.participant_id,
                    bond_id=order.bond_id,
                    order_type=order.order_type,
                    side=order.side,
                    quantity=quantity,
                    price=order.price,
                    time_in_force=order.time_in_force,
                    status=OrderStatus.PENDING,
                    metadata={
                        "parent_order_id": order.order_id,
                        "routing_decision": routing_decision.order_id,
                        "venue_id": venue_id
                    }
                )
                
                # Submit to appropriate venue
                if venue_id == "INTERNAL_CONTINUOUS":
                    # Submit to internal matching engine
                    event_id = await self.matching_engine.submit_order(split_order)
                    if event_id:
                        submitted_orders.append(split_order.order_id)
                        logger.info(f"Split order {split_order.order_id} submitted to internal continuous")
                
                elif venue_id == "INTERNAL_AUCTION":
                    # Submit to internal auction engine
                    # This would require auction-specific logic
                    logger.info(f"Split order {split_order.order_id} would be submitted to internal auction")
                    submitted_orders.append(split_order.order_id)
                
                else:
                    # Submit to external venue
                    adapter = self.venue_adapters.get(venue_id)
                    if adapter and adapter.is_connected:
                        try:
                            external_order_id = await adapter.submit_order(split_order)
                            if external_order_id:
                                submitted_orders.append(split_order.order_id)
                                logger.info(f"Split order {split_order.order_id} submitted to external venue {venue_id}")
                        except Exception as e:
                            logger.error(f"Error submitting to external venue {venue_id}: {e}")
                    else:
                        logger.warning(f"External venue {venue_id} not available")
            
            return submitted_orders
            
        except Exception as e:
            logger.error(f"Error executing routing decision: {e}")
            raise
    
    def get_venue_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all venues."""
        health_status = {}
        
        for venue_id, venue in self.venues.items():
            # Check if venue is active
            is_healthy = venue.is_active
            
            # Check last health check
            time_since_check = (datetime.utcnow() - venue.last_health_check).total_seconds()
            if time_since_check > 300:  # 5 minutes
                is_healthy = False
            
            # Check external venue connection
            if venue.venue_type in [VenueType.EXTERNAL_PRIMARY, VenueType.EXTERNAL_SECONDARY, VenueType.EXTERNAL_DARK_POOL]:
                adapter = self.venue_adapters.get(venue_id)
                if adapter:
                    is_healthy = is_healthy and adapter.is_connected
            
            health_status[venue_id] = {
                "venue_id": venue_id,
                "venue_type": venue.venue_type.value,
                "name": venue.name,
                "is_healthy": is_healthy,
                "is_active": venue.is_active,
                "latency_ms": venue.latency_ms,
                "last_health_check": venue.last_health_check,
                "time_since_check_seconds": time_since_check
            }
        
        return health_status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "orders_routed": self.orders_routed,
            "routing_decisions": self.routing_decisions,
            "total_venues": len(self.venues),
            "external_venues": len([v for v in self.venues.values() if v.venue_type not in [VenueType.INTERNAL_CONTINUOUS, VenueType.INTERNAL_AUCTION]]),
            "uptime_seconds": uptime,
            "orders_per_second": self.orders_routed / uptime if uptime > 0 else 0,
            "start_time": self.start_time,
            "default_strategy": self.default_strategy.value
        }


# Export classes
__all__ = ["OrderRouter", "VenueType", "RoutingStrategy", "VenueInfo", "ExecutionEstimate", 
           "RoutingDecision", "VenueAdapter", "MockExternalVenueAdapter"]
