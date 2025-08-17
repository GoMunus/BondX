"""
Sophisticated Auction Mechanisms for BondX.

This module implements various auction formats including Dutch, English, Sealed Bid,
Multi-Round, and Hybrid auctions with advanced clearing algorithms and allocation procedures.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from ..database.auction_models import (
    Auction, Bid, Allocation, AuctionType, AuctionStatus, BidStatus
)
from ..core.logging import get_logger

logger = get_logger(__name__)


class ClearingResult(NamedTuple):
    """Result of auction clearing process."""
    
    clearing_price: Decimal
    total_allocated: Decimal
    bid_to_cover_ratio: float
    allocations: List[Dict[str, Any]]
    demand_curve: List[Tuple[Decimal, Decimal]]
    clearing_rounds: int


class AllocationMethod(Enum):
    """Methods for allocating securities in auctions."""
    
    PRO_RATA = "PRO_RATA"  # Proportional allocation
    PRIORITY_BASED = "PRIORITY_BASED"  # Priority-based allocation
    TIME_PRIORITY = "TIME_PRIORITY"  # Time priority allocation
    SIZE_PRIORITY = "SIZE_PRIORITY"  # Size priority allocation
    HYBRID = "HYBRID"  # Hybrid allocation method


@dataclass
class DemandCurvePoint:
    """Point on the demand curve."""
    
    price: Decimal
    cumulative_quantity: Decimal
    bid_count: int
    participant_count: int


class BaseAuctionMechanism(ABC):
    """Base class for all auction mechanisms."""
    
    def __init__(self, db_session: Session):
        """Initialize the auction mechanism."""
        self.db_session = db_session
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def initialize_auction(self, auction: Auction) -> bool:
        """Initialize auction-specific parameters."""
        pass
    
    @abstractmethod
    async def start_bidding(self, auction: Auction) -> bool:
        """Start the bidding process."""
        pass
    
    @abstractmethod
    async def stop_bidding(self, auction: Auction) -> bool:
        """Stop the bidding process."""
        pass
    
    @abstractmethod
    async def process_auction(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process the auction and determine results."""
        pass
    
    def _build_demand_curve(self, bids: List[Bid], auction: Auction) -> List[DemandCurvePoint]:
        """Build comprehensive demand curve from bids."""
        try:
            # Sort bids by price (descending for most auction types)
            sorted_bids = sorted(bids, key=lambda x: x.bid_price, reverse=True)
            
            demand_curve = []
            cumulative_quantity = Decimal('0')
            current_price = None
            current_bids = []
            
            for bid in sorted_bids:
                if current_price is None or bid.bid_price != current_price:
                    # Process previous price level
                    if current_price is not None:
                        demand_curve.append(DemandCurvePoint(
                            price=current_price,
                            cumulative_quantity=cumulative_quantity,
                            bid_count=len(current_bids),
                            participant_count=len(set(b.participant_id for b in current_bids))
                        ))
                    
                    current_price = bid.bid_price
                    current_bids = [bid]
                else:
                    current_bids.append(bid)
                
                cumulative_quantity += bid.bid_quantity
            
            # Add final price level
            if current_price is not None:
                demand_curve.append(DemandCurvePoint(
                    price=current_price,
                    cumulative_quantity=cumulative_quantity,
                    bid_count=len(current_bids),
                    participant_count=len(set(b.participant_id for b in current_bids))
                ))
            
            return demand_curve
            
        except Exception as e:
            self.logger.error(f"Error building demand curve: {str(e)}")
            return []
    
    def _calculate_bid_to_cover_ratio(self, total_bids: Decimal, total_offered: Decimal) -> float:
        """Calculate bid-to-cover ratio."""
        try:
            if total_offered == 0:
                return 0.0
            return float(total_bids / total_offered)
        except Exception:
            return 0.0
    
    def _apply_allocation_rules(self, auction: Auction, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply auction-specific allocation rules."""
        try:
            if auction.allocation_method == AllocationMethod.PRO_RATA.value:
                return self._apply_pro_rata_allocation(auction, allocations)
            elif auction.allocation_method == AllocationMethod.PRIORITY_BASED.value:
                return self._apply_priority_allocation(auction, allocations)
            elif auction.allocation_method == AllocationMethod.TIME_PRIORITY.value:
                return self._apply_time_priority_allocation(auction, allocations)
            elif auction.allocation_method == AllocationMethod.SIZE_PRIORITY.value:
                return self._apply_size_priority_allocation(auction, allocations)
            else:
                return allocations
                
        except Exception as e:
            self.logger.error(f"Error applying allocation rules: {str(e)}")
            return allocations
    
    def _apply_pro_rata_allocation(self, auction: Auction, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply pro-rata allocation rules."""
        try:
            total_offered = auction.total_lot_size
            total_demand = sum(a['quantity'] for a in allocations)
            
            if total_demand <= total_offered:
                return allocations
            
            # Calculate allocation ratio
            allocation_ratio = total_offered / total_demand
            
            # Apply pro-rata allocation
            for allocation in allocations:
                original_quantity = allocation['quantity']
                allocation['quantity'] = (original_quantity * allocation_ratio).quantize(
                    Decimal('0.01'), rounding=ROUND_HALF_UP
                )
                allocation['allocation_method'] = 'PRO_RATA'
                allocation['allocation_ratio'] = float(allocation_ratio)
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in pro-rata allocation: {str(e)}")
            return allocations
    
    def _apply_priority_allocation(self, auction: Auction, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply priority-based allocation rules."""
        try:
            if not auction.priority_rules:
                return allocations
            
            # Sort by priority (higher priority first)
            priority_order = auction.priority_rules.get('order', [])
            if priority_order:
                allocations.sort(key=lambda x: priority_order.index(x.get('participant_type', 'OTHER')))
            
            # Apply priority allocation
            remaining_quantity = auction.total_lot_size
            for allocation in allocations:
                if remaining_quantity <= 0:
                    allocation['quantity'] = Decimal('0')
                    continue
                
                requested_quantity = allocation['quantity']
                allocated_quantity = min(requested_quantity, remaining_quantity)
                
                allocation['quantity'] = allocated_quantity
                allocation['allocation_method'] = 'PRIORITY'
                allocation['priority_level'] = priority_order.index(allocation.get('participant_type', 'OTHER'))
                
                remaining_quantity -= allocated_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in priority allocation: {str(e)}")
            return allocations
    
    def _apply_time_priority_allocation(self, auction: Auction, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply time priority allocation rules."""
        try:
            # Sort by submission time (earlier bids get priority)
            allocations.sort(key=lambda x: x.get('submission_time', datetime.max))
            
            # Apply time priority allocation
            remaining_quantity = auction.total_lot_size
            for allocation in allocations:
                if remaining_quantity <= 0:
                    allocation['quantity'] = Decimal('0')
                    continue
                
                requested_quantity = allocation['quantity']
                allocated_quantity = min(requested_quantity, remaining_quantity)
                
                allocation['quantity'] = allocated_quantity
                allocation['allocation_method'] = 'TIME_PRIORITY'
                allocation['time_priority'] = allocation.get('submission_time', datetime.max).isoformat()
                
                remaining_quantity -= allocated_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in time priority allocation: {str(e)}")
            return allocations
    
    def _apply_size_priority_allocation(self, auction: Auction, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply size priority allocation rules."""
        try:
            # Sort by quantity (larger bids get priority)
            allocations.sort(key=lambda x: x['quantity'], reverse=True)
            
            # Apply size priority allocation
            remaining_quantity = auction.total_lot_size
            for allocation in allocations:
                if remaining_quantity <= 0:
                    allocation['quantity'] = Decimal('0')
                    continue
                
                requested_quantity = allocation['quantity']
                allocated_quantity = min(requested_quantity, remaining_quantity)
                
                allocation['quantity'] = allocated_quantity
                allocation['allocation_method'] = 'SIZE_PRIORITY'
                allocation['size_priority'] = float(requested_quantity)
                
                remaining_quantity -= allocated_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in size priority allocation: {str(e)}")
            return allocations


class DutchAuction(BaseAuctionMechanism):
    """
    Dutch Auction implementation with sophisticated clearing algorithms.
    
    Dutch auctions start with a high price that decreases until demand meets supply.
    This implementation includes comprehensive demand curve analysis and optimal clearing.
    """
    
    async def initialize_auction(self, auction: Auction) -> bool:
        """Initialize Dutch auction parameters."""
        try:
            # Set default parameters for Dutch auction
            if not auction.maximum_price:
                # Set maximum price based on instrument characteristics
                auction.maximum_price = Decimal('100.00')  # 100% of face value
            
            if not auction.price_increment:
                auction.price_increment = Decimal('0.01')  # 1 basis point
            
            self.logger.info(f"Dutch auction {auction.auction_code} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Dutch auction: {str(e)}")
            return False
    
    async def start_bidding(self, auction: Auction) -> bool:
        """Start Dutch auction bidding process."""
        try:
            # Dutch auctions typically don't have traditional bidding periods
            # They run continuously with price discovery
            self.logger.info(f"Dutch auction {auction.auction_code} bidding started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Dutch auction bidding: {str(e)}")
            return False
    
    async def stop_bidding(self, auction: Auction) -> bool:
        """Stop Dutch auction bidding process."""
        try:
            self.logger.info(f"Dutch auction {auction.auction_code} bidding stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Dutch auction bidding: {str(e)}")
            return False
    
    async def process_auction(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process Dutch auction with sophisticated clearing algorithm."""
        try:
            self.logger.info(f"Processing Dutch auction {auction.auction_code}")
            
            if not bids:
                raise ValueError("No bids received for auction")
            
            # Build comprehensive demand curve
            demand_curve = self._build_demand_curve(bids, auction)
            
            # Find clearing price where demand meets supply
            clearing_price, total_allocated = self._find_clearing_price(
                demand_curve, auction.total_lot_size
            )
            
            # Create allocations based on clearing price
            allocations = self._create_dutch_allocations(
                auction, bids, clearing_price, total_allocated
            )
            
            # Apply allocation rules
            allocations = self._apply_allocation_rules(auction, allocations)
            
            # Calculate bid-to-cover ratio
            total_bids = sum(bid.bid_quantity for bid in bids)
            bid_to_cover_ratio = self._calculate_bid_to_cover_ratio(total_bids, auction.total_lot_size)
            
            # Create clearing result
            result = ClearingResult(
                clearing_price=clearing_price,
                total_allocated=total_allocated,
                bid_to_cover_ratio=bid_to_cover_ratio,
                allocations=allocations,
                demand_curve=[(float(p.price), float(p.cumulative_quantity)) for p in demand_curve],
                clearing_rounds=1
            )
            
            self.logger.info(f"Dutch auction {auction.auction_code} processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process Dutch auction: {str(e)}")
            raise
    
    def _find_clearing_price(self, demand_curve: List[DemandCurvePoint], total_offered: Decimal) -> Tuple[Decimal, Decimal]:
        """Find the clearing price where demand meets supply."""
        try:
            for point in demand_curve:
                if point.cumulative_quantity >= total_offered:
                    return point.price, total_offered
            
            # If no clearing price found, use the lowest bid price
            if demand_curve:
                lowest_price = demand_curve[-1].price
                return lowest_price, min(demand_curve[-1].cumulative_quantity, total_offered)
            
            return Decimal('0'), Decimal('0')
            
        except Exception as e:
            self.logger.error(f"Error finding clearing price: {str(e)}")
            return Decimal('0'), Decimal('0')
    
    def _create_dutch_allocations(self, auction: Auction, bids: List[Bid], 
                                 clearing_price: Decimal, total_allocated: Decimal) -> List[Dict[str, Any]]:
        """Create allocations for Dutch auction."""
        try:
            allocations = []
            
            # Filter bids at or above clearing price
            eligible_bids = [bid for bid in bids if bid.bid_price >= clearing_price]
            
            # Sort by price (descending) and then by submission time
            eligible_bids.sort(key=lambda x: (x.bid_price, x.submission_time))
            
            remaining_quantity = total_allocated
            
            for bid in eligible_bids:
                if remaining_quantity <= 0:
                    break
                
                # Calculate allocation for this bid
                allocation_quantity = min(bid.bid_quantity, remaining_quantity)
                
                allocation = {
                    'participant_id': bid.participant_id,
                    'bid_id': bid.id,
                    'quantity': allocation_quantity,
                    'price': clearing_price,
                    'allocation_method': 'DUTCH_CLEARING',
                    'clearing_price': float(clearing_price),
                    'submission_time': bid.submission_time
                }
                
                allocations.append(allocation)
                remaining_quantity -= allocation_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error creating Dutch allocations: {str(e)}")
            return []


class EnglishAuction(BaseAuctionMechanism):
    """
    English Auction implementation with ascending price mechanism.
    
    English auctions start with a low price that increases until only one bidder remains.
    This implementation includes multi-round bidding and price improvement mechanisms.
    """
    
    async def initialize_auction(self, auction: Auction) -> bool:
        """Initialize English auction parameters."""
        try:
            # Set default parameters for English auction
            if not auction.minimum_price:
                auction.minimum_price = Decimal('0.01')  # 1 basis point
            
            if not auction.price_increment:
                auction.price_increment = Decimal('0.01')  # 1 basis point
            
            self.logger.info(f"English auction {auction.auction_code} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize English auction: {str(e)}")
            return False
    
    async def start_bidding(self, auction: Auction) -> bool:
        """Start English auction bidding process."""
        try:
            self.logger.info(f"English auction {auction.auction_code} bidding started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start English auction bidding: {str(e)}")
            return False
    
    async def stop_bidding(self, auction: Auction) -> bool:
        """Stop English auction bidding process."""
        try:
            self.logger.info(f"English auction {auction.auction_code} bidding stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop English auction bidding: {str(e)}")
            return False
    
    async def process_auction(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process English auction with multi-round bidding simulation."""
        try:
            self.logger.info(f"Processing English auction {auction.auction_code}")
            
            if not bids:
                raise ValueError("No bids received for auction")
            
            # Simulate multi-round bidding process
            clearing_price, total_allocated, clearing_rounds = self._simulate_english_bidding(
                auction, bids
            )
            
            # Create allocations
            allocations = self._create_english_allocations(
                auction, bids, clearing_price, total_allocated
            )
            
            # Apply allocation rules
            allocations = self._apply_allocation_rules(auction, allocations)
            
            # Calculate bid-to-cover ratio
            total_bids = sum(bid.bid_quantity for bid in bids)
            bid_to_cover_ratio = self._calculate_bid_to_cover_ratio(total_bids, auction.total_lot_size)
            
            # Build demand curve
            demand_curve = self._build_demand_curve(bids, auction)
            
            # Create clearing result
            result = ClearingResult(
                clearing_price=clearing_price,
                total_allocated=total_allocated,
                bid_to_cover_ratio=bid_to_cover_ratio,
                allocations=allocations,
                demand_curve=[(float(p.price), float(p.cumulative_quantity)) for p in demand_curve],
                clearing_rounds=clearing_rounds
            )
            
            self.logger.info(f"English auction {auction.auction_code} processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process English auction: {str(e)}")
            raise
    
    def _simulate_english_bidding(self, auction: Auction, bids: List[Bid]) -> Tuple[Decimal, Decimal, int]:
        """Simulate multi-round English auction bidding."""
        try:
            current_price = auction.minimum_price or Decimal('0.01')
            price_increment = auction.price_increment or Decimal('0.01')
            remaining_quantity = auction.total_lot_size
            round_count = 0
            
            while remaining_quantity > 0 and current_price <= (auction.maximum_price or Decimal('1000.00')):
                round_count += 1
                
                # Count bids at current price level
                active_bids = [bid for bid in bids if bid.bid_price >= current_price]
                
                if not active_bids:
                    break
                
                # Calculate total demand at current price
                total_demand = sum(bid.bid_quantity for bid in active_bids)
                
                if total_demand <= remaining_quantity:
                    # All demand can be satisfied at this price
                    return current_price, total_demand, round_count
                
                # Move to next price level
                current_price += price_increment
            
            # Return best possible allocation
            best_bids = sorted(bids, key=lambda x: x.bid_price, reverse=True)
            total_allocated = Decimal('0')
            
            for bid in best_bids:
                if total_allocated >= auction.total_lot_size:
                    break
                total_allocated += min(bid.bid_quantity, auction.total_lot_size - total_allocated)
            
            return best_bids[0].bid_price if best_bids else Decimal('0'), total_allocated, round_count
            
        except Exception as e:
            self.logger.error(f"Error simulating English bidding: {str(e)}")
            return Decimal('0'), Decimal('0'), 0
    
    def _create_english_allocations(self, auction: Auction, bids: List[Bid], 
                                   clearing_price: Decimal, total_allocated: Decimal) -> List[Dict[str, Any]]:
        """Create allocations for English auction."""
        try:
            allocations = []
            
            # Filter bids at clearing price
            eligible_bids = [bid for bid in bids if bid.bid_price >= clearing_price]
            
            # Sort by price (descending) and then by submission time
            eligible_bids.sort(key=lambda x: (x.bid_price, x.submission_time))
            
            remaining_quantity = total_allocated
            
            for bid in eligible_bids:
                if remaining_quantity <= 0:
                    break
                
                # Calculate allocation for this bid
                allocation_quantity = min(bid.bid_quantity, remaining_quantity)
                
                allocation = {
                    'participant_id': bid.participant_id,
                    'bid_id': bid.id,
                    'quantity': allocation_quantity,
                    'price': clearing_price,
                    'allocation_method': 'ENGLISH_CLEARING',
                    'clearing_price': float(clearing_price),
                    'submission_time': bid.submission_time
                }
                
                allocations.append(allocation)
                remaining_quantity -= allocation_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error creating English allocations: {str(e)}")
            return []


class SealedBidAuction(BaseAuctionMechanism):
    """
    Sealed Bid Auction implementation with anti-manipulation features.
    
    Sealed bid auctions collect all bids simultaneously and reveal them at once.
    This implementation includes sophisticated bid validation and fairness monitoring.
    """
    
    async def initialize_auction(self, auction: Auction) -> bool:
        """Initialize Sealed Bid auction parameters."""
        try:
            # Set default parameters for Sealed Bid auction
            if not auction.minimum_price:
                auction.minimum_price = Decimal('0.01')
            
            if not auction.maximum_price:
                auction.maximum_price = Decimal('1000.00')
            
            self.logger.info(f"Sealed Bid auction {auction.auction_code} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sealed Bid auction: {str(e)}")
            return False
    
    async def start_bidding(self, auction: Auction) -> bool:
        """Start Sealed Bid auction bidding process."""
        try:
            self.logger.info(f"Sealed Bid auction {auction.auction_code} bidding started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Sealed Bid auction bidding: {str(e)}")
            return False
    
    async def stop_bidding(self, auction: Auction) -> bool:
        """Stop Sealed Bid auction bidding process."""
        try:
            self.logger.info(f"Sealed Bid auction {auction.auction_code} bidding stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Sealed Bid auction bidding: {str(e)}")
            return False
    
    async def process_auction(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process Sealed Bid auction with comprehensive analysis."""
        try:
            self.logger.info(f"Processing Sealed Bid auction {auction.auction_code}")
            
            if not bids:
                raise ValueError("No bids received for auction")
            
            # Perform anti-manipulation analysis
            self._analyze_bid_patterns(bids)
            
            # Build demand curve
            demand_curve = self._build_demand_curve(bids, auction)
            
            # Find clearing price
            clearing_price, total_allocated = self._find_sealed_clearing_price(
                demand_curve, auction.total_lot_size
            )
            
            # Create allocations
            allocations = self._create_sealed_allocations(
                auction, bids, clearing_price, total_allocated
            )
            
            # Apply allocation rules
            allocations = self._apply_allocation_rules(auction, allocations)
            
            # Calculate bid-to-cover ratio
            total_bids = sum(bid.bid_quantity for bid in bids)
            bid_to_cover_ratio = self._calculate_bid_to_cover_ratio(total_bids, auction.total_lot_size)
            
            # Create clearing result
            result = ClearingResult(
                clearing_price=clearing_price,
                total_allocated=total_allocated,
                bid_to_cover_ratio=bid_to_cover_ratio,
                allocations=allocations,
                demand_curve=[(float(p.price), float(p.cumulative_quantity)) for p in demand_curve],
                clearing_rounds=1
            )
            
            self.logger.info(f"Sealed Bid auction {auction.auction_code} processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process Sealed Bid auction: {str(e)}")
            raise
    
    def _analyze_bid_patterns(self, bids: List[Bid]) -> None:
        """Analyze bid patterns for potential manipulation."""
        try:
            # Check for unusual bid clustering
            prices = [bid.bid_price for bid in bids]
            if len(prices) > 1:
                price_std = (sum((p - sum(prices)/len(prices))**2 for p in prices) / len(prices)) ** 0.5
                if price_std < Decimal('0.001'):  # Very low price dispersion
                    self.logger.warning("Low price dispersion detected - potential manipulation")
            
            # Check for bid timing patterns
            submission_times = [bid.submission_time for bid in bids]
            if len(submission_times) > 1:
                time_diffs = [(submission_times[i+1] - submission_times[i]).total_seconds() 
                             for i in range(len(submission_times)-1)]
                if all(diff < 1.0 for diff in time_diffs):  # All bids within 1 second
                    self.logger.warning("Suspicious bid timing pattern detected")
            
            # Check for quantity patterns
            quantities = [bid.bid_quantity for bid in bids]
            if len(quantities) > 1:
                quantity_std = (sum((q - sum(quantities)/len(quantities))**2 for q in quantities) / len(quantities)) ** 0.5
                if quantity_std < Decimal('0.01'):  # Very low quantity dispersion
                    self.logger.warning("Low quantity dispersion detected - potential manipulation")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing bid patterns: {str(e)}")
    
    def _find_sealed_clearing_price(self, demand_curve: List[DemandCurvePoint], 
                                   total_offered: Decimal) -> Tuple[Decimal, Decimal]:
        """Find clearing price for sealed bid auction."""
        try:
            for point in demand_curve:
                if point.cumulative_quantity >= total_offered:
                    return point.price, total_offered
            
            # If no clearing price found, use the lowest bid price
            if demand_curve:
                lowest_price = demand_curve[-1].price
                return lowest_price, min(demand_curve[-1].cumulative_quantity, total_offered)
            
            return Decimal('0'), Decimal('0')
            
        except Exception as e:
            self.logger.error(f"Error finding sealed clearing price: {str(e)}")
            return Decimal('0'), Decimal('0')
    
    def _create_sealed_allocations(self, auction: Auction, bids: List[Bid], 
                                  clearing_price: Decimal, total_allocated: Decimal) -> List[Dict[str, Any]]:
        """Create allocations for sealed bid auction."""
        try:
            allocations = []
            
            # Filter bids at or above clearing price
            eligible_bids = [bid for bid in bids if bid.bid_price >= clearing_price]
            
            # Sort by price (descending) and then by submission time
            eligible_bids.sort(key=lambda x: (x.bid_price, x.submission_time))
            
            remaining_quantity = total_allocated
            
            for bid in eligible_bids:
                if remaining_quantity <= 0:
                    break
                
                # Calculate allocation for this bid
                allocation_quantity = min(bid.bid_quantity, remaining_quantity)
                
                allocation = {
                    'participant_id': bid.participant_id,
                    'bid_id': bid.id,
                    'quantity': allocation_quantity,
                    'price': clearing_price,
                    'allocation_method': 'SEALED_CLEARING',
                    'clearing_price': float(clearing_price),
                    'submission_time': bid.submission_time
                }
                
                allocations.append(allocation)
                remaining_quantity -= allocation_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error creating sealed allocations: {str(e)}")
            return []


class MultiRoundAuction(BaseAuctionMechanism):
    """
    Multi-Round Auction implementation with price improvement mechanisms.
    
    Multi-round auctions allow participants to improve their bids across multiple rounds
    to achieve better price discovery and liquidity formation.
    """
    
    async def initialize_auction(self, auction: Auction) -> bool:
        """Initialize Multi-Round auction parameters."""
        try:
            # Set default parameters for Multi-Round auction
            if not auction.minimum_price:
                auction.minimum_price = Decimal('0.01')
            
            if not auction.price_increment:
                auction.price_increment = Decimal('0.01')
            
            self.logger.info(f"Multi-Round auction {auction.auction_code} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Multi-Round auction: {str(e)}")
            return False
    
    async def start_bidding(self, auction: Auction) -> bool:
        """Start Multi-Round auction bidding process."""
        try:
            self.logger.info(f"Multi-Round auction {auction.auction_code} bidding started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Multi-Round auction bidding: {str(e)}")
            return False
    
    async def stop_bidding(self, auction: Auction) -> bool:
        """Stop Multi-Round auction bidding process."""
        try:
            self.logger.info(f"Multi-Round auction {auction.auction_code} bidding stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Multi-Round auction bidding: {str(e)}")
            return False
    
    async def process_auction(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process Multi-Round auction with sophisticated round management."""
        try:
            self.logger.info(f"Processing Multi-Round auction {auction.auction_code}")
            
            if not bids:
                raise ValueError("No bids received for auction")
            
            # Simulate multi-round bidding process
            clearing_price, total_allocated, clearing_rounds = self._simulate_multi_round_bidding(
                auction, bids
            )
            
            # Create allocations
            allocations = self._create_multi_round_allocations(
                auction, bids, clearing_price, total_allocated
            )
            
            # Apply allocation rules
            allocations = self._apply_allocation_rules(auction, allocations)
            
            # Calculate bid-to-cover ratio
            total_bids = sum(bid.bid_quantity for bid in bids)
            bid_to_cover_ratio = self._calculate_bid_to_cover_ratio(total_bids, auction.total_lot_size)
            
            # Build demand curve
            demand_curve = self._build_demand_curve(bids, auction)
            
            # Create clearing result
            result = ClearingResult(
                clearing_price=clearing_price,
                total_allocated=total_allocated,
                bid_to_cover_ratio=bid_to_cover_ratio,
                allocations=allocations,
                demand_curve=[(float(p.price), float(p.cumulative_quantity)) for p in demand_curve],
                clearing_rounds=clearing_rounds
            )
            
            self.logger.info(f"Multi-Round auction {auction.auction_code} processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process Multi-Round auction: {str(e)}")
            raise
    
    def _simulate_multi_round_bidding(self, auction: Auction, bids: List[Bid]) -> Tuple[Decimal, Decimal, int]:
        """Simulate multi-round bidding process."""
        try:
            current_price = auction.minimum_price or Decimal('0.01')
            price_increment = auction.price_increment or Decimal('0.01')
            remaining_quantity = auction.total_lot_size
            round_count = 0
            max_rounds = 10  # Maximum number of rounds
            
            while remaining_quantity > 0 and round_count < max_rounds:
                round_count += 1
                
                # Count bids at current price level
                active_bids = [bid for bid in bids if bid.bid_price >= current_price]
                
                if not active_bids:
                    break
                
                # Calculate total demand at current price
                total_demand = sum(bid.bid_quantity for bid in active_bids)
                
                if total_demand <= remaining_quantity:
                    # All demand can be satisfied at this price
                    return current_price, total_demand, round_count
                
                # Check if we should continue to next round
                if round_count < max_rounds:
                    # Simulate price improvement in next round
                    current_price += price_increment
                else:
                    # Use best available allocation
                    best_bids = sorted(active_bids, key=lambda x: x.bid_price, reverse=True)
                    total_allocated = Decimal('0')
                    
                    for bid in best_bids:
                        if total_allocated >= remaining_quantity:
                            break
                        total_allocated += min(bid.bid_quantity, remaining_quantity - total_allocated)
                    
                    return best_bids[0].bid_price, total_allocated, round_count
            
            return current_price, remaining_quantity, round_count
            
        except Exception as e:
            self.logger.error(f"Error simulating multi-round bidding: {str(e)}")
            return Decimal('0'), Decimal('0'), 0
    
    def _create_multi_round_allocations(self, auction: Auction, bids: List[Bid], 
                                       clearing_price: Decimal, total_allocated: Decimal) -> List[Dict[str, Any]]:
        """Create allocations for multi-round auction."""
        try:
            allocations = []
            
            # Filter bids at clearing price
            eligible_bids = [bid for bid in bids if bid.bid_price >= clearing_price]
            
            # Sort by price (descending) and then by submission time
            eligible_bids.sort(key=lambda x: (x.bid_price, x.submission_time))
            
            remaining_quantity = total_allocated
            
            for bid in eligible_bids:
                if remaining_quantity <= 0:
                    break
                
                # Calculate allocation for this bid
                allocation_quantity = min(bid.bid_quantity, remaining_quantity)
                
                allocation = {
                    'participant_id': bid.participant_id,
                    'bid_id': bid.id,
                    'quantity': allocation_quantity,
                    'price': clearing_price,
                    'allocation_method': 'MULTI_ROUND_CLEARING',
                    'clearing_price': float(clearing_price),
                    'submission_time': bid.submission_time
                }
                
                allocations.append(allocation)
                remaining_quantity -= allocation_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error creating multi-round allocations: {str(e)}")
            return []


class HybridAuction(BaseAuctionMechanism):
    """
    Hybrid Auction implementation combining multiple auction mechanisms.
    
    Hybrid auctions can switch between different auction types based on market conditions
    and participant preferences to optimize price discovery and liquidity.
    """
    
    async def initialize_auction(self, auction: Auction) -> bool:
        """Initialize Hybrid auction parameters."""
        try:
            # Set default parameters for Hybrid auction
            if not auction.minimum_price:
                auction.minimum_price = Decimal('0.01')
            
            if not auction.maximum_price:
                auction.maximum_price = Decimal('1000.00')
            
            if not auction.price_increment:
                auction.price_increment = Decimal('0.01')
            
            self.logger.info(f"Hybrid auction {auction.auction_code} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Hybrid auction: {str(e)}")
            return False
    
    async def start_bidding(self, auction: Auction) -> bool:
        """Start Hybrid auction bidding process."""
        try:
            self.logger.info(f"Hybrid auction {auction.auction_code} bidding started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Hybrid auction bidding: {str(e)}")
            return False
    
    async def stop_bidding(self, auction: Auction) -> bool:
        """Stop Hybrid auction bidding process."""
        try:
            self.logger.info(f"Hybrid auction {auction.auction_code} bidding stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Hybrid auction bidding: {str(e)}")
            return False
    
    async def process_auction(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process Hybrid auction with dynamic mechanism selection."""
        try:
            self.logger.info(f"Processing Hybrid auction {auction.auction_code}")
            
            if not bids:
                raise ValueError("No bids received for auction")
            
            # Determine optimal auction mechanism based on market conditions
            optimal_mechanism = self._select_optimal_mechanism(auction, bids)
            
            # Process using selected mechanism
            if optimal_mechanism == "DUTCH":
                return await self._process_as_dutch(auction, bids)
            elif optimal_mechanism == "ENGLISH":
                return await self._process_as_english(auction, bids)
            elif optimal_mechanism == "SEALED":
                return await self._process_as_sealed(auction, bids)
            elif optimal_mechanism == "MULTI_ROUND":
                return await self._process_as_multi_round(auction, bids)
            else:
                # Default to sealed bid
                return await self._process_as_sealed(auction, bids)
                
        except Exception as e:
            self.logger.error(f"Failed to process Hybrid auction: {str(e)}")
            raise
    
    def _select_optimal_mechanism(self, auction: Auction, bids: List[Bid]) -> str:
        """Select optimal auction mechanism based on market conditions."""
        try:
            # Analyze market conditions
            bid_count = len(bids)
            total_quantity = sum(bid.bid_quantity for bid in bids)
            price_range = max(bid.bid_price for bid in bids) - min(bid.bid_price for bid in bids)
            
            # Decision logic based on market characteristics
            if bid_count < 5:
                # Low participation - use sealed bid
                return "SEALED"
            elif price_range < Decimal('0.05'):
                # Low price dispersion - use English auction
                return "ENGLISH"
            elif total_quantity > auction.total_lot_size * Decimal('2'):
                # High oversubscription - use Dutch auction
                return "DUTCH"
            elif bid_count > 20:
                # High participation - use multi-round
                return "MULTI_ROUND"
            else:
                # Default to sealed bid
                return "SEALED"
                
        except Exception as e:
            self.logger.error(f"Error selecting optimal mechanism: {str(e)}")
            return "SEALED"
    
    async def _process_as_dutch(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process hybrid auction as Dutch auction."""
        dutch_auction = DutchAuction(self.db_session)
        return await dutch_auction.process_auction(auction, bids)
    
    async def _process_as_english(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process hybrid auction as English auction."""
        english_auction = EnglishAuction(self.db_session)
        return await english_auction.process_auction(auction, bids)
    
    async def _process_as_sealed(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process hybrid auction as Sealed Bid auction."""
        sealed_auction = SealedBidAuction(self.db_session)
        return await sealed_auction.process_auction(auction, bids)
    
    async def _process_as_multi_round(self, auction: Auction, bids: List[Bid]) -> ClearingResult:
        """Process hybrid auction as Multi-Round auction."""
        multi_round_auction = MultiRoundAuction(self.db_session)
        return await multi_round_auction.process_auction(auction, bids)
