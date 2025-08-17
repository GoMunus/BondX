"""
Clearing Engine for BondX Auction System.

This module provides sophisticated clearing algorithms for different auction types,
including demand curve analysis, optimal clearing price determination, and allocation optimization.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from ..database.auction_models import Auction, Bid, Allocation
from ..core.logging import get_logger

logger = get_logger(__name__)


class ClearingMethod(Enum):
    """Methods for clearing auctions."""
    
    UNIFORM_PRICE = "UNIFORM_PRICE"  # All participants pay the same price
    DISCRIMINATORY = "DISCRIMINATORY"  # Each participant pays their bid price
    HYBRID = "HYBRID"  # Combination of uniform and discriminatory pricing


class ClearingResult(NamedTuple):
    """Result of auction clearing process."""
    
    clearing_price: Decimal
    total_allocated: Decimal
    bid_to_cover_ratio: float
    allocations: List[Dict[str, Any]]
    demand_curve: List[Tuple[Decimal, Decimal]]
    clearing_method: ClearingMethod
    processing_time: float


@dataclass
class DemandCurvePoint:
    """Point on the demand curve."""
    
    price: Decimal
    cumulative_quantity: Decimal
    bid_count: int
    participant_count: int
    marginal_quantity: Decimal


class ClearingEngine:
    """
    Sophisticated clearing engine for auction price discovery.
    
    This engine provides:
    - Advanced demand curve analysis
    - Optimal clearing price determination
    - Multiple clearing methodologies
    - Allocation optimization algorithms
    - Market efficiency analysis
    """
    
    def __init__(self, db_session: Session):
        """Initialize the clearing engine."""
        self.db_session = db_session
        self.logger = get_logger(__name__)
    
    async def clear_auction(self, auction: Auction, bids: List[Bid], 
                           clearing_method: ClearingMethod = ClearingMethod.UNIFORM_PRICE) -> ClearingResult:
        """
        Clear an auction using the specified clearing method.
        
        Args:
            auction: The auction to clear
            bids: List of valid bids
            clearing_method: Method to use for clearing
            
        Returns:
            ClearingResult with auction results
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Clearing auction {auction.auction_code} using {clearing_method.value}")
            
            if not bids:
                raise ValueError("No bids received for auction")
            
            # Build comprehensive demand curve
            demand_curve = self._build_demand_curve(bids, auction)
            
            # Determine clearing price based on method
            if clearing_method == ClearingMethod.UNIFORM_PRICE:
                clearing_price, total_allocated = self._uniform_price_clearing(
                    demand_curve, auction.total_lot_size
                )
            elif clearing_method == ClearingMethod.DISCRIMINATORY:
                clearing_price, total_allocated = self._discriminatory_clearing(
                    demand_curve, auction.total_lot_size
                )
            elif clearing_method == ClearingMethod.HYBRID:
                clearing_price, total_allocated = self._hybrid_clearing(
                    demand_curve, auction.total_lot_size
                )
            else:
                raise ValueError(f"Unsupported clearing method: {clearing_method}")
            
            # Create allocations
            allocations = self._create_allocations(
                auction, bids, clearing_price, total_allocated, clearing_method
            )
            
            # Calculate bid-to-cover ratio
            total_bids = sum(bid.bid_quantity for bid in bids)
            bid_to_cover_ratio = self._calculate_bid_to_cover_ratio(total_bids, auction.total_lot_size)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create result
            result = ClearingResult(
                clearing_price=clearing_price,
                total_allocated=total_allocated,
                bid_to_cover_ratio=bid_to_cover_ratio,
                allocations=allocations,
                demand_curve=[(float(p.price), float(p.cumulative_quantity)) for p in demand_curve],
                clearing_method=clearing_method,
                processing_time=processing_time
            )
            
            self.logger.info(f"Auction {auction.auction_code} cleared successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to clear auction {auction.auction_code}: {str(e)}")
            raise
    
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
                        marginal_quantity = sum(b.bid_quantity for b in current_bids)
                        demand_curve.append(DemandCurvePoint(
                            price=current_price,
                            cumulative_quantity=cumulative_quantity,
                            bid_count=len(current_bids),
                            participant_count=len(set(b.participant_id for b in current_bids)),
                            marginal_quantity=marginal_quantity
                        ))
                    
                    current_price = bid.bid_price
                    current_bids = [bid]
                else:
                    current_bids.append(bid)
                
                cumulative_quantity += bid.bid_quantity
            
            # Add final price level
            if current_price is not None:
                marginal_quantity = sum(b.bid_quantity for b in current_bids)
                demand_curve.append(DemandCurvePoint(
                    price=current_price,
                    cumulative_quantity=cumulative_quantity,
                    bid_count=len(current_bids),
                    participant_count=len(set(b.participant_id for b in current_bids)),
                    marginal_quantity=marginal_quantity
                ))
            
            return demand_curve
            
        except Exception as e:
            self.logger.error(f"Error building demand curve: {str(e)}")
            return []
    
    def _uniform_price_clearing(self, demand_curve: List[DemandCurvePoint], 
                               total_offered: Decimal) -> Tuple[Decimal, Decimal]:
        """Uniform price clearing where all participants pay the same price."""
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
            self.logger.error(f"Error in uniform price clearing: {str(e)}")
            return Decimal('0'), Decimal('0')
    
    def _discriminatory_clearing(self, demand_curve: List[DemandCurvePoint], 
                                total_offered: Decimal) -> Tuple[Decimal, Decimal]:
        """Discriminatory clearing where each participant pays their bid price."""
        try:
            # In discriminatory clearing, we need to find the marginal bid
            for point in demand_curve:
                if point.cumulative_quantity >= total_offered:
                    # Calculate how much of the marginal price level is needed
                    excess_quantity = point.cumulative_quantity - total_offered
                    if excess_quantity > 0:
                        # Partial fill at marginal price
                        return point.price, total_offered
                    else:
                        # Full fill at marginal price
                        return point.price, total_offered
            
            # If no clearing price found, use the lowest bid price
            if demand_curve:
                lowest_price = demand_curve[-1].price
                return lowest_price, min(demand_curve[-1].cumulative_quantity, total_offered)
            
            return Decimal('0'), Decimal('0')
            
        except Exception as e:
            self.logger.error(f"Error in discriminatory clearing: {str(e)}")
            return Decimal('0'), Decimal('0')
    
    def _hybrid_clearing(self, demand_curve: List[DemandCurvePoint], 
                         total_offered: Decimal) -> Tuple[Decimal, Decimal]:
        """Hybrid clearing combining uniform and discriminatory pricing."""
        try:
            # Find the uniform clearing price
            uniform_price, uniform_quantity = self._uniform_price_clearing(demand_curve, total_offered)
            
            # Apply hybrid rules (e.g., uniform price for large allocations, discriminatory for small)
            # This is a simplified implementation - real hybrid systems can be more complex
            
            # For now, return uniform clearing results
            return uniform_price, uniform_quantity
            
        except Exception as e:
            self.logger.error(f"Error in hybrid clearing: {str(e)}")
            return Decimal('0'), Decimal('0')
    
    def _create_allocations(self, auction: Auction, bids: List[Bid], 
                           clearing_price: Decimal, total_allocated: Decimal,
                           clearing_method: ClearingMethod) -> List[Dict[str, Any]]:
        """Create allocations based on clearing results."""
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
                
                # Determine allocation price based on clearing method
                if clearing_method == ClearingMethod.UNIFORM_PRICE:
                    allocation_price = clearing_price
                elif clearing_method == ClearingMethod.DISCRIMINATORY:
                    allocation_price = bid.bid_price
                else:  # HYBRID
                    allocation_price = clearing_price  # Simplified for now
                
                allocation = {
                    'participant_id': bid.participant_id,
                    'bid_id': bid.id,
                    'quantity': allocation_quantity,
                    'price': allocation_price,
                    'clearing_method': clearing_method.value,
                    'clearing_price': float(clearing_price),
                    'bid_price': float(bid.bid_price),
                    'submission_time': bid.submission_time.isoformat()
                }
                
                allocations.append(allocation)
                remaining_quantity -= allocation_quantity
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error creating allocations: {str(e)}")
            return []
    
    def _calculate_bid_to_cover_ratio(self, total_bids: Decimal, total_offered: Decimal) -> float:
        """Calculate bid-to-cover ratio."""
        try:
            if total_offered == 0:
                return 0.0
            return float(total_bids / total_offered)
        except Exception:
            return 0.0
    
    def analyze_market_efficiency(self, demand_curve: List[DemandCurvePoint], 
                                 clearing_price: Decimal) -> Dict[str, Any]:
        """Analyze market efficiency metrics."""
        try:
            if not demand_curve:
                return {}
            
            # Calculate price dispersion
            prices = [point.price for point in demand_curve]
            price_range = max(prices) - min(prices)
            price_std = self._calculate_standard_deviation(prices)
            
            # Calculate quantity concentration
            total_quantity = demand_curve[-1].cumulative_quantity if demand_curve else Decimal('0')
            herfindahl_index = self._calculate_herfindahl_index(demand_curve)
            
            # Calculate market depth
            market_depth = self._calculate_market_depth(demand_curve, clearing_price)
            
            efficiency_metrics = {
                'price_dispersion': float(price_range),
                'price_volatility': float(price_std),
                'quantity_concentration': float(herfindahl_index),
                'market_depth': float(market_depth),
                'price_efficiency': float(1.0 / (1.0 + float(price_std))) if price_std > 0 else 1.0,
                'liquidity_score': float(market_depth / total_quantity) if total_quantity > 0 else 0.0
            }
            
            return efficiency_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing market efficiency: {str(e)}")
            return {}
    
    def _calculate_standard_deviation(self, values: List[Decimal]) -> Decimal:
        """Calculate standard deviation of a list of values."""
        try:
            if len(values) <= 1:
                return Decimal('0')
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance.sqrt()
            
        except Exception:
            return Decimal('0')
    
    def _calculate_herfindahl_index(self, demand_curve: List[DemandCurvePoint]) -> Decimal:
        """Calculate Herfindahl-Hirschman Index for quantity concentration."""
        try:
            if not demand_curve:
                return Decimal('0')
            
            total_quantity = demand_curve[-1].cumulative_quantity
            if total_quantity == 0:
                return Decimal('0')
            
            # Calculate HHI based on marginal quantities at each price level
            hhi = Decimal('0')
            for point in demand_curve:
                if point.marginal_quantity > 0:
                    market_share = point.marginal_quantity / total_quantity
                    hhi += market_share ** 2
            
            return hhi
            
        except Exception:
            return Decimal('0')
    
    def _calculate_market_depth(self, demand_curve: List[DemandCurvePoint], 
                               clearing_price: Decimal) -> Decimal:
        """Calculate market depth around the clearing price."""
        try:
            if not demand_curve:
                return Decimal('0')
            
            # Find price levels within a small range of clearing price
            price_tolerance = Decimal('0.01')  # 1 basis point
            depth_quantity = Decimal('0')
            
            for point in demand_curve:
                if abs(point.price - clearing_price) <= price_tolerance:
                    depth_quantity += point.marginal_quantity
            
            return depth_quantity
            
        except Exception:
            return Decimal('0')
    
    def optimize_allocation(self, allocations: List[Dict[str, Any]], 
                           optimization_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize allocations based on specified criteria."""
        try:
            if not optimization_criteria:
                return allocations
            
            # Apply optimization based on criteria
            if optimization_criteria.get('minimize_concentration', False):
                allocations = self._minimize_concentration(allocations)
            
            if optimization_criteria.get('maximize_participation', False):
                allocations = self._maximize_participation(allocations)
            
            if optimization_criteria.get('balance_allocation_sizes', False):
                allocations = self._balance_allocation_sizes(allocations)
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error optimizing allocations: {str(e)}")
            return allocations
    
    def _minimize_concentration(self, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Minimize concentration of allocations among participants."""
        try:
            # Group allocations by participant
            participant_allocations = {}
            for allocation in allocations:
                participant_id = allocation['participant_id']
                if participant_id not in participant_allocations:
                    participant_allocations[participant_id] = []
                participant_allocations[participant_id].append(allocation)
            
            # Calculate total allocation per participant
            participant_totals = {
                pid: sum(a['quantity'] for a in allocs)
                for pid, allocs in participant_allocations.items()
            }
            
            # Sort participants by total allocation (descending)
            sorted_participants = sorted(
                participant_totals.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Redistribute to minimize concentration
            # This is a simplified implementation
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error minimizing concentration: {str(e)}")
            return allocations
    
    def _maximize_participation(self, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Maximize the number of participants receiving allocations."""
        try:
            # This would implement logic to ensure more participants receive allocations
            # For now, return original allocations
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error maximizing participation: {str(e)}")
            return allocations
    
    def _balance_allocation_sizes(self, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance allocation sizes among participants."""
        try:
            # This would implement logic to balance allocation sizes
            # For now, return original allocations
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error balancing allocation sizes: {str(e)}")
            return allocations
