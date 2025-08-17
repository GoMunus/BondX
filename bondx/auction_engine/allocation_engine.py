"""
Allocation Engine for BondX Auction System.

This module provides sophisticated allocation algorithms for auction results,
including pro-rata, priority-based, and hybrid allocation methods.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from ..database.auction_models import Auction, Bid, Allocation, Participant
from ..core.logging import get_logger

logger = get_logger(__name__)


class AllocationMethod(Enum):
    """Methods for allocating securities in auctions."""
    
    PRO_RATA = "PRO_RATA"  # Proportional allocation
    PRIORITY_BASED = "PRIORITY_BASED"  # Priority-based allocation
    TIME_PRIORITY = "TIME_PRIORITY"  # Time priority allocation
    SIZE_PRIORITY = "SIZE_PRIORITY"  # Size priority allocation
    HYBRID = "HYBRID"  # Hybrid allocation method


@dataclass
class AllocationRequest:
    """Request for allocation processing."""
    
    participant_id: int
    bid_id: int
    requested_quantity: Decimal
    bid_price: Decimal
    bid_time: datetime
    priority_score: Optional[float] = None
    participant_type: Optional[str] = None


@dataclass
class AllocationResult:
    """Result of allocation processing."""
    
    participant_id: int
    bid_id: int
    allocated_quantity: Decimal
    allocation_price: Decimal
    allocation_method: str
    allocation_priority: Optional[int] = None
    allocation_round: Optional[int] = None


class AllocationEngine:
    """
    Sophisticated allocation engine for auction results.
    
    This engine provides:
    - Multiple allocation methodologies
    - Priority-based allocation algorithms
    - Pro-rata distribution with rounding optimization
    - Hybrid allocation strategies
    - Comprehensive allocation validation
    """
    
    def __init__(self, db_session: Session):
        """Initialize the allocation engine."""
        self.db_session = db_session
        self.logger = get_logger(__name__)
    
    async def create_allocations(self, auction: Auction, 
                                allocation_data: List[Dict[str, Any]]) -> List[Allocation]:
        """
        Create allocation records for auction results.
        
        Args:
            auction: The auction being allocated
            allocation_data: List of allocation data from clearing
            
        Returns:
            List of created allocation records
        """
        try:
            self.logger.info(f"Creating allocations for auction {auction.auction_code}")
            
            allocations = []
            
            for data in allocation_data:
                # Create allocation record
                allocation = Allocation(
                    auction_id=auction.id,
                    participant_id=data['participant_id'],
                    bid_id=data.get('bid_id'),
                    allocation_quantity=data['quantity'],
                    allocation_price=data['price'],
                    allocation_method=data.get('allocation_method', 'CLEARING'),
                    allocation_priority=data.get('allocation_priority'),
                    allocation_round=data.get('allocation_round'),
                    is_confirmed=False
                )
                
                self.db_session.add(allocation)
                allocations.append(allocation)
            
            self.db_session.commit()
            
            self.logger.info(f"Created {len(allocations)} allocation records for auction {auction.auction_code}")
            return allocations
            
        except Exception as e:
            self.logger.error(f"Failed to create allocations: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def process_allocation(self, auction: Auction, 
                                allocation_method: AllocationMethod = AllocationMethod.PRO_RATA) -> List[AllocationResult]:
        """
        Process allocation using the specified method.
        
        Args:
            auction: The auction to allocate
            allocation_method: Method to use for allocation
            
        Returns:
            List of allocation results
        """
        try:
            self.logger.info(f"Processing allocation for auction {auction.auction_code} using {allocation_method.value}")
            
            # Get all valid bids for the auction
            bids = self.db_session.query(Bid).filter(
                and_(
                    Bid.auction_id == auction.id,
                    Bid.status.in_([BidStatus.PENDING, BidStatus.ACCEPTED])
                )
            ).all()
            
            if not bids:
                raise ValueError("No valid bids found for auction")
            
            # Convert bids to allocation requests
            allocation_requests = []
            for bid in bids:
                request = AllocationRequest(
                    participant_id=bid.participant_id,
                    bid_id=bid.id,
                    requested_quantity=bid.bid_quantity,
                    bid_price=bid.bid_price,
                    bid_time=bid.submission_time,
                    participant_type=self._get_participant_type(bid.participant_id)
                )
                allocation_requests.append(request)
            
            # Process allocation based on method
            if allocation_method == AllocationMethod.PRO_RATA:
                results = self._pro_rata_allocation(auction, allocation_requests)
            elif allocation_method == AllocationMethod.PRIORITY_BASED:
                results = self._priority_based_allocation(auction, allocation_requests)
            elif allocation_method == AllocationMethod.TIME_PRIORITY:
                results = self._time_priority_allocation(auction, allocation_requests)
            elif allocation_method == AllocationMethod.SIZE_PRIORITY:
                results = self._size_priority_allocation(auction, allocation_requests)
            elif allocation_method == AllocationMethod.HYBRID:
                results = self._hybrid_allocation(auction, allocation_requests)
            else:
                raise ValueError(f"Unsupported allocation method: {allocation_method}")
            
            # Validate allocation results
            self._validate_allocation_results(auction, results)
            
            self.logger.info(f"Allocation processed successfully: {len(results)} allocations")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process allocation: {str(e)}")
            raise
    
    def _pro_rata_allocation(self, auction: Auction, 
                            requests: List[AllocationRequest]) -> List[AllocationResult]:
        """Pro-rata allocation based on bid quantities."""
        try:
            total_offered = auction.total_lot_size
            total_requested = sum(req.requested_quantity for req in requests)
            
            if total_requested <= 0:
                return []
            
            # Calculate allocation ratio
            allocation_ratio = total_offered / total_requested
            
            results = []
            for request in requests:
                # Calculate pro-rata allocation
                allocated_quantity = (request.requested_quantity * allocation_ratio).quantize(
                    Decimal('0.01'), rounding=ROUND_HALF_UP
                )
                
                # Ensure allocation doesn't exceed requested quantity
                allocated_quantity = min(allocated_quantity, request.requested_quantity)
                
                result = AllocationResult(
                    participant_id=request.participant_id,
                    bid_id=request.bid_id,
                    allocated_quantity=allocated_quantity,
                    allocation_price=auction.clearing_price or request.bid_price,
                    allocation_method=AllocationMethod.PRO_RATA.value
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pro-rata allocation: {str(e)}")
            return []
    
    def _priority_based_allocation(self, auction: Auction, 
                                  requests: List[AllocationRequest]) -> List[AllocationResult]:
        """Priority-based allocation using participant priorities."""
        try:
            # Sort requests by priority (higher priority first)
            priority_order = auction.priority_rules.get('order', []) if auction.priority_rules else []
            
            if priority_order:
                # Assign priority scores based on participant type
                for request in requests:
                    if request.participant_type in priority_order:
                        request.priority_score = priority_order.index(request.participant_type)
                    else:
                        request.priority_score = len(priority_order)  # Lowest priority
                
                # Sort by priority score (lower is higher priority)
                requests.sort(key=lambda x: x.priority_score)
            
            # Apply priority allocation
            remaining_quantity = auction.total_lot_size
            results = []
            
            for request in requests:
                if remaining_quantity <= 0:
                    # No more quantity to allocate
                    result = AllocationResult(
                        participant_id=request.participant_id,
                        bid_id=request.bid_id,
                        allocated_quantity=Decimal('0'),
                        allocation_price=auction.clearing_price or request.bid_price,
                        allocation_method=AllocationMethod.PRIORITY_BASED.value,
                        allocation_priority=request.priority_score
                    )
                    results.append(result)
                    continue
                
                # Allocate requested quantity or remaining quantity, whichever is smaller
                allocated_quantity = min(request.requested_quantity, remaining_quantity)
                
                result = AllocationResult(
                    participant_id=request.participant_id,
                    bid_id=request.bid_id,
                    allocated_quantity=allocated_quantity,
                    allocation_price=auction.clearing_price or request.bid_price,
                    allocation_method=AllocationMethod.PRIORITY_BASED.value,
                    allocation_priority=request.priority_score
                )
                
                results.append(result)
                remaining_quantity -= allocated_quantity
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in priority-based allocation: {str(e)}")
            return []
    
    def _time_priority_allocation(self, auction: Auction, 
                                 requests: List[AllocationRequest]) -> List[AllocationResult]:
        """Time priority allocation (earlier bids get priority)."""
        try:
            # Sort requests by submission time (earlier first)
            requests.sort(key=lambda x: x.bid_time)
            
            # Apply time priority allocation
            remaining_quantity = auction.total_lot_size
            results = []
            
            for request in requests:
                if remaining_quantity <= 0:
                    # No more quantity to allocate
                    result = AllocationResult(
                        participant_id=request.participant_id,
                        bid_id=request.bid_id,
                        allocated_quantity=Decimal('0'),
                        allocation_price=auction.clearing_price or request.bid_price,
                        allocation_method=AllocationMethod.TIME_PRIORITY.value
                    )
                    results.append(result)
                    continue
                
                # Allocate requested quantity or remaining quantity, whichever is smaller
                allocated_quantity = min(request.requested_quantity, remaining_quantity)
                
                result = AllocationResult(
                    participant_id=request.participant_id,
                    bid_id=request.bid_id,
                    allocated_quantity=allocated_quantity,
                    allocation_price=auction.clearing_price or request.bid_price,
                    allocation_method=AllocationMethod.TIME_PRIORITY.value
                )
                
                results.append(result)
                remaining_quantity -= allocated_quantity
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in time priority allocation: {str(e)}")
            return []
    
    def _size_priority_allocation(self, auction: Auction, 
                                 requests: List[AllocationRequest]) -> List[AllocationResult]:
        """Size priority allocation (larger bids get priority)."""
        try:
            # Sort requests by quantity (larger first)
            requests.sort(key=lambda x: x.requested_quantity, reverse=True)
            
            # Apply size priority allocation
            remaining_quantity = auction.total_lot_size
            results = []
            
            for request in requests:
                if remaining_quantity <= 0:
                    # No more quantity to allocate
                    result = AllocationResult(
                        participant_id=request.participant_id,
                        bid_id=request.bid_id,
                        allocated_quantity=Decimal('0'),
                        allocation_price=auction.clearing_price or request.bid_price,
                        allocation_method=AllocationMethod.SIZE_PRIORITY.value
                    )
                    results.append(result)
                    continue
                
                # Allocate requested quantity or remaining quantity, whichever is smaller
                allocated_quantity = min(request.requested_quantity, remaining_quantity)
                
                result = AllocationResult(
                    participant_id=request.participant_id,
                    bid_id=request.bid_id,
                    allocated_quantity=allocated_quantity,
                    allocation_price=auction.clearing_price or request.bid_price,
                    allocation_method=AllocationMethod.SIZE_PRIORITY.value
                )
                
                results.append(result)
                remaining_quantity -= allocated_quantity
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in size priority allocation: {str(e)}")
            return []
    
    def _hybrid_allocation(self, auction: Auction, 
                          requests: List[AllocationRequest]) -> List[AllocationResult]:
        """Hybrid allocation combining multiple methods."""
        try:
            # This is a simplified hybrid implementation
            # In practice, this could be much more sophisticated
            
            # First pass: Priority-based allocation for high-priority participants
            priority_requests = [req for req in requests if req.participant_type in ['PRIMARY_DEALER', 'INSTITUTIONAL']]
            priority_results = self._priority_based_allocation(auction, priority_requests)
            
            # Calculate remaining quantity after priority allocation
            allocated_quantity = sum(result.allocated_quantity for result in priority_results)
            remaining_quantity = auction.total_lot_size - allocated_quantity
            
            # Second pass: Pro-rata allocation for remaining participants
            remaining_requests = [req for req in requests if req.participant_type not in ['PRIMARY_DEALER', 'INSTITUTIONAL']]
            
            if remaining_requests and remaining_quantity > 0:
                # Create temporary auction with remaining quantity
                temp_auction = type('TempAuction', (), {
                    'total_lot_size': remaining_quantity,
                    'clearing_price': auction.clearing_price
                })()
                
                pro_rata_results = self._pro_rata_allocation(temp_auction, remaining_requests)
                
                # Combine results
                all_results = priority_results + pro_rata_results
            else:
                all_results = priority_results
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid allocation: {str(e)}")
            return []
    
    def _get_participant_type(self, participant_id: int) -> Optional[str]:
        """Get participant type for priority calculations."""
        try:
            participant = self.db_session.query(Participant).filter(
                Participant.id == participant_id
            ).first()
            
            if participant:
                return participant.participant_type.value
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting participant type: {str(e)}")
            return None
    
    def _validate_allocation_results(self, auction: Auction, results: List[AllocationResult]):
        """Validate allocation results for consistency."""
        try:
            if not results:
                return
            
            # Check total allocated quantity
            total_allocated = sum(result.allocated_quantity for result in results)
            if total_allocated > auction.total_lot_size:
                raise ValueError(f"Total allocated quantity ({total_allocated}) exceeds auction lot size ({auction.total_lot_size})")
            
            # Check for negative allocations
            for result in results:
                if result.allocated_quantity < 0:
                    raise ValueError(f"Negative allocation quantity: {result.allocated_quantity}")
            
            # Check for duplicate participant allocations
            participant_allocations = {}
            for result in results:
                if result.participant_id in participant_allocations:
                    participant_allocations[result.participant_id] += result.allocated_quantity
                else:
                    participant_allocations[result.participant_id] = result.allocated_quantity
            
            # Validate against maximum allocation per participant
            if auction.maximum_allocation_per_participant:
                for participant_id, total_allocated in participant_allocations.items():
                    if total_allocated > auction.maximum_allocation_per_participant:
                        raise ValueError(f"Participant {participant_id} allocation ({total_allocated}) exceeds maximum ({auction.maximum_allocation_per_participant})")
            
            self.logger.info("Allocation validation passed successfully")
            
        except Exception as e:
            self.logger.error(f"Allocation validation failed: {str(e)}")
            raise
    
    async def optimize_allocation(self, results: List[AllocationResult], 
                                 optimization_criteria: Dict[str, Any]) -> List[AllocationResult]:
        """
        Optimize allocation results based on specified criteria.
        
        Args:
            results: List of allocation results to optimize
            optimization_criteria: Criteria for optimization
            
        Returns:
            Optimized allocation results
        """
        try:
            if not optimization_criteria:
                return results
            
            # Apply optimization based on criteria
            if optimization_criteria.get('minimize_concentration', False):
                results = self._minimize_concentration(results)
            
            if optimization_criteria.get('maximize_participants', False):
                results = self._maximize_participants(results)
            
            if optimization_criteria.get('balance_allocation_sizes', False):
                results = self._balance_allocation_sizes(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error optimizing allocation: {str(e)}")
            return results
    
    def _minimize_concentration(self, results: List[AllocationResult]) -> List[AllocationResult]:
        """Minimize concentration of allocations among participants."""
        try:
            # Group allocations by participant
            participant_allocations = {}
            for result in results:
                if result.participant_id not in participant_allocations:
                    participant_allocations[result.participant_id] = []
                participant_allocations[result.participant_id].append(result)
            
            # Calculate total allocation per participant
            participant_totals = {
                pid: sum(r.allocated_quantity for r in allocs)
                for pid, allocs in participant_allocations.items()
            }
            
            # Sort participants by total allocation (descending)
            sorted_participants = sorted(
                participant_totals.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # This is a simplified implementation
            # In practice, this would implement more sophisticated redistribution logic
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error minimizing concentration: {str(e)}")
            return results
    
    def _maximize_participants(self, results: List[AllocationResult]) -> List[AllocationResult]:
        """Maximize the number of participants receiving allocations."""
        try:
            # This would implement logic to ensure more participants receive allocations
            # For now, return original results
            return results
            
        except Exception as e:
            self.logger.error(f"Error maximizing participants: {str(e)}")
            return results
    
    def _balance_allocation_sizes(self, results: List[AllocationResult]) -> List[AllocationResult]:
        """Balance allocation sizes among participants."""
        try:
            # This would implement logic to balance allocation sizes
            # For now, return original results
            return results
            
        except Exception as e:
            self.logger.error(f"Error balancing allocation sizes: {str(e)}")
            return results
    
    def get_allocation_statistics(self, results: List[AllocationResult]) -> Dict[str, Any]:
        """Get comprehensive statistics about allocation results."""
        try:
            if not results:
                return {}
            
            # Basic statistics
            total_allocations = len(results)
            total_quantity = sum(result.allocated_quantity for result in results)
            total_value = sum(result.allocated_quantity * result.allocation_price for result in results)
            
            # Participant statistics
            unique_participants = len(set(result.participant_id for result in results))
            
            # Method statistics
            method_counts = {}
            for result in results:
                method = result.allocation_method
                if method not in method_counts:
                    method_counts[method] = 0
                method_counts[method] += 1
            
            # Quantity distribution
            quantities = [float(result.allocated_quantity) for result in results]
            if quantities:
                min_quantity = min(quantities)
                max_quantity = max(quantities)
                avg_quantity = sum(quantities) / len(quantities)
            else:
                min_quantity = max_quantity = avg_quantity = 0.0
            
            statistics = {
                "total_allocations": total_allocations,
                "total_quantity": float(total_quantity),
                "total_value": float(total_value),
                "unique_participants": unique_participants,
                "allocation_methods": method_counts,
                "quantity_statistics": {
                    "minimum": min_quantity,
                    "maximum": max_quantity,
                    "average": avg_quantity
                },
                "participation_rate": unique_participants / total_allocations if total_allocations > 0 else 0.0
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error getting allocation statistics: {str(e)}")
            return {}
