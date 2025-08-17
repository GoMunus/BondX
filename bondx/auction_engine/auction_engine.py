"""
Main Auction Engine for BondX.

This module provides the core auction engine that orchestrates auction operations,
manages different auction types, and ensures regulatory compliance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..database.auction_models import (
    Auction, Bid, Allocation, Participant, AuctionStatus, BidStatus,
    AuctionType, SettlementStatus
)
from .auction_mechanisms import (
    DutchAuction, EnglishAuction, SealedBidAuction, 
    MultiRoundAuction, HybridAuction
)
from .clearing_engine import ClearingEngine
from .allocation_engine import AllocationEngine
from .risk_engine import RiskEngine
from .settlement_engine import SettlementEngine
from ..core.monitoring import MetricsCollector
from ..core.logging import get_logger

logger = get_logger(__name__)


class AuctionEngineState(Enum):
    """Auction engine operational states."""
    
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    MAINTENANCE = "MAINTENANCE"
    ERROR = "ERROR"


@dataclass
class AuctionResult:
    """Result of auction processing."""
    
    auction_id: int
    clearing_price: Decimal
    total_allocated: Decimal
    bid_to_cover_ratio: float
    allocations: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class AuctionEngine:
    """
    Main auction engine that orchestrates all auction operations.
    
    This engine provides:
    - Multi-format auction support (Dutch, English, Sealed Bid, Multi-Round, Hybrid)
    - Sophisticated clearing algorithms
    - Risk management and compliance checks
    - Real-time auction monitoring
    - Integration with settlement systems
    """
    
    def __init__(self, db_session: Session):
        """Initialize the auction engine."""
        self.db_session = db_session
        self.state = AuctionEngineState.IDLE
        self.metrics = MetricsCollector()
        
        # Initialize sub-engines
        self.clearing_engine = ClearingEngine(db_session)
        self.allocation_engine = AllocationEngine(db_session)
        self.risk_engine = RiskEngine(db_session)
        self.settlement_engine = SettlementEngine(db_session)
        
        # Auction mechanism instances
        self.auction_mechanisms = {
            AuctionType.DUTCH: DutchAuction(db_session),
            AuctionType.ENGLISH: EnglishAuction(db_session),
            AuctionType.SEALED_BID: SealedBidAuction(db_session),
            AuctionType.MULTI_ROUND: MultiRoundAuction(db_session),
            AuctionType.HYBRID: HybridAuction(db_session)
        }
        
        # Active auctions tracking
        self.active_auctions: Dict[int, Dict[str, Any]] = {}
        
        logger.info("Auction Engine initialized successfully")
    
    async def create_auction(self, auction_data: Dict[str, Any]) -> Auction:
        """
        Create a new auction with comprehensive validation.
        
        Args:
            auction_data: Auction configuration data
            
        Returns:
            Created auction instance
            
        Raises:
            ValueError: If auction data is invalid
            RuntimeError: If auction creation fails
        """
        try:
            logger.info(f"Creating auction: {auction_data.get('auction_name', 'Unknown')}")
            
            # Validate auction data
            self._validate_auction_data(auction_data)
            
            # Create auction instance
            auction = Auction(**auction_data)
            
            # Perform risk and compliance checks
            risk_check = await self.risk_engine.validate_auction(auction)
            if not risk_check.passed:
                raise ValueError(f"Risk check failed: {risk_check.reason}")
            
            # Save to database
            self.db_session.add(auction)
            self.db_session.commit()
            
            # Initialize auction mechanism
            auction_type = auction.auction_type
            if auction_type in self.auction_mechanisms:
                await self.auction_mechanisms[auction_type].initialize_auction(auction)
            
            # Track active auction
            self.active_auctions[auction.id] = {
                'auction': auction,
                'start_time': datetime.utcnow(),
                'status': auction.status
            }
            
            # Record metrics
            self.metrics.record_auction_created(auction)
            
            logger.info(f"Auction {auction.auction_code} created successfully")
            return auction
            
        except Exception as e:
            logger.error(f"Failed to create auction: {str(e)}")
            self.db_session.rollback()
            raise RuntimeError(f"Auction creation failed: {str(e)}")
    
    async def start_auction(self, auction_id: int) -> bool:
        """
        Start an auction and open bidding.
        
        Args:
            auction_id: ID of the auction to start
            
        Returns:
            True if auction started successfully
            
        Raises:
            ValueError: If auction cannot be started
        """
        try:
            auction = self.db_session.query(Auction).filter(Auction.id == auction_id).first()
            if not auction:
                raise ValueError(f"Auction {auction_id} not found")
            
            if auction.status != AuctionStatus.ANNOUNCED:
                raise ValueError(f"Auction {auction_id} is not in ANNOUNCED state")
            
            # Validate auction readiness
            readiness_check = await self._check_auction_readiness(auction)
            if not readiness_check['ready']:
                raise ValueError(f"Auction not ready: {readiness_check['reason']}")
            
            # Update auction status
            auction.status = AuctionStatus.BIDDING_OPEN
            auction.bidding_start_time = datetime.utcnow()
            
            # Start auction mechanism
            auction_type = auction.auction_type
            if auction_type in self.auction_mechanisms:
                await self.auction_mechanisms[auction_type].start_bidding(auction)
            
            # Update tracking
            if auction_id in self.active_auctions:
                self.active_auctions[auction_id]['status'] = AuctionStatus.BIDDING_OPEN
            
            self.db_session.commit()
            
            # Record metrics
            self.metrics.record_auction_started(auction)
            
            logger.info(f"Auction {auction.auction_code} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start auction {auction_id}: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def submit_bid(self, bid_data: Dict[str, Any]) -> Bid:
        """
        Submit a bid for an auction with comprehensive validation.
        
        Args:
            bid_data: Bid submission data
            
        Returns:
            Created bid instance
            
        Raises:
            ValueError: If bid is invalid
            RuntimeError: If bid submission fails
        """
        try:
            logger.info(f"Processing bid submission: {bid_data.get('bid_id', 'Unknown')}")
            
            # Validate bid data
            self._validate_bid_data(bid_data)
            
            # Check auction status
            auction = self.db_session.query(Auction).filter(
                Auction.id == bid_data['auction_id']
            ).first()
            
            if not auction or auction.status != AuctionStatus.BIDDING_OPEN:
                raise ValueError("Auction is not accepting bids")
            
            # Perform risk checks
            risk_check = await self.risk_engine.validate_bid(bid_data)
            if not risk_check.passed:
                raise ValueError(f"Risk check failed: {risk_check.reason}")
            
            # Create bid instance
            bid = Bid(**bid_data)
            bid.submission_time = datetime.utcnow()
            bid.status = BidStatus.PENDING
            
            # Save bid
            self.db_session.add(bid)
            self.db_session.commit()
            
            # Record metrics
            self.metrics.record_bid_submitted(bid)
            
            logger.info(f"Bid {bid.bid_id} submitted successfully")
            return bid
            
        except Exception as e:
            logger.error(f"Failed to submit bid: {str(e)}")
            self.db_session.rollback()
            raise RuntimeError(f"Bid submission failed: {str(e)}")
    
    async def close_auction(self, auction_id: int) -> bool:
        """
        Close an auction and stop accepting bids.
        
        Args:
            auction_id: ID of the auction to close
            
        Returns:
            True if auction closed successfully
        """
        try:
            auction = self.db_session.query(Auction).filter(Auction.id == auction_id).first()
            if not auction:
                raise ValueError(f"Auction {auction_id} not found")
            
            if auction.status != AuctionStatus.BIDDING_OPEN:
                raise ValueError(f"Auction {auction_id} is not in BIDDING_OPEN state")
            
            # Close bidding
            auction.status = AuctionStatus.BIDDING_CLOSED
            auction.bidding_end_time = datetime.utcnow()
            
            # Stop auction mechanism
            auction_type = auction.auction_type
            if auction_type in self.auction_mechanisms:
                await self.auction_mechanisms[auction_type].stop_bidding(auction)
            
            # Update tracking
            if auction_id in self.active_auctions:
                self.active_auctions[auction_id]['status'] = AuctionStatus.BIDDING_CLOSED
            
            self.db_session.commit()
            
            # Record metrics
            self.metrics.record_auction_closed(auction)
            
            logger.info(f"Auction {auction.auction_code} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close auction {auction_id}: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def process_auction(self, auction_id: int) -> AuctionResult:
        """
        Process a closed auction and determine results.
        
        Args:
            auction_id: ID of the auction to process
            
        Returns:
            AuctionResult with processing results
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing auction {auction_id}")
            
            # Update auction status
            auction = self.db_session.query(Auction).filter(Auction.id == auction_id).first()
            if not auction:
                raise ValueError(f"Auction {auction_id} not found")
            
            auction.status = AuctionStatus.PROCESSING
            
            # Get all bids for the auction
            bids = self.db_session.query(Bid).filter(
                and_(
                    Bid.auction_id == auction_id,
                    Bid.status.in_([BidStatus.PENDING, BidStatus.ACCEPTED])
                )
            ).all()
            
            if not bids:
                raise ValueError(f"No valid bids found for auction {auction_id}")
            
            # Process auction using appropriate mechanism
            auction_type = auction.auction_type
            if auction_type not in self.auction_mechanisms:
                raise ValueError(f"Unsupported auction type: {auction_type}")
            
            mechanism = self.auction_mechanisms[auction_type]
            clearing_result = await mechanism.process_auction(auction, bids)
            
            # Update auction with results
            auction.clearing_price = clearing_result.clearing_price
            auction.total_bids_received = len(bids)
            auction.total_lots_allocated = clearing_result.total_allocated
            auction.bid_to_cover_ratio = clearing_result.bid_to_cover_ratio
            
            # Create allocations
            allocations = await self.allocation_engine.create_allocations(
                auction, clearing_result.allocations
            )
            
            # Update bid statuses
            for bid in bids:
                allocation = next(
                    (a for a in allocations if a.bid_id == bid.id), None
                )
                if allocation:
                    if allocation.allocation_quantity == bid.bid_quantity:
                        bid.status = BidStatus.FILLED
                    else:
                        bid.status = BidStatus.PARTIALLY_FILLED
                    bid.allocation_quantity = allocation.allocation_quantity
                    bid.allocation_price = allocation.allocation_price
                else:
                    bid.status = BidStatus.REJECTED
            
            # Update auction status
            auction.status = AuctionStatus.SETTLED
            
            # Update tracking
            if auction_id in self.active_auctions:
                self.active_auctions[auction_id]['status'] = AuctionStatus.SETTLED
            
            self.db_session.commit()
            
            # Record metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.record_auction_processed(auction, processing_time)
            
            # Create result
            result = AuctionResult(
                auction_id=auction_id,
                clearing_price=clearing_result.clearing_price,
                total_allocated=clearing_result.total_allocated,
                bid_to_cover_ratio=clearing_result.bid_to_cover_ratio,
                allocations=[{
                    'participant_id': a.participant_id,
                    'quantity': float(a.allocation_quantity),
                    'price': float(a.allocation_price)
                } for a in allocations],
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"Auction {auction.auction_code} processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process auction {auction_id}: {str(e)}")
            
            # Update auction status to failed
            if auction:
                auction.status = AuctionStatus.FAILED
                self.db_session.commit()
            
            # Update tracking
            if auction_id in self.active_auctions:
                self.active_auctions[auction_id]['status'] = AuctionStatus.FAILED
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AuctionResult(
                auction_id=auction_id,
                clearing_price=Decimal('0'),
                total_allocated=Decimal('0'),
                bid_to_cover_ratio=0.0,
                allocations=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def get_auction_status(self, auction_id: int) -> Dict[str, Any]:
        """
        Get comprehensive status of an auction.
        
        Args:
            auction_id: ID of the auction
            
        Returns:
            Dictionary with auction status information
        """
        try:
            auction = self.db_session.query(Auction).filter(Auction.id == auction_id).first()
            if not auction:
                return {'error': 'Auction not found'}
            
            # Get bid statistics
            bid_stats = self.db_session.query(
                func.count(Bid.id).label('total_bids'),
                func.sum(Bid.bid_quantity).label('total_quantity'),
                func.avg(Bid.bid_price).label('average_price')
            ).filter(Bid.auction_id == auction_id).first()
            
            # Get allocation statistics
            allocation_stats = self.db_session.query(
                func.count(Allocation.id).label('total_allocations'),
                func.sum(Allocation.allocation_quantity).label('total_allocated'),
                func.avg(Allocation.allocation_price).label('average_allocation_price')
            ).filter(Allocation.auction_id == auction_id).first()
            
            status_info = {
                'auction_id': auction_id,
                'auction_code': auction.auction_code,
                'status': auction.status.value,
                'auction_type': auction.auction_type.value,
                'total_lot_size': float(auction.total_lot_size),
                'bidding_start_time': auction.bidding_start_time.isoformat() if auction.bidding_start_time else None,
                'bidding_end_time': auction.bidding_end_time.isoformat() if auction.bidding_end_time else None,
                'settlement_date': auction.settlement_date.isoformat() if auction.settlement_date else None,
                'bid_statistics': {
                    'total_bids': bid_stats.total_bids or 0,
                    'total_quantity': float(bid_stats.total_quantity) if bid_stats.total_quantity else 0.0,
                    'average_price': float(bid_stats.average_price) if bid_stats.average_price else 0.0
                },
                'allocation_statistics': {
                    'total_allocations': allocation_stats.total_allocations or 0,
                    'total_allocated': float(allocation_stats.total_allocated) if allocation_stats.total_allocated else 0.0,
                    'average_allocation_price': float(allocation_stats.average_allocation_price) if allocation_stats.average_allocation_price else 0.0
                },
                'results': {
                    'clearing_price': float(auction.clearing_price) if auction.clearing_price else None,
                    'bid_to_cover_ratio': auction.bid_to_cover_ratio,
                    'total_lots_allocated': float(auction.total_lots_allocated) if auction.total_lots_allocated else None
                }
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get auction status {auction_id}: {str(e)}")
            return {'error': f'Failed to get status: {str(e)}'}
    
    async def get_active_auctions(self) -> List[Dict[str, Any]]:
        """
        Get list of all active auctions.
        
        Returns:
            List of active auction information
        """
        try:
            active_auctions = self.db_session.query(Auction).filter(
                Auction.status.in_([
                    AuctionStatus.ANNOUNCED,
                    AuctionStatus.BIDDING_OPEN,
                    AuctionStatus.PROCESSING
                ])
            ).all()
            
            auction_list = []
            for auction in active_auctions:
                # Get bid count
                bid_count = self.db_session.query(func.count(Bid.id)).filter(
                    Bid.auction_id == auction.id
                ).scalar()
                
                auction_info = {
                    'auction_id': auction.id,
                    'auction_code': auction.auction_code,
                    'auction_name': auction.auction_name,
                    'status': auction.status.value,
                    'auction_type': auction.auction_type.value,
                    'total_lot_size': float(auction.total_lot_size),
                    'bidding_start_time': auction.bidding_start_time.isoformat() if auction.bidding_start_time else None,
                    'bidding_end_time': auction.bidding_end_time.isoformat() if auction.bidding_end_time else None,
                    'bid_count': bid_count,
                    'reserve_price': float(auction.reserve_price) if auction.reserve_price else None
                }
                auction_list.append(auction_info)
            
            return auction_list
            
        except Exception as e:
            logger.error(f"Failed to get active auctions: {str(e)}")
            return []
    
    def _validate_auction_data(self, auction_data: Dict[str, Any]) -> None:
        """Validate auction configuration data."""
        required_fields = ['auction_code', 'auction_name', 'auction_type', 'total_lot_size']
        for field in required_fields:
            if field not in auction_data:
                raise ValueError(f"Missing required field: {field}")
        
        if auction_data['total_lot_size'] <= 0:
            raise ValueError("Total lot size must be positive")
        
        if auction_data.get('minimum_lot_size', 0) <= 0:
            raise ValueError("Minimum lot size must be positive")
    
    def _validate_bid_data(self, bid_data: Dict[str, Any]) -> None:
        """Validate bid submission data."""
        required_fields = ['bid_id', 'auction_id', 'participant_id', 'bid_price', 'bid_quantity']
        for field in required_fields:
            if field not in bid_data:
                raise ValueError(f"Missing required field: {field}")
        
        if bid_data['bid_price'] <= 0:
            raise ValueError("Bid price must be positive")
        
        if bid_data['bid_quantity'] <= 0:
            raise ValueError("Bid quantity must be positive")
    
    async def _check_auction_readiness(self, auction: Auction) -> Dict[str, Any]:
        """Check if auction is ready to start."""
        readiness = {'ready': True, 'reason': None}
        
        # Check if bidding time is set
        if not auction.bidding_start_time:
            readiness['ready'] = False
            readiness['reason'] = "Bidding start time not set"
            return readiness
        
        # Check if settlement date is set
        if not auction.settlement_date:
            readiness['ready'] = False
            readiness['reason'] = "Settlement date not set"
            return readiness
        
        # Check if minimum requirements are met
        if auction.minimum_participation_requirement:
            # This would need more complex logic to check actual participation
            pass
        
        return readiness
    
    async def shutdown(self):
        """Shutdown the auction engine gracefully."""
        logger.info("Shutting down auction engine")
        
        # Close all active auctions
        for auction_id in list(self.active_auctions.keys()):
            try:
                auction = self.active_auctions[auction_id]['auction']
                if auction.status == AuctionStatus.BIDDING_OPEN:
                    await self.close_auction(auction_id)
            except Exception as e:
                logger.error(f"Error closing auction {auction_id}: {str(e)}")
        
        # Clear active auctions
        self.active_auctions.clear()
        
        # Set state to maintenance
        self.state = AuctionEngineState.MAINTENANCE
        
        logger.info("Auction engine shutdown complete")
