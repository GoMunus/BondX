"""
Auction API endpoints for BondX.

This module provides comprehensive REST API endpoints for auction operations,
including auction creation, bidding, processing, and settlement management.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ...database.base import get_db
from ...database.auction_models import (
    Auction, Bid, Allocation, Participant, AuctionStatus, BidStatus,
    AuctionType, SettlementStatus
)
from ...auction_engine.auction_engine import AuctionEngine
from ...auction_engine.settlement_engine import SettlementEngine
from ...auction_engine.websocket_manager import WebSocketManager, WebSocketMessage, MessageType
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auctions", tags=["auctions"])

# Global instances (in production, these would be properly managed)
auction_engine: Optional[AuctionEngine] = None
settlement_engine: Optional[SettlementEngine] = None
websocket_manager: Optional[WebSocketManager] = None


def get_auction_engine(db: Session = Depends(get_db)) -> AuctionEngine:
    """Get auction engine instance."""
    global auction_engine
    if auction_engine is None:
        auction_engine = AuctionEngine(db)
    return auction_engine


def get_settlement_engine(db: Session = Depends(get_db)) -> SettlementEngine:
    """Get settlement engine instance."""
    global settlement_engine
    if settlement_engine is None:
        settlement_engine = SettlementEngine(db)
    return settlement_engine


def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance."""
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = WebSocketManager()
    return websocket_manager


# Auction Management Endpoints

@router.post("/", response_model=Dict[str, Any])
async def create_auction(
    auction_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    auction_engine: AuctionEngine = Depends(get_auction_engine)
):
    """
    Create a new auction.
    
    This endpoint creates a new auction with comprehensive validation and risk checks.
    """
    try:
        logger.info(f"Creating auction: {auction_data.get('auction_name', 'Unknown')}")
        
        # Validate required fields
        required_fields = ['auction_code', 'auction_name', 'auction_type', 'total_lot_size']
        for field in required_fields:
            if field not in auction_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create auction
        auction = await auction_engine.create_auction(auction_data)
        
        # Send WebSocket notification
        websocket_manager = get_websocket_manager()
        if websocket_manager:
            notification = WebSocketMessage(
                message_type=MessageType.AUCTION_UPDATE,
                timestamp=datetime.utcnow(),
                data={
                    "action": "created",
                    "auction_id": auction.id,
                    "auction_code": auction.auction_code,
                    "auction_name": auction.auction_name,
                    "status": auction.status.value
                }
            )
            await websocket_manager.broadcast_message(notification)
        
        return {
            "success": True,
            "message": "Auction created successfully",
            "auction_id": auction.id,
            "auction_code": auction.auction_code,
            "status": auction.status.value
        }
        
    except ValueError as e:
        logger.error(f"Validation error creating auction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating auction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[Dict[str, Any]])
async def get_auctions(
    status: Optional[AuctionStatus] = Query(None, description="Filter by auction status"),
    auction_type: Optional[AuctionType] = Query(None, description="Filter by auction type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of auctions to return"),
    offset: int = Query(0, ge=0, description="Number of auctions to skip"),
    db: Session = Depends(get_db)
):
    """
    Get list of auctions with optional filtering.
    
    This endpoint returns a paginated list of auctions with comprehensive filtering options.
    """
    try:
        query = db.query(Auction)
        
        # Apply filters
        if status:
            query = query.filter(Auction.status == status)
        if auction_type:
            query = query.filter(Auction.auction_type == auction_type)
        
        # Apply pagination
        total_count = query.count()
        auctions = query.offset(offset).limit(limit).all()
        
        # Format response
        auction_list = []
        for auction in auctions:
            # Get bid count
            bid_count = db.query(Bid).filter(Bid.auction_id == auction.id).count()
            
            auction_data = {
                "auction_id": auction.id,
                "auction_code": auction.auction_code,
                "auction_name": auction.auction_name,
                "auction_type": auction.auction_type.value,
                "status": auction.status.value,
                "total_lot_size": float(auction.total_lot_size),
                "minimum_lot_size": float(auction.minimum_lot_size),
                "reserve_price": float(auction.reserve_price) if auction.reserve_price else None,
                "bidding_start_time": auction.bidding_start_time.isoformat() if auction.bidding_start_time else None,
                "bidding_end_time": auction.bidding_end_time.isoformat() if auction.bidding_end_time else None,
                "settlement_date": auction.settlement_date.isoformat() if auction.settlement_date else None,
                "bid_count": bid_count,
                "clearing_price": float(auction.clearing_price) if auction.clearing_price else None,
                "bid_to_cover_ratio": auction.bid_to_cover_ratio,
                "created_at": auction.created_at.isoformat()
            }
            auction_list.append(auction_data)
        
        return {
            "auctions": auction_list,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting auctions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{auction_id}", response_model=Dict[str, Any])
async def get_auction(
    auction_id: int = Path(..., description="ID of the auction"),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific auction.
    
    This endpoint returns comprehensive auction information including statistics and results.
    """
    try:
        auction = db.query(Auction).filter(Auction.id == auction_id).first()
        if not auction:
            raise HTTPException(status_code=404, detail="Auction not found")
        
        # Get bid statistics
        bid_stats = db.query(
            db.func.count(Bid.id).label('total_bids'),
            db.func.sum(Bid.bid_quantity).label('total_quantity'),
            db.func.avg(Bid.bid_price).label('average_price')
        ).filter(Bid.auction_id == auction_id).first()
        
        # Get allocation statistics
        allocation_stats = db.query(
            db.func.count(Allocation.id).label('total_allocations'),
            db.func.sum(Allocation.allocation_quantity).label('total_allocated'),
            db.func.avg(Allocation.allocation_price).label('average_allocation_price')
        ).filter(Allocation.auction_id == auction_id).first()
        
        # Format response
        auction_data = {
            "auction_id": auction.id,
            "auction_code": auction.auction_code,
            "auction_name": auction.auction_name,
            "auction_type": auction.auction_type.value,
            "status": auction.status.value,
            "total_lot_size": float(auction.total_lot_size),
            "minimum_lot_size": float(auction.minimum_lot_size),
            "lot_size_increment": float(auction.lot_size_increment),
            "reserve_price": float(auction.reserve_price) if auction.reserve_price else None,
            "minimum_price": float(auction.minimum_price) if auction.minimum_price else None,
            "maximum_price": float(auction.maximum_price) if auction.maximum_price else None,
            "price_increment": float(auction.price_increment) if auction.price_increment else None,
            "announcement_date": auction.announcement_date.isoformat() if auction.announcement_date else None,
            "bidding_start_time": auction.bidding_start_time.isoformat() if auction.bidding_start_time else None,
            "bidding_end_time": auction.bidding_end_time.isoformat() if auction.bidding_end_time else None,
            "settlement_date": auction.settlement_date.isoformat() if auction.settlement_date else None,
            "eligible_participants": auction.eligible_participants,
            "maximum_allocation_per_participant": float(auction.maximum_allocation_per_participant) if auction.maximum_allocation_per_participant else None,
            "allocation_method": auction.allocation_method,
            "bid_statistics": {
                "total_bids": bid_stats.total_bids or 0,
                "total_quantity": float(bid_stats.total_quantity) if bid_stats.total_quantity else 0.0,
                "average_price": float(bid_stats.average_price) if bid_stats.average_price else 0.0
            },
            "allocation_statistics": {
                "total_allocations": allocation_stats.total_allocations or 0,
                "total_allocated": float(allocation_stats.total_allocated) if allocation_stats.total_allocated else 0.0,
                "average_allocation_price": float(allocation_stats.average_allocation_price) if allocation_stats.average_allocation_price else 0.0
            },
            "results": {
                "clearing_price": float(auction.clearing_price) if auction.clearing_price else None,
                "bid_to_cover_ratio": auction.bid_to_cover_ratio,
                "total_lots_allocated": float(auction.total_lots_allocated) if auction.total_lots_allocated else None
            },
            "created_at": auction.created_at.isoformat(),
            "updated_at": auction.updated_at.isoformat()
        }
        
        return auction_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{auction_id}/start", response_model=Dict[str, Any])
async def start_auction(
    auction_id: int = Path(..., description="ID of the auction to start"),
    db: Session = Depends(get_db),
    auction_engine: AuctionEngine = Depends(get_auction_engine)
):
    """
    Start an auction and open bidding.
    
    This endpoint starts an auction and opens it for bidding with comprehensive validation.
    """
    try:
        logger.info(f"Starting auction {auction_id}")
        
        success = await auction_engine.start_auction(auction_id)
        
        if success:
            # Send WebSocket notification
            websocket_manager = get_websocket_manager()
            if websocket_manager:
                notification = WebSocketMessage(
                    message_type=MessageType.AUCTION_UPDATE,
                    timestamp=datetime.utcnow(),
                    data={
                        "action": "started",
                        "auction_id": auction_id,
                        "status": "BIDDING_OPEN",
                        "start_time": datetime.utcnow().isoformat()
                    }
                )
                await websocket_manager.broadcast_message(notification, topic=f"auction_{auction_id}")
            
            return {
                "success": True,
                "message": "Auction started successfully",
                "auction_id": auction_id,
                "status": "BIDDING_OPEN"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to start auction")
        
    except ValueError as e:
        logger.error(f"Error starting auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{auction_id}/close", response_model=Dict[str, Any])
async def close_auction(
    auction_id: int = Path(..., description="ID of the auction to close"),
    db: Session = Depends(get_db),
    auction_engine: AuctionEngine = Depends(get_auction_engine)
):
    """
    Close an auction and stop accepting bids.
    
    This endpoint closes an auction and stops accepting new bids.
    """
    try:
        logger.info(f"Closing auction {auction_id}")
        
        success = await auction_engine.close_auction(auction_id)
        
        if success:
            # Send WebSocket notification
            websocket_manager = get_websocket_manager()
            if websocket_manager:
                notification = WebSocketMessage(
                    message_type=MessageType.AUCTION_UPDATE,
                    timestamp=datetime.utcnow(),
                    data={
                        "action": "closed",
                        "auction_id": auction_id,
                        "status": "BIDDING_CLOSED",
                        "close_time": datetime.utcnow().isoformat()
                    }
                )
                await websocket_manager.broadcast_message(notification, topic=f"auction_{auction_id}")
            
            return {
                "success": True,
                "message": "Auction closed successfully",
                "auction_id": auction_id,
                "status": "BIDDING_CLOSED"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to close auction")
        
    except ValueError as e:
        logger.error(f"Error closing auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error closing auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{auction_id}/process", response_model=Dict[str, Any])
async def process_auction(
    auction_id: int = Path(..., description="ID of the auction to process"),
    db: Session = Depends(get_db),
    auction_engine: AuctionEngine = Depends(get_auction_engine)
):
    """
    Process a closed auction and determine results.
    
    This endpoint processes a closed auction, determines clearing price and allocations.
    """
    try:
        logger.info(f"Processing auction {auction_id}")
        
        result = await auction_engine.process_auction(auction_id)
        
        if result.success:
            # Send WebSocket notification
            websocket_manager = get_websocket_manager()
            if websocket_manager:
                notification = WebSocketMessage(
                    message_type=MessageType.ALLOCATION_UPDATE,
                    timestamp=datetime.utcnow(),
                    data={
                        "action": "processed",
                        "auction_id": auction_id,
                        "clearing_price": float(result.clearing_price),
                        "total_allocated": float(result.total_allocated),
                        "bid_to_cover_ratio": result.bid_to_cover_ratio,
                        "processing_time": result.processing_time
                    }
                )
                await websocket_manager.broadcast_message(notification, topic=f"auction_{auction_id}")
            
            return {
                "success": True,
                "message": "Auction processed successfully",
                "auction_id": auction_id,
                "clearing_price": float(result.clearing_price),
                "total_allocated": float(result.total_allocated),
                "bid_to_cover_ratio": result.bid_to_cover_ratio,
                "processing_time": result.processing_time,
                "allocations": result.allocations
            }
        else:
            return {
                "success": False,
                "message": "Auction processing failed",
                "auction_id": auction_id,
                "error_message": result.error_message,
                "processing_time": result.processing_time
            }
        
    except Exception as e:
        logger.error(f"Error processing auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Bidding Endpoints

@router.post("/{auction_id}/bids", response_model=Dict[str, Any])
async def submit_bid(
    auction_id: int = Path(..., description="ID of the auction"),
    bid_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    auction_engine: AuctionEngine = Depends(get_auction_engine)
):
    """
    Submit a bid for an auction.
    
    This endpoint accepts bid submissions with comprehensive validation and risk checks.
    """
    try:
        logger.info(f"Processing bid submission for auction {auction_id}")
        
        # Add auction_id to bid data
        bid_data['auction_id'] = auction_id
        
        # Validate required fields
        required_fields = ['bid_id', 'participant_id', 'bid_price', 'bid_quantity']
        for field in required_fields:
            if field not in bid_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Submit bid
        bid = await auction_engine.submit_bid(bid_data)
        
        # Send WebSocket notification
        websocket_manager = get_websocket_manager()
        if websocket_manager:
            notification = WebSocketMessage(
                message_type=MessageType.BID_UPDATE,
                timestamp=datetime.utcnow(),
                data={
                    "action": "submitted",
                    "bid_id": bid.bid_id,
                    "auction_id": auction_id,
                    "participant_id": bid.participant_id,
                    "bid_price": float(bid.bid_price),
                    "bid_quantity": float(bid.bid_quantity),
                    "status": bid.status.value,
                    "submission_time": bid.submission_time.isoformat()
                }
            )
            await websocket_manager.broadcast_message(notification, topic=f"auction_{auction_id}")
        
        return {
            "success": True,
            "message": "Bid submitted successfully",
            "bid_id": bid.bid_id,
            "auction_id": auction_id,
            "status": bid.status.value
        }
        
    except ValueError as e:
        logger.error(f"Validation error submitting bid: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting bid: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{auction_id}/bids", response_model=List[Dict[str, Any]])
async def get_auction_bids(
    auction_id: int = Path(..., description="ID of the auction"),
    status: Optional[BidStatus] = Query(None, description="Filter by bid status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of bids to return"),
    offset: int = Query(0, ge=0, description="Number of bids to skip"),
    db: Session = Depends(get_db)
):
    """
    Get bids for a specific auction.
    
    This endpoint returns a paginated list of bids for an auction with filtering options.
    """
    try:
        query = db.query(Bid).filter(Bid.auction_id == auction_id)
        
        # Apply filters
        if status:
            query = query.filter(Bid.status == status)
        
        # Apply pagination
        total_count = query.count()
        bids = query.offset(offset).limit(limit).all()
        
        # Format response
        bid_list = []
        for bid in bids:
            bid_data = {
                "bid_id": bid.bid_id,
                "auction_id": bid.auction_id,
                "participant_id": bid.participant_id,
                "bid_price": float(bid.bid_price),
                "bid_quantity": float(bid.bid_quantity),
                "bid_yield": float(bid.bid_yield) if bid.bid_yield else None,
                "status": bid.status.value,
                "allocation_quantity": float(bid.allocation_quantity) if bid.allocation_quantity else None,
                "allocation_price": float(bid.allocation_price) if bid.allocation_price else None,
                "submission_time": bid.submission_time.isoformat(),
                "last_modified": bid.last_modified.isoformat() if bid.last_modified else None,
                "order_type": bid.order_type.value,
                "time_in_force": bid.time_in_force.value,
                "risk_check_passed": bid.risk_check_passed
            }
            bid_list.append(bid_data)
        
        return {
            "bids": bid_list,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting bids for auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Settlement Endpoints

@router.post("/{auction_id}/settle", response_model=Dict[str, Any])
async def settle_auction(
    auction_id: int = Path(..., description="ID of the auction to settle"),
    db: Session = Depends(get_db),
    settlement_engine: SettlementEngine = Depends(get_settlement_engine)
):
    """
    Create settlement records for an auction.
    
    This endpoint creates settlement records for all auction allocations.
    """
    try:
        logger.info(f"Creating settlement for auction {auction_id}")
        
        # Get auction
        auction = db.query(Auction).filter(Auction.id == auction_id).first()
        if not auction:
            raise HTTPException(status_code=404, detail="Auction not found")
        
        # Get allocations
        allocations = db.query(Allocation).filter(Allocation.auction_id == auction_id).all()
        if not allocations:
            raise HTTPException(status_code=400, detail="No allocations found for auction")
        
        # Format allocations for settlement
        allocation_data = []
        for allocation in allocations:
            allocation_data.append({
                'participant_id': allocation.participant_id,
                'quantity': allocation.allocation_quantity,
                'price': allocation.allocation_price
            })
        
        # Create settlements
        settlements = await settlement_engine.create_auction_settlement(auction, allocation_data)
        
        # Send WebSocket notification
        websocket_manager = get_websocket_manager()
        if websocket_manager:
            notification = WebSocketMessage(
                message_type=MessageType.SETTLEMENT_UPDATE,
                timestamp=datetime.utcnow(),
                data={
                    "action": "created",
                    "auction_id": auction_id,
                    "settlement_count": len(settlements),
                    "settlement_ids": [s.settlement_id for s in settlements]
                }
            )
            await websocket_manager.broadcast_message(notification, topic=f"auction_{auction_id}")
        
        return {
            "success": True,
            "message": "Settlement records created successfully",
            "auction_id": auction_id,
            "settlement_count": len(settlements),
            "settlement_ids": [s.settlement_id for s in settlements]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating settlement for auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{auction_id}/allocations", response_model=List[Dict[str, Any]])
async def get_auction_allocations(
    auction_id: int = Path(..., description="ID of the auction"),
    db: Session = Depends(get_db)
):
    """
    Get allocations for a specific auction.
    
    This endpoint returns all allocations for an auction.
    """
    try:
        allocations = db.query(Allocation).filter(Allocation.auction_id == auction_id).all()
        
        allocation_list = []
        for allocation in allocations:
            allocation_data = {
                "allocation_id": allocation.id,
                "auction_id": allocation.auction_id,
                "participant_id": allocation.participant_id,
                "allocation_quantity": float(allocation.allocation_quantity),
                "allocation_price": float(allocation.allocation_price),
                "allocation_yield": float(allocation.allocation_yield) if allocation.allocation_yield else None,
                "allocation_method": allocation.allocation_method,
                "allocation_priority": allocation.allocation_priority,
                "allocation_round": allocation.allocation_round,
                "is_confirmed": allocation.is_confirmed,
                "confirmation_time": allocation.confirmation_time.isoformat() if allocation.confirmation_time else None,
                "created_at": allocation.created_at.isoformat()
            }
            allocation_list.append(allocation_data)
        
        return allocation_list
        
    except Exception as e:
        logger.error(f"Error getting allocations for auction {auction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# WebSocket Status Endpoint

@router.get("/websocket/status", response_model=Dict[str, Any])
async def get_websocket_status():
    """
    Get WebSocket system status.
    
    This endpoint returns the current status of the WebSocket communication system.
    """
    try:
        websocket_manager = get_websocket_manager()
        if websocket_manager:
            stats = websocket_manager.get_system_stats()
            return {
                "success": True,
                "websocket_status": stats
            }
        else:
            return {
                "success": False,
                "message": "WebSocket manager not available"
            }
        
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health Check Endpoint

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check for auction system.
    
    This endpoint provides a health check for the auction system components.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "auction_engine": "available" if auction_engine else "unavailable",
                "settlement_engine": "available" if settlement_engine else "unavailable",
                "websocket_manager": "available" if websocket_manager else "unavailable"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
