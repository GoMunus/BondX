"""Simple Bond Auction API for fractional ownership."""

from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/simple-auctions", tags=["Simple Auctions"])

class BidRequest(BaseModel):
    auction_id: str
    bid_amount: float = Field(gt=0)
    fraction_amount: float = Field(gt=0, le=1)

class AuctionSummary(BaseModel):
    auction_id: str
    bond_id: str
    bond_name: str
    issuer: str
    current_highest_bid: float
    available_fraction: float
    end_time: datetime
    time_remaining: int
    total_bids: int

# Storage
auctions = {}
bids = {}

def init_sample_auctions():
    for i in range(3):
        aid = f"auction_{i+1}"
        auctions[aid] = {
            "auction_id": aid,
            "bond_id": f"ISIN{i+1}",
            "bond_name": f"Sample Bond {i+1}",
            "issuer": f"Issuer {i+1}",
            "current_highest_bid": 100.0 + i*5,
            "available_fraction": 0.2,
            "end_time": datetime.now(timezone.utc) + timedelta(hours=1),
            "total_bids": i+2
        }
        bids[aid] = []

init_sample_auctions()

@router.get("/", response_model=List[AuctionSummary])
async def get_auctions():
    result = []
    for auction in auctions.values():
        time_remaining = max(0, int((auction["end_time"] - datetime.now(timezone.utc)).total_seconds()))
        result.append(AuctionSummary(
            auction_id=auction["auction_id"],
            bond_id=auction["bond_id"],
            bond_name=auction["bond_name"],
            issuer=auction["issuer"],
            current_highest_bid=auction["current_highest_bid"],
            available_fraction=auction["available_fraction"],
            end_time=auction["end_time"],
            time_remaining=time_remaining,
            total_bids=auction["total_bids"]
        ))
    return result

@router.post("/place-bid")
async def place_bid(request: BidRequest):
    if request.auction_id not in auctions:
        raise HTTPException(404, "Auction not found")
    
    bid_id = str(uuid.uuid4())
    auction = auctions[request.auction_id]
    
    if request.bid_amount > auction["current_highest_bid"]:
        auction["current_highest_bid"] = request.bid_amount
    
    auction["total_bids"] += 1
    
    bid = {
        "bid_id": bid_id,
        "bid_amount": request.bid_amount,
        "fraction_amount": request.fraction_amount,
        "timestamp": datetime.now(timezone.utc)
    }
    
    bids[request.auction_id].append(bid)
    
    return {"bid_id": bid_id, "status": "submitted", "message": "Bid placed successfully"}
