"""
Fractional Bond Ownership API endpoints.

This module provides REST API endpoints for fractional bond ownership,
including buying, selling, and viewing fractional bond positions.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ...core.data_loader import get_bonds_loader, CorporateBond
from ...core.logging import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter(prefix="/bonds", tags=["Fractional Bonds"])

# Pydantic models for request/response
class FractionalPosition(BaseModel):
    """Model for fractional bond position."""
    bond_id: str
    bond_name: str
    issuer: str
    sector: str
    face_value: float
    current_price: float
    yield_to_maturity: float
    owned_fraction: float
    owned_value: float
    purchase_price: float
    current_value: float
    unrealized_pnl: float
    ownership_percentage: float

class BuyFractionRequest(BaseModel):
    """Request model for buying bond fractions."""
    bond_id: str = Field(..., description="ISIN of the bond")
    fraction_amount: float = Field(..., gt=0, le=1, description="Fraction to buy (0-1)")
    max_price: Optional[float] = Field(None, description="Maximum price willing to pay")

class SellFractionRequest(BaseModel):
    """Request model for selling bond fractions."""
    bond_id: str = Field(..., description="ISIN of the bond")
    fraction_amount: float = Field(..., gt=0, le=1, description="Fraction to sell (0-1)")
    min_price: Optional[float] = Field(None, description="Minimum price to accept")

class TransactionResponse(BaseModel):
    """Response model for buy/sell transactions."""
    transaction_id: str
    status: str
    message: str
    bond_id: str
    fraction_amount: float
    price: float
    total_cost: float
    timestamp: datetime

class BondWithFractions(BaseModel):
    """Model for bond with fractional availability."""
    bond_id: str
    bond_name: str
    issuer: str
    sector: str
    face_value: float
    current_price: float
    yield_to_maturity: float
    rating: str
    maturity_date: str
    available_fraction: float
    min_investment: float
    total_volume: float
    daily_trades: int

# In-memory storage for demo (replace with database in production)
user_positions: Dict[str, List[FractionalPosition]] = {"demo_user": []}
transaction_history: List[Dict[str, Any]] = []

def get_current_user():
    """Get current user (simplified for demo)."""
    return "demo_user"

def simulate_market_data(bond: CorporateBond) -> Dict[str, Any]:
    """Simulate live market data for a bond."""
    import random
    
    base_price = bond.last_trade_price if bond.last_trade_price else 100.0
    # Add some realistic market movement
    price_change = random.uniform(-2, 2)
    current_price = max(50, base_price + price_change)
    
    return {
        "current_price": round(current_price, 2),
        "available_fraction": round(random.uniform(0.1, 0.8), 3),
        "min_investment": 1000,
        "daily_trades": random.randint(5, 50)
    }

@router.get("/", response_model=List[BondWithFractions])
async def get_available_bonds(
    sector: Optional[str] = None,
    min_yield: Optional[float] = None,
    max_yield: Optional[float] = None,
    limit: int = 50
) -> List[BondWithFractions]:
    """Get list of bonds available for fractional investment."""
    try:
        loader = get_bonds_loader()
        bonds = loader.get_all_bonds()
        
        # Apply filters
        if sector:
            bonds = [b for b in bonds if b.sector and sector.lower() in b.sector.lower()]
        if min_yield:
            bonds = [b for b in bonds if b.last_trade_yield and b.last_trade_yield >= min_yield]
        if max_yield:
            bonds = [b for b in bonds if b.last_trade_yield and b.last_trade_yield <= max_yield]
        
        # Convert to fractional bonds format
        fractional_bonds = []
        for bond in bonds[:limit]:
            market_data = simulate_market_data(bond)
            
            fractional_bond = BondWithFractions(
                bond_id=bond.isin,
                bond_name=bond.descriptor[:100] if bond.descriptor else "Bond",
                issuer=bond.issuer_name or "Unknown Issuer",
                sector=bond.sector or "Unknown",
                face_value=bond.face_value or 1000,
                current_price=market_data["current_price"],
                yield_to_maturity=bond.last_trade_yield or 0.0,
                rating=bond.bond_type or "NR",
                maturity_date=bond.maturity_date or "2030-12-31",
                available_fraction=market_data["available_fraction"],
                min_investment=market_data["min_investment"],
                total_volume=bond.value_lakhs or 0,
                daily_trades=market_data["daily_trades"]
            )
            fractional_bonds.append(fractional_bond)
        
        logger.info(f"Retrieved {len(fractional_bonds)} bonds with fractional availability")
        return fractional_bonds
        
    except Exception as e:
        logger.error(f"Error retrieving bonds: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve bonds")

@router.get("/positions", response_model=List[FractionalPosition])
async def get_user_positions(user_id: str = Depends(get_current_user)) -> List[FractionalPosition]:
    """Get user's fractional bond positions."""
    try:
        positions = user_positions.get(user_id, [])
        logger.info(f"Retrieved {len(positions)} positions for user {user_id}")
        return positions
        
    except Exception as e:
        logger.error(f"Error retrieving positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve positions")

@router.post("/buy-fraction", response_model=TransactionResponse)
async def buy_bond_fraction(
    request: BuyFractionRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
) -> TransactionResponse:
    """Buy a fraction of a bond."""
    try:
        # Get bond data
        loader = get_bonds_loader()
        bonds = loader.get_all_bonds()
        bond = next((b for b in bonds if b.isin == request.bond_id), None)
        
        if not bond:
            raise HTTPException(status_code=404, detail="Bond not found")
        
        # Simulate current market price
        market_data = simulate_market_data(bond)
        current_price = market_data["current_price"]
        
        # Check price limits
        if request.max_price and current_price > request.max_price:
            raise HTTPException(
                status_code=400, 
                detail=f"Current price {current_price} exceeds maximum price {request.max_price}"
            )
        
        # Calculate transaction details
        face_value = bond.face_value or 1000
        total_cost = current_price * request.fraction_amount * face_value / 100
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        transaction = TransactionResponse(
            transaction_id=transaction_id,
            status="completed",
            message="Fraction purchased successfully",
            bond_id=request.bond_id,
            fraction_amount=request.fraction_amount,
            price=current_price,
            total_cost=total_cost,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Update user positions
        if user_id not in user_positions:
            user_positions[user_id] = []
        
        # Check if user already owns this bond
        existing_position = next(
            (p for p in user_positions[user_id] if p.bond_id == request.bond_id), 
            None
        )
        
        if existing_position:
            # Update existing position
            existing_position.owned_fraction += request.fraction_amount
            existing_position.owned_value += total_cost
            existing_position.current_value = existing_position.owned_fraction * current_price * face_value / 100
            existing_position.unrealized_pnl = existing_position.current_value - existing_position.owned_value
            existing_position.ownership_percentage = existing_position.owned_fraction * 100
        else:
            # Create new position
            new_position = FractionalPosition(
                bond_id=request.bond_id,
                bond_name=bond.descriptor[:100] if bond.descriptor else "Bond",
                issuer=bond.issuer_name or "Unknown Issuer",
                sector=bond.sector or "Unknown",
                face_value=face_value,
                current_price=current_price,
                yield_to_maturity=bond.last_trade_yield or 0.0,
                owned_fraction=request.fraction_amount,
                owned_value=total_cost,
                purchase_price=current_price,
                current_value=total_cost,
                unrealized_pnl=0.0,
                ownership_percentage=request.fraction_amount * 100
            )
            user_positions[user_id].append(new_position)
        
        # Add to transaction history
        transaction_history.append({
            "transaction_id": transaction_id,
            "user_id": user_id,
            "type": "buy",
            "bond_id": request.bond_id,
            "fraction_amount": request.fraction_amount,
            "price": current_price,
            "total_cost": total_cost,
            "timestamp": transaction.timestamp
        })
        
        # Schedule background task to update portfolio metrics
        background_tasks.add_task(update_portfolio_metrics, user_id)
        
        logger.info(f"User {user_id} bought {request.fraction_amount} of bond {request.bond_id}")
        return transaction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buying bond fraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to buy bond fraction")

@router.post("/sell-fraction", response_model=TransactionResponse)
async def sell_bond_fraction(
    request: SellFractionRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
) -> TransactionResponse:
    """Sell a fraction of a bond."""
    try:
        # Check if user owns this bond
        user_positions_list = user_positions.get(user_id, [])
        existing_position = next(
            (p for p in user_positions_list if p.bond_id == request.bond_id), 
            None
        )
        
        if not existing_position:
            raise HTTPException(status_code=404, detail="Bond position not found")
        
        if existing_position.owned_fraction < request.fraction_amount:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient fraction. Owned: {existing_position.owned_fraction}, Requested: {request.fraction_amount}"
            )
        
        # Get current market price
        loader = get_bonds_loader()
        bonds = loader.get_all_bonds()
        bond = next((b for b in bonds if b.isin == request.bond_id), None)
        
        if not bond:
            raise HTTPException(status_code=404, detail="Bond not found")
        
        market_data = simulate_market_data(bond)
        current_price = market_data["current_price"]
        
        # Check price limits
        if request.min_price and current_price < request.min_price:
            raise HTTPException(
                status_code=400, 
                detail=f"Current price {current_price} below minimum price {request.min_price}"
            )
        
        # Calculate transaction details
        face_value = existing_position.face_value
        total_proceeds = current_price * request.fraction_amount * face_value / 100
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        transaction = TransactionResponse(
            transaction_id=transaction_id,
            status="completed",
            message="Fraction sold successfully",
            bond_id=request.bond_id,
            fraction_amount=request.fraction_amount,
            price=current_price,
            total_cost=total_proceeds,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Update position
        existing_position.owned_fraction -= request.fraction_amount
        existing_position.owned_value *= (existing_position.owned_fraction + request.fraction_amount) / (existing_position.owned_fraction + request.fraction_amount)
        existing_position.current_value = existing_position.owned_fraction * current_price * face_value / 100
        existing_position.unrealized_pnl = existing_position.current_value - existing_position.owned_value
        existing_position.ownership_percentage = existing_position.owned_fraction * 100
        
        # Remove position if fraction becomes zero
        if existing_position.owned_fraction <= 0:
            user_positions[user_id].remove(existing_position)
        
        # Add to transaction history
        transaction_history.append({
            "transaction_id": transaction_id,
            "user_id": user_id,
            "type": "sell",
            "bond_id": request.bond_id,
            "fraction_amount": request.fraction_amount,
            "price": current_price,
            "total_proceeds": total_proceeds,
            "timestamp": transaction.timestamp
        })
        
        # Schedule background task to update portfolio metrics
        background_tasks.add_task(update_portfolio_metrics, user_id)
        
        logger.info(f"User {user_id} sold {request.fraction_amount} of bond {request.bond_id}")
        return transaction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selling bond fraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to sell bond fraction")

async def update_portfolio_metrics(user_id: str):
    """Background task to update portfolio metrics after transactions."""
    try:
        # In a real application, this would trigger portfolio recalculation
        # and send WebSocket updates to connected clients
        logger.info(f"Portfolio metrics updated for user {user_id}")
        
        # Simulate WebSocket notification
        # await websocket_manager.broadcast_to_user(user_id, {
        #     "type": "portfolio_update",
        #     "message": "Portfolio updated after transaction"
        # })
        
    except Exception as e:
        logger.error(f"Error updating portfolio metrics: {e}")

@router.get("/transactions")
async def get_transaction_history(
    user_id: str = Depends(get_current_user),
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get user's transaction history."""
    try:
        user_transactions = [
            t for t in transaction_history 
            if t.get("user_id") == user_id
        ]
        
        # Sort by timestamp, most recent first
        user_transactions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return user_transactions[:limit]
        
    except Exception as e:
        logger.error(f"Error retrieving transaction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve transaction history")
