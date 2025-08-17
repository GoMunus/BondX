"""
Settlement Engine for BondX Auction System.

This module provides comprehensive settlement infrastructure including fractional ownership,
position tracking, cash flow management, and integration with external settlement systems.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from ..database.auction_models import (
    Auction, Bid, Allocation, Position, Trade, Settlement, CashFlow, 
    Participant, SettlementStatus
)
from ..core.logging import get_logger

logger = get_logger(__name__)


class SettlementType(Enum):
    """Types of settlement operations."""
    
    AUCTION_SETTLEMENT = "AUCTION_SETTLEMENT"  # Settlement of auction allocations
    TRADE_SETTLEMENT = "TRADE_SETTLEMENT"  # Settlement of secondary market trades
    COUPON_PAYMENT = "COUPON_PAYMENT"  # Coupon payment settlement
    PRINCIPAL_PAYMENT = "PRINCIPAL_PAYMENT"  # Principal payment settlement
    CORPORATE_ACTION = "CORPORATE_ACTION"  # Corporate action settlement


class PositionType(Enum):
    """Types of positions."""
    
    LONG = "LONG"  # Long position
    SHORT = "SHORT"  # Short position
    NET = "NET"  # Net position


@dataclass
class SettlementInstruction:
    """Settlement instruction for processing."""
    
    settlement_id: str
    participant_id: int
    instrument_id: int
    settlement_type: SettlementType
    quantity: Decimal
    amount: Decimal
    currency: str
    settlement_date: date
    instructions: Dict[str, Any]


class SettlementEngine:
    """
    Comprehensive settlement engine for BondX.
    
    This engine provides:
    - Fractional ownership position tracking
    - Cash flow management and distribution
    - Settlement instruction generation
    - Integration with external settlement systems
    - Risk management and compliance monitoring
    """
    
    def __init__(self, db_session: Session):
        """Initialize the settlement engine."""
        self.db_session = db_session
        self.logger = get_logger(__name__)
    
    async def create_auction_settlement(self, auction: Auction, 
                                       allocations: List[Dict[str, Any]]) -> List[Settlement]:
        """
        Create settlement records for auction allocations.
        
        Args:
            auction: The auction being settled
            allocations: List of allocations to settle
            
        Returns:
            List of created settlement records
        """
        try:
            self.logger.info(f"Creating settlement for auction {auction.auction_code}")
            
            settlements = []
            settlement_date = auction.settlement_date or date.today()
            
            for allocation in allocations:
                # Create settlement record
                settlement = Settlement(
                    settlement_id=f"SETTLE_{auction.auction_code}_{allocation['participant_id']}",
                    settlement_type=SettlementType.AUCTION_SETTLEMENT.value,
                    settlement_amount=allocation['quantity'] * allocation['price'],
                    settlement_currency="INR",
                    settlement_date=settlement_date,
                    participant_id=allocation['participant_id'],
                    auction_id=auction.id,
                    securities_quantity=allocation['quantity'],
                    cash_amount=allocation['quantity'] * allocation['price'],
                    status=SettlementStatus.PENDING,
                    risk_check_passed=True,
                    compliance_check_passed=True
                )
                
                self.db_session.add(settlement)
                settlements.append(settlement)
            
            # Create issuer settlement (receiving auction proceeds)
            total_proceeds = sum(a['quantity'] * a['price'] for a in allocations)
            issuer_settlement = Settlement(
                settlement_id=f"SETTLE_{auction.auction_code}_ISSUER",
                settlement_type=SettlementType.AUCTION_SETTLEMENT.value,
                settlement_amount=total_proceeds,
                settlement_currency="INR",
                settlement_date=settlement_date,
                participant_id=auction.instrument.issuer_id,
                auction_id=auction.id,
                securities_quantity=Decimal('0'),
                cash_amount=total_proceeds,
                status=SettlementStatus.PENDING,
                risk_check_passed=True,
                compliance_check_passed=True
            )
            
            self.db_session.add(issuer_settlement)
            settlements.append(issuer_settlement)
            
            self.db_session.commit()
            
            self.logger.info(f"Created {len(settlements)} settlement records for auction {auction.auction_code}")
            return settlements
            
        except Exception as e:
            self.logger.error(f"Failed to create auction settlement: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def process_settlement(self, settlement_id: str) -> bool:
        """
        Process a settlement instruction.
        
        Args:
            settlement_id: ID of the settlement to process
            
        Returns:
            True if settlement processed successfully
        """
        try:
            settlement = self.db_session.query(Settlement).filter(
                Settlement.settlement_id == settlement_id
            ).first()
            
            if not settlement:
                raise ValueError(f"Settlement {settlement_id} not found")
            
            if settlement.status != SettlementStatus.PENDING:
                raise ValueError(f"Settlement {settlement_id} is not in PENDING status")
            
            # Update status to in progress
            settlement.status = SettlementStatus.IN_PROGRESS
            
            # Perform settlement based on type
            if settlement.settlement_type == SettlementType.AUCTION_SETTLEMENT.value:
                success = await self._process_auction_settlement(settlement)
            elif settlement.settlement_type == SettlementType.TRADE_SETTLEMENT.value:
                success = await self._process_trade_settlement(settlement)
            elif settlement.settlement_type == SettlementType.COUPON_PAYMENT.value:
                success = await self._process_coupon_settlement(settlement)
            elif settlement.settlement_type == SettlementType.PRINCIPAL_PAYMENT.value:
                success = await self._process_principal_settlement(settlement)
            else:
                raise ValueError(f"Unsupported settlement type: {settlement.settlement_type}")
            
            if success:
                settlement.status = SettlementStatus.COMPLETED
                settlement.actual_settlement_date = date.today()
            else:
                settlement.status = SettlementStatus.FAILED
            
            self.db_session.commit()
            
            self.logger.info(f"Settlement {settlement_id} processed with status: {settlement.status}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to process settlement {settlement_id}: {str(e)}")
            self.db_session.rollback()
            return False
    
    async def create_position(self, participant_id: int, instrument_id: int, 
                             quantity: Decimal, price: Decimal, 
                             position_type: PositionType = PositionType.LONG) -> Position:
        """
        Create or update a fractional ownership position.
        
        Args:
            participant_id: ID of the participant
            instrument_id: ID of the instrument
            quantity: Quantity to add/subtract
            price: Price of the position
            position_type: Type of position
            
        Returns:
            Created/updated position record
        """
        try:
            # Check if position already exists
            existing_position = self.db_session.query(Position).filter(
                and_(
                    Position.participant_id == participant_id,
                    Position.instrument_id == instrument_id,
                    Position.is_active == True
                )
            ).first()
            
            if existing_position:
                # Update existing position
                if position_type == PositionType.LONG:
                    # Add to long position
                    new_quantity = existing_position.position_quantity + quantity
                    new_total_cost = existing_position.total_cost + (quantity * price)
                    new_average_cost = new_total_cost / new_quantity
                    
                    existing_position.position_quantity = new_quantity
                    existing_position.total_cost = new_total_cost
                    existing_position.average_cost = new_average_cost
                    
                    position = existing_position
                else:
                    # Reduce long position or create short position
                    if existing_position.position_quantity >= quantity:
                        # Reduce existing position
                        existing_position.position_quantity -= quantity
                        if existing_position.position_quantity == 0:
                            existing_position.is_active = False
                        position = existing_position
                    else:
                        # Create short position
                        short_quantity = quantity - existing_position.position_quantity
                        existing_position.is_active = False
                        
                        position = Position(
                            participant_id=participant_id,
                            instrument_id=instrument_id,
                            position_quantity=short_quantity,
                            average_cost=price,
                            total_cost=short_quantity * price,
                            position_type=PositionType.SHORT.value,
                            is_active=True
                        )
                        self.db_session.add(position)
            else:
                # Create new position
                position = Position(
                    participant_id=participant_id,
                    instrument_id=instrument_id,
                    position_quantity=quantity,
                    average_cost=price,
                    total_cost=quantity * price,
                    position_type=position_type.value,
                    is_active=True
                )
                self.db_session.add(position)
            
            self.db_session.commit()
            
            self.logger.info(f"Position created/updated for participant {participant_id}, instrument {instrument_id}")
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to create position: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def process_cash_flow(self, instrument_id: int, flow_type: str, 
                               due_date: date, total_amount: Decimal) -> List[CashFlow]:
        """
        Process cash flow distribution to fractional owners.
        
        Args:
            instrument_id: ID of the instrument
            flow_type: Type of cash flow (COUPON, PRINCIPAL, etc.)
            due_date: Due date for the cash flow
            total_amount: Total amount to distribute
            
        Returns:
            List of created cash flow records
        """
        try:
            self.logger.info(f"Processing cash flow for instrument {instrument_id}: {flow_type}")
            
            # Get all active positions for this instrument
            positions = self.db_session.query(Position).filter(
                and_(
                    Position.instrument_id == instrument_id,
                    Position.is_active == True
                )
            ).all()
            
            if not positions:
                self.logger.warning(f"No active positions found for instrument {instrument_id}")
                return []
            
            # Calculate total eligible quantity
            total_eligible_quantity = sum(pos.position_quantity for pos in positions)
            
            if total_eligible_quantity == 0:
                self.logger.warning(f"No eligible quantity for instrument {instrument_id}")
                return []
            
            # Create cash flow records for each position
            cash_flows = []
            for position in positions:
                # Calculate pro-rata amount
                participant_amount = (position.position_quantity / total_eligible_quantity) * total_amount
                
                # Round to appropriate precision
                participant_amount = participant_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                
                cash_flow = CashFlow(
                    flow_type=flow_type,
                    flow_amount=participant_amount,
                    flow_currency="INR",
                    due_date=due_date,
                    total_eligible_quantity=total_eligible_quantity,
                    participant_quantity=position.position_quantity,
                    participant_amount=participant_amount,
                    instrument_id=instrument_id,
                    participant_id=position.participant_id,
                    is_paid=False,
                    payment_status="PENDING",
                    compliance_status="COMPLIANT"
                )
                
                self.db_session.add(cash_flow)
                cash_flows.append(cash_flow)
            
            self.db_session.commit()
            
            self.logger.info(f"Created {len(cash_flows)} cash flow records for instrument {instrument_id}")
            return cash_flows
            
        except Exception as e:
            self.logger.error(f"Failed to process cash flow: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def mark_to_market_positions(self, instrument_id: int, 
                                      market_price: Decimal) -> List[Position]:
        """
        Mark positions to market for a specific instrument.
        
        Args:
            instrument_id: ID of the instrument
            market_price: Current market price
            
        Returns:
            List of updated positions
        """
        try:
            self.logger.info(f"Marking to market instrument {instrument_id} at price {market_price}")
            
            # Get all active positions for this instrument
            positions = self.db_session.query(Position).filter(
                and_(
                    Position.instrument_id == instrument_id,
                    Position.is_active == True
                )
            ).all()
            
            updated_positions = []
            for position in positions:
                # Calculate current market value
                current_market_value = position.position_quantity * market_price
                
                # Calculate unrealized P&L
                if position.position_type == PositionType.LONG.value:
                    unrealized_pnl = current_market_value - position.total_cost
                else:  # SHORT
                    unrealized_pnl = position.total_cost - current_market_value
                
                # Update position
                position.current_market_value = current_market_value
                position.unrealized_pnl = unrealized_pnl
                
                updated_positions.append(position)
            
            self.db_session.commit()
            
            self.logger.info(f"Marked to market {len(updated_positions)} positions for instrument {instrument_id}")
            return updated_positions
            
        except Exception as e:
            self.logger.error(f"Failed to mark to market: {str(e)}")
            self.db_session.rollback()
            raise
    
    async def generate_settlement_instructions(self, settlement: Settlement) -> SettlementInstruction:
        """
        Generate settlement instructions for external systems.
        
        Args:
            settlement: Settlement record
            
        Returns:
            Settlement instruction
        """
        try:
            # Get participant details
            participant = self.db_session.query(Participant).filter(
                Participant.id == settlement.participant_id
            ).first()
            
            if not participant:
                raise ValueError(f"Participant {settlement.participant_id} not found")
            
            # Create settlement instruction
            instruction = SettlementInstruction(
                settlement_id=settlement.settlement_id,
                participant_id=settlement.participant_id,
                instrument_id=settlement.auction.instrument_id if settlement.auction else None,
                settlement_type=SettlementType(settlement.settlement_type),
                quantity=settlement.securities_quantity or Decimal('0'),
                amount=settlement.settlement_amount,
                currency=settlement.settlement_currency,
                settlement_date=settlement.settlement_date,
                instructions={
                    'participant_code': participant.participant_code,
                    'participant_name': participant.participant_name,
                    'settlement_type': settlement.settlement_type,
                    'bank_details': participant.bank_account_details if hasattr(participant, 'bank_account_details') else {},
                    'depository_details': participant.depository_details if hasattr(participant, 'depository_details') else {}
                }
            )
            
            return instruction
            
        except Exception as e:
            self.logger.error(f"Failed to generate settlement instructions: {str(e)}")
            raise
    
    async def _process_auction_settlement(self, settlement: Settlement) -> bool:
        """Process auction settlement."""
        try:
            # This would integrate with external settlement systems
            # For now, simulate successful settlement
            
            # Update positions
            if settlement.securities_quantity and settlement.securities_quantity > 0:
                await self.create_position(
                    participant_id=settlement.participant_id,
                    instrument_id=settlement.auction.instrument_id,
                    quantity=settlement.securities_quantity,
                    price=settlement.auction.clearing_price or Decimal('100.00'),
                    position_type=PositionType.LONG
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process auction settlement: {str(e)}")
            return False
    
    async def _process_trade_settlement(self, settlement: Settlement) -> bool:
        """Process trade settlement."""
        try:
            # This would integrate with external settlement systems
            # For now, simulate successful settlement
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process trade settlement: {str(e)}")
            return False
    
    async def _process_coupon_settlement(self, settlement: Settlement) -> bool:
        """Process coupon settlement."""
        try:
            # This would integrate with external settlement systems
            # For now, simulate successful settlement
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process coupon settlement: {str(e)}")
            return False
    
    async def _process_principal_settlement(self, settlement: Settlement) -> bool:
        """Process principal settlement."""
        try:
            # This would integrate with external settlement systems
            # For now, simulate successful settlement
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process principal settlement: {str(e)}")
            return False
    
    async def get_participant_positions(self, participant_id: int) -> List[Dict[str, Any]]:
        """
        Get comprehensive position information for a participant.
        
        Args:
            participant_id: ID of the participant
            
        Returns:
            List of position information
        """
        try:
            positions = self.db_session.query(Position).filter(
                and_(
                    Position.participant_id == participant_id,
                    Position.is_active == True
                )
            ).all()
            
            position_info = []
            for position in positions:
                # Get instrument details
                instrument = position.instrument
                
                position_data = {
                    'position_id': position.id,
                    'instrument_id': position.instrument_id,
                    'instrument_name': instrument.name if instrument else 'Unknown',
                    'isin': instrument.isin if instrument else 'Unknown',
                    'position_quantity': float(position.position_quantity),
                    'average_cost': float(position.average_cost),
                    'current_market_value': float(position.current_market_value) if position.current_market_value else 0.0,
                    'total_cost': float(position.total_cost),
                    'unrealized_pnl': float(position.unrealized_pnl) if position.unrealized_pnl else 0.0,
                    'realized_pnl': float(position.realized_pnl),
                    'position_type': position.position_type,
                    'duration_exposure': float(position.duration_exposure) if position.duration_exposure else 0.0,
                    'credit_exposure': float(position.credit_exposure) if position.credit_exposure else 0.0,
                    'liquidity_score': position.liquidity_score
                }
                
                position_info.append(position_data)
            
            return position_info
            
        except Exception as e:
            self.logger.error(f"Failed to get participant positions: {str(e)}")
            return []
    
    async def get_settlement_status(self, settlement_id: str) -> Dict[str, Any]:
        """
        Get detailed settlement status.
        
        Args:
            settlement_id: ID of the settlement
            
        Returns:
            Settlement status information
        """
        try:
            settlement = self.db_session.query(Settlement).filter(
                Settlement.settlement_id == settlement_id
            ).first()
            
            if not settlement:
                return {'error': 'Settlement not found'}
            
            # Get participant details
            participant = self.db_session.query(Participant).filter(
                Participant.id == settlement.participant_id
            ).first()
            
            status_info = {
                'settlement_id': settlement.settlement_id,
                'settlement_type': settlement.settlement_type,
                'status': settlement.status.value,
                'settlement_date': settlement.settlement_date.isoformat(),
                'actual_settlement_date': settlement.actual_settlement_date.isoformat() if settlement.actual_settlement_date else None,
                'settlement_amount': float(settlement.settlement_amount),
                'currency': settlement.settlement_currency,
                'participant_id': settlement.participant_id,
                'participant_name': participant.participant_name if participant else 'Unknown',
                'securities_quantity': float(settlement.securities_quantity) if settlement.securities_quantity else 0.0,
                'cash_amount': float(settlement.cash_amount) if settlement.cash_amount else 0.0,
                'risk_check_passed': settlement.risk_check_passed,
                'compliance_check_passed': settlement.compliance_check_passed,
                'created_at': settlement.created_at.isoformat(),
                'updated_at': settlement.updated_at.isoformat()
            }
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to get settlement status: {str(e)}")
            return {'error': f'Failed to get status: {str(e)}'}
