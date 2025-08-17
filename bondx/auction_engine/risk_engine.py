"""
Risk Engine for BondX Auction System.

This module provides comprehensive risk management, compliance monitoring,
and anti-manipulation detection for auction operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from ..database.auction_models import (
    Auction, Bid, Allocation, Participant, ParticipantType
)
from ..core.logging import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk levels for risk assessment."""
    
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ComplianceStatus(Enum):
    """Compliance status indicators."""
    
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    BLOCKED = "BLOCKED"


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    
    passed: bool
    risk_level: RiskLevel
    compliance_status: ComplianceStatus
    reason: Optional[str] = None
    risk_score: float = 0.0
    details: Optional[Dict[str, Any]] = None


@dataclass
class ManipulationAlert:
    """Alert for potential market manipulation."""
    
    alert_id: str
    alert_type: str
    severity: RiskLevel
    description: str
    participant_id: Optional[int] = None
    auction_id: Optional[int] = None
    bid_id: Optional[int] = None
    timestamp: datetime
    evidence: Dict[str, Any]
    status: str = "OPEN"


class RiskEngine:
    """
    Comprehensive risk engine for auction operations.
    
    This engine provides:
    - Pre-trade risk checks
    - Compliance monitoring
    - Anti-manipulation detection
    - Position limit monitoring
    - Credit risk assessment
    - Market abuse prevention
    """
    
    def __init__(self, db_session: Session):
        """Initialize the risk engine."""
        self.db_session = db_session
        self.logger = get_logger(__name__)
        
        # Risk thresholds and limits
        self.risk_thresholds = {
            'max_position_concentration': 0.25,  # 25% of total auction size
            'max_bid_concentration': 0.15,  # 15% of total bids
            'min_bid_spread': 0.001,  # 1 basis point minimum spread
            'max_price_deviation': 0.05,  # 5% maximum price deviation
            'max_bid_frequency': 10,  # Maximum bids per minute per participant
            'max_credit_exposure': 1000000,  # 1M INR maximum credit exposure
        }
        
        # Manipulation detection patterns
        self.manipulation_patterns = {
            'bid_clustering': {'threshold': 0.8, 'description': 'Unusual bid clustering detected'},
            'price_manipulation': {'threshold': 0.05, 'description': 'Suspicious price manipulation'},
            'timing_patterns': {'threshold': 0.9, 'description': 'Suspicious bid timing patterns'},
            'quantity_patterns': {'threshold': 0.7, 'description': 'Unusual quantity patterns'},
            'participant_concentration': {'threshold': 0.3, 'description': 'High participant concentration'}
        }
    
    async def validate_auction(self, auction: Auction) -> RiskCheckResult:
        """
        Validate auction configuration for risk and compliance.
        
        Args:
            auction: Auction to validate
            
        Returns:
            RiskCheckResult with validation results
        """
        try:
            self.logger.info(f"Validating auction {auction.auction_code} for risk and compliance")
            
            risk_score = 0.0
            issues = []
            
            # Check auction size and lot configuration
            if auction.total_lot_size <= 0:
                issues.append("Invalid auction lot size")
                risk_score += 0.3
            
            if auction.minimum_lot_size <= 0:
                issues.append("Invalid minimum lot size")
                risk_score += 0.2
            
            if auction.minimum_lot_size > auction.total_lot_size:
                issues.append("Minimum lot size exceeds total lot size")
                risk_score += 0.4
            
            # Check price parameters
            if auction.reserve_price and auction.reserve_price <= 0:
                issues.append("Invalid reserve price")
                risk_score += 0.2
            
            if auction.minimum_price and auction.maximum_price:
                if auction.minimum_price >= auction.maximum_price:
                    issues.append("Invalid price range")
                    risk_score += 0.3
            
            # Check timing parameters
            if auction.bidding_start_time and auction.bidding_end_time:
                if auction.bidding_start_time >= auction.bidding_end_time:
                    issues.append("Invalid bidding time range")
                    risk_score += 0.2
            
            # Check eligibility and restrictions
            if not auction.eligible_participants:
                issues.append("No eligible participants specified")
                risk_score += 0.3
            
            # Determine risk level and compliance status
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
                compliance_status = ComplianceStatus.BLOCKED
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
                compliance_status = ComplianceStatus.VIOLATION
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
                compliance_status = ComplianceStatus.WARNING
            else:
                risk_level = RiskLevel.LOW
                compliance_status = ComplianceStatus.COMPLIANT
            
            passed = risk_score < 0.6
            reason = "; ".join(issues) if issues else "All checks passed"
            
            result = RiskCheckResult(
                passed=passed,
                risk_level=risk_level,
                compliance_status=compliance_status,
                reason=reason,
                risk_score=risk_score,
                details={
                    'issues': issues,
                    'auction_code': auction.auction_code,
                    'validation_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            self.logger.info(f"Auction validation completed: {result.compliance_status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating auction: {str(e)}")
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                compliance_status=ComplianceStatus.BLOCKED,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )
    
    async def validate_bid(self, bid_data: Dict[str, Any]) -> RiskCheckResult:
        """
        Validate bid submission for risk and compliance.
        
        Args:
            bid_data: Bid data to validate
            
        Returns:
            RiskCheckResult with validation results
        """
        try:
            self.logger.info(f"Validating bid {bid_data.get('bid_id', 'Unknown')} for risk and compliance")
            
            risk_score = 0.0
            issues = []
            
            # Get auction and participant information
            auction = self.db_session.query(Auction).filter(
                Auction.id == bid_data['auction_id']
            ).first()
            
            if not auction:
                return RiskCheckResult(
                    passed=False,
                    risk_level=RiskLevel.CRITICAL,
                    compliance_status=ComplianceStatus.BLOCKED,
                    reason="Auction not found",
                    risk_score=1.0
                )
            
            participant = self.db_session.query(Participant).filter(
                Participant.id == bid_data['participant_id']
            ).first()
            
            if not participant:
                return RiskCheckResult(
                    passed=False,
                    risk_level=RiskLevel.CRITICAL,
                    compliance_status=ComplianceStatus.BLOCKED,
                    reason="Participant not found",
                    risk_score=1.0
                )
            
            # Check auction status
            if auction.status.value not in ['BIDDING_OPEN']:
                issues.append("Auction is not accepting bids")
                risk_score += 0.4
            
            # Check participant eligibility
            if not self._check_participant_eligibility(participant, auction):
                issues.append("Participant not eligible for this auction")
                risk_score += 0.5
            
            # Check bid parameters
            if bid_data['bid_price'] <= 0:
                issues.append("Invalid bid price")
                risk_score += 0.3
            
            if bid_data['bid_quantity'] <= 0:
                issues.append("Invalid bid quantity")
                risk_score += 0.3
            
            if bid_data['bid_quantity'] < auction.minimum_lot_size:
                issues.append("Bid quantity below minimum lot size")
                risk_score += 0.2
            
            if bid_data['bid_quantity'] > auction.total_lot_size:
                issues.append("Bid quantity exceeds auction lot size")
                risk_score += 0.4
            
            # Check position limits
            position_check = await self._check_position_limits(
                participant.id, auction.instrument_id, bid_data['bid_quantity']
            )
            if not position_check['passed']:
                issues.append(position_check['reason'])
                risk_score += position_check['risk_score']
            
            # Check credit limits
            credit_check = await self._check_credit_limits(
                participant.id, bid_data['bid_price'], bid_data['bid_quantity']
            )
            if not credit_check['passed']:
                issues.append(credit_check['reason'])
                risk_score += credit_check['risk_score']
            
            # Check bid frequency
            frequency_check = await self._check_bid_frequency(
                participant.id, auction.id
            )
            if not frequency_check['passed']:
                issues.append(frequency_check['reason'])
                risk_score += frequency_check['risk_score']
            
            # Check for manipulation patterns
            manipulation_check = await self._check_manipulation_patterns(
                bid_data, auction, participant
            )
            if manipulation_check['detected']:
                issues.append(manipulation_check['description'])
                risk_score += manipulation_check['risk_score']
                
                # Create manipulation alert
                await self._create_manipulation_alert(
                    manipulation_check, bid_data, auction, participant
                )
            
            # Determine risk level and compliance status
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
                compliance_status = ComplianceStatus.BLOCKED
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
                compliance_status = ComplianceStatus.VIOLATION
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
                compliance_status = ComplianceStatus.WARNING
            else:
                risk_level = RiskLevel.LOW
                compliance_status = ComplianceStatus.COMPLIANT
            
            passed = risk_score < 0.6
            reason = "; ".join(issues) if issues else "All checks passed"
            
            result = RiskCheckResult(
                passed=passed,
                risk_level=risk_level,
                compliance_status=compliance_status,
                reason=reason,
                risk_score=risk_score,
                details={
                    'issues': issues,
                    'bid_id': bid_data.get('bid_id'),
                    'auction_id': bid_data['auction_id'],
                    'participant_id': bid_data['participant_id'],
                    'validation_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            self.logger.info(f"Bid validation completed: {result.compliance_status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating bid: {str(e)}")
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                compliance_status=ComplianceStatus.BLOCKED,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )
    
    def _check_participant_eligibility(self, participant: Participant, auction: Auction) -> bool:
        """Check if participant is eligible for the auction."""
        try:
            # Check compliance status
            if participant.compliance_status != "ACTIVE":
                return False
            
            # Check eligible auction types
            if auction.eligible_participants:
                participant_type = participant.participant_type.value
                if participant_type not in auction.eligible_participants:
                    return False
            
            # Check eligible instruments
            if participant.eligible_instruments:
                instrument_type = auction.instrument.bond_type.value
                if instrument_type not in participant.eligible_instruments:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking participant eligibility: {str(e)}")
            return False
    
    async def _check_position_limits(self, participant_id: int, instrument_id: int, 
                                    bid_quantity: Decimal) -> Dict[str, Any]:
        """Check position limits for the participant."""
        try:
            # Get current position
            current_position = self.db_session.query(func.sum(Allocation.allocation_quantity)).filter(
                and_(
                    Allocation.participant_id == participant_id,
                    Allocation.instrument_id == instrument_id
                )
            ).scalar() or Decimal('0')
            
            # Calculate new total position
            new_total_position = current_position + bid_quantity
            
            # Check against maximum position limit
            participant = self.db_session.query(Participant).filter(
                Participant.id == participant_id
            ).first()
            
            if participant and participant.maximum_position_limit:
                if new_total_position > participant.maximum_position_limit:
                    return {
                        'passed': False,
                        'reason': f"Position limit exceeded: {new_total_position} > {participant.maximum_position_limit}",
                        'risk_score': 0.4
                    }
            
            return {
                'passed': True,
                'reason': "Position limit check passed",
                'risk_score': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {str(e)}")
            return {
                'passed': False,
                'reason': f"Position limit check error: {str(e)}",
                'risk_score': 0.3
            }
    
    async def _check_credit_limits(self, participant_id: int, bid_price: Decimal, 
                                  bid_quantity: Decimal) -> Dict[str, Any]:
        """Check credit limits for the participant."""
        try:
            # Calculate bid value
            bid_value = bid_price * bid_quantity
            
            # Check against credit limit
            participant = self.db_session.query(Participant).filter(
                Participant.id == participant_id
            ).first()
            
            if participant and participant.credit_limit:
                if bid_value > participant.credit_limit:
                    return {
                        'passed': False,
                        'reason': f"Credit limit exceeded: {bid_value} > {participant.credit_limit}",
                        'risk_score': 0.5
                    }
            
            # Check against maximum credit exposure threshold
            if bid_value > self.risk_thresholds['max_credit_exposure']:
                return {
                    'passed': False,
                    'reason': f"Credit exposure threshold exceeded: {bid_value} > {self.risk_thresholds['max_credit_exposure']}",
                    'risk_score': 0.4
                }
            
            return {
                'passed': True,
                'reason': "Credit limit check passed",
                'risk_score': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking credit limits: {str(e)}")
            return {
                'passed': False,
                'reason': f"Credit limit check error: {str(e)}",
                'risk_score': 0.3
            }
    
    async def _check_bid_frequency(self, participant_id: int, auction_id: int) -> Dict[str, Any]:
        """Check bid frequency for the participant."""
        try:
            # Count bids in the last minute
            one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
            
            recent_bid_count = self.db_session.query(Bid).filter(
                and_(
                    Bid.participant_id == participant_id,
                    Bid.auction_id == auction_id,
                    Bid.submission_time >= one_minute_ago
                )
            ).count()
            
            if recent_bid_count > self.risk_thresholds['max_bid_frequency']:
                return {
                    'passed': False,
                    'reason': f"Bid frequency exceeded: {recent_bid_count} bids in last minute",
                    'risk_score': 0.3
                }
            
            return {
                'passed': True,
                'reason': "Bid frequency check passed",
                'risk_score': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking bid frequency: {str(e)}")
            return {
                'passed': False,
                'reason': f"Bid frequency check error: {str(e)}",
                'risk_score': 0.2
            }
    
    async def _check_manipulation_patterns(self, bid_data: Dict[str, Any], 
                                         auction: Auction, participant: Participant) -> Dict[str, Any]:
        """Check for potential market manipulation patterns."""
        try:
            manipulation_detected = False
            risk_score = 0.0
            description = ""
            
            # Check bid clustering
            clustering_score = self._check_bid_clustering(auction.id, bid_data)
            if clustering_score > self.manipulation_patterns['bid_clustering']['threshold']:
                manipulation_detected = True
                risk_score += 0.4
                description += "Bid clustering detected; "
            
            # Check price manipulation
            price_score = self._check_price_manipulation(auction.id, bid_data)
            if price_score > self.manipulation_patterns['price_manipulation']['threshold']:
                manipulation_detected = True
                risk_score += 0.5
                description += "Price manipulation detected; "
            
            # Check timing patterns
            timing_score = self._check_timing_patterns(auction.id, bid_data)
            if timing_score > self.manipulation_patterns['timing_patterns']['threshold']:
                manipulation_detected = True
                risk_score += 0.3
                description += "Suspicious timing patterns; "
            
            # Check quantity patterns
            quantity_score = self._check_quantity_patterns(auction.id, bid_data)
            if quantity_score > self.manipulation_patterns['quantity_patterns']['threshold']:
                manipulation_detected = True
                risk_score += 0.3
                description += "Unusual quantity patterns; "
            
            # Check participant concentration
            concentration_score = self._check_participant_concentration(auction.id, participant.id)
            if concentration_score > self.manipulation_patterns['participant_concentration']['threshold']:
                manipulation_detected = True
                risk_score += 0.4
                description += "High participant concentration; "
            
            return {
                'detected': manipulation_detected,
                'risk_score': risk_score,
                'description': description.strip() if description else "No manipulation detected"
            }
            
        except Exception as e:
            self.logger.error(f"Error checking manipulation patterns: {str(e)}")
            return {
                'detected': False,
                'risk_score': 0.0,
                'description': "Manipulation check error"
            }
    
    def _check_bid_clustering(self, auction_id: int, bid_data: Dict[str, Any]) -> float:
        """Check for bid clustering patterns."""
        try:
            # Get recent bids around the same price level
            price_tolerance = 0.001  # 1 basis point
            recent_bids = self.db_session.query(Bid).filter(
                and_(
                    Bid.auction_id == auction_id,
                    Bid.bid_price.between(
                        bid_data['bid_price'] - price_tolerance,
                        bid_data['bid_price'] + price_tolerance
                )
            ).all()
            
            if len(recent_bids) > 5:  # More than 5 bids at same price level
                return 0.9
            elif len(recent_bids) > 3:
                return 0.7
            elif len(recent_bids) > 1:
                return 0.5
            else:
                return 0.1
                
        except Exception as e:
            self.logger.error(f"Error checking bid clustering: {str(e)}")
            return 0.0
    
    def _check_price_manipulation(self, auction_id: int, bid_data: Dict[str, Any]) -> float:
        """Check for price manipulation patterns."""
        try:
            # Get all bids for the auction
            all_bids = self.db_session.query(Bid).filter(Bid.auction_id == auction_id).all()
            
            if not all_bids:
                return 0.0
            
            # Calculate price statistics
            prices = [bid.bid_price for bid in all_bids]
            avg_price = sum(prices) / len(prices)
            price_std = (sum((p - avg_price) ** 2 for p in prices) / len(prices)) ** 0.5
            
            # Check if bid price is significantly different from average
            if price_std > 0:
                price_deviation = abs(bid_data['bid_price'] - avg_price) / price_std
                if price_deviation > 3:  # More than 3 standard deviations
                    return 0.9
                elif price_deviation > 2:
                    return 0.7
                elif price_deviation > 1:
                    return 0.5
            
            return 0.1
            
        except Exception as e:
            self.logger.error(f"Error checking price manipulation: {str(e)}")
            return 0.0
    
    def _check_timing_patterns(self, auction_id: int, bid_data: Dict[str, Any]) -> float:
        """Check for suspicious bid timing patterns."""
        try:
            # Get recent bids from the same participant
            recent_bids = self.db_session.query(Bid).filter(
                and_(
                    Bid.auction_id == auction_id,
                    Bid.participant_id == bid_data['participant_id'],
                    Bid.submission_time >= datetime.utcnow() - timedelta(minutes=5)
                )
            ).order_by(Bid.submission_time).all()
            
            if len(recent_bids) > 3:
                # Check for regular intervals
                intervals = []
                for i in range(1, len(recent_bids)):
                    interval = (recent_bids[i].submission_time - recent_bids[i-1].submission_time).total_seconds()
                    intervals.append(interval)
                
                if len(intervals) > 1:
                    # Check if intervals are too regular (potential automation)
                    interval_std = (sum((i - sum(intervals)/len(intervals)) ** 2 for i in intervals) / len(intervals)) ** 0.5
                    if interval_std < 5:  # Very regular intervals
                        return 0.9
            
            return 0.1
            
        except Exception as e:
            self.logger.error(f"Error checking timing patterns: {str(e)}")
            return 0.0
    
    def _check_quantity_patterns(self, auction_id: int, bid_data: Dict[str, Any]) -> float:
        """Check for unusual quantity patterns."""
        try:
            # Get all bids for the auction
            all_bids = self.db_session.query(Bid).filter(Bid.auction_id == auction_id).all()
            
            if not all_bids:
                return 0.0
            
            # Calculate quantity statistics
            quantities = [bid.bid_quantity for bid in all_bids]
            avg_quantity = sum(quantities) / len(quantities)
            
            # Check if bid quantity is significantly different from average
            if avg_quantity > 0:
                quantity_ratio = bid_data['bid_quantity'] / avg_quantity
                if quantity_ratio > 5:  # 5x average
                    return 0.8
                elif quantity_ratio > 3:  # 3x average
                    return 0.6
                elif quantity_ratio < 0.1:  # 10% of average
                    return 0.5
            
            return 0.1
            
        except Exception as e:
            self.logger.error(f"Error checking quantity patterns: {str(e)}")
            return 0.0
    
    def _check_participant_concentration(self, auction_id: int, participant_id: int) -> float:
        """Check for high participant concentration."""
        try:
            # Get total bids for the auction
            total_bids = self.db_session.query(func.sum(Bid.bid_quantity)).filter(
                Bid.auction_id == auction_id
            ).scalar() or Decimal('0')
            
            if total_bids <= 0:
                return 0.0
            
            # Get participant's total bids
            participant_bids = self.db_session.query(func.sum(Bid.bid_quantity)).filter(
                and_(
                    Bid.auction_id == auction_id,
                    Bid.participant_id == participant_id
                )
            ).scalar() or Decimal('0')
            
            # Calculate concentration ratio
            concentration_ratio = float(participant_bids / total_bids)
            
            if concentration_ratio > 0.5:  # More than 50%
                return 0.9
            elif concentration_ratio > 0.3:  # More than 30%
                return 0.7
            elif concentration_ratio > 0.2:  # More than 20%
                return 0.5
            
            return 0.1
            
        except Exception as e:
            self.logger.error(f"Error checking participant concentration: {str(e)}")
            return 0.0
    
    async def _create_manipulation_alert(self, manipulation_check: Dict[str, Any], 
                                        bid_data: Dict[str, Any], auction: Auction, 
                                        participant: Participant):
        """Create manipulation alert for monitoring."""
        try:
            alert = ManipulationAlert(
                alert_id=f"ALERT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{participant.id}",
                alert_type="MANIPULATION_DETECTED",
                severity=RiskLevel.HIGH if manipulation_check['risk_score'] > 0.5 else RiskLevel.MEDIUM,
                description=manipulation_check['description'],
                participant_id=participant.id,
                auction_id=auction.id,
                bid_id=bid_data.get('bid_id'),
                timestamp=datetime.utcnow(),
                evidence={
                    'risk_score': manipulation_check['risk_score'],
                    'bid_data': bid_data,
                    'participant_info': {
                        'id': participant.id,
                        'code': participant.participant_code,
                        'type': participant.participant_type.value
                    },
                    'auction_info': {
                        'id': auction.id,
                        'code': auction.auction_code
                    }
                }
            )
            
            # In a real implementation, this would be stored in a database
            # and sent to monitoring systems
            self.logger.warning(f"Manipulation alert created: {alert.alert_id} - {alert.description}")
            
        except Exception as e:
            self.logger.error(f"Error creating manipulation alert: {str(e)}")
    
    async def get_risk_summary(self, auction_id: int) -> Dict[str, Any]:
        """Get comprehensive risk summary for an auction."""
        try:
            # Get auction information
            auction = self.db_session.query(Auction).filter(Auction.id == auction_id).first()
            if not auction:
                return {'error': 'Auction not found'}
            
            # Get bid statistics
            bids = self.db_session.query(Bid).filter(Bid.auction_id == auction_id).all()
            
            if not bids:
                return {'message': 'No bids found for auction'}
            
            # Calculate risk metrics
            total_bids = len(bids)
            total_quantity = sum(bid.bid_quantity for bid in bids)
            total_value = sum(bid.bid_price * bid.bid_quantity for bid in bids)
            
            # Participant concentration
            participant_bids = {}
            for bid in bids:
                if bid.participant_id not in participant_bids:
                    participant_bids[bid.participant_id] = {'quantity': Decimal('0'), 'value': Decimal('0'), 'count': 0}
                participant_bids[bid.participant_id]['quantity'] += bid.bid_quantity
                participant_bids[bid.participant_id]['value'] += bid.bid_price * bid.bid_quantity
                participant_bids[bid.participant_id]['count'] += 1
            
            # Calculate concentration ratios
            concentration_ratios = []
            for participant_id, data in participant_bids.items():
                quantity_ratio = float(data['quantity'] / total_quantity)
                value_ratio = float(data['value'] / total_value)
                concentration_ratios.append({
                    'participant_id': participant_id,
                    'quantity_ratio': quantity_ratio,
                    'value_ratio': value_ratio,
                    'bid_count': data['count']
                })
            
            # Sort by concentration
            concentration_ratios.sort(key=lambda x: x['quantity_ratio'], reverse=True)
            
            # Price dispersion
            prices = [bid.bid_price for bid in bids]
            if prices:
                price_range = max(prices) - min(prices)
                price_std = (sum((p - sum(prices)/len(prices)) ** 2 for p in prices) / len(prices)) ** 0.5
            else:
                price_range = price_std = 0
            
            risk_summary = {
                'auction_id': auction_id,
                'auction_code': auction.auction_code,
                'total_bids': total_bids,
                'total_quantity': float(total_quantity),
                'total_value': float(total_value),
                'unique_participants': len(participant_bids),
                'price_statistics': {
                    'range': float(price_range),
                    'standard_deviation': float(price_std),
                    'min_price': float(min(prices)) if prices else 0,
                    'max_price': float(max(prices)) if prices else 0
                },
                'concentration_analysis': {
                    'highest_concentration': concentration_ratios[0] if concentration_ratios else None,
                    'top_5_concentration': concentration_ratios[:5],
                    'herfindahl_index': sum(r['quantity_ratio'] ** 2 for r in concentration_ratios)
                },
                'risk_assessment': {
                    'concentration_risk': 'HIGH' if concentration_ratios[0]['quantity_ratio'] > 0.3 else 'MEDIUM' if concentration_ratios[0]['quantity_ratio'] > 0.2 else 'LOW',
                    'price_volatility': 'HIGH' if price_std > 0.02 else 'MEDIUM' if price_std > 0.01 else 'LOW',
                    'participation_risk': 'LOW' if len(participant_bids) > 10 else 'MEDIUM' if len(participant_bids) > 5 else 'HIGH'
                }
            }
            
            return risk_summary
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {str(e)}")
            return {'error': f'Failed to get risk summary: {str(e)}'}
