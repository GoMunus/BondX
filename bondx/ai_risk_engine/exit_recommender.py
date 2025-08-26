"""
Exit Recommender Service for BondX

This module provides intelligent exit route recommendations for bonds with
fill probability analysis and expected price deterioration estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
import warnings

from .liquidity_intelligence_service import (
    ExitPath, MarketMicrostructure, AuctionSignals, MarketMakerState
)

logger = logging.getLogger(__name__)

class ExitConstraint(Enum):
    """Types of constraints that may limit exit paths."""
    INVENTORY_LIMIT = "inventory_limit"
    WINDOW_CLOSED = "window_closed"
    LOT_SIZE = "lot_size"
    RATING_RESTRICTION = "rating_restriction"
    TIME_CONSTRAINT = "time_constraint"
    REGULATORY = "regulatory"

class ExitPriority(Enum):
    """Exit priority levels for recommendation ranking."""
    PRIMARY = "primary"      # Best option
    SECONDARY = "secondary"  # Good alternative
    FALLBACK = "fallback"    # Last resort
    UNAVAILABLE = "unavailable"  # Not available

@dataclass
class ExitPolicy:
    """Policy constraints for exit paths."""
    rating_caps: Dict[str, Dict[str, float]]  # rating -> {mm_cap, auction_cap, rfq_cap}
    tenor_restrictions: Dict[str, List[str]]  # tenor -> allowed_paths
    issuer_class_limits: Dict[str, Dict[str, float]]  # issuer_class -> limits
    mm_min_spread_bps: float = 5.0
    auction_cadence_hours: int = 4
    rfq_window_hours: int = 2
    tokenized_min_lot: float = 25000  # ₹25k minimum

@dataclass
class ExitRecommendation:
    """Complete exit recommendation for a specific path."""
    path: ExitPath
    priority: ExitPriority
    expected_price: float
    expected_spread_bps: float
    fill_probability: float  # 0-1
    expected_time_to_exit_minutes: float
    rationale: str
    constraints: List[ExitConstraint]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExitAnalysis:
    """Complete exit analysis for a bond."""
    isin: str
    as_of: datetime
    recommendations: List[ExitRecommendation]
    overall_confidence: float
    best_path: ExitPath
    expected_best_exit_time: float
    risk_warnings: List[str]
    data_freshness: str

class ExitRecommender:
    """
    Service for recommending optimal exit paths with fill probability analysis.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.exit_policy = self._load_default_policy()
        self.historical_fill_rates = {}  # Would load from database
        self._initialize_models()
        
    def _load_default_policy(self) -> ExitPolicy:
        """Load default exit policy configuration."""
        return ExitPolicy(
            rating_caps={
                'AAA': {'mm_cap': 1000000, 'auction_cap': 5000000, 'rfq_cap': 2000000},
                'AA': {'mm_cap': 800000, 'auction_cap': 4000000, 'rfq_cap': 1500000},
                'A': {'mm_cap': 600000, 'auction_cap': 3000000, 'rfq_cap': 1000000},
                'BBB': {'mm_cap': 400000, 'auction_cap': 2000000, 'rfq_cap': 800000},
                'BB': {'mm_cap': 200000, 'auction_cap': 1000000, 'rfq_cap': 500000},
                'B': {'mm_cap': 100000, 'auction_cap': 500000, 'rfq_cap': 250000},
                'C': {'mm_cap': 50000, 'auction_cap': 250000, 'rfq_cap': 100000},
                'D': {'mm_cap': 0, 'auction_cap': 0, 'rfq_cap': 0}
            },
            tenor_restrictions={
                '0-1Y': ['market_maker', 'auction', 'rfq_batch'],
                '1-3Y': ['market_maker', 'auction', 'rfq_batch', 'tokenized_p2p'],
                '3-5Y': ['market_maker', 'auction', 'rfq_batch', 'tokenized_p2p'],
                '5-10Y': ['market_maker', 'auction', 'rfq_batch', 'tokenized_p2p'],
                '10Y+': ['auction', 'rfq_batch', 'tokenized_p2p']
            },
            issuer_class_limits={
                'government': {'mm_cap': 5000000, 'auction_cap': 10000000, 'rfq_cap': 5000000},
                'psu': {'mm_cap': 3000000, 'auction_cap': 8000000, 'rfq_cap': 3000000},
                'corporate': {'mm_cap': 2000000, 'auction_cap': 5000000, 'rfq_cap': 2000000},
                'bank': {'mm_cap': 1500000, 'auction_cap': 4000000, 'rfq_cap': 1500000}
            }
        )
    
    def _initialize_models(self):
        """Initialize exit recommendation models."""
        logger.info("Initializing exit recommender models")
        
    def recommend_exit_paths(self, 
                            isin: str,
                            microstructure: MarketMicrostructure,
                            auction_signals: Optional[AuctionSignals] = None,
                            mm_state: Optional[MarketMakerState] = None,
                            bond_metadata: Dict[str, Any] = None,
                            trade_size: float = 100000) -> ExitAnalysis:
        """
        Recommend optimal exit paths for a bond.
        
        Args:
            isin: Bond ISIN
            microstructure: Market microstructure data
            auction_signals: Optional auction telemetry
            mm_state: Optional market maker state
            bond_metadata: Bond characteristics (rating, tenor, issuer class)
            trade_size: Size of position to exit
            
        Returns:
            Complete exit analysis with ranked recommendations
        """
        try:
            # Initialize bond metadata with defaults
            bond_metadata = bond_metadata or {}
            rating = bond_metadata.get('rating', 'BBB')
            tenor = bond_metadata.get('tenor', '3-5Y')
            issuer_class = bond_metadata.get('issuer_class', 'corporate')
            
            # Analyze each exit path
            recommendations = []
            
            # Market Maker analysis
            mm_rec = self._analyze_market_maker_path(
                microstructure, mm_state, bond_metadata, trade_size
            )
            if mm_rec:
                recommendations.append(mm_rec)
            
            # Auction analysis
            auction_rec = self._analyze_auction_path(
                microstructure, auction_signals, bond_metadata, trade_size
            )
            if auction_rec:
                recommendations.append(auction_rec)
            
            # RFQ Batch analysis
            rfq_rec = self._analyze_rfq_path(
                microstructure, bond_metadata, trade_size
            )
            if rfq_rec:
                recommendations.append(rfq_rec)
            
            # Tokenized P2P analysis
            p2p_rec = self._analyze_tokenized_p2p_path(
                microstructure, bond_metadata, trade_size
            )
            if p2p_rec:
                recommendations.append(p2p_rec)
            
            # Rank recommendations by fill probability and time
            recommendations.sort(
                key=lambda x: (x.fill_probability, -x.expected_time_to_exit_minutes),
                reverse=True
            )
            
            # Assign priorities
            for i, rec in enumerate(recommendations):
                if i == 0:
                    rec.priority = ExitPriority.PRIMARY
                elif i == 1:
                    rec.priority = ExitPriority.SECONDARY
                else:
                    rec.priority = ExitPriority.FALLBACK
            
            # Create exit analysis
            analysis = ExitAnalysis(
                isin=isin,
                as_of=datetime.now(),
                recommendations=recommendations,
                overall_confidence=self._calculate_overall_confidence(recommendations),
                best_path=recommendations[0].path if recommendations else None,
                expected_best_exit_time=recommendations[0].expected_time_to_exit_minutes if recommendations else 0,
                risk_warnings=self._generate_risk_warnings(recommendations, bond_metadata),
                data_freshness=self._assess_data_freshness(microstructure.timestamp)
            )
            
            logger.info(f"Generated exit analysis for {isin}: {len(recommendations)} paths, best={analysis.best_path}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error recommending exit paths for {isin}: {e}")
            raise
    
    def _analyze_market_maker_path(self, 
                                  microstructure: MarketMicrostructure,
                                  mm_state: Optional[MarketMakerState],
                                  bond_metadata: Dict[str, Any],
                                  trade_size: float) -> Optional[ExitRecommendation]:
        """Analyze market maker exit path."""
        try:
            if not mm_state or not mm_state.mm_online:
                return None
            
            # Check inventory constraints
            rating = bond_metadata.get('rating', 'BBB')
            issuer_class = bond_metadata.get('issuer_class', 'corporate')
            
            mm_cap = self.exit_policy.rating_caps.get(rating, {}).get('mm_cap', 100000)
            if trade_size > mm_cap:
                return None
            
            # Calculate fill probability
            fill_prob = self._calculate_mm_fill_probability(mm_state, microstructure, trade_size)
            
            # Calculate expected price and spread
            expected_price = microstructure.mid_price
            expected_spread = mm_state.last_quote_spread_bps
            
            # Estimate time to exit
            expected_tte = self._estimate_mm_exit_time(mm_state, trade_size)
            
            # Generate rationale
            rationale = f"Market maker online with {mm_state.quotes_last_24h} quotes in 24h. " \
                       f"Current spread: {expected_spread:.1f}bps. " \
                       f"Fill probability: {fill_prob:.1%}"
            
            # Check constraints
            constraints = []
            if mm_state.mm_inventory_band[2] < trade_size:
                constraints.append(ExitConstraint.INVENTORY_LIMIT)
            
            return ExitRecommendation(
                path=ExitPath.MARKET_MAKER,
                priority=ExitPriority.PRIMARY,  # Will be updated later
                expected_price=expected_price,
                expected_spread_bps=expected_spread,
                fill_probability=fill_prob,
                expected_time_to_exit_minutes=expected_tte,
                rationale=rationale,
                constraints=constraints,
                confidence=0.85,
                metadata={
                    'mm_online': True,
                    'quotes_24h': mm_state.quotes_last_24h,
                    'inventory_band': mm_state.mm_inventory_band
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market maker path: {e}")
            return None
    
    def _analyze_auction_path(self,
                              microstructure: MarketMicrostructure,
                              auction_signals: Optional[AuctionSignals],
                              bond_metadata: Dict[str, Any],
                              trade_size: float) -> Optional[ExitRecommendation]:
        """Analyze auction exit path."""
        try:
            if not auction_signals:
                return None
            
            # Check rating and size constraints
            rating = bond_metadata.get('rating', 'BBB')
            auction_cap = self.exit_policy.rating_caps.get(rating, {}).get('auction_cap', 1000000)
            if trade_size > auction_cap:
                return None
            
            # Calculate fill probability based on demand
            fill_prob = self._calculate_auction_fill_probability(auction_signals, trade_size)
            
            # Calculate expected price
            expected_price = auction_signals.clearing_price_estimate
            expected_spread = microstructure.spread_bps
            
            # Estimate time to exit (next auction window)
            time_to_window = (auction_signals.next_window - datetime.now()).total_seconds() / 60
            expected_tte = max(0, time_to_window) + 120  # 2 hours for auction + settlement
            
            # Generate rationale
            rationale = f"Next auction window: {auction_signals.next_window.strftime('%H:%M')}. " \
                       f"Current demand: {auction_signals.bids_count} bids. " \
                       f"Fill probability: {fill_prob:.1%}"
            
            # Check constraints
            constraints = []
            if time_to_window > 1440:  # >24 hours
                constraints.append(ExitConstraint.TIME_CONSTRAINT)
            
            return ExitRecommendation(
                path=ExitPath.AUCTION,
                priority=ExitPriority.PRIMARY,  # Will be updated later
                expected_price=expected_price,
                expected_spread_bps=expected_spread,
                fill_probability=fill_prob,
                expected_time_to_exit_minutes=expected_tte,
                rationale=rationale,
                constraints=constraints,
                confidence=0.75,
                metadata={
                    'next_window': auction_signals.next_window,
                    'bids_count': auction_signals.bids_count,
                    'lots_offered': auction_signals.lots_offered
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing auction path: {e}")
            return None
    
    def _analyze_rfq_path(self,
                          microstructure: MarketMicrostructure,
                          bond_metadata: Dict[str, Any],
                          trade_size: float) -> Optional[ExitRecommendation]:
        """Analyze RFQ batch exit path."""
        try:
            # Check rating and size constraints
            rating = bond_metadata.get('rating', 'BBB')
            rfq_cap = self.exit_policy.rating_caps.get(rating, {}).get('rfq_cap', 500000)
            if trade_size > rfq_cap:
                return None
            
            # Calculate fill probability (based on historical data)
            fill_prob = self._calculate_rfq_fill_probability(bond_metadata, trade_size)
            
            # Calculate expected price and spread
            expected_price = microstructure.mid_price
            expected_spread = microstructure.spread_bps * 1.2  # RFQ typically wider
            
            # Estimate time to exit (next RFQ window)
            next_rfq = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
            if datetime.now().hour >= 10:
                next_rfq += timedelta(days=1)
            
            time_to_window = (next_rfq - datetime.now()).total_seconds() / 60
            expected_tte = max(0, time_to_window) + 60  # 1 hour for batch processing
            
            # Generate rationale
            rationale = f"Next RFQ window: {next_rfq.strftime('%H:%M')}. " \
                       f"Batch processing with {fill_prob:.1%} fill probability. " \
                       f"Expected spread: {expected_spread:.1f}bps"
            
            # Check constraints
            constraints = []
            if time_to_window > 1440:  # >24 hours
                constraints.append(ExitConstraint.TIME_CONSTRAINT)
            
            return ExitRecommendation(
                path=ExitPath.RFQ_BATCH,
                priority=ExitPriority.PRIMARY,  # Will be updated later
                expected_price=expected_price,
                expected_spread_bps=expected_spread,
                fill_probability=fill_prob,
                expected_time_to_exit_minutes=expected_tte,
                rationale=rationale,
                constraints=constraints,
                confidence=0.70,
                metadata={
                    'next_window': next_rfq,
                    'batch_size': trade_size
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing RFQ path: {e}")
            return None
    
    def _analyze_tokenized_p2p_path(self,
                                   microstructure: MarketMicrostructure,
                                   bond_metadata: Dict[str, Any],
                                   trade_size: float) -> Optional[ExitRecommendation]:
        """Analyze tokenized P2P exit path."""
        try:
            # Check minimum lot size
            if trade_size < self.exit_policy.tokenized_min_lot:
                return None
            
            # Check tenor restrictions
            tenor = bond_metadata.get('tenor', '3-5Y')
            allowed_paths = self.exit_policy.tenor_restrictions.get(tenor, [])
            if 'tokenized_p2p' not in allowed_paths:
                return None
            
            # Calculate fill probability (conservative estimate)
            fill_prob = self._calculate_p2p_fill_probability(bond_metadata, trade_size)
            
            # Calculate expected price and spread
            expected_price = microstructure.mid_price * 0.995  # Slight discount for P2P
            expected_spread = microstructure.spread_bps * 1.5  # P2P typically wider
            
            # Estimate time to exit (conservative)
            expected_tte = 240  # 4 hours for P2P matching
            
            # Generate rationale
            rationale = f"Tokenized P2P available for {tenor} bonds. " \
                       f"Conservative fill probability: {fill_prob:.1%}. " \
                       f"Expected discount: 0.5% from mid"
            
            # Check constraints
            constraints = []
            if trade_size < 100000:  # <₹1L
                constraints.append(ExitConstraint.LOT_SIZE)
            
            return ExitRecommendation(
                path=ExitPath.TOKENIZED_P2P,
                priority=ExitPriority.PRIMARY,  # Will be updated later
                expected_price=expected_price,
                expected_spread_bps=expected_spread,
                fill_probability=fill_prob,
                expected_time_to_exit_minutes=expected_tte,
                rationale=rationale,
                constraints=constraints,
                confidence=0.60,
                metadata={
                    'tenor': tenor,
                    'min_lot': self.exit_policy.tokenized_min_lot
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing tokenized P2P path: {e}")
            return None
    
    def _calculate_mm_fill_probability(self, 
                                     mm_state: MarketMakerState,
                                     microstructure: MarketMicrostructure,
                                     trade_size: float) -> float:
        """Calculate market maker fill probability."""
        base_prob = 0.8
        
        # Adjust for quote frequency
        if mm_state.quotes_last_24h >= 20:
            base_prob += 0.1
        elif mm_state.quotes_last_24h < 5:
            base_prob -= 0.2
        
        # Adjust for spread quality
        if mm_state.last_quote_spread_bps < 20:
            base_prob += 0.1
        elif mm_state.last_quote_spread_bps > 100:
            base_prob -= 0.2
        
        # Adjust for trade size vs inventory
        low, target, high = mm_state.mm_inventory_band
        if trade_size <= target:
            base_prob += 0.1
        elif trade_size > high:
            base_prob -= 0.3
        
        return np.clip(base_prob, 0.1, 0.95)
    
    def _calculate_auction_fill_probability(self,
                                          auction_signals: AuctionSignals,
                                          trade_size: float) -> float:
        """Calculate auction fill probability."""
        base_prob = 0.6
        
        # Adjust for demand (more bids = higher probability)
        if auction_signals.bids_count >= 10:
            base_prob += 0.2
        elif auction_signals.bids_count >= 5:
            base_prob += 0.1
        elif auction_signals.bids_count < 2:
            base_prob -= 0.2
        
        # Adjust for demand curve slope
        if len(auction_signals.demand_curve_points) >= 2:
            prices, quantities = zip(*auction_signals.demand_curve_points)
            demand_slope = np.polyfit(quantities, prices, 1)[0]
            if demand_slope > 0:  # Upward sloping (good demand)
                base_prob += 0.1
            else:  # Downward sloping (weak demand)
                base_prob -= 0.1
        
        return np.clip(base_prob, 0.2, 0.9)
    
    def _calculate_rfq_fill_probability(self,
                                      bond_metadata: Dict[str, Any],
                                      trade_size: float) -> float:
        """Calculate RFQ fill probability."""
        base_prob = 0.5
        
        # Adjust for rating (higher rating = higher probability)
        rating = bond_metadata.get('rating', 'BBB')
        rating_adjustments = {
            'AAA': 0.2, 'AA': 0.15, 'A': 0.1, 'BBB': 0.0,
            'BB': -0.1, 'B': -0.2, 'C': -0.3, 'D': -0.4
        }
        base_prob += rating_adjustments.get(rating, 0.0)
        
        # Adjust for trade size
        if trade_size <= 500000:  # ≤₹5L
            base_prob += 0.1
        elif trade_size > 2000000:  # >₹20L
            base_prob -= 0.1
        
        return np.clip(base_prob, 0.2, 0.8)
    
    def _calculate_p2p_fill_probability(self,
                                       bond_metadata: Dict[str, Any],
                                       trade_size: float) -> float:
        """Calculate tokenized P2P fill probability."""
        base_prob = 0.4  # Conservative base
        
        # Adjust for rating
        rating = bond_metadata.get('rating', 'BBB')
        if rating in ['AAA', 'AA', 'A']:
            base_prob += 0.1
        
        # Adjust for trade size
        if trade_size >= 1000000:  # ≥₹10L
            base_prob += 0.1
        
        return np.clip(base_prob, 0.2, 0.7)
    
    def _estimate_mm_exit_time(self, mm_state: MarketMakerState, trade_size: float) -> float:
        """Estimate market maker exit time."""
        base_time = 15  # Base 15 minutes
        
        # Adjust for trade size
        if trade_size > 1000000:  # >₹10L
            base_time += 15
        
        # Adjust for quote frequency
        if mm_state.quotes_last_24h < 10:
            base_time += 30
        
        return base_time
    
    def _calculate_overall_confidence(self, recommendations: List[ExitRecommendation]) -> float:
        """Calculate overall confidence in exit analysis."""
        if not recommendations:
            return 0.0
        
        # Weight by priority
        priority_weights = {
            ExitPriority.PRIMARY: 0.5,
            ExitPriority.SECONDARY: 0.3,
            ExitPriority.FALLBACK: 0.2
        }
        
        weighted_confidence = sum(
            rec.confidence * priority_weights.get(rec.priority, 0.1)
            for rec in recommendations
        )
        
        return min(1.0, weighted_confidence)
    
    def _generate_risk_warnings(self, 
                               recommendations: List[ExitRecommendation],
                               bond_metadata: Dict[str, Any]) -> List[str]:
        """Generate risk warnings for exit analysis."""
        warnings = []
        
        # Check for high-risk scenarios
        if not recommendations:
            warnings.append("No viable exit paths available")
            return warnings
        
        # Check primary path constraints
        primary = recommendations[0]
        if primary.constraints:
            constraint_names = [c.value.replace('_', ' ').title() for c in primary.constraints]
            warnings.append(f"Primary exit path has constraints: {', '.join(constraint_names)}")
        
        # Check low fill probability
        if primary.fill_probability < 0.5:
            warnings.append(f"Low fill probability ({primary.fill_probability:.1%}) for primary exit path")
        
        # Check long exit time
        if primary.expected_time_to_exit_minutes > 240:  # >4 hours
            warnings.append(f"Long expected exit time ({primary.expected_time_to_exit_minutes:.0f} minutes)")
        
        # Check rating-based warnings
        rating = bond_metadata.get('rating', 'BBB')
        if rating in ['BB', 'B', 'C', 'D']:
            warnings.append(f"High credit risk rating ({rating}) may limit exit options")
        
        return warnings
    
    def _assess_data_freshness(self, timestamp: datetime) -> str:
        """Assess the freshness of market data."""
        age_minutes = (datetime.now() - timestamp).total_seconds() / 60
        
        if age_minutes < 1:
            return "real_time"
        elif age_minutes < 5:
            return "fresh"
        elif age_minutes < 15:
            return "recent"
        elif age_minutes < 60:
            return "stale"
        else:
            return "outdated"
