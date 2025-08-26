"""
Liquidity Intelligence Service for BondX

This module provides real-time liquidity profiling with actionable exit pathways.
It computes liquidity scores, depth metrics, spread analysis, and expected time-to-exit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class LiquidityLevel(Enum):
    """Liquidity level classifications."""
    EXCELLENT = "excellent"      # 80-100
    GOOD = "good"                # 60-79
    MODERATE = "moderate"        # 40-59
    POOR = "poor"                # 20-39
    ILLIQUID = "illiquid"        # 0-19

class ExitPath(Enum):
    """Available exit paths for bonds."""
    MARKET_MAKER = "market_maker"
    AUCTION = "auction"
    RFQ_BATCH = "rfq_batch"
    TOKENIZED_P2P = "tokenized_p2p"

@dataclass
class MarketMicrostructure:
    """Real-time market microstructure data."""
    timestamp: datetime
    isin: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    l2_depth_qty: float
    l2_levels: int
    trades_count: int
    vwap: float
    volume_face: float
    time_since_last_trade_s: int
    mid_price: float = field(init=False)
    spread_bps: float = field(init=False)
    
    def __post_init__(self):
        self.mid_price = (self.bid + self.ask) / 2
        self.spread_bps = ((self.ask - self.bid) / self.mid_price) * 10000

@dataclass
class AuctionSignals:
    """Auction telemetry data."""
    timestamp: datetime
    isin: str
    auction_id: str
    lots_offered: int
    bids_count: int
    demand_curve_points: List[Tuple[float, float]]  # (price/yield, qty)
    clearing_price_estimate: float
    next_window: datetime

@dataclass
class MarketMakerState:
    """Market maker telemetry data."""
    timestamp: datetime
    isin: str
    mm_online: bool
    mm_inventory_band: Tuple[float, float, float]  # (low, target, high)
    mm_min_spread_bps: float
    last_quote_spread_bps: float
    quotes_last_24h: int

@dataclass
class LiquidityProfile:
    """Comprehensive liquidity profile for a bond."""
    isin: str
    as_of: datetime
    liquidity_index: float  # 0-100
    spread_bps: float
    depth_score: float  # 0-100
    turnover_rank: float
    time_since_last_trade_s: int
    expected_time_to_exit_minutes: float
    liquidity_level: LiquidityLevel
    confidence: float
    data_freshness: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExitPathAnalysis:
    """Analysis of a specific exit path."""
    path: ExitPath
    expected_price: float
    expected_spread_bps: float
    fill_probability: float  # 0-1
    expected_time_to_exit_minutes: float
    rationale: str
    constraints: List[str]
    confidence: float

class LiquidityIntelligenceService:
    """
    Service for computing real-time liquidity intelligence and exit pathway analysis.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.liquidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.tte_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize and train the liquidity and TTE models."""
        # This would typically load pre-trained models
        # For now, we'll use placeholder models
        logger.info("Initializing liquidity intelligence models")
        
    def compute_liquidity_index(self, microstructure: MarketMicrostructure, 
                               auction_signals: Optional[AuctionSignals] = None,
                               mm_state: Optional[MarketMakerState] = None) -> float:
        """
        Compute liquidity index (0-100) based on multiple factors.
        
        Args:
            microstructure: Market microstructure data
            auction_signals: Optional auction telemetry
            mm_state: Optional market maker state
            
        Returns:
            Liquidity index score (0-100)
        """
        try:
            # Base liquidity score from microstructure
            base_score = self._compute_base_liquidity_score(microstructure)
            
            # Adjust for auction signals if available
            if auction_signals:
                auction_adjustment = self._compute_auction_adjustment(auction_signals)
                base_score += auction_adjustment
            
            # Adjust for market maker availability
            if mm_state:
                mm_adjustment = self._compute_mm_adjustment(mm_state)
                base_score += mm_adjustment
            
            # Normalize to 0-100 range
            liquidity_index = np.clip(base_score, 0, 100)
            
            logger.debug(f"Computed liquidity index {liquidity_index:.2f} for {microstructure.isin}")
            return liquidity_index
            
        except Exception as e:
            logger.error(f"Error computing liquidity index: {e}")
            return 50.0  # Default to moderate liquidity
    
    def _compute_base_liquidity_score(self, microstructure: MarketMicrostructure) -> float:
        """Compute base liquidity score from microstructure data."""
        # Normalize spread (lower is better)
        spread_score = max(0, 100 - (microstructure.spread_bps * 2))
        
        # Depth score based on available liquidity
        depth_score = min(100, (microstructure.l2_depth_qty / microstructure.volume_face) * 1000)
        
        # Turnover score based on recent activity
        turnover_score = max(0, 100 - (microstructure.time_since_last_trade_s / 3600))
        
        # Volume score
        volume_score = min(100, (microstructure.volume_face / 1000000) * 20)
        
        # Weighted combination
        weights = [0.3, 0.25, 0.25, 0.2]  # spread, depth, turnover, volume
        base_score = (spread_score * weights[0] + 
                     depth_score * weights[1] + 
                     turnover_score * weights[2] + 
                     volume_score * weights[3])
        
        return base_score
    
    def _compute_auction_adjustment(self, auction_signals: AuctionSignals) -> float:
        """Compute liquidity adjustment based on auction signals."""
        # More bids indicate higher demand/liquidity
        bid_adjustment = min(20, auction_signals.bids_count * 2)
        
        # Demand curve slope adjustment
        if len(auction_signals.demand_curve_points) >= 2:
            prices, quantities = zip(*auction_signals.demand_curve_points)
            demand_slope = np.polyfit(quantities, prices, 1)[0]
            slope_adjustment = max(-10, min(10, demand_slope * 100))
        else:
            slope_adjustment = 0
        
        return bid_adjustment + slope_adjustment
    
    def _compute_mm_adjustment(self, mm_state: MarketMakerState) -> float:
        """Compute liquidity adjustment based on market maker state."""
        if not mm_state.mm_online:
            return -30  # Significant penalty for offline MM
        
        # Spread quality adjustment
        spread_adjustment = max(-15, min(15, (50 - mm_state.last_quote_spread_bps) / 2))
        
        # Quote frequency adjustment
        quote_adjustment = min(10, mm_state.quotes_last_24h / 10)
        
        return spread_adjustment + quote_adjustment
    
    def estimate_time_to_exit(self, liquidity_index: float, 
                             exit_path: ExitPath,
                             trade_size: float,
                             market_conditions: Dict[str, Any]) -> float:
        """
        Estimate time to exit for a given bond and exit path.
        
        Args:
            liquidity_index: Liquidity index (0-100)
            exit_path: Chosen exit path
            trade_size: Size of position to exit
            market_conditions: Current market conditions
            
        Returns:
            Estimated time to exit in minutes
        """
        try:
            # Base TTE from liquidity index
            base_tte = self._get_base_tte(liquidity_index)
            
            # Path-specific adjustments
            path_adjustment = self._get_path_adjustment(exit_path, market_conditions)
            
            # Size adjustment
            size_adjustment = self._get_size_adjustment(trade_size, liquidity_index)
            
            # Market condition adjustment
            market_adjustment = self._get_market_adjustment(market_conditions)
            
            # Calculate final TTE
            final_tte = base_tte + path_adjustment + size_adjustment + market_adjustment
            
            # Ensure reasonable bounds
            final_tte = max(1, min(1440, final_tte))  # 1 minute to 24 hours
            
            logger.debug(f"Estimated TTE: {final_tte:.1f} minutes for {exit_path.value}")
            return final_tte
            
        except Exception as e:
            logger.error(f"Error estimating time to exit: {e}")
            return 60.0  # Default to 1 hour
    
    def _get_base_tte(self, liquidity_index: float) -> float:
        """Get base time to exit from liquidity index."""
        if liquidity_index >= 80:
            return 15  # Excellent liquidity: 15 minutes
        elif liquidity_index >= 60:
            return 30  # Good liquidity: 30 minutes
        elif liquidity_index >= 40:
            return 60  # Moderate liquidity: 1 hour
        elif liquidity_index >= 20:
            return 180  # Poor liquidity: 3 hours
        else:
            return 480  # Illiquid: 8 hours
    
    def _get_path_adjustment(self, exit_path: ExitPath, 
                            market_conditions: Dict[str, Any]) -> float:
        """Get time adjustment for specific exit path."""
        adjustments = {
            ExitPath.MARKET_MAKER: 0,      # Fastest
            ExitPath.AUCTION: 120,         # +2 hours for auction window
            ExitPath.RFQ_BATCH: 60,        # +1 hour for batch processing
            ExitPath.TOKENIZED_P2P: 240    # +4 hours for P2P matching
        }
        
        base_adjustment = adjustments.get(exit_path, 0)
        
        # Additional adjustments based on market conditions
        if exit_path == ExitPath.AUCTION:
            # Check if auction window is open
            if market_conditions.get('auction_window_open', False):
                base_adjustment -= 60  # Reduce if window is open
        
        return base_adjustment
    
    def _get_size_adjustment(self, trade_size: float, liquidity_index: float) -> float:
        """Get time adjustment based on trade size."""
        # Larger trades take longer to execute
        size_factor = min(2.0, trade_size / 1000000)  # Normalize to millions
        
        if liquidity_index >= 60:
            return size_factor * 15  # Good liquidity: smaller impact
        else:
            return size_factor * 45  # Poor liquidity: larger impact
    
    def _get_market_adjustment(self, market_conditions: Dict[str, Any]) -> float:
        """Get time adjustment based on market conditions."""
        adjustment = 0
        
        # Volatility adjustment
        volatility = market_conditions.get('volatility', 0)
        if volatility > 0.3:  # High volatility
            adjustment += 30
        
        # Market hours adjustment
        market_open = market_conditions.get('market_open', True)
        if not market_open:
            adjustment += 480  # +8 hours if market closed
        
        # News/events adjustment
        if market_conditions.get('high_impact_news', False):
            adjustment += 60
        
        return adjustment
    
    def compute_depth_score(self, microstructure: MarketMicrostructure) -> float:
        """Compute depth score (0-100) based on available liquidity."""
        try:
            # Normalize depth by volume
            depth_ratio = microstructure.l2_depth_qty / max(microstructure.volume_face, 1)
            
            # Convert to score (0-100)
            depth_score = min(100, depth_ratio * 1000)
            
            # Adjust for number of levels
            level_adjustment = min(20, microstructure.l2_levels * 2)
            
            final_score = min(100, depth_score + level_adjustment)
            return final_score
            
        except Exception as e:
            logger.error(f"Error computing depth score: {e}")
            return 50.0
    
    def compute_turnover_rank(self, microstructure: MarketMicrostructure, 
                             historical_data: pd.DataFrame) -> float:
        """Compute turnover rank based on historical data."""
        try:
            if historical_data.empty:
                return 50.0
            
            # Calculate turnover ratio
            turnover_ratio = microstructure.volume_face / max(microstructure.volume_face, 1)
            
            # Rank against historical distribution
            if 'turnover_ratio' in historical_data.columns:
                rank = stats.percentileofscore(
                    historical_data['turnover_ratio'], 
                    turnover_ratio
                )
                return rank
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error computing turnover rank: {e}")
            return 50.0
    
    def create_liquidity_profile(self, isin: str, 
                                microstructure: MarketMicrostructure,
                                auction_signals: Optional[AuctionSignals] = None,
                                mm_state: Optional[MarketMakerState] = None,
                                historical_data: Optional[pd.DataFrame] = None) -> LiquidityProfile:
        """
        Create comprehensive liquidity profile for a bond.
        
        Args:
            isin: Bond ISIN
            microstructure: Market microstructure data
            auction_signals: Optional auction telemetry
            mm_state: Optional market maker state
            historical_data: Optional historical data for ranking
            
        Returns:
            Complete liquidity profile
        """
        try:
            # Compute liquidity index
            liquidity_index = self.compute_liquidity_index(
                microstructure, auction_signals, mm_state
            )
            
            # Compute depth score
            depth_score = self.compute_depth_score(microstructure)
            
            # Compute turnover rank
            turnover_rank = self.compute_turnover_rank(microstructure, historical_data or pd.DataFrame())
            
            # Determine liquidity level
            liquidity_level = self._classify_liquidity_level(liquidity_index)
            
            # Estimate time to exit (default to market maker path)
            expected_tte = self.estimate_time_to_exit(
                liquidity_index, 
                ExitPath.MARKET_MAKER,
                microstructure.volume_face,
                {'market_open': True, 'volatility': 0.1}
            )
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(microstructure, auction_signals, mm_state)
            
            # Determine data freshness
            data_freshness = self._assess_data_freshness(microstructure.timestamp)
            
            profile = LiquidityProfile(
                isin=isin,
                as_of=datetime.now(),
                liquidity_index=liquidity_index,
                spread_bps=microstructure.spread_bps,
                depth_score=depth_score,
                turnover_rank=turnover_rank,
                time_since_last_trade_s=microstructure.time_since_last_trade_s,
                expected_time_to_exit_minutes=expected_tte,
                liquidity_level=liquidity_level,
                confidence=confidence,
                data_freshness=data_freshness,
                metadata={
                    'bid': microstructure.bid,
                    'ask': microstructure.ask,
                    'mid_price': microstructure.mid_price,
                    'volume_face': microstructure.volume_face,
                    'l2_levels': microstructure.l2_levels
                }
            )
            
            logger.info(f"Created liquidity profile for {isin}: index={liquidity_index:.1f}, level={liquidity_level.value}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating liquidity profile for {isin}: {e}")
            raise
    
    def _classify_liquidity_level(self, liquidity_index: float) -> LiquidityLevel:
        """Classify liquidity level based on index score."""
        if liquidity_index >= 80:
            return LiquidityLevel.EXCELLENT
        elif liquidity_index >= 60:
            return LiquidityLevel.GOOD
        elif liquidity_index >= 40:
            return LiquidityLevel.MODERATE
        elif liquidity_index >= 20:
            return LiquidityLevel.POOR
        else:
            return LiquidityLevel.ILLIQUID
    
    def _calculate_confidence(self, microstructure: MarketMicrostructure,
                            auction_signals: Optional[AuctionSignals],
                            mm_state: Optional[MarketMakerState]) -> float:
        """Calculate confidence in liquidity assessment."""
        confidence = 0.8  # Base confidence
        
        # Data freshness penalty
        if microstructure.time_since_last_trade_s > 3600:  # >1 hour
            confidence -= 0.2
        
        # Data availability bonus
        if auction_signals:
            confidence += 0.1
        if mm_state:
            confidence += 0.1
        
        # Spread quality indicator
        if microstructure.spread_bps < 50:  # Tight spread
            confidence += 0.1
        
        return np.clip(confidence, 0.1, 1.0)
    
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
