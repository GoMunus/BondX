"""
Liquidity and Market Impact Models for BondX Risk Management System

This module provides first-order liquidity analytics informing pre-trade checks and stress.

Features:
- Liquidity score per instrument using bid-ask, turnover, depth proxies
- Market impact models: linear and square-root impact
- Slicing strategies and cost estimator (bps) by notional
- Integration with pre-trade risk checks and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import warnings
from scipy import stats
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ImpactModel(Enum):
    """Available market impact models."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    POWER_LAW = "power_law"
    ADAPTIVE = "adaptive"

class LiquidityScoreMethod(Enum):
    """Available liquidity scoring methods."""
    COMPOSITE = "composite"
    BID_ASK_ONLY = "bid_ask_only"
    TURNOVER_ONLY = "turnover_only"
    DEPTH_ONLY = "depth_only"

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for an instrument."""
    bid_ask_spread: float  # In basis points
    turnover_ratio: float  # Daily turnover / outstanding amount
    market_depth: float  # Available liquidity at bid/ask
    days_since_last_trade: int
    issue_size: float  # Outstanding notional
    average_trade_size: float  # Average trade size
    volatility: float  # Price volatility
    rating: str  # Credit rating
    sector: str  # Industry sector
    maturity_tenor: float  # Years to maturity

@dataclass
class LiquidityConfig:
    """Configuration for liquidity modeling."""
    impact_model: ImpactModel = ImpactModel.SQUARE_ROOT
    liquidity_score_method: LiquidityScoreMethod = LiquidityScoreMethod.COMPOSITE
    bid_ask_weight: float = 0.4
    turnover_weight: float = 0.3
    depth_weight: float = 0.2
    volatility_weight: float = 0.1
    min_liquidity_score: float = 0.0
    max_liquidity_score: float = 100.0
    impact_calibration_factor: float = 1.0
    confidence_level: float = 0.95
    enable_stress_testing: bool = True
    stress_multiplier: float = 2.0
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class LiquidityScore:
    """Liquidity score result."""
    score: float  # 0-100 scale
    breakdown: Dict[str, float]  # Component scores
    confidence_band: Tuple[float, float]  # Confidence interval
    calculation_date: date
    metadata: Dict[str, Union[str, float, int]]

@dataclass
class MarketImpact:
    """Market impact estimation result."""
    expected_cost_bps: float  # Expected cost in basis points
    confidence_band: Tuple[float, float]  # Confidence interval
    impact_model: ImpactModel
    trade_size: float
    liquidity_score: float
    calculation_date: date
    metadata: Dict[str, Union[str, float, int]]

@dataclass
class SlicingStrategy:
    """Optimal trade slicing strategy."""
    optimal_slices: int
    slice_size: float
    total_cost_bps: float
    execution_time_hours: float
    risk_adjusted_cost: float
    strategy_type: str

class LiquidityModels:
    """
    Liquidity analytics and market impact models.
    
    Provides liquidity scoring and cost estimation for pre-trade analysis
    and stress testing scenarios.
    """
    
    def __init__(self, config: Optional[LiquidityConfig] = None):
        self.config = config or LiquidityConfig()
        self._cache: Dict[str, Union[LiquidityScore, MarketImpact]] = {}
        self._last_calculation: Optional[datetime] = None
        
    def calculate_liquidity_score(
        self,
        metrics: LiquidityMetrics,
        method: Optional[LiquidityScoreMethod] = None
    ) -> LiquidityScore:
        """
        Calculate liquidity score for an instrument.
        
        Args:
            metrics: Liquidity metrics for the instrument
            method: Scoring method override
            
        Returns:
            LiquidityScore with score and breakdown
        """
        method = method or self.config.liquidity_score_method
        
        # Calculate component scores
        bid_ask_score = self._calculate_bid_ask_score(metrics.bid_ask_spread)
        turnover_score = self._calculate_turnover_score(metrics.turnover_ratio)
        depth_score = self._calculate_depth_score(metrics.market_depth, metrics.issue_size)
        volatility_score = self._calculate_volatility_score(metrics.volatility)
        
        # Combine scores based on method
        if method == LiquidityScoreMethod.COMPOSITE:
            composite_score = (
                self.config.bid_ask_weight * bid_ask_score +
                self.config.turnover_weight * turnover_score +
                self.config.depth_weight * depth_score +
                self.config.volatility_weight * volatility_score
            )
        elif method == LiquidityScoreMethod.BID_ASK_ONLY:
            composite_score = bid_ask_score
        elif method == LiquidityScoreMethod.TURNOVER_ONLY:
            composite_score = turnover_score
        elif method == LiquidityScoreMethod.DEPTH_ONLY:
            composite_score = depth_score
        else:
            raise ValueError(f"Unknown liquidity score method: {method}")
        
        # Apply bounds
        composite_score = np.clip(
            composite_score,
            self.config.min_liquidity_score,
            self.config.max_liquidity_score
        )
        
        # Calculate confidence band
        confidence_band = self._calculate_confidence_band(
            composite_score,
            metrics,
            method
        )
        
        # Create result
        result = LiquidityScore(
            score=float(composite_score),
            breakdown={
                "bid_ask_score": float(bid_ask_score),
                "turnover_score": float(turnover_score),
                "depth_score": float(depth_score),
                "volatility_score": float(volatility_score)
            },
            confidence_band=confidence_band,
            calculation_date=datetime.now().date(),
            metadata={
                "method": method.value,
                "rating": metrics.rating,
                "sector": metrics.sector,
                "maturity_tenor": metrics.maturity_tenor,
                "issue_size": metrics.issue_size
            }
        )
        
        # Cache result
        if self.config.enable_caching:
            cache_key = self._generate_liquidity_cache_key(metrics, method)
            self._cache[cache_key] = result
        
        return result
    
    def estimate_market_impact(
        self,
        trade_size: float,
        liquidity_score: float,
        impact_model: Optional[ImpactModel] = None,
        confidence_level: Optional[float] = None
    ) -> MarketImpact:
        """
        Estimate market impact cost for a trade.
        
        Args:
            trade_size: Trade size in notional
            liquidity_score: Liquidity score (0-100)
            impact_model: Impact model override
            confidence_level: Confidence level override
            
        Returns:
            MarketImpact with cost estimation and confidence
        """
        impact_model = impact_model or self.config.impact_model
        confidence_level = confidence_level or self.config.confidence_level
        
        # Calculate base impact
        if impact_model == ImpactModel.LINEAR:
            base_impact = self._linear_impact(trade_size, liquidity_score)
        elif impact_model == ImpactModel.SQUARE_ROOT:
            base_impact = self._square_root_impact(trade_size, liquidity_score)
        elif impact_model == ImpactModel.POWER_LAW:
            base_impact = self._power_law_impact(trade_size, liquidity_score)
        elif impact_model == ImpactModel.ADAPTIVE:
            base_impact = self._adaptive_impact(trade_size, liquidity_score)
        else:
            raise ValueError(f"Unknown impact model: {impact_model}")
        
        # Apply calibration factor
        calibrated_impact = base_impact * self.config.impact_calibration_factor
        
        # Calculate confidence band
        confidence_band = self._calculate_impact_confidence_band(
            calibrated_impact,
            trade_size,
            liquidity_score,
            confidence_level
        )
        
        # Create result
        result = MarketImpact(
            expected_cost_bps=float(calibrated_impact),
            confidence_band=confidence_band,
            impact_model=impact_model,
            trade_size=trade_size,
            liquidity_score=liquidity_score,
            calculation_date=datetime.now().date(),
            metadata={
                "base_impact": float(base_impact),
                "calibration_factor": self.config.impact_calibration_factor,
                "confidence_level": confidence_level
            }
        )
        
        # Cache result
        if self.config.enable_caching:
            cache_key = self._generate_impact_cache_key(trade_size, liquidity_score, impact_model)
            self._cache[cache_key] = result
        
        return result
    
    def optimize_slicing_strategy(
        self,
        total_size: float,
        liquidity_score: float,
        time_horizon_hours: float = 8.0,
        risk_aversion: float = 1.0
    ) -> SlicingStrategy:
        """
        Optimize trade slicing strategy.
        
        Args:
            total_size: Total trade size
            liquidity_score: Liquidity score
            time_horizon_hours: Available execution time
            risk_aversion: Risk aversion parameter
            
        Returns:
            SlicingStrategy with optimal parameters
        """
        # Simple optimization: balance between cost and execution time
        # In production, use more sophisticated optimization
        
        # Estimate optimal number of slices
        if liquidity_score > 80:
            # High liquidity: fewer slices
            optimal_slices = max(1, int(total_size / 1000000))  # 1M per slice
        elif liquidity_score > 50:
            # Medium liquidity: moderate slices
            optimal_slices = max(2, int(total_size / 500000))   # 500K per slice
        else:
            # Low liquidity: more slices
            optimal_slices = max(4, int(total_size / 250000))   # 250K per slice
        
        # Calculate slice size
        slice_size = total_size / optimal_slices
        
        # Estimate total cost
        total_cost = self.estimate_market_impact(
            slice_size,
            liquidity_score
        ).expected_cost_bps * optimal_slices
        
        # Estimate execution time
        execution_time = time_horizon_hours / optimal_slices
        
        # Calculate risk-adjusted cost
        risk_adjusted_cost = total_cost * (1 + risk_aversion * (1 - liquidity_score / 100))
        
        return SlicingStrategy(
            optimal_slices=optimal_slices,
            slice_size=slice_size,
            total_cost_bps=float(total_cost),
            execution_time_hours=float(execution_time),
            risk_adjusted_cost=float(risk_adjusted_cost),
            strategy_type="balanced"
        )
    
    def stress_liquidity_score(
        self,
        base_score: LiquidityScore,
        stress_scenario: str = "liquidity_crunch"
    ) -> LiquidityScore:
        """
        Apply stress scenario to liquidity score.
        
        Args:
            base_score: Base liquidity score
            stress_scenario: Stress scenario type
            
        Returns:
            Stressed LiquidityScore
        """
        if not self.config.enable_stress_testing:
            return base_score
        
        # Apply stress multiplier based on scenario
        if stress_scenario == "liquidity_crunch":
            stress_multiplier = self.config.stress_multiplier
        elif stress_scenario == "market_crisis":
            stress_multiplier = self.config.stress_multiplier * 1.5
        elif stress_scenario == "sector_specific":
            stress_multiplier = self.config.stress_multiplier * 1.2
        else:
            stress_multiplier = 1.0
        
        # Apply stress to score (lower score = worse liquidity)
        stressed_score = base_score.score / stress_multiplier
        stressed_score = max(self.config.min_liquidity_score, stressed_score)
        
        # Create stressed result
        stressed_result = LiquidityScore(
            score=float(stressed_score),
            breakdown=base_score.breakdown,
            confidence_band=(
                base_score.confidence_band[0] / stress_multiplier,
                base_score.confidence_band[1] / stress_multiplier
            ),
            calculation_date=datetime.now().date(),
            metadata={
                **base_score.metadata,
                "stress_scenario": stress_scenario,
                "stress_multiplier": stress_multiplier,
                "original_score": base_score.score
            }
        )
        
        return stressed_result
    
    def _calculate_bid_ask_score(self, bid_ask_spread: float) -> float:
        """Calculate bid-ask component of liquidity score."""
        # Normalize bid-ask spread (lower is better)
        # Typical range: 0-100 bps
        if bid_ask_spread <= 1.0:  # 1 bp or less
            return 100.0
        elif bid_ask_spread <= 5.0:  # 5 bps
            return 90.0
        elif bid_ask_spread <= 10.0:  # 10 bps
            return 80.0
        elif bid_ask_spread <= 25.0:  # 25 bps
            return 60.0
        elif bid_ask_spread <= 50.0:  # 50 bps
            return 40.0
        else:  # 100+ bps
            return 20.0
    
    def _calculate_turnover_score(self, turnover_ratio: float) -> float:
        """Calculate turnover component of liquidity score."""
        # Normalize turnover ratio (higher is better)
        # Typical range: 0-10% daily
        if turnover_ratio >= 0.05:  # 5% or more
            return 100.0
        elif turnover_ratio >= 0.02:  # 2%
            return 80.0
        elif turnover_ratio >= 0.01:  # 1%
            return 60.0
        elif turnover_ratio >= 0.005:  # 0.5%
            return 40.0
        else:  # Less than 0.5%
            return 20.0
    
    def _calculate_depth_score(self, market_depth: float, issue_size: float) -> float:
        """Calculate market depth component of liquidity score."""
        # Normalize depth relative to issue size
        depth_ratio = market_depth / issue_size if issue_size > 0 else 0
        
        if depth_ratio >= 0.1:  # 10% or more
            return 100.0
        elif depth_ratio >= 0.05:  # 5%
            return 80.0
        elif depth_ratio >= 0.02:  # 2%
            return 60.0
        elif depth_ratio >= 0.01:  # 1%
            return 40.0
        else:  # Less than 1%
            return 20.0
    
    def _calculate_volatility_score(self, volatility: float) -> float:
        """Calculate volatility component of liquidity score."""
        # Normalize volatility (lower is better for liquidity)
        # Assuming annualized volatility in decimal form
        vol_bps = volatility * 10000  # Convert to basis points
        
        if vol_bps <= 50:  # 50 bps or less
            return 100.0
        elif vol_bps <= 100:  # 100 bps
            return 80.0
        elif vol_bps <= 200:  # 200 bps
            return 60.0
        elif vol_bps <= 500:  # 500 bps
            return 40.0
        else:  # 500+ bps
            return 20.0
    
    def _calculate_confidence_band(
        self,
        score: float,
        metrics: LiquidityMetrics,
        method: LiquidityScoreMethod
    ) -> Tuple[float, float]:
        """Calculate confidence band for liquidity score."""
        # Simple confidence band based on data quality
        # In production, use more sophisticated uncertainty quantification
        
        # Base uncertainty
        base_uncertainty = 5.0  # 5 points
        
        # Additional uncertainty based on data quality
        if metrics.days_since_last_trade > 30:
            base_uncertainty += 3.0
        if metrics.bid_ask_spread > 50:
            base_uncertainty += 2.0
        if metrics.turnover_ratio < 0.001:
            base_uncertainty += 2.0
        
        # Calculate confidence interval
        lower_bound = max(0, score - base_uncertainty)
        upper_bound = min(100, score + base_uncertainty)
        
        return (lower_bound, upper_bound)
    
    def _linear_impact(self, trade_size: float, liquidity_score: float) -> float:
        """Calculate linear market impact."""
        # Linear impact model: cost = k * size / liquidity
        k = 0.1  # Calibration constant
        return k * trade_size / (liquidity_score + 1)  # Add 1 to avoid division by zero
    
    def _square_root_impact(self, trade_size: float, liquidity_score: float) -> float:
        """Calculate square-root market impact."""
        # Square-root impact model: cost = k * sqrt(size) / liquidity
        k = 0.05  # Calibration constant
        return k * np.sqrt(trade_size) / (liquidity_score + 1)
    
    def _power_law_impact(self, trade_size: float, liquidity_score: float) -> float:
        """Calculate power-law market impact."""
        # Power-law impact model: cost = k * size^0.6 / liquidity
        k = 0.03  # Calibration constant
        return k * (trade_size ** 0.6) / (liquidity_score + 1)
    
    def _adaptive_impact(self, trade_size: float, liquidity_score: float) -> float:
        """Calculate adaptive market impact."""
        # Adaptive model: switches between linear and square-root based on size
        if trade_size < 1000000:  # 1M threshold
            return self._linear_impact(trade_size, liquidity_score)
        else:
            return self._square_root_impact(trade_size, liquidity_score)
    
    def _calculate_impact_confidence_band(
        self,
        base_impact: float,
        trade_size: float,
        liquidity_score: float,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence band for market impact."""
        # Simple confidence band based on model uncertainty
        # In production, use more sophisticated uncertainty quantification
        
        # Base uncertainty (20% of impact)
        base_uncertainty = base_impact * 0.2
        
        # Additional uncertainty for large trades
        if trade_size > 10000000:  # 10M
            base_uncertainty *= 1.5
        
        # Additional uncertainty for low liquidity
        if liquidity_score < 30:
            base_uncertainty *= 1.3
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * base_uncertainty
        
        lower_bound = max(0, base_impact - margin)
        upper_bound = base_impact + margin
        
        return (lower_bound, upper_bound)
    
    def _generate_liquidity_cache_key(
        self,
        metrics: LiquidityMetrics,
        method: LiquidityScoreMethod
    ) -> str:
        """Generate cache key for liquidity score results."""
        return f"liquidity_{metrics.rating}_{metrics.sector}_{method.value}"
    
    def _generate_impact_cache_key(
        self,
        trade_size: float,
        liquidity_score: float,
        impact_model: ImpactModel
    ) -> str:
        """Generate cache key for market impact results."""
        return f"impact_{trade_size}_{liquidity_score:.1f}_{impact_model.value}"
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Liquidity models cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "last_calculation": self._last_calculation.isoformat() if self._last_calculation else None
        }
