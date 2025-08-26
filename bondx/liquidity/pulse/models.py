"""
Pulse Models for Liquidity Pulse

This module implements the core models for calculating:
- Liquidity Index (0-100): Real-time liquidity nowcast
- Repayment Support (0-100): Medium-term fundamental support
- BondX Score (0-100): Combined weighted score
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import hashlib
import json

from ...core.logging import get_logger
from ...api.v1.schemas_liquidity import (
    Driver, DataFreshness, SignalQuality
)
from .feature_engine import FeatureSet
from .signal_adapters import ProcessedSignal

logger = get_logger(__name__)

@dataclass
class ModelWeights:
    """Weights for different components in the BondX Score."""
    liquidity_index_weight: float = 0.6
    repayment_support_weight: float = 0.4
    
    # Sector-specific adjustments
    sector_weights: Dict[str, Dict[str, float]] = None
    
    # Rating-specific adjustments
    rating_weights: Dict[str, Dict[str, float]] = None
    
    # Tenor-specific adjustments
    tenor_weights: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.sector_weights is None:
            self.sector_weights = {
                "FINANCIAL": {"liquidity": 0.7, "repayment": 0.3},
                "UTILITIES": {"liquidity": 0.4, "repayment": 0.6},
                "INDUSTRIAL": {"liquidity": 0.5, "repayment": 0.5},
                "GOVERNMENT": {"liquidity": 0.3, "repayment": 0.7}
            }
        
        if self.rating_weights is None:
            self.rating_weights = {
                "AAA": {"liquidity": 0.3, "repayment": 0.7},
                "AA": {"liquidity": 0.4, "repayment": 0.6},
                "A": {"liquidity": 0.5, "repayment": 0.5},
                "BBB": {"liquidity": 0.6, "repayment": 0.4},
                "BB": {"liquidity": 0.7, "repayment": 0.3}
            }
        
        if self.tenor_weights is None:
            self.tenor_weights = {
                "SHORT": {"liquidity": 0.8, "repayment": 0.2},      # < 2 years
                "MEDIUM": {"liquidity": 0.6, "repayment": 0.4},     # 2-7 years
                "LONG": {"liquidity": 0.4, "repayment": 0.6},       # 7-15 years
                "VERY_LONG": {"liquidity": 0.2, "repayment": 0.8}   # > 15 years
            }

class LiquidityIndexModel:
    """Model for calculating the Liquidity Index (0-100)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Feature weights for liquidity index
        self.feature_weights = config.get("liquidity_feature_weights", {
            "spread_bps": -0.3,           # Higher spread = lower liquidity
            "depth_density": 0.2,         # Higher depth = higher liquidity
            "turnover_velocity": 0.15,    # Higher turnover = higher liquidity
            "mm_online_ratio": 0.15,      # MM online = higher liquidity
            "auction_demand": 0.1,        # Higher demand = higher liquidity
            "time_since_last_trade": -0.1 # Longer time = lower liquidity
        })
        
        # Base liquidity score
        self.base_liquidity = config.get("base_liquidity", 50.0)
        
        # Calibration parameters
        self.calibration_params = config.get("calibration", {
            "min_spread_bps": 1.0,       # Minimum spread for calibration
            "max_spread_bps": 100.0,     # Maximum spread for calibration
            "spread_sensitivity": 0.5,   # How sensitive to spread changes
            "depth_threshold": 1000000,  # Depth threshold for full score
        })
    
    def calculate_liquidity_index(self, feature_set: FeatureSet, signals: List[ProcessedSignal]) -> Tuple[float, List[Driver]]:
        """Calculate liquidity index and identify drivers."""
        try:
            # Start with base liquidity
            liquidity_score = self.base_liquidity
            drivers = []
            
            # Apply feature contributions
            for feature_name, weight in self.feature_weights.items():
                if feature_name in feature_set.features:
                    feature_value = feature_set.features[feature_name]
                    contribution = self._calculate_feature_contribution(feature_name, feature_value, weight)
                    liquidity_score += contribution
                    
                    # Create driver entry
                    driver = Driver(
                        name=feature_name,
                        contribution=abs(contribution),
                        direction="↑" if contribution > 0 else "↓",
                        source="microstructure",
                        confidence=self._calculate_feature_confidence(feature_name, feature_value, signals)
                    )
                    drivers.append(driver)
            
            # Apply microstructure-specific adjustments
            microstructure_adjustments = self._apply_microstructure_adjustments(feature_set, signals)
            liquidity_score += microstructure_adjustments["total_adjustment"]
            
            # Add microstructure drivers
            for adj_name, adj_value in microstructure_adjustments["adjustments"].items():
                if abs(adj_value) > 0.1:  # Only include significant adjustments
                    driver = Driver(
                        name=f"microstructure_{adj_name}",
                        contribution=abs(adj_value),
                        direction="↑" if adj_value > 0 else "↓",
                        source="microstructure",
                        confidence=0.8
                    )
                    drivers.append(driver)
            
            # Clip to valid range
            liquidity_score = np.clip(liquidity_score, 0, 100)
            
            # Sort drivers by contribution magnitude
            drivers.sort(key=lambda x: x.contribution, reverse=True)
            
            # Limit to top drivers
            drivers = drivers[:10]
            
            return liquidity_score, drivers
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity index: {e}")
            return self.base_liquidity, []
    
    def _calculate_feature_contribution(self, feature_name: str, feature_value: float, weight: float) -> float:
        """Calculate contribution of a single feature to liquidity index."""
        try:
            if feature_name == "spread_bps":
                # Spread contribution: lower spread = higher liquidity
                spread_bps = max(feature_value, self.calibration_params["min_spread_bps"])
                spread_bps = min(spread_bps, self.calibration_params["max_spread_bps"])
                
                # Normalize spread to 0-1 range and invert
                spread_normalized = (spread_bps - self.calibration_params["min_spread_bps"]) / \
                                  (self.calibration_params["max_spread_bps"] - self.calibration_params["min_spread_bps"])
                spread_contribution = (1.0 - spread_normalized) * weight * 20  # Scale to reasonable range
                
                return spread_contribution
                
            elif feature_name == "depth_density":
                # Depth contribution: higher depth = higher liquidity
                depth_threshold = self.calibration_params["depth_threshold"]
                depth_normalized = min(feature_value / depth_threshold, 1.0)
                depth_contribution = depth_normalized * weight * 15
                
                return depth_contribution
                
            elif feature_name == "turnover_velocity":
                # Turnover contribution: higher turnover = higher liquidity
                turnover_normalized = min(feature_value / 1000000, 1.0)  # Normalize to 1M
                turnover_contribution = turnover_normalized * weight * 10
                
                return turnover_contribution
                
            elif feature_name == "mm_online_ratio":
                # MM online contribution: MM online = higher liquidity
                mm_contribution = feature_value * weight * 10
                
                return mm_contribution
                
            elif feature_name == "auction_demand":
                # Auction demand contribution: higher demand = higher liquidity
                demand_contribution = feature_value * weight * 10
                
                return demand_contribution
                
            elif feature_name == "time_since_last_trade":
                # Time decay contribution: longer time = lower liquidity
                time_hours = feature_value / 3600  # Convert seconds to hours
                time_normalized = min(time_hours / 24, 1.0)  # Normalize to 24 hours
                time_contribution = -time_normalized * weight * 5
                
                return time_contribution
                
            else:
                # Generic feature contribution
                return feature_value * weight * 5
                
        except Exception as e:
            self.logger.debug(f"Error calculating contribution for {feature_name}: {e}")
            return 0.0
    
    def _apply_microstructure_adjustments(self, feature_set: FeatureSet, signals: List[ProcessedSignal]) -> Dict[str, Any]:
        """Apply microstructure-specific adjustments to liquidity index."""
        adjustments = {}
        total_adjustment = 0.0
        
        try:
            # Order imbalance adjustment
            if "order_imbalance" in feature_set.features:
                imbalance = feature_set.features["order_imbalance"]
                # Extreme imbalance reduces liquidity
                if abs(imbalance) > 0.7:
                    imbalance_penalty = -5.0 * (abs(imbalance) - 0.7) / 0.3
                    adjustments["order_imbalance"] = imbalance_penalty
                    total_adjustment += imbalance_penalty
            
            # Volatility adjustment
            if "spread_volatility" in feature_set.features:
                volatility = feature_set.features["spread_volatility"]
                # High volatility reduces liquidity
                if volatility > 10.0:
                    volatility_penalty = -3.0 * (volatility - 10.0) / 10.0
                    adjustments["spread_volatility"] = volatility_penalty
                    total_adjustment += volatility_penalty
            
            # Market maker activity adjustment
            if "mm_spread_bps" in feature_set.features:
                mm_spread = feature_set.features["mm_spread_bps"]
                # Very wide MM spreads indicate poor liquidity
                if mm_spread > 50.0:
                    mm_penalty = -8.0 * (mm_spread - 50.0) / 50.0
                    adjustments["mm_spread"] = mm_penalty
                    total_adjustment += mm_penalty
            
            # Recent trade activity adjustment
            if "trades_count" in feature_set.features:
                trades_count = feature_set.features.get("trades_count", 0)
                # No recent trades indicate poor liquidity
                if trades_count == 0:
                    no_trades_penalty = -10.0
                    adjustments["no_recent_trades"] = no_trades_penalty
                    total_adjustment += no_trades_penalty
                elif trades_count < 5:
                    low_trades_penalty = -5.0 * (5 - trades_count) / 5
                    adjustments["low_trade_activity"] = low_trades_penalty
                    total_adjustment += low_trades_penalty
            
        except Exception as e:
            self.logger.debug(f"Error applying microstructure adjustments: {e}")
        
        return {
            "adjustments": adjustments,
            "total_adjustment": total_adjustment
        }
    
    def _calculate_feature_confidence(self, feature_name: str, feature_value: float, signals: List[ProcessedSignal]) -> float:
        """Calculate confidence in a feature's contribution."""
        try:
            # Base confidence
            confidence = 0.7
            
            # Check signal quality for this feature
            relevant_signals = [s for s in signals if feature_name in s.raw_data]
            if relevant_signals:
                avg_quality = np.mean([s.metadata.quality.value for s in relevant_signals])
                # Map quality to confidence
                quality_confidence = {
                    "excellent": 0.95,
                    "good": 0.85,
                    "moderate": 0.7,
                    "poor": 0.5,
                    "unreliable": 0.3
                }
                confidence = quality_confidence.get(avg_quality, 0.7)
            
            # Adjust confidence based on feature value stability
            if hasattr(self, '_historical_values') and feature_name in self._historical_values:
                historical_std = np.std(self._historical_values[feature_name])
                if historical_std > 0:
                    stability_factor = 1.0 / (1.0 + historical_std / abs(feature_value))
                    confidence *= stability_factor
            
            return np.clip(confidence, 0.1, 0.95)
            
        except Exception as e:
            self.logger.debug(f"Error calculating feature confidence: {e}")
            return 0.5

class RepaymentSupportModel:
    """Model for calculating the Repayment Support Index (0-100)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Feature weights for repayment support
        self.feature_weights = config.get("repayment_feature_weights", {
            "traffic_index": 0.25,           # Traffic as proxy for economic activity
            "utilities_index": 0.2,          # Utilities as proxy for stability
            "sentiment_intensity": 0.15,     # Sentiment as proxy for market perception
            "buzz_volume": 0.1,              # Buzz as proxy for attention
            "spread_stability": 0.2,         # Spread stability as proxy for market confidence
            "depth_stability": 0.1           # Depth stability as proxy for investor confidence
        })
        
        # Base repayment support score
        self.base_repayment = config.get("base_repayment", 60.0)
        
        # Sector-specific adjustments
        self.sector_adjustments = config.get("sector_adjustments", {
            "GOVERNMENT": 20,      # Government bonds get boost
            "UTILITIES": 15,       # Utilities get boost
            "FINANCIAL": -5,       # Financials get penalty
            "INDUSTRIAL": 0        # Industrials neutral
        })
    
    def calculate_repayment_support(self, feature_set: FeatureSet, signals: List[ProcessedSignal], 
                                  sector: str = "INDUSTRIAL") -> Tuple[float, List[Driver]]:
        """Calculate repayment support index and identify drivers."""
        try:
            # Start with base repayment support
            repayment_score = self.base_repayment
            drivers = []
            
            # Apply feature contributions
            for feature_name, weight in self.feature_weights.items():
                if feature_name in feature_set.features:
                    feature_value = feature_set.features[feature_name]
                    contribution = self._calculate_repayment_contribution(feature_name, feature_value, weight)
                    repayment_score += contribution
                    
                    # Create driver entry
                    driver = Driver(
                        name=feature_name,
                        contribution=abs(contribution),
                        direction="↑" if contribution > 0 else "↓",
                        source="fundamentals",
                        confidence=self._calculate_repayment_confidence(feature_name, feature_value, signals)
                    )
                    drivers.append(driver)
            
            # Apply sector adjustment
            sector_adjustment = self.sector_adjustments.get(sector, 0)
            repayment_score += sector_adjustment
            
            if sector_adjustment != 0:
                driver = Driver(
                    name=f"sector_{sector.lower()}",
                    contribution=abs(sector_adjustment),
                    direction="↑" if sector_adjustment > 0 else "↓",
                    source="sector_classification",
                    confidence=0.9
                )
                drivers.append(driver)
            
            # Apply stability adjustments
            stability_adjustments = self._apply_stability_adjustments(feature_set)
            repayment_score += stability_adjustments["total_adjustment"]
            
            # Add stability drivers
            for adj_name, adj_value in stability_adjustments["adjustments"].items():
                if abs(adj_value) > 0.1:
                    driver = Driver(
                        name=f"stability_{adj_name}",
                        contribution=abs(adj_value),
                        direction="↑" if adj_value > 0 else "↓",
                        source="stability_metrics",
                        confidence=0.8
                    )
                    drivers.append(driver)
            
            # Clip to valid range
            repayment_score = np.clip(repayment_score, 0, 100)
            
            # Sort drivers by contribution magnitude
            drivers.sort(key=lambda x: x.contribution, reverse=True)
            
            # Limit to top drivers
            drivers = drivers[:8]
            
            return repayment_score, drivers
            
        except Exception as e:
            self.logger.error(f"Error calculating repayment support: {e}")
            return self.base_repayment, []
    
    def _calculate_repayment_contribution(self, feature_name: str, feature_value: float, weight: float) -> float:
        """Calculate contribution of a single feature to repayment support."""
        try:
            if feature_name == "traffic_index":
                # Traffic contribution: higher traffic = higher economic activity = higher repayment support
                traffic_normalized = min(feature_value / 100, 1.0)
                traffic_contribution = traffic_normalized * weight * 20
                
                return traffic_contribution
                
            elif feature_name == "utilities_index":
                # Utilities contribution: higher utilities = higher stability = higher repayment support
                utilities_normalized = min(feature_value / 100, 1.0)
                utilities_contribution = utilities_normalized * weight * 15
                
                return utilities_contribution
                
            elif feature_name == "sentiment_intensity":
                # Sentiment contribution: positive sentiment = higher repayment support
                sentiment_contribution = feature_value * weight * 10
                
                return sentiment_contribution
                
            elif feature_name == "buzz_volume":
                # Buzz contribution: moderate buzz = higher repayment support, extreme buzz = lower
                buzz_normalized = min(feature_value / 100, 1.0)
                if buzz_normalized < 0.5:
                    buzz_contribution = buzz_normalized * weight * 10
                else:
                    buzz_contribution = (1.0 - buzz_normalized) * weight * 5
                
                return buzz_contribution
                
            elif feature_name == "spread_stability":
                # Spread stability contribution: higher stability = higher repayment support
                stability_contribution = feature_value * weight * 15
                
                return stability_contribution
                
            elif feature_name == "depth_stability":
                # Depth stability contribution: higher stability = higher repayment support
                stability_contribution = feature_value * weight * 10
                
                return stability_contribution
                
            else:
                # Generic feature contribution
                return feature_value * weight * 5
                
        except Exception as e:
            self.logger.debug(f"Error calculating repayment contribution for {feature_name}: {e}")
            return 0.0
    
    def _apply_stability_adjustments(self, feature_set: FeatureSet) -> Dict[str, Any]:
        """Apply stability-based adjustments to repayment support."""
        adjustments = {}
        total_adjustment = 0.0
        
        try:
            # Trend stability adjustment
            if "stability_spread_bps_trend_r2" in feature_set.stability:
                trend_stability = feature_set.stability["stability_spread_bps_trend_r2"]
                # High trend stability indicates good repayment prospects
                if trend_stability > 0.7:
                    trend_boost = 5.0 * (trend_stability - 0.7) / 0.3
                    adjustments["trend_stability"] = trend_boost
                    total_adjustment += trend_boost
            
            # Anomaly adjustment
            if "anomaly_spread_bps_anomaly_score" in feature_set.anomalies:
                anomaly_score = feature_set.anomalies["anomaly_spread_bps_anomaly_score"]
                # High anomaly indicates instability
                if anomaly_score > 0.5:
                    anomaly_penalty = -8.0 * (anomaly_score - 0.5) / 0.5
                    adjustments["spread_anomaly"] = anomaly_penalty
                    total_adjustment += anomaly_penalty
            
            # Seasonality adjustment
            if "seasonal_spread_bps_7d_adjustment" in feature_set.seasonality:
                seasonal_adj = feature_set.seasonality["seasonal_spread_bps_7d_adjustment"]
                # Large seasonal adjustments indicate instability
                if abs(seasonal_adj) > 5.0:
                    seasonal_penalty = -3.0 * (abs(seasonal_adj) - 5.0) / 5.0
                    adjustments["seasonal_instability"] = seasonal_penalty
                    total_adjustment += seasonal_penalty
            
        except Exception as e:
            self.logger.debug(f"Error applying stability adjustments: {e}")
        
        return {
            "adjustments": adjustments,
            "total_adjustment": total_adjustment
        }
    
    def _calculate_repayment_confidence(self, feature_name: str, feature_value: float, signals: List[ProcessedSignal]) -> float:
        """Calculate confidence in a repayment feature's contribution."""
        try:
            # Base confidence
            confidence = 0.7
            
            # Check signal quality for this feature
            relevant_signals = [s for s in signals if feature_name in s.raw_data]
            if relevant_signals:
                avg_quality = np.mean([s.metadata.quality.value for s in relevant_signals])
                # Map quality to confidence
                quality_confidence = {
                    "excellent": 0.95,
                    "good": 0.85,
                    "moderate": 0.7,
                    "poor": 0.5,
                    "unreliable": 0.3
                }
                confidence = quality_confidence.get(avg_quality, 0.7)
            
            # Alt-data features have lower confidence than market data
            if feature_name in ["traffic_index", "utilities_index", "buzz_volume"]:
                confidence *= 0.8
            
            return np.clip(confidence, 0.1, 0.95)
            
        except Exception as e:
            self.logger.debug(f"Error calculating repayment confidence: {e}")
            return 0.5

class BondXScoreModel:
    """Model for calculating the combined BondX Score (0-100)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model weights
        self.weights = ModelWeights(**config.get("bondx_weights", {}))
        
        # Uncertainty calculation parameters
        self.uncertainty_params = config.get("uncertainty", {
            "base_uncertainty": 0.2,
            "signal_quality_weight": 0.3,
            "data_freshness_weight": 0.3,
            "model_confidence_weight": 0.4
        })
    
    def calculate_bondx_score(self, liquidity_index: float, repayment_support: float,
                            liquidity_drivers: List[Driver], repayment_drivers: List[Driver],
                            sector: str = "INDUSTRIAL", rating: str = "BBB", tenor: str = "MEDIUM") -> Tuple[float, List[Driver], float]:
        """Calculate BondX Score and identify top drivers."""
        try:
            # Get weights for this instrument
            weights = self._get_instrument_weights(sector, rating, tenor)
            
            # Calculate weighted score
            bondx_score = (
                liquidity_index * weights["liquidity"] +
                repayment_support * weights["repayment"]
            )
            
            # Clip to valid range
            bondx_score = np.clip(bondx_score, 0, 100)
            
            # Combine and rank drivers
            all_drivers = liquidity_drivers + repayment_drivers
            
            # Adjust driver contributions based on weights
            for driver in all_drivers:
                if driver.source == "microstructure":
                    driver.contribution *= weights["liquidity"]
                elif driver.source == "fundamentals":
                    driver.contribution *= weights["repayment"]
            
            # Sort by adjusted contribution
            all_drivers.sort(key=lambda x: x.contribution, reverse=True)
            
            # Take top drivers
            top_drivers = all_drivers[:12]
            
            # Calculate overall uncertainty
            uncertainty = self._calculate_overall_uncertainty(
                liquidity_drivers, repayment_drivers, weights
            )
            
            return bondx_score, top_drivers, uncertainty
            
        except Exception as e:
            self.logger.error(f"Error calculating BondX Score: {e}")
            return 50.0, [], 0.5
    
    def _get_instrument_weights(self, sector: str, rating: str, tenor: str) -> Dict[str, float]:
        """Get weights for a specific instrument."""
        # Start with default weights
        weights = {
            "liquidity": self.weights.liquidity_index_weight,
            "repayment": self.weights.repayment_support_weight
        }
        
        # Apply sector adjustments
        if sector in self.weights.sector_weights:
            sector_weights = self.weights.sector_weights[sector]
            weights["liquidity"] = sector_weights.get("liquidity", weights["liquidity"])
            weights["repayment"] = sector_weights.get("repayment", weights["repayment"])
        
        # Apply rating adjustments
        if rating in self.weights.rating_weights:
            rating_weights = self.weights.rating_weights[rating]
            weights["liquidity"] = rating_weights.get("liquidity", weights["liquidity"])
            weights["repayment"] = rating_weights.get("repayment", weights["repayment"])
        
        # Apply tenor adjustments
        if tenor in self.weights.tenor_weights:
            tenor_weights = self.weights.tenor_weights[tenor]
            weights["liquidity"] = tenor_weights.get("liquidity", weights["liquidity"])
            weights["repayment"] = tenor_weights.get("repayment", weights["repayment"])
        
        # Normalize weights to sum to 1.0
        total_weight = weights["liquidity"] + weights["repayment"]
        if total_weight > 0:
            weights["liquidity"] /= total_weight
            weights["repayment"] /= total_weight
        
        return weights
    
    def _calculate_overall_uncertainty(self, liquidity_drivers: List[Driver], 
                                     repayment_drivers: List[Driver], weights: Dict[str, float]) -> float:
        """Calculate overall uncertainty in the BondX Score."""
        try:
            # Base uncertainty
            uncertainty = self.uncertainty_params["base_uncertainty"]
            
            # Signal quality component
            liquidity_confidence = np.mean([d.confidence for d in liquidity_drivers]) if liquidity_drivers else 0.5
            repayment_confidence = np.mean([d.confidence for d in repayment_drivers]) if repayment_drivers else 0.5
            
            signal_quality_uncertainty = (
                (1.0 - liquidity_confidence) * weights["liquidity"] +
                (1.0 - repayment_confidence) * weights["repayment"]
            )
            
            # Data freshness component (assume medium freshness for now)
            data_freshness_uncertainty = 0.3
            
            # Model confidence component
            model_confidence_uncertainty = 0.2
            
            # Combine uncertainty components
            total_uncertainty = (
                uncertainty * 0.1 +
                signal_quality_uncertainty * self.uncertainty_params["signal_quality_weight"] +
                data_freshness_uncertainty * self.uncertainty_params["data_freshness_weight"] +
                model_confidence_uncertainty * self.uncertainty_params["model_confidence_weight"]
            )
            
            return np.clip(total_uncertainty, 0.1, 0.9)
            
        except Exception as e:
            self.logger.debug(f"Error calculating overall uncertainty: {e}")
            return 0.5

class PulseModelRegistry:
    """Registry for all pulse models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize models
        self.liquidity_model = LiquidityIndexModel(config.get("liquidity_model", {}))
        self.repayment_model = RepaymentSupportModel(config.get("repayment_model", {}))
        self.bondx_model = BondXScoreModel(config.get("bondx_model", {}))
        
        self.logger.info("Pulse model registry initialized")
    
    def calculate_all_scores(self, feature_set: FeatureSet, signals: List[ProcessedSignal],
                           sector: str = "INDUSTRIAL", rating: str = "BBB", tenor: str = "MEDIUM") -> Dict[str, Any]:
        """Calculate all scores for an ISIN."""
        try:
            # Calculate liquidity index
            liquidity_index, liquidity_drivers = self.liquidity_model.calculate_liquidity_index(
                feature_set, signals
            )
            
            # Calculate repayment support
            repayment_support, repayment_drivers = self.repayment_model.calculate_repayment_support(
                feature_set, signals, sector
            )
            
            # Calculate BondX Score
            bondx_score, top_drivers, uncertainty = self.bondx_model.calculate_bondx_score(
                liquidity_index, repayment_support, liquidity_drivers, repayment_drivers,
                sector, rating, tenor
            )
            
            return {
                "liquidity_index": liquidity_index,
                "repayment_support": repayment_support,
                "bondx_score": bondx_score,
                "liquidity_drivers": liquidity_drivers,
                "repayment_drivers": repayment_drivers,
                "top_drivers": top_drivers,
                "uncertainty": uncertainty
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating all scores: {e}")
            return {
                "liquidity_index": 50.0,
                "repayment_support": 60.0,
                "bondx_score": 55.0,
                "liquidity_drivers": [],
                "repayment_drivers": [],
                "top_drivers": [],
                "uncertainty": 0.5
            }
