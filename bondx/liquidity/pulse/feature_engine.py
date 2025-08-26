"""
Feature Engine for Liquidity Pulse

This module computes rolling statistics, seasonality adjustments, stability metrics,
and derived features from raw signals for the liquidity pulse calculation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler

from ...core.logging import get_logger
from .signal_adapters import ProcessedSignal, SignalType

logger = get_logger(__name__)

@dataclass
class FeatureSet:
    """Complete feature set for an ISIN."""
    isin: str
    timestamp: datetime
    features: Dict[str, float]
    rolling_stats: Dict[str, Dict[str, float]]
    seasonality: Dict[str, float]
    stability: Dict[str, float]
    anomalies: Dict[str, float]
    metadata: Dict[str, Any]

class FeatureEngine:
    """Engine for computing features from raw signals."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Feature computation parameters
        self.rolling_windows = config.get("rolling_windows", [7, 30, 90])  # days
        self.seasonality_periods = config.get("seasonality_periods", [7, 30])  # days
        self.stability_threshold = config.get("stability_threshold", 0.1)
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0)  # z-score
        
        # Historical data storage for rolling calculations
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.max_history_days = max(self.rolling_windows) + 30  # Extra buffer
        
        # Feature scalers for normalization
        self.scalers: Dict[str, StandardScaler] = {}
        
        self.logger.info("Feature engine initialized")
    
    def compute_features(self, signals: List[ProcessedSignal], isin: str) -> FeatureSet:
        """Compute complete feature set for an ISIN."""
        try:
            # Group signals by type
            signal_groups = self._group_signals_by_type(signals)
            
            # Compute basic features
            basic_features = self._compute_basic_features(signal_groups)
            
            # Compute rolling statistics
            rolling_stats = self._compute_rolling_stats(signals, isin)
            
            # Compute seasonality adjustments
            seasonality = self._compute_seasonality(signals, isin)
            
            # Compute stability metrics
            stability = self._compute_stability_metrics(signals, isin)
            
            # Compute anomaly scores
            anomalies = self._compute_anomaly_scores(signals, isin)
            
            # Update historical data
            self._update_historical_data(signals, isin)
            
            # Combine all features
            all_features = {
                **basic_features,
                **{f"rolling_{k}": v for k, v in rolling_stats.items()},
                **{f"seasonal_{k}": v for k, v in seasonality.items()},
                **{f"stability_{k}": v for k, v in stability.items()},
                **{f"anomaly_{k}": v for k, v in anomalies.items()}
            }
            
            # Normalize features
            normalized_features = self._normalize_features(all_features, isin)
            
            return FeatureSet(
                isin=isin,
                timestamp=datetime.now(),
                features=normalized_features,
                rolling_stats=rolling_stats,
                seasonality=seasonality,
                stability=stability,
                anomalies=anomalies,
                metadata={
                    "signal_count": len(signals),
                    "signal_types": list(signal_groups.keys()),
                    "computation_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error computing features for {isin}: {e}")
            # Return minimal feature set on error
            return FeatureSet(
                isin=isin,
                timestamp=datetime.now(),
                features={},
                rolling_stats={},
                seasonality={},
                stability={},
                anomalies={},
                metadata={"error": str(e)}
            )
    
    def _group_signals_by_type(self, signals: List[ProcessedSignal]) -> Dict[str, List[ProcessedSignal]]:
        """Group signals by their type."""
        groups = {}
        for signal in signals:
            signal_type = signal.metadata.signal_type.value
            if signal_type not in groups:
                groups[signal_type] = []
            groups[signal_type].append(signal)
        return groups
    
    def _compute_basic_features(self, signal_groups: Dict[str, List[ProcessedSignal]]) -> Dict[str, float]:
        """Compute basic features from signal groups."""
        features = {}
        
        # Microstructure features
        if "microstructure" in signal_groups:
            micro_signals = signal_groups["microstructure"]
            if micro_signals:
                # Spread features
                spreads = [s.raw_data.get("spread_bps", 0) for s in micro_signals if s.raw_data.get("spread_bps")]
                if spreads:
                    features["spread_bps"] = np.mean(spreads)
                    features["spread_volatility"] = np.std(spreads)
                
                # Depth features
                depths = [s.raw_data.get("depth_qty", 0) for s in micro_signals if s.raw_data.get("depth_qty")]
                if depths:
                    features["depth_density"] = np.mean(depths)
                    features["depth_stability"] = 1.0 / (1.0 + np.std(depths) / np.mean(depths))
                
                # Turnover features
                turnovers = [s.raw_data.get("turnover", 0) for s in micro_signals if s.raw_data.get("turnover")]
                if turnovers:
                    features["turnover_velocity"] = np.mean(turnovers)
                    features["turnover_momentum"] = np.diff(turnovers).mean() if len(turnovers) > 1 else 0
        
        # Auction/MM features
        if "auction_mm" in signal_groups:
            auction_signals = signal_groups["auction_mm"]
            if auction_signals:
                # Auction demand
                demands = [s.raw_data.get("auction_demand_index", 0) for s in auction_signals if s.raw_data.get("auction_demand_index")]
                if demands:
                    features["auction_demand"] = np.mean(demands)
                    features["auction_volatility"] = np.std(demands)
                
                # MM activity
                mm_online = [s.raw_data.get("mm_online", False) for s in auction_signals if s.raw_data.get("mm_online") is not None]
                if mm_online:
                    features["mm_online_ratio"] = np.mean(mm_online)
                
                # MM spreads
                mm_spreads = [s.raw_data.get("mm_spread_bps", 999) for s in auction_signals if s.raw_data.get("mm_spread_bps")]
                if mm_spreads:
                    features["mm_spread_bps"] = np.mean(mm_spreads)
        
        # Sentiment features
        if "sentiment" in signal_groups:
            sentiment_signals = signal_groups["sentiment"]
            if sentiment_signals:
                # Sentiment scores
                scores = [s.raw_data.get("sentiment_score", 0) for s in sentiment_signals if s.raw_data.get("sentiment_score")]
                if scores:
                    features["sentiment_intensity"] = np.abs(np.mean(scores))
                    features["sentiment_volatility"] = np.std(scores)
                    features["sentiment_momentum"] = np.diff(scores).mean() if len(scores) > 1 else 0
                
                # Buzz volume
                buzz_volumes = [s.raw_data.get("buzz_volume", 0) for s in sentiment_signals if s.raw_data.get("buzz_volume")]
                if buzz_volumes:
                    features["buzz_volume"] = np.mean(buzz_volumes)
        
        # Alt data features
        if "alt_data" in signal_groups:
            alt_signals = signal_groups["alt_data"]
            if alt_signals:
                # Traffic/utilities data
                traffic_values = [s.raw_data.get("value", 0) for s in alt_signals if s.raw_data.get("source") == "traffic"]
                if traffic_values:
                    features["traffic_index"] = np.mean(traffic_values)
                
                utilities_values = [s.raw_data.get("value", 0) for s in alt_signals if s.raw_data.get("source") == "utilities"]
                if utilities_values:
                    features["utilities_index"] = np.mean(utilities_values)
        
        return features
    
    def _compute_rolling_stats(self, signals: List[ProcessedSignal], isin: str) -> Dict[str, Dict[str, float]]:
        """Compute rolling statistics for different windows."""
        rolling_stats = {}
        
        # Get historical data for this ISIN
        if isin not in self.historical_data:
            return rolling_stats
        
        df = self.historical_data[isin]
        if df.empty:
            return rolling_stats
        
        # Compute rolling statistics for each window
        for window_days in self.rolling_windows:
            window_key = f"{window_days}d"
            rolling_stats[window_key] = {}
            
            # Rolling mean and std for key features
            for feature in ["spread_bps", "depth_density", "turnover_velocity", "sentiment_intensity"]:
                if feature in df.columns:
                    rolling_mean = df[feature].rolling(window=window_days, min_periods=1).mean().iloc[-1]
                    rolling_std = df[feature].rolling(window=window_days, min_periods=1).std().iloc[-1]
                    
                    rolling_stats[window_key][f"{feature}_mean"] = rolling_mean
                    rolling_stats[window_key][f"{feature}_std"] = rolling_std
                    
                    # Stability metric (inverse of coefficient of variation)
                    if rolling_mean != 0:
                        rolling_stats[window_key][f"{feature}_stability"] = 1.0 / (1.0 + rolling_std / abs(rolling_mean))
                    else:
                        rolling_stats[window_key][f"{feature}_stability"] = 1.0
        
        return rolling_stats
    
    def _compute_seasonality(self, signals: List[ProcessedSignal], isin: str) -> Dict[str, float]:
        """Compute seasonality adjustments."""
        seasonality = {}
        
        if isin not in self.historical_data:
            return seasonality
        
        df = self.historical_data[isin]
        if df.empty or len(df) < 30:  # Need sufficient history
            return seasonality
        
        # Compute seasonality for different periods
        for period_days in self.seasonality_periods:
            period_key = f"{period_days}d"
            
            # Simple seasonal decomposition using moving averages
            for feature in ["spread_bps", "depth_density", "turnover_velocity"]:
                if feature in df.columns:
                    try:
                        # Remove trend using moving average
                        trend = df[feature].rolling(window=period_days, center=True).mean()
                        detrended = df[feature] - trend
                        
                        # Compute seasonal pattern
                        seasonal_pattern = detrended.groupby(df.index.dayofweek).mean()
                        current_day = datetime.now().weekday()
                        
                        if current_day in seasonal_pattern.index:
                            seasonality[f"{feature}_{period_key}_adjustment"] = seasonal_pattern[current_day]
                        else:
                            seasonality[f"{feature}_{period_key}_adjustment"] = 0.0
                            
                    except Exception as e:
                        self.logger.debug(f"Error computing seasonality for {feature}: {e}")
                        seasonality[f"{feature}_{period_key}_adjustment"] = 0.0
        
        return seasonality
    
    def _compute_stability_metrics(self, signals: List[ProcessedSignal], isin: str) -> Dict[str, float]:
        """Compute stability metrics for features."""
        stability = {}
        
        if isin not in self.historical_data:
            return stability
        
        df = self.historical_data[isin]
        if df.empty or len(df) < 7:  # Need at least a week of data
            return stability
        
        # Compute stability metrics for key features
        for feature in ["spread_bps", "depth_density", "turnover_velocity", "sentiment_intensity"]:
            if feature in df.columns:
                try:
                    # Coefficient of variation (lower = more stable)
                    values = df[feature].dropna()
                    if len(values) > 1 and values.mean() != 0:
                        cv = values.std() / abs(values.mean())
                        stability[f"{feature}_cv"] = cv
                        stability[f"{feature}_stability_score"] = 1.0 / (1.0 + cv)
                    else:
                        stability[f"{feature}_cv"] = 0.0
                        stability[f"{feature}_stability_score"] = 1.0
                    
                    # Trend stability (using linear regression R²)
                    if len(values) > 5:
                        x = np.arange(len(values))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                        stability[f"{feature}_trend_r2"] = r_value ** 2
                        stability[f"{feature}_trend_slope"] = slope
                    else:
                        stability[f"{feature}_trend_r2"] = 0.0
                        stability[f"{feature}_trend_slope"] = 0.0
                        
                except Exception as e:
                    self.logger.debug(f"Error computing stability for {feature}: {e}")
                    stability[f"{feature}_cv"] = 0.0
                    stability[f"{feature}_stability_score"] = 1.0
                    stability[f"{feature}_trend_r2"] = 0.0
                    stability[f"{feature}_trend_slope"] = 0.0
        
        return stability
    
    def _compute_anomaly_scores(self, signals: List[ProcessedSignal], isin: str) -> Dict[str, float]:
        """Compute anomaly scores using z-scores."""
        anomalies = {}
        
        if isin not in self.historical_data:
            return anomalies
        
        df = self.historical_data[isin]
        if df.empty or len(df) < 10:  # Need sufficient history for anomaly detection
            return anomalies
        
        # Compute z-scores for key features
        for feature in ["spread_bps", "depth_density", "turnover_velocity", "sentiment_intensity"]:
            if feature in df.columns:
                try:
                    values = df[feature].dropna()
                    if len(values) > 5:
                        # Compute z-score of current value vs historical distribution
                        current_value = values.iloc[-1]
                        historical_mean = values.iloc[:-1].mean()
                        historical_std = values.iloc[:-1].std()
                        
                        if historical_std > 0:
                            z_score = (current_value - historical_mean) / historical_std
                            anomalies[f"{feature}_zscore"] = z_score
                            anomalies[f"{feature}_anomaly_score"] = min(1.0, abs(z_score) / self.anomaly_threshold)
                        else:
                            anomalies[f"{feature}_zscore"] = 0.0
                            anomalies[f"{feature}_anomaly_score"] = 0.0
                    else:
                        anomalies[f"{feature}_zscore"] = 0.0
                        anomalies[f"{feature}_anomaly_score"] = 0.0
                        
                except Exception as e:
                    self.logger.debug(f"Error computing anomaly for {feature}: {e}")
                    anomalies[f"{feature}_zscore"] = 0.0
                    anomalies[f"{feature}_anomaly_score"] = 0.0
        
        return anomalies
    
    def _update_historical_data(self, signals: List[ProcessedSignal], isin: str):
        """Update historical data storage."""
        if not signals:
            return
        
        # Create current data point
        current_features = self._compute_basic_features(self._group_signals_by_type(signals))
        current_features["timestamp"] = datetime.now()
        
        # Initialize historical data if needed
        if isin not in self.historical_data:
            self.historical_data[isin] = pd.DataFrame()
        
        # Add current data point
        new_row = pd.DataFrame([current_features])
        self.historical_data[isin] = pd.concat([self.historical_data[isin], new_row], ignore_index=True)
        
        # Clean up old data
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        self.historical_data[isin] = self.historical_data[isin][
            self.historical_data[isin]["timestamp"] > cutoff_date
        ]
        
        # Limit to reasonable size
        if len(self.historical_data[isin]) > 1000:
            self.historical_data[isin] = self.historical_data[isin].tail(1000)
    
    def _normalize_features(self, features: Dict[str, float], isin: str) -> Dict[str, float]:
        """Normalize features using historical statistics."""
        normalized = {}
        
        for feature_name, feature_value in features.items():
            try:
                # Skip non-numeric features
                if not isinstance(feature_value, (int, float)) or np.isnan(feature_value):
                    normalized[feature_name] = 0.0
                    continue
                
                # Initialize scaler if needed
                if feature_name not in self.scalers:
                    self.scalers[feature_name] = StandardScaler()
                
                # Get historical values for this feature
                if isin in self.historical_data and feature_name in self.historical_data[isin].columns:
                    historical_values = self.historical_data[isin][feature_name].dropna()
                    
                    if len(historical_values) > 5:
                        # Fit scaler on historical data
                        historical_values_2d = historical_values.values.reshape(-1, 1)
                        self.scalers[feature_name].fit(historical_values_2d)
                        
                        # Transform current value
                        current_value_2d = np.array([[feature_value]])
                        normalized_value = self.scalers[feature_name].transform(current_value_2d)[0][0]
                        
                        # Clip to reasonable range
                        normalized[feature_name] = np.clip(normalized_value, -3, 3)
                    else:
                        # Insufficient history, use raw value
                        normalized[feature_name] = feature_value
                else:
                    # No historical data, use raw value
                    normalized[feature_name] = feature_value
                    
            except Exception as e:
                self.logger.debug(f"Error normalizing feature {feature_name}: {e}")
                normalized[feature_name] = feature_value
        
        return normalized
    
    def get_feature_summary(self, isin: str) -> Dict[str, Any]:
        """Get summary of computed features for an ISIN."""
        if isin not in self.historical_data:
            return {"error": "No historical data available"}
        
        df = self.historical_data[isin]
        if df.empty:
            return {"error": "Historical data is empty"}
        
        summary = {
            "isin": isin,
            "data_points": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns else "unknown",
                "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns else "unknown"
            },
            "features": {}
        }
        
        # Feature statistics
        for col in df.columns:
            if col != "timestamp" and df[col].dtype in [np.float64, np.int64]:
                summary["features"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "current": float(df[col].iloc[-1]) if len(df) > 0 else 0.0
                }
        
        return summary
    
    def cleanup_old_data(self):
        """Clean up old historical data."""
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        
        for isin in list(self.historical_data.keys()):
            if "timestamp" in self.historical_data[isin].columns:
                self.historical_data[isin] = self.historical_data[isin][
                    self.historical_data[isin]["timestamp"] > cutoff_date
                ]
                
                # Remove ISINs with no data
                if self.historical_data[isin].empty:
                    del self.historical_data[isin]
                    self.logger.debug(f"Removed empty historical data for {isin}")
