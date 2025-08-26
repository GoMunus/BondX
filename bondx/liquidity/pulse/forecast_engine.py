"""
Forecast Engine for Liquidity Pulse

This module provides short-term forecasting (T+1 to T+5) for the liquidity index
using temporal models with lagged features and calibrated confidence bands.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from ...core.logging import get_logger
from ...api.v1.schemas_liquidity import ForecastPoint
from .feature_engine import FeatureSet

logger = get_logger(__name__)

@dataclass
class ForecastModel:
    """Forecast model configuration and state."""
    model_id: str
    horizon_days: int
    model_type: str
    features: List[str]
    scaler: StandardScaler
    model: Any
    last_trained: datetime
    performance_metrics: Dict[str, float]
    confidence_calibration: Dict[str, float]

class ForecastEngine:
    """Engine for generating liquidity index forecasts."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Forecast configuration
        self.forecast_horizons = config.get("forecast_horizons", [1, 2, 3, 4, 5])  # days
        self.max_history_days = config.get("max_history_days", 90)
        self.min_training_samples = config.get("min_training_samples", 30)
        self.confidence_levels = config.get("confidence_levels", [0.68, 0.95])  # 1σ, 2σ
        
        # Model configuration
        self.model_type = config.get("model_type", "gradient_boosting")  # or "linear"
        self.feature_lags = config.get("feature_lags", [1, 2, 3, 5, 7])  # days
        self.target_lags = config.get("target_lags", [1, 2, 3, 5, 7])  # days
        
        # Model storage
        self.models: Dict[int, ForecastModel] = {}
        self.historical_forecasts: Dict[str, pd.DataFrame] = {}
        self.performance_history: Dict[str, pd.DataFrame] = {}
        
        # Initialize models for each horizon
        self._initialize_models()
        
        self.logger.info(f"Forecast engine initialized with {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize forecast models for each horizon."""
        for horizon in self.forecast_horizons:
            model_id = f"liquidity_forecast_{horizon}d"
            
            # Initialize scaler
            scaler = StandardScaler()
            
            # Initialize model
            if self.model_type == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42,
                    subsample=0.8
                )
            else:  # linear
                model = LinearRegression()
            
            # Create forecast model
            forecast_model = ForecastModel(
                model_id=model_id,
                horizon_days=horizon,
                model_type=self.model_type,
                features=[],
                scaler=scaler,
                model=model,
                last_trained=datetime.now(),
                performance_metrics={},
                confidence_calibration={}
            )
            
            self.models[horizon] = forecast_model
    
    def generate_forecasts(self, feature_set: FeatureSet, historical_liquidity: List[float]) -> List[ForecastPoint]:
        """Generate forecasts for all horizons."""
        forecasts = []
        
        try:
            for horizon in self.forecast_horizons:
                if horizon in self.models:
                    forecast = self._generate_single_forecast(
                        feature_set, historical_liquidity, horizon
                    )
                    if forecast:
                        forecasts.append(forecast)
                else:
                    # Create default forecast with high uncertainty
                    forecast = ForecastPoint(
                        horizon_d=horizon,
                        liquidity_index=feature_set.features.get("liquidity_index", 50.0),
                        confidence=0.1,  # Low confidence for untrained model
                        upper_bound=None,
                        lower_bound=None
                    )
                    forecasts.append(forecast)
            
            # Sort by horizon
            forecasts.sort(key=lambda x: x.horizon_d)
            
        except Exception as e:
            self.logger.error(f"Error generating forecasts: {e}")
            # Return default forecasts on error
            forecasts = [
                ForecastPoint(
                    horizon_d=horizon,
                    liquidity_index=50.0,
                    confidence=0.1,
                    upper_bound=None,
                    lower_bound=None
                )
                for horizon in self.forecast_horizons
            ]
        
        return forecasts
    
    def _generate_single_forecast(self, feature_set: FeatureSet, historical_liquidity: List[float], horizon: int) -> Optional[ForecastPoint]:
        """Generate forecast for a single horizon."""
        try:
            model = self.models[horizon]
            
            # Prepare features for forecasting
            forecast_features = self._prepare_forecast_features(feature_set, historical_liquidity, horizon)
            
            if forecast_features is None:
                return None
            
            # Scale features
            features_scaled = model.scaler.transform([forecast_features])
            
            # Generate prediction
            prediction = model.model.predict(features_scaled)[0]
            
            # Clip to valid range
            prediction = np.clip(prediction, 0, 100)
            
            # Calculate confidence
            confidence = self._calculate_forecast_confidence(model, forecast_features, horizon)
            
            # Calculate confidence bounds
            upper_bound, lower_bound = self._calculate_confidence_bounds(
                prediction, confidence, model.confidence_calibration
            )
            
            return ForecastPoint(
                horizon_d=horizon,
                liquidity_index=float(prediction),
                confidence=float(confidence),
                upper_bound=float(upper_bound) if upper_bound is not None else None,
                lower_bound=float(lower_bound) if lower_bound is not None else None
            )
            
        except Exception as e:
            self.logger.error(f"Error generating forecast for horizon {horizon}: {e}")
            return None
    
    def _prepare_forecast_features(self, feature_set: FeatureSet, historical_liquidity: List[float], horizon: int) -> Optional[np.ndarray]:
        """Prepare features for forecasting."""
        try:
            features = []
            
            # Current feature values
            for feature_name in self._get_feature_names():
                if feature_name in feature_set.features:
                    features.append(feature_set.features[feature_name])
                else:
                    features.append(0.0)  # Default value for missing features
            
            # Lagged liquidity values
            for lag in self.target_lags:
                if len(historical_liquidity) >= lag:
                    features.append(historical_liquidity[-lag])
                else:
                    features.append(50.0)  # Default value if insufficient history
            
            # Lagged feature values (if available)
            if hasattr(feature_set, 'historical_features') and feature_set.historical_features:
                for lag in self.feature_lags:
                    if len(feature_set.historical_features) >= lag:
                        lagged_features = feature_set.historical_features[-lag]
                        for feature_name in self._get_feature_names():
                            if feature_name in lagged_features:
                                features.append(lagged_features[feature_name])
                            else:
                                features.append(0.0)
                    else:
                        # Add default values for missing lags
                        for _ in self._get_feature_names():
                            features.append(0.0)
            
            # Add time-based features
            now = datetime.now()
            features.extend([
                now.weekday() / 6.0,  # Day of week (0-1)
                now.hour / 23.0,      # Hour of day (0-1)
                now.month / 12.0,     # Month (0-1)
            ])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error preparing forecast features: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names used in forecasting."""
        return [
            "spread_bps", "depth_density", "turnover_velocity", "sentiment_intensity",
            "auction_demand", "mm_online_ratio", "mm_spread_bps",
            "traffic_index", "utilities_index", "buzz_volume"
        ]
    
    def _calculate_forecast_confidence(self, model: ForecastModel, features: np.ndarray, horizon: int) -> float:
        """Calculate forecast confidence based on model performance and feature quality."""
        try:
            # Base confidence from model performance
            base_confidence = 0.5  # Default confidence
            
            if model.performance_metrics:
                # Use historical performance metrics
                mae = model.performance_metrics.get("mae", 10.0)
                rmse = model.performance_metrics.get("rmse", 15.0)
                
                # Convert error metrics to confidence (lower error = higher confidence)
                error_confidence = max(0.1, 1.0 - (rmse / 100.0))
                base_confidence = error_confidence
            
            # Adjust confidence based on horizon (longer horizon = lower confidence)
            horizon_decay = 1.0 / (1.0 + horizon * 0.2)
            
            # Adjust confidence based on feature quality
            feature_quality = self._assess_feature_quality(features)
            
            # Combine confidence factors
            final_confidence = base_confidence * horizon_decay * feature_quality
            
            # Clip to valid range
            return np.clip(final_confidence, 0.1, 0.95)
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast confidence: {e}")
            return 0.3  # Default low confidence
    
    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """Assess the quality of input features."""
        try:
            # Check for missing or extreme values
            if features is None or len(features) == 0:
                return 0.1
            
            # Count non-zero features (proxy for data availability)
            non_zero_count = np.count_nonzero(features)
            total_features = len(features)
            
            if total_features == 0:
                return 0.1
            
            availability_score = non_zero_count / total_features
            
            # Check for extreme values (outliers)
            if len(features) > 0:
                z_scores = np.abs((features - np.mean(features)) / (np.std(features) + 1e-8))
                outlier_ratio = np.mean(z_scores > 3.0)  # Features with z-score > 3
                outlier_penalty = 1.0 - outlier_ratio
            else:
                outlier_penalty = 1.0
            
            # Combine scores
            quality_score = availability_score * outlier_penalty
            
            return np.clip(quality_score, 0.1, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error assessing feature quality: {e}")
            return 0.5  # Default medium quality
    
    def _calculate_confidence_bounds(self, prediction: float, confidence: float, calibration: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate confidence bounds for the forecast."""
        try:
            if not calibration:
                # Use default calibration if not available
                default_std = 10.0  # Default standard deviation
            else:
                # Use calibrated standard deviation
                default_std = calibration.get("std", 10.0)
            
            # Adjust standard deviation based on confidence
            adjusted_std = default_std * (1.0 - confidence)
            
            # Calculate bounds
            upper_bound = min(100.0, prediction + adjusted_std)
            lower_bound = max(0.0, prediction - adjusted_std)
            
            return upper_bound, lower_bound
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence bounds: {e}")
            return None, None
    
    def train_models(self, training_data: Dict[str, List[Tuple[FeatureSet, float]]]):
        """Train forecast models using historical data."""
        try:
            for horizon in self.forecast_horizons:
                if horizon in self.models:
                    self._train_single_model(horizon, training_data)
            
            self.logger.info("Forecast models training completed")
            
        except Exception as e:
            self.logger.error(f"Error training forecast models: {e}")
    
    def _train_single_model(self, horizon: int, training_data: Dict[str, List[Tuple[FeatureSet, float]]]):
        """Train a single forecast model."""
        try:
            model = self.models[horizon]
            
            # Prepare training data for this horizon
            X_train, y_train = self._prepare_training_data(horizon, training_data)
            
            if len(X_train) < self.min_training_samples:
                self.logger.warning(f"Insufficient training data for horizon {horizon}: {len(X_train)} samples")
                return
            
            # Scale features
            X_scaled = model.scaler.fit_transform(X_train)
            
            # Train model
            model.model.fit(X_scaled, y_train)
            
            # Update model features
            model.features = self._get_feature_names()
            
            # Evaluate performance
            y_pred = model.model.predict(X_scaled)
            mae = mean_absolute_error(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            
            # Update performance metrics
            model.performance_metrics = {
                "mae": mae,
                "rmse": rmse,
                "r2": model.model.score(X_scaled, y_train),
                "training_samples": len(X_train)
            }
            
            # Update confidence calibration
            residuals = y_train - y_pred
            model.confidence_calibration = {
                "std": float(np.std(residuals)),
                "mean": float(np.mean(residuals)),
                "q95": float(np.percentile(np.abs(residuals), 95))
            }
            
            # Update training timestamp
            model.last_trained = datetime.now()
            
            self.logger.info(f"Model {horizon}d trained with {len(X_train)} samples, RMSE: {rmse:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error training model for horizon {horizon}: {e}")
    
    def _prepare_training_data(self, horizon: int, training_data: Dict[str, List[Tuple[FeatureSet, float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific horizon."""
        X_train = []
        y_train = []
        
        try:
            for isin, data_points in training_data.items():
                for i, (feature_set, liquidity_value) in enumerate(data_points):
                    # Skip if we don't have enough future data for this horizon
                    if i + horizon >= len(data_points):
                        continue
                    
                    # Target is the liquidity value at horizon days ahead
                    target_liquidity = data_points[i + horizon][1]
                    
                    # Prepare features
                    features = self._prepare_forecast_features(feature_set, [dp[1] for dp in data_points[:i+1]], horizon)
                    
                    if features is not None:
                        X_train.append(features)
                        y_train.append(target_liquidity)
            
            return np.array(X_train), np.array(y_train)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data for horizon {horizon}: {e}")
            return np.array([]), np.array([])
    
    def update_performance_history(self, isin: str, actual_liquidity: float, forecasts: List[ForecastPoint]):
        """Update performance history with actual vs forecasted values."""
        try:
            if isin not in self.performance_history:
                self.performance_history[isin] = pd.DataFrame()
            
            # Create performance record
            for forecast in forecasts:
                performance_record = {
                    "timestamp": datetime.now(),
                    "horizon_d": forecast.horizon_d,
                    "forecasted": forecast.liquidity_index,
                    "actual": actual_liquidity,
                    "error": actual_liquidity - forecast.liquidity_index,
                    "abs_error": abs(actual_liquidity - forecast.liquidity_index),
                    "confidence": forecast.confidence
                }
                
                # Add to performance history
                new_row = pd.DataFrame([performance_record])
                self.performance_history[isin] = pd.concat([self.performance_history[isin], new_row], ignore_index=True)
            
            # Clean up old performance data
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            self.performance_history[isin] = self.performance_history[isin][
                self.performance_history[isin]["timestamp"] > cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error updating performance history for {isin}: {e}")
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all models."""
        performance = {}
        
        for horizon, model in self.models.items():
            performance[f"{horizon}d"] = {
                "model_type": model.model_type,
                "last_trained": model.last_trained.isoformat(),
                "training_samples": model.performance_metrics.get("training_samples", 0),
                "mae": model.performance_metrics.get("mae", 0.0),
                "rmse": model.performance_metrics.get("rmse", 0.0),
                "r2": model.performance_metrics.get("r2", 0.0),
                "confidence_calibration": model.confidence_calibration
            }
        
        return performance
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        try:
            model_data = {}
            for horizon, model in self.models.items():
                model_data[horizon] = {
                    "model": model.model,
                    "scaler": model.scaler,
                    "features": model.features,
                    "performance_metrics": model.performance_metrics,
                    "confidence_calibration": model.confidence_calibration
                }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        try:
            model_data = joblib.load(filepath)
            
            for horizon, data in model_data.items():
                if horizon in self.models:
                    self.models[horizon].model = data["model"]
                    self.models[horizon].scaler = data["scaler"]
                    self.models[horizon].features = data["features"]
                    self.models[horizon].performance_metrics = data["performance_metrics"]
                    self.models[horizon].confidence_calibration = data["confidence_calibration"]
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def cleanup_old_data(self):
        """Clean up old performance history data."""
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        
        for isin in list(self.performance_history.keys()):
            if "timestamp" in self.performance_history[isin].columns:
                self.performance_history[isin] = self.performance_history[isin][
                    self.performance_history[isin]["timestamp"] > cutoff_date
                ]
                
                # Remove ISINs with no data
                if self.performance_history[isin].empty:
                    del self.performance_history[isin]
                    self.logger.debug(f"Removed empty performance history for {isin}")
