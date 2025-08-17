"""
Advanced Yield Prediction Models

This module implements sophisticated yield forecasting models using multiple machine
learning approaches including ensemble methods, regime-switching models, and factor models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Advanced ML imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

# Statistical imports
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX

# GARCH models for volatility
from arch import arch_model

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of yield prediction models"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    ARIMA = "arima"
    VAR = "var"
    GARCH = "garch"
    ENSEMBLE = "ensemble"

class RegimeType(Enum):
    """Market regime classifications"""
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    MONETARY_TIGHTENING = "monetary_tightening"
    MONETARY_EASING = "monetary_easing"

@dataclass
class YieldPrediction:
    """Yield prediction data structure"""
    predicted_yield: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: int  # days
    model_type: ModelType
    feature_importance: Dict[str, float]
    prediction_time: datetime
    model_confidence: float

@dataclass
class YieldCurveFactors:
    """Yield curve principal components"""
    level: float
    slope: float
    curvature: float
    explained_variance: List[float]

class YieldPredictionEngine:
    """
    Advanced yield prediction engine using multiple ML approaches
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.feature_names = []
        self.is_trained = False
        
        # Initialize base models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all prediction models"""
        # Linear models
        self.models[ModelType.LINEAR_REGRESSION] = LinearRegression()
        self.models[ModelType.RIDGE] = Ridge(alpha=1.0)
        self.models[ModelType.LASSO] = Lasso(alpha=0.1)
        
        # Tree-based models
        self.models[ModelType.RANDOM_FOREST] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models[ModelType.GRADIENT_BOOSTING] = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        
        # Advanced boosting models
        self.models[ModelType.XGBOOST] = xgb.XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.models[ModelType.LIGHTGBM] = lgb.LGBMRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        
        # Neural network
        self.models[ModelType.NEURAL_NETWORK] = MLPRegressor(
            hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
        )
        
        # Time series models
        self.models[ModelType.ARIMA] = None  # Will be initialized per bond
        self.models[ModelType.VAR] = None    # Will be initialized per bond
        self.models[ModelType.GARCH] = None  # Will be initialized per bond
        
    def extract_yield_curve_factors(
        self,
        yield_data: pd.DataFrame,
        maturities: List[str]
    ) -> YieldCurveFactors:
        """
        Extract principal components from yield curve data
        
        Args:
            yield_data: DataFrame with yield data across maturities
            maturities: List of maturity labels
            
        Returns:
            Yield curve factors (level, slope, curvature)
        """
        try:
            # Clean and prepare data
            clean_data = yield_data.dropna()
            
            if clean_data.empty:
                raise ValueError("No valid yield data available")
            
            # Fit PCA
            self.pca.fit(clean_data)
            
            # Extract factors
            level = self.pca.components_[0]
            slope = self.pca.components_[1]
            curvature = self.pca.components_[2]
            
            # Calculate explained variance
            explained_variance = self.pca.explained_variance_ratio_
            
            return YieldCurveFactors(
                level=level,
                slope=slope,
                curvature=curvature,
                explained_variance=explained_variance.tolist()
            )
            
        except Exception as e:
            logger.error(f"Error extracting yield curve factors: {e}")
            # Return default factors
            return YieldCurveFactors(
                level=np.ones(len(maturities)) / len(maturities),
                slope=np.linspace(-1, 1, len(maturities)),
                curvature=np.array([1, 0, -1, 0, 1][:len(maturities)]),
                explained_variance=[0.8, 0.15, 0.05]
            )
    
    def engineer_features(
        self,
        bond_data: pd.DataFrame,
        market_data: pd.DataFrame,
        macroeconomic_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer comprehensive features for yield prediction
        
        Args:
            bond_data: Bond-specific data
            market_data: Market-wide data
            macroeconomic_data: Macroeconomic indicators
            
        Returns:
            Feature matrix for ML models
        """
        try:
            features = pd.DataFrame()
            
            # Bond-specific features
            if not bond_data.empty:
                features['coupon_rate'] = bond_data.get('coupon_rate', 0)
                features['time_to_maturity'] = bond_data.get('time_to_maturity', 0)
                features['credit_rating_numeric'] = bond_data.get('credit_rating_numeric', 0)
                features['issue_size'] = bond_data.get('issue_size', 0)
                features['liquidity_score'] = bond_data.get('liquidity_score', 0)
            
            # Market features
            if not market_data.empty:
                features['benchmark_yield'] = market_data.get('benchmark_yield', 0)
                features['yield_curve_slope'] = market_data.get('yield_curve_slope', 0)
                features['yield_curve_curvature'] = market_data.get('yield_curve_curvature', 0)
                features['market_volatility'] = market_data.get('market_volatility', 0)
                features['credit_spread'] = market_data.get('credit_spread', 0)
                features['liquidity_premium'] = market_data.get('liquidity_premium', 0)
            
            # Macroeconomic features
            if not macroeconomic_data.empty:
                features['gdp_growth'] = macroeconomic_data.get('gdp_growth', 0)
                features['inflation_rate'] = macroeconomic_data.get('inflation_rate', 0)
                features['repo_rate'] = macroeconomic_data.get('repo_rate', 0)
                features['crr_rate'] = macroeconomic_data.get('crr_rate', 0)
                features['slr_rate'] = macroeconomic_data.get('slr_rate', 0)
                features['fiscal_deficit'] = macroeconomic_data.get('fiscal_deficit', 0)
                features['current_account_deficit'] = macroeconomic_data.get('current_account_deficit', 0)
            
            # Technical indicators
            if not market_data.empty:
                # Moving averages
                if 'benchmark_yield' in market_data.columns:
                    features['yield_ma_5'] = market_data['benchmark_yield'].rolling(5).mean()
                    features['yield_ma_20'] = market_data['benchmark_yield'].rolling(20).mean()
                    features['yield_ma_50'] = market_data['benchmark_yield'].rolling(50).mean()
                
                # Momentum indicators
                if 'benchmark_yield' in market_data.columns:
                    features['yield_momentum_5'] = market_data['benchmark_yield'].pct_change(5)
                    features['yield_momentum_20'] = market_data['benchmark_yield'].pct_change(20)
                
                # Volatility indicators
                if 'benchmark_yield' in market_data.columns:
                    features['yield_volatility_20'] = market_data['benchmark_yield'].rolling(20).std()
                    features['yield_volatility_50'] = market_data['benchmark_yield'].rolling(50).std()
            
            # Lagged features
            if 'benchmark_yield' in features.columns:
                features['yield_lag_1'] = features['benchmark_yield'].shift(1)
                features['yield_lag_5'] = features['benchmark_yield'].shift(5)
                features['yield_lag_20'] = features['benchmark_yield'].shift(20)
            
            # Interaction features
            if 'coupon_rate' in features.columns and 'time_to_maturity' in features.columns:
                features['coupon_maturity_interaction'] = features['coupon_rate'] * features['time_to_maturity']
            
            if 'credit_rating_numeric' in features.columns and 'time_to_maturity' in features.columns:
                features['rating_maturity_interaction'] = features['credit_rating_numeric'] * features['time_to_maturity']
            
            # Clean features
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Store feature names
            self.feature_names = features.columns.tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return pd.DataFrame()
    
    def train_models(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all prediction models
        
        Args:
            features: Feature matrix
            targets: Target yield values
            validation_split: Fraction of data for validation
            
        Returns:
            Training results and model performance
        """
        try:
            if features.empty or targets.empty:
                raise ValueError("Empty features or targets")
            
            # Prepare data
            X = features.values
            y = targets.values
            
            # Split data (time series aware)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            training_results = {}
            
            # Train ML models
            for model_type, model in self.models.items():
                if model_type in [ModelType.ARIMA, ModelType.VAR, ModelType.GARCH]:
                    continue  # Skip time series models for now
                
                try:
                    # Train model
                    if hasattr(model, 'fit'):
                        model.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred_train = model.predict(X_train_scaled)
                        y_pred_val = model.predict(X_val_scaled)
                        
                        # Calculate metrics
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                        train_r2 = r2_score(y_train, y_pred_train)
                        val_r2 = r2_score(y_val, y_pred_val)
                        
                        # Feature importance
                        feature_importance = {}
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                        elif hasattr(model, 'coef_'):
                            feature_importance = dict(zip(self.feature_names, model.coef_))
                        
                        training_results[model_type.value] = {
                            'model': model,
                            'train_rmse': train_rmse,
                            'val_rmse': val_rmse,
                            'train_r2': train_r2,
                            'val_r2': val_r2,
                            'feature_importance': feature_importance,
                            'is_trained': True
                        }
                        
                except Exception as e:
                    logger.error(f"Error training {model_type.value}: {e}")
                    training_results[model_type.value] = {
                        'model': model,
                        'error': str(e),
                        'is_trained': False
                    }
            
            # Train time series models
            self._train_time_series_models(features, targets, training_results)
            
            self.is_trained = True
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def _train_time_series_models(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        training_results: Dict
    ):
        """Train time series models (ARIMA, VAR, GARCH)"""
        try:
            # ARIMA model
            try:
                # Determine optimal ARIMA parameters
                p, d, q = self._find_optimal_arima_params(targets)
                arima_model = ARIMA(targets, order=(p, d, q))
                arima_fitted = arima_model.fit()
                
                training_results[ModelType.ARIMA.value] = {
                    'model': arima_fitted,
                    'arima_params': (p, d, q),
                    'aic': arima_fitted.aic,
                    'is_trained': True
                }
                
            except Exception as e:
                logger.error(f"Error training ARIMA: {e}")
                training_results[ModelType.ARIMA.value] = {
                    'error': str(e),
                    'is_trained': False
                }
            
            # VAR model (if we have multiple time series)
            try:
                if len(features.columns) > 1:
                    # Use first few features for VAR
                    var_features = features.iloc[:, :3]  # First 3 features
                    var_model = VARMAX(var_features, order=(2, 0))
                    var_fitted = var_model.fit(disp=False)
                    
                    training_results[ModelType.VAR.value] = {
                        'model': var_fitted,
                        'var_order': (2, 0),
                        'aic': var_fitted.aic,
                        'is_trained': True
                    }
                    
            except Exception as e:
                logger.error(f"Error training VAR: {e}")
                training_results[ModelType.VAR.value] = {
                    'error': str(e),
                    'is_trained': False
                }
            
            # GARCH model for volatility
            try:
                # Use yield changes for GARCH
                yield_changes = targets.pct_change().dropna()
                if len(yield_changes) > 50:  # Need sufficient data
                    garch_model = arch_model(yield_changes, vol='GARCH', p=1, q=1)
                    garch_fitted = garch_model.fit(disp='off')
                    
                    training_results[ModelType.GARCH.value] = {
                        'model': garch_fitted,
                        'garch_params': (1, 1),
                        'aic': garch_fitted.aic,
                        'is_trained': True
                    }
                    
            except Exception as e:
                logger.error(f"Error training GARCH: {e}")
                training_results[ModelType.GARCH.value] = {
                    'error': str(e),
                    'is_trained': False
                }
                
        except Exception as e:
            logger.error(f"Error in time series model training: {e}")
    
    def _find_optimal_arima_params(self, series: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA parameters using AIC"""
        try:
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            # Grid search for optimal parameters
            for p in range(0, 4):
                for d in range(0, 3):
                    for q in range(0, 4):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error finding optimal ARIMA params: {e}")
            return (1, 1, 1)  # Default parameters
    
    def predict_yield(
        self,
        features: pd.DataFrame,
        model_type: ModelType = ModelType.ENSEMBLE,
        prediction_horizon: int = 1
    ) -> YieldPrediction:
        """
        Make yield prediction using specified model
        
        Args:
            features: Feature matrix for prediction
            model_type: Type of model to use
            prediction_horizon: Prediction horizon in days
            
        Returns:
            Yield prediction with confidence intervals
        """
        try:
            if not self.is_trained:
                raise ValueError("Models not trained yet")
            
            if model_type == ModelType.ENSEMBLE:
                return self._ensemble_prediction(features, prediction_horizon)
            
            # Get model results
            model_results = self.models.get(model_type)
            if not model_results or not model_results.get('is_trained', False):
                raise ValueError(f"Model {model_type.value} not available or not trained")
            
            # Make prediction
            if model_type in [ModelType.ARIMA, ModelType.VAR, ModelType.GARCH]:
                prediction = self._time_series_prediction(
                    model_results['model'], model_type, prediction_horizon
                )
            else:
                # ML model prediction
                X_scaled = self.scaler.transform(features.values)
                prediction = model_results['model'].predict(X_scaled)[0]
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                prediction, model_type, model_results
            )
            
            # Get feature importance
            feature_importance = model_results.get('feature_importance', {})
            
            # Calculate model confidence
            model_confidence = self._calculate_model_confidence(model_results)
            
            return YieldPrediction(
                predicted_yield=prediction,
                confidence_interval=confidence_interval,
                prediction_horizon=prediction_horizon,
                model_type=model_type,
                feature_importance=feature_importance,
                prediction_time=datetime.now(),
                model_confidence=model_confidence
            )
            
        except Exception as e:
            logger.error(f"Error making yield prediction: {e}")
            # Return default prediction
            return YieldPrediction(
                predicted_yield=0.0,
                confidence_interval=(0.0, 0.0),
                prediction_horizon=prediction_horizon,
                model_type=model_type,
                feature_importance={},
                prediction_time=datetime.now(),
                model_confidence=0.0
            )
    
    def _ensemble_prediction(
        self,
        features: pd.DataFrame,
        prediction_horizon: int
    ) -> YieldPrediction:
        """Make ensemble prediction using multiple models"""
        try:
            predictions = []
            weights = []
            
            # Collect predictions from all trained models
            for model_type, model_results in self.models.items():
                if model_results.get('is_trained', False):
                    try:
                        pred = self.predict_yield(features, model_type, prediction_horizon)
                        predictions.append(pred.predicted_yield)
                        
                        # Weight by model performance (R² score)
                        weight = model_results.get('val_r2', 0.5)
                        weights.append(max(weight, 0.1))  # Minimum weight
                        
                    except Exception as e:
                        logger.warning(f"Error in ensemble prediction for {model_type.value}: {e}")
                        continue
            
            if not predictions:
                raise ValueError("No valid predictions for ensemble")
            
            # Calculate weighted average
            weights = np.array(weights) / np.sum(weights)
            ensemble_prediction = np.average(predictions, weights=weights)
            
            # Calculate ensemble confidence interval
            confidence_interval = (
                np.percentile(predictions, 5),
                np.percentile(predictions, 95)
            )
            
            # Aggregate feature importance
            feature_importance = self._aggregate_feature_importance()
            
            return YieldPrediction(
                predicted_yield=ensemble_prediction,
                confidence_interval=confidence_interval,
                prediction_horizon=prediction_horizon,
                model_type=ModelType.ENSEMBLE,
                feature_importance=feature_importance,
                prediction_time=datetime.now(),
                model_confidence=0.8  # High confidence for ensemble
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def _time_series_prediction(
        self,
        model,
        model_type: ModelType,
        prediction_horizon: int
    ) -> float:
        """Make prediction using time series models"""
        try:
            if model_type == ModelType.ARIMA:
                forecast = model.forecast(steps=prediction_horizon)
                return forecast[0] if prediction_horizon == 1 else forecast[-1]
            
            elif model_type == ModelType.VAR:
                forecast = model.forecast(steps=prediction_horizon)
                return forecast.iloc[-1, 0]  # First variable
            
            elif model_type == ModelType.GARCH:
                # GARCH models predict volatility, not levels
                # Use last known value as approximation
                return model.data.mean()
            
            else:
                raise ValueError(f"Unsupported time series model: {model_type}")
                
        except Exception as e:
            logger.error(f"Error in time series prediction: {e}")
            return 0.0
    
    def _calculate_confidence_interval(
        self,
        prediction: float,
        model_type: ModelType,
        model_results: Dict
    ) -> Tuple[float, float]:
        """Calculate prediction confidence interval"""
        try:
            # Base confidence interval
            base_interval = 0.1  # 10% of prediction
            
            # Adjust based on model performance
            if 'val_r2' in model_results:
                r2 = model_results['val_r2']
                if r2 > 0.8:
                    base_interval *= 0.5
                elif r2 < 0.5:
                    base_interval *= 2.0
            
            # Adjust based on model type
            if model_type in [ModelType.ARIMA, ModelType.VAR]:
                base_interval *= 1.5  # Time series models typically have higher uncertainty
            
            confidence_interval = (
                prediction * (1 - base_interval),
                prediction * (1 + base_interval)
            )
            
            return confidence_interval
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (prediction * 0.9, prediction * 1.1)
    
    def _calculate_model_confidence(self, model_results: Dict) -> float:
        """Calculate model confidence score"""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on R² score
            if 'val_r2' in model_results:
                r2 = model_results['val_r2']
                confidence += r2 * 0.4  # R² contributes up to 40%
            
            # Adjust based on validation RMSE
            if 'val_rmse' in model_results:
                rmse = model_results['val_rmse']
                # Normalize RMSE (assume typical yield range is 0-15%)
                normalized_rmse = min(rmse / 0.15, 1.0)
                confidence += (1 - normalized_rmse) * 0.1  # RMSE contributes up to 10%
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating model confidence: {e}")
            return 0.5
    
    def _aggregate_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across all models"""
        try:
            aggregated_importance = {}
            
            for model_type, model_results in self.models.items():
                if model_results.get('is_trained', False) and 'feature_importance' in model_results:
                    feature_importance = model_results['feature_importance']
                    weight = model_results.get('val_r2', 0.5)
                    
                    for feature, importance in feature_importance.items():
                        if feature not in aggregated_importance:
                            aggregated_importance[feature] = 0.0
                        aggregated_importance[feature] += importance * weight
            
            # Normalize
            if aggregated_importance:
                max_importance = max(aggregated_importance.values())
                if max_importance > 0:
                    aggregated_importance = {
                        k: v / max_importance for k, v in aggregated_importance.items()
                    }
            
            return aggregated_importance
            
        except Exception as e:
            logger.error(f"Error aggregating feature importance: {e}")
            return {}
    
    def detect_market_regime(
        self,
        market_data: pd.DataFrame,
        window_size: int = 60
    ) -> RegimeType:
        """
        Detect current market regime using volatility and correlation analysis
        
        Args:
            market_data: Market data for regime detection
            window_size: Rolling window size for analysis
            
        Returns:
            Detected market regime
        """
        try:
            if market_data.empty:
                return RegimeType.LOW_VOLATILITY  # Default regime
            
            # Calculate volatility
            if 'benchmark_yield' in market_data.columns:
                yields = market_data['benchmark_yield']
                volatility = yields.rolling(window_size).std()
                current_volatility = volatility.iloc[-1]
                
                # Volatility threshold
                vol_threshold = yields.std() * 1.5
                
                # Calculate correlation with risk assets
                if 'equity_index' in market_data.columns:
                    equity_correlation = yields.rolling(window_size).corr(
                        market_data['equity_index']
                    ).iloc[-1]
                    
                    # Regime classification
                    if current_volatility > vol_threshold:
                        if equity_correlation < -0.3:
                            return RegimeType.RISK_OFF
                        else:
                            return RegimeType.HIGH_VOLATILITY
                    else:
                        if equity_correlation > 0.3:
                            return RegimeType.RISK_ON
                        else:
                            return RegimeType.LOW_VOLATILITY
            
            # Default classification based on volatility only
            if 'current_volatility' in locals() and current_volatility > vol_threshold:
                return RegimeType.HIGH_VOLATILITY
            else:
                return RegimeType.LOW_VOLATILITY
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return RegimeType.LOW_VOLATILITY
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performances"""
        try:
            summary = {}
            
            for model_type, model_results in self.models.items():
                if model_results.get('is_trained', False):
                    summary[model_type.value] = {
                        'is_trained': True,
                        'validation_rmse': model_results.get('val_rmse', None),
                        'validation_r2': model_results.get('val_r2', None),
                        'aic': model_results.get('aic', None)
                    }
                else:
                    summary[model_type.value] = {
                        'is_trained': False,
                        'error': model_results.get('error', 'Unknown error')
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {}
