"""
Spread Prediction Model

This module provides ML-based spread prediction capabilities including:
- Feature engineering for bond data
- Model training and validation
- SHAP-based explainability
- Model persistence and metadata
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    TimeSeriesSplit, validation_curve
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Advanced ML imports
import xgboost as xgb
import lightgbm as lgb
import shap

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"

@dataclass
class ModelMetadata:
    """Model metadata for persistence"""
    model_name: str
    model_type: str
    feature_columns: List[str]
    target_column: str
    training_date: datetime
    seed: int
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    data_shape: Tuple[int, int]
    version: str = "1.0.0"

@dataclass
class PredictionResult:
    """Prediction result with confidence and explanations"""
    predicted_spread: float
    confidence: float
    feature_contributions: Dict[str, float]
    top_features: List[str]
    model_metadata: ModelMetadata

class SpreadPredictionModel:
    """ML model for predicting bond yield spreads"""
    
    def __init__(self, seed: int = 42, model_type: ModelType = ModelType.XGBOOST):
        """Initialize the spread prediction model"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        
        # Model and pipeline
        self.model = None
        self.pipeline = None
        self.feature_columns = []
        self.target_column = 'spread_bps'
        
        # Preprocessing components
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        
        # Performance tracking
        self.training_history = []
        self.feature_importance = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.seed
            )
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.seed
            )
        elif self.model_type == ModelType.XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.seed,
                eval_metric='rmse'
            )
        elif self.model_type == ModelType.LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.seed,
                verbose=-1
            )
        elif self.model_type == ModelType.LINEAR:
            self.model = LinearRegression()
        elif self.model_type == ModelType.RIDGE:
            self.model = Ridge(alpha=1.0, random_state=self.seed)
        elif self.model_type == ModelType.LASSO:
            self.model = Lasso(alpha=0.1, random_state=self.seed)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for spread prediction
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Ensure liquidity index exists
        if 'liquidity_index' not in df_features.columns:
            from ..analysis.liquidity_insights import LiquidityInsightsEngine
            liq_engine = LiquidityInsightsEngine(seed=self.seed)
            df_features['liquidity_index'] = liq_engine.calculate_liquidity_index(df_features)
        
        # Ensure ESG scores exist
        if 'esg_composite' not in df_features.columns:
            from ..analysis.heatmaps import SectorHeatmapEngine
            heatmap_engine = SectorHeatmapEngine(seed=self.seed)
            df_features = heatmap_engine.calculate_esg_scores(df_features)
        
        # Basic numerical features
        numerical_features = [
            'yield_to_maturity', 'volume_traded', 'bid_ask_spread',
            'time_to_maturity', 'coupon_rate', 'issue_size',
            'liquidity_index', 'esg_composite'
        ]
        
        # Categorical features
        categorical_features = ['sector', 'credit_rating', 'issuer_class']
        
        # Create feature encodings
        for col in categorical_features:
            if col in df_features.columns:
                # Label encoding for categorical variables
                le = LabelEncoder()
                df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
                self.label_encoders[col] = le
        
        # Create interaction features
        df_features['liquidity_volume'] = df_features['liquidity_index'] * df_features['volume_traded']
        df_features['spread_liquidity'] = df_features['bid_ask_spread'] * df_features['liquidity_index']
        df_features['yield_maturity'] = df_features['yield_to_maturity'] * df_features['time_to_maturity']
        
        # Create polynomial features for key numerical variables
        df_features['liquidity_squared'] = df_features['liquidity_index'] ** 2
        df_features['volume_squared'] = df_features['volume_traded'] ** 2
        df_features['spread_squared'] = df_features['bid_ask_spread'] ** 2
        
        # Create sector-specific features
        for sector in df_features['sector'].unique():
            sector_mask = df_features['sector'] == sector
            df_features[f'{sector}_indicator'] = sector_mask.astype(int)
        
        # Create rating-specific features
        rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
                       'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 
                       'B+', 'B', 'B-', 'CCC', 'CC', 'C']
        
        for i, rating in enumerate(rating_order):
            rating_mask = df_features['credit_rating'] == rating
            df_features[f'rating_{rating}'] = rating_mask.astype(int)
        
        # Create time-based features (if available)
        if 'last_trade_date' in df_features.columns:
            df_features['last_trade_date'] = pd.to_datetime(df_features['last_trade_date'])
            df_features['days_since_trade'] = (datetime.now() - df_features['last_trade_date']).dt.days
        
        # Create market depth proxy
        df_features['market_depth'] = df_features['volume_traded'] / df_features['bid_ask_spread']
        df_features['market_depth'] = df_features['market_depth'].replace([np.inf, -np.inf], 0)
        
        # Create credit quality proxy
        df_features['credit_quality'] = df_features['esg_composite'] * df_features['liquidity_index']
        
        # Handle missing values
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numerical_cols] = df_features[numerical_cols].fillna(df_features[numerical_cols].median())
        
        return df_features
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (feature DataFrame, feature column names)
        """
        # Select numerical features for modeling
        feature_candidates = [
            'yield_to_maturity', 'volume_traded', 'bid_ask_spread',
            'time_to_maturity', 'coupon_rate', 'issue_size',
            'liquidity_index', 'esg_composite', 'liquidity_volume',
            'spread_liquidity', 'yield_maturity', 'liquidity_squared',
            'volume_squared', 'spread_squared', 'market_depth',
            'credit_quality', 'days_since_trade'
        ]
        
        # Add encoded categorical features
        for col in ['sector', 'credit_rating', 'issuer_class']:
            if f'{col}_encoded' in df.columns:
                feature_candidates.append(f'{col}_encoded')
        
        # Add sector indicators
        sector_cols = [col for col in df.columns if col.endswith('_indicator')]
        feature_candidates.extend(sector_cols)
        
        # Add rating indicators
        rating_cols = [col for col in df.columns if col.startswith('rating_')]
        feature_candidates.extend(rating_cols)
        
        # Filter to available features
        available_features = [col for col in feature_candidates if col in df.columns]
        
        # Ensure target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Create feature matrix
        X = df[available_features].copy()
        y = df[self.target_column].copy()
        
        # Remove rows with missing values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.feature_columns = available_features
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, 
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the spread prediction model
        
        Args:
            df: DataFrame with bond data
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting model training with {self.model_type.value}")
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df_features)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Target vector shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )
        
        # Create preprocessing pipeline
        self.pipeline = self._create_preprocessing_pipeline()
        
        # Fit preprocessing pipeline
        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        
        # Train model
        self.model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_processed)
        y_test_pred = self.model.predict(X_test_processed)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, "train")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, "test")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_processed, y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        # Feature importance
        self.feature_importance = self._extract_feature_importance(X.columns)
        
        # Store training history
        training_result = {
            'model_type': self.model_type.value,
            'feature_count': len(self.feature_columns),
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat()
        }
        
        self.training_history.append(training_result)
        
        self.logger.info(f"Training completed. Test RMSE: {test_metrics['rmse']:.2f}")
        
        return training_result
    
    def _create_preprocessing_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline"""
        # Numerical features preprocessing
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.feature_columns)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          prefix: str = "") -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred)
        }
        
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        return metrics
    
    def _extract_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            # Fallback: equal importance
            importance = np.ones(len(feature_names)) / len(feature_names)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return feature_importance
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            Array of predicted spreads
        """
        if self.pipeline is None or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Prepare features
        X, _ = self.prepare_features(df_features)
        
        # Apply preprocessing
        X_processed = self.pipeline.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_with_explanation(self, df: pd.DataFrame, 
                               top_features: int = 10) -> List[PredictionResult]:
        """
        Make predictions with SHAP-based explanations
        
        Args:
            df: DataFrame with bond data
            top_features: Number of top features to include
            
        Returns:
            List of prediction results with explanations
        """
        if self.model_type not in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]:
            self.logger.warning("SHAP explanations not available for this model type")
            return self._simple_predictions(df)
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Prepare features
        X, _ = self.prepare_features(df_features)
        
        # Apply preprocessing
        X_processed = self.pipeline.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Calculate SHAP values
        try:
            if self.model_type == ModelType.XGBOOST:
                explainer = shap.TreeExplainer(self.model)
            elif self.model_type == ModelType.LIGHTGBM:
                explainer = shap.TreeExplainer(self.model)
            elif self.model_type == ModelType.RANDOM_FOREST:
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = None
            
            if explainer:
                shap_values = explainer.shap_values(X_processed)
                
                # Create prediction results
                results = []
                for i, pred in enumerate(predictions):
                    # Get feature contributions for this prediction
                    if len(shap_values.shape) == 2:
                        contributions = dict(zip(self.feature_columns, shap_values[i]))
                    else:
                        contributions = dict(zip(self.feature_columns, shap_values))
                    
                    # Sort features by absolute contribution
                    sorted_features = sorted(
                        contributions.items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )
                    
                    top_features_list = [f[0] for f in sorted_features[:top_features]]
                    
                    # Calculate confidence (based on feature importance consistency)
                    confidence = self._calculate_prediction_confidence(contributions)
                    
                    result = PredictionResult(
                        predicted_spread=float(pred),
                        confidence=confidence,
                        feature_contributions=contributions,
                        top_features=top_features_list,
                        model_metadata=self._create_model_metadata()
                    )
                    
                    results.append(result)
                
                return results
            else:
                return self._simple_predictions(df)
                
        except Exception as e:
            self.logger.warning(f"SHAP calculation failed: {e}")
            return self._simple_predictions(df)
    
    def _simple_predictions(self, df: pd.DataFrame) -> List[PredictionResult]:
        """Fallback to simple predictions without SHAP"""
        predictions = self.predict(df)
        
        results = []
        for pred in predictions:
            result = PredictionResult(
                predicted_spread=float(pred),
                confidence=0.8,  # Default confidence
                feature_contributions={},
                top_features=self.feature_columns[:5],
                model_metadata=self._create_model_metadata()
            )
            results.append(result)
        
        return results
    
    def _calculate_prediction_confidence(self, contributions: Dict[str, float]) -> float:
        """Calculate prediction confidence based on feature contributions"""
        if not contributions:
            return 0.8
        
        # Calculate confidence based on consistency of feature contributions
        abs_contributions = [abs(v) for v in contributions.values()]
        total_contribution = sum(abs_contributions)
        
        if total_contribution == 0:
            return 0.5
        
        # Normalize contributions
        normalized_contributions = [v / total_contribution for v in abs_contributions]
        
        # Calculate entropy (lower entropy = higher confidence)
        entropy = -sum(p * np.log2(p + 1e-10) for p in normalized_contributions if p > 0)
        max_entropy = np.log2(len(normalized_contributions))
        
        if max_entropy == 0:
            return 0.8
        
        # Convert entropy to confidence (0-1)
        confidence = 1 - (entropy / max_entropy)
        
        return max(0.1, min(0.95, confidence))
    
    def _create_model_metadata(self) -> ModelMetadata:
        """Create model metadata"""
        if not self.training_history:
            raise ValueError("Model not trained. Call train() first.")
        
        latest_training = self.training_history[-1]
        
        return ModelMetadata(
            model_name=f"spread_prediction_{self.model_type.value}",
            model_type=self.model_type.value,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            training_date=datetime.fromisoformat(latest_training['training_date']),
            seed=self.seed,
            performance_metrics=latest_training['test_metrics'],
            feature_importance=self.feature_importance,
            hyperparameters=self._get_hyperparameters(),
            data_shape=(latest_training['training_samples'], len(self.feature_columns))
        )
    
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return {'model_type': self.model_type.value}
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = f"{filepath}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save pipeline
        pipeline_path = f"{filepath}_pipeline.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # Save metadata
        metadata_path = f"{filepath}_metadata.json"
        metadata = self._create_model_metadata()
        with open(metadata_path, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)
        
        # Save feature importance
        importance_path = f"{filepath}_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and metadata"""
        # Load model
        model_path = f"{filepath}_model.pkl"
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load pipeline
        pipeline_path = f"{filepath}_pipeline.pkl"
        with open(pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Update instance variables
        self.feature_columns = metadata_dict['feature_columns']
        self.target_column = metadata_dict['target_column']
        self.seed = metadata_dict['seed']
        
        # Load feature importance
        importance_path = f"{filepath}_importance.json"
        with open(importance_path, 'r') as f:
            self.feature_importance = json.load(f)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def evaluate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df_features)
        
        # Apply preprocessing
        X_processed = self.pipeline.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred, "evaluation")
        
        # Calculate additional metrics
        metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y - y_pred) / y)) * 100
        
        # Feature importance analysis
        feature_analysis = {
            'top_features': list(self.feature_importance.keys())[:10],
            'feature_importance_scores': dict(list(self.feature_importance.items())[:10])
        }
        
        evaluation_result = {
            'metrics': metrics,
            'feature_analysis': feature_analysis,
            'evaluation_date': datetime.now().isoformat(),
            'sample_count': len(X)
        }
        
        return evaluation_result
