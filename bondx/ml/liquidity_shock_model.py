"""
Liquidity Shock Model

This module provides ML-based liquidity shock prediction including:
- Directional classification of liquidity changes
- Confidence scoring for predictions
- Feature engineering for liquidity analysis
- Model training and validation
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
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

# Advanced ML imports
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC = "logistic"

class ShockDirection(Enum):
    """Liquidity shock directions"""
    IMPROVEMENT = "improvement"
    DETERIORATION = "deterioration"
    STABLE = "stable"

@dataclass
class LiquidityShockPrediction:
    """Liquidity shock prediction result"""
    issuer_name: str
    sector: str
    current_liquidity_index: float
    predicted_shock_direction: ShockDirection
    predicted_shock_magnitude: float
    confidence: float
    top_shock_factors: List[str]
    model_metadata: Dict[str, Any]

class LiquidityShockModel:
    """ML model for predicting liquidity shocks"""
    
    def __init__(self, seed: int = 42, model_type: ModelType = ModelType.XGBOOST):
        """Initialize the liquidity shock model"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        
        # Model and pipeline
        self.model = None
        self.pipeline = None
        self.feature_columns = []
        self.target_column = 'liquidity_shock_direction'
        
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
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.seed,
                class_weight='balanced'
            )
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.seed
            )
        elif self.model_type == ModelType.XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.seed,
                eval_metric='logloss',
                scale_pos_weight=1.0
            )
        elif self.model_type == ModelType.LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.seed,
                verbose=-1,
                class_weight='balanced'
            )
        elif self.model_type == ModelType.LOGISTIC:
            self.model = LogisticRegression(
                random_state=self.seed,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def calculate_liquidity_shock_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity shock target variable
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            Series with liquidity shock direction (0: stable, 1: deterioration, 2: improvement)
        """
        # Simulate liquidity shock targets based on current metrics
        # In a real scenario, this would be based on historical liquidity changes
        
        shock_targets = pd.Series(0, index=df.index)  # Default to stable
        
        for idx, row in df.iterrows():
            shock_score = 0
            
            # Current liquidity index (if available)
            if 'liquidity_index' in df.columns:
                liq_idx = row['liquidity_index']
                if liq_idx < 0.3:
                    shock_score += 0.3  # Low liquidity increases deterioration risk
                elif liq_idx > 0.8:
                    shock_score += 0.2  # High liquidity increases improvement potential
            
            # Spread risk
            spread = row['spread_bps']
            if spread > 400:
                shock_score += 0.3  # Wide spreads increase deterioration risk
            elif spread < 100:
                shock_score += 0.2  # Tight spreads increase improvement potential
            
            # Sector volatility
            sector = row['sector']
            volatile_sectors = ['Energy', 'Mining', 'Real Estate', 'Financial']
            if sector in volatile_sectors:
                shock_score += 0.2
            
            # Credit rating risk
            rating = row['credit_rating']
            if rating in ['BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C']:
                shock_score += 0.2
            
            # Volume volatility
            volume = row['volume_traded']
            volume_volatility = np.random.normal(0, 0.2)
            shock_score += volume_volatility
            
            # Add market-wide shock component
            market_shock = np.random.normal(0, 0.15)
            shock_score += market_shock
            
            # Determine shock direction
            if shock_score > 0.3:
                shock_targets.iloc[idx] = 1  # Deterioration
            elif shock_score < -0.3:
                shock_targets.iloc[idx] = 2  # Improvement
            else:
                shock_targets.iloc[idx] = 0  # Stable
        
        return shock_targets
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for liquidity shock prediction
        
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
        
        # Calculate liquidity shock target
        df_features['liquidity_shock_direction'] = self.calculate_liquidity_shock_target(df_features)
        
        # Basic numerical features
        numerical_features = [
            'yield_to_maturity', 'volume_traded', 'bid_ask_spread',
            'time_to_maturity', 'coupon_rate', 'issue_size',
            'liquidity_index', 'esg_composite', 'spread_bps'
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
        
        # Create liquidity-specific features
        df_features['liquidity_volatility'] = df_features['liquidity_index'] * (1 - df_features['liquidity_index'])
        df_features['spread_liquidity_ratio'] = df_features['spread_bps'] / (df_features['liquidity_index'] + 0.01)
        df_features['volume_liquidity_ratio'] = df_features['volume_traded'] / (df_features['liquidity_index'] + 0.01)
        
        # Create market stress indicators
        df_features['market_stress'] = (
            (df_features['spread_bps'] > df_features['spread_bps'].quantile(0.8)).astype(int) +
            (df_features['bid_ask_spread'] > df_features['bid_ask_spread'].quantile(0.8)).astype(int) +
            (df_features['liquidity_index'] < df_features['liquidity_index'].quantile(0.2)).astype(int)
        )
        
        # Create sector volatility indicators
        sector_volatility_map = {
            'Energy': 0.8, 'Mining': 0.9, 'Real Estate': 0.7,
            'Financial': 0.6, 'Technology': 0.4, 'Healthcare': 0.3,
            'Consumer Goods': 0.3, 'Utilities': 0.2, 'Construction': 0.6,
            'Automotive': 0.5, 'Infrastructure': 0.5, 'Telecommunications': 0.4
        }
        
        df_features['sector_volatility'] = df_features['sector'].map(sector_volatility_map).fillna(0.5)
        
        # Create rating volatility indicators
        rating_volatility_map = {
            'AAA': 0.1, 'AA+': 0.15, 'AA': 0.2, 'AA-': 0.25,
            'A+': 0.3, 'A': 0.35, 'A-': 0.4, 'BBB+': 0.45,
            'BBB': 0.5, 'BBB-': 0.55, 'BB+': 0.6, 'BB': 0.65,
            'BB-': 0.7, 'B+': 0.75, 'B': 0.8, 'B-': 0.85,
            'CCC': 0.9, 'CC': 0.95, 'C': 1.0
        }
        
        df_features['rating_volatility'] = df_features['credit_rating'].map(rating_volatility_map).fillna(0.5)
        
        # Create interaction features
        df_features['liquidity_sector_volatility'] = df_features['liquidity_index'] * df_features['sector_volatility']
        df_features['liquidity_rating_volatility'] = df_features['liquidity_index'] * df_features['rating_volatility']
        df_features['spread_sector_volatility'] = df_features['spread_bps'] * df_features['sector_volatility']
        
        # Create polynomial features
        df_features['liquidity_squared'] = df_features['liquidity_index'] ** 2
        df_features['spread_squared'] = df_features['spread_bps'] ** 2
        df_features['volume_squared'] = df_features['volume_traded'] ** 2
        
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
            df_features['trade_recency_factor'] = 1 / (df_features['days_since_trade'] + 1)
        
        # Create market depth proxy
        df_features['market_depth'] = df_features['volume_traded'] / (df_features['bid_ask_spread'] + 0.01)
        df_features['market_depth'] = df_features['market_depth'].replace([np.inf, -np.inf], 0)
        
        # Create credit quality composite
        df_features['credit_quality'] = (
            df_features['esg_composite'] * 0.4 +
            df_features['liquidity_index'] * 0.3 +
            (1 - df_features['spread_bps'] / df_features['spread_bps'].max()) * 0.3
        )
        
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
            'liquidity_index', 'esg_composite', 'spread_bps',
            'liquidity_volatility', 'spread_liquidity_ratio', 'volume_liquidity_ratio',
            'market_stress', 'sector_volatility', 'rating_volatility',
            'liquidity_sector_volatility', 'liquidity_rating_volatility',
            'spread_sector_volatility', 'liquidity_squared', 'spread_squared',
            'volume_squared', 'market_depth', 'credit_quality'
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
        
        # Add time-based features
        time_features = ['days_since_trade', 'trade_recency_factor']
        feature_candidates.extend([f for f in time_features if f in df.columns])
        
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
        Train the liquidity shock model
        
        Args:
            df: DataFrame with bond data
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting liquidity shock model training with {self.model_type.value}")
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Prepare features
        X, y = self.prepare_features(df_features)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
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
        
        # Get prediction probabilities
        y_train_proba = self.model.predict_proba(X_train_processed)
        y_test_proba = self.model.predict_proba(X_test_processed)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba, "train")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba, "test")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_processed, y_train, 
            cv=cv_folds, scoring='f1_macro'
        )
        
        # Feature importance
        self.feature_importance = self._extract_feature_importance(X.columns)
        
        # Store training history
        training_result = {
            'model_type': self.model_type.value,
            'feature_count': len(self.feature_columns),
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'target_distribution': y.value_counts().to_dict(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat()
        }
        
        self.training_history.append(training_result)
        
        self.logger.info(f"Training completed. Test F1: {test_metrics['f1']:.3f}")
        
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
                          y_proba: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Calculate ROC AUC for each class
        try:
            if y_proba.shape[1] == 3:  # 3 classes
                # One-vs-rest ROC AUC
                roc_auc_scores = []
                for i in range(3):
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
                        roc_auc_scores.append(auc)
                    else:
                        roc_auc_scores.append(0.5)
                
                metrics['roc_auc_macro'] = np.mean(roc_auc_scores)
                metrics['roc_auc_weighted'] = np.mean(roc_auc_scores)
            else:
                metrics['roc_auc_macro'] = 0.5
                metrics['roc_auc_weighted'] = 0.5
        except:
            metrics['roc_auc_macro'] = 0.5
            metrics['roc_auc_weighted'] = 0.5
        
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        return metrics
    
    def _extract_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For multiclass, take mean of absolute coefficients
            importance = np.mean(np.abs(self.model.coef_), axis=0)
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
    
    def predict_liquidity_shocks(self, df: pd.DataFrame) -> List[LiquidityShockPrediction]:
        """
        Predict liquidity shocks for bonds
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            List of liquidity shock predictions
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
        y_pred = self.model.predict(X_processed)
        y_proba = self.model.predict_proba(X_processed)
        
        # Create prediction results
        predictions = []
        for i, (idx, row) in enumerate(df_features.iterrows()):
            # Get original data row
            orig_row = df.loc[idx]
            
            # Determine shock direction
            pred_class = y_pred[i]
            if pred_class == 0:
                shock_direction = ShockDirection.STABLE
            elif pred_class == 1:
                shock_direction = ShockDirection.DETERIORATION
            else:
                shock_direction = ShockDirection.IMPROVEMENT
            
            # Calculate shock magnitude (confidence in prediction)
            pred_proba = y_proba[i]
            predicted_shock_magnitude = np.max(pred_proba)
            
            # Get top shock factors
            top_factors = self._get_top_shock_factors(X.iloc[i])
            
            # Calculate confidence based on feature importance consistency
            confidence = self._calculate_prediction_confidence(X.iloc[i])
            
            prediction = LiquidityShockPrediction(
                issuer_name=orig_row['issuer_name'],
                sector=orig_row['sector'],
                current_liquidity_index=row.get('liquidity_index', 0.5),
                predicted_shock_direction=shock_direction,
                predicted_shock_magnitude=float(predicted_shock_magnitude),
                confidence=confidence,
                top_shock_factors=top_factors,
                model_metadata={
                    'model_type': self.model_type.value,
                    'training_date': self.training_history[-1]['training_date'] if self.training_history else None
                }
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _get_top_shock_factors(self, features: pd.Series, top_n: int = 5) -> List[str]:
        """Get top shock factors for a prediction"""
        # Get feature importance for available features
        available_features = [col for col in self.feature_columns if col in features.index]
        
        if not available_features:
            return []
        
        # Calculate shock contribution for each feature
        shock_contributions = {}
        for feature in available_features:
            if feature in self.feature_importance:
                # Normalize feature value and multiply by importance
                feature_value = features[feature]
                importance = self.feature_importance[feature]
                shock_contributions[feature] = abs(feature_value) * importance
        
        # Sort by shock contribution
        sorted_factors = sorted(
            shock_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [factor[0] for factor in sorted_factors[:top_n]]
    
    def _calculate_prediction_confidence(self, features: pd.Series) -> float:
        """Calculate prediction confidence based on feature values"""
        # Simple confidence calculation based on feature importance consistency
        available_features = [col for col in self.feature_columns if col in features.index]
        
        if not available_features:
            return 0.8
        
        # Calculate weighted confidence based on feature importance
        total_importance = 0
        weighted_confidence = 0
        
        for feature in available_features:
            if feature in self.feature_importance:
                importance = self.feature_importance[feature]
                feature_value = features[feature]
                
                # Normalize feature value to 0-1 range
                if feature_value != 0:
                    normalized_value = min(1.0, abs(feature_value) / 100)  # Rough normalization
                else:
                    normalized_value = 0.5
                
                weighted_confidence += importance * normalized_value
                total_importance += importance
        
        if total_importance == 0:
            return 0.8
        
        confidence = weighted_confidence / total_importance
        return max(0.1, min(0.95, confidence))
    
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
        metadata = {
            'model_type': self.model_type.value,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'seed': self.seed
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
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
        self.training_history = metadata_dict.get('training_history', [])
        self.feature_importance = metadata_dict.get('feature_importance', {})
        
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
        y_proba = self.model.predict_proba(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred, y_proba, "evaluation")
        
        # Generate classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Feature importance analysis
        feature_analysis = {
            'top_features': list(self.feature_importance.keys())[:10],
            'feature_importance_scores': dict(list(self.feature_importance.items())[:10])
        }
        
        evaluation_result = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'feature_analysis': feature_analysis,
            'evaluation_date': datetime.now().isoformat(),
            'sample_count': len(X)
        }
        
        return evaluation_result
