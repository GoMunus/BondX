"""
Downgrade Risk Model

This module provides ML-based downgrade risk prediction including:
- Binary and multiclass classification for rating deterioration
- Feature engineering for credit risk analysis
- Model training and validation
- Risk probability calibration
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

class RiskLevel(Enum):
    """Risk levels for downgrade classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DowngradePrediction:
    """Downgrade risk prediction result"""
    issuer_name: str
    current_rating: str
    predicted_downgrade_prob: float
    risk_level: RiskLevel
    confidence: float
    top_risk_factors: List[str]
    model_metadata: Dict[str, Any]

class DowngradeRiskModel:
    """ML model for predicting bond rating downgrade risk"""
    
    def __init__(self, seed: int = 42, model_type: ModelType = ModelType.XGBOOST):
        """Initialize the downgrade risk model"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        
        # Model and pipeline
        self.model = None
        self.pipeline = None
        self.feature_columns = []
        self.target_column = 'downgrade_risk'
        
        # Preprocessing components
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        
        # Performance tracking
        self.training_history = []
        self.feature_importance = {}
        
        # Rating hierarchy for downgrade calculation
        self.rating_hierarchy = {
            'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
            'A+': 17, 'A': 16, 'A-': 15, 'BBB+': 14,
            'BBB': 13, 'BBB-': 12, 'BB+': 11, 'BB': 10,
            'BB-': 9, 'B+': 8, 'B': 7, 'B-': 6,
            'CCC': 5, 'CC': 4, 'C': 3, 'D': 2, 'NR': 1
        }
        
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
    
    def calculate_downgrade_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate downgrade risk target variable
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            Series with downgrade risk (1 for downgrade, 0 for no downgrade)
        """
        # Simulate downgrade risk based on current metrics
        # In a real scenario, this would be based on historical rating changes
        
        downgrade_risk = pd.Series(0, index=df.index)
        
        for idx, row in df.iterrows():
            risk_score = 0
            
            # Credit rating risk
            rating = row['credit_rating']
            if rating in ['BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C']:
                risk_score += 0.4  # High yield bonds have higher downgrade risk
            
            # Spread risk
            spread = row['spread_bps']
            if spread > 300:
                risk_score += 0.3
            elif spread > 200:
                risk_score += 0.2
            elif spread > 100:
                risk_score += 0.1
            
            # Sector risk
            sector = row['sector']
            high_risk_sectors = ['Energy', 'Mining', 'Real Estate']
            if sector in high_risk_sectors:
                risk_score += 0.2
            
            # Liquidity risk (if available)
            if 'liquidity_index' in df.columns:
                liq_idx = row['liquidity_index']
                if liq_idx < 0.3:
                    risk_score += 0.2
                elif liq_idx < 0.5:
                    risk_score += 0.1
            
            # ESG risk (if available)
            if 'esg_composite' in df.columns:
                esg_score = row['esg_composite']
                if esg_score < 0.4:
                    risk_score += 0.2
                elif esg_score < 0.6:
                    risk_score += 0.1
            
            # Add some randomness to simulate real-world uncertainty
            risk_score += np.random.normal(0, 0.1)
            risk_score = max(0, min(1, risk_score))
            
            # Convert to binary target (threshold at 0.5)
            downgrade_risk.iloc[idx] = 1 if risk_score > 0.5 else 0
        
        return downgrade_risk
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for downgrade risk prediction
        
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
        
        # Calculate downgrade target
        df_features['downgrade_risk'] = self.calculate_downgrade_target(df_features)
        
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
        
        # Create rating-based features
        df_features['rating_numeric'] = df_features['credit_rating'].map(self.rating_hierarchy)
        df_features['rating_numeric'] = df_features['rating_numeric'].fillna(10)  # Default to BB
        
        # Create interaction features
        df_features['spread_liquidity'] = df_features['spread_bps'] * df_features['liquidity_index']
        df_features['yield_maturity'] = df_features['yield_to_maturity'] * df_features['time_to_maturity']
        df_features['esg_liquidity'] = df_features['esg_composite'] * df_features['liquidity_index']
        
        # Create polynomial features for key numerical variables
        df_features['spread_squared'] = df_features['spread_bps'] ** 2
        df_features['liquidity_squared'] = df_features['liquidity_index'] ** 2
        df_features['esg_squared'] = df_features['esg_composite'] ** 2
        
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
        
        # Create market stress indicators
        df_features['market_stress'] = (
            (df_features['spread_bps'] > df_features['spread_bps'].quantile(0.8)).astype(int) +
            (df_features['bid_ask_spread'] > df_features['bid_ask_spread'].quantile(0.8)).astype(int) +
            (df_features['liquidity_index'] < df_features['liquidity_index'].quantile(0.2)).astype(int)
        )
        
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
            'liquidity_index', 'esg_composite', 'rating_numeric',
            'spread_liquidity', 'yield_maturity', 'esg_liquidity',
            'spread_squared', 'liquidity_squared', 'esg_squared',
            'market_stress', 'credit_quality'
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
        Train the downgrade risk model
        
        Args:
            df: DataFrame with bond data
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting downgrade risk model training with {self.model_type.value}")
        
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
        y_train_proba = self.model.predict_proba(X_train_processed)[:, 1]
        y_test_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba, "train")
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba, "test")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_processed, y_train, 
            cv=cv_folds, scoring='roc_auc'
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
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().isoformat()
        }
        
        self.training_history.append(training_result)
        
        self.logger.info(f"Training completed. Test AUC: {test_metrics['roc_auc']:.3f}")
        
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
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
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
            importance = np.abs(self.model.coef_[0])
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
    
    def predict_downgrade_risk(self, df: pd.DataFrame) -> List[DowngradePrediction]:
        """
        Predict downgrade risk for bonds
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            List of downgrade risk predictions
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
        y_proba = self.model.predict_proba(X_processed)[:, 1]
        
        # Create prediction results
        predictions = []
        for i, (idx, row) in enumerate(df_features.iterrows()):
            # Get original data row
            orig_row = df.loc[idx]
            
            # Determine risk level
            prob = y_proba[i]
            if prob < 0.25:
                risk_level = RiskLevel.LOW
            elif prob < 0.5:
                risk_level = RiskLevel.MEDIUM
            elif prob < 0.75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Get top risk factors
            top_factors = self._get_top_risk_factors(X.iloc[i])
            
            # Calculate confidence based on feature importance consistency
            confidence = self._calculate_prediction_confidence(X.iloc[i])
            
            prediction = DowngradePrediction(
                issuer_name=orig_row['issuer_name'],
                current_rating=orig_row['credit_rating'],
                predicted_downgrade_prob=float(prob),
                risk_level=risk_level,
                confidence=confidence,
                top_risk_factors=top_factors,
                model_metadata={
                    'model_type': self.model_type.value,
                    'training_date': self.training_history[-1]['training_date'] if self.training_history else None
                }
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _get_top_risk_factors(self, features: pd.Series, top_n: int = 5) -> List[str]:
        """Get top risk factors for a prediction"""
        # Get feature importance for available features
        available_features = [col for col in self.feature_columns if col in features.index]
        
        if not available_features:
            return []
        
        # Calculate risk contribution for each feature
        risk_contributions = {}
        for feature in available_features:
            if feature in self.feature_importance:
                # Normalize feature value and multiply by importance
                feature_value = features[feature]
                importance = self.feature_importance[feature]
                risk_contributions[feature] = abs(feature_value) * importance
        
        # Sort by risk contribution
        sorted_factors = sorted(
            risk_contributions.items(), 
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
        y_proba = self.model.predict_proba(X_processed)[:, 1]
        
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
