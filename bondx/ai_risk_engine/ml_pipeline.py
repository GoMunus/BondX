"""
Machine Learning Pipeline Architecture

This module implements comprehensive ML pipelines that handle feature engineering,
model training, validation, and deployment in production environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import joblib
from pathlib import Path
import hashlib

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
    cross_val_score, validation_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)

# Advanced ML imports
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.sklearn

# Feature engineering
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stages"""
    DATA_INGESTION = "data_ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"

class FeatureType(Enum):
    """Types of features"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    DERIVED = "derived"

class ValidationMethod(Enum):
    """Validation methods"""
    TIME_SERIES_CV = "time_series_cv"
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    feature_name: str
    feature_type: FeatureType
    source_columns: List[str]
    transformation: str
    parameters: Dict = field(default_factory=dict)
    validation_rules: Dict = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_name: str
    model_type: str
    hyperparameters: Dict
    feature_columns: List[str]
    target_column: str
    validation_method: ValidationMethod
    validation_params: Dict = field(default_factory=dict)

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    training_time: float
    validation_score: float
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]
    data_quality_metrics: Dict[str, float]
    timestamp: datetime

class MLPipeline:
    """
    Comprehensive ML pipeline for bond risk analytics
    """
    
    def __init__(self, config: Dict = None, pipeline_name: str = "bond_risk_pipeline"):
        self.config = config or {}
        self.pipeline_name = pipeline_name
        self.pipeline = None
        self.feature_configs = []
        self.model_configs = []
        self.feature_store = {}
        self.model_store = {}
        self.metrics_history = []
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Initialize pipeline components
        self._initialize_pipeline_components()
        
    def _setup_mlflow(self):
        """Setup MLflow for experiment tracking"""
        try:
            mlflow.set_tracking_uri(self.config.get('mlflow_tracking_uri', 'sqlite:///mlflow.db'))
            mlflow.set_experiment(self.pipeline_name)
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    
    def _initialize_pipeline_components(self):
        """Initialize pipeline components"""
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        self.feature_selectors = {
            'kbest': SelectKBest(score_func=f_regression),
            'mutual_info': SelectKBest(score_func=mutual_info_regression),
            'rfe': RFE(estimator=RandomForestRegressor(n_estimators=10)),
            'select_from_model': SelectFromModel(RandomForestRegressor(n_estimators=10))
        }
        
        self.dimension_reducers = {
            'pca': PCA(),
            'ica': FastICA()
        }
    
    def add_feature_config(self, feature_config: FeatureConfig):
        """Add feature configuration to pipeline"""
        self.feature_configs.append(feature_config)
        logger.info(f"Added feature config: {feature_config.feature_name}")
    
    def add_model_config(self, model_config: ModelConfig):
        """Add model configuration to pipeline"""
        self.model_configs.append(model_config)
        logger.info(f"Added model config: {model_config.model_name}")
    
    def engineer_features(
        self,
        raw_data: pd.DataFrame,
        feature_configs: List[FeatureConfig] = None
    ) -> pd.DataFrame:
        """
        Engineer features based on configuration
        
        Args:
            raw_data: Raw input data
            feature_configs: List of feature configurations (uses pipeline configs if None)
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if feature_configs is None:
                feature_configs = self.feature_configs
            
            if not feature_configs:
                logger.warning("No feature configurations provided")
                return raw_data
            
            engineered_data = raw_data.copy()
            
            for config in feature_configs:
                try:
                    feature_value = self._create_feature(
                        raw_data, config
                    )
                    
                    if feature_value is not None:
                        engineered_data[config.feature_name] = feature_value
                        
                        # Store feature metadata
                        self.feature_store[config.feature_name] = {
                            'config': config,
                            'created_at': datetime.now(),
                            'data_type': str(type(feature_value)),
                            'validation_status': 'created'
                        }
                        
                except Exception as e:
                    logger.error(f"Error creating feature {config.feature_name}: {e}")
                    continue
            
            # Validate engineered features
            self._validate_features(engineered_data)
            
            logger.info(f"Engineered {len(engineered_data.columns) - len(raw_data.columns)} new features")
            return engineered_data
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return raw_data
    
    def _create_feature(
        self,
        data: pd.DataFrame,
        config: FeatureConfig
    ) -> Union[pd.Series, float, int, str]:
        """Create individual feature based on configuration"""
        try:
            if config.feature_type == FeatureType.NUMERICAL:
                return self._create_numerical_feature(data, config)
            elif config.feature_type == FeatureType.CATEGORICAL:
                return self._create_categorical_feature(data, config)
            elif config.feature_type == FeatureType.TEMPORAL:
                return self._create_temporal_feature(data, config)
            elif config.feature_type == FeatureType.TEXT:
                return self._create_text_feature(data, config)
            elif config.feature_type == FeatureType.DERIVED:
                return self._create_derived_feature(data, config)
            else:
                raise ValueError(f"Unsupported feature type: {config.feature_type}")
                
        except Exception as e:
            logger.error(f"Error creating feature {config.feature_name}: {e}")
            return None
    
    def _create_numerical_feature(self, data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
        """Create numerical feature"""
        transformation = config.transformation.lower()
        
        if transformation == "sum":
            return data[config.source_columns].sum(axis=1)
        elif transformation == "mean":
            return data[config.source_columns].mean(axis=1)
        elif transformation == "std":
            return data[config.source_columns].std(axis=1)
        elif transformation == "min":
            return data[config.source_columns].min(axis=1)
        elif transformation == "max":
            return data[config.source_columns].max(axis=1)
        elif transformation == "median":
            return data[config.source_columns].median(axis=1)
        elif transformation == "rolling_mean":
            window = config.parameters.get('window', 5)
            return data[config.source_columns[0]].rolling(window).mean()
        elif transformation == "rolling_std":
            window = config.parameters.get('window', 5)
            return data[config.source_columns[0]].rolling(window).std()
        elif transformation == "pct_change":
            periods = config.parameters.get('periods', 1)
            return data[config.source_columns[0]].pct_change(periods)
        elif transformation == "diff":
            periods = config.parameters.get('periods', 1)
            return data[config.source_columns[0]].diff(periods)
        elif transformation == "zscore":
            return (data[config.source_columns[0]] - data[config.source_columns[0]].mean()) / data[config.source_columns[0]].std()
        else:
            raise ValueError(f"Unsupported numerical transformation: {transformation}")
    
    def _create_categorical_feature(self, data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
        """Create categorical feature"""
        transformation = config.transformation.lower()
        
        if transformation == "bin":
            column = config.source_columns[0]
            bins = config.parameters.get('bins', 5)
            labels = config.parameters.get('labels', None)
            return pd.cut(data[column], bins=bins, labels=labels)
        elif transformation == "qcut":
            column = config.source_columns[0]
            q = config.parameters.get('q', 5)
            labels = config.parameters.get('labels', None)
            return pd.qcut(data[column], q=q, labels=labels)
        elif transformation == "combine":
            separator = config.parameters.get('separator', '_')
            return data[config.source_columns].astype(str).agg(separator.join, axis=1)
        else:
            raise ValueError(f"Unsupported categorical transformation: {transformation}")
    
    def _create_temporal_feature(self, data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
        """Create temporal feature"""
        transformation = config.transformation.lower()
        column = config.source_columns[0]
        
        if transformation == "hour":
            return pd.to_datetime(data[column]).dt.hour
        elif transformation == "day":
            return pd.to_datetime(data[column]).dt.day
        elif transformation == "month":
            return pd.to_datetime(data[column]).dt.month
        elif transformation == "quarter":
            return pd.to_datetime(data[column]).dt.quarter
        elif transformation == "year":
            return pd.to_datetime(data[column]).dt.year
        elif transformation == "dayofweek":
            return pd.to_datetime(data[column]).dt.dayofweek
        elif transformation == "is_month_end":
            return pd.to_datetime(data[column]).dt.is_month_end.astype(int)
        elif transformation == "is_quarter_end":
            return pd.to_datetime(data[column]).dt.is_quarter_end.astype(int)
        else:
            raise ValueError(f"Unsupported temporal transformation: {transformation}")
    
    def _create_text_feature(self, data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
        """Create text feature"""
        transformation = config.transformation.lower()
        column = config.source_columns[0]
        
        if transformation == "length":
            return data[column].astype(str).str.len()
        elif transformation == "word_count":
            return data[column].astype(str).str.split().str.len()
        elif transformation == "has_numeric":
            return data[column].astype(str).str.contains(r'\d').astype(int)
        elif transformation == "has_special":
            return data[column].astype(str).str.contains(r'[^a-zA-Z0-9\s]').astype(int)
        else:
            raise ValueError(f"Unsupported text transformation: {transformation}")
    
    def _create_derived_feature(self, data: pd.DataFrame, config: FeatureConfig) -> pd.Series:
        """Create derived feature using custom logic"""
        transformation = config.transformation.lower()
        
        if transformation == "ratio":
            if len(config.source_columns) != 2:
                raise ValueError("Ratio transformation requires exactly 2 source columns")
            return data[config.source_columns[0]] / data[config.source_columns[1]]
        elif transformation == "product":
            return data[config.source_columns].prod(axis=1)
        elif transformation == "polynomial":
            degree = config.parameters.get('degree', 2)
            column = config.source_columns[0]
            return data[column] ** degree
        elif transformation == "interaction":
            if len(config.source_columns) != 2:
                raise ValueError("Interaction transformation requires exactly 2 source columns")
            return data[config.source_columns[0]] * data[config.source_columns[1]]
        else:
            raise ValueError(f"Unsupported derived transformation: {transformation}")
    
    def select_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        method: str = "kbest",
        n_features: int = 20,
        threshold: float = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most important features
        
        Args:
            data: Feature matrix
            target: Target variable
            method: Feature selection method
            n_features: Number of features to select
            threshold: Feature importance threshold
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        try:
            if method not in self.feature_selectors:
                raise ValueError(f"Unsupported feature selection method: {method}")
            
            selector = self.feature_selectors[method]
            
            if method == "kbest":
                selector.set_params(k=n_features)
            elif method == "select_from_model":
                if threshold:
                    selector.set_params(threshold=threshold)
            
            # Fit and transform
            selected_features = selector.fit_transform(data, target)
            
            # Get selected feature names
            if hasattr(selector, 'get_support'):
                selected_mask = selector.get_support()
                selected_feature_names = data.columns[selected_mask].tolist()
            else:
                selected_feature_names = data.columns.tolist()
            
            # Create DataFrame with selected features
            selected_data = pd.DataFrame(
                selected_features,
                columns=selected_feature_names,
                index=data.index
            )
            
            logger.info(f"Selected {len(selected_feature_names)} features using {method}")
            return selected_data, selected_feature_names
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return data, data.columns.tolist()
    
    def reduce_dimensions(
        self,
        data: pd.DataFrame,
        method: str = "pca",
        n_components: int = 10
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Reduce dimensionality of features
        
        Args:
            data: Feature matrix
            method: Dimensionality reduction method
            n_components: Number of components to keep
            
        Returns:
            Tuple of (reduced_features, reducer_object)
        """
        try:
            if method not in self.dimension_reducers:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")
            
            reducer = self.dimension_reducers[method]
            reducer.set_params(n_components=min(n_components, data.shape[1]))
            
            # Fit and transform
            reduced_features = reducer.fit_transform(data)
            
            # Create column names
            column_names = [f"{method.upper()}_{i+1}" for i in range(reduced_features.shape[1])]
            
            # Create DataFrame
            reduced_data = pd.DataFrame(
                reduced_features,
                columns=column_names,
                index=data.index
            )
            
            logger.info(f"Reduced dimensions from {data.shape[1]} to {reduced_features.shape[1]} using {method}")
            return reduced_data, reducer
            
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            return data, None
    
    def optimize_hyperparameters(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        method: str = "grid_search",
        cv_folds: int = 5,
        n_trials: int = 100
    ) -> Tuple[Any, Dict]:
        """
        Optimize hyperparameters using various methods
        
        Args:
            model: Base model to optimize
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for optimization
            method: Optimization method
            cv_folds: Number of CV folds
            n_trials: Number of trials for random search
            
        Returns:
            Tuple of (best_model, best_params)
        """
        try:
            if method == "grid_search":
                # Time series aware cross-validation
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=1
                )
                
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
            elif method == "random_search":
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                
                random_search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_trials, cv=tscv,
                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
                )
                
                random_search.fit(X, y)
                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
                
            elif method == "optuna":
                best_model, best_params = self._optimize_with_optuna(
                    model, X, y, param_grid, n_trials, cv_folds
                )
                
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            logger.info(f"Best parameters: {best_params}")
            return best_model, best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return model, {}
    
    def _optimize_with_optuna(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        n_trials: int,
        cv_folds: int
    ) -> Tuple[Any, Dict]:
        """Optimize hyperparameters using Optuna"""
        try:
            def objective(trial):
                # Suggest parameters
                params = {}
                for param_name, param_range in param_grid.items():
                    if isinstance(param_range, list):
                        if isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        elif isinstance(param_range[0], float):
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_categorical(param_name, param_range)
                    else:
                        params[param_name] = param_range
                
                # Set parameters
                model.set_params(**params)
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                
                return scores.mean()
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Optimize
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_model = model.set_params(**best_params)
            
            return best_model, best_params
            
        except Exception as e:
            logger.error(f"Error in Optuna optimization: {e}")
            return model, {}
    
    def validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        validation_method: ValidationMethod = ValidationMethod.TIME_SERIES_CV,
        validation_params: Dict = None
    ) -> Dict[str, float]:
        """
        Validate model using specified validation method
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            validation_method: Validation method to use
            validation_params: Parameters for validation
            
        Returns:
            Dictionary of validation metrics
        """
        try:
            validation_params = validation_params or {}
            
            if validation_method == ValidationMethod.TIME_SERIES_CV:
                return self._time_series_cv_validation(model, X, y, validation_params)
            elif validation_method == ValidationMethod.WALK_FORWARD:
                return self._walk_forward_validation(model, X, y, validation_params)
            elif validation_method == ValidationMethod.EXPANDING_WINDOW:
                return self._expanding_window_validation(model, X, y, validation_params)
            elif validation_method == ValidationMethod.ROLLING_WINDOW:
                return self._rolling_window_validation(model, X, y, validation_params)
            else:
                raise ValueError(f"Unsupported validation method: {validation_method}")
                
        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            return {}
    
    def _time_series_cv_validation(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict
    ) -> Dict[str, float]:
        """Time series cross-validation"""
        n_splits = params.get('n_splits', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        
        return {
            'cv_mean_rmse': np.sqrt(-scores.mean()),
            'cv_std_rmse': np.sqrt(-scores.std()),
            'cv_scores': scores.tolist()
        }
    
    def _walk_forward_validation(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict
    ) -> Dict[str, float]:
        """Walk-forward validation"""
        train_size = params.get('train_size', 0.7)
        step_size = params.get('step_size', 1)
        
        train_idx = int(len(X) * train_size)
        predictions = []
        actuals = []
        
        while train_idx < len(X):
            # Train on expanding window
            X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:train_idx+step_size]
            y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:train_idx+step_size]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            
            train_idx += step_size
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'walk_forward_rmse': rmse,
            'walk_forward_mae': mae,
            'walk_forward_r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def _expanding_window_validation(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict
    ) -> Dict[str, float]:
        """Expanding window validation"""
        min_train_size = params.get('min_train_size', 100)
        step_size = params.get('step_size', 20)
        
        predictions = []
        actuals = []
        
        for i in range(min_train_size, len(X), step_size):
            X_train, X_test = X.iloc[:i], X.iloc[i:i+step_size]
            y_train, y_test = y.iloc[:i], y.iloc[i:i+step_size]
            
            if len(X_test) == 0:
                break
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'expanding_window_rmse': rmse,
            'expanding_window_mae': mae,
            'expanding_window_r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def _rolling_window_validation(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict
    ) -> Dict[str, float]:
        """Rolling window validation"""
        window_size = params.get('window_size', 200)
        step_size = params.get('step_size', 20)
        
        predictions = []
        actuals = []
        
        for i in range(window_size, len(X), step_size):
            X_train, X_test = X.iloc[i-window_size:i], X.iloc[i:i+step_size]
            y_train, y_test = y.iloc[i-window_size:i], y.iloc[i:i+step_size]
            
            if len(X_test) == 0:
                break
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'rolling_window_rmse': rmse,
            'rolling_window_mae': mae,
            'rolling_window_r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def save_model(self, model, model_name: str, save_path: str = None):
        """Save trained model"""
        try:
            if save_path is None:
                save_path = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save to MLflow if available
            try:
                mlflow.sklearn.log_model(model, model_name)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        return {
            'pipeline_name': self.pipeline_name,
            'feature_configs': len(self.feature_configs),
            'model_configs': len(self.model_configs),
            'features_created': len(self.feature_store),
            'models_trained': len(self.model_store),
            'metrics_history': len(self.metrics_history),
            'last_updated': datetime.now().isoformat()
        }
