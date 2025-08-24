"""
Enhanced Retraining Pipeline Module

This module provides comprehensive model retraining capabilities including:
- Automated retraining pipelines with seeded randomness
- Integration with drift detection
- Model evaluation and comparison
- Pipeline orchestration and monitoring
- Integration with experiment tracking
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import random
import hashlib
import pickle
from dataclasses import dataclass, asdict
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import yaml
import semver

from .config import MLOpsConfig
from .tracking import ExperimentTracker
from .registry import ModelRegistry, ModelStage, ModelMetadata
from .drift import DriftMonitor

logger = logging.getLogger(__name__)

@dataclass
class RetrainConfig:
    """Configuration for retraining pipeline"""
    model_type: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    evaluation_metrics: List[str] = None
    early_stopping_patience: int = 10
    max_training_time_minutes: int = 60
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "f1_score", "precision", "recall"]

@dataclass
class RetrainResult:
    """Result of retraining pipeline"""
    success: bool
    model_path: str
    model_version: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time_seconds: float
    model_size_bytes: int
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

class RetrainPipeline:
    """Enhanced retraining pipeline with automated orchestration"""
    
    def __init__(self, config: MLOpsConfig, tracker: ExperimentTracker, 
                 registry: ModelRegistry, drift_monitor: DriftMonitor):
        """Initialize retraining pipeline"""
        self.config = config
        self.tracker = tracker
        self.registry = registry
        self.drift_monitor = drift_monitor
        
        # Setup storage paths
        self.retrain_data_path = Path(config.data_storage_path) / "retrain"
        self.retrain_data_path.mkdir(parents=True, exist_ok=True)
        
        # Load retraining configurations
        self.retrain_configs = self._load_retrain_configs()
        
        logger.info("Retraining pipeline initialized")
    
    def _load_retrain_configs(self) -> Dict[str, RetrainConfig]:
        """Load retraining configurations from config files"""
        configs = {}
        config_path = Path("mlops/configs")
        
        if config_path.exists():
            for config_file in config_path.glob("retrain_*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                        model_type = config_file.stem.replace("retrain_", "")
                        configs[model_type] = RetrainConfig(**config_data)
                except Exception as e:
                    logger.warning(f"Failed to load retrain config {config_file}: {e}")
        
        return configs
    
    def trigger_retrain(self, model_type: str, reason: str = "drift_detected",
                       data_path: Optional[str] = None) -> str:
        """Trigger retraining for a specific model type"""
        
        try:
            # Start experiment tracking
            run_id = self.tracker.start_experiment(
                experiment_name=f"retrain_{model_type}",
                run_name=f"retrain_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"reason": reason, "model_type": model_type}
            )
            
            # Log retraining trigger
            self.tracker.log_parameters({
                "model_type": model_type,
                "reason": reason,
                "data_path": data_path or "default"
            })
            
            # Execute retraining pipeline
            result = self._execute_retrain_pipeline(model_type, data_path, run_id)
            
            if result.success:
                # Register new model
                model_version = self._register_retrained_model(result, run_id)
                
                # Log success
                self.tracker.log_metrics({
                    "retrain_success": 1,
                    "new_model_version": float(model_version.replace(".", "")),
                    "training_time_seconds": result.training_time_seconds
                })
                
                logger.info(f"Retraining completed successfully for {model_type}. New version: {model_version}")
                
            else:
                # Log failure
                self.tracker.log_metrics({
                    "retrain_success": 0,
                    "error": 1
                })
                
                logger.error(f"Retraining failed for {model_type}: {result.error_message}")
            
            # End experiment
            self.tracker.end_experiment()
            
            return run_id
            
        except Exception as e:
            logger.error(f"Error triggering retrain for {model_type}: {e}")
            if self.tracker.current_run:
                self.tracker.end_experiment()
            raise
    
    def _execute_retrain_pipeline(self, model_type: str, data_path: Optional[str], 
                                 run_id: str) -> RetrainResult:
        """Execute the retraining pipeline"""
        
        start_time = datetime.now()
        
        try:
            # Load retraining configuration
            if model_type not in self.retrain_configs:
                raise ValueError(f"No retraining configuration found for {model_type}")
            
            retrain_config = self.retrain_configs[model_type]
            
            # Load training data
            training_data = self._load_training_data(model_type, data_path)
            
            # Set random seeds for reproducibility
            self._set_random_seeds(run_id)
            
            # Prepare features and target
            X, y = self._prepare_features_target(training_data, retrain_config)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=retrain_config.test_size, 
                random_state=retrain_config.random_state, stratify=y if self._is_classification(y) else None
            )
            
            # Initialize model
            model = self._initialize_model(retrain_config)
            
            # Train model
            training_metrics = self._train_model(model, X_train, y_train, retrain_config)
            
            # Evaluate model
            validation_metrics = self._evaluate_model(model, X_train, y_train, retrain_config)
            test_metrics = self._evaluate_model(model, X_test, y_test, retrain_config)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(model, retrain_config.feature_columns)
            
            # Save model
            model_path = self._save_model(model, model_type, run_id)
            
            # Calculate model size
            model_size_bytes = os.path.getsize(model_path)
            
            # Calculate training time
            training_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = RetrainResult(
                success=True,
                model_path=model_path,
                model_version="",  # Will be set by registry
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                training_time_seconds=training_time_seconds,
                model_size_bytes=model_size_bytes,
                hyperparameters=retrain_config.hyperparameters,
                feature_importance=feature_importance
            )
            
            # Log metrics during training
            self.tracker.log_metrics(training_metrics, step=0)
            self.tracker.log_metrics(validation_metrics, step=1)
            self.tracker.log_metrics(test_metrics, step=2)
            
            return result
            
        except Exception as e:
            training_time_seconds = (datetime.now() - start_time).total_seconds()
            
            return RetrainResult(
                success=False,
                model_path="",
                model_version="",
                training_metrics={},
                validation_metrics={},
                test_metrics={},
                training_time_seconds=training_time_seconds,
                model_size_bytes=0,
                hyperparameters={},
                error_message=str(e)
            )
    
    def _set_random_seeds(self, run_id: str):
        """Set random seeds for reproducibility"""
        # Generate deterministic seed from run_id
        seed = int(hashlib.md5(run_id.encode()).hexdigest()[:8], 16)
        
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"Set random seeds to {seed} for run {run_id}")
    
    def _load_training_data(self, model_type: str, data_path: Optional[str]) -> pd.DataFrame:
        """Load training data for retraining"""
        
        if data_path and os.path.exists(data_path):
            # Load from specified path
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_path}")
        else:
            # Load from default location
            default_path = self.retrain_data_path / f"{model_type}_training_data.parquet"
            if default_path.exists():
                data = pd.read_parquet(default_path)
            else:
                raise FileNotFoundError(f"No training data found for {model_type}")
        
        logger.info(f"Loaded training data: {len(data)} samples, {len(data.columns)} features")
        return data
    
    def _prepare_features_target(self, data: pd.DataFrame, config: RetrainConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        
        # Select features
        X = data[config.feature_columns].copy()
        y = data[config.target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.median())
        
        # Convert categorical features to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
        return X, y
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Determine if target is classification or regression"""
        unique_values = y.nunique()
        return unique_values <= 20  # Simple heuristic
    
    def _initialize_model(self, config: RetrainConfig):
        """Initialize model based on configuration"""
        
        algorithm = config.algorithm.lower()
        
        if algorithm == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if self._is_classification(y):
                model = RandomForestClassifier(**config.hyperparameters, random_state=42)
            else:
                model = RandomForestRegressor(**config.hyperparameters, random_state=42)
        
        elif algorithm == "xgboost":
            import xgboost as xgb
            if self._is_classification(y):
                model = xgb.XGBClassifier(**config.hyperparameters, random_state=42)
            else:
                model = xgb.XGBRegressor(**config.hyperparameters, random_state=42)
        
        elif algorithm == "lightgbm":
            import lightgbm as lgb
            if self._is_classification(y):
                model = lgb.LGBMClassifier(**config.hyperparameters, random_state=42)
            else:
                model = lgb.LGBMRegressor(**config.hyperparameters, random_state=42)
        
        elif algorithm == "neural_network":
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            if self._is_classification(y):
                model = MLPClassifier(**config.hyperparameters, random_state=42)
            else:
                model = MLPRegressor(**config.hyperparameters, random_state=42)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        logger.info(f"Initialized {algorithm} model with hyperparameters: {config.hyperparameters}")
        return model
    
    def _train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                    config: RetrainConfig) -> Dict[str, float]:
        """Train the model and return training metrics"""
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_pred = model.predict(X_train)
        
        if self._is_classification(y_train):
            metrics = {
                "train_accuracy": accuracy_score(y_train, y_pred),
                "train_precision": precision_score(y_train, y_pred, average='weighted'),
                "train_recall": recall_score(y_train, y_pred, average='weighted'),
                "train_f1_score": f1_score(y_train, y_pred, average='weighted')
            }
        else:
            metrics = {
                "train_mae": mean_absolute_error(y_train, y_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred)),
                "train_r2": r2_score(y_train, y_pred)
            }
        
        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics
    
    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                       config: RetrainConfig) -> Dict[str, float]:
        """Evaluate model performance"""
        
        y_pred = model.predict(X)
        
        if self._is_classification(y):
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average='weighted'),
                "recall": recall_score(y, y_pred, average='weighted'),
                "f1_score": f1_score(y, y_pred, average='weighted')
            }
            
            # Add ROC AUC if binary classification
            if y.nunique() == 2:
                try:
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y, y_pred_proba)
                except:
                    pass
        else:
            metrics = {
                "mae": mean_absolute_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "r2": r2_score(y, y_pred)
            }
        
        return metrics
    
    def _get_feature_importance(self, model, feature_columns: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model"""
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                return None
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_columns, importance))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def _save_model(self, model, model_type: str, run_id: str) -> str:
        """Save the trained model"""
        
        model_path = self.retrain_data_path / f"{model_type}_{run_id}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def _register_retrained_model(self, result: RetrainResult, run_id: str) -> str:
        """Register the retrained model in the registry"""
        
        # Get current model info
        current_model = self.registry.get_model(result.model_path.split('_')[0])
        
        # Generate new version
        if current_model:
            # Increment minor version for retraining
            new_version = self._increment_version(current_model.version, version_type="minor")
        else:
            new_version = "1.0.0"
        
        # Create model metadata
        metadata = ModelMetadata(
            model_type=result.model_path.split('_')[0],
            algorithm="retrained",  # This should come from config
            hyperparameters=result.hyperparameters,
            feature_columns=[],  # This should come from config
            target_column="",    # This should come from config
            training_data_info={"samples": 0},  # This should be calculated
            validation_metrics=result.validation_metrics,
            model_size_bytes=result.model_size_bytes,
            dependencies={},
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Register model
        model_version = self.registry.register_model(
            model_type=result.model_path.split('_')[0],
            model_path=result.model_path,
            metadata=metadata,
            experiment_run_id=run_id,
            git_sha="retrained",  # This should come from git
            performance_metrics=result.test_metrics,
            description="Automatically retrained model"
        )
        
        return model_version
    
    def _increment_version(self, current_version: str, version_type: str = "patch") -> str:
        """Increment semantic version"""
        
        try:
            version_info = semver.VersionInfo.parse(current_version)
            
            if version_type == "major":
                new_version = semver.VersionInfo(version_info.major + 1, 0, 0)
            elif version_type == "minor":
                new_version = semver.VersionInfo(version_info.major, version_info.minor + 1, 0)
            else:  # patch
                new_version = semver.VersionInfo(version_info.major, version_info.minor, version_info.patch + 1)
            
            return str(new_version)
            
        except ValueError:
            # If version parsing fails, return a default
            return "1.0.0"
    
    def compare_models(self, model_type: str, old_version: str, new_version: str) -> Dict[str, Any]:
        """Compare old and new model versions"""
        
        old_model = self.registry.get_model(model_type, old_version)
        new_model = self.registry.get_model(model_type, new_version)
        
        if not old_model or not new_model:
            raise ValueError("One or both models not found")
        
        comparison = {
            "old_version": old_version,
            "new_version": new_version,
            "performance_improvement": {},
            "size_change": new_model.model_size_bytes - old_model.model_size_bytes,
            "training_date_diff": (new_model.created_at - old_model.created_at).days
        }
        
        # Compare performance metrics
        for metric in old_model.performance_metrics:
            if metric in new_model.performance_metrics:
                old_value = old_model.performance_metrics[metric]
                new_value = new_model.performance_metrics[metric]
                
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    improvement = new_value - old_value
                    comparison["performance_improvement"][metric] = improvement
        
        return comparison
    
    def get_retrain_history(self, model_type: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get retraining history for a model type"""
        
        # This would typically query the experiment tracker
        # For now, return basic information
        return [
            {
                "model_type": model_type,
                "last_retrain": "2024-01-01",
                "retrain_count": 5,
                "success_rate": 0.8
            }
        ]
