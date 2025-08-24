"""
MLOps Configuration Module

This module provides configuration management for MLOps operations including:
- Drift detection thresholds
- Canary deployment policies
- Model promotion gates
- Experiment tracking settings
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import yaml


class DriftThresholds(BaseModel):
    """Drift detection thresholds for different model types"""
    
    # Statistical drift thresholds
    ks_test_threshold: float = Field(0.05, description="Kolmogorov-Smirnov test p-value threshold")
    psi_threshold: float = Field(0.25, description="Population Stability Index threshold")
    feature_drift_threshold: float = Field(0.1, description="Feature distribution drift threshold")
    
    # Target drift thresholds
    target_drift_threshold: float = Field(0.15, description="Target variable drift threshold")
    residual_drift_threshold: float = Field(0.2, description="Model residual drift threshold")
    
    # Time-based drift detection
    drift_check_interval_hours: int = Field(24, description="Hours between drift checks")
    drift_window_days: int = Field(30, description="Days of data to use for drift detection")


class CanaryPolicy(BaseModel):
    """Canary deployment policy configuration"""
    
    # Traffic splitting
    initial_canary_percentage: float = Field(0.05, description="Initial traffic to canary model (5%)")
    max_canary_percentage: float = Field(0.20, description="Maximum traffic to canary model (20%)")
    promotion_threshold: float = Field(0.95, description="Performance threshold for promotion")
    
    # Evaluation criteria
    evaluation_metrics: List[str] = Field(
        ["mae", "rmse", "r2_score"], 
        description="Metrics to evaluate for promotion"
    )
    evaluation_window_hours: int = Field(48, description="Hours to evaluate canary performance")
    
    # Rollback conditions
    rollback_threshold: float = Field(0.8, description="Performance threshold below which to rollback")
    max_rollback_count: int = Field(3, description="Maximum rollback attempts")


class ModelPromotionGates(BaseModel):
    """Model promotion validation gates"""
    
    # Performance gates
    min_accuracy: float = Field(0.85, description="Minimum accuracy for production")
    max_drift: float = Field(0.1, description="Maximum allowed drift for production")
    
    # Quality gates
    require_explainability: bool = Field(True, description="Require SHAP explanations")
    require_feature_importance: bool = Field(True, description="Require feature importance analysis")
    require_bias_testing: bool = Field(True, description="Require bias/fairness testing")
    
    # Security gates
    require_security_scan: bool = Field(True, description="Require security vulnerability scan")
    require_compliance_check: bool = Field(True, description="Require regulatory compliance check")


class ExperimentTracking(BaseModel):
    """Experiment tracking configuration"""
    
    # Backend settings
    tracking_backend: str = Field("local", description="Tracking backend (local/mlflow/weights_biases)")
    tracking_uri: Optional[str] = Field(None, description="Tracking server URI")
    
    # Logging settings
    log_parameters: bool = Field(True, description="Log model parameters")
    log_metrics: bool = Field(True, description="Log training metrics")
    log_artifacts: bool = Field(True, description="Log model artifacts")
    log_environment: bool = Field(True, description="Log environment information")
    log_git_sha: bool = Field(True, description="Log git commit SHA")
    
    # Artifact storage
    artifact_storage_path: str = Field("./mlruns", description="Local artifact storage path")
    max_artifact_size_mb: int = Field(100, description="Maximum artifact size in MB")


class MLOpsConfig(BaseModel):
    """Main MLOps configuration"""
    
    # Core settings
    environment: str = Field("development", description="Environment (development/staging/production)")
    seed: int = Field(42, description="Random seed for reproducibility")
    
    # Model types
    supported_models: List[str] = Field(
        ["spread", "downgrade", "liquidity_shock", "anomaly"],
        description="Supported model types"
    )
    
    # Configuration sections
    drift: DriftThresholds = Field(default_factory=DriftThresholds)
    canary: CanaryPolicy = Field(default_factory=CanaryPolicy)
    promotion: ModelPromotionGates = Field(default_factory=ModelPromotionGates)
    tracking: ExperimentTracking = Field(default_factory=ExperimentTracking)
    
    # Paths
    model_storage_path: str = Field("./models", description="Path to store trained models")
    data_storage_path: str = Field("./data", description="Path to store training data")
    log_path: str = Field("./logs", description="Path to store logs")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "MLOpsConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_data = self.model_dump()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    @classmethod
    def get_default_config(cls) -> "MLOpsConfig":
        """Get default configuration"""
        return cls()
    
    def validate_environment(self) -> bool:
        """Validate configuration for current environment"""
        if self.environment == "production":
            # Production-specific validations
            if not self.tracking.tracking_backend in ["mlflow", "weights_biases"]:
                raise ValueError("Production environment requires remote tracking backend")
            
            if not self.promotion.require_security_scan:
                raise ValueError("Production environment requires security scanning")
        
        return True


# Default configuration
DEFAULT_CONFIG = MLOpsConfig.get_default_config()
