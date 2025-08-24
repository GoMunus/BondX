"""
Enhanced Canary Deployment Module

This module provides comprehensive canary deployment capabilities including:
- Traffic splitting and routing
- Performance monitoring and comparison
- Automatic promotion and rollback
- Policy-driven decision making
- Integration with model registry and tracking
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import time
import asyncio
from dataclasses import dataclass, asdict
import yaml
from enum import Enum

from .config import MLOpsConfig
from .tracking import ExperimentTracker
from .registry import ModelRegistry, ModelStage, ModelVersion

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class CanaryStage(Enum):
    """Canary deployment stages"""
    INITIAL = "initial"
    EXPANDING = "expanding"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"

@dataclass
class CanaryConfig:
    """Configuration for canary deployment"""
    initial_percentage: float = 0.05  # 5%
    max_percentage: float = 0.20      # 20%
    evaluation_window_hours: int = 48
    promotion_threshold: float = 0.95  # 95% confidence
    rollback_threshold: float = 0.8    # 80% confidence
    max_rollback_count: int = 3
    metrics_to_monitor: List[str] = None
    health_check_interval_seconds: int = 300  # 5 minutes
    
    def __post_init__(self):
        if self.metrics_to_monitor is None:
            self.metrics_to_monitor = ["accuracy", "latency", "error_rate", "throughput"]

@dataclass
class CanaryDeployment:
    """Canary deployment instance"""
    deployment_id: str
    model_type: str
    candidate_version: str
    production_version: str
    status: DeploymentStatus
    stage: CanaryStage
    start_time: datetime
    current_percentage: float
    target_percentage: float
    metrics_history: List[Dict[str, Any]]
    health_checks: List[Dict[str, Any]]
    rollback_count: int = 0
    end_time: Optional[datetime] = None
    final_status: Optional[str] = None

@dataclass
class DeploymentMetrics:
    """Metrics for deployment comparison"""
    timestamp: datetime
    production_metrics: Dict[str, float]
    canary_metrics: Dict[str, float]
    comparison_score: float
    is_healthy: bool
    alerts: List[str]

class CanaryDeploymentManager:
    """Enhanced canary deployment manager"""
    
    def __init__(self, config: MLOpsConfig, registry: ModelRegistry, 
                 tracker: Optional[ExperimentTracker] = None):
        """Initialize canary deployment manager"""
        self.config = config
        self.registry = registry
        self.tracker = tracker
        
        # Load canary configuration
        self.canary_config = self._load_canary_config()
        
        # Active deployments
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        
        # Deployment storage
        self.deployment_path = Path("mlops/deployments")
        self.deployment_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing deployments
        self._load_deployments()
        
        logger.info("Canary deployment manager initialized")
    
    def _load_canary_config(self) -> CanaryConfig:
        """Load canary deployment configuration"""
        try:
            config_path = Path("mlops/configs/canary.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    return CanaryConfig(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load canary config: {e}")
        
        # Return default configuration
        return CanaryConfig()
    
    def _load_deployments(self):
        """Load existing deployments from storage"""
        deployments_file = self.deployment_path / "deployments.json"
        
        if deployments_file.exists():
            try:
                with open(deployments_file, 'r') as f:
                    deployments_data = json.load(f)
                    
                    for deployment_data in deployments_data:
                        # Convert back to CanaryDeployment object
                        deployment = CanaryDeployment(**deployment_data)
                        deployment.start_time = datetime.fromisoformat(deployment_data["start_time"])
                        if deployment_data.get("end_time"):
                            deployment.end_time = datetime.fromisoformat(deployment_data["end_time"])
                        
                        # Only load active deployments
                        if deployment.status in [DeploymentStatus.PENDING, DeploymentStatus.RUNNING]:
                            self.active_deployments[deployment.deployment_id] = deployment
                            
            except Exception as e:
                logger.error(f"Failed to load deployments: {e}")
    
    def _save_deployments(self):
        """Save deployments to storage"""
        deployments_file = self.deployment_path / "deployments.json"
        
        # Convert deployments to serializable format
        deployments_data = []
        for deployment in self.active_deployments.values():
            deployment_dict = asdict(deployment)
            deployment_dict["start_time"] = deployment_dict["start_time"].isoformat()
            if deployment_dict.get("end_time"):
                deployment_dict["end_time"] = deployment_dict["end_time"].isoformat()
            deployments_data.append(deployment_dict)
        
        with open(deployments_file, 'w') as f:
            json.dump(deployments_data, f, indent=2, default=str)
    
    def start_canary_deployment(self, model_type: str, candidate_version: str,
                               initial_percentage: Optional[float] = None) -> str:
        """Start a new canary deployment"""
        
        try:
            # Validate model versions
            candidate_model = self.registry.get_model(model_type, candidate_version)
            if not candidate_model:
                raise ValueError(f"Candidate model {candidate_version} not found")
            
            production_model = self.registry.get_model(model_type, stage=ModelStage.PRODUCTION)
            if not production_model:
                raise ValueError(f"No production model found for {model_type}")
            
            # Check if deployment already exists
            for deployment in self.active_deployments.values():
                if (deployment.model_type == model_type and 
                    deployment.candidate_version == candidate_version):
                    raise ValueError(f"Canary deployment already exists for {model_type} {candidate_version}")
            
            # Generate deployment ID
            deployment_id = f"canary_{model_type}_{candidate_version}_{int(time.time())}"
            
            # Set initial percentage
            if initial_percentage is None:
                initial_percentage = self.canary_config.initial_percentage
            
            # Create deployment
            deployment = CanaryDeployment(
                deployment_id=deployment_id,
                model_type=model_type,
                candidate_version=candidate_version,
                production_version=production_model.version,
                status=DeploymentStatus.PENDING,
                stage=CanaryStage.INITIAL,
                start_time=datetime.now(),
                current_percentage=0.0,
                target_percentage=initial_percentage,
                metrics_history=[],
                health_checks=[]
            )
            
            # Add to active deployments
            self.active_deployments[deployment_id] = deployment
            
            # Save deployments
            self._save_deployments()
            
            # Log deployment start
            if self.tracker:
                self.tracker.log_metrics({
                    "canary_deployment_started": 1,
                    "initial_percentage": initial_percentage
                })
            
            logger.info(f"Started canary deployment {deployment_id} for {model_type} {candidate_version}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to start canary deployment: {e}")
            raise
    
    def update_traffic_split(self, deployment_id: str, new_percentage: float) -> bool:
        """Update traffic split for a canary deployment"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        # Validate percentage
        if new_percentage < 0 or new_percentage > self.canary_config.max_percentage:
            raise ValueError(f"Invalid percentage: {new_percentage}. Must be between 0 and {self.canary_config.max_percentage}")
        
        # Update deployment
        deployment.current_percentage = new_percentage
        deployment.target_percentage = new_percentage
        
        # Update stage
        if new_percentage > 0:
            deployment.stage = CanaryStage.EXPANDING
            deployment.status = DeploymentStatus.RUNNING
        
        # Save deployments
        self._save_deployments()
        
        logger.info(f"Updated traffic split for {deployment_id} to {new_percentage:.1%}")
        return True
    
    def record_metrics(self, deployment_id: str, production_metrics: Dict[str, float],
                      canary_metrics: Dict[str, float]) -> bool:
        """Record metrics for deployment comparison"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        # Calculate comparison score
        comparison_score = self._calculate_comparison_score(production_metrics, canary_metrics)
        
        # Determine health status
        is_healthy = comparison_score >= self.canary_config.promotion_threshold
        
        # Generate alerts
        alerts = self._generate_alerts(production_metrics, canary_metrics, comparison_score)
        
        # Create metrics record
        metrics_record = DeploymentMetrics(
            timestamp=datetime.now(),
            production_metrics=production_metrics,
            canary_metrics=canary_metrics,
            comparison_score=comparison_score,
            is_healthy=is_healthy,
            alerts=alerts
        )
        
        # Add to deployment history
        deployment.metrics_history.append(asdict(metrics_record))
        
        # Update deployment stage
        if len(deployment.metrics_history) >= 3:  # Need some history for evaluation
            deployment.stage = CanaryStage.EVALUATING
        
        # Save deployments
        self._save_deployments()
        
        # Log metrics
        if self.tracker:
            self.tracker.log_metrics({
                "canary_comparison_score": comparison_score,
                "canary_is_healthy": int(is_healthy),
                "canary_alerts_count": len(alerts)
            })
        
        logger.info(f"Recorded metrics for {deployment_id}. Score: {comparison_score:.3f}, Healthy: {is_healthy}")
        return True
    
    def _calculate_comparison_score(self, production_metrics: Dict[str, float],
                                  canary_metrics: Dict[str, float]) -> float:
        """Calculate overall comparison score between production and canary"""
        
        if not production_metrics or not canary_metrics:
            return 0.0
        
        scores = []
        
        for metric in self.canary_config.metrics_to_monitor:
            if metric in production_metrics and metric in canary_metrics:
                prod_value = production_metrics[metric]
                canary_value = canary_metrics[metric]
                
                if prod_value == 0:
                    continue
                
                # Calculate relative performance
                if metric in ["accuracy", "precision", "recall", "f1_score", "r2"]:
                    # Higher is better
                    relative_score = min(canary_value / prod_value, 1.0)
                elif metric in ["mae", "rmse", "latency", "error_rate"]:
                    # Lower is better
                    relative_score = max(prod_value / canary_value, 0.0)
                else:
                    # Default to ratio
                    relative_score = min(canary_value / prod_value, 1.0)
                
                scores.append(relative_score)
        
        if not scores:
            return 0.0
        
        # Return average score
        return np.mean(scores)
    
    def _generate_alerts(self, production_metrics: Dict[str, float],
                        canary_metrics: Dict[str, float], comparison_score: float) -> List[str]:
        """Generate alerts based on metrics comparison"""
        
        alerts = []
        
        # Check for significant degradation
        if comparison_score < self.canary_config.rollback_threshold:
            alerts.append(f"Critical: Canary performance significantly degraded (score: {comparison_score:.3f})")
        
        # Check individual metrics
        for metric in self.canary_config.metrics_to_monitor:
            if metric in production_metrics and metric in canary_metrics:
                prod_value = production_metrics[metric]
                canary_value = canary_metrics[metric]
                
                if metric in ["accuracy", "precision", "recall", "f1_score", "r2"]:
                    if canary_value < prod_value * 0.9:  # 10% degradation
                        alerts.append(f"Warning: {metric} degraded from {prod_value:.3f} to {canary_value:.3f}")
                
                elif metric in ["mae", "rmse", "latency", "error_rate"]:
                    if canary_value > prod_value * 1.1:  # 10% increase
                        alerts.append(f"Warning: {metric} increased from {prod_value:.3f} to {canary_value:.3f}")
        
        return alerts
    
    def evaluate_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Evaluate a canary deployment and make promotion/rollback decision"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        if deployment.stage != CanaryStage.EVALUATING:
            return {"status": "not_ready", "reason": f"Deployment in {deployment.stage.value} stage"}
        
        # Check evaluation window
        evaluation_duration = datetime.now() - deployment.start_time
        if evaluation_duration.total_seconds() < self.canary_config.evaluation_window_hours * 3600:
            return {"status": "not_ready", "reason": "Evaluation window not met"}
        
        # Calculate average comparison score
        if not deployment.metrics_history:
            return {"status": "not_ready", "reason": "No metrics recorded"}
        
        recent_metrics = deployment.metrics_history[-10:]  # Last 10 measurements
        avg_score = np.mean([m["comparison_score"] for m in recent_metrics])
        
        # Make decision
        if avg_score >= self.canary_config.promotion_threshold:
            return self._promote_deployment(deployment_id)
        elif avg_score < self.canary_config.rollback_threshold:
            return self._rollback_deployment(deployment_id)
        else:
            return {"status": "continue", "reason": f"Score {avg_score:.3f} in uncertain range"}
    
    def _promote_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Promote canary deployment to production"""
        
        deployment = self.active_deployments[deployment_id]
        deployment.stage = CanaryStage.PROMOTING
        
        try:
            # Promote model in registry
            self.registry.promote_model(
                deployment.model_type,
                deployment.candidate_version,
                ModelStage.PRODUCTION,
                reason="Canary deployment successful"
            )
            
            # Update deployment status
            deployment.status = DeploymentStatus.SUCCESS
            deployment.end_time = datetime.now()
            deployment.final_status = "promoted"
            
            # Remove from active deployments
            del self.active_deployments[deployment_id]
            
            # Save deployments
            self._save_deployments()
            
            # Log promotion
            if self.tracker:
                self.tracker.log_metrics({
                    "canary_promoted": 1,
                    "final_score": deployment.metrics_history[-1]["comparison_score"] if deployment.metrics_history else 0.0
                })
            
            logger.info(f"Promoted canary deployment {deployment_id} to production")
            
            return {
                "status": "promoted",
                "reason": "Canary deployment successful",
                "final_score": deployment.metrics_history[-1]["comparison_score"] if deployment.metrics_history else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to promote deployment {deployment_id}: {e}")
            deployment.stage = CanaryStage.EVALUATING
            return {"status": "error", "reason": str(e)}
    
    def _rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback canary deployment"""
        
        deployment = self.active_deployments[deployment_id]
        deployment.stage = CanaryStage.ROLLING_BACK
        deployment.rollback_count += 1
        
        try:
            # Check if we should rollback the production model
            if deployment.rollback_count >= self.canary_config.max_rollback_count:
                # Rollback production model to previous version
                self.registry.rollback_model(
                    deployment.model_type,
                    deployment.production_version,
                    ModelStage.STAGING,
                    reason="Multiple canary failures"
                )
            
            # Update deployment status
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.end_time = datetime.now()
            deployment.final_status = "rolled_back"
            
            # Remove from active deployments
            del self.active_deployments[deployment_id]
            
            # Save deployments
            self._save_deployments()
            
            # Log rollback
            if self.tracker:
                self.tracker.log_metrics({
                    "canary_rolled_back": 1,
                    "rollback_count": deployment.rollback_count
                })
            
            logger.info(f"Rolled back canary deployment {deployment_id}")
            
            return {
                "status": "rolled_back",
                "reason": "Canary deployment failed",
                "rollback_count": deployment.rollback_count
            }
            
        except Exception as e:
            logger.error(f"Failed to rollback deployment {deployment_id}: {e}")
            deployment.stage = CanaryStage.EVALUATING
            return {"status": "error", "reason": str(e)}
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific deployment"""
        
        if deployment_id not in self.active_deployments:
            return None
        
        deployment = self.active_deployments[deployment_id]
        
        return {
            "deployment_id": deployment.deployment_id,
            "model_type": deployment.model_type,
            "candidate_version": deployment.candidate_version,
            "production_version": deployment.production_version,
            "status": deployment.status.value,
            "stage": deployment.stage.value,
            "start_time": deployment.start_time.isoformat(),
            "current_percentage": deployment.current_percentage,
            "target_percentage": deployment.target_percentage,
            "metrics_count": len(deployment.metrics_history),
            "health_checks_count": len(deployment.health_checks),
            "rollback_count": deployment.rollback_count
        }
    
    def list_deployments(self, status: Optional[DeploymentStatus] = None) -> List[Dict[str, Any]]:
        """List all deployments with optional filtering"""
        
        deployments = []
        
        for deployment in self.active_deployments.values():
            if status and deployment.status != status:
                continue
            
            deployments.append(self.get_deployment_status(deployment.deployment_id))
        
        # Sort by start time (newest first)
        deployments.sort(key=lambda x: x["start_time"], reverse=True)
        return deployments
    
    def cleanup_completed_deployments(self, days_to_keep: int = 30) -> int:
        """Clean up completed deployments older than specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        # Load all deployments from storage
        deployments_file = self.deployment_path / "deployments.json"
        if not deployments_file.exists():
            return 0
        
        try:
            with open(deployments_file, 'r') as f:
                all_deployments = json.load(f)
            
            # Filter out old completed deployments
            recent_deployments = []
            for deployment_data in all_deployments:
                start_time = datetime.fromisoformat(deployment_data["start_time"])
                
                if (start_time >= cutoff_date or 
                    deployment_data["status"] in ["pending", "running"]):
                    recent_deployments.append(deployment_data)
                else:
                    cleaned_count += 1
            
            # Save filtered deployments
            with open(deployments_file, 'w') as f:
                json.dump(recent_deployments, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to cleanup deployments: {e}")
            return 0
        
        logger.info(f"Cleaned up {cleaned_count} old deployments")
        return cleaned_count
    
    def get_deployment_metrics(self, deployment_id: str, 
                              hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for a deployment"""
        
        if deployment_id not in self.active_deployments:
            return []
        
        deployment = self.active_deployments[deployment_id]
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter metrics by time
        recent_metrics = []
        for metrics in deployment.metrics_history:
            timestamp = datetime.fromisoformat(metrics["timestamp"])
            if timestamp >= cutoff_time:
                recent_metrics.append(metrics)
        
        return recent_metrics
    
    def force_rollback(self, deployment_id: str, reason: str = "Manual rollback") -> bool:
        """Force rollback of a canary deployment"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        # Perform rollback
        result = self._rollback_deployment(deployment_id)
        
        if result["status"] == "rolled_back":
            logger.info(f"Force rollback of {deployment_id} successful: {reason}")
            return True
        else:
            logger.error(f"Force rollback of {deployment_id} failed: {result}")
            return False
