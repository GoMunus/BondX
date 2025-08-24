"""
Enhanced Model Registry Module

This module provides comprehensive model registry capabilities including:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Stage management (Development → Staging → Production → Archived)
- Model metadata and lineage tracking
- Integration with experiment tracking
- Model promotion and rollback workflows
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import yaml
import semver
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .config import MLOpsConfig
from .tracking import ExperimentTracker

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    model_type: str
    stage: ModelStage
    model_path: str
    metadata_path: str
    created_at: datetime
    created_by: str
    experiment_run_id: str
    git_sha: str
    performance_metrics: Dict[str, float]
    drift_metrics: Optional[Dict[str, float]] = None
    tags: Dict[str, str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.drift_metrics is None:
            self.drift_metrics = {}

@dataclass
class ModelMetadata:
    """Detailed model metadata"""
    model_type: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    training_data_info: Dict[str, Any]
    validation_metrics: Dict[str, float]
    model_size_bytes: int
    dependencies: Dict[str, str]
    created_at: datetime
    last_updated: datetime

class ModelRegistry:
    """Enhanced model registry with semantic versioning and stage management"""
    
    def __init__(self, config: MLOpsConfig, tracker: Optional[ExperimentTracker] = None):
        """Initialize model registry"""
        self.config = config
        self.tracker = tracker
        
        # Setup storage paths
        self.registry_path = Path(config.model_storage_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create stage directories
        for stage in ModelStage:
            (self.registry_path / stage.value).mkdir(exist_ok=True)
        
        # Create metadata directory
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        
        # Load registry index
        self.registry_index = self._load_registry_index()
        
        logger.info(f"Model registry initialized at {self.registry_path}")
    
    def _load_registry_index(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load registry index from disk"""
        index_path = self.registry_path / "registry_index.json"
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    
                    # Convert back to ModelVersion objects
                    index = {}
                    for model_type, versions in data.items():
                        index[model_type] = {}
                        for version_str, version_data in versions.items():
                            # Convert stage string back to enum
                            version_data['stage'] = ModelStage(version_data['stage'])
                            # Convert datetime strings back to datetime objects
                            version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                            index[model_type][version_str] = ModelVersion(**version_data)
                    
                    return index
            except Exception as e:
                logger.error(f"Error loading registry index: {e}")
        
        return {}
    
    def _save_registry_index(self):
        """Save registry index to disk"""
        index_path = self.registry_path / "registry_index.json"
        
        # Convert ModelVersion objects to serializable format
        serializable_index = {}
        for model_type, versions in self.registry_index.items():
            serializable_index[model_type] = {}
            for version_str, version in versions.items():
                version_dict = asdict(version)
                # Convert datetime to string for JSON serialization
                version_dict['created_at'] = version_dict['created_at'].isoformat()
                serializable_index[model_type][version_str] = version_dict
        
        with open(index_path, 'w') as f:
            json.dump(serializable_index, f, indent=2)
    
    def _generate_version(self, model_type: str, major: Optional[int] = None, 
                         minor: Optional[int] = None, patch: Optional[int] = None) -> str:
        """Generate semantic version for a model"""
        
        # Get existing versions for this model type
        existing_versions = self.registry_index.get(model_type, {})
        
        if not existing_versions:
            # First version
            return "1.0.0"
        
        # Find the highest version
        highest_version = "0.0.0"
        for version_str in existing_versions.keys():
            try:
                if semver.compare(version_str, highest_version) > 0:
                    highest_version = version_str
            except ValueError:
                logger.warning(f"Invalid version format: {version_str}")
                continue
        
        # Parse the highest version
        try:
            current = semver.VersionInfo.parse(highest_version)
            
            if major is not None:
                new_version = semver.VersionInfo(major, 0, 0)
            elif minor is not None:
                new_version = semver.VersionInfo(current.major, minor, 0)
            else:
                new_version = semver.VersionInfo(current.major, current.minor, current.patch + 1)
            
            return str(new_version)
            
        except ValueError as e:
            logger.error(f"Error parsing version {highest_version}: {e}")
            return "1.0.0"
    
    def register_model(self, model_type: str, model_path: str, metadata: ModelMetadata,
                      experiment_run_id: str, git_sha: str, 
                      performance_metrics: Dict[str, float],
                      version: Optional[str] = None, description: str = "",
                      tags: Optional[Dict[str, str]] = None) -> str:
        """Register a new model version"""
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_type)
        
        # Validate version format
        try:
            semver.VersionInfo.parse(version)
        except ValueError:
            raise ValueError(f"Invalid version format: {version}. Use semantic versioning (e.g., 1.0.0)")
        
        # Check if version already exists
        if model_type in self.registry_index and version in self.registry_index[model_type]:
            raise ValueError(f"Version {version} already exists for model type {model_type}")
        
        # Create model version
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            stage=ModelStage.DEVELOPMENT,
            model_path=model_path,
            metadata_path=str(self.registry_path / "metadata" / f"{model_type}_{version}_metadata.json"),
            created_at=datetime.now(),
            created_by=os.getenv("USER", "unknown"),
            experiment_run_id=experiment_run_id,
            git_sha=git_sha,
            performance_metrics=performance_metrics,
            tags=tags or {},
            description=description
        )
        
        # Save model metadata
        self._save_model_metadata(model_version, metadata)
        
        # Add to registry index
        if model_type not in self.registry_index:
            self.registry_index[model_type] = {}
        
        self.registry_index[model_type][version] = model_version
        
        # Save registry index
        self._save_registry_index()
        
        # Log registration
        if self.tracker:
            self.tracker.log_metrics({
                "model_registered": 1,
                "model_version": float(version.replace(".", ""))
            })
        
        logger.info(f"Registered model {model_type} version {version}")
        return version
    
    def _save_model_metadata(self, model_version: ModelVersion, metadata: ModelMetadata):
        """Save detailed model metadata"""
        metadata_path = Path(model_version.metadata_path)
        
        # Convert metadata to serializable format
        metadata_dict = asdict(metadata)
        metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
        metadata_dict['last_updated'] = metadata_dict['last_updated'].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def promote_model(self, model_type: str, version: str, target_stage: ModelStage,
                     reason: str = "") -> bool:
        """Promote a model to a new stage"""
        
        if model_type not in self.registry_index or version not in self.registry_index[model_type]:
            raise ValueError(f"Model {model_type} version {version} not found")
        
        model_version = self.registry_index[model_type][version]
        current_stage = model_version.stage
        
        # Validate stage transition
        if not self._is_valid_stage_transition(current_stage, target_stage):
            raise ValueError(f"Invalid stage transition from {current_stage.value} to {target_stage.value}")
        
        # Check promotion gates
        if not self._check_promotion_gates(model_version, target_stage):
            raise ValueError(f"Promotion gates not met for stage {target_stage.value}")
        
        # Update stage
        old_stage = model_version.stage
        model_version.stage = target_stage
        
        # Move model files to new stage directory
        self._move_model_to_stage(model_version, old_stage, target_stage)
        
        # Update registry index
        self._save_registry_index()
        
        # Log promotion
        if self.tracker:
            self.tracker.log_metrics({
                "model_promoted": 1,
                "from_stage": old_stage.value,
                "to_stage": target_stage.value
            })
        
        logger.info(f"Promoted model {model_type} version {version} from {old_stage.value} to {target_stage.value}")
        return True
    
    def _is_valid_stage_transition(self, current: ModelStage, target: ModelStage) -> bool:
        """Check if stage transition is valid"""
        valid_transitions = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: []  # No transitions from archived
        }
        
        return target in valid_transitions.get(current, [])
    
    def _check_promotion_gates(self, model_version: ModelVersion, target_stage: ModelStage) -> bool:
        """Check if promotion gates are met"""
        
        if target_stage == ModelStage.STAGING:
            # Basic quality gates for staging
            return (model_version.performance_metrics.get('accuracy', 0) >= 0.8 and
                    model_version.performance_metrics.get('f1_score', 0) >= 0.75)
        
        elif target_stage == ModelStage.PRODUCTION:
            # Stricter gates for production
            return (model_version.performance_metrics.get('accuracy', 0) >= 0.85 and
                    model_version.performance_metrics.get('f1_score', 0) >= 0.8 and
                    model_version.drift_metrics is not None and
                    model_version.drift_metrics.get('psi', 1.0) <= 0.25)
        
        return True
    
    def _move_model_to_stage(self, model_version: ModelVersion, old_stage: ModelStage, new_stage: ModelStage):
        """Move model files to new stage directory"""
        
        old_path = Path(model_version.model_path)
        new_path = self.registry_path / new_stage.value / f"{model_version.model_type}_{model_version.version}.pkl"
        
        if old_path.exists():
            import shutil
            shutil.move(str(old_path), str(new_path))
            model_version.model_path = str(new_path)
    
    def rollback_model(self, model_type: str, version: str, target_stage: ModelStage,
                      reason: str = "") -> bool:
        """Rollback a model to a previous stage"""
        
        if model_type not in self.registry_index or version not in self.registry_index[model_type]:
            raise ValueError(f"Model {model_type} version {version} not found")
        
        model_version = self.registry_index[model_type][version]
        current_stage = model_version.stage
        
        # Validate rollback
        if not self._is_valid_stage_transition(target_stage, current_stage):
            raise ValueError(f"Invalid rollback from {current_stage.value} to {target_stage.value}")
        
        # Update stage
        old_stage = model_version.stage
        model_version.stage = target_stage
        
        # Move model files to new stage directory
        self._move_model_to_stage(model_version, old_stage, target_stage)
        
        # Update registry index
        self._save_registry_index()
        
        # Log rollback
        if self.tracker:
            self.tracker.log_metrics({
                "model_rollback": 1,
                "from_stage": old_stage.value,
                "to_stage": target_stage.value
            })
        
        logger.info(f"Rolled back model {model_type} version {version} from {old_stage.value} to {target_stage.value}")
        return True
    
    def get_model(self, model_type: str, version: Optional[str] = None, 
                  stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """Get a model version"""
        
        if model_type not in self.registry_index:
            return None
        
        versions = self.registry_index[model_type]
        
        if version:
            return versions.get(version)
        
        if stage:
            # Find the latest version in the specified stage
            stage_versions = [v for v in versions.values() if v.stage == stage]
            if stage_versions:
                return max(stage_versions, key=lambda x: semver.VersionInfo.parse(x.version))
        
        # Return the latest version overall
        if versions:
            return max(versions.values(), key=lambda x: semver.VersionInfo.parse(x.version))
        
        return None
    
    def list_models(self, model_type: Optional[str] = None, 
                    stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List models with optional filtering"""
        
        models = []
        
        for mt, versions in self.registry_index.items():
            if model_type and mt != model_type:
                continue
            
            for version in versions.values():
                if stage and version.stage != stage:
                    continue
                models.append(version)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    def delete_model(self, model_type: str, version: str) -> bool:
        """Delete a model version"""
        
        if model_type not in self.registry_index or version not in self.registry_index[model_type]:
            raise ValueError(f"Model {model_type} version {version} not found")
        
        model_version = self.registry_index[model_type][version]
        
        # Check if model is in production
        if model_version.stage == ModelStage.PRODUCTION:
            raise ValueError("Cannot delete production models")
        
        # Remove model files
        model_path = Path(model_version.model_path)
        if model_path.exists():
            model_path.unlink()
        
        metadata_path = Path(model_version.metadata_path)
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from registry index
        del self.registry_index[model_type][version]
        
        # Clean up empty model types
        if not self.registry_index[model_type]:
            del self.registry_index[model_type]
        
        # Save registry index
        self._save_registry_index()
        
        logger.info(f"Deleted model {model_type} version {version}")
        return True
    
    def update_drift_metrics(self, model_type: str, version: str, 
                           drift_metrics: Dict[str, float]) -> bool:
        """Update drift metrics for a model"""
        
        if model_type not in self.registry_index or version not in self.registry_index[model_type]:
            raise ValueError(f"Model {model_type} version {version} not found")
        
        model_version = self.registry_index[model_type][version]
        model_version.drift_metrics = drift_metrics
        
        # Update registry index
        self._save_registry_index()
        
        logger.info(f"Updated drift metrics for model {model_type} version {version}")
        return True
    
    def get_model_lineage(self, model_type: str, version: str) -> List[ModelVersion]:
        """Get model lineage (previous versions)"""
        
        if model_type not in self.registry_index:
            return []
        
        versions = self.registry_index[model_type]
        current_version = semver.VersionInfo.parse(version)
        
        # Find all versions that are older than the current one
        lineage = []
        for v in versions.values():
            try:
                v_semver = semver.VersionInfo.parse(v.version)
                if v_semver < current_version:
                    lineage.append(v)
            except ValueError:
                continue
        
        # Sort by version (oldest first)
        lineage.sort(key=lambda x: semver.VersionInfo.parse(x.version))
        return lineage
    
    def cleanup_old_versions(self, model_type: str, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the specified number"""
        
        if model_type not in self.registry_index:
            return 0
        
        versions = self.registry_index[model_type]
        
        # Sort versions by creation date
        sorted_versions = sorted(versions.values(), key=lambda x: x.created_at)
        
        # Keep the most recent versions
        versions_to_delete = sorted_versions[:-keep_versions]
        
        deleted_count = 0
        for version in versions_to_delete:
            try:
                if self.delete_model(model_type, version.version):
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete version {version.version}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old versions for model type {model_type}")
        return deleted_count
