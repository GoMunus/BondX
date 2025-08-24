"""
Enhanced Experiment Tracking Module

This module provides comprehensive experiment tracking capabilities including:
- Parameter logging with validation
- Metric tracking and visualization
- Artifact management and versioning
- Environment and git information capture
- Integration with MLflow and other backends
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import git
import yaml
import mlflow
import mlflow.sklearn
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .config import MLOpsConfig

logger = logging.getLogger(__name__)

@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking"""
    experiment_name: str
    run_name: str
    run_id: str
    git_sha: str
    git_branch: str
    environment: str
    python_version: str
    dependencies: Dict[str, str]
    timestamp: datetime
    user: str
    tags: Dict[str, str]

@dataclass
class ModelArtifact:
    """Model artifact information"""
    artifact_path: str
    artifact_type: str
    size_bytes: int
    checksum: str
    created_at: datetime
    metadata: Dict[str, Any]

class ExperimentTracker:
    """Enhanced experiment tracking with MLflow integration"""
    
    def __init__(self, config: MLOpsConfig):
        """Initialize experiment tracker"""
        self.config = config
        self.current_run = None
        self.current_experiment = None
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Setup local tracking
        self._setup_local_tracking()
        
        logger.info(f"Experiment tracker initialized with backend: {config.tracking_backend}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        if self.config.tracking_backend == "mlflow":
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
            else:
                mlflow.set_tracking_uri(f"file://{self.config.artifact_storage_path}")
            
            # Set experiment name
            mlflow.set_experiment("bondx-mlops")
    
    def _setup_local_tracking(self):
        """Setup local tracking directories"""
        self.local_tracking_path = Path(self.config.artifact_storage_path)
        self.local_tracking_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.local_tracking_path / "experiments").mkdir(exist_ok=True)
        (self.local_tracking_path / "models").mkdir(exist_ok=True)
        (self.local_tracking_path / "artifacts").mkdir(exist_ok=True)
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                "git_sha": repo.head.object.hexsha,
                "git_branch": repo.active_branch.name,
                "git_remote": repo.remotes.origin.url if repo.remotes else "local"
            }
        except Exception as e:
            logger.warning(f"Could not get git info: {e}")
            return {
                "git_sha": "unknown",
                "git_branch": "unknown",
                "git_remote": "unknown"
            }
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information"""
        return {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.sys.platform,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "user": os.getenv("USER", "unknown")
        }
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current dependencies"""
        try:
            import pkg_resources
            return {dist.project_name: dist.version for dist in pkg_resources.working_set}
        except Exception as e:
            logger.warning(f"Could not get dependencies: {e}")
            return {}
    
    def start_experiment(self, experiment_name: str, run_name: str, 
                        tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run"""
        
        # Generate run ID
        run_id = hashlib.md5(f"{experiment_name}_{run_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # Get metadata
        git_info = self._get_git_info()
        env_info = self._get_environment_info()
        deps = self._get_dependencies()
        
        metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            run_name=run_name,
            run_id=run_id,
            git_sha=git_info["git_sha"],
            git_branch=git_info["git_branch"],
            environment=env_info["environment"],
            python_version=env_info["python_version"],
            dependencies=deps,
            timestamp=datetime.now(),
            user=env_info["user"],
            tags=tags or {}
        )
        
        # Start MLflow run if configured
        if self.config.tracking_backend == "mlflow":
            mlflow.start_run(run_name=run_name, tags=tags or {})
            mlflow.log_param("run_id", run_id)
            mlflow.log_param("git_sha", git_info["git_sha"])
            mlflow.log_param("environment", env_info["environment"])
        
        # Store metadata locally
        self._store_metadata_locally(run_id, metadata)
        
        self.current_run = run_id
        self.current_experiment = experiment_name
        
        logger.info(f"Started experiment run: {run_id} for {experiment_name}")
        return run_id
    
    def _store_metadata_locally(self, run_id: str, metadata: ExperimentMetadata):
        """Store experiment metadata locally"""
        metadata_path = self.local_tracking_path / "experiments" / f"{run_id}_metadata.json"
        
        # Convert datetime to string for JSON serialization
        metadata_dict = asdict(metadata)
        metadata_dict["timestamp"] = metadata_dict["timestamp"].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log experiment parameters"""
        if not self.current_run:
            raise ValueError("No active experiment run")
        
        # Log to MLflow
        if self.config.tracking_backend == "mlflow":
            mlflow.log_params(params)
        
        # Log locally
        self._log_locally("parameters", params)
        
        logger.info(f"Logged {len(params)} parameters for run {self.current_run}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log experiment metrics"""
        if not self.current_run:
            raise ValueError("No active experiment run")
        
        # Log to MLflow
        if self.config.tracking_backend == "mlflow":
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step=step)
        
        # Log locally
        self._log_locally("metrics", metrics, step)
        
        logger.info(f"Logged {len(metrics)} metrics for run {self.current_run}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)"""
        if not self.current_run:
            raise ValueError("No active experiment run")
        
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact not found: {local_path}")
        
        # Calculate artifact info
        size_bytes = local_path.stat().st_size
        checksum = self._calculate_checksum(local_path)
        
        artifact_info = ModelArtifact(
            artifact_path=artifact_path or local_path.name,
            artifact_type=local_path.suffix,
            size_bytes=size_bytes,
            checksum=checksum,
            created_at=datetime.now(),
            metadata={}
        )
        
        # Log to MLflow
        if self.config.tracking_backend == "mlflow":
            mlflow.log_artifact(str(local_path), artifact_path)
        
        # Store locally
        self._store_artifact_locally(local_path, artifact_info)
        
        logger.info(f"Logged artifact: {local_path.name} ({size_bytes} bytes)")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _store_artifact_locally(self, local_path: Path, artifact_info: ModelArtifact):
        """Store artifact information locally"""
        artifacts_dir = self.local_tracking_path / "artifacts" / self.current_run
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy artifact
        import shutil
        shutil.copy2(local_path, artifacts_dir / local_path.name)
        
        # Store metadata
        metadata_path = artifacts_dir / f"{local_path.stem}_info.json"
        artifact_dict = asdict(artifact_info)
        artifact_dict["created_at"] = artifact_dict["created_at"].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(artifact_dict, f, indent=2)
    
    def _log_locally(self, log_type: str, data: Any, step: Optional[int] = None):
        """Log data locally"""
        if not self.current_run:
            return
        
        log_dir = self.local_tracking_path / "experiments" / self.current_run
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        if step is not None:
            filename = f"{log_type}_{step}_{timestamp}.json"
        else:
            filename = f"{log_type}_{timestamp}.json"
        
        log_path = log_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        if isinstance(data, dict):
            data = {k: convert_numpy(v) for k, v in data.items()}
        else:
            data = convert_numpy(data)
        
        with open(log_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def end_experiment(self):
        """End the current experiment run"""
        if not self.current_run:
            logger.warning("No active experiment run to end")
            return
        
        # End MLflow run
        if self.config.tracking_backend == "mlflow":
            mlflow.end_run()
        
        # Log completion
        self.log_metrics({"experiment_duration_seconds": 
                         (datetime.now() - datetime.fromisoformat(
                             self._get_metadata()["timestamp"])).total_seconds()})
        
        logger.info(f"Ended experiment run: {self.current_run}")
        
        self.current_run = None
        self.current_experiment = None
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for current run"""
        if not self.current_run:
            return {}
        
        metadata_path = self.local_tracking_path / "experiments" / f"{self.current_run}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_experiment_history(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Get experiment history for a specific experiment"""
        experiments_dir = self.local_tracking_path / "experiments"
        if not experiments_dir.exists():
            return []
        
        history = []
        for metadata_file in experiments_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get("experiment_name") == experiment_name:
                        history.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history
    
    def cleanup_old_experiments(self, days_to_keep: int = 30):
        """Clean up old experiment data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        experiments_dir = self.local_tracking_path / "experiments"
        if not experiments_dir.exists():
            return
        
        cleaned_count = 0
        for metadata_file in experiments_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    timestamp = datetime.fromisoformat(metadata.get("timestamp", ""))
                    
                    if timestamp < cutoff_date:
                        run_id = metadata.get("run_id")
                        if run_id:
                            # Remove experiment directory
                            run_dir = experiments_dir / run_id
                            if run_dir.exists():
                                import shutil
                                shutil.rmtree(run_dir)
                            
                            # Remove metadata file
                            metadata_file.unlink()
                            
                            # Remove artifacts
                            artifacts_dir = self.local_tracking_path / "artifacts" / run_id
                            if artifacts_dir.exists():
                                shutil.rmtree(artifacts_dir)
                            
                            cleaned_count += 1
                            
            except Exception as e:
                logger.warning(f"Error processing metadata file {metadata_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} old experiments")
