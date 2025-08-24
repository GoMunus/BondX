"""
Enhanced Drift Detection Module

This module provides comprehensive drift detection capabilities including:
- Population Stability Index (PSI) calculation
- Kolmogorov-Smirnov tests for feature distributions
- Target drift detection
- Residual drift monitoring
- Automated drift threshold management
- Integration with experiment tracking
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from .config import MLOpsConfig
from .tracking import ExperimentTracker

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    """Result of drift detection analysis"""
    feature_name: str
    drift_type: str  # "feature", "target", "residual"
    drift_score: float
    threshold: float
    is_drifted: bool
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_metrics: Optional[Dict[str, Any]] = None

@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: datetime
    model_type: str
    model_version: str
    overall_drift_score: float
    drifted_features: List[str]
    drift_results: List[DriftResult]
    recommendations: List[str]
    metadata: Dict[str, Any]

class DriftMonitor:
    """Enhanced drift detection monitor"""
    
    def __init__(self, config: MLOpsConfig, tracker: Optional[ExperimentTracker] = None):
        """Initialize drift monitor"""
        self.config = config
        self.tracker = tracker
        
        # Setup storage paths
        self.drift_data_path = Path(config.data_storage_path) / "drift"
        self.drift_data_path.mkdir(parents=True, exist_ok=True)
        
        # Load baseline data
        self.baselines = self._load_baselines()
        
        # Drift thresholds from config
        self.ks_threshold = config.drift.ks_test_threshold
        self.psi_threshold = config.drift.psi_threshold
        self.feature_drift_threshold = config.drift.feature_drift_threshold
        self.target_drift_threshold = config.drift.target_drift_threshold
        self.residual_drift_threshold = config.drift.residual_drift_threshold
        
        logger.info("Drift monitor initialized")
    
    def _load_baselines(self) -> Dict[str, pd.DataFrame]:
        """Load baseline datasets for drift detection"""
        baselines = {}
        baseline_path = self.drift_data_path / "baselines"
        baseline_path.mkdir(exist_ok=True)
        
        for baseline_file in baseline_path.glob("*.parquet"):
            try:
                model_type = baseline_file.stem
                baselines[model_type] = pd.read_parquet(baseline_file)
                logger.info(f"Loaded baseline for {model_type}: {len(baselines[model_type])} samples")
            except Exception as e:
                logger.warning(f"Failed to load baseline {baseline_file}: {e}")
        
        return baselines
    
    def set_baseline(self, model_type: str, baseline_data: pd.DataFrame, 
                     save: bool = True) -> bool:
        """Set baseline data for drift detection"""
        try:
            # Validate baseline data
            if baseline_data.empty:
                raise ValueError("Baseline data cannot be empty")
            
            # Store baseline
            self.baselines[model_type] = baseline_data.copy()
            
            # Save baseline if requested
            if save:
                baseline_path = self.drift_data_path / "baselines" / f"{model_type}.parquet"
                baseline_data.to_parquet(baseline_path, index=False)
                logger.info(f"Saved baseline for {model_type}: {len(baseline_data)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set baseline for {model_type}: {e}")
            return False
    
    def detect_feature_drift(self, model_type: str, current_data: pd.DataFrame,
                           features: Optional[List[str]] = None) -> List[DriftResult]:
        """Detect feature drift using multiple statistical tests"""
        
        if model_type not in self.baselines:
            raise ValueError(f"No baseline data found for model type: {model_type}")
        
        baseline_data = self.baselines[model_type]
        
        if features is None:
            # Use common features between baseline and current data
            features = list(set(baseline_data.columns) & set(current_data.columns))
        
        drift_results = []
        
        for feature in features:
            if feature not in baseline_data.columns or feature not in current_data.columns:
                continue
            
            try:
                # Get feature data
                baseline_feature = baseline_data[feature].dropna()
                current_feature = current_data[feature].dropna()
                
                if len(baseline_feature) == 0 or len(current_feature) == 0:
                    continue
                
                # Calculate drift metrics
                drift_score, p_value, confidence_interval = self._calculate_feature_drift(
                    baseline_feature, current_feature
                )
                
                # Determine if drifted
                is_drifted = drift_score > self.feature_drift_threshold
                
                # Create drift result
                drift_result = DriftResult(
                    feature_name=feature,
                    drift_type="feature",
                    drift_score=drift_score,
                    threshold=self.feature_drift_threshold,
                    is_drifted=is_drifted,
                    p_value=p_value,
                    confidence_interval=confidence_interval,
                    additional_metrics={
                        "baseline_samples": len(baseline_feature),
                        "current_samples": len(current_feature),
                        "baseline_mean": float(baseline_feature.mean()),
                        "current_mean": float(current_feature.mean()),
                        "baseline_std": float(baseline_feature.std()),
                        "current_std": float(current_feature.std())
                    }
                )
                
                drift_results.append(drift_result)
                
            except Exception as e:
                logger.warning(f"Error calculating drift for feature {feature}: {e}")
                continue
        
        return drift_results
    
    def _calculate_feature_drift(self, baseline: pd.Series, current: pd.Series) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate feature drift using multiple statistical tests"""
        
        # Convert to numeric if needed
        baseline = pd.to_numeric(baseline, errors='coerce').dropna()
        current = pd.to_numeric(current, errors='coerce').dropna()
        
        if len(baseline) == 0 or len(current) == 0:
            return 1.0, 1.0, (0.0, 1.0)
        
        # Kolmogorov-Smirnov test
        try:
            ks_statistic, ks_p_value = stats.ks_2samp(baseline, current)
        except Exception:
            ks_statistic, ks_p_value = 1.0, 1.0
        
        # Population Stability Index (PSI)
        try:
            psi_score = self._calculate_psi(baseline, current)
        except Exception:
            psi_score = 1.0
        
        # Wasserstein distance (Earth Mover's Distance)
        try:
            wasserstein_distance = stats.wasserstein_distance(baseline, current)
            # Normalize by standard deviation
            normalized_wasserstein = wasserstein_distance / (baseline.std() + current.std()) * 2
        except Exception:
            normalized_wasserstein = 1.0
        
        # Combined drift score (weighted average)
        drift_score = (
            0.4 * ks_statistic +      # KS test weight
            0.4 * psi_score +         # PSI weight
            0.2 * normalized_wasserstein  # Wasserstein weight
        )
        
        # Confidence interval (simplified)
        confidence_interval = (max(0, drift_score - 0.1), min(1, drift_score + 0.1))
        
        return drift_score, ks_p_value, confidence_interval
    
    def _calculate_psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins based on baseline data
        baseline_bins = pd.cut(baseline, bins=bins, include_lowest=True)
        current_bins = pd.cut(current, bins=bins, include_lowest=True)
        
        # Calculate bin counts
        baseline_counts = baseline_bins.value_counts(normalize=True)
        current_counts = current_bins.value_counts(normalize=True)
        
        # Align bins
        all_bins = set(baseline_counts.index) | set(current_counts.index)
        baseline_counts = baseline_counts.reindex(all_bins, fill_value=0.001)  # Avoid log(0)
        current_counts = current_counts.reindex(all_bins, fill_value=0.001)
        
        # Calculate PSI
        psi = 0.0
        for bin_name in all_bins:
            baseline_pct = baseline_counts[bin_name]
            current_pct = current_counts[bin_name]
            
            if baseline_pct > 0 and current_pct > 0:
                psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        
        return abs(psi)
    
    def detect_target_drift(self, model_type: str, baseline_targets: pd.Series,
                           current_targets: pd.Series) -> DriftResult:
        """Detect target variable drift"""
        
        try:
            # Calculate target drift
            drift_score, p_value, confidence_interval = self._calculate_feature_drift(
                baseline_targets, current_targets
            )
            
            # Determine if drifted
            is_drifted = drift_score > self.target_drift_threshold
            
            return DriftResult(
                feature_name="target",
                drift_type="target",
                drift_score=drift_score,
                threshold=self.target_drift_threshold,
                is_drifted=is_drifted,
                p_value=p_value,
                confidence_interval=confidence_interval,
                additional_metrics={
                    "baseline_samples": len(baseline_targets),
                    "current_samples": len(current_targets),
                    "baseline_mean": float(baseline_targets.mean()),
                    "current_mean": float(current_targets.mean())
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating target drift: {e}")
            return DriftResult(
                feature_name="target",
                drift_type="target",
                drift_score=1.0,
                threshold=self.target_drift_threshold,
                is_drifted=True,
                additional_metrics={"error": str(e)}
            )
    
    def detect_residual_drift(self, model_type: str, baseline_residuals: pd.Series,
                             current_residuals: pd.Series) -> DriftResult:
        """Detect residual drift"""
        
        try:
            # Calculate residual drift
            drift_score, p_value, confidence_interval = self._calculate_feature_drift(
                baseline_residuals, current_residuals
            )
            
            # Determine if drifted
            is_drifted = drift_score > self.residual_drift_threshold
            
            return DriftResult(
                feature_name="residuals",
                drift_type="residual",
                drift_score=drift_score,
                threshold=self.residual_drift_threshold,
                is_drifted=is_drifted,
                p_value=p_value,
                confidence_interval=confidence_interval,
                additional_metrics={
                    "baseline_samples": len(baseline_residuals),
                    "current_samples": len(current_residuals),
                    "baseline_rmse": float(np.sqrt(mean_squared_error(baseline_residuals, np.zeros_like(baseline_residuals)))),
                    "current_rmse": float(np.sqrt(mean_squared_error(current_residuals, np.zeros_like(current_residuals))))
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating residual drift: {e}")
            return DriftResult(
                feature_name="residuals",
                drift_type="residual",
                drift_score=1.0,
                threshold=self.residual_drift_threshold,
                is_drifted=True,
                additional_metrics={"error": str(e)}
            )
    
    def comprehensive_drift_check(self, model_type: str, current_data: pd.DataFrame,
                                current_targets: Optional[pd.Series] = None,
                                current_residuals: Optional[pd.Series] = None,
                                features: Optional[List[str]] = None) -> DriftReport:
        """Perform comprehensive drift detection"""
        
        if model_type not in self.baselines:
            raise ValueError(f"No baseline data found for model type: {model_type}")
        
        baseline_data = self.baselines[model_type]
        
        # Detect feature drift
        feature_drift_results = self.detect_feature_drift(model_type, current_data, features)
        
        # Detect target drift if targets provided
        target_drift_result = None
        if current_targets is not None and 'target' in baseline_data.columns:
            target_drift_result = self.detect_target_drift(
                model_type, baseline_data['target'], current_targets
            )
        
        # Detect residual drift if residuals provided
        residual_drift_result = None
        if current_residuals is not None:
            # Use baseline residuals if available, otherwise assume zero
            baseline_residuals = baseline_data.get('residuals', pd.Series(0, index=baseline_data.index))
            residual_drift_result = self.detect_residual_drift(
                model_type, baseline_residuals, current_residuals
            )
        
        # Combine all results
        all_results = feature_drift_results
        if target_drift_result:
            all_results.append(target_drift_result)
        if residual_drift_result:
            all_results.append(residual_drift_result)
        
        # Calculate overall drift score
        if all_results:
            overall_drift_score = np.mean([r.drift_score for r in all_results])
        else:
            overall_drift_score = 0.0
        
        # Identify drifted features
        drifted_features = [r.feature_name for r in all_results if r.is_drifted]
        
        # Generate recommendations
        recommendations = self._generate_drift_recommendations(all_results)
        
        # Create drift report
        report = DriftReport(
            timestamp=datetime.now(),
            model_type=model_type,
            model_version="latest",  # This should be passed in
            overall_drift_score=overall_drift_score,
            drifted_features=drifted_features,
            drift_results=all_results,
            recommendations=recommendations,
            metadata={
                "baseline_samples": len(baseline_data),
                "current_samples": len(current_data),
                "drift_thresholds": {
                    "feature": self.feature_drift_threshold,
                    "target": self.target_drift_threshold,
                    "residual": self.residual_drift_threshold
                }
            }
        )
        
        # Log drift detection
        if self.tracker:
            self.tracker.log_metrics({
                "drift_check_performed": 1,
                "overall_drift_score": overall_drift_score,
                "drifted_features_count": len(drifted_features)
            })
        
        # Save drift report
        self._save_drift_report(report)
        
        logger.info(f"Drift check completed for {model_type}. Overall score: {overall_drift_score:.3f}")
        return report
    
    def _generate_drift_recommendations(self, drift_results: List[DriftResult]) -> List[str]:
        """Generate recommendations based on drift results"""
        
        recommendations = []
        
        # Count drifted features by type
        feature_drift_count = sum(1 for r in drift_results if r.drift_type == "feature" and r.is_drifted)
        target_drift_count = sum(1 for r in drift_results if r.drift_type == "target" and r.is_drifted)
        residual_drift_count = sum(1 for r in drift_results if r.drift_type == "residual" and r.is_drifted)
        
        if feature_drift_count > 0:
            recommendations.append(f"Feature drift detected in {feature_drift_count} features. Consider retraining the model.")
        
        if target_drift_count > 0:
            recommendations.append("Target drift detected. Data distribution has changed significantly.")
        
        if residual_drift_count > 0:
            recommendations.append("Residual drift detected. Model performance may be degrading.")
        
        if not recommendations:
            recommendations.append("No significant drift detected. Model appears stable.")
        
        return recommendations
    
    def _save_drift_report(self, report: DriftReport):
        """Save drift report to disk"""
        try:
            report_path = self.drift_data_path / "reports" / f"{report.model_type}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            # Convert report to serializable format
            report_dict = vars(report) # Use vars() to get all dataclass fields
            report_dict["timestamp"] = report_dict["timestamp"].isoformat()
            
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save drift report: {e}")
    
    def get_drift_history(self, model_type: str, days: int = 30) -> List[DriftReport]:
        """Get drift detection history"""
        
        reports = []
        reports_path = self.drift_data_path / "reports"
        
        if not reports_path.exists():
            return reports
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for report_file in reports_path.glob(f"{model_type}_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    timestamp = datetime.fromisoformat(report_data["timestamp"])
                    
                    if timestamp >= cutoff_date:
                        # Convert back to DriftReport object
                        report = DriftReport(**report_data)
                        report.timestamp = timestamp
                        reports.append(report)
                        
            except Exception as e:
                logger.warning(f"Error reading drift report {report_file}: {e}")
        
        # Sort by timestamp (newest first)
        reports.sort(key=lambda x: x.timestamp, reverse=True)
        return reports
    
    def plot_drift_analysis(self, drift_results: List[DriftResult], 
                           save_path: Optional[str] = None) -> None:
        """Create visualization of drift analysis"""
        
        if not drift_results:
            logger.warning("No drift results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Drift Analysis Results", fontsize=16)
        
        # Plot 1: Drift scores by feature
        feature_names = [r.feature_name for r in drift_results]
        drift_scores = [r.drift_score for r in drift_results]
        colors = ['red' if r.is_drifted else 'green' for r in drift_results]
        
        axes[0, 0].barh(feature_names, drift_scores, color=colors)
        axes[0, 0].axvline(x=self.feature_drift_threshold, color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Drift Score')
        axes[0, 0].set_title('Feature Drift Scores')
        axes[0, 0].legend()
        
        # Plot 2: Drift scores by type
        drift_types = [r.drift_type for r in drift_results]
        type_scores = [r.drift_score for r in drift_results]
        
        axes[0, 1].scatter(drift_types, type_scores, c=colors, s=100, alpha=0.7)
        axes[0, 1].set_ylabel('Drift Score')
        axes[0, 1].set_title('Drift Scores by Type')
        
        # Plot 3: Threshold comparison
        thresholds = [r.threshold for r in drift_results]
        axes[1, 0].scatter(drift_scores, thresholds, c=colors, s=100, alpha=0.7)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('Drift Score')
        axes[1, 0].set_ylabel('Threshold')
        axes[1, 0].set_title('Drift Score vs Threshold')
        
        # Plot 4: Summary statistics
        drifted_count = sum(1 for r in drift_results if r.is_drifted)
        total_count = len(drift_results)
        
        axes[1, 1].pie([drifted_count, total_count - drifted_count], 
                       labels=['Drifted', 'Stable'], 
                       colors=['red', 'green'],
                       autopct='%1.1f%%')
        axes[1, 1].set_title('Drift Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drift analysis plot saved to {save_path}")
        
        plt.show()
    
    def update_drift_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """Update drift detection thresholds"""
        try:
            if 'ks_test_threshold' in new_thresholds:
                self.ks_threshold = new_thresholds['ks_test_threshold']
            if 'psi_threshold' in new_thresholds:
                self.psi_threshold = new_thresholds['psi_threshold']
            if 'feature_drift_threshold' in new_thresholds:
                self.feature_drift_threshold = new_thresholds['feature_drift_threshold']
            if 'target_drift_threshold' in new_thresholds:
                self.target_drift_threshold = new_thresholds['target_drift_threshold']
            if 'residual_drift_threshold' in new_thresholds:
                self.residual_drift_threshold = new_thresholds['residual_drift_threshold']
            
            logger.info("Drift thresholds updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update drift thresholds: {e}")
            return False
