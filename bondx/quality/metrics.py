"""
Metrics Collection for BondX Quality Assurance

This module provides unified KPI collection for coverage %, freshness %, model accuracy,
anomaly precision, and data drift indicators with JSON output and logging.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a dataset."""
    dataset_name: str
    timestamp: str
    run_id: str
    dataset_version: str
    
    # Coverage metrics
    total_rows: int
    total_columns: int
    overall_coverage_pct: float
    column_coverage: Dict[str, float]
    
    # Freshness metrics
    max_staleness_minutes: Optional[float]
    median_staleness_minutes: Optional[float]
    stale_records_pct: Optional[float]
    
    # Distribution metrics
    numeric_columns: List[str]
    distribution_stats: Dict[str, Dict[str, float]]
    
    # Quality indicators
    validation_results_summary: Dict[str, int]
    quality_gates_summary: Dict[str, Any]
    
    # Data drift indicators (if baseline available)
    drift_metrics: Optional[Dict[str, float]] = None

class MetricsCollector:
    """Collects and analyzes quality metrics across datasets."""
    
    def __init__(self, baseline_path: Optional[str] = None):
        self.baseline_metrics = self._load_baseline(baseline_path)
        self.current_metrics: List[QualityMetrics] = []
        
    def _load_baseline(self, baseline_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load baseline metrics for drift comparison."""
        if baseline_path and Path(baseline_path).exists():
            try:
                with open(baseline_path, 'r') as f:
                    baseline = json.load(f)
                    logger.info(f"Loaded baseline metrics from {baseline_path}")
                    return baseline
            except Exception as e:
                logger.warning(f"Failed to load baseline from {baseline_path}: {e}")
        return None
    
    def calculate_coverage_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive coverage metrics."""
        total_rows = len(data)
        total_columns = len(data.columns)
        
        if total_rows == 0:
            return {
                "total_rows": 0,
                "total_columns": total_columns,
                "overall_coverage_pct": 0.0,
                "column_coverage": {}
            }
        
        # Overall coverage
        non_null_counts = data.notna().sum()
        overall_coverage = (non_null_counts.sum() / (total_rows * total_columns)) * 100
        
        # Per-column coverage
        column_coverage = {}
        for col in data.columns:
            col_coverage = (data[col].notna().sum() / total_rows) * 100
            column_coverage[col] = round(col_coverage, 2)
        
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "overall_coverage_pct": round(overall_coverage, 2),
            "column_coverage": column_coverage
        }
    
    def calculate_freshness_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data freshness metrics."""
        # Look for timestamp columns
        timestamp_columns = [col for col in data.columns if any(keyword in col.lower() 
                                                              for keyword in ['time', 'date', 'timestamp', 'updated'])]
        
        if not timestamp_columns:
            return {
                "max_staleness_minutes": None,
                "median_staleness_minutes": None,
                "stale_records_pct": None
            }
        
        # Use the first timestamp column found
        timestamp_col = timestamp_columns[0]
        try:
            # Convert to datetime if needed
            if data[timestamp_col].dtype == 'object':
                timestamps = pd.to_datetime(data[timestamp_col], errors='coerce')
            else:
                timestamps = data[timestamp_col]
            
            # Calculate staleness relative to now
            now = pd.Timestamp.now()
            staleness = (now - timestamps).dt.total_seconds() / 60  # Convert to minutes
            
            # Remove invalid timestamps
            valid_staleness = staleness[staleness.notna() & (staleness >= 0)]
            
            if len(valid_staleness) == 0:
                return {
                    "max_staleness_minutes": None,
                    "median_staleness_minutes": None,
                    "stale_records_pct": None
                }
            
            max_staleness = valid_staleness.max()
            median_staleness = valid_staleness.median()
            
            # Calculate percentage of stale records (older than 1 hour)
            stale_threshold = 60  # minutes
            stale_records = (valid_staleness > stale_threshold).sum()
            stale_pct = (stale_records / len(valid_staleness)) * 100
            
            return {
                "max_staleness_minutes": round(max_staleness, 2),
                "median_staleness_minutes": round(median_staleness, 2),
                "stale_records_pct": round(stale_pct, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating freshness metrics: {e}")
            return {
                "max_staleness_minutes": None,
                "median_staleness_minutes": None,
                "stale_records_pct": None
            }
    
    def calculate_distribution_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate distribution statistics for numeric columns."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        distribution_stats = {}
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
                
            try:
                stats_dict = {
                    "count": len(col_data),
                    "mean": round(col_data.mean(), 4),
                    "std": round(col_data.std(), 4),
                    "min": round(col_data.min(), 4),
                    "max": round(col_data.max(), 4),
                    "median": round(col_data.median(), 4),
                    "q25": round(col_data.quantile(0.25), 4),
                    "q75": round(col_data.quantile(0.75), 4),
                    "skewness": round(stats.skew(col_data), 4) if len(col_data) > 2 else None,
                    "kurtosis": round(stats.kurtosis(col_data), 4) if len(col_data) > 2 else None
                }
                distribution_stats[col] = stats_dict
            except Exception as e:
                logger.warning(f"Error calculating stats for column {col}: {e}")
                distribution_stats[col] = {"error": str(e)}
        
        return {
            "numeric_columns": numeric_columns,
            "distribution_stats": distribution_stats
        }
    
    def calculate_drift_metrics(self, current_data: pd.DataFrame, baseline_metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate data drift metrics compared to baseline."""
        if not baseline_metrics or 'distribution_stats' not in baseline_metrics:
            return None
        
        drift_metrics = {}
        
        try:
            # Compare distributions for common numeric columns
            current_stats = self.calculate_distribution_metrics(current_data)
            baseline_stats = baseline_metrics.get('distribution_stats', {})
            
            common_columns = set(current_stats['numeric_columns']) & set(baseline_stats.keys())
            
            for col in common_columns:
                if col in current_stats['distribution_stats'] and col in baseline_stats:
                    current_mean = current_stats['distribution_stats'][col].get('mean')
                    baseline_mean = baseline_stats[col].get('mean')
                    
                    if current_mean is not None and baseline_mean is not None and baseline_mean != 0:
                        # Calculate relative drift
                        mean_drift = abs(current_mean - baseline_mean) / abs(baseline_mean) * 100
                        drift_metrics[f"{col}_mean_drift_pct"] = round(mean_drift, 2)
                        
                        # Calculate distribution similarity (simplified)
                        current_std = current_stats['distribution_stats'][col].get('std', 1)
                        baseline_std = baseline_stats[col].get('std', 1)
                        
                        if current_std > 0 and baseline_std > 0:
                            std_drift = abs(current_std - baseline_std) / baseline_std * 100
                            drift_metrics[f"{col}_std_drift_pct"] = round(std_drift, 2)
            
            # Overall drift score (average of all drifts)
            if drift_metrics:
                overall_drift = np.mean(list(drift_metrics.values()))
                drift_metrics['overall_drift_score'] = round(overall_drift, 2)
                
        except Exception as e:
            logger.warning(f"Error calculating drift metrics: {e}")
            return None
        
        return drift_metrics
    
    def collect_metrics(self, data: pd.DataFrame, dataset_name: str, 
                       validation_results: List[Any] = None,
                       quality_gates_results: List[Any] = None,
                       run_id: str = None,
                       dataset_version: str = "1.0") -> QualityMetrics:
        """Collect comprehensive metrics for a dataset."""
        logger.info(f"Collecting metrics for dataset: {dataset_name}")
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate all metrics
        coverage_metrics = self.calculate_coverage_metrics(data)
        freshness_metrics = self.calculate_freshness_metrics(data)
        distribution_metrics = self.calculate_distribution_metrics(data)
        
        # Calculate drift if baseline available
        drift_metrics = None
        if self.baseline_metrics:
            drift_metrics = self.calculate_drift_metrics(data, self.baseline_metrics)
        
        # Summarize validation results
        validation_summary = {"total": 0, "passes": 0, "failures": 0, "warnings": 0}
        if validation_results:
            validation_summary["total"] = len(validation_results)
            validation_summary["passes"] = sum(1 for r in validation_results if r.is_valid)
            validation_summary["failures"] = sum(1 for r in validation_results if r.severity == "FAIL")
            validation_summary["warnings"] = sum(1 for r in validation_results if r.severity == "WARN")
        
        # Summarize quality gates
        quality_gates_summary = {"total": 0, "passed": 0, "failed": 0}
        if quality_gates_results:
            quality_gates_summary["total"] = len(quality_gates_results)
            quality_gates_summary["passed"] = sum(1 for r in quality_gates_results if r.passed)
            quality_gates_summary["failed"] = sum(1 for r in quality_gates_results if not r.passed)
        
        # Create metrics object
        metrics = QualityMetrics(
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            run_id=run_id,
            dataset_version=dataset_version,
            total_rows=coverage_metrics["total_rows"],
            total_columns=coverage_metrics["total_columns"],
            overall_coverage_pct=coverage_metrics["overall_coverage_pct"],
            column_coverage=coverage_metrics["column_coverage"],
            max_staleness_minutes=freshness_metrics["max_staleness_minutes"],
            median_staleness_minutes=freshness_metrics["median_staleness_minutes"],
            stale_records_pct=freshness_metrics["stale_records_pct"],
            numeric_columns=distribution_metrics["numeric_columns"],
            distribution_stats=distribution_metrics["distribution_stats"],
            validation_results_summary=validation_summary,
            quality_gates_summary=quality_gates_summary,
            drift_metrics=drift_metrics
        )
        
        self.current_metrics.append(metrics)
        return metrics
    
    def export_metrics_json(self, output_path: str = "quality/last_run_report.json") -> str:
        """Export all collected metrics to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to dictionaries
        metrics_data = {
            "run_summary": {
                "total_datasets": len(self.current_metrics),
                "timestamp": datetime.now().isoformat(),
                "run_id": self.current_metrics[0].run_id if self.current_metrics else "unknown"
            },
            "datasets": [asdict(metrics) for metrics in self.current_metrics]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            logger.info(f"Metrics exported to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return ""
    
    def print_summary_table(self):
        """Print human-readable summary table."""
        if not self.current_metrics:
            print("No metrics collected yet.")
            return
        
        print("\n" + "="*80)
        print("QUALITY METRICS SUMMARY")
        print("="*80)
        
        for metrics in self.current_metrics:
            print(f"\nDataset: {metrics.dataset_name}")
            print(f"Run ID: {metrics.run_id}")
            print(f"Timestamp: {metrics.timestamp}")
            print("-" * 50)
            
            # Coverage
            print(f"Coverage: {metrics.overall_coverage_pct}% ({metrics.total_rows} rows, {metrics.total_columns} cols)")
            
            # Freshness
            if metrics.max_staleness_minutes is not None:
                print(f"Freshness: Max {metrics.max_staleness_minutes:.1f} min, Median {metrics.median_staleness_minutes:.1f} min")
                print(f"Stale Records: {metrics.stale_records_pct}%")
            
            # Validation
            val_summary = metrics.validation_results_summary
            print(f"Validation: {val_summary['passes']}/{val_summary['total']} passed, {val_summary['failures']} failed, {val_summary['warnings']} warnings")
            
            # Quality Gates
            gate_summary = metrics.quality_gates_summary
            print(f"Quality Gates: {gate_summary['passed']}/{gate_summary['total']} passed, {gate_summary['failed']} failed")
            
            # Drift
            if metrics.drift_metrics:
                overall_drift = metrics.drift_metrics.get('overall_drift_score', 'N/A')
                print(f"Data Drift Score: {overall_drift}")
            
            print("-" * 50)
        
        print("\n" + "="*80)
    
    def get_overall_health_score(self) -> float:
        """Calculate overall health score (0-100) based on all metrics."""
        if not self.current_metrics:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        for metrics in self.current_metrics:
            # Coverage score (30% weight)
            coverage_score = min(metrics.overall_coverage_pct, 100.0)
            total_score += coverage_score * 0.3
            max_score += 100.0 * 0.3
            
            # Validation score (30% weight)
            val_summary = metrics.validation_results_summary
            if val_summary['total'] > 0:
                validation_score = (val_summary['passes'] / val_summary['total']) * 100
                total_score += validation_score * 0.3
                max_score += 100.0 * 0.3
            
            # Quality gates score (25% weight)
            gate_summary = metrics.quality_gates_summary
            if gate_summary['total'] > 0:
                gate_score = (gate_summary['passed'] / gate_summary['total']) * 100
                total_score += gate_score * 0.25
                max_score += 100.0 * 0.25
            
            # Freshness score (15% weight)
            if metrics.stale_records_pct is not None:
                freshness_score = max(0, 100 - metrics.stale_records_pct)
                total_score += freshness_score * 0.15
                max_score += 100.0 * 0.15
        
        if max_score == 0:
            return 0.0
        
        return round((total_score / max_score) * 100, 2)
