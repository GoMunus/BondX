"""
Quality Gates for BondX Data Pipeline

This module provides centralized, configurable thresholds for accept/reject decisions
and warnings to ensure data quality before downstream processing.
"""

import logging
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .validators import ValidationResult, DataValidator

logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate evaluation."""
    gate_name: str
    passed: bool
    severity: str  # 'FAIL', 'WARN', 'INFO'
    message: str
    threshold: Any
    actual_value: Any
    dataset: str

class QualityGateManager:
    """Manages quality gates with configurable thresholds and policies."""
    
    def __init__(self, policy_path: Optional[str] = None):
        self.policy = self._load_policy(policy_path)
        self.gate_results: List[QualityGateResult] = []
        
    def _load_policy(self, policy_path: Optional[str]) -> Dict[str, Any]:
        """Load quality policy from YAML file."""
        default_policy = {
            "thresholds": {
                "coverage_min": 90.0,  # Minimum coverage percentage
                "esg_missing_max": 20.0,  # Maximum allowed missing ESG data
                "liquidity_index_median_min": 30.0,  # Minimum median liquidity index
                "negative_spreads_max_pct": 1.0,  # Maximum percentage of negative spreads
                "maturity_anomalies_max_pct": 0.1,  # Maximum percentage of maturity anomalies
                "stale_quotes_minutes": 60,  # Minutes after which quotes are considered stale
            },
            "severity_mapping": {
                "coverage_threshold": "WARN",
                "esg_missing": "WARN", 
                "liquidity_index_low": "WARN",
                "negative_spreads": "FAIL",
                "maturity_anomalies": "FAIL",
                "stale_quotes": "WARN"
            },
            "ignore_list": []
        }
        
        if policy_path and Path(policy_path).exists():
            try:
                with open(policy_path, 'r') as f:
                    custom_policy = yaml.safe_load(f)
                    # Merge custom policy with defaults
                    default_policy.update(custom_policy)
                    logger.info(f"Loaded custom quality policy from {policy_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom policy from {policy_path}: {e}")
                logger.info("Using default quality policy")
        
        return default_policy
    
    def evaluate_coverage_gate(self, data: pd.DataFrame, dataset_name: str) -> QualityGateResult:
        """Evaluate coverage quality gate."""
        total_rows = len(data)
        if total_rows == 0:
            return QualityGateResult(
                gate_name="coverage_gate",
                passed=False,
                severity="FAIL",
                message="Dataset is empty",
                threshold=self.policy["thresholds"]["coverage_min"],
                actual_value=0.0,
                dataset=dataset_name
            )
        
        # Calculate overall coverage
        non_null_counts = data.notna().sum()
        coverage_pct = (non_null_counts.sum() / (total_rows * len(data.columns))) * 100
        
        threshold = self.policy["thresholds"]["coverage_min"]
        passed = coverage_pct >= threshold
        severity = self.policy["severity_mapping"].get("coverage_threshold", "WARN")
        
        result = QualityGateResult(
            gate_name="coverage_gate",
            passed=passed,
            severity=severity,
            message=f"Dataset coverage: {coverage_pct:.1f}% (threshold: {threshold}%)",
            threshold=threshold,
            actual_value=coverage_pct,
            dataset=dataset_name
        )
        
        self.gate_results.append(result)
        return result
    
    def evaluate_esg_completeness_gate(self, data: pd.DataFrame, dataset_name: str) -> QualityGateResult:
        """Evaluate ESG completeness quality gate."""
        esg_columns = [col for col in data.columns if col.startswith('esg_')]
        if not esg_columns:
            return QualityGateResult(
                gate_name="esg_completeness_gate",
                passed=True,
                severity="INFO",
                message="No ESG columns found in dataset",
                threshold=self.policy["thresholds"]["esg_missing_max"],
                actual_value=0.0,
                dataset=dataset_name
            )
        
        total_rows = len(data)
        esg_missing_counts = data[esg_columns].isna().sum().sum()
        esg_missing_pct = (esg_missing_counts / (total_rows * len(esg_columns))) * 100
        
        threshold = self.policy["thresholds"]["esg_missing_max"]
        passed = esg_missing_pct <= threshold
        severity = self.policy["severity_mapping"].get("esg_missing", "WARN")
        
        result = QualityGateResult(
            gate_name="esg_completeness_gate",
            passed=passed,
            severity=severity,
            message=f"ESG missing data: {esg_missing_pct:.1f}% (threshold: {threshold}%)",
            threshold=threshold,
            actual_value=esg_missing_pct,
            dataset=dataset_name
        )
        
        self.gate_results.append(result)
        return result
    
    def evaluate_liquidity_index_gate(self, data: pd.DataFrame, dataset_name: str) -> QualityGateResult:
        """Evaluate liquidity index quality gate."""
        if 'liquidity_index_0_100' not in data.columns:
            return QualityGateResult(
                gate_name="liquidity_index_gate",
                passed=True,
                severity="INFO",
                message="No liquidity index column found in dataset",
                threshold=self.policy["thresholds"]["liquidity_index_median_min"],
                actual_value=None,
                dataset=dataset_name
            )
        
        liquidity_data = data['liquidity_index_0_100'].dropna()
        if len(liquidity_data) == 0:
            return QualityGateResult(
                gate_name="liquidity_index_gate",
                passed=False,
                severity="WARN",
                message="All liquidity index values are null",
                threshold=self.policy["thresholds"]["liquidity_index_median_min"],
                actual_value=None,
                dataset=dataset_name
            )
        
        median_liquidity = liquidity_data.median()
        threshold = self.policy["thresholds"]["liquidity_index_median_min"]
        passed = median_liquidity >= threshold
        severity = self.policy["severity_mapping"].get("liquidity_index_low", "WARN")
        
        result = QualityGateResult(
            gate_name="liquidity_index_gate",
            passed=passed,
            severity=severity,
            message=f"Median liquidity index: {median_liquidity:.1f} (threshold: {threshold})",
            threshold=threshold,
            actual_value=median_liquidity,
            dataset=dataset_name
        )
        
        self.gate_results.append(result)
        return result
    
    def evaluate_negative_spreads_gate(self, data: pd.DataFrame, dataset_name: str) -> QualityGateResult:
        """Evaluate negative spreads quality gate."""
        if 'liquidity_spread_bps' not in data.columns:
            return QualityGateResult(
                gate_name="negative_spreads_gate",
                passed=True,
                severity="INFO",
                message="No spread column found in dataset",
                threshold=self.policy["thresholds"]["negative_spreads_max_pct"],
                actual_value=0.0,
                dataset=dataset_name
            )
        
        total_rows = len(data)
        negative_spreads = data[data['liquidity_spread_bps'] < 0]
        negative_pct = (len(negative_spreads) / total_rows) * 100
        
        threshold = self.policy["thresholds"]["negative_spreads_max_pct"]
        passed = negative_pct <= threshold
        severity = self.policy["severity_mapping"].get("negative_spreads", "FAIL")
        
        result = QualityGateResult(
            gate_name="negative_spreads_gate",
            passed=passed,
            severity=severity,
            message=f"Negative spreads: {negative_pct:.2f}% (threshold: {threshold}%)",
            threshold=threshold,
            actual_value=negative_pct,
            dataset=dataset_name
        )
        
        self.gate_results.append(result)
        return result
    
    def evaluate_maturity_anomalies_gate(self, data: pd.DataFrame, dataset_name: str) -> QualityGateResult:
        """Evaluate maturity anomalies quality gate."""
        if 'maturity_years' not in data.columns:
            return QualityGateResult(
                gate_name="maturity_anomalies_gate",
                passed=True,
                severity="INFO",
                message="No maturity column found in dataset",
                threshold=self.policy["thresholds"]["maturity_anomalies_max_pct"],
                actual_value=0.0,
                dataset=dataset_name
            )
        
        total_rows = len(data)
        # Check for maturity values outside reasonable bounds
        anomalies = data[(data['maturity_years'] < 0.1) | (data['maturity_years'] > 50)]
        anomaly_pct = (len(anomalies) / total_rows) * 100
        
        threshold = self.policy["thresholds"]["maturity_anomalies_max_pct"]
        passed = anomaly_pct <= threshold
        severity = self.policy["severity_mapping"].get("maturity_anomalies", "FAIL")
        
        result = QualityGateResult(
            gate_name="maturity_anomalies_gate",
            passed=passed,
            severity=severity,
            message=f"Maturity anomalies: {anomaly_pct:.2f}% (threshold: {threshold}%)",
            threshold=threshold,
            actual_value=anomaly_pct,
            dataset=dataset_name
        )
        
        self.gate_results.append(result)
        return result
    
    def run_all_gates(self, data: pd.DataFrame, dataset_name: str) -> List[QualityGateResult]:
        """Run all quality gates on the dataset."""
        logger.info(f"Running quality gates on dataset: {dataset_name}")
        
        gates = [
            self.evaluate_coverage_gate,
            self.evaluate_esg_completeness_gate,
            self.evaluate_liquidity_index_gate,
            self.evaluate_negative_spreads_gate,
            self.evaluate_maturity_anomalies_gate
        ]
        
        results = []
        for gate_func in gates:
            try:
                result = gate_func(data, dataset_name)
                results.append(result)
                logger.info(f"Gate {result.gate_name}: {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                logger.error(f"Error running gate {gate_func.__name__}: {e}")
                # Create a failed result for the gate
                failed_result = QualityGateResult(
                    gate_name=gate_func.__name__,
                    passed=False,
                    severity="FAIL",
                    message=f"Gate execution error: {str(e)}",
                    threshold=None,
                    actual_value=None,
                    dataset=dataset_name
                )
                results.append(failed_result)
        
        return results
    
    def has_critical_failures(self) -> bool:
        """Check if any gates have critical failures."""
        return any(result.severity == "FAIL" and not result.passed for result in self.gate_results)
    
    def get_failed_gates(self) -> List[QualityGateResult]:
        """Get all failed gates."""
        return [result for result in self.gate_results if not result.passed]
    
    def get_passed_gates(self) -> List[QualityGateResult]:
        """Get all passed gates."""
        return [result for result in self.gate_results if result.passed]
    
    def get_gate_summary(self) -> Dict[str, Any]:
        """Get summary of all gate results."""
        total_gates = len(self.gate_results)
        passed_gates = len(self.get_passed_gates())
        failed_gates = len(self.get_failed_gates())
        
        severity_counts = {}
        for result in self.gate_results:
            severity = result.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "pass_rate": (passed_gates / total_gates * 100) if total_gates > 0 else 0,
            "severity_breakdown": severity_counts,
            "has_critical_failures": self.has_critical_failures()
        }
