#!/usr/bin/env python3
"""
Quality Validator for BondX AI Autonomous Training Loop
Integrates with Golden Dataset Vault to validate datasets and ML model outputs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from ..quality import DataValidator, QualityGateManager, MetricsCollector
from ..quality.config.quality_policy import load_quality_policy

class QualityValidator:
    """
    Validates quality of generated datasets and ML model outputs against
    Golden Dataset Vault baselines and quality policy thresholds.
    """
    
    def __init__(self, 
                 golden_vault_path: str = "data/golden",
                 quality_policy_path: str = "bondx/quality/config/quality_policy.yaml",
                 output_dir: str = "bondx/ml/quality_reports"):
        """
        Initialize the quality validator.
        
        Args:
            golden_vault_path: Path to Golden Dataset Vault
            quality_policy_path: Path to quality policy configuration
            output_dir: Directory for quality reports
        """
        self.golden_vault_path = Path(golden_vault_path)
        self.quality_policy_path = Path(quality_policy_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quality components
        self.quality_policy = load_quality_policy(self.quality_policy_path)
        self.data_validator = DataValidator(self.quality_policy)
        self.quality_gate_manager = QualityGateManager(self.quality_policy)
        self.metrics_collector = MetricsCollector()
        
        # Setup logging
        self._setup_logging()
        
        # Load golden dataset baselines
        self.golden_baselines = self._load_golden_baselines()
        
        logger.info("Quality Validator initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        global logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/quality_validation.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
    
    def _load_golden_baselines(self) -> Dict[str, Any]:
        """Load golden dataset baselines for comparison."""
        baselines = {}
        
        try:
            # Load baseline files
            baseline_dir = self.golden_vault_path / "baselines"
            if baseline_dir.exists():
                for baseline_file in baseline_dir.glob("*.json"):
                    dataset_name = baseline_file.stem.replace("_baseline", "")
                    with open(baseline_file, 'r') as f:
                        baselines[dataset_name] = json.load(f)
            
            logger.info(f"Loaded {len(baselines)} golden dataset baselines")
            
        except Exception as e:
            logger.warning(f"Could not load golden baselines: {e}")
        
        return baselines
    
    def validate_dataset_quality(self, dataset_path: str, dataset_name: str = "enhanced_bonds") -> Dict[str, Any]:
        """
        Validate dataset quality against quality policy and golden baselines.
        
        Args:
            dataset_path: Path to dataset CSV file
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating dataset quality: {dataset_path}")
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded: {df.shape}")
            
            # Run quality validation pipeline
            validation_results = self._run_quality_pipeline(df, dataset_name)
            
            # Compare against golden baselines if available
            baseline_comparison = self._compare_against_baselines(validation_results, dataset_name)
            
            # Generate quality report
            quality_report = self._generate_quality_report(
                dataset_name, validation_results, baseline_comparison
            )
            
            # Save report
            report_path = self.output_dir / f"quality_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            logger.info(f"Quality validation complete. Report saved to: {report_path}")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error validating dataset quality: {e}")
            raise
    
    def _run_quality_pipeline(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Run the quality validation pipeline on the dataset."""
        
        # Data validation
        validation_results = self.data_validator.validate_dataframe(df)
        
        # Quality gate evaluation
        gate_results = self.quality_gate_manager.evaluate_gates(df, validation_results)
        
        # Metrics collection
        metrics = self.metrics_collector.collect_metrics(df, validation_results, gate_results)
        
        # Generate summary
        summary = self._generate_validation_summary(validation_results, gate_results, metrics)
        
        return {
            'validation_results': validation_results,
            'gate_results': gate_results,
            'metrics': metrics,
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'dataset_shape': df.shape
        }
    
    def _generate_validation_summary(self, validation_results: Dict, gate_results: Dict, metrics: Dict) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        
        # Count validation issues by severity
        issue_counts = {}
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            issue_counts[severity] = len([
                issue for issue in validation_results.get('issues', [])
                if issue.get('severity') == severity
            ])
        
        # Count gate outcomes
        gate_counts = {}
        for outcome in ['PASS', 'WARN', 'FAIL']:
            gate_counts[outcome] = len([
                gate for gate in gate_results.get('gate_results', [])
                if gate.get('outcome') == outcome
            ])
        
        # Calculate overall quality score
        total_issues = sum(issue_counts.values())
        total_gates = sum(gate_counts.values())
        
        if total_gates > 0:
            quality_score = (gate_counts.get('PASS', 0) / total_gates) * 100
        else:
            quality_score = 0
        
        return {
            'issue_counts': issue_counts,
            'gate_counts': gate_counts,
            'total_issues': total_issues,
            'total_gates': total_gates,
            'quality_score': quality_score,
            'overall_status': 'PASS' if quality_score >= 90 else 'WARN' if quality_score >= 70 else 'FAIL'
        }
    
    def _compare_against_baselines(self, validation_results: Dict, dataset_name: str) -> Dict[str, Any]:
        """Compare validation results against golden dataset baselines."""
        
        baseline_comparison = {
            'baseline_available': False,
            'comparison_results': {},
            'drift_detected': False,
            'drift_summary': {}
        }
        
        # Find matching baseline
        matching_baseline = None
        for baseline_name, baseline_data in self.golden_baselines.items():
            if dataset_name in baseline_name or 'enhanced' in baseline_name:
                matching_baseline = baseline_data
                break
        
        if not matching_baseline:
            logger.info("No matching golden baseline found for comparison")
            return baseline_comparison
        
        baseline_comparison['baseline_available'] = True
        baseline_comparison['baseline_name'] = list(self.golden_baselines.keys())[0]
        
        # Compare key metrics
        current_metrics = validation_results.get('metrics', {})
        baseline_metrics = matching_baseline.get('metrics', {})
        
        comparison_results = {}
        drift_detected = False
        
        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                current_value = current_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                # Calculate drift
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if baseline_value != 0:
                        drift_pct = abs(current_value - baseline_value) / abs(baseline_value) * 100
                    else:
                        drift_pct = 0
                    
                    # Check if drift exceeds threshold (5%)
                    drift_threshold = 5.0
                    significant_drift = drift_pct > drift_threshold
                    
                    comparison_results[metric_name] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'drift_pct': drift_pct,
                        'significant_drift': significant_drift
                    }
                    
                    if significant_drift:
                        drift_detected = True
                        logger.warning(f"Significant drift detected in {metric_name}: {drift_pct:.2f}%")
        
        baseline_comparison['comparison_results'] = comparison_results
        baseline_comparison['drift_detected'] = drift_detected
        
        # Generate drift summary
        if drift_detected:
            drifted_metrics = [
                metric for metric, result in comparison_results.items()
                if result.get('significant_drift', False)
            ]
            baseline_comparison['drift_summary'] = {
                'drifted_metrics': drifted_metrics,
                'total_drifted': len(drifted_metrics),
                'drift_threshold': 5.0
            }
        
        return baseline_comparison
    
    def _generate_quality_report(self, dataset_name: str, validation_results: Dict, 
                                baseline_comparison: Dict) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        summary = validation_results.get('summary', {})
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'validator_version': '1.0.0',
                'quality_policy_version': self.quality_policy.get('version', 'unknown')
            },
            'dataset_info': {
                'shape': validation_results.get('dataset_shape', (0, 0)),
                'total_records': validation_results.get('dataset_shape', (0, 0))[0],
                'total_features': validation_results.get('dataset_shape', (0, 0))[1]
            },
            'validation_summary': summary,
            'quality_gates': validation_results.get('gate_results', {}),
            'detailed_validation': validation_results.get('validation_results', {}),
            'metrics': validation_results.get('metrics', {}),
            'baseline_comparison': baseline_comparison,
            'quality_assessment': {
                'overall_quality_score': summary.get('quality_score', 0),
                'overall_status': summary.get('overall_status', 'UNKNOWN'),
                'quality_gates_passed': summary.get('gate_counts', {}).get('PASS', 0),
                'quality_gates_total': summary.get('total_gates', 0),
                'critical_issues': summary.get('issue_counts', {}).get('CRITICAL', 0),
                'high_issues': summary.get('issue_counts', {}).get('HIGH', 0),
                'drift_detected': baseline_comparison.get('drift_detected', False)
            },
            'recommendations': self._generate_recommendations(summary, baseline_comparison)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict, baseline_comparison: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        # Quality score recommendations
        quality_score = summary.get('quality_score', 0)
        if quality_score < 70:
            recommendations.append("Quality score below 70%. Review data generation process and quality gates.")
        elif quality_score < 90:
            recommendations.append("Quality score below 90%. Consider improving data quality or adjusting quality gates.")
        
        # Issue-based recommendations
        critical_issues = summary.get('issue_counts', {}).get('CRITICAL', 0)
        high_issues = summary.get('issue_counts', {}).get('HIGH', 0)
        
        if critical_issues > 0:
            recommendations.append(f"Address {critical_issues} critical issues before proceeding with model training.")
        
        if high_issues > 0:
            recommendations.append(f"Review {high_issues} high-severity issues to improve data quality.")
        
        # Drift-based recommendations
        if baseline_comparison.get('drift_detected', False):
            drifted_count = baseline_comparison.get('drift_summary', {}).get('total_drifted', 0)
            recommendations.append(f"Significant drift detected in {drifted_count} metrics. Review against golden baselines.")
        
        # Gate-based recommendations
        gate_counts = summary.get('gate_counts', {})
        failed_gates = gate_counts.get('FAIL', 0)
        if failed_gates > 0:
            recommendations.append(f"{failed_gates} quality gates failed. Review gate configurations and thresholds.")
        
        if not recommendations:
            recommendations.append("Data quality meets all standards. Proceed with confidence.")
        
        return recommendations
    
    def validate_ml_model_outputs(self, model_outputs: Dict[str, Any], 
                                 model_name: str = "unknown") -> Dict[str, Any]:
        """
        Validate ML model outputs for quality and consistency.
        
        Args:
            model_outputs: Dictionary containing model outputs and metrics
            model_name: Name of the model being validated
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating ML model outputs: {model_name}")
        
        validation_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'validation_passed': True,
            'issues': [],
            'metrics_validation': {},
            'quality_gates': {}
        }
        
        try:
            # Validate model performance metrics
            metrics_validation = self._validate_model_metrics(model_outputs)
            validation_results['metrics_validation'] = metrics_validation
            
            # Validate model outputs structure
            structure_validation = self._validate_model_structure(model_outputs)
            validation_results['structure_validation'] = structure_validation
            
            # Check quality gates for ML models
            ml_quality_gates = self._evaluate_ml_quality_gates(model_outputs)
            validation_results['quality_gates'] = ml_quality_gates
            
            # Overall validation result
            all_metrics_valid = all(metrics_validation.values())
            all_structure_valid = all(structure_validation.values())
            all_gates_passed = all(ml_quality_gates.values())
            
            validation_results['validation_passed'] = all_metrics_valid and all_structure_valid and all_gates_passed
            
            # Generate issues list
            if not all_metrics_valid:
                validation_results['issues'].append("Model performance metrics below acceptable thresholds")
            
            if not all_structure_valid:
                validation_results['issues'].append("Model output structure validation failed")
            
            if not all_gates_passed:
                validation_results['issues'].append("ML quality gates not met")
            
            logger.info(f"ML model validation complete: {'PASS' if validation_results['validation_passed'] else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error validating ML model outputs: {e}")
            validation_results['validation_passed'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _validate_model_metrics(self, model_outputs: Dict[str, Any]) -> Dict[str, bool]:
        """Validate model performance metrics against thresholds."""
        
        validation_results = {}
        
        # Extract metrics from model outputs
        metrics = model_outputs.get('metrics', {})
        
        # Validate spread model metrics
        if 'spread' in metrics:
            spread_metrics = metrics['spread']
            validation_results['spread_r2'] = spread_metrics.get('r2', 0) >= 0.7
            validation_results['spread_rmse'] = spread_metrics.get('rmse', float('inf')) <= 100
        
        # Validate downgrade model metrics
        if 'downgrade' in metrics:
            downgrade_metrics = metrics['downgrade']
            validation_results['downgrade_accuracy'] = downgrade_metrics.get('accuracy', 0) >= 0.8
            validation_results['downgrade_f1'] = downgrade_metrics.get('f1', 0) >= 0.7
        
        # Validate liquidity shock model metrics
        if 'liquidity_shock' in metrics:
            liquidity_metrics = metrics['liquidity_shock']
            validation_results['liquidity_accuracy'] = liquidity_metrics.get('accuracy', 0) >= 0.8
            validation_results['liquidity_f1'] = liquidity_metrics.get('f1', 0) >= 0.7
        
        return validation_results
    
    def _validate_model_structure(self, model_outputs: Dict[str, Any]) -> Dict[str, bool]:
        """Validate model output structure and completeness."""
        
        validation_results = {}
        
        # Check required keys
        required_keys = ['metrics', 'model_files', 'metadata']
        for key in required_keys:
            validation_results[f'has_{key}'] = key in model_outputs
        
        # Check metrics structure
        if 'metrics' in model_outputs:
            metrics = model_outputs['metrics']
            validation_results['metrics_not_empty'] = len(metrics) > 0
            
            # Check if all expected model types are present
            expected_models = ['spread', 'downgrade', 'liquidity_shock', 'anomaly']
            for model_type in expected_models:
                validation_results[f'has_{model_type}_metrics'] = model_type in metrics
        
        # Check model files
        if 'model_files' in model_outputs:
            model_files = model_outputs['model_files']
            validation_results['model_files_not_empty'] = len(model_files) > 0
            
            # Check if all expected model files are present
            expected_model_files = ['spread', 'downgrade', 'liquidity_shock', 'anomaly']
            for model_file in expected_model_files:
                validation_results[f'has_{model_file}_file'] = model_file in model_files
        
        return validation_results
    
    def _evaluate_ml_quality_gates(self, model_outputs: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate quality gates specific to ML models."""
        
        quality_gates = {}
        
        # Overall quality gate
        overall_quality = model_outputs.get('final_quality', {}).get('overall', False)
        quality_gates['overall_quality'] = overall_quality
        
        # Quality pass rate gate
        quality_pass_rate = model_outputs.get('final_quality', {}).get('pass_rate', 0)
        quality_gates['quality_pass_rate'] = quality_pass_rate >= 0.9
        
        # Convergence gate
        converged = model_outputs.get('converged', False)
        quality_gates['converged'] = converged
        
        # Training completion gate
        training_complete = model_outputs.get('training_complete', False)
        quality_gates['training_complete'] = training_complete
        
        return quality_gates
    
    def generate_quality_dashboard_metrics(self, quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metrics suitable for quality dashboard visualization."""
        
        dashboard_metrics = {
            'timestamp': quality_report.get('report_metadata', {}).get('generated_at', ''),
            'dataset_name': quality_report.get('dataset_name', ''),
            'overall_quality_score': quality_report.get('quality_assessment', {}).get('overall_quality_score', 0),
            'overall_status': quality_report.get('quality_assessment', {}).get('overall_status', 'UNKNOWN'),
            'quality_gates_passed': quality_report.get('quality_assessment', {}).get('quality_gates_passed', 0),
            'quality_gates_total': quality_report.get('quality_assessment', {}).get('quality_gates_total', 0),
            'critical_issues': quality_report.get('quality_assessment', {}).get('critical_issues', 0),
            'high_issues': quality_report.get('quality_assessment', {}).get('high_issues', 0),
            'drift_detected': quality_report.get('quality_assessment', {}).get('drift_detected', False),
            'pass_rate_percentage': (
                quality_report.get('quality_assessment', {}).get('quality_gates_passed', 0) /
                max(quality_report.get('quality_assessment', {}).get('quality_gates_total', 1), 1) * 100
            )
        }
        
        return dashboard_metrics

def main():
    """Main function to demonstrate quality validation."""
    
    # Initialize validator
    validator = QualityValidator()
    
    # Validate enhanced dataset
    dataset_path = "data/synthetic/enhanced_bonds_150plus.csv"
    if Path(dataset_path).exists():
        quality_report = validator.validate_dataset_quality(dataset_path, "enhanced_bonds_150plus")
        
        print(f"\nQuality Validation Complete!")
        print(f"Overall Quality Score: {quality_report['quality_assessment']['overall_quality_score']:.1f}%")
        print(f"Overall Status: {quality_report['quality_assessment']['overall_status']}")
        print(f"Quality Gates Passed: {quality_report['quality_assessment']['quality_gates_passed']}/{quality_report['quality_assessment']['quality_gates_total']}")
        print(f"Drift Detected: {quality_report['quality_assessment']['drift_detected']}")
        
        # Generate dashboard metrics
        dashboard_metrics = validator.generate_quality_dashboard_metrics(quality_report)
        print(f"\nDashboard Metrics: {dashboard_metrics}")
        
    else:
        print(f"Dataset not found: {dataset_path}")

if __name__ == "__main__":
    main()
