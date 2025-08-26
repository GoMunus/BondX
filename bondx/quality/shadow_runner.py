#!/usr/bin/env python3
"""
Shadow Runner for Hybrid Testing

Compares quality outcomes between synthetic and real data slices to detect
discrepancies and validate synthetic data quality.

Usage:
    python -m bondx.quality.shadow_runner --real data/real_shadow --synthetic data/synthetic_subset --out quality/shadow
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Add bondx to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bondx.quality.validators import DataValidator
from bondx.quality.quality_gates import QualityGateManager
from bondx.quality.metrics import MetricsCollector

class ShadowRunner:
    """Compares quality outcomes between real and synthetic data."""
    
    def __init__(self, tolerance_config: Optional[Dict[str, Any]] = None):
        """Initialize the shadow runner."""
        self.logger = logging.getLogger(__name__)
        
        # Default tolerance thresholds
        self.default_tolerances = {
            'fail_rate_variance': 0.05,      # ±5% acceptable variance
            'warn_rate_variance': 0.10,      # ±10% acceptable variance
            'coverage_variance': 0.03,       # ±3% acceptable variance
            'esg_completeness_variance': 0.05,  # ±5% acceptable variance
            'liquidity_variance': 0.08,      # ±8% acceptable variance
            'validation_count_variance': 0.15,  # ±15% acceptable variance
            'gate_outcome_variance': 0.10    # ±10% acceptable variance
        }
        
        # Merge with config if available
        if tolerance_config:
            self.tolerances = {**self.default_tolerances, **tolerance_config}
        else:
            self.tolerances = self.default_tolerances
    
    def load_dataset(self, dataset_path: str, dataset_type: str) -> pd.DataFrame:
        """Load dataset from various formats."""
        try:
            path = Path(dataset_path)
            
            if path.suffix.lower() == '.csv':
                data = pd.read_csv(path)
            elif path.suffix.lower() == '.json':
                data = pd.read_json(path)
            elif path.suffix.lower() == '.parquet':
                data = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            self.logger.info(f"Loaded {dataset_type} dataset: {len(data)} records, {len(data.columns)} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"Could not load {dataset_type} dataset from {dataset_path}: {e}")
            raise
    
    def anonymize_data(self, data: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Anonymize real data for shadow testing."""
        if dataset_type != 'real':
            return data
        
        self.logger.info("Anonymizing real data for shadow testing...")
        
        # Create a copy to avoid modifying original
        anonymized = data.copy()
        
        # Hash sensitive identifiers
        if 'issuer_name' in anonymized.columns:
            anonymized['issuer_name'] = anonymized['issuer_name'].apply(
                lambda x: f"ANON_{hash(str(x)) % 10000:04d}" if pd.notna(x) else x
            )
        
        if 'record_id' in anonymized.columns:
            anonymized['record_id'] = anonymized['record_id'].apply(
                lambda x: f"SHADOW_{hash(str(x)) % 10000:04d}" if pd.notna(x) else x
            )
        
        # Remove or hash other potentially sensitive fields
        sensitive_fields = ['email', 'phone', 'address', 'tax_id', 'ssn']
        for field in sensitive_fields:
            if field in anonymized.columns:
                anonymized[field] = anonymized[field].apply(
                    lambda x: f"HASH_{hash(str(x)) % 10000:04d}" if pd.notna(x) else x
                )
        
        # Add anonymization metadata
        anonymized['_anonymized'] = True
        anonymized['_anonymization_timestamp'] = datetime.now().isoformat()
        anonymized['_original_record_count'] = len(data)
        
        self.logger.info("Data anonymization completed")
        return anonymized
    
    def run_quality_pipeline(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Run quality pipeline on dataset."""
        self.logger.info(f"Running quality pipeline on {dataset_name}")
        
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        metrics_collector = MetricsCollector()
        
        # Run validators
        self.logger.info("Running validators...")
        bond_validation = validator.validate_bond_data(data)
        liquidity_validation = validator.validate_liquidity_data(data)
        esg_validation = validator.validate_esg_data(data)
        completeness_validation = validator.validate_dataset_completeness(data, dataset_name)
        
        all_validation_results = (bond_validation + liquidity_validation + 
                                esg_validation + completeness_validation)
        
        # Run quality gates
        self.logger.info("Running quality gates...")
        gate_results = gate_manager.run_all_gates(data, dataset_name)
        
        # Collect metrics
        self.logger.info("Collecting metrics...")
        metrics = metrics_collector.collect_metrics(
            data=data,
            dataset_name=dataset_name,
            validation_results=all_validation_results,
            quality_gates_results=gate_results,
            dataset_version="shadow"
        )
        
        # Prepare results
        results = {
            'validation_results': [result.__dict__ for result in all_validation_results],
            'gate_results': gate_results,
            'metrics': metrics_collector.get_metrics_dict(),
            'dataset_info': {
                'name': dataset_name,
                'record_count': len(data),
                'column_count': len(data.columns),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return results
    
    def calculate_quality_metrics(self, validation_results: List[Dict], 
                                 gate_results: Dict, metrics: Dict) -> Dict[str, Any]:
        """Calculate key quality metrics from results."""
        # Count validation results by severity
        severity_counts = {}
        for result in validation_results:
            severity = result.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        total_validations = len(validation_results)
        
        # Calculate rates
        pass_rate = severity_counts.get('PASS', 0) / total_validations if total_validations > 0 else 0
        warn_rate = severity_counts.get('WARN', 0) / total_validations if total_validations > 0 else 0
        fail_rate = severity_counts.get('FAIL', 0) / total_validations if total_validations > 0 else 0
        
        # Count gate results
        gate_pass = sum(1 for g in gate_results.values() if g.get('status') == 'PASS')
        gate_warn = sum(1 for g in gate_results.values() if g.get('status') == 'WARN')
        gate_fail = sum(1 for g in gate_results.values() if g.get('status') == 'FAIL')
        total_gates = len(gate_results)
        
        # Calculate gate rates
        gate_pass_rate = gate_pass / total_gates if total_gates > 0 else 0
        gate_warn_rate = gate_warn / total_gates if total_gates > 0 else 0
        gate_fail_rate = gate_fail / total_gates if total_gates > 0 else 0
        
        return {
            'total_validations': total_validations,
            'pass_rate': pass_rate,
            'warn_rate': warn_rate,
            'fail_rate': fail_rate,
            'total_gates': total_gates,
            'gate_pass_rate': gate_pass_rate,
            'gate_warn_rate': gate_warn_rate,
            'gate_fail_rate': gate_fail_rate,
            'coverage_percent': metrics.get('coverage_percent', 0),
            'esg_completeness_percent': metrics.get('esg_completeness_percent', 0),
            'liquidity_index_median': metrics.get('liquidity_index_median', 0),
            'data_freshness_minutes': metrics.get('data_freshness_minutes', 0)
        }
    
    def compare_metrics(self, real_metrics: Dict[str, Any], 
                       synthetic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics between real and synthetic datasets."""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'metrics_comparison': {},
            'tolerance_checks': {},
            'overall_assessment': 'PASS'
        }
        
        # Compare each metric
        for metric_name in real_metrics.keys():
            if metric_name in synthetic_metrics:
                real_value = real_metrics[metric_name]
                synthetic_value = synthetic_metrics[metric_name]
                
                # Skip non-numeric metrics
                if not isinstance(real_value, (int, float)) or not isinstance(synthetic_value, (int, float)):
                    continue
                
                # Calculate variance
                if real_value != 0:
                    variance = abs(synthetic_value - real_value) / real_value
                else:
                    variance = 0 if synthetic_value == 0 else float('inf')
                
                # Get tolerance threshold
                tolerance_key = f"{metric_name.replace('_', '_').replace('percent', '')}_variance"
                tolerance = self.tolerances.get(tolerance_key, 0.10)  # Default 10%
                
                # Check if within tolerance
                within_tolerance = variance <= tolerance
                
                comparison['metrics_comparison'][metric_name] = {
                    'real_value': real_value,
                    'synthetic_value': synthetic_value,
                    'variance': variance,
                    'tolerance': tolerance,
                    'within_tolerance': within_tolerance
                }
                
                comparison['tolerance_checks'][metric_name] = within_tolerance
                
                # Update overall assessment
                if not within_tolerance:
                    comparison['overall_assessment'] = 'FAIL'
        
        # Calculate overall tolerance score
        total_checks = len(comparison['tolerance_checks'])
        passed_checks = sum(comparison['tolerance_checks'].values())
        
        if total_checks > 0:
            tolerance_score = passed_checks / total_checks
            comparison['tolerance_score'] = tolerance_score
            
            if tolerance_score >= 0.9:
                comparison['overall_assessment'] = 'PASS'
            elif tolerance_score >= 0.7:
                comparison['overall_assessment'] = 'WARN'
            else:
                comparison['overall_assessment'] = 'FAIL'
        
        return comparison
    
    def generate_delta_report(self, real_results: Dict[str, Any], 
                             synthetic_results: Dict[str, Any],
                             comparison: Dict[str, Any]) -> str:
        """Generate human-readable delta report."""
        # Calculate metrics for both datasets
        real_metrics = self.calculate_quality_metrics(
            real_results['validation_results'],
            real_results['gate_results'],
            real_results['metrics']
        )
        
        synthetic_metrics = self.calculate_quality_metrics(
            synthetic_results['validation_results'],
            synthetic_results['gate_results'],
            synthetic_results['metrics']
        )
        
        report = f"""SHADOW RUNNER DELTA REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL ASSESSMENT: {comparison['overall_assessment']}
Tolerance Score: {comparison.get('tolerance_score', 'N/A'):.1%}

DATASET COMPARISON:
Real Dataset:
- Records: {real_results['dataset_info']['record_count']:,}
- Columns: {real_results['dataset_info']['column_count']}
- Validations: {real_metrics['total_validations']:,}
- Quality Gates: {real_metrics['total_gates']}

Synthetic Dataset:
- Records: {synthetic_results['dataset_info']['record_count']:,}
- Columns: {synthetic_results['dataset_info']['column_count']}
- Validations: {synthetic_metrics['total_validations']:,}
- Quality Gates: {synthetic_metrics['total_gates']}

METRICS COMPARISON:
"""
        
        for metric_name, comparison_data in comparison['metrics_comparison'].items():
            status_icon = "✅" if comparison_data['within_tolerance'] else "❌"
            report += f"{status_icon} {metric_name}:\n"
            report += f"   Real: {comparison_data['real_value']:.3f}\n"
            report += f"   Synthetic: {comparison_data['synthetic_value']:.3f}\n"
            report += f"   Variance: {comparison_data['variance']:.1%}\n"
            report += f"   Tolerance: ±{comparison_data['tolerance']:.1%}\n"
            report += f"   Status: {'PASS' if comparison_data['within_tolerance'] else 'FAIL'}\n\n"
        
        # Summary of tolerance checks
        passed_checks = sum(comparison['tolerance_checks'].values())
        total_checks = len(comparison['tolerance_checks'])
        
        report += f"TOLERANCE CHECK SUMMARY:\n"
        report += f"Passed: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)\n"
        report += f"Failed: {total_checks - passed_checks}/{total_checks} ({(total_checks - passed_checks)/total_checks*100:.1f}%)\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        if comparison['overall_assessment'] == 'PASS':
            report += "✅ Synthetic data quality is within acceptable tolerances\n"
            report += "   Continue using synthetic data for testing and development\n"
        elif comparison['overall_assessment'] == 'WARN':
            report += "⚠️  Some metrics are outside tolerances\n"
            report += "   Review synthetic data generation for affected metrics\n"
            report += "   Consider adjusting synthetic data characteristics\n"
        else:
            report += "❌ Significant discrepancies detected\n"
            report += "   Investigate synthetic data generation process\n"
            report += "   Review quality validation logic\n"
            report += "   Consider regenerating synthetic datasets\n"
        
        return report
    
    def run_shadow_comparison(self, real_data_path: str, synthetic_data_path: str,
                             output_dir: str = "quality/shadow") -> Dict[str, Any]:
        """Run complete shadow comparison between real and synthetic data."""
        self.logger.info("Starting shadow runner comparison...")
        
        # Load datasets
        real_data = self.load_dataset(real_data_path, 'real')
        synthetic_data = self.load_dataset(synthetic_data_path, 'synthetic')
        
        # Anonymize real data
        real_data_anon = self.anonymize_data(real_data, 'real')
        
        # Run quality pipeline on both datasets
        self.logger.info("Running quality pipeline on real data...")
        real_results = self.run_quality_pipeline(real_data_anon, 'real_shadow')
        
        self.logger.info("Running quality pipeline on synthetic data...")
        synthetic_results = self.run_quality_pipeline(synthetic_data, 'synthetic_subset')
        
        # Calculate and compare metrics
        real_metrics = self.calculate_quality_metrics(
            real_results['validation_results'],
            real_results['gate_results'],
            real_results['metrics']
        )
        
        synthetic_metrics = self.calculate_quality_metrics(
            synthetic_results['validation_results'],
            synthetic_results['gate_results'],
            synthetic_results['metrics']
        )
        
        comparison = self.compare_metrics(real_metrics, synthetic_metrics)
        
        # Generate delta report
        delta_report = self.generate_delta_report(real_results, synthetic_results, comparison)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        real_results_file = output_path / f"real_shadow_results_{timestamp}.json"
        with open(real_results_file, 'w') as f:
            json.dump(real_results, f, indent=2)
        
        synthetic_results_file = output_path / f"synthetic_subset_results_{timestamp}.json"
        with open(synthetic_results_file, 'w') as f:
            json.dump(synthetic_results, f, indent=2)
        
        # Save comparison
        comparison_file = output_path / f"shadow_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save delta report
        delta_report_file = output_path / f"shadow_delta_report_{timestamp}.txt"
        with open(delta_report_file, 'w') as f:
            f.write(delta_report)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_assessment': comparison['overall_assessment'],
            'tolerance_score': comparison.get('tolerance_score', 0),
            'files_generated': {
                'real_results': str(real_results_file),
                'synthetic_results': str(synthetic_results_file),
                'comparison': str(comparison_file),
                'delta_report': str(delta_report_file)
            },
            'dataset_sizes': {
                'real': len(real_data),
                'synthetic': len(synthetic_data)
            }
        }
        
        summary_file = output_path / f"shadow_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Shadow comparison completed. Results saved to {output_path}")
        
        return {
            'summary': summary,
            'comparison': comparison,
            'delta_report': delta_report,
            'output_files': summary['files_generated']
        }

def main():
    """Main function for shadow runner."""
    parser = argparse.ArgumentParser(description="Run shadow comparison between real and synthetic data")
    parser.add_argument('--real', required=True, help='Path to real data slice')
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data subset')
    parser.add_argument('--out', default='quality/shadow', help='Output directory')
    parser.add_argument('--tolerance-config', help='Path to tolerance configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load tolerance configuration if provided
    tolerance_config = None
    if args.tolerance_config:
        try:
            with open(args.tolerance_config, 'r') as f:
                tolerance_config = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load tolerance config: {e}")
    
    # Initialize shadow runner
    runner = ShadowRunner(tolerance_config)
    
    try:
        # Run shadow comparison
        results = runner.run_shadow_comparison(
            real_data_path=args.real,
            synthetic_data_path=args.synthetic,
            output_dir=args.out
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SHADOW RUNNER COMPLETED")
        print("="*60)
        print(f"Overall Assessment: {results['summary']['overall_assessment']}")
        print(f"Tolerance Score: {results['summary']['tolerance_score']:.1%}")
        print(f"Real Dataset: {results['summary']['dataset_sizes']['real']:,} records")
        print(f"Synthetic Dataset: {results['summary']['dataset_sizes']['synthetic']:,} records")
        
        print(f"\nOutput Files:")
        for file_type, file_path in results['summary']['files_generated'].items():
            print(f"  {file_type}: {file_path}")
        
        # Print delta report
        print(f"\n{results['delta_report']}")
        
        # Exit with appropriate code
        if results['summary']['overall_assessment'] == 'FAIL':
            print("\n❌ Shadow comparison failed - investigate discrepancies")
            return 1
        elif results['summary']['overall_assessment'] == 'WARN':
            print("\n⚠️  Shadow comparison passed with warnings")
            return 0
        else:
            print("\n✅ Shadow comparison passed successfully")
            return 0
        
    except Exception as e:
        print(f"\n❌ Error running shadow comparison: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
