#!/usr/bin/env python3
"""
Golden Dataset Validation Harness

Runs quality pipeline on golden datasets and compares outputs to baselines
to detect silent drift in quality behavior.

Usage:
    python validate_golden.py                    # Validate all datasets
    python validate_golden.py --dataset v1_dirty # Validate specific dataset
    python validate_golden.py --verbose          # Verbose output
"""

import argparse
import json
import logging
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

# Add bondx to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bondx.quality.run import run_quality_pipeline
from bondx.quality.validators import DataValidator
from bondx.quality.quality_gates import QualityGateManager
from bondx.quality.metrics import MetricsCollector

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('golden_validation.log')
        ]
    )

def load_baseline(dataset_name: str, baseline_dir: Path) -> Dict[str, Any]:
    """Load baseline files for a dataset."""
    baseline_path = baseline_dir / dataset_name
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found for dataset {dataset_name}")
    
    baseline = {}
    
    # Load last_run_report.json
    report_file = baseline_path / "last_run_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            baseline['report'] = json.load(f)
    
    # Load metrics.json
    metrics_file = baseline_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            baseline['metrics'] = json.load(f)
    
    # Load summary.txt
    summary_file = baseline_path / "summary.txt"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            baseline['summary'] = f.read()
    
    return baseline

def run_quality_on_dataset(dataset_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run quality pipeline on a golden dataset."""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_path}")
    data = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(data)} records")
    
    # Initialize components
    validator = DataValidator()
    gate_manager = QualityGateManager()
    metrics_collector = MetricsCollector()
    
    # Run validators
    logger.info("Running validators...")
    bond_validation = validator.validate_bond_data(data)
    liquidity_validation = validator.validate_liquidity_data(data)
    esg_validation = validator.validate_esg_data(data)
    completeness_validation = validator.validate_dataset_completeness(data, "golden_dataset")
    
    all_validation_results = (bond_validation + liquidity_validation + 
                            esg_validation + completeness_validation)
    
    # Run quality gates
    logger.info("Running quality gates...")
    gate_results = gate_manager.run_all_gates(data, "golden_dataset")
    
    # Collect metrics
    logger.info("Collecting metrics...")
    metrics = metrics_collector.collect_metrics(
        data=data,
        dataset_name="golden_dataset",
        validation_results=all_validation_results,
        quality_gates_results=gate_results,
        dataset_version="golden"
    )
    
    # Export results
    results = {
        'validation_results': [result.__dict__ for result in all_validation_results],
        'gate_results': gate_results,
        'metrics': metrics_collector.get_metrics_dict()
    }
    
    # Save results
    report_file = output_dir / "last_run_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_collector.get_metrics_dict(), f, indent=2)
    
    # Generate summary
    summary = generate_summary(all_validation_results, gate_results, metrics_collector.get_metrics_dict())
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    logger.info(f"Results saved to {output_dir}")
    return results

def generate_summary(validation_results: List, gate_results: Dict, metrics: Dict) -> str:
    """Generate human-readable summary of validation results."""
    # Count validation results by severity
    pass_count = sum(1 for r in validation_results if r.severity == "PASS")
    warn_count = sum(1 for r in validation_results if r.severity == "WARN")
    fail_count = sum(1 for r in validation_results if r.severity == "FAIL")
    
    total_validations = len(validation_results)
    
    # Count gate results
    gate_pass = sum(1 for g in gate_results.values() if g.get('status') == 'PASS')
    gate_warn = sum(1 for g in gate_results.values() if g.get('status') == 'WARN')
    gate_fail = sum(1 for g in gate_results.values() if g.get('status') == 'FAIL')
    
    total_gates = len(gate_results)
    
    summary = f"""Golden Dataset Validation Summary
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

VALIDATION RESULTS:
Total Validations: {total_validations}
- PASS: {pass_count} ({pass_count/total_validations*100:.1f}%)
- WARN: {warn_count} ({warn_count/total_validations*100:.1f}%)
- FAIL: {fail_count} ({fail_count/total_validations*100:.1f}%)

QUALITY GATES:
Total Gates: {total_gates}
- PASS: {gate_pass} ({gate_pass/total_gates*100:.1f}%)
- WARN: {gate_warn} ({gate_warn/total_gates*100:.1f}%)
- FAIL: {gate_fail} ({gate_fail/total_gates*100:.1f}%)

KEY METRICS:
Coverage: {metrics.get('coverage_percent', 'N/A')}%
ESG Completeness: {metrics.get('esg_completeness_percent', 'N/A')}%
Liquidity Index Median: {metrics.get('liquidity_index_median', 'N/A')}
Data Freshness: {metrics.get('data_freshness_minutes', 'N/A')} minutes

VIOLATIONS BY TYPE:
"""
    
    # Group violations by rule
    violations_by_rule = {}
    for result in validation_results:
        if result.severity in ['WARN', 'FAIL']:
            rule = result.rule_name
            if rule not in violations_by_rule:
                violations_by_rule[rule] = {'warn': 0, 'fail': 0}
            violations_by_rule[rule][result.severity.lower()] += 1
    
    for rule, counts in violations_by_rule.items():
        summary += f"{rule}: {counts['fail']} FAIL, {counts['warn']} WARN\n"
    
    return summary

def normalize_for_comparison(data: Any) -> str:
    """Normalize data for comparison to handle non-deterministic elements."""
    if isinstance(data, dict):
        # Sort keys and normalize timestamps
        normalized = {}
        for key in sorted(data.keys()):
            value = data[key]
            if key in ['timestamp', 'generation_timestamp', 'last_quote_time']:
                # Normalize timestamps to remove seconds/microseconds
                if isinstance(value, str):
                    try:
                        dt = pd.to_datetime(value)
                        normalized[key] = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        normalized[key] = value
                else:
                    normalized[key] = value
            else:
                normalized[key] = normalize_for_comparison(value)
        return json.dumps(normalized, sort_keys=True, default=str)
    elif isinstance(data, list):
        # Sort lists for consistent comparison
        normalized = [normalize_for_comparison(item) for item in data]
        return json.dumps(sorted(normalized), sort_keys=True, default=str)
    else:
        return str(data)

def compare_outputs(current: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Compare current outputs to baseline outputs."""
    differences = {}
    all_match = True
    
    # Compare report
    if 'report' in baseline and 'validation_results' in current:
        current_normalized = normalize_for_comparison(current['validation_results'])
        baseline_normalized = normalize_for_comparison(baseline['report'].get('validation_results', []))
        
        if current_normalized != baseline_normalized:
            differences['validation_results'] = {
                'current_hash': hashlib.md5(current_normalized.encode()).hexdigest()[:8],
                'baseline_hash': hashlib.md5(baseline_normalized.encode()).hexdigest()[:8]
            }
            all_match = False
    
    # Compare metrics
    if 'metrics' in baseline and 'metrics' in current:
        current_normalized = normalize_for_comparison(current['metrics'])
        baseline_normalized = normalize_for_comparison(baseline['metrics'])
        
        if current_normalized != baseline_normalized:
            differences['metrics'] = {
                'current_hash': hashlib.md5(current_normalized.encode()).hexdigest()[:8],
                'baseline_hash': hashlib.md5(baseline_normalized.encode()).hexdigest()[:8]
            }
            all_match = False
    
    return all_match, differences

def validate_dataset(dataset_name: str, golden_dir: Path, baseline_dir: Path, 
                    verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Validate a single golden dataset against its baseline."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating dataset: {dataset_name}")
    
    # Check if dataset exists
    dataset_path = golden_dir / dataset_name / f"{dataset_name}.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False, {'error': f'Dataset not found: {dataset_path}'}
    
    # Check if baseline exists
    try:
        baseline = load_baseline(dataset_name, baseline_dir)
    except FileNotFoundError:
        logger.warning(f"No baseline found for {dataset_name}, skipping comparison")
        return True, {'warning': f'No baseline found for {dataset_name}'}
    
    # Run quality pipeline
    output_dir = golden_dir / "validation_outputs" / dataset_name
    current_results = run_quality_on_dataset(dataset_path, output_dir)
    
    # Compare to baseline
    matches, differences = compare_outputs(current_results, baseline)
    
    if matches:
        logger.info(f"✅ {dataset_name}: Outputs match baseline")
        return True, {'status': 'match', 'dataset': dataset_name}
    else:
        logger.error(f"❌ {dataset_name}: Outputs differ from baseline")
        logger.error(f"Differences: {differences}")
        return False, {
            'status': 'mismatch', 
            'dataset': dataset_name, 
            'differences': differences
        }

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate golden datasets against baselines")
    parser.add_argument('--dataset', help='Specific dataset to validate (e.g., v1_dirty)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--golden-dir', default='data/golden', help='Golden datasets directory')
    parser.add_argument('--baseline-dir', default='data/golden/baselines', help='Baselines directory')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    golden_dir = Path(args.golden_dir)
    baseline_dir = Path(args.baseline_dir)
    
    if not golden_dir.exists():
        logger.error(f"Golden directory not found: {golden_dir}")
        sys.exit(1)
    
    # Determine which datasets to validate
    if args.dataset:
        datasets_to_validate = [args.dataset]
    else:
        # Find all dataset directories
        datasets_to_validate = []
        for item in golden_dir.iterdir():
            if item.is_dir() and item.name.startswith('v1_') and not item.name.startswith('baselines'):
                datasets_to_validate.append(item.name)
    
    if not datasets_to_validate:
        logger.error("No datasets found to validate")
        sys.exit(1)
    
    logger.info(f"Validating {len(datasets_to_validate)} datasets: {', '.join(datasets_to_validate)}")
    
    # Validate each dataset
    results = {}
    all_passed = True
    
    for dataset_name in datasets_to_validate:
        passed, result = validate_dataset(dataset_name, golden_dir, baseline_dir, args.verbose)
        results[dataset_name] = result
        
        if not passed and 'error' not in result:
            all_passed = False
    
    # Generate unified report
    unified_report = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'datasets_validated': len(datasets_to_validate),
        'all_passed': all_passed,
        'results': results
    }
    
    # Save unified report
    report_file = golden_dir / "last_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(unified_report, f, indent=2)
    
    logger.info(f"Validation complete. Report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("GOLDEN DATASET VALIDATION SUMMARY")
    print("="*60)
    
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"❌ {dataset_name}: ERROR - {result['error']}")
        elif 'warning' in result:
            print(f"⚠️  {dataset_name}: WARNING - {result['warning']}")
        elif result.get('status') == 'match':
            print(f"✅ {dataset_name}: MATCH")
        else:
            print(f"❌ {dataset_name}: MISMATCH")
    
    print(f"\nOverall Status: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print(f"Report: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
