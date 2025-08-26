#!/usr/bin/env python3
"""
BondX Quality Assurance CLI Runner

This script orchestrates the complete quality assurance pipeline:
1. Data validation
2. Quality gates evaluation
3. Metrics collection
4. Report generation

Usage:
    python -m bondx.quality.run --data-root data/ --policy bondx/quality/config/quality_policy.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from .validators import DataValidator
from .quality_gates import QualityGateManager
from .metrics import MetricsCollector

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('quality_run.log')
        ]
    )

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from various formats."""
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.csv':
        return pd.read_csv(data_path)
    elif data_path.suffix.lower() == '.json':
        return pd.read_json(data_path)
    elif data_path.suffix.lower() == '.parquet':
        return pd.read_parquet(data_path)
    elif data_path.suffix.lower() == '.xlsx':
        return pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

def run_quality_pipeline(data_path: str, policy_path: str = None, 
                        output_dir: str = "quality", verbose: bool = False) -> int:
    """Run the complete quality assurance pipeline."""
    
    # Setup logging
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {data_path}")
        data = load_dataset(data_path)
        logger.info(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
        
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager(policy_path)
        metrics_collector = MetricsCollector()
        
        # Step 1: Run validators
        logger.info("Running data validators...")
        bond_validation = validator.validate_bond_data(data)
        liquidity_validation = validator.validate_liquidity_data(data)
        esg_validation = validator.validate_esg_data(data)
        completeness_validation = validator.validate_dataset_completeness(data, "synthetic_bonds")
        
        all_validation_results = (bond_validation + liquidity_validation + 
                                esg_validation + completeness_validation)
        
        # Step 2: Run quality gates
        logger.info("Running quality gates...")
        gate_results = gate_manager.run_all_gates(data, "synthetic_bonds")
        
        # Step 3: Collect metrics
        logger.info("Collecting quality metrics...")
        metrics = metrics_collector.collect_metrics(
            data=data,
            dataset_name="synthetic_bonds",
            validation_results=all_validation_results,
            quality_gates_results=gate_results,
            dataset_version="1.0"
        )
        
        # Step 4: Generate reports
        logger.info("Generating quality reports...")
        output_path = Path(output_dir) / "last_run_report.json"
        metrics_collector.export_metrics_json(str(output_path))
        
        # Step 5: Print summary
        print("\n" + "="*80)
        print("QUALITY ASSURANCE PIPELINE COMPLETED")
        print("="*80)
        
        # Validation summary
        failures = validator.get_failures()
        warnings = validator.get_warnings()
        
        print(f"\nVALIDATION RESULTS:")
        print(f"  Total checks: {len(all_validation_results)}")
        print(f"  Passed: {len(all_validation_results) - len(failures) - len(warnings)}")
        print(f"  Failures: {len(failures)}")
        print(f"  Warnings: {len(warnings)}")
        
        if failures:
            print(f"\nCRITICAL FAILURES:")
            for failure in failures:
                print(f"  ❌ {failure.rule_name}: {failure.message}")
        
        if warnings:
            print(f"\nWARNINGS:")
            for warning in warnings:
                print(f"  ⚠️  {warning.rule_name}: {warning.message}")
        
        # Quality gates summary
        gate_summary = gate_manager.get_gate_summary()
        print(f"\nQUALITY GATES:")
        print(f"  Total gates: {gate_summary['total_gates']}")
        print(f"  Passed: {gate_summary['passed_gates']}")
        print(f"  Failed: {gate_summary['failed_gates']}")
        print(f"  Pass rate: {gate_summary['pass_rate']:.1f}%")
        
        # Overall health score
        health_score = metrics_collector.get_overall_health_score()
        print(f"\nOVERALL HEALTH SCORE: {health_score}/100")
        
        # Metrics summary
        metrics_collector.print_summary_table()
        
        # Determine exit code
        if validator.has_critical_failures() or gate_manager.has_critical_failures():
            logger.error("Quality pipeline failed due to critical failures")
            return 1
        elif len(failures) > 0 or len(gate_manager.get_failed_gates()) > 0:
            logger.warning("Quality pipeline completed with warnings")
            return 0
        else:
            logger.info("Quality pipeline completed successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Quality pipeline failed: {e}")
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BondX Quality Assurance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m bondx.quality.run --data-root data/synthetic/sample_bond_dataset.csv
  
  # Run with custom policy
  python -m bondx.quality.run --data-root data/synthetic/sample_bond_dataset.csv --policy bondx/quality/config/quality_policy.yaml
  
  # Run with custom output directory
  python -m bondx.quality.run --data-root data/synthetic/sample_bond_dataset.csv --output-dir custom_output/
        """
    )
    
    parser.add_argument(
        "--data-root", "-d",
        required=True,
        help="Path to the dataset file (CSV, JSON, Parquet, or Excel)"
    )
    
    parser.add_argument(
        "--policy", "-p",
        help="Path to quality policy YAML file (optional)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="quality",
        help="Output directory for reports (default: quality/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.data_root).exists():
        print(f"Error: Data file not found: {args.data_root}")
        sys.exit(1)
    
    # Run pipeline
    exit_code = run_quality_pipeline(
        data_path=args.data_root,
        policy_path=args.policy,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
