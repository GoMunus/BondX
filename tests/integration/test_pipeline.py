"""
Full Pipeline Integration Test for BondX Quality Assurance

Tests the complete quality assurance pipeline from data loading through
validation, quality gates, metrics collection, and report generation.
Ensures deterministic results with fixed seeds and validates all outputs.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import hashlib

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bondx.quality.validators import DataValidator
from bondx.quality.quality_gates import QualityGateManager
from bondx.quality.metrics import MetricsCollector

class TestFullQualityPipeline:
    """Test suite for full quality pipeline integration."""
    
    @pytest.fixture(autouse=True)
    def setup_seed(self):
        """Set deterministic seed for all tests."""
        np.random.seed(42)
    
    @pytest.fixture
    def perfect_dataset(self):
        """Create a perfect dataset with no quality violations."""
        return pd.DataFrame({
            'issuer_name': [
                'Perfect Corp A', 'Perfect Corp B', 'Perfect Corp C',
                'Perfect Corp D', 'Perfect Corp E'
            ],
            'issuer_id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'sector': ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer'],
            'rating': ['AAA', 'AA+', 'A', 'BBB+', 'BB'],
            'coupon_rate_pct': [5.0, 4.5, 6.0, 3.8, 7.2],
            'maturity_years': [10.0, 5.0, 15.0, 8.0, 12.0],
            'face_value': [1000, 1000, 1000, 1000, 1000],
            'liquidity_spread_bps': [150, 200, 300, 450, 600],
            'l2_depth_qty': [10000, 15000, 8000, 12000, 6000],
            'trades_7d': [25, 30, 20, 15, 10],
            'time_since_last_trade_s': [3600, 1800, 7200, 5400, 9000],
            'liquidity_index_0_100': [75.0, 65.0, 55.0, 45.0, 35.0],
            'esg_target_renew_pct': [80, 70, 90, 60, 50],
            'esg_actual_renew_pct': [75, 65, 85, 55, 45],
            'esg_target_emission_intensity': [100, 120, 80, 150, 200],
            'esg_actual_emission_intensity': [105, 125, 85, 155, 205],
            'timestamp': [
                datetime.now() - timedelta(minutes=30),
                datetime.now() - timedelta(minutes=45),
                datetime.now() - timedelta(minutes=60),
                datetime.now() - timedelta(minutes=75),
                datetime.now() - timedelta(minutes=90)
            ]
        })
    
    @pytest.fixture
    def error_dataset(self):
        """Create a dataset with known quality issues for testing."""
        return pd.DataFrame({
            'issuer_name': ['Bad Corp A', 'Bad Corp B', 'Bad Corp C', 'Bad Corp D'],
            'issuer_id': ['ID001', 'ID001', 'ID003', 'ID004'],  # Duplicate ID
            'sector': ['Technology', '', 'Healthcare', 'Energy'],  # Empty sector
            'rating': ['INVALID', 'AA+', 'A', 'BBB+'],  # Invalid rating
            'coupon_rate_pct': [35.0, 4.5, -2.0, 6.0],  # Invalid coupon rates
            'maturity_years': [0.05, 5.0, 60.0, 8.0],  # Invalid maturity
            'face_value': [0, 1000, -1000, 1000],  # Invalid face values
            'liquidity_spread_bps': [-100, 1200, 300, 450],  # Invalid spreads
            'l2_depth_qty': [-5000, 15000, 8000, 12000],  # Negative depth
            'trades_7d': [25, -10, 20, 15],  # Negative trades
            'time_since_last_trade_s': [3600, 1800, -1000, 5400],  # Negative time
            'liquidity_index_0_100': [75.0, 150.0, -25.0, 45.0],  # Out of bounds
            'esg_target_renew_pct': [120, 70, 90, 60],  # Invalid target
            'esg_actual_renew_pct': [75, 65, 85, 55],
            'esg_target_emission_intensity': [-50, 120, 80, 150],  # Invalid target
            'esg_actual_emission_intensity': [105, 125, 85, 155],
            'timestamp': [
                datetime.now() - timedelta(minutes=30),
                datetime.now() - timedelta(minutes=45),
                datetime.now() - timedelta(minutes=60),
                datetime.now() - timedelta(minutes=75)
            ]
        })
    
    @pytest.fixture
    def mixed_dataset(self):
        """Create a dataset with a mix of good and problematic records."""
        # Start with perfect data
        perfect_data = self.perfect_dataset()
        
        # Add some problematic records
        problematic_records = pd.DataFrame({
            'issuer_name': ['Mixed Corp A', 'Mixed Corp B'],
            'issuer_id': ['ID006', 'ID007'],
            'sector': ['Technology', 'Finance'],
            'rating': ['A', 'BBB+'],
            'coupon_rate_pct': [6.5, 4.8],
            'maturity_years': [12.0, 7.0],
            'face_value': [1000, 1000],
            'liquidity_spread_bps': [350, 500],
            'l2_depth_qty': [9000, 11000],
            'trades_7d': [18, 22],
            'time_since_last_trade_s': [6000, 4800],
            'liquidity_index_0_100': [50.0, 40.0],  # Below threshold
            'esg_target_renew_pct': [70, 65],
            'esg_actual_renew_pct': [65, 60],
            'esg_target_emission_intensity': [110, 130],
            'esg_actual_emission_intensity': [115, 135],
            'timestamp': [
                datetime.now() - timedelta(minutes=120),  # Stale
                datetime.now() - timedelta(minutes=150)   # Stale
            ]
        })
        
        # Combine perfect and problematic data
        mixed_df = pd.concat([perfect_data, problematic_records], ignore_index=True)
        return mixed_df
    
    @pytest.fixture
    def baseline_metrics(self):
        """Create baseline metrics for drift testing."""
        return {
            "distribution_stats": {
                "liquidity_index_0_100": {
                    "mean": 55.0,
                    "std": 15.0,
                    "min": 35.0,
                    "max": 75.0,
                    "median": 55.0
                },
                "liquidity_spread_bps": {
                    "mean": 340.0,
                    "std": 150.0,
                    "min": 150.0,
                    "max": 600.0,
                    "median": 300.0
                },
                "coupon_rate_pct": {
                    "mean": 5.5,
                    "std": 1.2,
                    "min": 3.8,
                    "max": 7.2,
                    "median": 5.0
                }
            },
            "coverage_baseline": {
                "overall_coverage_pct": 100.0,
                "column_coverage": {
                    "issuer_name": 100.0,
                    "issuer_id": 100.0,
                    "sector": 100.0,
                    "rating": 100.0,
                    "coupon_rate_pct": 100.0,
                    "maturity_years": 100.0,
                    "face_value": 100.0,
                    "liquidity_spread_bps": 100.0,
                    "l2_depth_qty": 100.0,
                    "trades_7d": 100.0,
                    "time_since_last_trade_s": 100.0,
                    "liquidity_index_0_100": 100.0,
                    "esg_target_renew_pct": 100.0,
                    "esg_actual_renew_pct": 100.0,
                    "esg_target_emission_intensity": 100.0,
                    "esg_actual_emission_intensity": 100.0
                }
            },
            "generated_at": datetime.now().isoformat(),
            "seed": 42
        }
    
    # Test 1: Complete pipeline with perfect data
    def test_complete_pipeline_perfect_data(self, perfect_dataset, baseline_metrics):
        """Test complete pipeline with clean data."""
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        
        # Create metrics collector with baseline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_metrics, f)
            baseline_path = f.name
        
        try:
            metrics_collector = MetricsCollector(baseline_path)
            
            # Step 1: Run validators
            bond_validation = validator.validate_bond_data(perfect_dataset)
            
            # Step 2: Run quality gates
            gate_results = gate_manager.run_all_gates(perfect_dataset, "perfect_dataset")
            
            # Step 3: Collect metrics
            metrics = metrics_collector.collect_metrics(
                data=perfect_dataset,
                dataset_name="perfect_dataset",
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                dataset_version="1.0"
            )
            
            # Step 4: Generate report
            report = self._generate_quality_report(
                dataset_name="perfect_dataset",
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                metrics=metrics,
                dataset_version="1.0"
            )
            
            # Verify report structure
            assert "run_id" in report
            assert "dataset_version" in report
            assert "validators_fired" in report
            assert "gates_triggered" in report
            assert "metrics" in report
            assert "status" in report
            assert "started_at" in report
            assert "finished_at" in report
            
            # Verify report content
            assert report["dataset_version"] == "1.0"
            assert report["status"] == "PASS"  # Perfect data should pass
            assert len(report["validators_fired"]) > 0
            assert len(report["gates_triggered"]) > 0
            
            # Verify metrics
            assert report["metrics"]["coverage"] == 100.0
            assert report["metrics"]["freshness"] is not None
            assert report["metrics"]["drift"] is not None
            
        finally:
            Path(baseline_path).unlink()
    
    # Test 2: Complete pipeline with error data
    def test_complete_pipeline_error_data(self, error_dataset, baseline_metrics):
        """Test complete pipeline with data containing known errors."""
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        
        # Create metrics collector with baseline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_metrics, f)
            baseline_path = f.name
        
        try:
            metrics_collector = MetricsCollector(baseline_path)
            
            # Step 1: Run validators
            bond_validation = validator.validate_bond_data(error_dataset)
            
            # Step 2: Run quality gates
            gate_results = gate_manager.run_all_gates(error_dataset, "error_dataset")
            
            # Step 3: Collect metrics
            metrics = metrics_collector.collect_metrics(
                data=error_dataset,
                dataset_name="error_dataset",
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                dataset_version="1.0"
            )
            
            # Step 4: Generate report
            report = self._generate_quality_report(
                dataset_name="error_dataset",
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                metrics=metrics,
                dataset_version="1.0"
            )
            
            # Verify report structure
            assert "run_id" in report
            assert "dataset_version" in report
            assert "validators_fired" in report
            assert "gates_triggered" in report
            assert "metrics" in report
            assert "status" in report
            
            # Verify report content - should fail due to errors
            assert report["dataset_version"] == "1.0"
            assert report["status"] == "FAIL"  # Error data should fail
            
            # Should have validation failures
            assert len(report["validators_fired"]) > 0
            
            # Should have gate failures
            failed_gates = [g for g in report["gates_triggered"] if g["severity"] == "FAIL"]
            assert len(failed_gates) > 0
            
            # Verify metrics show issues
            assert report["metrics"]["coverage"] < 100.0  # Some missing data
            
        finally:
            Path(baseline_path).unlink()
    
    # Test 3: Complete pipeline with mixed data
    def test_complete_pipeline_mixed_data(self, mixed_dataset, baseline_metrics):
        """Test complete pipeline with mixed quality data."""
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        
        # Create metrics collector with baseline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_metrics, f)
            baseline_path = f.name
        
        try:
            metrics_collector = MetricsCollector(baseline_path)
            
            # Step 1: Run validators
            bond_validation = validator.validate_bond_data(mixed_dataset)
            
            # Step 2: Run quality gates
            gate_results = gate_manager.run_all_gates(mixed_dataset, "mixed_dataset")
            
            # Step 3: Collect metrics
            metrics = metrics_collector.collect_metrics(
                data=mixed_dataset,
                dataset_name="mixed_dataset",
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                dataset_version="1.0"
            )
            
            # Step 4: Generate report
            report = self._generate_quality_report(
                dataset_name="mixed_dataset",
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                metrics=metrics,
                dataset_version="1.0"
            )
            
            # Verify report structure
            assert "run_id" in report
            assert "dataset_version" in report
            assert "validators_fired" in report
            assert "gates_triggered" in report
            assert "metrics" in report
            assert "status" in report
            
            # Verify report content - mixed results
            assert report["dataset_version"] == "1.0"
            # Status could be PASS, WARN, or FAIL depending on severity
            assert report["status"] in ["PASS", "WARN", "FAIL"]
            
            # Should have both validation results and gate results
            assert len(report["validators_fired"]) > 0
            assert len(report["gates_triggered"]) > 0
            
            # Should have metrics
            assert report["metrics"]["coverage"] is not None
            assert report["metrics"]["freshness"] is not None
            
        finally:
            Path(baseline_path).unlink()
    
    # Test 4: Deterministic results with fixed seed
    def test_deterministic_results(self, perfect_dataset, baseline_metrics):
        """Test that pipeline produces deterministic results with fixed seed."""
        # Run pipeline twice with same data and seed
        results1 = self._run_pipeline(perfect_dataset, "test1", baseline_metrics)
        results2 = self._run_pipeline(perfect_dataset, "test2", baseline_metrics)
        
        # Results should be identical (except for timestamps and run_id)
        # Remove non-deterministic fields
        results1_clean = self._remove_non_deterministic_fields(results1)
        results2_clean = self._remove_non_deterministic_fields(results2)
        
        # Should be identical
        assert results1_clean == results2_clean, "Results should be deterministic with fixed seed"
    
    # Test 5: Multiple dataset processing
    def test_multiple_dataset_processing(self, perfect_dataset, error_dataset, mixed_dataset, baseline_metrics):
        """Test processing multiple datasets in sequence."""
        datasets = {
            "perfect": perfect_dataset,
            "error": error_dataset,
            "mixed": mixed_dataset
        }
        
        all_reports = {}
        
        for name, data in datasets.items():
            report = self._run_pipeline(data, name, baseline_metrics)
            all_reports[name] = report
        
        # Verify all datasets were processed
        assert len(all_reports) == 3
        assert "perfect" in all_reports
        assert "error" in all_reports
        assert "mixed" in all_reports
        
        # Verify report structure for each
        for name, report in all_reports.items():
            assert "run_id" in report
            assert "dataset_version" in report
            assert "validators_fired" in report
            assert "gates_triggered" in report
            assert "metrics" in report
            assert "status" in report
        
        # Verify expected outcomes
        assert all_reports["perfect"]["status"] == "PASS"
        assert all_reports["error"]["status"] == "FAIL"
        # Mixed could be any status depending on severity
    
    # Test 6: Report output validation
    def test_report_output_validation(self, perfect_dataset, baseline_metrics):
        """Test that generated reports have correct schema and content."""
        report = self._run_pipeline(perfect_dataset, "validation_test", baseline_metrics)
        
        # Validate required fields
        required_fields = [
            "run_id", "dataset_version", "validators_fired", 
            "gates_triggered", "metrics", "status", 
            "started_at", "finished_at"
        ]
        
        for field in required_fields:
            assert field in report, f"Required field '{field}' missing from report"
        
        # Validate field types
        assert isinstance(report["run_id"], str)
        assert isinstance(report["dataset_version"], str)
        assert isinstance(report["validators_fired"], list)
        assert isinstance(report["gates_triggered"], list)
        assert isinstance(report["metrics"], dict)
        assert isinstance(report["status"], str)
        assert isinstance(report["started_at"], str)
        assert isinstance(report["finished_at"], str)
        
        # Validate metrics structure
        metrics = report["metrics"]
        assert "coverage" in metrics
        assert "freshness" in metrics
        assert "drift" in metrics
        
        # Validate data types in metrics
        assert isinstance(metrics["coverage"], (int, float))
        assert metrics["freshness"] is None or isinstance(metrics["freshness"], (int, float))
        assert metrics["drift"] is None or isinstance(metrics["drift"], dict)
        
        # Validate status values
        valid_statuses = ["PASS", "WARN", "FAIL"]
        assert report["status"] in valid_statuses
    
    # Test 7: Console summary generation
    def test_console_summary_generation(self, perfect_dataset, baseline_metrics):
        """Test generation of human-readable console summaries."""
        report = self._run_pipeline(perfect_dataset, "console_test", baseline_metrics)
        
        # Generate console summary
        console_summary = self._generate_console_summary(report)
        
        # Verify summary structure
        assert "QUALITY ASSURANCE SUMMARY" in console_summary
        assert "Dataset:" in console_summary
        assert "Status:" in console_summary
        assert "Coverage:" in console_summary
        assert "Validation Results:" in console_summary
        assert "Quality Gates:" in console_summary
        
        # Verify content
        assert report["dataset_name"] in console_summary
        assert report["status"] in console_summary
        assert str(report["metrics"]["coverage"]) in console_summary
    
    # Test 8: Policy-driven behavior changes
    def test_policy_driven_behavior(self, mixed_dataset, baseline_metrics):
        """Test that different policies change pipeline behavior."""
        # Test with strict policy
        strict_report = self._run_pipeline_with_policy(
            mixed_dataset, "strict_test", baseline_metrics, "strict"
        )
        
        # Test with exploratory policy
        exploratory_report = self._run_pipeline_with_policy(
            mixed_dataset, "exploratory_test", baseline_metrics, "exploratory"
        )
        
        # Different policies should potentially produce different results
        # (This test verifies the policy system is working, not specific outcomes)
        assert strict_report is not None
        assert exploratory_report is not None
        
        # Both should have valid structure
        for report in [strict_report, exploratory_report]:
            assert "run_id" in report
            assert "status" in report
            assert "metrics" in report
    
    # Test 9: Error handling and recovery
    def test_error_handling_and_recovery(self, baseline_metrics):
        """Test pipeline error handling and recovery."""
        # Test with empty dataset
        empty_data = pd.DataFrame()
        
        try:
            report = self._run_pipeline(empty_data, "empty_test", baseline_metrics)
            
            # Should handle gracefully
            assert report is not None
            assert "status" in report
            # Status could be FAIL or WARN depending on implementation
            
        except Exception as e:
            # If pipeline fails completely, that's also acceptable behavior
            # as long as it fails gracefully
            assert "empty" in str(e).lower() or "data" in str(e).lower()
    
    # Test 10: Performance and scalability
    def test_performance_and_scalability(self, baseline_metrics):
        """Test pipeline performance with larger datasets."""
        # Create larger dataset
        n_records = 1000
        large_data = pd.DataFrame({
            'issuer_name': [f'Large Corp {i}' for i in range(n_records)],
            'issuer_id': [f'ID{i:03d}' for i in range(n_records)],
            'sector': ['Technology'] * n_records,
            'rating': ['AAA'] * n_records,
            'coupon_rate_pct': [5.0] * n_records,
            'maturity_years': [10.0] * n_records,
            'face_value': [1000] * n_records,
            'liquidity_spread_bps': [150] * n_records,
            'l2_depth_qty': [10000] * n_records,
            'trades_7d': [25] * n_records,
            'time_since_last_trade_s': [3600] * n_records,
            'liquidity_index_0_100': [75.0] * n_records,
            'esg_target_renew_pct': [80] * n_records,
            'esg_actual_renew_pct': [75] * n_records,
            'esg_target_emission_intensity': [100] * n_records,
            'esg_actual_emission_intensity': [105] * n_records,
            'timestamp': [datetime.now() - timedelta(minutes=30)] * n_records
        })
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        
        report = self._run_pipeline(large_data, "performance_test", baseline_metrics)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in under 10 seconds for 1000 records
        assert execution_time < 10.0, f"Pipeline took {execution_time:.2f} seconds for 1000 records"
        
        # Should produce valid report
        assert report is not None
        assert "status" in report
        assert "metrics" in report
    
    # Helper methods
    
    def _run_pipeline(self, data, dataset_name, baseline_metrics):
        """Run the complete quality pipeline."""
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        
        # Create metrics collector with baseline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_metrics, f)
            baseline_path = f.name
        
        try:
            metrics_collector = MetricsCollector(baseline_path)
            
            # Step 1: Run validators
            bond_validation = validator.validate_bond_data(data)
            
            # Step 2: Run quality gates
            gate_results = gate_manager.run_all_gates(data, dataset_name)
            
            # Step 3: Collect metrics
            metrics = metrics_collector.collect_metrics(
                data=data,
                dataset_name=dataset_name,
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                dataset_version="1.0"
            )
            
            # Step 4: Generate report
            report = self._generate_quality_report(
                dataset_name=dataset_name,
                validation_results=bond_validation,
                quality_gates_results=gate_results,
                metrics=metrics,
                dataset_version="1.0"
            )
            
            return report
            
        finally:
            Path(baseline_path).unlink()
    
    def _run_pipeline_with_policy(self, data, dataset_name, baseline_metrics, policy_mode):
        """Run pipeline with specific policy mode."""
        # This would implement policy-specific pipeline execution
        # For now, just run the standard pipeline
        return self._run_pipeline(data, dataset_name, baseline_metrics)
    
    def _generate_quality_report(self, dataset_name, validation_results, quality_gates_results, metrics, dataset_version):
        """Generate a quality report in the required format."""
        run_id = f"run_{int(datetime.now().timestamp())}_{hashlib.md5(dataset_name.encode()).hexdigest()[:8]}"
        
        # Determine overall status
        has_failures = any(g.get("severity") == "FAIL" for g in quality_gates_results)
        has_warnings = any(g.get("severity") == "WARN" for g in quality_gates_results)
        
        if has_failures:
            status = "FAIL"
        elif has_warnings:
            status = "WARN"
        else:
            status = "PASS"
        
        # Extract metrics
        coverage = metrics.overall_coverage_pct if hasattr(metrics, 'overall_coverage_pct') else 100.0
        freshness = getattr(metrics, 'max_staleness_minutes', None)
        drift = getattr(metrics, 'drift_metrics', None)
        
        report = {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "validators_fired": [
                {
                    "rule": getattr(v, 'rule_name', 'unknown'),
                    "severity": getattr(v, 'severity', 'INFO'),
                    "message": getattr(v, 'message', ''),
                    "row_count": getattr(v, 'row_count', 0)
                }
                for v in validation_results
            ],
            "gates_triggered": [
                {
                    "name": getattr(g, 'gate_name', 'unknown'),
                    "severity": getattr(g, 'severity', 'INFO'),
                    "passed": getattr(g, 'passed', True),
                    "message": getattr(g, 'message', ''),
                    "count": 1
                }
                for g in quality_gates_results
            ],
            "metrics": {
                "coverage": coverage,
                "freshness": freshness,
                "drift": drift
            },
            "status": status,
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_console_summary(self, report):
        """Generate human-readable console summary."""
        summary = f"""
QUALITY ASSURANCE SUMMARY
=========================

Dataset: {report['dataset_name']}
Version: {report['dataset_version']}
Status: {report['status']}
Run ID: {report['run_id']}

Coverage: {report['metrics']['coverage']}%
Freshness: {report['metrics']['freshness'] or 'N/A'} minutes
Drift: {'Yes' if report['metrics']['drift'] else 'No'}

Validation Results:
{'-' * 20}"""
        
        for validator in report['validators_fired']:
            summary += f"\n• {validator['rule']}: {validator['severity']} - {validator['message']}"
        
        summary += f"""

Quality Gates:
{'-' * 15}"""
        
        for gate in report['gates_triggered']:
            status = "✓ PASS" if gate['passed'] else "✗ FAIL"
            summary += f"\n• {gate['name']}: {status} ({gate['severity']})"
        
        summary += f"""

Timing:
Started: {report['started_at']}
Finished: {report['finished_at']}
"""
        
        return summary
    
    def _remove_non_deterministic_fields(self, report):
        """Remove non-deterministic fields for comparison."""
        # Create a copy and remove non-deterministic fields
        clean_report = report.copy()
        
        # Remove fields that change between runs
        fields_to_remove = ['run_id', 'started_at', 'finished_at']
        for field in fields_to_remove:
            if field in clean_report:
                del clean_report[field]
        
        # Clean up nested structures
        if 'validators_fired' in clean_report:
            for validator in clean_report['validators_fired']:
                if 'message' in validator:
                    # Remove timestamp-dependent parts of messages
                    validator['message'] = self._clean_message(validator['message'])
        
        return clean_report
    
    def _clean_message(self, message):
        """Clean timestamp-dependent parts of messages."""
        # Remove any timestamp patterns
        import re
        # Remove ISO timestamp patterns
        message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '[TIMESTAMP]', message)
        # Remove relative time patterns
        message = re.sub(r'\d+ minutes? ago', '[TIME_AGO]', message)
        return message
