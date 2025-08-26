"""
End-to-End Integration Test for BondX Quality Pipeline

Tests the complete quality assurance pipeline from data loading through
validation, quality gates, metrics collection, and report generation.
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bondx.quality.validators import DataValidator
from bondx.quality.quality_gates import QualityGateManager
from bondx.quality.metrics import MetricsCollector

class TestE2EQualityPipeline:
    """Test suite for end-to-end quality pipeline."""
    
    @pytest.fixture
    def mini_synthetic_dataset(self):
        """Create a small synthetic dataset for testing."""
        return pd.DataFrame({
            'issuer_name': [
                'SX-Test Corp A', 'SX-Test Corp B', 'SX-Test Corp C',
                'SX-Test Corp D', 'SX-Test Corp E'
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
            'esg_actual_emission_intensity': [105, 125, 85, 155, 205]
        })
    
    @pytest.fixture
    def bad_dataset_with_issues(self):
        """Create a dataset with known quality issues for testing."""
        return pd.DataFrame({
            'issuer_name': ['SX-Bad Corp A', 'SX-Bad Corp B', 'SX-Bad Corp C'],
            'issuer_id': ['ID001', 'ID001', 'ID003'],  # Duplicate ID
            'sector': ['Technology', '', 'Healthcare'],  # Empty sector
            'rating': ['INVALID', 'AA+', 'A'],  # Invalid rating
            'coupon_rate_pct': [35.0, 4.5, -2.0],  # Invalid coupon rates
            'maturity_years': [0.05, 5.0, 60.0],  # Invalid maturity
            'face_value': [0, 1000, -1000],  # Invalid face values
            'liquidity_spread_bps': [-100, 1200, 300],  # Invalid spreads
            'l2_depth_qty': [-5000, 15000, 8000],  # Negative depth
            'trades_7d': [25, -10, 20],  # Negative trades
            'time_since_last_trade_s': [3600, 1800, -1000],  # Negative time
            'liquidity_index_0_100': [75.0, 150.0, -25.0],  # Out of bounds
            'esg_target_renew_pct': [120, 70, 90],  # Invalid target
            'esg_actual_renew_pct': [75, 65, 85],
            'esg_target_emission_intensity': [-50, 120, 80],  # Invalid target
            'esg_actual_emission_intensity': [105, 125, 85]
        })
    
    def test_complete_pipeline_success(self, mini_synthetic_dataset):
        """Test complete pipeline with clean data."""
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        metrics_collector = MetricsCollector()
        
        # Step 1: Run validators
        bond_validation = validator.validate_bond_data(mini_synthetic_dataset)
        liquidity_validation = validator.validate_liquidity_data(mini_synthetic_dataset)
        esg_validation = validator.validate_esg_data(mini_synthetic_dataset)
        completeness_validation = validator.validate_dataset_completeness(mini_synthetic_dataset, "mini_synthetic")
        
        all_validation_results = (bond_validation + liquidity_validation + 
                                esg_validation + completeness_validation)
        
        # Step 2: Run quality gates
        gate_results = gate_manager.run_all_gates(mini_synthetic_dataset, "mini_synthetic")
        
        # Step 3: Collect metrics
        metrics = metrics_collector.collect_metrics(
            data=mini_synthetic_dataset,
            dataset_name="mini_synthetic",
            validation_results=all_validation_results,
            quality_gates_results=gate_results,
            dataset_version="1.0"
        )
        
        # Step 4: Verify results
        # No critical failures
        assert not validator.has_critical_failures()
        assert not gate_manager.has_critical_failures()
        
        # All validations should pass
        failures = validator.get_failures()
        assert len(failures) == 0, f"Expected no failures, got {len(failures)}"
        
        # Quality gates should pass
        failed_gates = gate_manager.get_failed_gates()
        assert len(failed_gates) == 0, f"Expected no failed gates, got {len(failed_gates)}"
        
        # Metrics should be collected
        assert metrics is not None
        assert metrics.dataset_name == "mini_synthetic"
        assert metrics.total_rows == 5
        assert metrics.total_columns == 16
        
        # Health score should be high
        health_score = metrics_collector.get_overall_health_score()
        assert health_score > 80, f"Expected health score > 80, got {health_score}"
    
    def test_pipeline_with_data_issues(self, bad_dataset_with_issues):
        """Test pipeline with known data quality issues."""
        # Initialize components
        validator = DataValidator()
        gate_manager = QualityGateManager()
        metrics_collector = MetricsCollector()
        
        # Step 1: Run validators
        bond_validation = validator.validate_bond_data(bad_dataset_with_issues)
        liquidity_validation = validator.validate_liquidity_data(bad_dataset_with_issues)
        esg_validation = validator.validate_esg_data(bad_dataset_with_issues)
        completeness_validation = validator.validate_dataset_completeness(bad_dataset_with_issues, "bad_dataset")
        
        all_validation_results = (bond_validation + liquidity_validation + 
                                esg_validation + completeness_validation)
        
        # Step 2: Run quality gates
        gate_results = gate_manager.run_all_gates(bad_dataset_with_issues, "bad_dataset")
        
        # Step 3: Collect metrics
        metrics = metrics_collector.collect_metrics(
            data=bad_dataset_with_issues,
            dataset_name="bad_dataset",
            validation_results=all_validation_results,
            quality_gates_results=gate_results,
            dataset_version="1.0"
        )
        
        # Step 4: Verify issues are caught
        # Should have critical failures
        assert validator.has_critical_failures()
        
        # Should have validation failures
        failures = validator.get_failures()
        assert len(failures) > 0, "Expected failures to be caught"
        
        # Check specific issues are caught
        failure_rules = [f.rule_name for f in failures]
        assert "unique_primary_keys" in failure_rules, "Duplicate keys should be caught"
        assert "non_empty_sectors" in failure_rules, "Empty sectors should be caught"
        assert "valid_rating_buckets" in failure_rules, "Invalid ratings should be caught"
        assert "coupon_rate_bounds" in failure_rules, "Invalid coupon rates should be caught"
        assert "maturity_bounds" in failure_rules, "Invalid maturity should be caught"
        assert "positive_face_value" in failure_rules, "Invalid face values should be caught"
        assert "no_negative_spreads" in failure_rules, "Negative spreads should be caught"
        assert "spread_bounds" in failure_rules, "Out of bounds spreads should be caught"
        
        # Health score should be low
        health_score = metrics_collector.get_overall_health_score()
        assert health_score < 50, f"Expected low health score for bad data, got {health_score}"
    
    def test_metrics_export_and_import(self, mini_synthetic_dataset):
        """Test metrics export and import functionality."""
        # Initialize components and run pipeline
        validator = DataValidator()
        gate_manager = QualityGateManager()
        metrics_collector = MetricsCollector()
        
        # Run validations and gates
        bond_validation = validator.validate_bond_data(mini_synthetic_dataset)
        gate_results = gate_manager.run_all_gates(mini_synthetic_dataset, "mini_synthetic")
        
        # Collect metrics
        metrics = metrics_collector.collect_metrics(
            data=mini_synthetic_dataset,
            dataset_name="mini_synthetic",
            validation_results=bond_validation,
            quality_gates_results=gate_results
        )
        
        # Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "test_metrics.json"
            exported_path = metrics_collector.export_metrics_json(str(export_path))
            
            assert exported_path != ""
            assert Path(exported_path).exists()
            
            # Test import by reading the file
            with open(exported_path, 'r') as f:
                exported_data = json.load(f)
            
            # Verify structure
            assert "run_summary" in exported_data
            assert "datasets" in exported_data
            assert len(exported_data["datasets"]) == 1
            
            dataset_metrics = exported_data["datasets"][0]
            assert dataset_metrics["dataset_name"] == "mini_synthetic"
            assert dataset_metrics["total_rows"] == 5
            assert dataset_metrics["total_columns"] == 16
    
    def test_policy_override_functionality(self, mini_synthetic_dataset):
        """Test that quality policy overrides work correctly."""
        # Create a custom policy with different thresholds
        custom_policy = {
            "thresholds": {
                "coverage_min": 95.0,  # Higher coverage requirement
                "esg_missing_max": 10.0,  # Stricter ESG requirement
                "liquidity_index_median_min": 50.0  # Higher liquidity requirement
            }
        }
        
        # Write custom policy to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(custom_policy, f)
            policy_path = f.name
        
        try:
            # Initialize with custom policy
            gate_manager = QualityGateManager(policy_path)
            
            # Run gates
            gate_results = gate_manager.run_all_gates(mini_synthetic_dataset, "mini_synthetic")
            
            # Check that custom thresholds are applied
            # The median liquidity index is 55.0, which should pass the 50.0 threshold
            liquidity_gate = next((g for g in gate_results if g.gate_name == "liquidity_index_gate"), None)
            assert liquidity_gate is not None
            assert liquidity_gate.passed, "Liquidity gate should pass with custom threshold"
            
        finally:
            # Clean up
            os.unlink(policy_path)
    
    def test_pipeline_reproducibility(self, mini_synthetic_dataset):
        """Test that pipeline produces consistent results."""
        # Run pipeline twice
        results1 = self._run_pipeline_once(mini_synthetic_dataset)
        results2 = self._run_pipeline_once(mini_synthetic_dataset)
        
        # Results should be identical
        assert results1["validation_failures"] == results2["validation_failures"]
        assert results1["gate_failures"] == results2["gate_failures"]
        assert results1["health_score"] == results2["health_score"]
    
    def _run_pipeline_once(self, dataset):
        """Helper method to run pipeline once and return key metrics."""
        validator = DataValidator()
        gate_manager = QualityGateManager()
        metrics_collector = MetricsCollector()
        
        # Run validations
        bond_validation = validator.validate_bond_data(dataset)
        liquidity_validation = validator.validate_liquidity_data(dataset)
        esg_validation = validator.validate_esg_data(dataset)
        completeness_validation = validator.validate_dataset_completeness(dataset, "test")
        
        all_validation_results = (bond_validation + liquidity_validation + 
                                esg_validation + completeness_validation)
        
        # Run gates
        gate_results = gate_manager.run_all_gates(dataset, "test")
        
        # Collect metrics
        metrics = metrics_collector.collect_metrics(
            data=dataset,
            dataset_name="test",
            validation_results=all_validation_results,
            quality_gates_results=gate_results
        )
        
        return {
            "validation_failures": len(validator.get_failures()),
            "gate_failures": len(gate_manager.get_failed_gates()),
            "health_score": metrics_collector.get_overall_health_score()
        }
    
    def test_error_handling(self):
        """Test pipeline handles errors gracefully."""
        # Test with empty dataset
        empty_data = pd.DataFrame()
        
        validator = DataValidator()
        gate_manager = QualityGateManager()
        metrics_collector = MetricsCollector()
        
        # Should handle empty dataset gracefully
        completeness_results = validator.validate_dataset_completeness(empty_data, "empty")
        assert len(completeness_results) == 1
        assert completeness_results[0].severity == "FAIL"
        
        # Gates should handle empty dataset
        gate_results = gate_manager.run_all_gates(empty_data, "empty")
        assert len(gate_results) > 0
        
        # Metrics should handle empty dataset
        metrics = metrics_collector.collect_metrics(
            data=empty_data,
            dataset_name="empty",
            validation_results=completeness_results,
            quality_gates_results=gate_results
        )
        assert metrics.total_rows == 0
        assert metrics.overall_coverage_pct == 0.0
