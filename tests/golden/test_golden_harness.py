#!/usr/bin/env python3
"""
Tests for Golden Dataset Harness

Tests the validation harness to ensure:
1. Mismatches are detected correctly
2. CI fails when outputs drift
3. Baseline comparison works accurately
4. Normalization handles non-deterministic elements
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys

# Add bondx to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.golden.validate_golden import (
    normalize_for_comparison,
    compare_outputs,
    generate_summary,
    load_baseline
)

class TestGoldenHarness(unittest.TestCase):
    """Test cases for golden dataset validation harness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)
        
        # Create mock validation results
        self.mock_validation_result = Mock()
        self.mock_validation_result.severity = "FAIL"
        self.mock_validation_result.rule_name = "test_rule"
        
        # Create mock gate results
        self.mock_gate_results = {
            "coverage_gate": {"status": "PASS"},
            "esg_gate": {"status": "WARN"},
            "liquidity_gate": {"status": "FAIL"}
        }
        
        # Create mock metrics
        self.mock_metrics = {
            "coverage_percent": 95.0,
            "esg_completeness_percent": 85.0,
            "liquidity_index_median": 45.0,
            "data_freshness_minutes": 30
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_normalize_for_comparison_dict(self):
        """Test normalization of dictionary data."""
        test_data = {
            "b": "value2",
            "a": "value1",
            "timestamp": "2024-01-15 14:30:25.123456"
        }
        
        normalized = normalize_for_comparison(test_data)
        
        # Should be sorted by keys
        self.assertIn('"a": "value1"', normalized)
        self.assertIn('"b": "value2"', normalized)
        
        # Timestamp should be normalized to minute precision
        self.assertIn('"timestamp": "2024-01-15 14:30"', normalized)
    
    def test_normalize_for_comparison_list(self):
        """Test normalization of list data."""
        test_data = ["c", "a", "b"]
        
        normalized = normalize_for_comparison(test_data)
        
        # Should be sorted
        self.assertIn('"a"', normalized)
        self.assertIn('"b"', normalized)
        self.assertIn('"c"', normalized)
    
    def test_normalize_for_comparison_timestamps(self):
        """Test timestamp normalization."""
        test_data = {
            "timestamp": "2024-01-15 14:30:25.123456",
            "last_quote_time": "2024-01-15 14:30:25.789012",
            "other_field": "value"
        }
        
        normalized = normalize_for_comparison(test_data)
        
        # Timestamps should be normalized to minute precision
        self.assertIn('"timestamp": "2024-01-15 14:30"', normalized)
        self.assertIn('"last_quote_time": "2024-01-15 14:30"', normalized)
        self.assertIn('"other_field": "value"', normalized)
    
    def test_compare_outputs_match(self):
        """Test comparison when outputs match."""
        current = {
            "validation_results": [{"rule": "test", "severity": "PASS"}],
            "metrics": {"coverage": 95.0}
        }
        
        baseline = {
            "report": {
                "validation_results": [{"rule": "test", "severity": "PASS"}]
            },
            "metrics": {"coverage": 95.0}
        }
        
        matches, differences = compare_outputs(current, baseline)
        
        self.assertTrue(matches)
        self.assertEqual(differences, {})
    
    def test_compare_outputs_mismatch(self):
        """Test comparison when outputs don't match."""
        current = {
            "validation_results": [{"rule": "test", "severity": "FAIL"}],
            "metrics": {"coverage": 85.0}
        }
        
        baseline = {
            "report": {
                "validation_results": [{"rule": "test", "severity": "PASS"}]
            },
            "metrics": {"coverage": 95.0}
        }
        
        matches, differences = compare_outputs(current, baseline)
        
        self.assertFalse(matches)
        self.assertIn('validation_results', differences)
        self.assertIn('metrics', differences)
    
    def test_compare_outputs_missing_baseline(self):
        """Test comparison when baseline is missing."""
        current = {
            "validation_results": [{"rule": "test", "severity": "PASS"}],
            "metrics": {"coverage": 95.0}
        }
        
        baseline = {}
        
        matches, differences = compare_outputs(current, baseline)
        
        # Should match when no baseline to compare against
        self.assertTrue(matches)
        self.assertEqual(differences, {})
    
    def test_generate_summary(self):
        """Test summary generation."""
        # Create mock validation results
        validation_results = []
        for i in range(10):
            result = Mock()
            if i < 6:
                result.severity = "PASS"
            elif i < 8:
                result.severity = "WARN"
            else:
                result.severity = "FAIL"
            result.rule_name = f"rule_{i}"
            validation_results.append(result)
        
        summary = generate_summary(validation_results, self.mock_gate_results, self.mock_metrics)
        
        # Check summary contains expected information
        self.assertIn("Total Validations: 10", summary)
        self.assertIn("PASS: 6 (60.0%)", summary)
        self.assertIn("WARN: 2 (20.0%)", summary)
        self.assertIn("FAIL: 2 (20.0%)", summary)
        self.assertIn("Total Gates: 3", summary)
        self.assertIn("Coverage: 95.0%", summary)
    
    def test_load_baseline_success(self):
        """Test successful baseline loading."""
        # Create test baseline files
        baseline_dir = self.test_dir / "test_baseline"
        baseline_dir.mkdir()
        
        # Create report file
        report_data = {"validation_results": [{"rule": "test", "severity": "PASS"}]}
        with open(baseline_dir / "last_run_report.json", 'w') as f:
            json.dump(report_data, f)
        
        # Create metrics file
        metrics_data = {"coverage": 95.0}
        with open(baseline_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f)
        
        # Create summary file
        with open(baseline_dir / "summary.txt", 'w') as f:
            f.write("Test summary")
        
        baseline = load_baseline("test_baseline", self.test_dir)
        
        self.assertIn('report', baseline)
        self.assertIn('metrics', baseline)
        self.assertIn('summary', baseline)
        self.assertEqual(baseline['report'], report_data)
        self.assertEqual(baseline['metrics'], metrics_data)
        self.assertEqual(baseline['summary'], "Test summary")
    
    def test_load_baseline_missing(self):
        """Test baseline loading when files are missing."""
        baseline_dir = self.test_dir / "missing_baseline"
        baseline_dir.mkdir()
        
        # Only create report file
        report_data = {"validation_results": []}
        with open(baseline_dir / "last_run_report.json", 'w') as f:
            json.dump(report_data, f)
        
        baseline = load_baseline("missing_baseline", self.test_dir)
        
        self.assertIn('report', baseline)
        self.assertNotIn('metrics', baseline)
        self.assertNotIn('summary', baseline)
    
    def test_load_baseline_not_found(self):
        """Test baseline loading when directory doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_baseline("nonexistent", self.test_dir)
    
    def test_deterministic_comparison(self):
        """Test that comparison is deterministic across runs."""
        data1 = {
            "timestamp": "2024-01-15 14:30:25.123456",
            "list_data": ["c", "a", "b"],
            "nested": {"z": 3, "x": 1, "y": 2}
        }
        
        data2 = {
            "timestamp": "2024-01-15 14:30:25.789012",
            "list_data": ["b", "c", "a"],
            "nested": {"y": 2, "x": 1, "z": 3}
        }
        
        # Both should normalize to the same string
        normalized1 = normalize_for_comparison(data1)
        normalized2 = normalize_for_comparison(data2)
        
        self.assertEqual(normalized1, normalized2)
    
    def test_hash_consistency(self):
        """Test that hash generation is consistent."""
        data = {
            "validation_results": [{"rule": "test", "severity": "PASS"}],
            "metrics": {"coverage": 95.0}
        }
        
        normalized = normalize_for_comparison(data)
        
        # Hash should be consistent
        import hashlib
        hash1 = hashlib.md5(normalized.encode()).hexdigest()
        hash2 = hashlib.md5(normalized.encode()).hexdigest()
        
        self.assertEqual(hash1, hash2)
    
    def test_edge_case_empty_data(self):
        """Test handling of empty data structures."""
        empty_data = {}
        normalized = normalize_for_comparison(empty_data)
        self.assertEqual(normalized, '{}')
        
        empty_list = []
        normalized = normalize_for_comparison(empty_list)
        self.assertEqual(normalized, '[]')
    
    def test_edge_case_none_values(self):
        """Test handling of None values."""
        data_with_none = {
            "field1": "value",
            "field2": None,
            "field3": "another_value"
        }
        
        normalized = normalize_for_comparison(data_with_none)
        
        # Should handle None values gracefully
        self.assertIn('"field1": "value"', normalized)
        self.assertIn('"field2": null', normalized)
        self.assertIn('"field3": "another_value"', normalized)

class TestGoldenHarnessIntegration(unittest.TestCase):
    """Integration tests for golden harness."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)
        
        # Create golden dataset structure
        self.golden_dir = self.test_dir / "golden"
        self.golden_dir.mkdir()
        
        # Create baseline directory
        self.baseline_dir = self.golden_dir / "baselines"
        self.baseline_dir.mkdir()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_validation_flow(self):
        """Test complete validation flow from dataset to baseline."""
        # This would test the full flow when we have the actual quality pipeline
        # For now, we'll test the structure and file handling
        
        dataset_dir = self.golden_dir / "v1_test"
        dataset_dir.mkdir()
        
        # Create a simple test dataset
        test_data = pd.DataFrame({
            "record_id": ["TEST_001", "TEST_002"],
            "issuer_name": ["Test Corp", "Test Inc"],
            "sector": ["Technology", "Finance"],
            "rating": ["A", "BBB"],
            "spread_bps": [100, 200],
            "liquidity_index": [80, 70]
        })
        
        dataset_file = dataset_dir / "v1_test.csv"
        test_data.to_csv(dataset_file, index=False)
        
        # Verify dataset was created
        self.assertTrue(dataset_file.exists())
        self.assertEqual(len(test_data), 2)
        
        # Test baseline directory structure
        baseline_dataset_dir = self.baseline_dir / "v1_test"
        baseline_dataset_dir.mkdir()
        
        # Create mock baseline files
        baseline_report = {"validation_results": []}
        with open(baseline_dataset_dir / "last_run_report.json", 'w') as f:
            json.dump(baseline_report, f)
        
        baseline_metrics = {"coverage": 100.0}
        with open(baseline_dataset_dir / "metrics.json", 'w') as f:
            json.dump(baseline_metrics, f)
        
        # Verify baseline structure
        self.assertTrue((baseline_dataset_dir / "last_run_report.json").exists())
        self.assertTrue((baseline_dataset_dir / "metrics.json").exists())

if __name__ == "__main__":
    unittest.main()
