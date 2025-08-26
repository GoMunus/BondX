"""
Unit tests for BondX Data Validators

Tests all validation rules and edge cases for data integrity.
Comprehensive coverage of schema fields, ranges, chronology, categorical validity,
duplicates, and staleness with deterministic seeds.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from datetime import datetime, timedelta

from bondx.quality.validators import DataValidator, ValidationResult

class TestDataValidator:
    """Test suite for DataValidator class."""
    
    @pytest.fixture(autouse=True)
    def setup_seed(self):
        """Set deterministic seed for all tests."""
        np.random.seed(42)
    
    @pytest.fixture
    def validator(self):
        """Create a fresh validator instance for each test."""
        return DataValidator()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample bond data for testing."""
        return pd.DataFrame({
            'issuer_name': ['Test Corp A', 'Test Corp B', 'Test Corp C'],
            'issuer_id': ['ID001', 'ID002', 'ID003'],
            'sector': ['Technology', 'Finance', 'Healthcare'],
            'rating': ['AAA', 'AA+', 'A'],
            'coupon_rate_pct': [5.0, 4.5, 6.0],
            'maturity_years': [10.0, 5.0, 15.0],
            'face_value': [1000, 1000, 1000],
            'liquidity_spread_bps': [150, 200, 300],
            'l2_depth_qty': [10000, 15000, 8000],
            'trades_7d': [25, 30, 20],
            'time_since_last_trade_s': [3600, 1800, 7200],
            'liquidity_index_0_100': [75.0, 65.0, 55.0],
            'esg_target_renew_pct': [80, 70, 90],
            'esg_actual_renew_pct': [75, 65, 85],
            'esg_target_emission_intensity': [100, 120, 80],
            'esg_actual_emission_intensity': [105, 125, 85],
            'timestamp': [
                datetime.now() - timedelta(minutes=30),
                datetime.now() - timedelta(minutes=45),
                datetime.now() - timedelta(minutes=60)
            ]
        })
    
    # Test 1: Critical field presence validation
    def test_validate_bond_data_success(self, validator, sample_data):
        """Test successful bond data validation."""
        results = validator.validate_bond_data(sample_data)
        
        # Should have no failures
        failures = [r for r in results if r.severity == "FAIL"]
        assert len(failures) == 0, f"Expected no failures, got {len(failures)}"
        
        # Should have some validation results
        assert len(results) > 0
    
    # Test 2: Missing critical fields
    def test_validate_bond_data_missing_critical_fields(self, validator):
        """Test validation fails when critical fields are missing."""
        incomplete_data = pd.DataFrame({
            'issuer_name': ['Test Corp'],
            'issuer_id': ['ID001']
            # Missing critical fields
        })
        
        results = validator.validate_bond_data(incomplete_data)
        
        # Should fail immediately due to missing critical fields
        assert len(results) > 0
        assert any(r.rule_name == "critical_fields_present" for r in results)
    
    # Test 3: Negative spreads validation
    def test_validate_bond_data_negative_spreads(self, validator, sample_data):
        """Test validation catches negative spreads."""
        # Introduce negative spreads
        sample_data.loc[0, 'liquidity_spread_bps'] = -50
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch negative spreads
        negative_spread_results = [r for r in results if r.rule_name == "no_negative_spreads"]
        assert len(negative_spread_results) == 1
        assert negative_spread_results[0].severity == "FAIL"
        assert negative_spread_results[0].row_count == 1
    
    # Test 4: Invalid coupon rates
    def test_validate_bond_data_invalid_coupon_rates(self, validator, sample_data):
        """Test validation catches invalid coupon rates."""
        # Introduce invalid coupon rates
        sample_data.loc[0, 'coupon_rate_pct'] = 35.0  # Above 30%
        sample_data.loc[1, 'coupon_rate_pct'] = -2.0   # Below 0%
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid coupon rates
        invalid_coupon_results = [r for r in results if r.rule_name == "coupon_rate_bounds"]
        assert len(invalid_coupon_results) == 1
        assert invalid_coupon_results[0].severity == "FAIL"
        assert invalid_coupon_results[0].row_count == 2
    
    # Test 5: Invalid maturity values
    def test_validate_bond_data_invalid_maturity(self, validator, sample_data):
        """Test validation catches invalid maturity values."""
        # Introduce invalid maturity values
        sample_data.loc[0, 'maturity_years'] = 0.05  # Below 0.1
        sample_data.loc[1, 'maturity_years'] = 60.0  # Above 50
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid maturity values
        invalid_maturity_results = [r for r in results if r.rule_name == "maturity_bounds"]
        assert len(invalid_maturity_results) == 1
        assert invalid_maturity_results[0].severity == "FAIL"
        assert invalid_maturity_results[0].row_count == 2
    
    # Test 6: Invalid face values
    def test_validate_bond_data_invalid_face_values(self, validator, sample_data):
        """Test validation catches invalid face values."""
        # Introduce invalid face values
        sample_data.loc[0, 'face_value'] = 0
        sample_data.loc[1, 'face_value'] = -1000
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid face values
        invalid_face_value_results = [r for r in results if r.rule_name == "face_value_positive"]
        assert len(invalid_face_value_results) == 1
        assert invalid_face_value_results[0].severity == "FAIL"
        assert invalid_face_value_results[0].row_count == 2
    
    # Test 7: Invalid liquidity index bounds
    def test_validate_bond_data_invalid_liquidity_index(self, validator, sample_data):
        """Test validation catches invalid liquidity index values."""
        # Introduce invalid liquidity index values
        sample_data.loc[0, 'liquidity_index_0_100'] = 150.0  # Above 100
        sample_data.loc[1, 'liquidity_index_0_100'] = -25.0  # Below 0
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid liquidity index values
        invalid_liquidity_results = [r for r in results if r.rule_name == "liquidity_index_bounds"]
        assert len(invalid_liquidity_results) == 1
        assert invalid_liquidity_results[0].severity == "FAIL"
        assert invalid_liquidity_results[0].row_count == 2
    
    # Test 8: Invalid L2 depth quantities
    def test_validate_bond_data_invalid_l2_depth(self, validator, sample_data):
        """Test validation catches invalid L2 depth quantities."""
        # Introduce invalid L2 depth values
        sample_data.loc[0, 'l2_depth_qty'] = -5000
        sample_data.loc[1, 'l2_depth_qty'] = 0
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid L2 depth values
        invalid_depth_results = [r for r in results if r.rule_name == "l2_depth_positive"]
        assert len(invalid_depth_results) == 1
        assert invalid_depth_results[0].severity == "FAIL"
        assert invalid_depth_results[0].row_count == 2
    
    # Test 9: Invalid trade counts
    def test_validate_bond_data_invalid_trade_counts(self, validator, sample_data):
        """Test validation catches invalid trade counts."""
        # Introduce invalid trade counts
        sample_data.loc[0, 'trades_7d'] = -10
        sample_data.loc[1, 'trades_7d'] = 0
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid trade counts
        invalid_trade_results = [r for r in results if r.rule_name == "trades_7d_non_negative"]
        assert len(invalid_trade_results) == 1
        assert invalid_trade_results[0].severity == "FAIL"
        assert invalid_trade_results[0].row_count == 2
    
    # Test 10: Invalid time since last trade
    def test_validate_bond_data_invalid_time_since_trade(self, validator, sample_data):
        """Test validation catches invalid time since last trade."""
        # Introduce invalid time values
        sample_data.loc[0, 'time_since_last_trade_s'] = -1000
        sample_data.loc[1, 'time_since_last_trade_s'] = 0
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid time values
        invalid_time_results = [r for r in results if r.rule_name == "time_since_trade_non_negative"]
        assert len(invalid_time_results) == 1
        assert invalid_time_results[0].severity == "FAIL"
        assert invalid_time_results[0].row_count == 2
    
    # Test 11: ESG target percentage bounds
    def test_validate_bond_data_invalid_esg_targets(self, validator, sample_data):
        """Test validation catches invalid ESG target percentages."""
        # Introduce invalid ESG target values
        sample_data.loc[0, 'esg_target_renew_pct'] = 120  # Above 100
        sample_data.loc[1, 'esg_target_renew_pct'] = -50  # Below 0
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid ESG target values
        invalid_esg_results = [r for r in results if r.rule_name == "esg_target_bounds"]
        assert len(invalid_esg_results) == 1
        assert invalid_esg_results[0].severity == "FAIL"
        assert invalid_esg_results[0].row_count == 2
    
    # Test 12: ESG actual percentage bounds
    def test_validate_bond_data_invalid_esg_actuals(self, validator, sample_data):
        """Test validation catches invalid ESG actual percentages."""
        # Introduce invalid ESG actual values
        sample_data.loc[0, 'esg_actual_renew_pct'] = 150  # Above 100
        sample_data.loc[1, 'esg_actual_renew_pct'] = -25  # Below 0
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid ESG actual values
        invalid_esg_results = [r for r in results if r.rule_name == "esg_actual_bounds"]
        assert len(invalid_esg_results) == 1
        assert invalid_esg_results[0].severity == "FAIL"
        assert invalid_esg_results[0].row_count == 2
    
    # Test 13: ESG emission intensity bounds
    def test_validate_bond_data_invalid_emission_intensity(self, validator, sample_data):
        """Test validation catches invalid emission intensity values."""
        # Introduce invalid emission intensity values
        sample_data.loc[0, 'esg_target_emission_intensity'] = -50  # Below 0
        sample_data.loc[1, 'esg_actual_emission_intensity'] = 0  # At 0 (edge case)
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid emission intensity values
        invalid_emission_results = [r for r in results if r.rule_name == "emission_intensity_bounds"]
        assert len(invalid_emission_results) == 1
        assert invalid_emission_results[0].severity == "FAIL"
        assert invalid_emission_results[0].row_count == 1  # Only negative value should fail
    
    # Test 14: Rating validation
    def test_validate_bond_data_invalid_ratings(self, validator, sample_data):
        """Test validation catches invalid rating values."""
        # Introduce invalid rating values
        sample_data.loc[0, 'rating'] = 'INVALID'
        sample_data.loc[1, 'rating'] = 'ZZZ'
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid rating values
        invalid_rating_results = [r for r in results if r.rule_name == "rating_valid"]
        assert len(invalid_rating_results) == 1
        assert invalid_rating_results[0].severity == "FAIL"
        assert invalid_rating_results[0].row_count == 2
    
    # Test 15: Sector validation
    def test_validate_bond_data_invalid_sectors(self, validator, sample_data):
        """Test validation catches invalid sector values."""
        # Introduce invalid sector values
        sample_data.loc[0, 'sector'] = ''  # Empty sector
        sample_data.loc[1, 'sector'] = 'INVALID_SECTOR'
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid sector values
        invalid_sector_results = [r for r in results if r.rule_name == "sector_valid"]
        assert len(invalid_sector_results) == 1
        assert invalid_sector_results[0].severity == "FAIL"
        assert invalid_sector_results[0].row_count == 2
    
    # Test 16: Duplicate issuer ID validation
    def test_validate_bond_data_duplicate_issuer_ids(self, validator, sample_data):
        """Test validation catches duplicate issuer IDs."""
        # Introduce duplicate issuer ID
        sample_data.loc[1, 'issuer_id'] = 'ID001'  # Duplicate of first record
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch duplicate issuer IDs
        duplicate_id_results = [r for r in results if r.rule_name == "no_duplicate_issuer_ids"]
        assert len(duplicate_id_results) == 1
        assert duplicate_id_results[0].severity == "FAIL"
        assert duplicate_id_results[0].row_count == 2
    
    # Test 17: Chronological validation (maturity > issue date)
    def test_validate_bond_data_maturity_chronology(self, validator, sample_data):
        """Test validation catches maturity date before issue date."""
        # Add issue date column and set invalid chronology
        sample_data['issue_date'] = [
            datetime.now() - timedelta(days=365*5),   # 5 years ago
            datetime.now() - timedelta(days=365*10),  # 10 years ago
            datetime.now() - timedelta(days=365*2)    # 2 years ago
        ]
        
        # Set maturity to be before issue date for one record
        sample_data.loc[0, 'maturity_years'] = 3.0  # Matures in 3 years, but issued 5 years ago
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch chronological issues
        chronology_results = [r for r in results if r.rule_name == "maturity_after_issue"]
        assert len(chronology_results) == 1
        assert chronology_results[0].severity == "FAIL"
        assert chronology_results[0].row_count == 1
    
    # Test 18: Bid-ask spread validation
    def test_validate_bond_data_bid_ask_spread(self, validator, sample_data):
        """Test validation catches invalid bid-ask spreads."""
        # Add bid and ask columns
        sample_data['bid_price'] = [99.5, 98.0, 97.5]
        sample_data['ask_price'] = [99.0, 98.5, 97.0]  # Ask < Bid for first record
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch invalid bid-ask spreads
        spread_results = [r for r in results if r.rule_name == "bid_less_than_ask"]
        assert len(spread_results) == 1
        assert spread_results[0].severity == "FAIL"
        assert spread_results[0].row_count == 1
    
    # Test 19: Data staleness validation
    def test_validate_bond_data_staleness(self, validator, sample_data):
        """Test validation catches stale data."""
        # Set very old timestamps
        old_timestamp = datetime.now() - timedelta(days=30)  # 30 days old
        sample_data.loc[0, 'timestamp'] = old_timestamp
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch stale data
        staleness_results = [r for r in results if r.rule_name == "data_freshness"]
        assert len(staleness_results) == 1
        assert staleness_results[0].severity == "WARN"
        assert staleness_results[0].row_count == 1
    
    # Test 20: Comprehensive validation with multiple issues
    def test_validate_bond_data_multiple_issues(self, validator, sample_data):
        """Test validation catches multiple types of issues in one dataset."""
        # Introduce multiple issues
        sample_data.loc[0, 'liquidity_spread_bps'] = -100  # Negative spread
        sample_data.loc[1, 'coupon_rate_pct'] = 35.0       # Invalid coupon
        sample_data.loc[2, 'maturity_years'] = 0.05        # Invalid maturity
        
        results = validator.validate_bond_data(sample_data)
        
        # Should catch all issues
        failures = [r for r in results if r.severity == "FAIL"]
        assert len(failures) >= 3, f"Expected at least 3 failures, got {len(failures)}"
        
        # Check specific rule violations
        rule_names = [r.rule_name for r in failures]
        assert "no_negative_spreads" in rule_names
        assert "coupon_rate_bounds" in rule_names
        assert "maturity_bounds" in rule_names
    
    # Test 21: Edge case validation - boundary values
    def test_validate_bond_data_boundary_values(self, validator, sample_data):
        """Test validation handles boundary values correctly."""
        # Set boundary values
        sample_data.loc[0, 'coupon_rate_pct'] = 0.0      # Minimum valid
        sample_data.loc[1, 'coupon_rate_pct'] = 30.0     # Maximum valid
        sample_data.loc[2, 'maturity_years'] = 0.1       # Minimum valid
        sample_data.loc[0, 'maturity_years'] = 50.0      # Maximum valid
        
        results = validator.validate_bond_data(sample_data)
        
        # Boundary values should pass validation
        failures = [r for r in results if r.severity == "FAIL"]
        # Should not fail due to boundary values
        assert not any(r.rule_name in ["coupon_rate_bounds", "maturity_bounds"] for r in failures)
    
    # Test 22: Empty dataset handling
    def test_validate_bond_data_empty_dataset(self, validator):
        """Test validation handles empty dataset gracefully."""
        empty_data = pd.DataFrame()
        
        results = validator.validate_bond_data(empty_data)
        
        # Should handle empty dataset appropriately
        assert len(results) > 0
        # Should indicate missing critical fields
        assert any(r.rule_name == "critical_fields_present" for r in results)
    
    # Test 23: Single record validation
    def test_validate_bond_data_single_record(self, validator):
        """Test validation works correctly with single record."""
        single_record = pd.DataFrame({
            'issuer_name': ['Single Corp'],
            'issuer_id': ['ID001'],
            'sector': ['Technology'],
            'rating': ['AAA'],
            'coupon_rate_pct': [5.0],
            'maturity_years': [10.0],
            'face_value': [1000],
            'liquidity_spread_bps': [150],
            'l2_depth_qty': [10000],
            'trades_7d': [25],
            'time_since_last_trade_s': [3600],
            'liquidity_index_0_100': [75.0],
            'esg_target_renew_pct': [80],
            'esg_actual_renew_pct': [75],
            'esg_target_emission_intensity': [100],
            'esg_actual_emission_intensity': [105]
        })
        
        results = validator.validate_bond_data(single_record)
        
        # Should validate single record successfully
        failures = [r for r in results if r.severity == "FAIL"]
        assert len(failures) == 0, f"Expected no failures for valid single record, got {len(failures)}"
    
    # Test 24: Large dataset validation performance
    def test_validate_bond_data_large_dataset(self, validator):
        """Test validation performance with larger dataset."""
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
            'esg_actual_emission_intensity': [105] * n_records
        })
        
        # Introduce one issue to test detection
        large_data.loc[500, 'liquidity_spread_bps'] = -50
        
        results = validator.validate_bond_data(large_data)
        
        # Should catch the single issue
        failures = [r for r in results if r.severity == "FAIL"]
        assert len(failures) >= 1, "Should catch at least one failure in large dataset"
        
        # Should identify the specific issue
        negative_spread_results = [r for r in results if r.rule_name == "no_negative_spreads"]
        assert len(negative_spread_results) == 1
        assert negative_spread_results[0].row_count == 1
