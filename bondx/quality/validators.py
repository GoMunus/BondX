"""
Data Validators for BondX Quality Assurance

This module provides strict schema validation and domain rules across all core datasets.
Implements enterprise-grade data integrity checks for investor/regulator use.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    rule_name: str
    dataset: str
    field: str
    message: str
    severity: str  # 'FAIL', 'WARN', 'INFO'
    row_count: int = 0
    sample_violations: List[Dict[str, Any]] = None

class DataValidator:
    """Core data validator implementing enterprise-grade quality checks."""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        
    def validate_bond_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate bond instrument data."""
        results = []
        
        # Critical field presence
        critical_fields = ['issuer_name', 'issuer_id', 'sector', 'rating', 'coupon_rate_pct', 
                         'maturity_years', 'face_value', 'liquidity_spread_bps']
        
        missing_fields = [field for field in critical_fields if field not in data.columns]
        if missing_fields:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="critical_fields_present",
                dataset="bonds",
                field=",".join(missing_fields),
                message=f"Missing critical fields: {missing_fields}",
                severity="FAIL"
            ))
            return results  # Can't proceed without critical fields
        
        # No negative yields/spreads
        negative_spreads = data[data['liquidity_spread_bps'] < 0]
        if len(negative_spreads) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="no_negative_spreads",
                dataset="bonds",
                field="liquidity_spread_bps",
                message=f"Found {len(negative_spreads)} bonds with negative spreads",
                severity="FAIL",
                row_count=len(negative_spreads),
                sample_violations=negative_spreads.head(3).to_dict('records')
            ))
        
        # Coupon rate bounds (0-30%)
        invalid_coupons = data[(data['coupon_rate_pct'] < 0) | (data['coupon_rate_pct'] > 30)]
        if len(invalid_coupons) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="coupon_rate_bounds",
                dataset="bonds",
                field="coupon_rate_pct",
                message=f"Found {len(invalid_coupons)} bonds with invalid coupon rates (0-30%)",
                severity="FAIL",
                row_count=len(invalid_coupons),
                sample_violations=invalid_coupons.head(3).to_dict('records')
            ))
        
        # Maturity years sanity check (0.1 to 50 years)
        invalid_maturity = data[(data['maturity_years'] < 0.1) | (data['maturity_years'] > 50)]
        if len(invalid_maturity) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="maturity_bounds",
                dataset="bonds",
                field="maturity_years",
                message=f"Found {len(invalid_maturity)} bonds with invalid maturity years (0.1-50)",
                severity="FAIL",
                row_count=len(invalid_maturity),
                sample_violations=invalid_maturity.head(3).to_dict('records')
            ))
        
        # Face value positive
        invalid_face_value = data[data['face_value'] <= 0]
        if len(invalid_face_value) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="positive_face_value",
                dataset="bonds",
                field="face_value",
                message=f"Found {len(invalid_face_value)} bonds with non-positive face value",
                severity="FAIL",
                row_count=len(invalid_face_value),
                sample_violations=invalid_face_value.head(3).to_dict('records')
            ))
        
        # Rating bucket validity
        valid_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 
                        'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
        invalid_ratings = data[~data['rating'].isin(valid_ratings)]
        if len(invalid_ratings) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="valid_rating_buckets",
                dataset="bonds",
                field="rating",
                message=f"Found {len(invalid_ratings)} bonds with invalid ratings",
                severity="FAIL",
                row_count=len(invalid_ratings),
                sample_violations=invalid_ratings.head(3).to_dict('records')
            ))
        
        # Sector non-empty
        empty_sectors = data[data['sector'].isna() | (data['sector'] == '')]
        if len(empty_sectors) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="non_empty_sectors",
                dataset="bonds",
                field="sector",
                message=f"Found {len(empty_sectors)} bonds with empty sectors",
                severity="FAIL",
                row_count=len(empty_sectors),
                sample_violations=empty_sectors.head(3).to_dict('records')
            ))
        
        # Duplicate primary keys
        duplicates = data[data.duplicated(subset=['issuer_id'], keep=False)]
        if len(duplicates) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="unique_primary_keys",
                dataset="bonds",
                field="issuer_id",
                message=f"Found {len(duplicates)} duplicate issuer IDs",
                severity="FAIL",
                row_count=len(duplicates),
                sample_violations=duplicates.head(3).to_dict('records')
            ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_liquidity_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate liquidity-related data."""
        results = []
        
        # Spread bounds (0-1000 bps)
        invalid_spreads = data[(data['liquidity_spread_bps'] < 0) | (data['liquidity_spread_bps'] > 1000)]
        if len(invalid_spreads) > 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="spread_bounds",
                dataset="liquidity",
                field="liquidity_spread_bps",
                message=f"Found {len(invalid_spreads)} records with invalid spreads (0-1000 bps)",
                severity="FAIL",
                row_count=len(invalid_spreads),
                sample_violations=invalid_spreads.head(3).to_dict('records')
            ))
        
        # L2 depth quantity non-negative
        if 'l2_depth_qty' in data.columns:
            invalid_depth = data[data['l2_depth_qty'] < 0]
            if len(invalid_depth) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="non_negative_l2_depth",
                    dataset="liquidity",
                    field="l2_depth_qty",
                    message=f"Found {len(invalid_depth)} records with negative L2 depth",
                    severity="FAIL",
                    row_count=len(invalid_depth),
                    sample_violations=invalid_depth.head(3).to_dict('records')
                ))
        
        # Trades 7d non-negative
        if 'trades_7d' in data.columns:
            invalid_trades = data[data['trades_7d'] < 0]
            if len(invalid_trades) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="non_negative_trades",
                    dataset="liquidity",
                    field="trades_7d",
                    message=f"Found {len(invalid_trades)} records with negative trade counts",
                    severity="FAIL",
                    row_count=len(invalid_trades),
                    sample_violations=invalid_trades.head(3).to_dict('records')
                ))
        
        # Time since last trade non-negative
        if 'time_since_last_trade_s' in data.columns:
            invalid_time = data[data['time_since_last_trade_s'] < 0]
            if len(invalid_time) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="non_negative_time_since_trade",
                    dataset="liquidity",
                    field="time_since_last_trade_s",
                    message=f"Found {len(invalid_time)} records with negative time since last trade",
                    severity="FAIL",
                    row_count=len(invalid_time),
                    sample_violations=invalid_time.head(3).to_dict('records')
                ))
        
        # Liquidity index bounds (0-100)
        if 'liquidity_index_0_100' in data.columns:
            invalid_liquidity = data[(data['liquidity_index_0_100'] < 0) | (data['liquidity_index_0_100'] > 100)]
            if len(invalid_liquidity) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="liquidity_index_bounds",
                    dataset="liquidity",
                    field="liquidity_index_0_100",
                    message=f"Found {len(invalid_liquidity)} records with invalid liquidity index (0-100)",
                    severity="FAIL",
                    row_count=len(invalid_liquidity),
                    sample_violations=invalid_liquidity.head(3).to_dict('records')
                ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_esg_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate ESG-related data."""
        results = []
        
        # ESG target ranges (0-100%)
        if 'esg_target_renew_pct' in data.columns:
            invalid_renew_target = data[(data['esg_target_renew_pct'] < 0) | (data['esg_target_renew_pct'] > 100)]
            if len(invalid_renew_target) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="renewable_target_bounds",
                    dataset="esg",
                    field="esg_target_renew_pct",
                    message=f"Found {len(invalid_renew_target)} records with invalid renewable target (0-100%)",
                    severity="FAIL",
                    row_count=len(invalid_renew_target),
                    sample_violations=invalid_renew_target.head(3).to_dict('records')
                ))
        
        if 'esg_target_emission_intensity' in data.columns:
            invalid_emission_target = data[data['esg_target_emission_intensity'] < 0]
            if len(invalid_emission_target) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="emission_target_bounds",
                    dataset="esg",
                    field="esg_target_emission_intensity",
                    message=f"Found {len(invalid_emission_target)} records with negative emission targets",
                    severity="FAIL",
                    row_count=len(invalid_emission_target),
                    sample_violations=invalid_emission_target.head(3).to_dict('records')
                ))
        
        # ESG actual vs target comparison
        if 'esg_actual_renew_pct' in data.columns and 'esg_target_renew_pct' in data.columns:
            invalid_comparison = data[data['esg_actual_renew_pct'] > data['esg_target_renew_pct'] + 10]
            if len(invalid_comparison) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    rule_name="esg_actual_vs_target",
                    dataset="esg",
                    field="esg_actual_renew_pct,esg_target_renew_pct",
                    message=f"Found {len(invalid_comparison)} records where actual renewable % exceeds target by >10%",
                    severity="WARN",
                    row_count=len(invalid_comparison),
                    sample_violations=invalid_comparison.head(3).to_dict('records')
                ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_dataset_completeness(self, data: pd.DataFrame, dataset_name: str) -> List[ValidationResult]:
        """Validate dataset completeness and coverage."""
        results = []
        
        total_rows = len(data)
        if total_rows == 0:
            results.append(ValidationResult(
                is_valid=False,
                rule_name="non_empty_dataset",
                dataset=dataset_name,
                field="all",
                message=f"Dataset {dataset_name} is empty",
                severity="FAIL"
            ))
            return results
        
        # Coverage check (non-null values)
        coverage_stats = {}
        for col in data.columns:
            non_null_count = data[col].notna().sum()
            coverage_pct = (non_null_count / total_rows) * 100
            coverage_stats[col] = coverage_pct
            
            # Warn if coverage < 90%
            if coverage_pct < 90:
                results.append(ValidationResult(
                    is_valid=True,  # Not a hard fail
                    rule_name="coverage_threshold",
                    dataset=dataset_name,
                    field=col,
                    message=f"Column {col} has {coverage_pct:.1f}% coverage (below 90% threshold)",
                    severity="WARN",
                    row_count=total_rows - non_null_count
                ))
        
        # Store coverage metrics for later use
        self.coverage_metrics = coverage_stats
        
        self.validation_results.extend(results)
        return results
    
    def get_all_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.validation_results
    
    def get_failures(self) -> List[ValidationResult]:
        """Get only failed validations."""
        return [r for r in self.validation_results if r.severity == "FAIL"]
    
    def get_warnings(self) -> List[ValidationResult]:
        """Get only warnings."""
        return [r for r in self.validation_results if r.severity == "WARN"]
    
    def has_critical_failures(self) -> bool:
        """Check if there are any critical failures."""
        return len(self.get_failures()) > 0
