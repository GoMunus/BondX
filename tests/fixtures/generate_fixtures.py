#!/usr/bin/env python3
"""
Generate deterministic test fixtures for BondX quality testing.

This script creates test datasets with known characteristics for comprehensive
testing of the quality assurance pipeline.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import uuid

# Set global seed for deterministic generation
np.random.seed(42)

def generate_perfect_dataset():
    """Generate a small perfect dataset with no violations."""
    data = {
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
    }
    
    df = pd.DataFrame(data)
    return df

def generate_dataset_with_known_errors():
    """Generate a dataset with known quality issues for testing."""
    data = {
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
    }
    
    df = pd.DataFrame(data)
    return df

def generate_mixed_dataset():
    """Generate a dataset with a mix of good and problematic records."""
    # Start with perfect data
    perfect_data = generate_perfect_dataset()
    
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

def generate_baseline_metrics():
    """Generate baseline metrics for drift testing."""
    baseline = {
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
    
    return baseline

def main():
    """Generate all test fixtures."""
    fixtures_dir = Path(__file__).parent
    
    print("Generating test fixtures...")
    
    # Generate perfect dataset
    perfect_df = generate_perfect_dataset()
    perfect_path = fixtures_dir / "small_perfect_dataset" / "bonds.csv"
    perfect_df.to_csv(perfect_path, index=False)
    print(f"Generated perfect dataset: {perfect_path}")
    
    # Generate dataset with known errors
    error_df = generate_dataset_with_known_errors()
    error_path = fixtures_dir / "dataset_with_known_errors" / "bonds.csv"
    error_df.to_csv(error_path, index=False)
    print(f"Generated error dataset: {error_path}")
    
    # Generate mixed dataset
    mixed_df = generate_mixed_dataset()
    mixed_path = fixtures_dir / "mixed_dataset" / "bonds.csv"
    mixed_df.to_csv(mixed_path, index=False)
    print(f"Generated mixed dataset: {mixed_path}")
    
    # Generate baseline metrics
    baseline = generate_baseline_metrics()
    baseline_path = fixtures_dir / "baselines" / "baseline_metrics.json"
    with open(baseline_path, 'w') as f:
        json.dump(baseline, f, indent=2, default=str)
    print(f"Generated baseline metrics: {baseline_path}")
    
    # Generate metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "seed": 42,
        "datasets": {
            "small_perfect_dataset": {
                "description": "Clean dataset with no quality violations",
                "records": len(perfect_df),
                "expected_validation_results": "All validations should pass"
            },
            "dataset_with_known_errors": {
                "description": "Dataset with intentional quality violations for testing",
                "records": len(error_df),
                "expected_validation_results": "Multiple FAIL severity results expected"
            },
            "mixed_dataset": {
                "description": "Combination of clean and problematic records",
                "records": len(mixed_df),
                "expected_validation_results": "Mix of PASS, WARN, and FAIL results"
            }
        }
    }
    
    metadata_path = fixtures_dir / "fixtures_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Generated metadata: {metadata_path}")
    
    print("\nAll test fixtures generated successfully!")
    print(f"Seed used: 42 (for deterministic testing)")

if __name__ == "__main__":
    main()
