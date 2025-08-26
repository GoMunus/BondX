#!/usr/bin/env python3
"""
Golden Dataset Generator for BondX Quality Assurance

Generates three curated datasets with known violations and expected outcomes:
1. v1_clean: Perfect dataset (100% PASS)
2. v1_dirty: Known violations (mixed PASS/WARN/FAIL)
3. v1_mixed: Balanced mix (edge cases and thresholds)

Usage:
    python generate_golden_datasets.py
"""

import csv
import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Synthetic company names by sector
SYNTHETIC_COMPANIES = {
    "Financials": ["SX-Apex Capital", "SX-BlueStone Finance", "SX-Cascade Holdings"],
    "Utilities": ["SX-Alpha Grid", "SX-BrightPeak Power", "SX-CityLink Utilities"],
    "Technology": ["SX-AltWave Software", "SX-ByteField Analytics", "SX-CypherGrid Security"],
    "Healthcare": ["SX-ApexBio Therapeutics", "SX-BioNova Labs", "SX-CareSphere Clinics"],
    "Consumer": ["SX-Aurora Retail", "SX-BasketHub Commerce", "SX-CityMart Stores"]
}

# Rating categories
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-"]

# Sectors
SECTORS = ["Financials", "Utilities", "Technology", "Healthcare", "Consumer"]

def generate_clean_record(record_id: int) -> Dict[str, Any]:
    """Generate a perfect record with no violations."""
    sector = random.choice(SECTORS)
    company = random.choice(SYNTHETIC_COMPANIES[sector])
    rating = random.choice(RATINGS[:8])  # Only high-quality ratings
    
    return {
        "record_id": f"GOLDEN_CLEAN_{record_id:03d}",
        "issuer_name": company,
        "sector": sector,
        "rating": rating,
        "coupon_rate": round(random.uniform(2.0, 8.0), 2),
        "maturity_date": (datetime.now() + timedelta(days=random.randint(365, 3650))).strftime("%Y-%m-%d"),
        "issue_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        "face_value": random.choice([1000, 5000, 10000, 25000, 50000]),
        "spread_bps": random.randint(50, 300),
        "liquidity_index": random.randint(60, 95),
        "esg_score": random.randint(70, 95),
        "esg_environmental": random.randint(75, 95),
        "esg_social": random.randint(70, 90),
        "esg_governance": random.randint(80, 95),
        "last_quote_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "GOLDEN_VAULT",
        "dataset_version": "v1_clean"
    }

def generate_dirty_record(record_id: int) -> Dict[str, Any]:
    """Generate a record with known violations."""
    sector = random.choice(SECTORS)
    company = random.choice(SYNTHETIC_COMPANIES[sector])
    
    # Introduce known violations based on record_id patterns
    violation_type = record_id % 20  # 20 different violation patterns
    
    if violation_type < 5:  # 5% negative spreads (FAIL)
        spread = random.randint(-200, -50)
        rating = random.choice(RATINGS[8:])  # Lower ratings
    elif violation_type < 15:  # 10% invalid ratings (FAIL)
        spread = random.randint(50, 300)
        rating = "INVALID_RATING"  # Invalid rating
    elif violation_type < 30:  # 15% bad maturity dates (FAIL)
        spread = random.randint(50, 300)
        rating = random.choice(RATINGS[:8])
        maturity_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")  # Past maturity
    elif violation_type < 50:  # 20% stale quotes (WARN)
        spread = random.randint(50, 300)
        rating = random.choice(RATINGS[:8])
        last_quote_time = (datetime.now() - timedelta(hours=random.randint(2, 24))).strftime("%Y-%m-%d %H:%M:%S")
    elif violation_type < 75:  # 25% missing ESG data (WARN)
        spread = random.randint(50, 300)
        rating = random.choice(RATINGS[:8])
        esg_score = None
        esg_environmental = None
        esg_social = None
        esg_governance = None
    else:  # Normal record
        spread = random.randint(50, 300)
        rating = random.choice(RATINGS[:8])
    
    # Set default values for non-violation cases
    if violation_type >= 30:  # Not a maturity violation
        maturity_date = (datetime.now() + timedelta(days=random.randint(365, 3650))).strftime("%Y-%m-%d")
    
    if violation_type >= 50:  # Not a stale quote violation
        last_quote_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if violation_type >= 75:  # Not an ESG violation
        esg_score = random.randint(70, 95)
        esg_environmental = random.randint(75, 95)
        esg_social = random.randint(70, 90)
        esg_governance = random.randint(80, 95)
    
    return {
        "record_id": f"GOLDEN_DIRTY_{record_id:03d}",
        "issuer_name": company,
        "sector": sector,
        "rating": rating,
        "coupon_rate": round(random.uniform(2.0, 8.0), 2),
        "maturity_date": maturity_date,
        "issue_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        "face_value": random.choice([1000, 5000, 10000, 25000, 50000]),
        "spread_bps": spread,
        "liquidity_index": random.randint(60, 95),
        "esg_score": esg_score,
        "esg_environmental": esg_environmental,
        "esg_social": esg_social,
        "esg_governance": esg_governance,
        "last_quote_time": last_quote_time,
        "data_source": "GOLDEN_VAULT",
        "dataset_version": "v1_dirty"
    }

def generate_mixed_record(record_id: int) -> Dict[str, Any]:
    """Generate a record with edge cases and threshold violations."""
    sector = random.choice(SECTORS)
    company = random.choice(SYNTHETIC_COMPANIES[sector])
    
    # Create edge cases around thresholds
    edge_type = record_id % 15  # 15 different edge case patterns
    
    if edge_type < 3:  # Borderline maturity (close to issue date)
        days_to_maturity = random.randint(30, 90)  # Very short maturity
        maturity_date = (datetime.now() + timedelta(days=days_to_maturity)).strftime("%Y-%m-%d")
    elif edge_type < 6:  # Borderline liquidity (close to threshold)
        liquidity_index = random.randint(25, 35)  # Close to 30% threshold
    elif edge_type < 9:  # Borderline ESG (close to missing threshold)
        esg_score = random.randint(65, 75)  # Close to 70% threshold
    elif edge_type < 12:  # Borderline spread (close to negative)
        spread = random.randint(0, 10)  # Very low positive spread
    else:  # Normal record
        maturity_date = (datetime.now() + timedelta(days=random.randint(365, 3650))).strftime("%Y-%m-%d")
        liquidity_index = random.randint(60, 95)
        esg_score = random.randint(70, 95)
        spread = random.randint(50, 300)
    
    # Set defaults for non-edge cases
    if edge_type >= 3:
        maturity_date = (datetime.now() + timedelta(days=random.randint(365, 3650))).strftime("%Y-%m-%d")
    
    if edge_type >= 6:
        liquidity_index = random.randint(60, 95)
    
    if edge_type >= 9:
        esg_score = random.randint(70, 95)
    
    if edge_type >= 12:
        spread = random.randint(50, 300)
    
    return {
        "record_id": f"GOLDEN_MIXED_{record_id:03d}",
        "issuer_name": company,
        "sector": sector,
        "rating": random.choice(RATINGS[:8]),
        "coupon_rate": round(random.uniform(2.0, 8.0), 2),
        "maturity_date": maturity_date,
        "issue_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        "face_value": random.choice([1000, 5000, 10000, 25000, 50000]),
        "spread_bps": spread,
        "liquidity_index": liquidity_index,
        "esg_score": esg_score,
        "esg_environmental": random.randint(70, 95),
        "esg_social": random.randint(70, 90),
        "esg_governance": random.randint(75, 95),
        "last_quote_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "GOLDEN_VAULT",
        "dataset_version": "v1_mixed"
    }

def save_dataset(data: List[Dict[str, Any]], filename: str, output_dir: Path):
    """Save dataset to CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    if not data:
        return
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Generated {len(data)} records: {filepath}")

def generate_metadata(dataset_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate metadata for the dataset."""
    # Count violations for dirty dataset
    violations = {}
    if dataset_name == "v1_dirty":
        negative_spreads = sum(1 for r in records if r.get('spread_bps', 0) < 0)
        invalid_ratings = sum(1 for r in records if r.get('rating') == "INVALID_RATING")
        past_maturity = sum(1 for r in records if r.get('maturity_date') and 
                          datetime.strptime(r['maturity_date'], "%Y-%m-%d") < datetime.now())
        stale_quotes = sum(1 for r in records if r.get('last_quote_time') and 
                          (datetime.now() - datetime.strptime(r['last_quote_time'], "%Y-%m-%d %H:%M:%S")).total_seconds() > 7200)
        missing_esg = sum(1 for r in records if r.get('esg_score') is None)
        
        violations = {
            "negative_spreads": negative_spreads,
            "invalid_ratings": invalid_ratings,
            "past_maturity": past_maturity,
            "stale_quotes": stale_quotes,
            "missing_esg": missing_esg
        }
    
    return {
        "dataset_name": dataset_name,
        "generation_timestamp": datetime.now().isoformat(),
        "record_count": len(records),
        "random_seed": RANDOM_SEED,
        "violations": violations,
        "expected_outcome": get_expected_outcome(dataset_name, violations)
    }

def get_expected_outcome(dataset_name: str, violations: Dict[str, int]) -> Dict[str, Any]:
    """Get expected validation outcomes for the dataset."""
    if dataset_name == "v1_clean":
        return {
            "pass_rate": "100%",
            "warn_rate": "0%",
            "fail_rate": "0%",
            "description": "Perfect dataset with no violations"
        }
    elif dataset_name == "v1_dirty":
        total_violations = sum(violations.values())
        total_records = 100
        pass_rate = (total_records - total_violations) / total_records * 100
        return {
            "pass_rate": f"{pass_rate:.1f}%",
            "warn_rate": "~35%",
            "fail_rate": "~20%",
            "description": "Known violations dataset for testing failure detection"
        }
    else:  # v1_mixed
        return {
            "pass_rate": "~60%",
            "warn_rate": "~25%",
            "fail_rate": "~15%",
            "description": "Mixed dataset with edge cases and thresholds"
        }

def main():
    """Generate all golden datasets."""
    print("Generating BondX Golden Datasets...")
    print(f"Using random seed: {RANDOM_SEED}")
    
    # Create output directory
    output_dir = Path("data/golden")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate v1_clean dataset (100 perfect records)
    print("\n1. Generating v1_clean dataset...")
    clean_records = [generate_clean_record(i) for i in range(100)]
    save_dataset(clean_records, "v1_clean.csv", output_dir / "v1_clean")
    
    # Generate v1_dirty dataset (100 records with known violations)
    print("\n2. Generating v1_dirty dataset...")
    dirty_records = [generate_dirty_record(i) for i in range(100)]
    save_dataset(dirty_records, "v1_dirty.csv", output_dir / "v1_dirty")
    
    # Generate v1_mixed dataset (100 records with edge cases)
    print("\n3. Generating v1_mixed dataset...")
    mixed_records = [generate_mixed_record(i) for i in range(100)]
    save_dataset(mixed_records, "v1_mixed.csv", output_dir / "v1_mixed")
    
    # Generate metadata for each dataset
    print("\n4. Generating metadata...")
    for dataset_name, records in [("v1_clean", clean_records), 
                                 ("v1_dirty", dirty_records), 
                                 ("v1_mixed", mixed_records)]:
        metadata = generate_metadata(dataset_name, records)
        metadata_file = output_dir / dataset_name / "metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  {dataset_name}: {metadata['expected_outcome']['description']}")
    
    # Generate summary
    print("\n5. Generating summary...")
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "datasets": {
            "v1_clean": {
                "file": "v1_clean/v1_clean.csv",
                "records": 100,
                "expected": "100% PASS, 0% WARN, 0% FAIL"
            },
            "v1_dirty": {
                "file": "v1_dirty/v1_dirty.csv",
                "records": 100,
                "expected": "~45% PASS, ~35% WARN, ~20% FAIL"
            },
            "v1_mixed": {
                "file": "v1_mixed/v1_mixed.csv",
                "records": 100,
                "expected": "~60% PASS, ~25% WARN, ~15% FAIL"
            }
        },
        "usage": "Run quality pipeline on these datasets and compare outputs to baselines",
        "next_steps": [
            "1. Run quality pipeline on each dataset",
            "2. Save outputs as baselines in data/golden/baselines/",
            "3. Use validate_golden.py to check for drift"
        ]
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGolden datasets generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Summary: {summary_file}")
    print("\nNext: Run quality pipeline on these datasets to create baselines")

if __name__ == "__main__":
    main()
