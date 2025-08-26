#!/usr/bin/env python3
"""
Simple test script for BondX synthetic dataset validation.
"""

import sys
import os

# Add current directory to path to import the generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_dataset import generate_dataset, SYNTHETIC_COMPANIES, RATINGS

def test_dataset_generation():
    """Test basic dataset generation."""
    print("Testing dataset generation...")
    
    # Generate dataset
    dataset = generate_dataset()
    
    # Basic checks
    assert len(dataset) == 260, f"Expected 260 companies, got {len(dataset)}"
    print(f"✓ Dataset size: {len(dataset)} companies")
    
    # Check all companies have SX- prefix
    for company in dataset:
        assert company['issuer_name'].startswith('SX-'), f"Company {company['issuer_name']} missing SX- prefix"
    print("✓ All companies have SX- prefix")
    
    # Check unique company names
    company_names = [company['issuer_name'] for company in dataset]
    assert len(company_names) == len(set(company_names)), "Duplicate company names found"
    print("✓ All company names are unique")
    
    # Check unique issuer IDs
    issuer_ids = [company['issuer_id'] for company in dataset]
    assert len(issuer_ids) == len(set(issuer_ids)), "Duplicate issuer IDs found"
    print("✓ All issuer IDs are unique")
    
    # Check sector distribution
    expected_total = sum(len(companies) for companies in SYNTHETIC_COMPANIES.values())
    assert len(dataset) == expected_total, f"Expected {expected_total} companies, got {len(dataset)}"
    print("✓ Sector distribution matches expected counts")
    
    # Check rating distribution
    ratings = [company['rating'] for company in dataset]
    rating_counts = {}
    for rating in ratings:
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    print("\nRating Distribution:")
    for rating in RATINGS:
        count = rating_counts.get(rating, 0)
        percentage = (count / len(dataset)) * 100
        print(f"  {rating}: {count} ({percentage:.1f}%)")
    
    # Check value ranges
    print("\nChecking value ranges...")
    
    for company in dataset:
        # Coupon rate
        assert 0 <= company['coupon_rate_pct'] <= 14.5, f"Invalid coupon rate: {company['coupon_rate_pct']}"
        
        # Maturity
        assert 0.5 <= company['maturity_years'] <= 20, f"Invalid maturity: {company['maturity_years']}"
        
        # Spread
        assert 40 <= company['liquidity_spread_bps'] <= 350, f"Invalid spread: {company['liquidity_spread_bps']}"
        
        # Liquidity metrics
        assert 0 <= company['liquidity_index_0_100'] <= 100, f"Invalid liquidity index: {company['liquidity_index_0_100']}"
        assert 0 <= company['repayment_support_0_100'] <= 100, f"Invalid repayment support: {company['repayment_support_0_100']}"
        assert 0 <= company['bondx_score_0_100'] <= 100, f"Invalid BondX score: {company['bondx_score_0_100']}"
        
        # Alt-data
        assert 0 <= company['alt_traffic_index'] <= 200, f"Invalid traffic index: {company['alt_traffic_index']}"
        assert 0 <= company['alt_utility_load'] <= 200, f"Invalid utility load: {company['alt_utility_load']}"
        assert -1 <= company['sentiment_score'] <= 1, f"Invalid sentiment score: {company['sentiment_score']}"
        assert 0 <= company['buzz_volume'] <= 1000, f"Invalid buzz volume: {company['buzz_volume']}"
        
        # ESG
        assert 20 <= company['esg_target_renew_pct'] <= 100, f"Invalid renewable target: {company['esg_target_renew_pct']}"
        assert 0 <= company['esg_actual_renew_pct'] <= 100, f"Invalid renewable actual: {company['esg_actual_renew_pct']}"
        assert 50 <= company['esg_target_emission_intensity'] <= 200, f"Invalid emission target: {company['esg_target_emission_intensity']}"
        assert 50 <= company['esg_actual_emission_intensity'] <= 250, f"Invalid emission actual: {company['esg_actual_emission_intensity']}"
    
    print("✓ All value ranges are valid")
    
    # Check cross-field coherence
    print("\nChecking cross-field coherence...")
    
    # Higher rating should generally have lower spread
    rating_spreads = {}
    for rating in RATINGS:
        rating_data = [company for company in dataset if company['rating'] == rating]
        if rating_data:
            rating_spreads[rating] = sum(company['liquidity_spread_bps'] for company in rating_data) / len(rating_data)
    
    print("Average spreads by rating:")
    for rating in RATINGS:
        if rating in rating_spreads:
            print(f"  {rating}: {rating_spreads[rating]:.1f} bps")
    
    # Check that higher ratings generally have lower spreads
    rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    for i in range(len(rating_order) - 1):
        current = rating_order[i]
        next_rating = rating_order[i + 1]
        if current in rating_spreads and next_rating in rating_spreads:
            if rating_spreads[next_rating] > rating_spreads[current]:
                print(f"  ⚠️  {next_rating} has higher average spread than {current}")
            else:
                print(f"  ✓ {next_rating} has lower average spread than {current}")
    
    print("\n✓ Basic dataset validation complete!")

def test_reproducibility():
    """Test that dataset generation is reproducible."""
    print("\nTesting reproducibility...")
    
    # Generate dataset twice
    dataset1 = generate_dataset()
    dataset2 = generate_dataset()
    
    # Check they're identical
    assert len(dataset1) == len(dataset2), "Dataset sizes differ"
    
    for i, (company1, company2) in enumerate(zip(dataset1, dataset2)):
        assert company1 == company2, f"Companies at index {i} differ: {company1['issuer_name']} vs {company2['issuer_name']}"
    
    print("✓ Dataset generation is reproducible")

if __name__ == "__main__":
    try:
        test_dataset_generation()
        test_reproducibility()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
