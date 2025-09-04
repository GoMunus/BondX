#!/usr/bin/env python3
"""
Test script for Corporate Bonds CSV Integration

This script tests the complete integration of corporate bonds data
from CSV file into the BondX backend and frontend system.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bondx.core.data_loader import get_bonds_loader, refresh_bonds_data
from bondx.core.logging import get_logger

logger = get_logger(__name__)

async def test_csv_data_loading():
    """Test CSV data loading functionality."""
    print("\n🔍 Testing CSV Data Loading...")
    
    try:
        # Get the bonds loader
        bonds_loader = get_bonds_loader()
        bonds = bonds_loader.get_all_bonds()
        
        print(f"✅ Successfully loaded {len(bonds)} corporate bonds from CSV")
        
        # Test market summary
        market_summary = bonds_loader.get_market_summary()
        print(f"✅ Market Summary:")
        print(f"   Total Bonds: {market_summary['total_bonds']}")
        print(f"   Total Value: ₹{market_summary['total_value_lakhs']:,.2f} Lakhs")
        print(f"   Total Trades: {market_summary['total_trades']:,}")
        print(f"   Average Yield: {market_summary['average_yield']:.2f}%")
        
        # Test sector breakdown
        sectors = market_summary['sector_breakdown']['counts']
        print(f"✅ Sector Breakdown: {list(sectors.keys())}")
        
        # Test top bonds by volume
        top_bonds = bonds_loader.get_top_bonds_by_volume(5)
        print(f"✅ Top 5 Bonds by Volume:")
        for i, bond in enumerate(top_bonds, 1):
            print(f"   {i}. {bond.issuer_name}: ₹{bond.value_lakhs:,.2f}L @ {bond.last_trade_yield:.2f}% YTM")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV Data Loading Failed: {str(e)}")
        return False

async def test_api_endpoints():
    """Test API endpoints using real data."""
    print("\n🔍 Testing API Endpoints...")
    
    try:
        # Test direct data access (simulating API calls)
        bonds_loader = get_bonds_loader()
        
        # Test corporate bonds endpoint
        bonds = bonds_loader.get_all_bonds()[:10]  # Limit for testing
        print(f"✅ Corporate Bonds Endpoint: {len(bonds)} bonds retrieved")
        
        # Test sector filtering
        financial_bonds = bonds_loader.get_bonds_by_sector("Financial Services")
        print(f"✅ Sector Filtering: {len(financial_bonds)} Financial Services bonds")
        
        # Test yield filtering
        high_yield_bonds = bonds_loader.get_bonds_by_yield_range(8.0, 12.0)
        print(f"✅ Yield Filtering: {len(high_yield_bonds)} bonds with 8-12% yield")
        
        # Test market summary
        market_summary = bonds_loader.get_market_summary()
        print(f"✅ Market Summary Endpoint: {market_summary['total_bonds']} total bonds")
        
        return True
        
    except Exception as e:
        print(f"❌ API Endpoints Test Failed: {str(e)}")
        return False

async def test_data_quality():
    """Test data quality and parsing accuracy."""
    print("\n🔍 Testing Data Quality...")
    
    try:
        bonds_loader = get_bonds_loader()
        bonds = bonds_loader.get_all_bonds()
        
        # Check for required fields
        bonds_with_isin = [b for b in bonds if b.isin]
        bonds_with_yield = [b for b in bonds if b.last_trade_yield > 0]
        bonds_with_price = [b for b in bonds if b.last_trade_price > 0]
        bonds_with_issuer = [b for b in bonds if b.issuer_name]
        
        print(f"✅ Data Quality Check:")
        print(f"   Bonds with ISIN: {len(bonds_with_isin)}/{len(bonds)} ({100*len(bonds_with_isin)/len(bonds):.1f}%)")
        print(f"   Bonds with Yield: {len(bonds_with_yield)}/{len(bonds)} ({100*len(bonds_with_yield)/len(bonds):.1f}%)")
        print(f"   Bonds with Price: {len(bonds_with_price)}/{len(bonds)} ({100*len(bonds_with_price)/len(bonds):.1f}%)")
        print(f"   Bonds with Issuer: {len(bonds_with_issuer)}/{len(bonds)} ({100*len(bonds_with_issuer)/len(bonds):.1f}%)")
        
        # Check yield range validity
        valid_yields = [b for b in bonds if 0 < b.last_trade_yield < 50]  # Reasonable yield range
        print(f"   Valid Yields (0-50%): {len(valid_yields)}/{len(bonds)} ({100*len(valid_yields)/len(bonds):.1f}%)")
        
        # Check sector assignment
        bonds_with_sector = [b for b in bonds if b.sector != "Corporate"]
        print(f"   Bonds with Specific Sector: {len(bonds_with_sector)}/{len(bonds)} ({100*len(bonds_with_sector)/len(bonds):.1f}%)")
        
        # Sample some data
        print(f"✅ Sample Bond Data:")
        sample_bonds = bonds[:3]
        for i, bond in enumerate(sample_bonds, 1):
            print(f"   {i}. {bond.isin}: {bond.issuer_name}")
            print(f"      Sector: {bond.sector}, Yield: {bond.last_trade_yield:.2f}%")
            print(f"      Price: ₹{bond.last_trade_price:.4f}, Volume: ₹{bond.value_lakhs:.2f}L")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Quality Test Failed: {str(e)}")
        return False

async def test_frontend_data_format():
    """Test data format compatibility with frontend."""
    print("\n🔍 Testing Frontend Data Format...")
    
    try:
        bonds_loader = get_bonds_loader()
        bonds = bonds_loader.get_all_bonds()[:5]
        market_summary = bonds_loader.get_market_summary()
        
        # Simulate API response format
        api_response = {
            "timestamp": "2024-01-15T10:30:00Z",
            "total_bonds": len(bonds),
            "bonds": [],
            "market_summary": market_summary,
            "filters_applied": {
                "sector": None,
                "min_yield": None,
                "max_yield": None,
                "sort_by": "volume",
                "limit": 5
            }
        }
        
        # Format bonds for frontend
        for bond in bonds:
            bond_data = {
                "isin": bond.isin,
                "descriptor": bond.descriptor,
                "issuer_name": bond.issuer_name,
                "sector": bond.sector,
                "bond_type": bond.bond_type,
                "coupon_rate": float(bond.coupon_rate) if bond.coupon_rate else None,
                "maturity_date": bond.maturity_date.isoformat() if bond.maturity_date else None,
                "weighted_avg_price": float(bond.weighted_avg_price),
                "last_trade_price": float(bond.last_trade_price),
                "weighted_avg_yield": float(bond.weighted_avg_yield),
                "last_trade_yield": float(bond.last_trade_yield),
                "value_lakhs": float(bond.value_lakhs),
                "num_trades": bond.num_trades,
                "face_value": float(bond.face_value) if bond.face_value else None
            }
            api_response["bonds"].append(bond_data)
        
        # Verify JSON serialization
        json_data = json.dumps(api_response, indent=2, default=str)
        parsed_data = json.loads(json_data)
        
        print(f"✅ Frontend Data Format:")
        print(f"   JSON Serializable: ✓")
        print(f"   Contains {len(parsed_data['bonds'])} bond records")
        print(f"   Market summary included: ✓")
        print(f"   Sample bond keys: {list(parsed_data['bonds'][0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend Data Format Test Failed: {str(e)}")
        return False

async def test_performance():
    """Test loading and processing performance."""
    print("\n🔍 Testing Performance...")
    
    try:
        import time
        
        # Test data loading performance
        start_time = time.time()
        refresh_bonds_data()
        bonds_loader = get_bonds_loader()
        load_time = time.time() - start_time
        
        bonds = bonds_loader.get_all_bonds()
        print(f"✅ Performance Metrics:")
        print(f"   Data loading time: {load_time:.3f} seconds")
        print(f"   Bonds processed: {len(bonds)}")
        print(f"   Processing rate: {len(bonds)/load_time:.0f} bonds/second")
        
        # Test query performance
        start_time = time.time()
        top_bonds = bonds_loader.get_top_bonds_by_volume(50)
        query_time = time.time() - start_time
        print(f"   Top bonds query: {query_time:.3f} seconds")
        
        start_time = time.time()
        sector_bonds = bonds_loader.get_bonds_by_sector("Financial Services")
        sector_query_time = time.time() - start_time
        print(f"   Sector filtering: {sector_query_time:.3f} seconds")
        
        start_time = time.time()
        market_summary = bonds_loader.get_market_summary()
        summary_time = time.time() - start_time
        print(f"   Market summary: {summary_time:.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Test Failed: {str(e)}")
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting Corporate Bonds CSV Integration Test")
    print("=" * 60)
    
    tests = [
        ("CSV Data Loading", test_csv_data_loading),
        ("API Endpoints", test_api_endpoints),
        ("Data Quality", test_data_quality),
        ("Frontend Data Format", test_frontend_data_format),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        result = await test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed_tests = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("\n🎉 All tests passed! Corporate bonds integration is working correctly.")
        return True
    else:
        print(f"\n⚠️ {len(tests) - passed_tests} test(s) failed. Please review the integration.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
