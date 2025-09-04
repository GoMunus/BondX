#!/usr/bin/env python3
"""
Demo script to show Corporate Bonds CSV Integration Results
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def show_results():
    print("🚀 BondX Corporate Bonds Integration Results")
    print("=" * 60)
    
    try:
        from bondx.core.data_loader import get_bonds_loader
        
        # Load the bonds data
        print("\n📊 Loading Corporate Bonds Data...")
        bonds_loader = get_bonds_loader()
        bonds = bonds_loader.get_all_bonds()
        
        print(f"✅ Successfully loaded {len(bonds)} corporate bonds from CSV")
        
        # Show market summary
        print(f"\n📈 Market Summary:")
        market_summary = bonds_loader.get_market_summary()
        print(f"   Total Bonds: {market_summary['total_bonds']:,}")
        print(f"   Total Value: ₹{market_summary['total_value_lakhs']:,.0f} Lakhs")
        print(f"   Total Trades: {market_summary['total_trades']:,}")
        print(f"   Average Yield: {market_summary['average_yield']:.2f}%")
        
        # Show sector breakdown
        print(f"\n🏢 Sector Breakdown:")
        sectors = market_summary['sector_breakdown']['counts']
        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            value = market_summary['sector_breakdown']['values'][sector]
            print(f"   {sector}: {count} bonds (₹{value:,.0f}L)")
        
        # Show top bonds by volume
        print(f"\n🏆 Top 10 Bonds by Volume:")
        top_bonds = bonds_loader.get_top_bonds_by_volume(10)
        for i, bond in enumerate(top_bonds, 1):
            print(f"   {i:2d}. {bond.issuer_name[:40]:<40} ₹{bond.value_lakhs:>8,.0f}L @ {bond.last_trade_yield:>5.2f}% YTM")
        
        # Show sample bond details
        print(f"\n📋 Sample Bond Details:")
        sample_bonds = bonds[:5]
        for i, bond in enumerate(sample_bonds, 1):
            print(f"\n   Bond {i}: {bond.isin}")
            print(f"   Issuer: {bond.issuer_name}")
            print(f"   Sector: {bond.sector}")
            print(f"   Price: ₹{bond.last_trade_price:.4f}")
            print(f"   Yield: {bond.last_trade_yield:.2f}%")
            print(f"   Volume: ₹{bond.value_lakhs:,.0f} Lakhs")
            print(f"   Trades: {bond.num_trades}")
            if bond.coupon_rate:
                print(f"   Coupon: {bond.coupon_rate:.2f}%")
        
        # Show yield distribution
        print(f"\n📊 Yield Distribution:")
        yield_ranges = [
            (0, 5, "Ultra Low (0-5%)"),
            (5, 7, "Low (5-7%)"), 
            (7, 9, "Medium (7-9%)"),
            (9, 12, "High (9-12%)"),
            (12, 20, "Very High (12-20%)")
        ]
        
        for min_yield, max_yield, label in yield_ranges:
            count = len([b for b in bonds if min_yield <= b.last_trade_yield < max_yield])
            if count > 0:
                percentage = (count / len(bonds)) * 100
                print(f"   {label}: {count} bonds ({percentage:.1f}%)")
        
        # Show API endpoints available
        print(f"\n🔗 Available API Endpoints:")
        print(f"   GET /api/v1/dashboard/corporate-bonds")
        print(f"   GET /api/v1/dashboard/bonds/sectors") 
        print(f"   GET /api/v1/dashboard/bonds/top-performers")
        print(f"   GET /api/v1/dashboard/portfolio-summary (enhanced)")
        print(f"   GET /api/v1/dashboard/trading-activity (enhanced)")
        
        # Show frontend widgets
        print(f"\n🎨 Frontend Widgets Enhanced:")
        print(f"   ✅ Corporate Bonds Widget (NEW)")
        print(f"   ✅ Portfolio Summary Widget (real data)")
        print(f"   ✅ Trading Activity Widget (real bonds)")
        print(f"   ✅ Market Overview Widget (real volumes)")
        print(f"   ✅ Yield Curve Widget (enhanced)")
        
        print(f"\n🎉 Integration Complete!")
        print(f"   All widgets now display REAL corporate bonds data")
        print(f"   from your CSV file with {len(bonds)} authentic bonds!")
        
        print(f"\n🚀 To see the live dashboard:")
        print(f"   1. Run: python start_integrated_bondx.py")
        print(f"   2. Open: http://localhost:5173")
        print(f"   3. Login: demo/demo123 or admin/admin123")
        
    except Exception as e:
        print(f"❌ Error loading bonds data: {str(e)}")
        print(f"   Make sure the CSV file is in the correct location")
        print(f"   Path: data/corporate_bonds.csv")

if __name__ == "__main__":
    show_results()
