#!/usr/bin/env python3
"""
Simple script to show Corporate Bonds CSV Data
"""

import csv
from pathlib import Path

def show_bonds_data():
    print("🚀 BondX Corporate Bonds Data")
    print("=" * 60)
    
    # Path to the CSV file
    csv_file = Path("data/corporate_bonds.csv")
    
    if not csv_file.exists():
        print(f"❌ CSV file not found at: {csv_file}")
        return
    
    print(f"📊 Loading Corporate Bonds Data from: {csv_file}")
    
    bonds = []
    total_value = 0
    total_trades = 0
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            # Find the actual data rows (skip empty lines and headers)
            data_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('"ISIN') and not line.startswith('"columns'):
                    data_lines.append(line)
            
            print(f"Found {len(data_lines)} data lines")
            
            for line in data_lines:
                try:
                    # Parse the CSV line manually since it has complex structure
                    if line.count('","') >= 7:  # Ensure we have enough columns
                        # Split by '","' and clean up
                        parts = line.split('","')
                        if len(parts) >= 8:
                            isin = parts[0].strip('"')
                            descriptor = parts[1].strip('"')
                            wap = parts[2].strip('"')
                            ltp = parts[3].strip('"')
                            way = parts[4].strip('"')
                            lty = parts[5].strip('"')
                            value_str = parts[6].strip('"')
                            trades_str = parts[7].strip('"')
                            
                            # Skip if no ISIN or invalid data
                            if not isin or isin == "ISIN" or isin == "-":
                                continue
                            
                            # Parse values
                            try:
                                value_lakhs = float(value_str.replace(',', ''))
                                num_trades = int(trades_str)
                                last_trade_yield = float(lty)
                                last_trade_price = float(ltp)
                            except:
                                continue
                            
                            bonds.append({
                                'isin': isin,
                                'descriptor': descriptor,
                                'price': last_trade_price,
                                'yield': last_trade_yield,
                                'value_lakhs': value_lakhs,
                                'trades': num_trades
                            })
                            
                            total_value += value_lakhs
                            total_trades += num_trades
                            
                except Exception as e:
                    continue
        
        print(f"✅ Successfully loaded {len(bonds)} corporate bonds")
        
        if not bonds:
            print("❌ No valid bonds found in the CSV file")
            return
        
        # Show market summary
        print(f"\n📈 Market Summary:")
        print(f"   Total Bonds: {len(bonds):,}")
        print(f"   Total Value: ₹{total_value:,.0f} Lakhs")
        print(f"   Total Trades: {total_trades:,}")
        
        avg_yield = sum(b['yield'] for b in bonds) / len(bonds)
        print(f"   Average Yield: {avg_yield:.2f}%")
        
        # Show top bonds by volume
        print(f"\n🏆 Top 10 Bonds by Volume:")
        top_bonds = sorted(bonds, key=lambda x: x['value_lakhs'], reverse=True)[:10]
        for i, bond in enumerate(top_bonds, 1):
            issuer = bond['descriptor'][:40] if len(bond['descriptor']) > 40 else bond['descriptor']
            print(f"   {i:2d}. {issuer:<40} ₹{bond['value_lakhs']:>8,.0f}L @ {bond['yield']:>5.2f}% YTM")
        
        # Show sample bond details
        print(f"\n📋 Sample Bond Details:")
        sample_bonds = bonds[:5]
        for i, bond in enumerate(sample_bonds, 1):
            print(f"\n   Bond {i}: {bond['isin']}")
            print(f"   Description: {bond['descriptor']}")
            print(f"   Price: ₹{bond['price']:.4f}")
            print(f"   Yield: {bond['yield']:.2f}%")
            print(f"   Volume: ₹{bond['value_lakhs']:,.0f} Lakhs")
            print(f"   Trades: {bond['trades']}")
        
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
            count = len([b for b in bonds if min_yield <= b['yield'] < max_yield])
            if count > 0:
                percentage = (count / len(bonds)) * 100
                print(f"   {label}: {count} bonds ({percentage:.1f}%)")
        
        # Show bond types breakdown
        print(f"\n🏢 Bond Types:")
        ncd_count = len([b for b in bonds if 'NCD' in b['descriptor']])
        bd_count = len([b for b in bonds if 'BD' in b['descriptor']])
        other_count = len(bonds) - ncd_count - bd_count
        
        print(f"   NCD (Non-Convertible Debentures): {ncd_count} bonds")
        print(f"   BD (Bonds): {bd_count} bonds")
        print(f"   Other: {other_count} bonds")
        
        # Show issuer breakdown
        print(f"\n🏭 Top Issuers:")
        issuer_counts = {}
        for bond in bonds:
            # Extract issuer name (first part before rate/maturity)
            descriptor = bond['descriptor']
            if descriptor and descriptor != "-":
                parts = descriptor.split()
                issuer_parts = []
                for part in parts:
                    if any(char.isdigit() for char in part) and ('.' in part or '%' in part):
                        break
                    issuer_parts.append(part)
                
                if issuer_parts:
                    issuer_name = " ".join(issuer_parts).title()
                    issuer_counts[issuer_name] = issuer_counts.get(issuer_name, 0) + 1
        
        # Show top 5 issuers
        top_issuers = sorted(issuer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for issuer, count in top_issuers:
            print(f"   {issuer}: {count} bonds")
        
        print(f"\n🎉 Data Analysis Complete!")
        print(f"   Found {len(bonds)} authentic corporate bonds")
        print(f"   Total market value: ₹{total_value:,.0f} Lakhs")
        
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_bonds_data()
