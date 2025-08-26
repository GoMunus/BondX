#!/usr/bin/env python3
"""
Synthetic Dataset Generator for BondX
Generates 260 synthetic issuers with realistic bond and liquidity attributes.
All data is synthetic and clearly labeled as such.
"""

import csv
import json
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Synthetic company names by sector
SYNTHETIC_COMPANIES = {
    "Financials and Banking": [
        "SX-Apex Capital Bank", "SX-BlueStone Finance", "SX-Cascade Holdings",
        "SX-Delta Trust Bank", "SX-Equinox Financial", "SX-First Meridian Credit",
        "SX-Granite Securities", "SX-Horizon Mutual", "SX-IronGate Banking Corp",
        "SX-JadeBridge Capital", "SX-Kestrel Asset Group", "SX-Lighthouse Investments",
        "SX-Midway Finance PLC", "SX-NorthArc Funds", "SX-Oakline Credit Union",
        "SX-Pillarstone Bank", "SX-Quanta Capital", "SX-Riverbend Trust",
        "SX-Stronghold Finance", "SX-Trident Securities", "SX-UnionGate Financial",
        "SX-Vertex Capital Markets", "SX-Westward Finance", "SX-Yellowtail Mutual",
        "SX-Zenith Bank Group"
    ],
    "Infrastructure, Power, Utilities": [
        "SX-Alpha Grid Transmission", "SX-BrightPeak Power", "SX-CityLink Utilities",
        "SX-DeepRiver Hydro", "SX-EdgeVolt Renewables", "SX-Flux Energy Systems",
        "SX-GreenArc Solar", "SX-Helios Wind Partners", "SX-InfraCore Roads",
        "SX-Jetstream Gas", "SX-Kiln Thermal Power", "SX-Lattice Water Works",
        "SX-MegaRail Logistics", "SX-NovaMetro Transit", "SX-Orbit Fiber Networks",
        "SX-PowerBridge T&D", "SX-Quantum SmartGrid", "SX-Resonance Utilities",
        "SX-Stoneport Ports", "SX-Turbine Peak Power", "SX-UrbanFlow Sewage",
        "SX-Valence Energy Storage", "SX-WaveLine Desalination", "SX-Xenon City Gas",
        "SX-Yardley InfraDev"
    ],
    "Industrial, Manufacturing": [
        "SX-AeroForge Systems", "SX-BetaMach Precision", "SX-Covalent Materials",
        "SX-DriveChain Motors", "SX-Enginuity Tools", "SX-ForgeLink Metals",
        "SX-GigaFab Components", "SX-Hexon Plastics", "SX-IndiSteel Works",
        "SX-Juno Robotics", "SX-Kronos Machinery", "SX-Laminar Pumps",
        "SX-MicroAlloy Foundry", "SX-Nimbus Aerospace", "SX-OrthoTech Medical Devices",
        "SX-PrimeGear Industrial", "SX-Quantum Bearings", "SX-RotorCore Turbines",
        "SX-Sintered Parts Co", "SX-TitanCast Foundries", "SX-UltraFab Electronics",
        "SX-Vector Engines", "SX-WeldRight Systems", "SX-XPress Conveyors",
        "SX-Yotta Materials", "SX-Zenufacture Labs"
    ],
    "Technology, Software, Data": [
        "SX-AltWave Software", "SX-ByteField Analytics", "SX-CypherGrid Security",
        "SX-DataHarbor Cloud", "SX-EdgeFox Networks", "SX-FusionStack Systems",
        "SX-GlowAI Platforms", "SX-HyperLoop Compute", "SX-InsightForge BI",
        "SX-Jigsaw DevTools", "SX-KiteMesh IoT", "SX-LucidSignal Data",
        "SX-Mindline AI", "SX-NexusLayer Infra", "SX-Orbita Quantum",
        "SX-PixelPeak Media", "SX-QuarkLabs R&D", "SX-RippleCode DevOps",
        "SX-Spectral Semicon", "SX-TerraNode DB", "SX-Uplink Systems",
        "SX-VoxelVision AR", "SX-Wireframe UX", "SX-XStream CDN",
        "SX-Yosei Robotics", "SX-ZettaCompute"
    ],
    "Healthcare, Pharma, Biotech": [
        "SX-ApexBio Therapeutics", "SX-BioNova Labs", "SX-CareSphere Clinics",
        "SX-Dermatek Pharma", "SX-Enviva Diagnostics", "SX-GenCrest Biologics",
        "SX-HelixPath Genomics", "SX-ImmunoCore Health", "SX-JunoLife Hospitals",
        "SX-KardioTech Devices", "SX-LumenRx Pharma", "SX-MediGrid Services",
        "SX-NeuroAxis Care", "SX-OncoPath Research", "SX-PharmaVista",
        "SX-QureWellness", "SX-Reviva Biotech", "SX-SurgiLine Tools",
        "SX-TheraLink", "SX-UroMed Devices", "SX-VitalCore Care",
        "SX-Wellfield Diagnostics", "SX-XenRx Labs", "SX-YoungLife Health"
    ],
    "Consumer, Retail, E-commerce": [
        "SX-Aurora Retail", "SX-BasketHub Commerce", "SX-CityMart Stores",
        "SX-Delight Foods", "SX-Everglo Home", "SX-FashionCircle",
        "SX-GreenGrove Organics", "SX-HomeBlend", "SX-Intown Electronics",
        "SX-JoyRide Mobility", "SX-KartNook", "SX-Luxora Beauty",
        "SX-MetroMart", "SX-NimbleWear", "SX-OpenCartel", "SX-PureNest",
        "SX-QuickBasket", "SX-RetailLoop", "SX-StyleHive", "SX-TruValue Bazaar",
        "SX-UrbanThreads", "SX-VitaFoods", "SX-Wishlane", "SX-XenoWear",
        "SX-YummyCart", "SX-ZenCart"
    ],
    "Telecom, Media": [
        "SX-AirWave Mobile", "SX-Broadcast One", "SX-ChannelPeak Media",
        "SX-DigiVerse OTT", "SX-EchoLine Telecom", "SX-FlexCast Studios",
        "SX-GigaBeam Wireless", "SX-HoloStream", "SX-InfiniTalk",
        "SX-JetCast Radio", "SX-Kinetik Media", "SX-LinkSphere",
        "SX-MediaCrest", "SX-NetPulse", "SX-OmniCast", "SX-PolyWave",
        "SX-QuantaTel", "SX-RadioNova", "SX-SkyBridge Cable", "SX-TeleCore",
        "SX-UnityFiber", "SX-ViewPort Media", "SX-WaveCell", "SX-XpressTel",
        "SX-YoMedia", "SX-ZenCast"
    ],
    "Logistics, Transportation": [
        "SX-AeroPort Cargo", "SX-BaseLine Freight", "SX-CityLift Logistics",
        "SX-DriveFleet", "SX-ExpressHaul", "SX-FleetAxis", "SX-Gateway Shipping",
        "SX-HarborLink", "SX-Inland RailCo", "SX-JetFreight", "SX-KargoChain",
        "SX-LoadRunner", "SX-MetroShip", "SX-NorthStar Freight", "SX-OceanWing",
        "SX-Portline Terminals", "SX-QuickRail", "SX-RoadBridge", "SX-SkyLift Air",
        "SX-TransNova", "SX-UrbanPorts", "SX-Vantage Logistics", "SX-WayPoint Movers",
        "SX-Xport Global", "SX-Yardline", "SX-ZoneShip"
    ],
    "Real Estate, Construction": [
        "SX-Arcadia Estates", "SX-BlueRock Developers", "SX-Crestline Realty",
        "SX-DeltaHomes", "SX-Everstone REIT", "SX-Foundry Builders",
        "SX-GrandBuild", "SX-HabitatWorks", "SX-InfraMason", "SX-Jaguar Homes",
        "SX-Keystone Constructions", "SX-Landmark Realty", "SX-MetroLiving",
        "SX-NestCore", "SX-Orbit Homes", "SX-PrimeSquare", "SX-Quantum Realty",
        "SX-Riverstone RE", "SX-StructaBuild", "SX-TerraHomes", "SX-UrbanForge",
        "SX-ValleyView", "SX-Worksite Infra", "SX-XenoBuild", "SX-YieldStone",
        "SX-ZenHabitats"
    ],
    "Agriculture, Commodities": [
        "SX-AgriLine Mills", "SX-BioHarvest", "SX-CropCircle", "SX-DeltaAgro",
        "SX-EcoFarms", "SX-FreshRoot", "SX-GrainGate", "SX-HarvestPeak",
        "SX-Irrigo Systems", "SX-JuxtaAgri", "SX-KrishiCore", "SX-Leaf&Loam",
        "SX-MandiLink", "SX-NatureNest", "SX-OrchardWorks", "SX-PrimeAg Commodities",
        "SX-QuinoaQuarry", "SX-RuralBridge", "SX-SoilSync", "SX-TerraFarm",
        "SX-UrbanSprout", "SX-VerdantCo", "SX-Wetland Produce", "SX-Xylem Agro",
        "SX-YieldNova", "SX-ZenCrop"
    ],
    "Energy Trading, Metals, Mining": [
        "SX-AlloyCore Mining", "SX-BulkMetals", "SX-CoreMinerals", "SX-DrillStone Energy"
    ]
}

# Rating buckets with weights for realistic distribution
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]
RATING_WEIGHTS = [0.05, 0.08, 0.12, 0.15, 0.20, 0.18, 0.12, 0.05, 0.03, 0.02]

# Coupon types
COUPON_TYPES = ["fixed", "floating", "zero", "amortizing"]
COUPON_TYPE_WEIGHTS = [0.60, 0.25, 0.10, 0.05]

# Face values
FACE_VALUES = [1000, 5000, 10000, 25000, 50000, 100000]

def generate_deterministic_uuid(company_name: str) -> str:
    """Generate deterministic UUID from company name using hash."""
    hash_object = hashlib.md5(company_name.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert hash to UUID format
    uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str

def get_rating_base_spread(rating: str) -> Tuple[int, int]:
    """Get base spread range for rating (in basis points)."""
    base_spreads = {
        "AAA": (40, 80),
        "AA+": (50, 90),
        "AA": (60, 120),
        "AA-": (70, 140),
        "A+": (80, 160),
        "A": (90, 180),
        "A-": (100, 200),
        "BBB+": (120, 240),
        "BBB": (140, 300),
        "BBB-": (160, 350)
    }
    return base_spreads.get(rating, (100, 200))

def generate_company_data(company_name: str, sector: str, company_index: int) -> Dict[str, Any]:
    """Generate synthetic data for a single company."""
    
    # Deterministic generation based on company index
    local_rng = random.Random(RANDOM_SEED + company_index)
    
    # Basic company info
    issuer_id = generate_deterministic_uuid(company_name)
    
    # Rating (weighted random)
    rating = local_rng.choices(RATINGS, weights=RATING_WEIGHTS)[0]
    
    # Coupon details
    coupon_type = local_rng.choices(COUPON_TYPES, weights=COUPON_TYPE_WEIGHTS)[0]
    
    if coupon_type == "zero":
        coupon_rate = 0.0
    elif coupon_type == "floating":
        coupon_rate = round(local_rng.uniform(3.0, 8.0), 2)
    else:  # fixed or amortizing
        coupon_rate = round(local_rng.uniform(3.0, 14.5), 2)
    
    # Maturity and face value
    maturity_years = round(local_rng.uniform(0.5, 20.0), 1)
    face_value = local_rng.choice(FACE_VALUES)
    
    # Liquidity metrics
    base_spread_min, base_spread_max = get_rating_base_spread(rating)
    spread_bps = local_rng.randint(base_spread_min, base_spread_max)
    
    # Add maturity premium
    if maturity_years > 10:
        spread_bps += local_rng.randint(20, 50)
    elif maturity_years > 5:
        spread_bps += local_rng.randint(10, 30)
    
    # L2 depth and trading activity
    l2_depth_qty = local_rng.randint(1000, 50000)
    trades_7d = local_rng.randint(5, 200)
    time_since_last_trade_s = local_rng.randint(0, 604800)  # 0 to 7 days
    
    # Alt-data features
    traffic_index = local_rng.randint(0, 200)
    utility_load_index = local_rng.randint(0, 200)
    sentiment_score = round(local_rng.uniform(-1.0, 1.0), 3)
    buzz_volume = local_rng.randint(0, 1000)
    
    # ESG KPIs
    renewables_target = local_rng.randint(20, 100)
    renewables_actual = max(0, renewables_target - local_rng.randint(0, 30))
    emissions_target = local_rng.randint(50, 200)
    emissions_actual = emissions_target + local_rng.randint(0, 50)
    
    # Liquidity pulse metrics
    liquidity_index = max(0, min(100, 
        (100 - spread_bps/4) + 
        (l2_depth_qty/1000) + 
        (trades_7d/2) - 
        (time_since_last_trade_s/10000)
    ))
    
    repayment_support = max(0, min(100,
        (traffic_index + utility_load_index) / 4 +
        (sentiment_score + 1) * 25 +
        (buzz_volume / 20)
    ))
    
    bondx_score = max(0, min(100,
        (liquidity_index * 0.4) +
        (repayment_support * 0.3) +
        (100 - spread_bps/3) * 0.3
    ))
    
    # Exit paths
    mm_available = liquidity_index > 50 and trades_7d > 20
    
    if liquidity_index > 70 and mm_available:
        auction_next_time = None
        rfq_window = None
    elif liquidity_index > 40:
        # Generate auction time in next 30 days
        days_ahead = local_rng.randint(1, 30)
        auction_next_time = (datetime.now() + timedelta(days=days_ahead)).isoformat()
        rfq_window = None
    else:
        auction_next_time = None
        # Generate RFQ window in next 14 days
        start_days = local_rng.randint(1, 14)
        end_days = start_days + local_rng.randint(1, 7)
        start_time = datetime.now() + timedelta(days=start_days)
        end_time = datetime.now() + timedelta(days=end_days)
        rfq_window = f"{start_time.isoformat()}/{end_time.isoformat()}"
    
    tokenized_eligible = liquidity_index < 40 or (liquidity_index < 60 and spread_bps > 200)
    
    return {
        "issuer_name": company_name,
        "issuer_id": issuer_id,
        "sector": sector,
        "rating": rating,
        "coupon_type": coupon_type,
        "coupon_rate_pct": coupon_rate,
        "maturity_years": maturity_years,
        "face_value": face_value,
        "liquidity_spread_bps": spread_bps,
        "l2_depth_qty": l2_depth_qty,
        "trades_7d": trades_7d,
        "time_since_last_trade_s": time_since_last_trade_s,
        "alt_traffic_index": traffic_index,
        "alt_utility_load": utility_load_index,
        "sentiment_score": sentiment_score,
        "buzz_volume": buzz_volume,
        "esg_target_renew_pct": renewables_target,
        "esg_actual_renew_pct": renewables_actual,
        "esg_target_emission_intensity": emissions_target,
        "esg_actual_emission_intensity": emissions_actual,
        "liquidity_index_0_100": round(liquidity_index, 1),
        "repayment_support_0_100": round(repayment_support, 1),
        "bondx_score_0_100": round(bondx_score, 1),
        "mm_available": mm_available,
        "auction_next_time": auction_next_time,
        "rfq_window": rfq_window,
        "tokenized_eligible": tokenized_eligible
    }

def generate_dataset() -> List[Dict[str, Any]]:
    """Generate the complete synthetic dataset."""
    dataset = []
    company_index = 0
    
    for sector, companies in SYNTHETIC_COMPANIES.items():
        for company_name in companies:
            company_data = generate_company_data(company_name, sector, company_index)
            dataset.append(company_data)
            company_index += 1
    
    return dataset

def save_csv(dataset: List[Dict[str, Any]], filename: str):
    """Save dataset to CSV file."""
    if not dataset:
        return
    
    fieldnames = dataset[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

def save_jsonl(dataset: List[Dict[str, Any]], filename: str):
    """Save dataset to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as jsonlfile:
        for record in dataset:
            jsonlfile.write(json.dumps(record) + '\n')

def create_readme():
    """Create README documentation for the synthetic dataset."""
    readme_content = """# Synthetic BondX Dataset

## Overview
This directory contains synthetic training data for BondX liquidity and risk features. 
**ALL DATA IS SYNTHETIC AND CLEARLY LABELED AS SUCH.**

## Files
- `bondx_issuers_260.csv` - CSV format dataset
- `bondx_issuers_260.jsonl` - JSON Lines format dataset
- `README.md` - This documentation

## Schema

### Company Information
- `issuer_name` - Synthetic company name (SX- prefix)
- `issuer_id` - Deterministic UUID hash of company name
- `sector` - Business sector classification

### Bond Characteristics
- `rating` - Credit rating (AAA to BBB-)
- `coupon_type` - fixed/floating/zero/amortizing
- `coupon_rate_pct` - Annual coupon rate (0.00-14.50%)
- `maturity_years` - Time to maturity (0.5-20 years)
- `face_value` - Bond face value

### Liquidity Metrics
- `liquidity_spread_bps` - Bid-ask spread in basis points
- `l2_depth_qty` - Level 2 market depth quantity
- `trades_7d` - Number of trades in last 7 days
- `time_since_last_trade_s` - Seconds since last trade

### Alternative Data Features
- `alt_traffic_index` - Web traffic index (0-200)
- `alt_utility_load` - Utility load index (0-200)
- `sentiment_score` - Sentiment score (-1.0 to 1.0)
- `buzz_volume` - Social media buzz volume (0-1000)

### ESG KPIs
- `esg_target_renew_pct` - Renewable energy target percentage
- `esg_actual_renew_pct` - Actual renewable energy percentage
- `esg_target_emission_intensity` - Target emissions intensity
- `esg_actual_emission_intensity` - Actual emissions intensity

### Liquidity Pulse Metrics
- `liquidity_index_0_100` - Liquidity index (0-100)
- `repayment_support_0_100` - Repayment support score (0-100)
- `bondx_score_0_100` - Composite BondX score (0-100)

### Exit Paths
- `mm_available` - Market maker availability (boolean)
- `auction_next_time` - Next auction time (ISO format or null)
- `rfq_window` - RFQ window (ISO range or null)
- `tokenized_eligible` - Tokenization eligibility (boolean)

## Generation Rules

### Rating Distribution
- Skewed toward AA/A ratings for realism
- AAA: 5%, AA+: 8%, AA: 12%, AA-: 15%
- A+: 20%, A: 18%, A-: 12%, BBB+: 5%, BBB: 3%, BBB-: 2%

### Spread Logic
- Base spreads by rating: AAA (40-80 bps) to BBB- (160-350 bps)
- Longer maturity increases spread
- Higher rating decreases spread

### Liquidity Index Calculation
- Based on spread, depth, trading activity, and time since last trade
- Higher values indicate better liquidity

### Exit Path Logic
- High liquidity (>70) + MM available → prefer market maker
- Medium liquidity (40-70) → auction/RFQ windows
- Low liquidity (<40) → often tokenization eligible

## Data Quality

### Reproducibility
- Fixed random seed (42) ensures identical output on reruns
- Deterministic UUID generation from company names
- No external dependencies or network calls

### Realism Constraints
- Values within realistic ranges for bond markets
- Cross-field coherence maintained
- Sector-appropriate characteristics

## Disclaimer
**WARNING: This is synthetic training data. None of the companies, ratings, or financial data represent real entities or actual market conditions. Use only for testing, development, and training purposes.**

## Seed Information
- Random seed: 42
- Generation timestamp: {timestamp}
- Total companies: 260
- Sectors: 11

## Usage
```python
import pandas as pd

# Load CSV
df = pd.read_csv('bondx_issuers_260.csv')

# Load JSONL
data = []
with open('bondx_issuers_260.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
```
"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    readme_content = readme_content.format(timestamp=timestamp)
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    """Main function to generate and save the synthetic dataset."""
    print("Generating synthetic BondX dataset...")
    
    # Generate dataset
    dataset = generate_dataset()
    
    print(f"Generated {len(dataset)} synthetic companies")
    
    # Save files
    csv_filename = 'bondx_issuers_260.csv'
    jsonl_filename = 'bondx_issuers_260.jsonl'
    
    save_csv(dataset, csv_filename)
    save_jsonl(dataset, jsonl_filename)
    
    print(f"Saved CSV: {csv_filename}")
    print(f"Saved JSONL: {jsonl_filename}")
    
    # Create documentation
    create_readme()
    print("Created README.md")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total companies: {len(dataset)}")
    
    # Rating distribution
    ratings = [company['rating'] for company in dataset]
    rating_counts = {}
    for rating in ratings:
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    print("\nRating Distribution:")
    for rating in RATINGS:
        count = rating_counts.get(rating, 0)
        percentage = (count / len(dataset)) * 100
        print(f"  {rating}: {count} ({percentage:.1f}%)")
    
    # Sector distribution
    sectors = [company['sector'] for company in dataset]
    sector_counts = {}
    for sector in sectors:
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    print("\nSector Distribution:")
    for sector, count in sector_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {sector}: {count} ({percentage:.1f}%)")
    
    print(f"\nDataset generation complete! Seed: {RANDOM_SEED}")

if __name__ == "__main__":
    main()
