#!/usr/bin/env python3
"""
Enterprise-Grade Synthetic Corporate Bond Dataset Generator for BondX AI
Generates 1000+ synthetic corporate bonds with realistic correlations, macroeconomic factors,
and stress testing scenarios. All data is synthetic and clearly labeled as such.
"""

import csv
import json
import hashlib
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Enhanced company names by sector (1000+ companies across expanded sectors)
SYNTHETIC_COMPANIES = {
    "Technology": [
        "SX-TechCorp", "SX-DataFlow", "SX-CloudSync", "SX-AINexus", "SX-CyberShield",
        "SX-QuantumCore", "SX-BlockChain", "SX-VirtualSpace", "SX-SmartGrid", "SX-DigiBank",
        "SX-MobileFirst", "SX-WebScale", "SX-DevOps", "SX-MicroServices", "SX-API Gateway",
        "SX-DataLake", "SX-MachineLearn", "SX-NeuralNet", "SX-ComputerVision", "SX-NaturalLang",
        "SX-Robotics", "SX-Automation", "SX-IoT Platform", "SX-EdgeCompute", "SX-5G Network",
        "SX-AltWave Software", "SX-ByteField Analytics", "SX-CypherGrid Security", "SX-DataHarbor Cloud",
        "SX-EdgeFox Networks", "SX-FusionStack Systems", "SX-GlowAI Platforms", "SX-HyperLoop Compute",
        "SX-InsightForge BI", "SX-Jigsaw DevTools", "SX-KiteMesh IoT", "SX-LucidSignal Data",
        "SX-Mindline AI", "SX-NexusLayer Infra", "SX-Orbita Quantum", "SX-PixelPeak Media",
        "SX-QuarkLabs R&D", "SX-RippleCode DevOps", "SX-Spectral Semicon", "SX-TerraNode DB",
        "SX-Uplink Systems", "SX-VoxelVision AR", "SX-Wireframe UX", "SX-XStream CDN",
        "SX-Yosei Robotics", "SX-ZettaCompute", "SX-QuantumAI", "SX-NeuralBridge", "SX-CryptoCore"
    ],
    "Finance": [
        "SX-GlobalBank", "SX-InvestmentCorp", "SX-CreditUnion", "SX-AssetManagement", "SX-InsuranceCo",
        "SX-PensionFund", "SX-HedgeFund", "SX-PrivateEquity", "SX-VentureCapital", "SX-RealEstateTrust",
        "SX-CommercialBank", "SX-RetailBank", "SX-CorporateBank", "SX-TradingDesk", "SX-ClearingHouse",
        "SX-CustodyBank", "SX-PrimeBroker", "SX-InvestmentBank", "SX-MerchantBank", "SX-SavingsBank",
        "SX-CreditCard", "SX-PaymentProcessor", "SX-Fintech", "SX-CryptoExchange", "SX-Tokenization",
        "SX-Apex Capital Bank", "SX-BlueStone Finance", "SX-Cascade Holdings", "SX-Delta Trust Bank",
        "SX-Equinox Financial", "SX-First Meridian Credit", "SX-Granite Securities", "SX-Horizon Mutual",
        "SX-IronGate Banking Corp", "SX-JadeBridge Capital", "SX-Kestrel Asset Group", "SX-Lighthouse Investments",
        "SX-Midway Finance PLC", "SX-NorthArc Funds", "SX-Oakline Credit Union", "SX-Pillarstone Bank",
        "SX-Quanta Capital", "SX-Riverbend Trust", "SX-Stronghold Finance", "SX-Trident Securities",
        "SX-UnionGate Financial", "SX-Vertex Capital Markets", "SX-Westward Finance", "SX-Yellowtail Mutual",
        "SX-Zenith Bank Group", "SX-DigitalBank", "SX-CryptoFund", "SX-DeFi Protocol", "SX-TokenVault"
    ],
    "Energy": [
        "SX-SolarPower", "SX-WindEnergy", "SX-NuclearCorp", "SX-HydroElectric", "SX-Geothermal",
        "SX-BioFuel", "SX-OilRefinery", "SX-NaturalGas", "SX-CoalMining", "SX-EnergyStorage",
        "SX-GridOperator", "SX-Transmission", "SX-Distribution", "SX-EnergyTrading", "SX-CarbonCapture",
        "SX-HydrogenFuel", "SX-FusionPower", "SX-TidalEnergy", "SX-WavePower", "SX-SmartMeter",
        "SX-Alpha Grid Transmission", "SX-BrightPeak Power", "SX-CityLink Utilities", "SX-DeepRiver Hydro",
        "SX-EdgeVolt Renewables", "SX-Flux Energy Systems", "SX-GreenArc Solar", "SX-Helios Wind Partners",
        "SX-InfraCore Roads", "SX-Jetstream Gas", "SX-Kiln Thermal Power", "SX-Lattice Water Works",
        "SX-MegaRail Logistics", "SX-NovaMetro Transit", "SX-Orbit Fiber Networks", "SX-PowerBridge T&D",
        "SX-Quantum SmartGrid", "SX-Resonance Utilities", "SX-Stoneport Ports", "SX-Turbine Peak Power",
        "SX-UrbanFlow Sewage", "SX-Valence Energy Storage", "SX-WaveLine Desalination", "SX-Xenon City Gas",
        "SX-Yardley InfraDev", "SX-CleanEnergy", "SX-GreenGrid", "SX-SolarFarm", "SX-WindTurbine"
    ],
    "Industrial": [
        "SX-HeavyMachinery", "SX-Aerospace", "SX-Automotive", "SX-Shipbuilding", "SX-Construction",
        "SX-MiningEquipment", "SX-SteelMill", "SX-ChemicalPlant", "SX-Pharmaceutical", "SX-FoodProcessing",
        "SX-TextileMill", "SX-PaperMill", "SX-GlassFactory", "SX-CementPlant", "SX-PlasticFactory",
        "SX-Electronics", "SX-Semiconductor", "SX-TelecomEquipment", "SX-MedicalDevice", "SX-Defense",
        "SX-AeroForge Systems", "SX-BetaMach Precision", "SX-Covalent Materials", "SX-DriveChain Motors",
        "SX-Enginuity Tools", "SX-ForgeLink Metals", "SX-GigaFab Components", "SX-Hexon Plastics",
        "SX-IndiSteel Works", "SX-Juno Robotics", "SX-Kronos Machinery", "SX-Laminar Pumps",
        "SX-MicroAlloy Foundry", "SX-Nimbus Aerospace", "SX-OrthoTech Medical Devices", "SX-PrimeGear Industrial",
        "SX-Quantum Bearings", "SX-RotorCore Turbines", "SX-Sintered Parts Co", "SX-TitanCast Foundries",
        "SX-UltraFab Electronics", "SX-Vector Engines", "SX-WeldRight Systems", "SX-XPress Conveyors",
        "SX-Yotta Materials", "SX-Zenufacture Labs", "SX-IndustrialAI", "SX-SmartFactory", "SX-RobotAssembly"
    ],
    "Consumer Goods": [
        "SX-RetailChain", "SX-FoodBrand", "SX-BeverageCo", "SX-ClothingBrand", "SX-ElectronicsRetail",
        "SX-HomeImprovement", "SX-AutoDealer", "SX-PharmacyChain", "SX-DepartmentStore", "SX-Supermarket",
        "SX-FastFood", "SX-CasualDining", "SX-LuxuryBrand", "SX-SportingGoods", "SX-ToyStore",
        "SX-BookStore", "SX-JewelryStore", "SX-FurnitureStore", "SX-GardenCenter", "SX-PetStore",
        "SX-Aurora Retail", "SX-BasketHub Commerce", "SX-CityMart Stores", "SX-Delight Foods", "SX-Everglo Home",
        "SX-FashionCircle", "SX-GreenGrove Organics", "SX-HomeBlend", "SX-Intown Electronics", "SX-JoyRide Mobility",
        "SX-KartNook", "SX-Luxora Beauty", "SX-MetroMart", "SX-NimbleWear", "SX-OpenCartel", "SX-PureNest",
        "SX-QuickBasket", "SX-RetailLoop", "SX-StyleHive", "SX-TruValue Bazaar", "SX-UrbanThreads",
        "SX-VitaFoods", "SX-Wishlane", "SX-XenoWear", "SX-YummyCart", "SX-ZenCart", "SX-DigitalRetail",
        "SX-Ecommerce", "SX-MobileCommerce", "SX-SocialCommerce", "SX-VirtualStore"
    ],
    "Healthcare": [
        "SX-HospitalChain", "SX-Pharmaceutical", "SX-Biotech", "SX-MedicalDevice", "SX-HealthInsurance",
        "SX-DiagnosticLab", "SX-ResearchInstitute", "SX-ClinicalTrial", "SX-GeneticTesting", "SX-Telemedicine",
        "SX-NursingHome", "SX-HomeHealthcare", "SX-MentalHealth", "SX-Rehabilitation", "SX-PreventiveCare",
        "SX-EmergencyServices", "SX-SpecialtyClinic", "SX-UrgentCare", "SX-PediatricCare", "SX-GeriatricCare",
        "SX-ApexBio Therapeutics", "SX-BioNova Labs", "SX-CareSphere Clinics", "SX-Dermatek Pharma",
        "SX-Enviva Diagnostics", "SX-GenCrest Biologics", "SX-HelixPath Genomics", "SX-ImmunoCore Health",
        "SX-JunoLife Hospitals", "SX-KardioTech Devices", "SX-LumenRx Pharma", "SX-MediGrid Services",
        "SX-NeuroAxis Care", "SX-OncoPath Research", "SX-PharmaVista", "SX-QureWellness", "SX-Reviva Biotech",
        "SX-SurgiLine Tools", "SX-TheraLink", "SX-UroMed Devices", "SX-VitalCore Care", "SX-Wellfield Diagnostics",
        "SX-XenRx Labs", "SX-YoungLife Health", "SX-AI Diagnostics", "SX-Genomic Medicine", "SX-Precision Health"
    ],
    "Telecom": [
        "SX-AirWave Mobile", "SX-Broadcast One", "SX-ChannelPeak Media", "SX-DigiVerse OTT", "SX-EchoLine Telecom",
        "SX-FlexCast Studios", "SX-GigaBeam Wireless", "SX-HoloStream", "SX-InfiniTalk", "SX-JetCast Radio",
        "SX-Kinetik Media", "SX-LinkSphere", "SX-MediaCrest", "SX-NetPulse", "SX-OmniCast", "SX-PolyWave",
        "SX-QuantaTel", "SX-RadioNova", "SX-SkyBridge Cable", "SX-TeleCore", "SX-UnityFiber", "SX-ViewPort Media",
        "SX-WaveCell", "SX-XpressTel", "SX-YoMedia", "SX-ZenCast", "SX-5G Networks", "SX-Fiber Optics",
        "SX-Satellite Communications", "SX-Cloud Communications", "SX-Virtual Networks"
    ],
    "Logistics": [
        "SX-AeroPort Cargo", "SX-BaseLine Freight", "SX-CityLift Logistics", "SX-DriveFleet", "SX-ExpressHaul",
        "SX-FleetAxis", "SX-Gateway Shipping", "SX-HarborLink", "SX-Inland RailCo", "SX-JetFreight",
        "SX-KargoChain", "SX-LoadRunner", "SX-MetroShip", "SX-NorthStar Freight", "SX-OceanWing",
        "SX-Portline Terminals", "SX-QuickRail", "SX-RoadBridge", "SX-SkyLift Air", "SX-TransNova",
        "SX-UrbanPorts", "SX-Vantage Logistics", "SX-WayPoint Movers", "SX-Xport Global", "SX-Yardline",
        "SX-ZoneShip", "SX-Drone Delivery", "SX-Autonomous Trucks", "SX-Smart Warehousing", "SX-Last Mile"
    ],
    "Real Estate": [
        "SX-Arcadia Estates", "SX-BlueRock Developers", "SX-Crestline Realty", "SX-DeltaHomes", "SX-Everstone REIT",
        "SX-Foundry Builders", "SX-GrandBuild", "SX-HabitatWorks", "SX-InfraMason", "SX-Jaguar Homes",
        "SX-Keystone Constructions", "SX-Landmark Realty", "SX-MetroLiving", "SX-NestCore", "SX-Orbit Homes",
        "SX-PrimeSquare", "SX-Quantum Realty", "SX-Riverstone RE", "SX-StructaBuild", "SX-TerraHomes",
        "SX-UrbanForge", "SX-ValleyView", "SX-Worksite Infra", "SX-XenoBuild", "SX-YieldStone",
        "SX-ZenHabitats", "SX-Smart Buildings", "SX-Green Construction", "SX-Prefab Housing", "SX-Urban Planning"
    ],
    "Agriculture": [
        "SX-AgriLine Mills", "SX-BioHarvest", "SX-CropCircle", "SX-DeltaAgro", "SX-EcoFarms",
        "SX-FreshRoot", "SX-GrainGate", "SX-HarvestPeak", "SX-Irrigo Systems", "SX-JuxtaAgri",
        "SX-KrishiCore", "SX-Leaf&Loam", "SX-MandiLink", "SX-NatureNest", "SX-OrchardWorks",
        "SX-PrimeAg Commodities", "SX-QuinoaQuarry", "SX-RuralBridge", "SX-SoilSync", "SX-TerraFarm",
        "SX-UrbanSprout", "SX-VerdantCo", "SX-Wetland Produce", "SX-Xylem Agro", "SX-YieldNova",
        "SX-ZenCrop", "SX-Precision Agriculture", "SX-Vertical Farming", "SX-Hydroponics", "SX-Aquaponics"
    ],
    "Mining": [
        "SX-AlloyCore Mining", "SX-BulkMetals", "SX-CoreMinerals", "SX-DrillStone Energy",
        "SX-EarthMovers", "SX-FossilFuel Corp", "SX-Gemstone Mining", "SX-HeavyMinerals",
        "SX-IronOre Corp", "SX-JewelMine", "SX-KingCoal Mining", "SX-LithiumMine", "SX-MetalCorp",
        "SX-NobleMetals", "SX-OreProcessors", "SX-PreciousMinerals", "SX-QuarryCorp", "SX-RareEarth",
        "SX-SilverMine", "SX-TitaniumCorp", "SX-UraniumMine", "SX-ValuableMetals", "SX-WaterMining",
        "SX-XenonMining", "SX-YellowGold", "SX-ZincMine", "SX-DeepSea Mining", "SX-Asteroid Mining"
    ],
    "Sovereign": [
        "SX-Republic of Atlantia", "SX-Federation of Borealis", "SX-Commonwealth of Celestia",
        "SX-Empire of Draconia", "SX-Kingdom of Elysium", "SX-Republic of Fantasia",
        "SX-Grand Duchy of Galaxia", "SX-Confederation of Hyperion", "SX-Union of Icarus",
        "SX-Alliance of Jovian", "SX-Federation of Kronos", "SX-Republic of Lyra",
        "SX-Empire of Meridian", "SX-Kingdom of Nebula", "SX-Commonwealth of Orion",
        "SX-Union of Phoenix", "SX-Confederation of Quantum", "SX-Republic of Radiant",
        "SX-Federation of Stellar", "SX-Alliance of Terra", "SX-Union of Ultima",
        "SX-Confederation of Vega", "SX-Republic of Xenon", "SX-Empire of Zenith"
    ],
    "Tech Startups": [
        "SX-StartupAlpha", "SX-StartupBeta", "SX-StartupGamma", "SX-StartupDelta", "SX-StartupEpsilon",
        "SX-StartupZeta", "SX-StartupEta", "SX-StartupTheta", "SX-StartupIota", "SX-StartupKappa",
        "SX-StartupLambda", "SX-StartupMu", "SX-StartupNu", "SX-StartupXi", "SX-StartupOmicron",
        "SX-StartupPi", "SX-StartupRho", "SX-StartupSigma", "SX-StartupTau", "SX-StartupUpsilon",
        "SX-StartupPhi", "SX-StartupChi", "SX-StartupPsi", "SX-StartupOmega", "SX-StartupNova",
        "SX-StartupPulsar", "SX-StartupQuasar", "SX-StartupNebula", "SX-StartupGalaxy", "SX-StartupCosmos"
    ],
    "Distressed Debt": [
        "SX-DistressedAlpha", "SX-DistressedBeta", "SX-DistressedGamma", "SX-DistressedDelta",
        "SX-DistressedEpsilon", "SX-DistressedZeta", "SX-DistressedEta", "SX-DistressedTheta",
        "SX-DistressedIota", "SX-DistressedKappa", "SX-DistressedLambda", "SX-DistressedMu",
        "SX-DistressedNu", "SX-DistressedXi", "SX-DistressedOmicron", "SX-DistressedPi",
        "SX-DistressedRho", "SX-DistressedSigma", "SX-DistressedTau", "SX-DistressedUpsilon",
        "SX-DistressedPhi", "SX-DistressedChi", "SX-DistressedPsi", "SX-DistressedOmega"
    ]
}

# Credit ratings with realistic weights - Enhanced for enterprise
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
RATING_WEIGHTS = [0.02, 0.03, 0.05, 0.07, 0.12, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# Sectors with risk weights (higher = higher default risk) - Enhanced for enterprise
SECTOR_RISK_WEIGHTS = {
    "Technology": 0.12,
    "Finance": 0.08,
    "Energy": 0.18,
    "Industrial": 0.16,
    "Consumer Goods": 0.10,
    "Healthcare": 0.12,
    "Telecom": 0.14,
    "Logistics": 0.13,
    "Real Estate": 0.15,
    "Agriculture": 0.11,
    "Mining": 0.20,
    "Sovereign": 0.05,
    "Tech Startups": 0.25,
    "Distressed Debt": 0.35
}

# Macroeconomic factors for stress testing
MACRO_FACTORS = {
    "interest_rate_shock": [0.5, 1.0, 1.5, 2.0, 2.5],  # Percentage point changes
    "inflation_shock": [0.2, 0.5, 1.0, 1.5, 2.0],      # Percentage point changes
    "fx_volatility": [0.1, 0.2, 0.3, 0.4, 0.5],        # Volatility multiplier
    "liquidity_freeze": [0.1, 0.2, 0.3, 0.4, 0.5],     # Liquidity reduction factor
    "economic_growth_shock": [-0.5, -1.0, -1.5, -2.0, -2.5]  # GDP growth impact
}

def generate_deterministic_uuid(company_name: str, bond_id: str) -> str:
    """Generate deterministic UUID from company name and bond ID."""
    combined = f"{company_name}_{bond_id}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert hash to UUID format
    uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str

def get_rating_base_spread(rating: str) -> Tuple[int, int]:
    """Get base spread range for rating (in basis points) - Enhanced for enterprise."""
    base_spreads = {
        "AAA": (25, 50), "AA+": (35, 60), "AA": (45, 80), "AA-": (55, 100),
        "A+": (65, 120), "A": (75, 140), "A-": (85, 160), "BBB+": (100, 180),
        "BBB": (120, 220), "BBB-": (140, 260), "BB+": (180, 350), "BB": (220, 450),
        "BB-": (280, 550), "B+": (350, 650), "B": (450, 800), "B-": (550, 950),
        "CCC+": (700, 1200), "CCC": (900, 1500), "CCC-": (1100, 1800),
        "CC": (1400, 2200), "C": (1800, 2800), "D": (2500, 4000)
    }
    return base_spreads.get(rating, (100, 200))

def calculate_yield_to_maturity(coupon_rate: float, market_price: float, face_value: float, 
                               years_to_maturity: float) -> float:
    """Calculate yield to maturity using simplified approximation."""
    if years_to_maturity <= 0:
        return coupon_rate
    
    # Simplified YTM calculation
    annual_coupon = face_value * (coupon_rate / 100)
    price_diff = face_value - market_price
    years_avg = years_to_maturity / 2
    
    ytm = (annual_coupon + (price_diff / years_to_maturity)) / ((face_value + market_price) / 2)
    return ytm * 100

def generate_macroeconomic_scenarios() -> Dict[str, float]:
    """Generate macroeconomic scenarios for stress testing."""
    scenarios = {}
    for factor, values in MACRO_FACTORS.items():
        scenarios[factor] = random.choice(values)
    return scenarios

def apply_macroeconomic_stress(bond_data: Dict[str, Any], scenarios: Dict[str, float]) -> Dict[str, Any]:
    """Apply macroeconomic stress scenarios to bond data."""
    stressed_data = bond_data.copy()
    
    # Interest rate shock impact
    if 'interest_rate_shock' in scenarios:
        ir_shock = scenarios['interest_rate_shock']
        stressed_data['Yield_to_Maturity(%)'] = bond_data.get('Yield_to_Maturity(%)', 5.0) + ir_shock
        # Adjust spread based on duration
        duration = bond_data.get('Duration_Years', 5.0)
        spread_adjustment = ir_shock * duration * 0.1
        stressed_data['Liquidity_Spread(bps)'] = bond_data.get('Liquidity_Spread(bps', 100) + spread_adjustment
    
    # Inflation shock impact
    if 'inflation_shock' in scenarios:
        inflation_shock = scenarios['inflation_shock']
        stressed_data['ESG_Score'] = max(0, bond_data.get('ESG_Score', 70) - inflation_shock * 5)
    
    # FX volatility impact
    if 'fx_volatility' in scenarios:
        fx_vol = scenarios['fx_volatility']
        stressed_data['Liquidity_Score(0-100)'] = max(0, bond_data.get('Liquidity_Score(0-100)', 60) - fx_vol * 20)
    
    # Liquidity freeze impact
    if 'liquidity_freeze' in scenarios:
        liquidity_freeze = scenarios['liquidity_freeze']
        stressed_data['Liquidity_Score(0-100)'] = max(0, bond_data.get('Liquidity_Score(0-100)', 60) * (1 - liquidity_freeze))
        stressed_data['Liquidity_Spread(bps)'] = bond_data.get('Liquidity_Spread(bps', 100) * (1 + liquidity_freeze)
    
    return stressed_data

def generate_enhanced_synthetic_dataset(target_size: int = 1000) -> pd.DataFrame:
    """Generate enhanced synthetic dataset with expanded diversity and macroeconomic factors."""
    logger.info(f"Generating enhanced synthetic dataset with {target_size} bonds")
    
    bonds_data = []
    company_count = 0
    
    # Generate bonds for each sector
    for sector, companies in SYNTHETIC_COMPANIES.items():
        sector_weight = SECTOR_RISK_WEIGHTS.get(sector, 0.15)
        sector_bonds = max(1, int(target_size * sector_weight))
        
        for i in range(sector_bonds):
            company_name = companies[i % len(companies)]
            company_count += 1
            
            # Generate bond with realistic correlations
            bond_data = generate_single_bond(company_name, sector, company_count)
            
            # Apply macroeconomic stress scenarios
            scenarios = generate_macroeconomic_scenarios()
            stressed_bond = apply_macroeconomic_stress(bond_data, scenarios)
            
            bonds_data.append(stressed_bond)
            
            if len(bonds_data) >= target_size:
                break
        
        if len(bonds_data) >= target_size:
            break
    
    # Ensure we have exactly target_size bonds
    bonds_data = bonds_data[:target_size]
    
    # Convert to DataFrame
    df = pd.DataFrame(bonds_data)
    
    # Add macroeconomic scenario metadata
    df['Macro_Scenario'] = df.index.map(lambda x: f"scenario_{x % len(MACRO_FACTORS)}")
    df['Stress_Test_Level'] = df.index.map(lambda x: (x % 5) + 1)
    
    logger.info(f"Generated {len(df)} bonds across {len(SYNTHETIC_COMPANIES)} sectors")
    return df

def generate_single_bond(company_name: str, sector: str, company_id: int) -> Dict[str, Any]:
    """Generate a single synthetic bond with realistic correlations."""
    
    # Generate bond ID
    bond_id = f"BOND-{company_id:06d}"
    
    # Select rating based on sector risk
    sector_risk = SECTOR_RISK_WEIGHTS.get(sector, 0.15)
    rating_weights = np.array(RATING_WEIGHTS)
    
    # Adjust weights based on sector risk
    if sector_risk > 0.2:  # High risk sectors
        rating_weights[:8] *= 0.5  # Reduce AAA-BBB weights
        rating_weights[8:] *= 2.0  # Increase BB-D weights
    elif sector_risk < 0.1:  # Low risk sectors
        rating_weights[:8] *= 2.0  # Increase AAA-BBB weights
        rating_weights[8:] *= 0.5  # Reduce BB-D weights
    
    rating_weights = rating_weights / rating_weights.sum()
    rating = np.random.choice(RATINGS, p=rating_weights)
    
    # Get base spread for rating
    base_spread_min, base_spread_max = get_rating_base_spread(rating)
    base_spread = random.uniform(base_spread_min, base_spread_max)
    
    # Add sector-specific adjustments
    sector_adjustment = (sector_risk - 0.15) * 100
    adjusted_spread = base_spread + sector_adjustment
    
    # Generate other bond characteristics
    maturity_years = random.uniform(0.5, 30.0)
    coupon_rate = random.uniform(2.0, 15.0)
    face_value = random.choice([1000, 5000, 10000, 25000, 50000, 100000])
    
    # Calculate derived values
    market_price = face_value * (1 + random.uniform(-0.2, 0.2))
    ytm = calculate_yield_to_maturity(coupon_rate, market_price, face_value, maturity_years)
    
    # Generate liquidity metrics
    liquidity_score = max(0, min(100, 80 - (adjusted_spread / 10) + random.uniform(-10, 10)))
    liquidity_spread = max(0, adjusted_spread + random.uniform(-20, 20))
    
    # Generate ESG metrics
    esg_score = max(0, min(100, 85 - (sector_risk * 100) + random.uniform(-15, 15)))
    
    # Generate alt-data metrics
    alt_traffic = random.uniform(0, 200)
    alt_utility = random.uniform(0, 200)
    sentiment = random.uniform(-1, 1)
    buzz_volume = random.uniform(0, 1000)
    
    # Generate BondX score
    bondx_score = max(0, min(100, 
        (liquidity_score * 0.3 + 
         esg_score * 0.2 + 
         (1 - sector_risk) * 100 * 0.2 + 
         (1 - adjusted_spread / 1000) * 100 * 0.3) +
        random.uniform(-5, 5)
    ))
    
    # Generate exit path indicators
    mm_available = liquidity_score > 70
    auction_next_time = None
    rfq_window = None
    tokenized_eligible = liquidity_score < 40
    
    if liquidity_score >= 40 and liquidity_score < 70:
        if random.random() > 0.5:
            auction_next_time = (datetime.now() + timedelta(days=random.randint(1, 30))).isoformat()
        else:
            rfq_window = f"{(datetime.now() + timedelta(days=1)).isoformat()}/{datetime.now() + timedelta(days=7).isoformat()}"
    
    # Generate macroeconomic stress indicators
    stress_level = random.randint(1, 5)
    macro_impact = random.uniform(0.8, 1.2)
    
    return {
        'Issuer_Name': company_name,
        'Issuer_ID': generate_deterministic_uuid(company_name, bond_id),
        'Bond_ID': bond_id,
        'Sector': sector,
        'Rating': rating,
        'Coupon_Rate(%)': round(coupon_rate, 2),
        'Maturity_Years': round(maturity_years, 1),
        'Face_Value($)': face_value,
        'Market_Price($)': round(market_price, 2),
        'Yield_to_Maturity(%)': round(ytm, 2),
        'Liquidity_Spread(bps)': round(liquidity_spread),
        'Liquidity_Score(0-100)': round(liquidity_score),
        'ESG_Score': round(esg_score),
        'BondX_Score': round(bondx_score),
        'Alt_Traffic_Index': round(alt_traffic),
        'Alt_Utility_Load': round(alt_utility),
        'Sentiment_Score': round(sentiment, 3),
        'Buzz_Volume': round(buzz_volume),
        'Duration_Years': round(maturity_years * 0.8, 1),
        'Convexity': round(maturity_years * maturity_years * 0.1, 3),
        'MM_Available': mm_available,
        'Auction_Next_Time': auction_next_time,
        'RFQ_Window': rfq_window,
        'Tokenized_Eligible': tokenized_eligible,
        'Stress_Level': stress_level,
        'Macro_Impact': round(macro_impact, 3),
        'Sector_Risk_Weight': round(sector_risk, 3),
        'Rating_Numeric': RATINGS.index(rating) + 1
    }

def export_datasets(bonds_data: pd.DataFrame, output_dir: Path):
    """Export datasets to multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Export to CSV
    csv_path = output_dir / "enhanced_bonds_1000plus.csv"
    bonds_data.to_csv(csv_path, index=False)
    logger.info(f"Exported CSV to {csv_path}")
    
    # Export to JSONL
    jsonl_path = output_dir / "enhanced_bonds_1000plus.jsonl"
    with open(jsonl_path, 'w') as f:
        for _, row in bonds_data.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    logger.info(f"Exported JSONL to {jsonl_path}")
    
    # Export to Parquet
    parquet_path = output_dir / "enhanced_bonds_1000plus.parquet"
    bonds_data.to_parquet(parquet_path, index=False)
    logger.info(f"Exported Parquet to {parquet_path}")
    
    # Export metadata
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "total_bonds": len(bonds_data),
        "sectors": bonds_data['Sector'].nunique(),
        "ratings": bonds_data['Rating'].nunique(),
        "seed": RANDOM_SEED,
        "macro_factors": list(MACRO_FACTORS.keys()),
        "stress_levels": bonds_data['Stress_Test_Level'].nunique(),
        "schema_version": "2.0.0"
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Exported metadata to {metadata_path}")

if __name__ == "__main__":
    # Generate enhanced dataset
    bonds_data = generate_enhanced_synthetic_dataset(1000)
    
    # Export datasets
    export_datasets(bonds_data, Path("data/synthetic"))
    
    print(f"Generated {len(bonds_data)} synthetic bonds")
    print(f"Sectors: {bonds_data['Sector'].nunique()}")
    print(f"Ratings: {bonds_data['Rating'].nunique()}")
    print(f"Stress levels: {bonds_data['Stress_Test_Level'].nunique()}")
