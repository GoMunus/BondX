# Cursor Prompt for BondX Synthetic Dataset Generation

Create a synthetic, reproducible training dataset for BondX liquidity and risk features using the provided 260 synthetic issuer names (prefix SX-). Do not use or infer any real companies. Generate programmatically with a seeded RNG so reruns are identical.

## Data Goals

Provide 260 rows (one per issuer) with realistic-but-synthetic bond and liquidity attributes for model/e2e testing.

Maintain cross-field coherence (e.g., higher rating → typically lower spread; longer maturity → higher duration; stronger alt-data → higher repayment support).

Include fields for risk, liquidity, alt-data, ESG, and exit-path planning aligned to BondX modules.

## Schema (Columns)

**issuer_name, issuer_id (UUID or deterministic hash of name), sector**

**rating (AAA..BBB-), coupon_type (fixed/floating/zero/amortizing), coupon_rate_pct (3.00–14.50), maturity_years (0.5–20), face_value**

**liquidity_spread_bps, l2_depth_qty, trades_7d, time_since_last_trade_s**

**alt_traffic_index (0–200), alt_utility_load (0–200), sentiment_score (-1..1), buzz_volume (0..1000)**

**esg_target_renew_pct, esg_actual_renew_pct, esg_target_emission_intensity, esg_actual_emission_intensity**

**liquidity_index_0_100, repayment_support_0_100, bondx_score_0_100**

**mm_available (bool), auction_next_time (ISO or null), rfq_window (ISO range or null), tokenized_eligible (bool)**

## Generation Rules

**Ratings distribution skewed toward AA/A; a smaller tail in BBB.**

Map rating to base spread: AAA ~ 40–80 bps, AA ~ 60–120, A ~ 90–180, BBB ~ 140–300; add noise.

Longer maturity slightly increases spread and decreases liquidity_index.

Higher l2_depth_qty and trades_7d increase liquidity_index and reduce expected time-to-exit.

Alt-data (traffic/utility) correlates positively with repayment_support; sentiment_score affects bondx_score moderately.

ESG targets vs actual: generate gaps; larger misses reduce bondx_score slightly.

Exit path logic: if liquidity_index>70 and mm available → prefer MM; if 40–70 → auction/RFQ windows; if <40 → tokenized_eligible often true.

## Output

Produce two artifacts:

**CSV file** `data/synthetic/bondx_issuers_260.csv` with the full schema and 260 rows.

**JSONLines file** `data/synthetic/bondx_issuers_260.jsonl` mirroring the CSV.

Include a `data/README.md` documenting schema, generation rules, seed, and disclaimers ("Synthetic; not real entities").

Add unit tests that verify: distributions by rating; monotonic trends (higher rating → lower average spread); correlations (depth/trades with liquidity_index); reproducibility with fixed seed.

## Constraints

No real-world names or identifiers.

Deterministic seed; no external network calls.

Keep values within realistic ranges and coherent across fields.

## Company Universe

Use these 260 synthetic companies across 11 sectors:

**Financials and Banking (25):** SX-Apex Capital Bank, SX-BlueStone Finance, SX-Cascade Holdings, SX-Delta Trust Bank, SX-Equinox Financial, SX-First Meridian Credit, SX-Granite Securities, SX-Horizon Mutual, SX-IronGate Banking Corp, SX-JadeBridge Capital, SX-Kestrel Asset Group, SX-Lighthouse Investments, SX-Midway Finance PLC, SX-NorthArc Funds, SX-Oakline Credit Union, SX-Pillarstone Bank, SX-Quanta Capital, SX-Riverbend Trust, SX-Stronghold Finance, SX-Trident Securities, SX-UnionGate Financial, SX-Vertex Capital Markets, SX-Westward Finance, SX-Yellowtail Mutual, SX-Zenith Bank Group

**Infrastructure, Power, Utilities (25):** SX-Alpha Grid Transmission, SX-BrightPeak Power, SX-CityLink Utilities, SX-DeepRiver Hydro, SX-EdgeVolt Renewables, SX-Flux Energy Systems, SX-GreenArc Solar, SX-Helios Wind Partners, SX-InfraCore Roads, SX-Jetstream Gas, SX-Kiln Thermal Power, SX-Lattice Water Works, SX-MegaRail Logistics, SX-NovaMetro Transit, SX-Orbit Fiber Networks, SX-PowerBridge T&D, SX-Quantum SmartGrid, SX-Resonance Utilities, SX-Stoneport Ports, SX-Turbine Peak Power, SX-UrbanFlow Sewage, SX-Valence Energy Storage, SX-WaveLine Desalination, SX-Xenon City Gas, SX-Yardley InfraDev

**Industrial, Manufacturing (26):** SX-AeroForge Systems, SX-BetaMach Precision, SX-Covalent Materials, SX-DriveChain Motors, SX-Enginuity Tools, SX-ForgeLink Metals, SX-GigaFab Components, SX-Hexon Plastics, SX-IndiSteel Works, SX-Juno Robotics, SX-Kronos Machinery, SX-Laminar Pumps, SX-MicroAlloy Foundry, SX-Nimbus Aerospace, SX-OrthoTech Medical Devices, SX-PrimeGear Industrial, SX-Quantum Bearings, SX-RotorCore Turbines, SX-Sintered Parts Co, SX-TitanCast Foundries, SX-UltraFab Electronics, SX-Vector Engines, SX-WeldRight Systems, SX-XPress Conveyors, SX-Yotta Materials, SX-Zenufacture Labs

**Technology, Software, Data (26):** SX-AltWave Software, SX-ByteField Analytics, SX-CypherGrid Security, SX-DataHarbor Cloud, SX-EdgeFox Networks, SX-FusionStack Systems, SX-GlowAI Platforms, SX-HyperLoop Compute, SX-InsightForge BI, SX-Jigsaw DevTools, SX-KiteMesh IoT, SX-LucidSignal Data, SX-Mindline AI, SX-NexusLayer Infra, SX-Orbita Quantum, SX-PixelPeak Media, SX-QuarkLabs R&D, SX-RippleCode DevOps, SX-Spectral Semicon, SX-TerraNode DB, SX-Uplink Systems, SX-VoxelVision AR, SX-Wireframe UX, SX-XStream CDN, SX-Yosei Robotics, SX-ZettaCompute

**Healthcare, Pharma, Biotech (24):** SX-ApexBio Therapeutics, SX-BioNova Labs, SX-CareSphere Clinics, SX-Dermatek Pharma, SX-Enviva Diagnostics, SX-GenCrest Biologics, SX-HelixPath Genomics, SX-ImmunoCore Health, SX-JunoLife Hospitals, SX-KardioTech Devices, SX-LumenRx Pharma, SX-MediGrid Services, SX-NeuroAxis Care, SX-OncoPath Research, SX-PharmaVista, SX-QureWellness, SX-Reviva Biotech, SX-SurgiLine Tools, SX-TheraLink, SX-UroMed Devices, SX-VitalCore Care, SX-Wellfield Diagnostics, SX-XenRx Labs, SX-YoungLife Health

**Consumer, Retail, E-commerce (26):** SX-Aurora Retail, SX-BasketHub Commerce, SX-CityMart Stores, SX-Delight Foods, SX-Everglo Home, SX-FashionCircle, SX-GreenGrove Organics, SX-HomeBlend, SX-Intown Electronics, SX-JoyRide Mobility, SX-KartNook, SX-Luxora Beauty, SX-MetroMart, SX-NimbleWear, SX-OpenCartel, SX-PureNest, SX-QuickBasket, SX-RetailLoop, SX-StyleHive, SX-TruValue Bazaar, SX-UrbanThreads, SX-VitaFoods, SX-Wishlane, SX-XenoWear, SX-YummyCart, SX-ZenCart

**Telecom, Media (26):** SX-AirWave Mobile, SX-Broadcast One, SX-ChannelPeak Media, SX-DigiVerse OTT, SX-EchoLine Telecom, SX-FlexCast Studios, SX-GigaBeam Wireless, SX-HoloStream, SX-InfiniTalk, SX-JetCast Radio, SX-Kinetik Media, SX-LinkSphere, SX-MediaCrest, SX-NetPulse, SX-OmniCast, SX-PolyWave, SX-QuantaTel, SX-RadioNova, SX-SkyBridge Cable, SX-TeleCore, SX-UnityFiber, SX-ViewPort Media, SX-WaveCell, SX-XpressTel, SX-YoMedia, SX-ZenCast

**Logistics, Transportation (26):** SX-AeroPort Cargo, SX-BaseLine Freight, SX-CityLift Logistics, SX-DriveFleet, SX-ExpressHaul, SX-FleetAxis, SX-Gateway Shipping, SX-HarborLink, SX-Inland RailCo, SX-JetFreight, SX-KargoChain, SX-LoadRunner, SX-MetroShip, SX-NorthStar Freight, SX-OceanWing, SX-Portline Terminals, SX-QuickRail, SX-RoadBridge, SX-SkyLift Air, SX-TransNova, SX-UrbanPorts, SX-Vantage Logistics, SX-WayPoint Movers, SX-Xport Global, SX-Yardline, SX-ZoneShip

**Real Estate, Construction (26):** SX-Arcadia Estates, SX-BlueRock Developers, SX-Crestline Realty, SX-DeltaHomes, SX-Everstone REIT, SX-Foundry Builders, SX-GrandBuild, SX-HabitatWorks, SX-InfraMason, SX-Jaguar Homes, SX-Keystone Constructions, SX-Landmark Realty, SX-MetroLiving, SX-NestCore, SX-Orbit Homes, SX-PrimeSquare, SX-Quantum Realty, SX-Riverstone RE, SX-StructaBuild, SX-TerraHomes, SX-UrbanForge, SX-ValleyView, SX-Worksite Infra, SX-XenoBuild, SX-YieldStone, SX-ZenHabitats

**Agriculture, Commodities (26):** SX-AgriLine Mills, SX-BioHarvest, SX-CropCircle, SX-DeltaAgro, SX-EcoFarms, SX-FreshRoot, SX-GrainGate, SX-HarvestPeak, SX-Irrigo Systems, SX-JuxtaAgri, SX-KrishiCore, SX-Leaf&Loam, SX-MandiLink, SX-NatureNest, SX-OrchardWorks, SX-PrimeAg Commodities, SX-QuinoaQuarry, SX-RuralBridge, SX-SoilSync, SX-TerraFarm, SX-UrbanSprout, SX-VerdantCo, SX-Wetland Produce, SX-Xylem Agro, SX-YieldNova, SX-ZenCrop

**Energy Trading, Metals, Mining (4):** SX-AlloyCore Mining, SX-BulkMetals, SX-CoreMinerals, SX-DrillStone Energy

This gives Cursor clear instructions to generate high-quality, clearly synthetic training data without risking confusion or misuse.
