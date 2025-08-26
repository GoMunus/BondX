# BondX Synthetic Dataset System

## Overview

This directory contains a complete synthetic dataset generation system for BondX liquidity and risk features. **ALL DATA IS SYNTHETIC AND CLEARLY LABELED AS SUCH.**

The system generates 260 synthetic issuers across 11 business sectors with realistic bond characteristics, liquidity metrics, ESG KPIs, and exit path planning data.

## What's Included

### Core Files
- `generate_synthetic_dataset.py` - Main dataset generator (260 companies)
- `test_dataset.py` - Validation and testing script
- `run_system.py` - Simple launcher script
- `CURSOR_PROMPT.md` - Ready-to-use Cursor prompt
- `requirements.txt` - Optional dependencies
- `README.md` - This documentation

### Generated Output
- `bondx_issuers_260.csv` - CSV format dataset
- `bondx_issuers_260.jsonl` - JSON Lines format dataset

## Quick Start

### Prerequisites
- Python 3.7+ (standard library only - no external packages required)
- Windows PowerShell or Command Prompt

### Basic Usage

1. **Generate Dataset:**
   ```bash
   python generate_synthetic_dataset.py
   ```

2. **Run Tests:**
   ```bash
   python test_dataset.py
   ```

3. **Run Complete System:**
   ```bash
   python run_system.py
   ```

## Dataset Schema

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

## Company Sectors

1. **Financials and Banking** (25 companies)
2. **Infrastructure, Power, Utilities** (25 companies)
3. **Industrial, Manufacturing** (26 companies)
4. **Technology, Software, Data** (26 companies)
5. **Healthcare, Pharma, Biotech** (24 companies)
6. **Consumer, Retail, E-commerce** (26 companies)
7. **Telecom, Media** (26 companies)
8. **Logistics, Transportation** (26 companies)
9. **Real Estate, Construction** (26 companies)
10. **Agriculture, Commodities** (26 companies)
11. **Energy Trading, Metals, Mining** (4 companies)

**Total: 260 synthetic companies**

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

## Data Quality Features

### Reproducibility
- Fixed random seed (42) ensures identical output on reruns
- Deterministic UUID generation from company names
- No external dependencies or network calls

### Realism Constraints
- Values within realistic ranges for bond markets
- Cross-field coherence maintained
- Sector-appropriate characteristics

### Validation
- Comprehensive test suite validates data quality
- Checks for value ranges, uniqueness, and coherence
- Verifies rating-spread relationships and correlations

## Use Cases

### Development & Testing
- Model training and validation
- End-to-end testing
- Performance benchmarking
- Feature development

### Training & Education
- Bond market concepts
- Liquidity analysis
- Risk management
- ESG integration

### Research & Prototyping
- Algorithm development
- System architecture testing
- Data pipeline validation
- API testing

## Cursor Integration

The `CURSOR_PROMPT.md` file contains a ready-to-use prompt for Cursor AI to generate similar synthetic datasets or extend the existing system.

## Troubleshooting

### Python Not Found
If you get "Python was not found" error:
1. Install Python from [python.org](https://python.org)
2. Ensure Python is added to PATH
3. Try using `python3` instead of `python`

### Import Errors
The system uses only Python standard library - no external packages required.

### File Permissions
Ensure you have write permissions in the current directory.

## Disclaimer

**WARNING: This is synthetic training data. None of the companies, ratings, or financial data represent real entities or actual market conditions. Use only for testing, development, and training purposes.**

## Seed Information

- Random seed: 42
- Generation timestamp: Generated on first run
- Total companies: 260
- Sectors: 11

## Support

For issues or questions:
1. Check the test output for validation errors
2. Verify Python installation and PATH
3. Ensure all files are in the same directory

---

**Ready to generate your synthetic BondX dataset!** 🚀
