"""
Corporate Bonds Data Loader for BondX

This module handles loading and processing real corporate bonds data
from CSV files and other data sources.
"""

import csv
import re
from decimal import Decimal
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CorporateBond:
    """Corporate bond data structure."""
    isin: str
    descriptor: str
    weighted_avg_price: Decimal
    last_trade_price: Decimal
    weighted_avg_yield: Decimal
    last_trade_yield: Decimal
    value_lakhs: Decimal
    num_trades: int
    
    # Derived fields
    issuer_name: str = ""
    bond_type: str = ""
    coupon_rate: Optional[Decimal] = None
    maturity_date: Optional[date] = None
    face_value: Optional[Decimal] = None
    sector: str = "Corporate"
    rating: str = "A"  # Default rating
    
    def __post_init__(self):
        """Extract additional information from descriptor."""
        self._parse_descriptor()
    
    def _parse_descriptor(self):
        """Parse bond descriptor to extract issuer, coupon, maturity, etc."""
        if not self.descriptor or self.descriptor == "-":
            return
        
        # Extract issuer name (first part before rate/maturity info)
        parts = self.descriptor.split()
        if parts:
            # Find where the numerical/date info starts
            issuer_parts = []
            for part in parts:
                if any(char.isdigit() for char in part) and ('.' in part or '%' in part):
                    break
                issuer_parts.append(part)
            
            self.issuer_name = " ".join(issuer_parts).title()
        
        # Extract coupon rate
        coupon_match = re.search(r'(\d+\.?\d*)\s*(?:%|NCD|BD)', self.descriptor)
        if coupon_match:
            try:
                self.coupon_rate = Decimal(coupon_match.group(1))
            except:
                pass
        
        # Extract maturity date
        date_match = re.search(r'(\d{2}[A-Z]{2}\d{2,4})', self.descriptor)
        if date_match:
            try:
                date_str = date_match.group(1)
                # Convert to proper date format
                self.maturity_date = self._parse_date(date_str)
            except:
                pass
        
        # Extract face value
        fv_match = re.search(r'FVRS?(\d+(?:LAC|CR)?)', self.descriptor)
        if fv_match:
            fv_str = fv_match.group(1)
            try:
                if 'LAC' in fv_str:
                    self.face_value = Decimal(fv_str.replace('LAC', '')) * 100000
                elif 'CR' in fv_str:
                    self.face_value = Decimal(fv_str.replace('CR', '')) * 10000000
                else:
                    self.face_value = Decimal(fv_str)
            except:
                pass
        
        # Determine bond type
        if 'NCD' in self.descriptor:
            self.bond_type = 'NCD'  # Non-Convertible Debenture
        elif 'BD' in self.descriptor:
            self.bond_type = 'Bond'
        else:
            self.bond_type = 'Other'
        
        # Assign sector based on issuer name
        self.sector = self._determine_sector()
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string like '29AP30' to proper date."""
        try:
            # Map month abbreviations
            month_map = {
                'JN': 'JAN', 'FB': 'FEB', 'MR': 'MAR', 'AP': 'APR',
                'MY': 'MAY', 'JU': 'JUN', 'JL': 'JUL', 'AG': 'AUG',
                'SP': 'SEP', 'OT': 'OCT', 'NV': 'NOV', 'DC': 'DEC'
            }
            
            if len(date_str) == 6:  # DDMMYY format
                day = int(date_str[:2])
                month_abbr = date_str[2:4]
                year = int(date_str[4:])
                
                if month_abbr in month_map:
                    month_abbr = month_map[month_abbr]
                
                # Convert 2-digit year to 4-digit
                if year < 50:
                    year += 2000
                else:
                    year += 1900
                
                # Map month abbreviation to number
                month_num_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                
                month = month_num_map.get(month_abbr, 1)
                return date(year, month, day)
        except:
            pass
        return None
    
    def _determine_sector(self) -> str:
        """Determine sector based on issuer name."""
        if not self.issuer_name:
            return "Corporate"
        
        name_upper = self.issuer_name.upper()
        
        # Banking and Financial Services
        if any(keyword in name_upper for keyword in [
            'BANK', 'FINANCE', 'FINANCIAL', 'CAPITAL', 'CREDIT'
        ]):
            return "Financial Services"
        
        # Energy and Oil
        if any(keyword in name_upper for keyword in [
            'PETROLEUM', 'OIL', 'POWER', 'ENERGY', 'ELECTRIFICATION'
        ]):
            return "Energy"
        
        # Infrastructure
        if any(keyword in name_upper for keyword in [
            'INFRASTRUCTURE', 'AIRPORT', 'GRID', 'DEVELOPMENT'
        ]):
            return "Infrastructure"
        
        # Automotive
        if any(keyword in name_upper for keyword in [
            'BAJAJ', 'MAHINDRA', 'HERO'
        ]):
            return "Automotive"
        
        # Steel and Metals
        if any(keyword in name_upper for keyword in [
            'STEEL', 'JSW'
        ]):
            return "Metals"
        
        return "Corporate"

class CorporateBondsLoader:
    """Loader for corporate bonds data."""
    
    def __init__(self, csv_file_path: str = None):
        """Initialize the loader."""
        if csv_file_path is None:
            # Default to the project's data directory
            project_root = Path(__file__).parent.parent.parent
            csv_file_path = project_root / "data" / "corporate_bonds.csv"
        
        self.csv_file_path = Path(csv_file_path)
        self.bonds: List[CorporateBond] = []
        self._load_bonds()
    
    def _clean_value(self, value: str) -> str:
        """Clean CSV value by removing quotes and extra whitespace."""
        if not value:
            return ""
        return value.strip().strip('"').strip()
    
    def _parse_decimal(self, value: str) -> Decimal:
        """Parse decimal value from string, handling commas and special cases."""
        if not value or value == "-":
            return Decimal('0')
        
        # Remove commas and extra whitespace
        cleaned = self._clean_value(value).replace(',', '')
        
        try:
            return Decimal(cleaned)
        except:
            logger.warning(f"Could not parse decimal value: {value}")
            return Decimal('0')
    
    def _parse_int(self, value: str) -> int:
        """Parse integer value from string."""
        if not value or value == "-":
            return 0
        
        cleaned = self._clean_value(value).replace(',', '')
        try:
            return int(float(cleaned))  # Convert via float to handle decimals
        except:
            logger.warning(f"Could not parse integer value: {value}")
            return 0
    
    def _load_bonds(self):
        """Load bonds from CSV file."""
        try:
            if not self.csv_file_path.exists():
                logger.error(f"CSV file not found: {self.csv_file_path}")
                return
            
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                # Read all lines and handle the multi-line header
                lines = file.readlines()
                
                # Find the actual header line (should contain ISIN)
                header_line_idx = 0
                for i, line in enumerate(lines):
                    if 'ISIN' in line and 'DESCRIPTOR' in line:
                        header_line_idx = i
                        break
                
                # Skip to data lines (after header)
                data_lines = lines[header_line_idx + 1:]
                
                # Parse CSV data
                csv_reader = csv.reader(data_lines)
                
                for row_idx, row in enumerate(csv_reader):
                    if len(row) < 8:  # Skip incomplete rows
                        continue
                    
                    try:
                        # Extract and clean values
                        isin = self._clean_value(row[0])
                        descriptor = self._clean_value(row[1])
                        weighted_avg_price = self._parse_decimal(row[2])
                        last_trade_price = self._parse_decimal(row[3])
                        weighted_avg_yield = self._parse_decimal(row[4])
                        last_trade_yield = self._parse_decimal(row[5])
                        value_lakhs = self._parse_decimal(row[6])
                        num_trades = self._parse_int(row[7])
                        
                        # Skip rows with empty ISIN
                        if not isin:
                            continue
                        
                        # Create bond object
                        bond = CorporateBond(
                            isin=isin,
                            descriptor=descriptor,
                            weighted_avg_price=weighted_avg_price,
                            last_trade_price=last_trade_price,
                            weighted_avg_yield=weighted_avg_yield,
                            last_trade_yield=last_trade_yield,
                            value_lakhs=value_lakhs,
                            num_trades=num_trades
                        )
                        
                        self.bonds.append(bond)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing row {row_idx}: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.bonds)} corporate bonds from {self.csv_file_path}")
            
        except Exception as e:
            logger.error(f"Error loading bonds from CSV: {e}")
    
    def get_all_bonds(self) -> List[CorporateBond]:
        """Get all loaded bonds."""
        return self.bonds
    
    def get_bonds_by_sector(self, sector: str) -> List[CorporateBond]:
        """Get bonds filtered by sector."""
        return [bond for bond in self.bonds if bond.sector == sector]
    
    def get_bonds_by_issuer(self, issuer: str) -> List[CorporateBond]:
        """Get bonds filtered by issuer."""
        return [bond for bond in self.bonds if issuer.lower() in bond.issuer_name.lower()]
    
    def get_top_bonds_by_volume(self, limit: int = 10) -> List[CorporateBond]:
        """Get top bonds by trading volume."""
        return sorted(self.bonds, key=lambda x: x.value_lakhs, reverse=True)[:limit]
    
    def get_bonds_by_yield_range(self, min_yield: float, max_yield: float) -> List[CorporateBond]:
        """Get bonds within a yield range."""
        return [
            bond for bond in self.bonds 
            if min_yield <= float(bond.last_trade_yield) <= max_yield
        ]
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary statistics."""
        if not self.bonds:
            return {}
        
        total_value = sum(bond.value_lakhs for bond in self.bonds)
        total_trades = sum(bond.num_trades for bond in self.bonds)
        avg_yield = sum(bond.last_trade_yield for bond in self.bonds) / len(self.bonds)
        
        # Sector breakdown
        sector_counts = {}
        sector_values = {}
        for bond in self.bonds:
            sector = bond.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            sector_values[sector] = sector_values.get(sector, Decimal('0')) + bond.value_lakhs
        
        return {
            "total_bonds": len(self.bonds),
            "total_value_lakhs": float(total_value),
            "total_trades": total_trades,
            "average_yield": float(avg_yield),
            "sector_breakdown": {
                "counts": sector_counts,
                "values": {k: float(v) for k, v in sector_values.items()}
            }
        }

# Global instance
_bonds_loader = None

def get_bonds_loader() -> CorporateBondsLoader:
    """Get the global bonds loader instance."""
    global _bonds_loader
    if _bonds_loader is None:
        _bonds_loader = CorporateBondsLoader()
    return _bonds_loader

def refresh_bonds_data():
    """Refresh the bonds data by reloading from CSV."""
    global _bonds_loader
    _bonds_loader = CorporateBondsLoader()
    logger.info("Corporate bonds data refreshed")

# Export key classes and functions
__all__ = ["CorporateBond", "CorporateBondsLoader", "get_bonds_loader", "refresh_bonds_data"]
