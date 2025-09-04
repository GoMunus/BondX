"""
Automated Regulatory Capital Engine for BondX

This module implements automated regulatory capital calculations:
- Basel III/IV frameworks (Standardized vs Internal Models Approach)
- Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR)
- SEBI/RBI compliance reporting
- Automated capital requirement calculations
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from pathlib import Path

from ..core.logging import get_logger
from ..core.monitoring import MetricsCollector
from .portfolio_risk import PortfolioRiskManager
from ..mathematics.bond_pricing import BondPricingEngine
from ..database.models import Portfolio, Position, Bond

logger = get_logger(__name__)

class BaselFramework(Enum):
    """Basel framework versions"""
    BASEL_III = "basel_iii"
    BASEL_IV = "basel_iv"

class ApproachType(Enum):
    """Regulatory approach types"""
    STANDARDIZED = "standardized"
    INTERNAL_MODELS = "internal_models"
    FOUNDATION_IRB = "foundation_irb"
    ADVANCED_IRB = "advanced_irb"

class AssetClass(Enum):
    """Asset classification for regulatory purposes"""
    SOVEREIGN = "sovereign"
    PUBLIC_SECTOR = "public_sector"
    BANKS = "banks"
    CORPORATES = "corporates"
    COVERED_BONDS = "covered_bonds"
    RESIDENTIAL_MORTGAGES = "residential_mortgages"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    EQUITY = "equity"
    OTHER = "other"

@dataclass
class RegulatoryConfig:
    """Configuration for regulatory calculations"""
    basel_framework: BaselFramework = BaselFramework.BASEL_IV
    approach: ApproachType = ApproachType.STANDARDIZED
    jurisdiction: str = "INDIA"  # SEBI/RBI specific
    reporting_frequency: str = "DAILY"
    confidence_level: float = 0.999  # 99.9% for regulatory capital
    holding_period: int = 10  # 10-day holding period
    multiplier: float = 3.0  # Regulatory multiplier

@dataclass
class CapitalRequirement:
    """Regulatory capital requirement result"""
    portfolio_id: str
    calculation_date: date
    total_capital: float
    credit_risk_capital: float
    market_risk_capital: float
    operational_risk_capital: float
    lcr_requirement: float
    nsfr_requirement: float
    leverage_ratio: float
    methodology: str
    metadata: Dict[str, Any]

@dataclass
class LCRResult:
    """Liquidity Coverage Ratio calculation result"""
    portfolio_id: str
    calculation_date: date
    lcr_ratio: float
    high_quality_liquid_assets: float
    net_cash_outflows: float
    required_liquidity: float
    stress_scenarios: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class NSFRResult:
    """Net Stable Funding Ratio calculation result"""
    portfolio_id: str
    calculation_date: date
    nsfr_ratio: float
    available_stable_funding: float
    required_stable_funding: float
    funding_stability: Dict[str, float]
    metadata: Dict[str, Any]

class RegulatoryCapitalEngine:
    """Automated regulatory capital calculation engine"""
    
    def __init__(self, config: RegulatoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Load regulatory parameters
        self.risk_weights = self._load_risk_weights()
        self.liquidity_factors = self._load_liquidity_factors()
        self.funding_factors = self._load_funding_factors()
        
        # Performance tracking
        self.performance_metrics = {
            'total_calculations': 0,
            'average_calculation_time': 0.0,
            'last_calculation': None
        }
    
    def _load_risk_weights(self) -> Dict[str, Dict[str, float]]:
        """Load risk weights for different asset classes"""
        # Basel IV risk weights (simplified)
        return {
            AssetClass.SOVEREIGN.value: {
                "AAA": 0.0,
                "AA": 0.0,
                "A": 0.2,
                "BBB": 0.5,
                "BB": 1.0,
                "B": 1.5,
                "CCC": 3.0,
                "D": 10.0
            },
            AssetClass.PUBLIC_SECTOR.value: {
                "AAA": 0.0,
                "AA": 0.2,
                "A": 0.2,
                "BBB": 0.5,
                "BB": 1.0,
                "B": 1.5,
                "CCC": 3.0,
                "D": 10.0
            },
            AssetClass.BANKS.value: {
                "AAA": 0.2,
                "AA": 0.2,
                "A": 0.3,
                "BBB": 0.5,
                "BB": 1.0,
                "B": 1.5,
                "CCC": 3.0,
                "D": 10.0
            },
            AssetClass.CORPORATES.value: {
                "AAA": 0.2,
                "AA": 0.2,
                "A": 0.5,
                "BBB": 1.0,
                "BB": 1.5,
                "B": 2.0,
                "CCC": 3.0,
                "D": 10.0
            }
        }
    
    def _load_liquidity_factors(self) -> Dict[str, float]:
        """Load liquidity factors for LCR calculation"""
        return {
            "sovereign_bonds": 1.0,  # 100% HQLA
            "central_bank_reserves": 1.0,
            "high_grade_corporate_bonds": 0.5,  # 50% HQLA
            "covered_bonds": 0.7,
            "residential_mortgages": 0.3,
            "commercial_real_estate": 0.2,
            "equity": 0.0,  # Not HQLA
            "other_assets": 0.0
        }
    
    def _load_funding_factors(self) -> Dict[str, float]:
        """Load funding stability factors for NSFR"""
        return {
            "stable_deposits": 0.95,  # 95% stable
            "less_stable_deposits": 0.5,
            "wholesale_funding": 0.0,  # 0% stable
            "equity": 1.0,  # 100% stable
            "long_term_debt": 0.5,
            "short_term_debt": 0.0
        }
    
    async def calculate_regulatory_capital(self, portfolio: Portfolio) -> CapitalRequirement:
        """Calculate total regulatory capital requirement"""
        start_time = datetime.utcnow()
        
        try:
            # Calculate individual capital components
            credit_capital = await self._calculate_credit_risk_capital(portfolio)
            market_capital = await self._calculate_market_risk_capital(portfolio)
            operational_capital = await self._calculate_operational_risk_capital(portfolio)
            
            # Total capital requirement
            total_capital = credit_capital + market_capital + operational_capital
            
            # Apply regulatory multiplier
            total_capital *= self.config.multiplier
            
            # Calculate additional ratios
            lcr_result = await self.calculate_lcr(portfolio)
            nsfr_result = await self.calculate_nsfr(portfolio)
            
            calculation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_performance_metrics(calculation_time)
            
            return CapitalRequirement(
                portfolio_id=portfolio.id,
                calculation_date=date.today(),
                total_capital=total_capital,
                credit_risk_capital=credit_capital,
                market_risk_capital=market_capital,
                operational_risk_capital=operational_capital,
                lcr_requirement=lcr_result.lcr_ratio,
                nsfr_requirement=nsfr_result.nsfr_ratio,
                leverage_ratio=total_capital / portfolio.total_value if portfolio.total_value > 0 else 0.0,
                methodology=self.config.approach.value,
                metadata={
                    "basel_framework": self.config.basel_framework.value,
                    "calculation_time_seconds": calculation_time,
                    "regulatory_multiplier": self.config.multiplier
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating regulatory capital: {e}")
            raise
    
    async def _calculate_credit_risk_capital(self, portfolio: Portfolio) -> float:
        """Calculate credit risk capital using standardized approach"""
        if self.config.approach == ApproachType.STANDARDIZED:
            return await self._standardized_credit_risk(portfolio)
        else:
            return await self._internal_models_credit_risk(portfolio)
    
    async def _standardized_credit_risk(self, portfolio: Portfolio) -> float:
        """Standardized approach for credit risk capital"""
        total_capital = 0.0
        
        for position in portfolio.positions:
            # Get asset class and rating
            asset_class = self._classify_asset(position.bond)
            rating = position.bond.credit_rating
            
            # Get risk weight
            risk_weight = self.risk_weights.get(asset_class, {}).get(rating, 1.0)
            
            # Calculate capital requirement
            exposure = position.notional_value
            capital = exposure * risk_weight * 0.08  # 8% minimum capital ratio
            
            total_capital += capital
        
        return total_capital
    
    async def _internal_models_credit_risk(self, portfolio: Portfolio) -> float:
        """Internal models approach for credit risk capital"""
        # This would use internal PD/LGD/EAD models
        # For now, return a simplified calculation
        total_exposure = sum(pos.notional_value for pos in portfolio.positions)
        
        # Simplified internal model: 2% of total exposure
        return total_exposure * 0.02
    
    async def _calculate_market_risk_capital(self, portfolio: Portfolio) -> float:
        """Calculate market risk capital"""
        if self.config.approach == ApproachType.STANDARDIZED:
            return await self._standardized_market_risk(portfolio)
        else:
            return await self._internal_models_market_risk(portfolio)
    
    async def _standardized_market_risk(self, portfolio: Portfolio) -> float:
        """Standardized approach for market risk capital"""
        # Interest rate risk
        interest_rate_risk = await self._calculate_interest_rate_risk(portfolio)
        
        # Credit spread risk
        credit_spread_risk = await self._calculate_credit_spread_risk(portfolio)
        
        # Equity risk
        equity_risk = await self._calculate_equity_risk(portfolio)
        
        # Total market risk (simplified correlation)
        total_risk = np.sqrt(interest_rate_risk**2 + credit_spread_risk**2 + equity_risk**2)
        
        return total_risk
    
    async def _internal_models_market_risk(self, portfolio: Portfolio) -> float:
        """Internal models approach for market risk capital"""
        # This would use VaR/ES models with regulatory parameters
        # For now, return a simplified calculation
        total_value = portfolio.total_value
        
        # Simplified internal model: 1% of portfolio value
        return total_value * 0.01
    
    async def _calculate_interest_rate_risk(self, portfolio: Portfolio) -> float:
        """Calculate interest rate risk capital"""
        total_risk = 0.0
        
        for position in portfolio.positions:
            duration = position.bond.duration
            convexity = position.bond.convexity
            notional = position.notional_value
            
            # Simplified interest rate risk calculation
            # Based on duration and convexity
            risk = notional * (duration * 0.02 + 0.5 * convexity * 0.02**2)
            total_risk += risk
        
        return total_risk
    
    async def _calculate_credit_spread_risk(self, portfolio: Portfolio) -> float:
        """Calculate credit spread risk capital"""
        total_risk = 0.0
        
        for position in portfolio.positions:
            duration = position.bond.duration
            notional = position.notional_value
            rating = position.bond.credit_rating
            
            # Credit spread risk based on rating
            spread_volatility = self._get_spread_volatility(rating)
            risk = notional * duration * spread_volatility
            
            total_risk += risk
        
        return total_risk
    
    async def _calculate_equity_risk(self, portfolio: Portfolio) -> float:
        """Calculate equity risk capital"""
        # For bond portfolios, equity risk is typically minimal
        # This would be more relevant for hybrid portfolios
        return 0.0
    
    def _get_spread_volatility(self, rating: str) -> float:
        """Get credit spread volatility by rating"""
        spread_vols = {
            "AAA": 0.005,
            "AA": 0.008,
            "A": 0.012,
            "BBB": 0.020,
            "BB": 0.035,
            "B": 0.050,
            "CCC": 0.080,
            "D": 0.150
        }
        return spread_vols.get(rating, 0.025)
    
    async def _calculate_operational_risk_capital(self, portfolio: Portfolio) -> float:
        """Calculate operational risk capital"""
        # Simplified operational risk calculation
        # In production, this would use more sophisticated models
        
        total_value = portfolio.total_value
        
        # Basic indicator approach: 15% of average gross income
        # For bond portfolios, use 0.1% of portfolio value as proxy
        return total_value * 0.001
    
    async def calculate_lcr(self, portfolio: Portfolio) -> LCRResult:
        """Calculate Liquidity Coverage Ratio"""
        # Calculate high-quality liquid assets
        hqla = await self._calculate_hqla(portfolio)
        
        # Calculate net cash outflows
        net_outflows = await self._calculate_net_cash_outflows(portfolio)
        
        # Calculate LCR
        lcr_ratio = hqla / net_outflows if net_outflows > 0 else float('inf')
        
        # Required liquidity (100% of net outflows)
        required_liquidity = net_outflows
        
        # Stress scenarios
        stress_scenarios = {
            "30_day_outflow": net_outflows,
            "7_day_outflow": net_outflows * 0.3,
            "1_day_outflow": net_outflows * 0.1
        }
        
        return LCRResult(
            portfolio_id=portfolio.id,
            calculation_date=date.today(),
            lcr_ratio=lcr_ratio,
            high_quality_liquid_assets=hqla,
            net_cash_outflows=net_outflows,
            required_liquidity=required_liquidity,
            stress_scenarios=stress_scenarios,
            metadata={"calculation_method": "standardized"}
        )
    
    async def _calculate_hqla(self, portfolio: Portfolio) -> float:
        """Calculate high-quality liquid assets"""
        hqla = 0.0
        
        for position in portfolio.positions:
            asset_type = self._classify_asset(position.bond)
            liquidity_factor = self.liquidity_factors.get(asset_type, 0.0)
            
            hqla += position.market_value * liquidity_factor
        
        return hqla
    
    async def _calculate_net_cash_outflows(self, portfolio: Portfolio) -> float:
        """Calculate net cash outflows for LCR"""
        # Simplified calculation
        # In production, this would consider actual cash flow schedules
        
        total_outflows = 0.0
        
        for position in portfolio.positions:
            # Assume 5% of portfolio value as potential outflows
            total_outflows += position.market_value * 0.05
        
        return total_outflows
    
    async def calculate_nsfr(self, portfolio: Portfolio) -> NSFRResult:
        """Calculate Net Stable Funding Ratio"""
        # Calculate available stable funding
        asf = await self._calculate_available_stable_funding(portfolio)
        
        # Calculate required stable funding
        rsf = await self._calculate_required_stable_funding(portfolio)
        
        # Calculate NSFR
        nsfr_ratio = asf / rsf if rsf > 0 else float('inf')
        
        # Funding stability breakdown
        funding_stability = {
            "stable_funding": asf,
            "required_funding": rsf,
            "funding_gap": asf - rsf
        }
        
        return NSFRResult(
            portfolio_id=portfolio.id,
            calculation_date=date.today(),
            nsfr_ratio=nsfr_ratio,
            available_stable_funding=asf,
            required_stable_funding=rsf,
            funding_stability=funding_stability,
            metadata={"calculation_method": "standardized"}
        )
    
    async def _calculate_available_stable_funding(self, portfolio: Portfolio) -> float:
        """Calculate available stable funding"""
        # Simplified calculation
        # In production, this would consider actual funding sources
        
        total_value = portfolio.total_value
        
        # Assume 80% of portfolio value is funded by stable sources
        return total_value * 0.8
    
    async def _calculate_required_stable_funding(self, portfolio: Portfolio) -> float:
        """Calculate required stable funding"""
        # Simplified calculation
        # In production, this would consider asset liquidity profiles
        
        total_value = portfolio.total_value
        
        # Assume 60% of portfolio value requires stable funding
        return total_value * 0.6
    
    def _classify_asset(self, bond: Bond) -> str:
        """Classify bond for regulatory purposes"""
        # Simplified classification
        # In production, this would use more sophisticated logic
        
        if bond.issuer_type == "GOVERNMENT":
            return "sovereign_bonds"
        elif bond.issuer_type == "PUBLIC_SECTOR":
            return "public_sector"
        elif bond.issuer_type == "BANK":
            return "banks"
        elif bond.issuer_type == "CORPORATE":
            return "high_grade_corporate_bonds"
        else:
            return "other_assets"
    
    def _update_performance_metrics(self, calculation_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_calculations'] += 1
        
        current_avg = self.performance_metrics['average_calculation_time']
        total_calcs = self.performance_metrics['total_calculations']
        
        self.performance_metrics['average_calculation_time'] = (
            (current_avg * (total_calcs - 1) + calculation_time) / total_calcs
        )
        
        self.performance_metrics['last_calculation'] = datetime.utcnow()
        
        if self.performance_metrics['total_calculations'] % 10 == 0:
            self.logger.info(f"Regulatory capital performance: "
                           f"Avg time: {self.performance_metrics['average_calculation_time']:.2f}s, "
                           f"Total calculations: {self.performance_metrics['total_calculations']}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    async def generate_regulatory_report(self, portfolio: Portfolio, 
                                       report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        start_time = datetime.utcnow()
        
        try:
            # Calculate all regulatory metrics
            capital_req = await self.calculate_regulatory_capital(portfolio)
            lcr_result = await self.calculate_lcr(portfolio)
            nsfr_result = await self.calculate_nsfr(portfolio)
            
            # Generate report
            report = {
                "report_metadata": {
                    "portfolio_id": portfolio.id,
                    "calculation_date": date.today().isoformat(),
                    "basel_framework": self.config.basel_framework.value,
                    "approach": self.config.approach.value,
                    "jurisdiction": self.config.jurisdiction,
                    "report_type": report_type,
                    "generation_timestamp": datetime.utcnow().isoformat()
                },
                "capital_requirements": {
                    "total_regulatory_capital": capital_req.total_capital,
                    "credit_risk_capital": capital_req.credit_risk_capital,
                    "market_risk_capital": capital_req.market_risk_capital,
                    "operational_risk_capital": capital_req.operational_risk_capital,
                    "leverage_ratio": capital_req.leverage_ratio
                },
                "liquidity_metrics": {
                    "lcr_ratio": lcr_result.lcr_ratio,
                    "high_quality_liquid_assets": lcr_result.high_quality_liquid_assets,
                    "net_cash_outflows": lcr_result.net_cash_outflows,
                    "lcr_compliance": lcr_result.lcr_ratio >= 1.0
                },
                "funding_metrics": {
                    "nsfr_ratio": nsfr_result.nsfr_ratio,
                    "available_stable_funding": nsfr_result.available_stable_funding,
                    "required_stable_funding": nsfr_result.required_stable_funding,
                    "nsfr_compliance": nsfr_result.nsfr_ratio >= 1.0
                },
                "portfolio_summary": {
                    "total_value": portfolio.total_value,
                    "total_positions": len(portfolio.positions),
                    "asset_allocation": self._get_asset_allocation(portfolio),
                    "credit_rating_distribution": self._get_rating_distribution(portfolio)
                },
                "compliance_status": {
                    "capital_adequacy": capital_req.total_capital > 0,  # Simplified
                    "liquidity_compliance": lcr_result.lcr_ratio >= 1.0,
                    "funding_compliance": nsfr_result.nsfr_ratio >= 1.0,
                    "overall_compliance": all([
                        capital_req.total_capital > 0,
                        lcr_result.lcr_ratio >= 1.0,
                        nsfr_result.nsfr_ratio >= 1.0
                    ])
                }
            }
            
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Regulatory report generated in {generation_time:.2f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating regulatory report: {e}")
            raise
    
    def _get_asset_allocation(self, portfolio: Portfolio) -> Dict[str, float]:
        """Get asset allocation breakdown"""
        allocation = {}
        
        for position in portfolio.positions:
            asset_type = self._classify_asset(position.bond)
            current_value = allocation.get(asset_type, 0.0)
            allocation[asset_type] = current_value + position.market_value
        
        # Convert to percentages
        total_value = portfolio.total_value
        if total_value > 0:
            allocation = {k: v / total_value * 100 for k, v in allocation.items()}
        
        return allocation
    
    def _get_rating_distribution(self, portfolio: Portfolio) -> Dict[str, float]:
        """Get credit rating distribution"""
        rating_dist = {}
        
        for position in portfolio.positions:
            rating = position.bond.credit_rating
            current_value = rating_dist.get(rating, 0.0)
            rating_dist[rating] = current_value + position.market_value
        
        # Convert to percentages
        total_value = portfolio.total_value
        if total_value > 0:
            rating_dist = {k: v / total_value * 100 for k, v in rating_dist.items()}
        
        return rating_dist
