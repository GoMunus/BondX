"""
Automated Regulatory Capital Engine for Phase D

This module implements comprehensive regulatory compliance:
- Basel III/IV frameworks (Standardized vs Internal Models Approach)
- Liquidity Coverage Ratio (LCR) calculations
- Net Stable Funding Ratio (NSFR) calculations
- SEBI/RBI compliant reporting
- Automated capital requirement calculations
- Performance target: Generate reports in <10s for 100k+ instruments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import joblib
from pathlib import Path
import hashlib
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, Empty

# Financial calculations
from scipy import stats
from scipy.optimize import minimize
import arch

logger = logging.getLogger(__name__)

class BaselFramework(Enum):
    """Basel framework versions"""
    BASEL_II = "basel_ii"
    BASEL_III = "basel_iii"
    BASEL_IV = "basel_iv"

class CapitalApproach(Enum):
    """Capital calculation approaches"""
    STANDARDIZED = "standardized"
    INTERNAL_MODELS = "internal_models"
    FOUNDATION_IRB = "foundation_irb"
    ADVANCED_IRB = "advanced_irb"

class AssetClass(Enum):
    """Asset classes for capital calculations"""
    SOVEREIGN = "sovereign"
    BANKS = "banks"
    CORPORATES = "corporates"
    RETAIL = "retail"
    EQUITY = "equity"
    SECURITIZATION = "securitization"
    OTHER = "other"

class RiskWeightCategory(Enum):
    """Risk weight categories"""
    ZERO = "zero"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class RegulatoryInstrument:
    """Regulatory instrument for capital calculations"""
    instrument_id: str
    asset_class: AssetClass
    risk_weight: float
    maturity: float  # in years
    notional_amount: float
    market_value: float
    credit_rating: str
    issuer_type: str
    country_of_issuer: str
    sector: str
    liquidity_category: str
    collateral_type: str
    guarantor_type: str
    
    # Additional risk factors
    correlation_factor: float = 1.0
    maturity_factor: float = 1.0
    size_factor: float = 1.0
    concentration_factor: float = 1.0

@dataclass
class CapitalRequirements:
    """Capital requirement calculations"""
    credit_risk_capital: float
    market_risk_capital: float
    operational_risk_capital: float
    total_capital: float
    tier_1_capital: float
    tier_2_capital: float
    additional_tier_1: float
    capital_adequacy_ratio: float
    leverage_ratio: float

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for regulatory compliance"""
    lcr_numerator: float
    lcr_denominator: float
    lcr_ratio: float
    nsfr_numerator: float
    nsfr_denominator: float
    nsfr_ratio: float
    net_stable_funding: float
    required_stable_funding: float

@dataclass
class RegulatoryReport:
    """Regulatory compliance report"""
    report_date: datetime
    institution_id: str
    framework_version: str
    capital_requirements: CapitalRequirements
    liquidity_metrics: LiquidityMetrics
    risk_weighted_assets: Dict[str, float]
    concentration_limits: Dict[str, float]
    stress_test_results: Dict[str, float]
    compliance_status: Dict[str, bool]
    generation_time_seconds: float

class BaselCapitalCalculator:
    """Basel framework capital calculator"""
    
    def __init__(self, framework: BaselFramework = BaselFramework.BASEL_IV):
        self.framework = framework
        self.risk_weights = self._initialize_risk_weights()
        self.correlation_factors = self._initialize_correlation_factors()
        
        logger.info(f"Basel Capital Calculator initialized for {framework.value}")
    
    def _initialize_risk_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk weights based on Basel framework"""
        if self.framework == BaselFramework.BASEL_IV:
            return {
                AssetClass.SOVEREIGN.value: {
                    'AAA': 0.0, 'AA': 0.0, 'A': 0.2, 'BBB': 0.5, 'BB': 1.0, 'B': 1.5, 'CCC': 5.0, 'D': 15.0
                },
                AssetClass.BANKS.value: {
                    'AAA': 0.2, 'AA': 0.2, 'A': 0.3, 'BBB': 0.5, 'BB': 1.0, 'B': 1.5, 'CCC': 5.0, 'D': 15.0
                },
                AssetClass.CORPORATES.value: {
                    'AAA': 0.2, 'AA': 0.2, 'A': 0.3, 'BBB': 0.5, 'BB': 1.0, 'B': 1.5, 'CCC': 5.0, 'D': 15.0
                },
                AssetClass.RETAIL.value: {
                    'AAA': 0.75, 'AA': 0.75, 'A': 0.75, 'BBB': 0.75, 'BB': 0.75, 'B': 0.75, 'CCC': 0.75, 'D': 0.75
                },
                AssetClass.EQUITY.value: {
                    'listed': 2.0, 'unlisted': 4.0
                }
            }
        else:
            # Basel III risk weights (simplified)
            return {
                AssetClass.SOVEREIGN.value: {'AAA': 0.0, 'AA': 0.0, 'A': 0.2, 'BBB': 0.5, 'BB': 1.0, 'B': 1.5, 'CCC': 5.0, 'D': 15.0},
                AssetClass.BANKS.value: {'AAA': 0.2, 'AA': 0.2, 'A': 0.3, 'BBB': 0.5, 'BB': 1.0, 'B': 1.5, 'CCC': 5.0, 'D': 15.0},
                AssetClass.CORPORATES.value: {'AAA': 0.2, 'AA': 0.2, 'A': 0.3, 'BBB': 0.5, 'BB': 1.0, 'B': 1.5, 'CCC': 5.0, 'D': 15.0},
                AssetClass.RETAIL.value: {'AAA': 0.75, 'AA': 0.75, 'A': 0.75, 'BBB': 0.75, 'BB': 0.75, 'B': 0.75, 'CCC': 0.75, 'D': 0.75},
                AssetClass.EQUITY.value: {'listed': 2.0, 'unlisted': 4.0}
            }
    
    def _initialize_correlation_factors(self) -> Dict[str, float]:
        """Initialize correlation factors for Basel calculations"""
        return {
            AssetClass.SOVEREIGN.value: 0.0,
            AssetClass.BANKS.value: 0.2,
            AssetClass.CORPORATES.value: 0.24,
            AssetClass.RETAIL.value: 0.16,
            AssetClass.EQUITY.value: 0.0
        }
    
    def calculate_credit_risk_capital(self, 
                                    instruments: List[RegulatoryInstrument],
                                    approach: CapitalApproach = CapitalApproach.STANDARDIZED) -> float:
        """Calculate credit risk capital requirements"""
        
        if approach == CapitalApproach.STANDARDIZED:
            return self._calculate_standardized_credit_risk(instruments)
        elif approach == CapitalApproach.INTERNAL_MODELS:
            return self._calculate_internal_models_credit_risk(instruments)
        else:
            raise ValueError(f"Unsupported capital approach: {approach}")
    
    def _calculate_standardized_credit_risk(self, instruments: List[RegulatoryInstrument]) -> float:
        """Calculate standardized approach credit risk capital"""
        total_capital = 0.0
        
        for instrument in instruments:
            # Get risk weight
            risk_weight = self._get_risk_weight(instrument)
            
            # Calculate exposure at default (EAD)
            ead = instrument.notional_amount
            
            # Apply maturity adjustment for Basel IV
            if self.framework == BaselFramework.BASEL_IV:
                maturity_factor = self._calculate_maturity_factor(instrument.maturity)
                risk_weight *= maturity_factor
            
            # Calculate capital requirement
            capital_requirement = ead * risk_weight * 0.08  # 8% minimum capital ratio
            
            total_capital += capital_requirement
        
        return total_capital
    
    def _calculate_internal_models_credit_risk(self, instruments: List[RegulatoryInstrument]) -> float:
        """Calculate internal models approach credit risk capital"""
        # This would implement the more complex internal models approach
        # For now, return a simplified calculation
        
        total_capital = 0.0
        
        for instrument in instruments:
            # Simplified internal models calculation
            pd = self._estimate_probability_of_default(instrument)
            lgd = self._estimate_loss_given_default(instrument)
            ead = instrument.notional_amount
            maturity = instrument.maturity
            
            # Vasicek model for capital calculation
            r = 0.12  # Asset correlation
            capital_requirement = self._vasicek_capital(pd, lgd, ead, maturity, r)
            
            total_capital += capital_requirement
        
        return total_capital
    
    def _get_risk_weight(self, instrument: RegulatoryInstrument) -> float:
        """Get risk weight for an instrument"""
        asset_class = instrument.asset_class.value
        rating = instrument.credit_rating
        
        if asset_class in self.risk_weights:
            if rating in self.risk_weights[asset_class]:
                return self.risk_weights[asset_class][rating]
            else:
                # Default to highest risk weight
                return max(self.risk_weights[asset_class].values())
        else:
            # Default risk weight for unknown asset class
            return 1.0
    
    def _calculate_maturity_factor(self, maturity: float) -> float:
        """Calculate maturity adjustment factor for Basel IV"""
        if maturity <= 1.0:
            return 1.0
        elif maturity <= 5.0:
            return 1.0 + 0.2 * (maturity - 1.0)
        else:
            return 1.8 + 0.1 * (maturity - 5.0)
    
    def _estimate_probability_of_default(self, instrument: RegulatoryInstrument) -> float:
        """Estimate probability of default based on credit rating"""
        rating_pd_map = {
            'AAA': 0.0001, 'AA': 0.0005, 'A': 0.001, 'BBB': 0.01,
            'BB': 0.05, 'B': 0.15, 'CCC': 0.30, 'D': 1.0
        }
        
        return rating_pd_map.get(instrument.credit_rating, 0.05)
    
    def _estimate_loss_given_default(self, instrument: RegulatoryInstrument) -> float:
        """Estimate loss given default"""
        # Simplified LGD estimation
        if instrument.collateral_type == "secured":
            return 0.45
        elif instrument.collateral_type == "unsecured":
            return 0.75
        else:
            return 0.60
    
    def _vasicek_capital(self, pd: float, lgd: float, ead: float, maturity: float, r: float) -> float:
        """Calculate capital using Vasicek model"""
        # Vasicek asymptotic single risk factor model
        alpha = 0.999  # Confidence level
        
        # Normal inverse function approximation
        k = stats.norm.ppf(alpha)
        pd_norm = stats.norm.ppf(pd)
        
        # Capital requirement
        capital = lgd * ead * (stats.norm.cdf((k + np.sqrt(r) * pd_norm) / np.sqrt(1 - r)) - pd)
        
        return capital

class LiquidityCalculator:
    """Liquidity metrics calculator for regulatory compliance"""
    
    def __init__(self):
        self.lcr_weights = self._initialize_lcr_weights()
        self.nsfr_weights = self._initialize_nsfr_weights()
        
        logger.info("Liquidity Calculator initialized")
    
    def _initialize_lcr_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize LCR weights for different asset categories"""
        return {
            'level_1_assets': {
                'cash': 1.0,
                'central_bank_reserves': 1.0,
                'sovereign_bonds_aaa': 1.0,
                'sovereign_bonds_aa': 1.0
            },
            'level_2a_assets': {
                'sovereign_bonds_a': 0.85,
                'corporate_bonds_aaa': 0.85,
                'covered_bonds_aaa': 0.85
            },
            'level_2b_assets': {
                'corporate_bonds_aa': 0.5,
                'equity_shares': 0.5,
                'residential_mortgage_backed_securities': 0.5
            }
        }
    
    def _initialize_nsfr_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize NSFR weights for different funding sources and uses"""
        return {
            'available_stable_funding': {
                'tier_1_capital': 1.0,
                'tier_2_capital': 1.0,
                'stable_deposits': 0.95,
                'less_stable_deposits': 0.5
            },
            'required_stable_funding': {
                'sovereign_bonds': 0.05,
                'corporate_bonds_aaa': 0.15,
                'corporate_bonds_aa': 0.25,
                'corporate_bonds_a': 0.35,
                'corporate_bonds_bbb': 0.50,
                'equity': 1.0
            }
        }
    
    def calculate_lcr(self, 
                     high_quality_liquid_assets: Dict[str, float],
                     net_cash_outflows: Dict[str, float]) -> LiquidityMetrics:
        """Calculate Liquidity Coverage Ratio"""
        
        # Calculate LCR numerator (HQLA)
        lcr_numerator = 0.0
        for asset_type, amount in high_quality_liquid_assets.items():
            weight = self._get_lcr_weight(asset_type)
            lcr_numerator += amount * weight
        
        # Calculate LCR denominator (net cash outflows)
        lcr_denominator = sum(net_cash_outflows.values())
        
        # Calculate LCR ratio
        lcr_ratio = lcr_numerator / lcr_denominator if lcr_denominator > 0 else float('inf')
        
        return LiquidityMetrics(
            lcr_numerator=lcr_numerator,
            lcr_denominator=lcr_denominator,
            lcr_ratio=lcr_ratio,
            nsfr_numerator=0.0,  # Will be calculated separately
            nsfr_denominator=0.0,
            nsfr_ratio=0.0,
            net_stable_funding=0.0,
            required_stable_funding=0.0
        )
    
    def calculate_nsfr(self, 
                      available_stable_funding: Dict[str, float],
                      required_stable_funding: Dict[str, float]) -> LiquidityMetrics:
        """Calculate Net Stable Funding Ratio"""
        
        # Calculate ASF (Available Stable Funding)
        nsfr_numerator = 0.0
        for funding_type, amount in available_stable_funding.items():
            weight = self._get_nsfr_asf_weight(funding_type)
            nsfr_numerator += amount * weight
        
        # Calculate RSF (Required Stable Funding)
        nsfr_denominator = 0.0
        for asset_type, amount in required_stable_funding.items():
            weight = self._get_nsfr_rsf_weight(asset_type)
            nsfr_denominator += amount * weight
        
        # Calculate NSFR ratio
        nsfr_ratio = nsfr_numerator / nsfr_denominator if nsfr_denominator > 0 else float('inf')
        
        return LiquidityMetrics(
            lcr_numerator=0.0,  # Will be calculated separately
            lcr_denominator=0.0,
            lcr_ratio=0.0,
            nsfr_numerator=nsfr_numerator,
            nsfr_denominator=nsfr_denominator,
            nsfr_ratio=nsfr_ratio,
            net_stable_funding=nsfr_numerator,
            required_stable_funding=nsfr_denominator
        )
    
    def _get_lcr_weight(self, asset_type: str) -> float:
        """Get LCR weight for asset type"""
        for level, assets in self.lcr_weights.items():
            if asset_type in assets:
                return assets[asset_type]
        
        # Default weight for unknown asset types
        return 0.0
    
    def _get_nsfr_asf_weight(self, funding_type: str) -> float:
        """Get NSFR ASF weight for funding type"""
        if 'available_stable_funding' in self.nsfr_weights:
            return self.nsfr_weights['available_stable_funding'].get(funding_type, 0.0)
        return 0.0
    
    def _get_nsfr_rsf_weight(self, asset_type: str) -> float:
        """Get NSFR RSF weight for asset type"""
        if 'required_stable_funding' in self.nsfr_weights:
            return self.nsfr_weights['required_stable_funding'].get(asset_type, 1.0)
        return 1.0

class RegulatoryCapitalEngine:
    """Main regulatory capital engine integrating all compliance frameworks"""
    
    def __init__(self, 
                 basel_framework: BaselFramework = BaselFramework.BASEL_IV,
                 capital_approach: CapitalApproach = CapitalApproach.STANDARDIZED):
        
        self.basel_framework = basel_framework
        self.capital_approach = capital_approach
        
        # Initialize components
        self.basel_calculator = BaselCapitalCalculator(basel_framework)
        self.liquidity_calculator = LiquidityCalculator()
        
        # Performance tracking
        self.performance_metrics = {}
        self.report_history = []
        
        logger.info(f"Regulatory Capital Engine initialized for {basel_framework.value}")
    
    def calculate_regulatory_capital(self, 
                                   instruments: List[RegulatoryInstrument],
                                   liquidity_data: Dict[str, Any]) -> RegulatoryReport:
        """Calculate comprehensive regulatory capital requirements"""
        
        start_time = time.time()
        
        try:
            # Calculate credit risk capital
            credit_risk_capital = self.basel_calculator.calculate_credit_risk_capital(
                instruments, self.capital_approach
            )
            
            # Calculate market risk capital (simplified)
            market_risk_capital = self._calculate_market_risk_capital(instruments)
            
            # Calculate operational risk capital (simplified)
            operational_risk_capital = self._calculate_operational_risk_capital(instruments)
            
            # Calculate total capital
            total_capital = credit_risk_capital + market_risk_capital + operational_risk_capital
            
            # Calculate capital adequacy ratio
            total_assets = sum(instrument.market_value for instrument in instruments)
            capital_adequacy_ratio = total_capital / total_assets if total_assets > 0 else 0.0
            
            # Create capital requirements object
            capital_requirements = CapitalRequirements(
                credit_risk_capital=credit_risk_capital,
                market_risk_capital=market_risk_capital,
                operational_risk_capital=operational_risk_capital,
                total_capital=total_capital,
                tier_1_capital=total_capital * 0.6,  # Simplified
                tier_2_capital=total_capital * 0.4,  # Simplified
                additional_tier_1=total_capital * 0.1,  # Simplified
                capital_adequacy_ratio=capital_adequacy_ratio,
                leverage_ratio=total_capital / total_assets if total_assets > 0 else 0.0
            )
            
            # Calculate liquidity metrics
            lcr_metrics = self.liquidity_calculator.calculate_lcr(
                liquidity_data.get('high_quality_liquid_assets', {}),
                liquidity_data.get('net_cash_outflows', {})
            )
            
            nsfr_metrics = self.liquidity_calculator.calculate_nsfr(
                liquidity_data.get('available_stable_funding', {}),
                liquidity_data.get('required_stable_funding', {})
            )
            
            # Combine liquidity metrics
            liquidity_metrics = LiquidityMetrics(
                lcr_numerator=lcr_metrics.lcr_numerator,
                lcr_denominator=lcr_metrics.lcr_denominator,
                lcr_ratio=lcr_metrics.lcr_ratio,
                nsfr_numerator=nsfr_metrics.nsfr_numerator,
                nsfr_denominator=nsfr_metrics.nsfr_denominator,
                nsfr_ratio=nsfr_metrics.nsfr_ratio,
                net_stable_funding=nsfr_metrics.net_stable_funding,
                required_stable_funding=nsfr_metrics.required_stable_funding
            )
            
            # Calculate risk-weighted assets
            risk_weighted_assets = self._calculate_risk_weighted_assets(instruments)
            
            # Calculate concentration limits
            concentration_limits = self._calculate_concentration_limits(instruments)
            
            # Run stress tests
            stress_test_results = self._run_stress_tests(instruments, capital_requirements)
            
            # Check compliance status
            compliance_status = self._check_compliance_status(
                capital_requirements, liquidity_metrics, stress_test_results
            )
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Create regulatory report
            report = RegulatoryReport(
                report_date=datetime.now(),
                institution_id=self._generate_institution_id(),
                framework_version=self.basel_framework.value,
                capital_requirements=capital_requirements,
                liquidity_metrics=liquidity_metrics,
                risk_weighted_assets=risk_weighted_assets,
                concentration_limits=concentration_limits,
                stress_test_results=stress_test_results,
                compliance_status=compliance_status,
                generation_time_seconds=generation_time
            )
            
            # Store report history
            self.report_history.append(report)
            
            # Update performance metrics
            self._update_performance_metrics(generation_time)
            
            logger.info(f"Regulatory capital report generated in {generation_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Regulatory capital calculation failed: {e}")
            raise
    
    def _calculate_market_risk_capital(self, instruments: List[RegulatoryInstrument]) -> float:
        """Calculate market risk capital (simplified)"""
        # Simplified market risk calculation
        total_market_value = sum(instrument.market_value for instrument in instruments)
        
        # Assume 8% market risk capital requirement
        market_risk_capital = total_market_value * 0.08
        
        return market_risk_capital
    
    def _calculate_operational_risk_capital(self, instruments: List[RegulatoryInstrument]) -> float:
        """Calculate operational risk capital (simplified)"""
        # Simplified operational risk calculation using Basic Indicator Approach
        total_assets = sum(instrument.market_value for instrument in instruments)
        
        # Assume 15% of gross income as operational risk capital
        # For simplicity, use 1% of total assets as proxy for gross income
        gross_income = total_assets * 0.01
        operational_risk_capital = gross_income * 0.15
        
        return operational_risk_capital
    
    def _calculate_risk_weighted_assets(self, instruments: List[RegulatoryInstrument]) -> Dict[str, float]:
        """Calculate risk-weighted assets by category"""
        rwa_by_category = {}
        
        for instrument in instruments:
            asset_class = instrument.asset_class.value
            risk_weight = self.basel_calculator._get_risk_weight(instrument)
            rwa = instrument.notional_amount * risk_weight
            
            if asset_class not in rwa_by_category:
                rwa_by_category[asset_class] = 0.0
            
            rwa_by_category[asset_class] += rwa
        
        return rwa_by_category
    
    def _calculate_concentration_limits(self, instruments: List[RegulatoryInstrument]) -> Dict[str, float]:
        """Calculate concentration limits and exposures"""
        concentration_metrics = {}
        
        # Calculate sector concentration
        sector_exposures = {}
        for instrument in instruments:
            sector = instrument.sector
            if sector not in sector_exposures:
                sector_exposures[sector] = 0.0
            sector_exposures[sector] += instrument.market_value
        
        # Calculate issuer concentration
        issuer_exposures = {}
        for instrument in instruments:
            issuer = instrument.instrument_id.split('_')[0]  # Simplified issuer extraction
            if issuer not in issuer_exposures:
                issuer_exposures[issuer] = 0.0
            issuer_exposures[issuer] += instrument.market_value
        
        # Calculate total portfolio value
        total_portfolio_value = sum(instrument.market_value for instrument in instruments)
        
        # Calculate concentration ratios
        concentration_metrics['sector_concentration'] = {
            sector: exposure / total_portfolio_value 
            for sector, exposure in sector_exposures.items()
        }
        
        concentration_metrics['issuer_concentration'] = {
            issuer: exposure / total_portfolio_value 
            for issuer, exposure in issuer_exposures.items()
        }
        
        return concentration_metrics
    
    def _run_stress_tests(self, 
                          instruments: List[RegulatoryInstrument],
                          capital_requirements: CapitalRequirements) -> Dict[str, float]:
        """Run regulatory stress tests"""
        stress_results = {}
        
        # Interest rate stress test
        interest_rate_stress = self._interest_rate_stress_test(instruments)
        stress_results['interest_rate_stress'] = interest_rate_stress
        
        # Credit spread stress test
        credit_spread_stress = self._credit_spread_stress_test(instruments)
        stress_results['credit_spread_stress'] = credit_spread_stress
        
        # Liquidity stress test
        liquidity_stress = self._liquidity_stress_test(instruments)
        stress_results['liquidity_stress'] = liquidity_stress
        
        # Market volatility stress test
        volatility_stress = self._volatility_stress_test(instruments)
        stress_results['volatility_stress'] = volatility_stress
        
        return stress_results
    
    def _interest_rate_stress_test(self, instruments: List[RegulatoryInstrument]) -> float:
        """Interest rate stress test"""
        # Parallel shift of 200 basis points
        rate_shock = 0.02
        
        total_impact = 0.0
        for instrument in instruments:
            # Duration-based impact
            impact = -instrument.duration * instrument.market_value * rate_shock
            total_impact += impact
        
        return total_impact
    
    def _credit_spread_stress_test(self, instruments: List[RegulatoryInstrument]) -> float:
        """Credit spread stress test"""
        # Credit spread widening of 100 basis points
        spread_shock = 0.01
        
        total_impact = 0.0
        for instrument in instruments:
            # Duration-based impact
            impact = -instrument.duration * instrument.market_value * spread_shock
            total_impact += impact
        
        return total_impact
    
    def _liquidity_stress_test(self, instruments: List[RegulatoryInstrument]) -> float:
        """Liquidity stress test"""
        # Liquidity score deterioration
        total_impact = 0.0
        for instrument in instruments:
            # Assume 50% deterioration in liquidity score
            liquidity_impact = instrument.market_value * (1 - instrument.liquidity_score) * 0.5
            total_impact += liquidity_impact
        
        return total_impact
    
    def _volatility_stress_test(self, instruments: List[RegulatoryInstrument]) -> float:
        """Market volatility stress test"""
        # Volatility increase impact on option-like instruments
        total_impact = 0.0
        for instrument in instruments:
            # Simplified volatility impact
            if instrument.convexity > 0:
                volatility_impact = instrument.convexity * instrument.market_value * 0.5
                total_impact += volatility_impact
        
        return total_impact
    
    def _check_compliance_status(self, 
                                capital_requirements: CapitalRequirements,
                                liquidity_metrics: LiquidityMetrics,
                                stress_test_results: Dict[str, float]) -> Dict[str, bool]:
        """Check regulatory compliance status"""
        compliance_status = {}
        
        # Capital adequacy ratio (minimum 8%)
        compliance_status['capital_adequacy'] = capital_requirements.capital_adequacy_ratio >= 0.08
        
        # LCR ratio (minimum 100%)
        compliance_status['lcr_compliance'] = liquidity_metrics.lcr_ratio >= 1.0
        
        # NSFR ratio (minimum 100%)
        compliance_status['nsfr_compliance'] = liquidity_metrics.nsfr_ratio >= 1.0
        
        # Leverage ratio (minimum 3%)
        compliance_status['leverage_ratio'] = capital_requirements.leverage_ratio >= 0.03
        
        # Stress test compliance (capital should remain above minimum after stress)
        total_stress_impact = sum(stress_test_results.values())
        compliance_status['stress_test_compliance'] = (
            capital_requirements.total_capital + total_stress_impact >= 
            capital_requirements.total_capital * 0.08
        )
        
        return compliance_status
    
    def _generate_institution_id(self) -> str:
        """Generate unique institution identifier"""
        return f"institution_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _update_performance_metrics(self, generation_time: float):
        """Update performance tracking metrics"""
        if 'generation_times' not in self.performance_metrics:
            self.performance_metrics['generation_times'] = []
        
        self.performance_metrics['generation_times'].append(generation_time)
        
        # Keep only last 1000 measurements
        if len(self.performance_metrics['generation_times']) > 1000:
            self.performance_metrics['generation_times'] = self.performance_metrics['generation_times'][-1000:]
        
        # Update statistics
        times = self.performance_metrics['generation_times']
        self.performance_metrics['avg_generation_time'] = np.mean(times)
        self.performance_metrics['p95_generation_time'] = np.percentile(times, 95)
        self.performance_metrics['p99_generation_time'] = np.percentile(times, 99)
        self.performance_metrics['min_generation_time'] = np.min(times)
        self.performance_metrics['max_generation_time'] = np.max(times)
    
    def generate_sebi_report(self, regulatory_report: RegulatoryReport) -> Dict[str, Any]:
        """Generate SEBI-compliant regulatory report"""
        sebi_report = {
            'report_type': 'SEBI_Regulatory_Report',
            'report_date': regulatory_report.report_date.isoformat(),
            'institution_id': regulatory_report.institution_id,
            'capital_adequacy_ratio': regulatory_report.capital_requirements.capital_adequacy_ratio,
            'lcr_ratio': regulatory_report.liquidity_metrics.lcr_ratio,
            'nsfr_ratio': regulatory_report.liquidity_metrics.nsfr_ratio,
            'risk_weighted_assets': regulatory_report.risk_weighted_assets,
            'concentration_limits': regulatory_report.concentration_limits,
            'compliance_status': regulatory_report.compliance_status,
            'stress_test_summary': {
                'total_impact': sum(regulatory_report.stress_test_results.values()),
                'scenarios_tested': len(regulatory_report.stress_test_results)
            }
        }
        
        return sebi_report
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'basel_framework': self.basel_framework.value,
            'capital_approach': self.capital_approach.value,
            'reports_generated': len(self.report_history),
            'performance_metrics': self.performance_metrics,
            'last_report_date': self.report_history[-1].report_date.isoformat() if self.report_history else None
        }
