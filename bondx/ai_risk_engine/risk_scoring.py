"""
Multi-Layered Risk Scoring Architecture

This module implements a comprehensive risk assessment framework that combines
quantitative and qualitative factors to produce nuanced risk scores for bonds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class RatingAgency(Enum):
    """Rating agencies for Indian bonds"""
    CRISIL = "CRISIL"
    ICRA = "ICRA"
    CARE = "CARE"
    INDIA_RATINGS = "INDIA_RATINGS"
    BRICKWORK = "BRICKWORK"
    INFOMERICS = "INFOMERICS"

class RiskFactor(Enum):
    """Risk factors for bond analysis"""
    CREDIT_RISK = "credit_risk"
    INTEREST_RATE_RISK = "interest_rate_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    ESG_RISK = "esg_risk"
    OPERATIONAL_RISK = "operational_risk"

@dataclass
class RiskScore:
    """Risk score data structure"""
    overall_score: float
    factor_scores: Dict[RiskFactor, float]
    confidence_interval: Tuple[float, float]
    last_updated: pd.Timestamp
    methodology_version: str

@dataclass
class CreditRiskMetrics:
    """Credit risk metrics"""
    default_probability: float
    loss_given_default: float
    expected_loss: float
    distance_to_default: float
    rating_outlook: str
    rating_watch: str

class RiskScoringEngine:
    """
    Multi-layered risk assessment framework for bond investments
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rating_mappings = self._initialize_rating_mappings()
        self.risk_weights = self._initialize_risk_weights()
        self.scaler = StandardScaler()
        self.credit_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def _initialize_rating_mappings(self) -> Dict:
        """Initialize rating agency mappings with numerical scores"""
        mappings = {}
        
        # CRISIL rating scale (AAA to D)
        mappings[RatingAgency.CRISIL] = {
            'AAA': 1.0, 'AA+': 2.0, 'AA': 3.0, 'AA-': 4.0,
            'A+': 5.0, 'A': 6.0, 'A-': 7.0,
            'BBB+': 8.0, 'BBB': 9.0, 'BBB-': 10.0,
            'BB+': 11.0, 'BB': 12.0, 'BB-': 13.0,
            'B+': 14.0, 'B': 15.0, 'B-': 16.0,
            'C': 17.0, 'D': 18.0
        }
        
        # ICRA rating scale
        mappings[RatingAgency.ICRA] = {
            'AAA': 1.0, 'AA+': 2.0, 'AA': 3.0, 'AA-': 4.0,
            'A+': 5.0, 'A': 6.0, 'A-': 7.0,
            'BBB+': 8.0, 'BBB': 9.0, 'BBB-': 10.0,
            'BB+': 11.0, 'BB': 12.0, 'BB-': 13.0,
            'B+': 14.0, 'B': 15.0, 'B-': 16.0,
            'C': 17.0, 'D': 18.0
        }
        
        # CARE rating scale
        mappings[RatingAgency.CARE] = {
            'CARE AAA': 1.0, 'CARE AA+': 2.0, 'CARE AA': 3.0, 'CARE AA-': 4.0,
            'CARE A+': 5.0, 'CARE A': 6.0, 'CARE A-': 7.0,
            'CARE BBB+': 8.0, 'CARE BBB': 9.0, 'CARE BBB-': 10.0,
            'CARE BB+': 11.0, 'CARE BB': 12.0, 'CARE BB-': 13.0,
            'CARE B+': 14.0, 'CARE B': 15.0, 'CARE B-': 16.0,
            'CARE C': 17.0, 'CARE D': 18.0
        }
        
        return mappings
    
    def _initialize_risk_weights(self) -> Dict[RiskFactor, float]:
        """Initialize risk factor weights"""
        return {
            RiskFactor.CREDIT_RISK: 0.35,
            RiskFactor.INTEREST_RATE_RISK: 0.25,
            RiskFactor.LIQUIDITY_RISK: 0.20,
            RiskFactor.CONCENTRATION_RISK: 0.10,
            RiskFactor.ESG_RISK: 0.08,
            RiskFactor.OPERATIONAL_RISK: 0.02
        }
    
    def calculate_credit_risk_score(
        self,
        rating: str,
        rating_agency: RatingAgency,
        outlook: str = "stable",
        financial_ratios: Dict = None,
        macroeconomic_factors: Dict = None
    ) -> float:
        """
        Calculate comprehensive credit risk score
        
        Args:
            rating: Bond rating (e.g., 'AAA', 'AA+')
            rating_agency: Rating agency
            outlook: Rating outlook (stable, positive, negative)
            financial_ratios: Key financial ratios
            macroeconomic_factors: Macroeconomic indicators
            
        Returns:
            Credit risk score (0-100, higher = higher risk)
        """
        try:
            # Base score from rating
            base_score = self._get_base_rating_score(rating, rating_agency)
            
            # Outlook adjustment
            outlook_adjustment = self._calculate_outlook_adjustment(outlook)
            
            # Financial ratios adjustment
            financial_adjustment = self._calculate_financial_adjustment(financial_ratios)
            
            # Macroeconomic adjustment
            macro_adjustment = self._calculate_macroeconomic_adjustment(macroeconomic_factors)
            
            # Calculate final credit risk score
            credit_score = base_score + outlook_adjustment + financial_adjustment + macro_adjustment
            
            # Normalize to 0-100 scale
            credit_score = np.clip(credit_score, 0, 100)
            
            return credit_score
            
        except Exception as e:
            logger.error(f"Error calculating credit risk score: {e}")
            return 50.0  # Default moderate risk score
    
    def _get_base_rating_score(self, rating: str, agency: RatingAgency) -> float:
        """Get base numerical score from rating"""
        if agency in self.rating_mappings and rating in self.rating_mappings[agency]:
            base_score = self.rating_mappings[agency][rating]
            # Convert to 0-100 scale (1 = lowest risk, 18 = highest risk)
            return (base_score - 1) * (100 / 17)
        return 50.0  # Default score
    
    def _calculate_outlook_adjustment(self, outlook: str) -> float:
        """Calculate rating outlook adjustment"""
        outlook_adjustments = {
            "positive": -5.0,
            "stable": 0.0,
            "negative": 5.0,
            "developing": 2.5,
            "on_watch": 3.0
        }
        return outlook_adjustments.get(outlook.lower(), 0.0)
    
    def _calculate_financial_adjustment(self, ratios: Dict) -> float:
        """Calculate adjustment based on financial ratios"""
        if not ratios:
            return 0.0
        
        adjustment = 0.0
        
        # Debt-to-EBITDA ratio
        if 'debt_to_ebitda' in ratios:
            debt_ebitda = ratios['debt_to_ebitda']
            if debt_ebitda > 6.0:
                adjustment += 10.0
            elif debt_ebitda > 4.0:
                adjustment += 5.0
            elif debt_ebitda < 2.0:
                adjustment -= 5.0
        
        # Interest coverage ratio
        if 'interest_coverage' in ratios:
            coverage = ratios['interest_coverage']
            if coverage < 1.5:
                adjustment += 15.0
            elif coverage < 2.5:
                adjustment += 8.0
            elif coverage > 5.0:
                adjustment -= 5.0
        
        # Current ratio
        if 'current_ratio' in ratios:
            current = ratios['current_ratio']
            if current < 1.0:
                adjustment += 10.0
            elif current > 2.0:
                adjustment -= 3.0
        
        return adjustment
    
    def _calculate_macroeconomic_adjustment(self, factors: Dict) -> float:
        """Calculate adjustment based on macroeconomic factors"""
        if not factors:
            return 0.0
        
        adjustment = 0.0
        
        # GDP growth rate
        if 'gdp_growth' in factors:
            gdp_growth = factors['gdp_growth']
            if gdp_growth < 4.0:
                adjustment += 8.0
            elif gdp_growth > 7.0:
                adjustment -= 3.0
        
        # Inflation rate
        if 'inflation_rate' in factors:
            inflation = factors['inflation_rate']
            if inflation > 6.0:
                adjustment += 5.0
            elif inflation < 3.0:
                adjustment -= 2.0
        
        # RBI repo rate
        if 'repo_rate' in factors:
            repo_rate = factors['repo_rate']
            if repo_rate > 7.0:
                adjustment += 3.0
            elif repo_rate < 4.0:
                adjustment -= 2.0
        
        return adjustment
    
    def calculate_interest_rate_risk_score(
        self,
        modified_duration: float,
        convexity: float,
        yield_curve_slope: float,
        embedded_options: bool = False
    ) -> float:
        """
        Calculate interest rate risk score
        
        Args:
            modified_duration: Modified duration of the bond
            convexity: Convexity of the bond
            yield_curve_slope: Current yield curve slope
            embedded_options: Whether bond has embedded options
            
        Returns:
            Interest rate risk score (0-100, higher = higher risk)
        """
        try:
            # Base score from duration
            duration_score = min(modified_duration * 10, 50)  # Cap at 50
            
            # Convexity adjustment
            convexity_score = min(abs(convexity) * 0.1, 20)  # Cap at 20
            
            # Yield curve slope adjustment
            slope_score = 0.0
            if abs(yield_curve_slope) > 0.02:  # 200 bps slope
                slope_score = min(abs(yield_curve_slope) * 100, 15)
            
            # Embedded options penalty
            options_penalty = 10.0 if embedded_options else 0.0
            
            # Calculate total interest rate risk score
            total_score = duration_score + convexity_score + slope_score + options_penalty
            
            return np.clip(total_score, 0, 100)
            
        except Exception as e:
            logger.error(f"Error calculating interest rate risk score: {e}")
            return 25.0  # Default moderate risk score
    
    def calculate_liquidity_risk_score(
        self,
        bid_ask_spread: float,
        trading_volume: float,
        issue_size: float,
        days_to_maturity: int,
        sector_liquidity: str = "medium"
    ) -> float:
        """
        Calculate liquidity risk score
        
        Args:
            bid_ask_spread: Bid-ask spread in basis points
            trading_volume: Average daily trading volume
            issue_size: Total issue size in crores
            days_to_maturity: Days remaining to maturity
            sector_liquidity: Sector liquidity classification
            
        Returns:
            Liquidity risk score (0-100, higher = higher risk)
        """
        try:
            # Spread-based score
            spread_score = min(bid_ask_spread * 0.5, 30)  # Cap at 30
            
            # Volume-based score
            volume_score = 0.0
            if trading_volume < 10:  # Less than 10 crores daily volume
                volume_score = 25.0
            elif trading_volume < 50:
                volume_score = 15.0
            elif trading_volume < 100:
                volume_score = 8.0
            
            # Issue size score
            size_score = 0.0
            if issue_size < 100:  # Less than 100 crores
                size_score = 20.0
            elif issue_size < 500:
                size_score = 10.0
            
            # Maturity score
            maturity_score = 0.0
            if days_to_maturity > 1825:  # More than 5 years
                maturity_score = 15.0
            elif days_to_maturity > 1095:  # More than 3 years
                maturity_score = 8.0
            
            # Sector liquidity score
            sector_scores = {
                "high": 0.0,
                "medium": 5.0,
                "low": 15.0
            }
            sector_score = sector_scores.get(sector_liquidity.lower(), 5.0)
            
            # Calculate total liquidity risk score
            total_score = spread_score + volume_score + size_score + maturity_score + sector_score
            
            return np.clip(total_score, 0, 100)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk score: {e}")
            return 20.0  # Default moderate risk score
    
    def calculate_concentration_risk_score(
        self,
        sector_exposure: Dict[str, float],
        issuer_exposure: Dict[str, float],
        rating_exposure: Dict[str, float],
        maturity_exposure: Dict[str, float]
    ) -> float:
        """
        Calculate concentration risk score
        
        Args:
            sector_exposure: Sector-wise exposure percentages
            issuer_exposure: Issuer-wise exposure percentages
            rating_exposure: Rating-wise exposure percentages
            maturity_exposure: Maturity-wise exposure percentages
            
        Returns:
            Concentration risk score (0-100, higher = higher risk)
        """
        try:
            # Herfindahl-Hirschman Index for sector concentration
            sector_hhi = sum(exp ** 2 for exp in sector_exposure.values())
            sector_score = min(sector_hhi * 100, 30)
            
            # Issuer concentration score
            issuer_hhi = sum(exp ** 2 for exp in issuer_exposure.values())
            issuer_score = min(issuer_hhi * 100, 25)
            
            # Rating concentration score
            rating_score = 0.0
            if any(exp > 0.3 for exp in rating_exposure.values()):  # >30% in any rating
                rating_score = 20.0
            elif any(exp > 0.2 for exp in rating_exposure.values()):  # >20% in any rating
                rating_score = 15.0
            
            # Maturity concentration score
            maturity_score = 0.0
            if any(exp > 0.4 for exp in maturity_exposure.values()):  # >40% in any maturity
                maturity_score = 15.0
            elif any(exp > 0.25 for exp in maturity_exposure.values()):  # >25% in any maturity
                maturity_score = 10.0
            
            # Calculate total concentration risk score
            total_score = sector_score + issuer_score + rating_score + maturity_score
            
            return np.clip(total_score, 0, 100)
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk score: {e}")
            return 15.0  # Default moderate risk score
    
    def calculate_esg_risk_score(
        self,
        environmental_score: float,
        social_score: float,
        governance_score: float,
        regulatory_risk: str = "medium",
        sustainability_trends: str = "stable"
    ) -> float:
        """
        Calculate ESG risk score
        
        Args:
            environmental_score: Environmental risk score (0-100)
            social_score: Social risk score (0-100)
            governance_score: Governance risk score (0-100)
            regulatory_risk: Regulatory risk classification
            sustainability_trends: Sustainability trend classification
            
        Returns:
            ESG risk score (0-100, higher = higher risk)
        """
        try:
            # Weighted average of ESG scores
            esg_weights = {"E": 0.4, "S": 0.3, "G": 0.3}
            weighted_esg = (
                environmental_score * esg_weights["E"] +
                social_score * esg_weights["S"] +
                governance_score * esg_weights["G"]
            )
            
            # Regulatory risk adjustment
            regulatory_adjustments = {
                "low": -5.0,
                "medium": 0.0,
                "high": 10.0
            }
            regulatory_adjustment = regulatory_adjustments.get(regulatory_risk.lower(), 0.0)
            
            # Sustainability trends adjustment
            trend_adjustments = {
                "improving": -8.0,
                "stable": 0.0,
                "deteriorating": 12.0
            }
            trend_adjustment = trend_adjustments.get(sustainability_trends.lower(), 0.0)
            
            # Calculate final ESG risk score
            total_score = weighted_esg + regulatory_adjustment + trend_adjustment
            
            return np.clip(total_score, 0, 100)
            
        except Exception as e:
            logger.error(f"Error calculating ESG risk score: {e}")
            return 30.0  # Default moderate risk score
    
    def calculate_overall_risk_score(
        self,
        credit_score: float,
        interest_rate_score: float,
        liquidity_score: float,
        concentration_score: float,
        esg_score: float,
        operational_score: float = 10.0
    ) -> RiskScore:
        """
        Calculate overall risk score combining all risk factors
        
        Args:
            credit_score: Credit risk score (0-100)
            interest_rate_score: Interest rate risk score (0-100)
            liquidity_score: Liquidity risk score (0-100)
            concentration_score: Concentration risk score (0-100)
            esg_score: ESG risk score (0-100)
            operational_score: Operational risk score (0-100)
            
        Returns:
            Comprehensive risk score object
        """
        try:
            # Calculate weighted average
            factor_scores = {
                RiskFactor.CREDIT_RISK: credit_score,
                RiskFactor.INTEREST_RATE_RISK: interest_rate_score,
                RiskFactor.LIQUIDITY_RISK: liquidity_score,
                RiskFactor.CONCENTRATION_RISK: concentration_score,
                RiskFactor.ESG_RISK: esg_score,
                RiskFactor.OPERATIONAL_RISK: operational_score
            }
            
            overall_score = sum(
                score * self.risk_weights[factor]
                for factor, score in factor_scores.items()
            )
            
            # Calculate confidence interval (simplified)
            confidence_interval = (
                max(0, overall_score - 5),
                min(100, overall_score + 5)
            )
            
            return RiskScore(
                overall_score=overall_score,
                factor_scores=factor_scores,
                confidence_interval=confidence_interval,
                last_updated=pd.Timestamp.now(),
                methodology_version="1.0.0"
            )
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            # Return default risk score
            return RiskScore(
                overall_score=50.0,
                factor_scores={
                    RiskFactor.CREDIT_RISK: 50.0,
                    RiskFactor.INTEREST_RATE_RISK: 50.0,
                    RiskFactor.LIQUIDITY_RISK: 50.0,
                    RiskFactor.CONCENTRATION_RISK: 50.0,
                    RiskFactor.ESG_RISK: 50.0,
                    RiskFactor.OPERATIONAL_RISK: 50.0
                },
                confidence_interval=(45.0, 55.0),
                last_updated=pd.Timestamp.now(),
                methodology_version="1.0.0"
            )
    
    def stress_test_portfolio(
        self,
        portfolio_risk_scores: List[RiskScore],
        stress_scenarios: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Perform stress testing on portfolio risk scores
        
        Args:
            portfolio_risk_scores: List of risk scores for portfolio bonds
            stress_scenarios: Dictionary of stress scenarios and their parameters
            
        Returns:
            Stress test results for each scenario
        """
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            try:
                stressed_scores = []
                
                for risk_score in portfolio_risk_scores:
                    # Apply stress scenario adjustments
                    stressed_score = self._apply_stress_scenario(
                        risk_score, scenario_params
                    )
                    stressed_scores.append(stressed_score)
                
                # Calculate portfolio-level metrics
                avg_stressed_score = np.mean([s.overall_score for s in stressed_scores])
                max_stressed_score = np.max([s.overall_score for s in stressed_scores])
                score_increase = avg_stressed_score - np.mean([s.overall_score for s in portfolio_risk_scores])
                
                results[scenario_name] = {
                    "average_stressed_score": avg_stressed_score,
                    "maximum_stressed_score": max_stressed_score,
                    "average_score_increase": score_increase,
                    "bonds_above_threshold": sum(1 for s in stressed_scores if s.overall_score > 70),
                    "scenario_parameters": scenario_params
                }
                
            except Exception as e:
                logger.error(f"Error in stress test scenario {scenario_name}: {e}")
                results[scenario_name] = {"error": str(e)}
        
        return results
    
    def _apply_stress_scenario(
        self,
        risk_score: RiskScore,
        scenario_params: Dict
    ) -> RiskScore:
        """Apply stress scenario to individual risk score"""
        # Create a copy of the risk score
        stressed_score = RiskScore(
            overall_score=risk_score.overall_score,
            factor_scores=risk_score.factor_scores.copy(),
            confidence_interval=risk_score.confidence_interval,
            last_updated=risk_score.last_updated,
            methodology_version=risk_score.methodology_version
        )
        
        # Apply credit stress
        if "credit_downgrade" in scenario_params:
            downgrade = scenario_params["credit_downgrade"]
            stressed_score.factor_scores[RiskFactor.CREDIT_RISK] = min(
                100, stressed_score.factor_scores[RiskFactor.CREDIT_RISK] + downgrade
            )
        
        # Apply interest rate stress
        if "rate_shock" in scenario_params:
            rate_shock = scenario_params["rate_shock"]
            stressed_score.factor_scores[RiskFactor.INTEREST_RATE_RISK] = min(
                100, stressed_score.factor_scores[RiskFactor.INTEREST_RATE_RISK] + rate_shock
            )
        
        # Apply liquidity stress
        if "liquidity_crunch" in scenario_params:
            liquidity_stress = scenario_params["liquidity_crunch"]
            stressed_score.factor_scores[RiskFactor.LIQUIDITY_RISK] = min(
                100, stressed_score.factor_scores[RiskFactor.LIQUIDITY_RISK] + liquidity_stress
            )
        
        # Recalculate overall score
        stressed_score.overall_score = sum(
            score * self.risk_weights[factor]
            for factor, score in stressed_score.factor_scores.items()
        )
        
        return stressed_score
