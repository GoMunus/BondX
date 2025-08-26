"""
Stress Testing Engine

This module provides comprehensive stress testing capabilities including:
- Scenario definition and application
- Shock application to various dimensions
- Before/after comparison and diff generation
- Portfolio impact analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

class ShockType(Enum):
    """Types of shocks that can be applied"""
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    MULTIPLIER = "multiplier"
    RATING_DOWNGRADE = "rating_downgrade"
    SECTOR_SHOCK = "sector_shock"

class ShockDimension(Enum):
    """Dimensions that can be shocked"""
    LIQUIDITY_INDEX = "liquidity_index"
    SPREAD_BPS = "spread_bps"
    BID_ASK_SPREAD = "bid_ask_spread"
    VOLUME_TRADED = "volume_traded"
    ISSUE_SIZE = "issue_size"
    CREDIT_RATING = "credit_rating"
    YIELD_TO_MATURITY = "yield_to_maturity"

@dataclass
class ShockScenario:
    """Definition of a shock scenario"""
    name: str
    description: str
    shocks: List[Dict[str, Any]]
    probability: float = 0.01
    severity: str = "moderate"
    tags: List[str] = field(default_factory=list)

@dataclass
class ShockResult:
    """Result of applying a shock"""
    scenario_name: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    impact_summary: Dict[str, Any]
    affected_issuers: List[str]
    risk_flags: List[str]

class StressTestingEngine:
    """Engine for stress testing bond portfolios"""
    
    def __init__(self, seed: int = 42):
        """Initialize the stress testing engine"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        
        # Rating downgrade mappings
        self.rating_downgrades = {
            'AAA': 'AA+', 'AA+': 'AA', 'AA': 'AA-', 'AA-': 'A+',
            'A+': 'A', 'A': 'A-', 'A-': 'BBB+', 'BBB+': 'BBB',
            'BBB': 'BBB-', 'BBB-': 'BB+', 'BB+': 'BB', 'BB': 'BB-',
            'BB-': 'B+', 'B+': 'B', 'B': 'B-', 'B-': 'CCC',
            'CCC': 'CC', 'CC': 'C', 'C': 'C'
        }
        
        # Rating to spread adjustment factors
        self.rating_spread_adjustments = {
            'AAA': 1.0, 'AA+': 1.1, 'AA': 1.2, 'AA-': 1.3,
            'A+': 1.4, 'A': 1.5, 'A-': 1.6, 'BBB+': 1.7,
            'BBB': 1.8, 'BBB-': 1.9, 'BB+': 2.0, 'BB': 2.2,
            'BB-': 2.4, 'B+': 2.6, 'B': 2.8, 'B-': 3.0,
            'CCC': 3.5, 'CC': 4.0, 'C': 4.5
        }
        
        # Sector volatility factors
        self.sector_volatility = {
            'Energy': 1.4, 'Mining': 1.5, 'Real Estate': 1.3,
            'Financial': 1.2, 'Technology': 1.0, 'Healthcare': 1.1,
            'Consumer Goods': 1.0, 'Utilities': 0.9, 'Construction': 1.2,
            'Automotive': 1.1, 'Infrastructure': 1.3, 'Telecommunications': 1.2
        }
    
    def create_scenario(self, name: str, description: str, 
                       shocks: List[Dict[str, Any]], **kwargs) -> ShockScenario:
        """
        Create a new shock scenario
        
        Args:
            name: Scenario name
            description: Scenario description
            shocks: List of shock definitions
            **kwargs: Additional scenario parameters
            
        Returns:
            ShockScenario object
        """
        return ShockScenario(
            name=name,
            description=description,
            shocks=shocks,
            **kwargs
        )
    
    def apply_shock(self, df: pd.DataFrame, shock: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a single shock to the dataset
        
        Args:
            df: DataFrame with bond data
            shock: Shock definition dictionary
            
        Returns:
            DataFrame with applied shock
        """
        df_shocked = df.copy()
        shock_type = shock.get('type', ShockType.ABSOLUTE.value)
        dimension = shock.get('dimension')
        value = shock.get('value')
        filters = shock.get('filters', {})
        
        # Apply filters to identify affected bonds
        mask = self._apply_filters(df_shocked, filters)
        
        if not mask.any():
            self.logger.warning(f"No bonds match filters for shock: {shock}")
            return df_shocked
        
        if shock_type == ShockType.ABSOLUTE.value:
            df_shocked.loc[mask, dimension] = value
        elif shock_type == ShockType.PERCENTAGE.value:
            df_shocked.loc[mask, dimension] *= (1 + value / 100)
        elif shock_type == ShockType.MULTIPLIER.value:
            df_shocked.loc[mask, dimension] *= value
        elif shock_type == ShockType.RATING_DOWNGRADE.value:
            df_shocked = self._apply_rating_downgrade(df_shocked, mask, value)
        elif shock_type == ShockType.SECTOR_SHOCK.value:
            df_shocked = self._apply_sector_shock(df_shocked, mask, dimension, value)
        
        return df_shocked
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.Series:
        """Apply filters to create a boolean mask"""
        mask = pd.Series([True] * len(df), index=df.index)
        
        for key, value in filters.items():
            if key == 'credit_rating':
                if isinstance(value, list):
                    mask &= df['credit_rating'].isin(value)
                else:
                    mask &= df['credit_rating'] == value
            elif key == 'sector':
                if isinstance(value, list):
                    mask &= df['sector'].isin(value)
                else:
                    mask &= df['sector'] == value
            elif key == 'min_liquidity_index':
                mask &= df['liquidity_index'] >= value
            elif key == 'max_liquidity_index':
                mask &= df['liquidity_index'] <= value
            elif key == 'min_spread_bps':
                mask &= df['spread_bps'] >= value
            elif key == 'max_spread_bps':
                mask &= df['spread_bps'] <= value
            elif key == 'min_issue_size':
                mask &= df['issue_size'] >= value
            elif key == 'max_issue_size':
                mask &= df['issue_size'] <= value
        
        return mask
    
    def _apply_rating_downgrade(self, df: pd.DataFrame, mask: pd.Series, 
                               downgrade_steps: int = 1) -> pd.DataFrame:
        """Apply rating downgrade to affected bonds"""
        df_shocked = df.copy()
        
        for _ in range(downgrade_steps):
            df_shocked.loc[mask, 'credit_rating'] = df_shocked.loc[mask, 'credit_rating'].map(
                lambda x: self.rating_downgrades.get(x, x)
            )
        
        # Adjust spreads based on new ratings
        for idx in df_shocked[mask].index:
            old_rating = df.loc[idx, 'credit_rating']
            new_rating = df_shocked.loc[idx, 'credit_rating']
            
            if old_rating != new_rating:
                old_factor = self.rating_spread_adjustments.get(old_rating, 1.0)
                new_factor = self.rating_spread_adjustments.get(new_rating, 1.0)
                
                # Adjust spread proportionally
                spread_adjustment = new_factor / old_factor
                df_shocked.loc[idx, 'spread_bps'] *= spread_adjustment
                
                # Adjust bid-ask spread
                df_shocked.loc[idx, 'bid_ask_spread'] *= spread_adjustment
        
        return df_shocked
    
    def _apply_sector_shock(self, df: pd.DataFrame, mask: pd.Series, 
                           dimension: str, shock_value: float) -> pd.DataFrame:
        """Apply sector-specific shock with volatility adjustment"""
        df_shocked = df.copy()
        
        for idx in df_shocked[mask].index:
            sector = df_shocked.loc[idx, 'sector']
            volatility_factor = self.sector_volatility.get(sector, 1.0)
            
            # Apply shock with sector volatility adjustment
            adjusted_shock = shock_value * volatility_factor
            
            if dimension == 'spread_bps':
                df_shocked.loc[idx, 'spread_bps'] += adjusted_shock
            elif dimension == 'liquidity_index':
                df_shocked.loc[idx, 'liquidity_index'] = max(0, 
                    df_shocked.loc[idx, 'liquidity_index'] - adjusted_shock)
            elif dimension == 'volume_traded':
                df_shocked.loc[idx, 'volume_traded'] *= (1 - adjusted_shock)
        
        return df_shocked
    
    def run_scenario(self, df: pd.DataFrame, scenario: ShockScenario) -> ShockResult:
        """
        Run a complete stress scenario
        
        Args:
            df: DataFrame with bond data
            scenario: ShockScenario to run
            
        Returns:
            ShockResult with before/after analysis
        """
        self.logger.info(f"Running stress scenario: {scenario.name}")
        
        # Calculate baseline metrics
        baseline_metrics = self._calculate_portfolio_metrics(df)
        
        # Apply all shocks in sequence
        df_shocked = df.copy()
        for shock in scenario.shocks:
            df_shocked = self.apply_shock(df_shocked, shock)
        
        # Calculate shocked metrics
        shocked_metrics = self._calculate_portfolio_metrics(df_shocked)
        
        # Calculate impact summary
        impact_summary = self._calculate_impact_summary(baseline_metrics, shocked_metrics)
        
        # Identify affected issuers
        affected_issuers = self._identify_affected_issuers(df, df_shocked)
        
        # Generate risk flags
        risk_flags = self._generate_risk_flags(impact_summary, affected_issuers)
        
        return ShockResult(
            scenario_name=scenario.name,
            before_metrics=baseline_metrics,
            after_metrics=shocked_metrics,
            impact_summary=impact_summary,
            affected_issuers=affected_issuers,
            risk_flags=risk_flags
        )
    
    def _calculate_portfolio_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        # Ensure liquidity index exists
        if 'liquidity_index' not in df.columns:
            from .liquidity_insights import LiquidityInsightsEngine
            liq_engine = LiquidityInsightsEngine(seed=self.seed)
            df = df.copy()
            df['liquidity_index'] = liq_engine.calculate_liquidity_index(df)
            df['expected_tte_days'] = liq_engine.calculate_expected_tte(df, df['liquidity_index'])
        
        total_par = df['issue_size'].sum()
        
        metrics = {
            'portfolio_summary': {
                'total_par_value': total_par,
                'total_issuers': df['issuer_name'].nunique(),
                'total_bonds': len(df),
                'weighted_liquidity_index': (df['liquidity_index'] * df['issue_size']).sum() / total_par,
                'weighted_spread_bps': (df['spread_bps'] * df['issue_size']).sum() / total_par,
                'weighted_bid_ask_spread': (df['bid_ask_spread'] * df['issue_size']).sum() / total_par,
                'weighted_expected_tte_days': (df['expected_tte_days'] * df['issue_size']).sum() / total_par
            },
            'sector_exposure': df.groupby('sector')['issue_size'].sum().to_dict(),
            'rating_exposure': df.groupby('credit_rating')['issue_size'].sum().to_dict(),
            'liquidity_tier_exposure': pd.cut(df['liquidity_index'], 
                                            bins=[0, 0.2, 0.4, 0.6, 1.0],
                                            labels=['Illiquid', 'Low', 'Medium', 'High']).value_counts().to_dict(),
            'risk_metrics': {
                'high_yield_exposure': df[df['credit_rating'].isin(['BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C'])]['issue_size'].sum() / total_par,
                'illiquid_exposure': df[df['liquidity_index'] < 0.3]['issue_size'].sum() / total_par,
                'long_tte_exposure': df[df['expected_tte_days'] > 30]['issue_size'].sum() / total_par,
                'wide_spread_exposure': df[df['spread_bps'] > 300]['issue_size'].sum() / total_par
            }
        }
        
        return metrics
    
    def _calculate_impact_summary(self, before: Dict[str, Any], 
                                 after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate impact summary between before and after metrics"""
        impact = {}
        
        # Portfolio summary impacts
        for key in before['portfolio_summary']:
            if key == 'total_issuers' or key == 'total_bonds':
                continue
            
            before_val = before['portfolio_summary'][key]
            after_val = after['portfolio_summary'][key]
            
            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                if before_val != 0:
                    pct_change = ((after_val - before_val) / before_val) * 100
                else:
                    pct_change = 0
                
                impact[f"{key}_change"] = after_val - before_val
                impact[f"{key}_pct_change"] = pct_change
        
        # Sector exposure impacts
        sector_impact = {}
        for sector in set(before['sector_exposure'].keys()) | set(after['sector_exposure'].keys()):
            before_exposure = before['sector_exposure'].get(sector, 0)
            after_exposure = after['sector_exposure'].get(sector, 0)
            
            if before_exposure != 0:
                pct_change = ((after_exposure - before_exposure) / before_exposure) * 100
            else:
                pct_change = 0
            
            sector_impact[sector] = {
                'exposure_change': after_exposure - before_exposure,
                'exposure_pct_change': pct_change
            }
        
        impact['sector_impacts'] = sector_impact
        
        # Risk metric impacts
        risk_impact = {}
        for key in before['risk_metrics']:
            before_val = before['risk_metrics'][key]
            after_val = after['risk_metrics'][key]
            
            if before_val != 0:
                pct_change = ((after_val - before_val) / before_val) * 100
            else:
                pct_change = 0
            
            risk_impact[f"{key}_change"] = after_val - before_val
            risk_impact[f"{key}_pct_change"] = pct_change
        
        impact['risk_impacts'] = risk_impact
        
        return impact
    
    def _identify_affected_issuers(self, df_before: pd.DataFrame, 
                                  df_after: pd.DataFrame) -> List[str]:
        """Identify issuers affected by the stress scenario"""
        affected = set()
        
        # Compare key metrics
        for col in ['spread_bps', 'bid_ask_spread', 'liquidity_index', 'credit_rating']:
            if col in df_before.columns and col in df_after.columns:
                # Find bonds with significant changes
                if col == 'credit_rating':
                    changed_mask = df_before[col] != df_after[col]
                else:
                    # For numerical columns, check for >5% change
                    pct_change = abs((df_after[col] - df_before[col]) / df_before[col])
                    changed_mask = pct_change > 0.05
                
                affected_issuers = df_before.loc[changed_mask, 'issuer_name'].unique()
                affected.update(affected_issuers)
        
        return list(affected)
    
    def _generate_risk_flags(self, impact_summary: Dict[str, Any], 
                            affected_issuers: List[str]) -> List[str]:
        """Generate risk flags based on impact summary"""
        flags = []
        
        # Check for significant portfolio impacts
        if impact_summary.get('weighted_liquidity_index_pct_change', 0) < -20:
            flags.append("Significant liquidity deterioration")
        
        if impact_summary.get('weighted_spread_bps_pct_change', 0) > 30:
            flags.append("Material spread widening")
        
        if impact_summary.get('weighted_expected_tte_days_pct_change', 0) > 50:
            flags.append("Substantial increase in exit time")
        
        # Check risk metric changes
        risk_impacts = impact_summary.get('risk_impacts', {})
        if risk_impacts.get('high_yield_exposure_pct_change', 0) > 20:
            flags.append("Increased high-yield exposure")
        
        if risk_impacts.get('illiquid_exposure_pct_change', 0) > 25:
            flags.append("Increased illiquid exposure")
        
        # Check sector concentration
        sector_impacts = impact_summary.get('sector_impacts', {})
        for sector, impact in sector_impacts.items():
            if impact.get('exposure_pct_change', 0) > 40:
                flags.append(f"Concentration risk in {sector} sector")
        
        # Check number of affected issuers
        if len(affected_issuers) > 10:
            flags.append("Widespread issuer impact")
        
        return flags
    
    def run_multiple_scenarios(self, df: pd.DataFrame, 
                             scenarios: List[ShockScenario]) -> Dict[str, ShockResult]:
        """
        Run multiple stress scenarios
        
        Args:
            df: DataFrame with bond data
            scenarios: List of scenarios to run
            
        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        
        for scenario in scenarios:
            try:
                result = self.run_scenario(df, scenario)
                results[scenario.name] = result
                self.logger.info(f"Completed scenario: {scenario.name}")
            except Exception as e:
                self.logger.error(f"Error running scenario {scenario.name}: {e}")
                continue
        
        return results
    
    def generate_scenario_report(self, results: Dict[str, ShockResult], 
                               output_dir: str = "outputs") -> Dict[str, str]:
        """
        Generate comprehensive scenario report
        
        Args:
            results: Dictionary of scenario results
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate summary report
        summary_data = {
            'scenario_summary': {},
            'portfolio_impacts': {},
            'risk_analysis': {},
            'generated_at': datetime.now().isoformat()
        }
        
        for scenario_name, result in results.items():
            # Scenario summary
            summary_data['scenario_summary'][scenario_name] = {
                'affected_issuers_count': len(result.affected_issuers),
                'risk_flags_count': len(result.risk_flags),
                'liquidity_impact_pct': result.impact_summary.get('weighted_liquidity_index_pct_change', 0),
                'spread_impact_pct': result.impact_summary.get('weighted_spread_bps_pct_change', 0)
            }
            
            # Portfolio impacts
            summary_data['portfolio_impacts'][scenario_name] = result.impact_summary
            
            # Risk analysis
            summary_data['risk_analysis'][scenario_name] = {
                'risk_flags': result.risk_flags,
                'affected_issuers': result.affected_issuers
            }
        
        # Export detailed results
        detailed_path = output_path / "scenario_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Export summary report
        summary_path = output_path / "scenario_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Export CSV comparison
        comparison_data = []
        for scenario_name, result in results.items():
            comparison_data.append({
                'scenario_name': scenario_name,
                'liquidity_index_before': result.before_metrics['portfolio_summary']['weighted_liquidity_index'],
                'liquidity_index_after': result.after_metrics['portfolio_summary']['weighted_liquidity_index'],
                'liquidity_change_pct': result.impact_summary.get('weighted_liquidity_index_pct_change', 0),
                'spread_bps_before': result.before_metrics['portfolio_summary']['weighted_spread_bps'],
                'spread_bps_after': result.after_metrics['portfolio_summary']['weighted_spread_bps'],
                'spread_change_pct': result.impact_summary.get('weighted_spread_bps_pct_change', 0),
                'affected_issuers': len(result.affected_issuers),
                'risk_flags': len(result.risk_flags)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        csv_path = output_path / "scenario_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        return {
            'detailed': str(detailed_path),
            'summary': str(summary_path),
            'comparison': str(csv_path)
        }
    
    def create_preset_scenarios(self) -> List[ShockScenario]:
        """Create preset stress scenarios for common use cases"""
        scenarios = []
        
        # Market stress scenario
        market_stress = ShockScenario(
            name="Market Stress",
            description="General market stress affecting all bonds",
            probability=0.05,
            severity="moderate",
            tags=["market", "systemic"],
            shocks=[
                {
                    "type": "percentage",
                    "dimension": "spread_bps",
                    "value": 25,
                    "filters": {}
                },
                {
                    "type": "percentage",
                    "dimension": "bid_ask_spread",
                    "value": 20,
                    "filters": {}
                },
                {
                    "type": "percentage",
                    "dimension": "liquidity_index",
                    "value": -15,
                    "filters": {}
                }
            ]
        )
        scenarios.append(market_stress)
        
        # High yield stress
        hy_stress = ShockScenario(
            name="High Yield Stress",
            description="Stress specifically affecting high yield bonds",
            probability=0.03,
            severity="high",
            tags=["high_yield", "credit"],
            shocks=[
                {
                    "type": "percentage",
                    "dimension": "spread_bps",
                    "value": 50,
                    "filters": {"credit_rating": ["BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C"]}
                },
                {
                    "type": "percentage",
                    "dimension": "liquidity_index",
                    "value": -30,
                    "filters": {"credit_rating": ["BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C"]}
                }
            ]
        )
        scenarios.append(hy_stress)
        
        # Sector-specific stress
        energy_stress = ShockScenario(
            name="Energy Sector Stress",
            description="Stress affecting energy and mining sectors",
            probability=0.02,
            severity="high",
            tags=["sector", "energy"],
            shocks=[
                {
                    "type": "sector_shock",
                    "dimension": "spread_bps",
                    "value": 100,
                    "filters": {"sector": ["Energy", "Mining"]}
                },
                {
                    "type": "sector_shock",
                    "dimension": "liquidity_index",
                    "value": 0.3,
                    "filters": {"sector": ["Energy", "Mining"]}
                }
            ]
        )
        scenarios.append(energy_stress)
        
        # Rating downgrade wave
        downgrade_wave = ShockScenario(
            name="Rating Downgrade Wave",
            description="Widespread rating downgrades",
            probability=0.01,
            severity="severe",
            tags=["rating", "credit"],
            shocks=[
                {
                    "type": "rating_downgrade",
                    "dimension": "credit_rating",
                    "value": 1,
                    "filters": {"credit_rating": ["BBB", "BBB-", "BB+", "BB"]}
                }
            ]
        )
        scenarios.append(downgrade_wave)
        
        return scenarios
