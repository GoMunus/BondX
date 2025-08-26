"""
Liquidity Insights Engine

This module provides comprehensive liquidity analysis including:
- Issuer ranking by liquidity metrics
- Expected time-to-exit calculations
- Exit route recommendations
- Portfolio rollups and snapshots
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ExitRoute(Enum):
    """Exit route options for bond positions"""
    MARKET_MAKER = "MM"
    AUCTION = "Auction"
    RFQ = "RFQ"
    TOKENIZED = "Tokenized"

class LiquidityTier(Enum):
    """Liquidity tiers based on metrics"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    ILLIQUID = "Illiquid"

@dataclass
class ExitRouteRecommendation:
    """Exit route recommendation with confidence and reasoning"""
    primary_route: ExitRoute
    secondary_route: ExitRoute
    confidence: float
    reasoning: str
    expected_fill_time_hours: float
    expected_cost_bps: float

@dataclass
class IssuerSnapshot:
    """Comprehensive issuer liquidity snapshot"""
    issuer_name: str
    sector: str
    credit_rating: str
    liquidity_score: float
    liquidity_tier: LiquidityTier
    expected_tte_days: float
    exit_route_recommendation: ExitRouteRecommendation
    risk_flags: List[str]
    recent_changes: Dict[str, Any]
    portfolio_weight: float = 0.0

class LiquidityInsightsEngine:
    """Engine for generating liquidity insights and recommendations"""
    
    def __init__(self, seed: int = 42):
        """Initialize the liquidity insights engine"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        
        # Liquidity thresholds
        self.liquidity_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'illiquid': 0.2
        }
        
        # Exit route parameters
        self.exit_route_params = {
            ExitRoute.MARKET_MAKER: {
                'min_liquidity': 0.6,
                'max_size': 10000000,
                'fill_time_hours': 2,
                'cost_bps': 5
            },
            ExitRoute.AUCTION: {
                'min_liquidity': 0.4,
                'max_size': 50000000,
                'fill_time_hours': 24,
                'cost_bps': 15
            },
            ExitRoute.RFQ: {
                'min_liquidity': 0.3,
                'max_size': 25000000,
                'fill_time_hours': 48,
                'cost_bps': 25
            },
            ExitRoute.TOKENIZED: {
                'min_liquidity': 0.2,
                'max_size': 10000000,
                'fill_time_hours': 72,
                'cost_bps': 40
            }
        }
    
    def calculate_liquidity_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite liquidity index from multiple factors
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            Series with liquidity index values (0-1)
        """
        # Normalize individual factors
        volume_norm = (df['volume_traded'] - df['volume_traded'].min()) / \
                     (df['volume_traded'].max() - df['volume_traded'].min())
        
        spread_norm = 1 - (df['bid_ask_spread'] - df['bid_ask_spread'].min()) / \
                     (df['bid_ask_spread'].max() - df['bid_ask_spread'].min())
        
        size_norm = (df['issue_size'] - df['issue_size'].min()) / \
                   (df['issue_size'].max() - df['issue_size'].min())
        
        # Weighted composite index
        liquidity_index = (
            0.4 * volume_norm +
            0.3 * spread_norm +
            0.3 * size_norm
        )
        
        return liquidity_index.clip(0, 1)
    
    def calculate_expected_tte(self, df: pd.DataFrame, liquidity_index: pd.Series) -> pd.Series:
        """
        Calculate expected time-to-exit based on liquidity metrics
        
        Args:
            df: DataFrame with bond data
            liquidity_index: Series with liquidity index values
            
        Returns:
            Series with expected TTE in days
        """
        # Base TTE from liquidity index
        base_tte = 30 * (1 - liquidity_index)  # 0-30 days
        
        # Adjust for credit rating
        rating_adjustments = {
            'AAA': 0.7, 'AA+': 0.75, 'AA': 0.8, 'AA-': 0.85,
            'A+': 0.9, 'A': 0.95, 'A-': 1.0,
            'BBB+': 1.1, 'BBB': 1.2, 'BBB-': 1.3,
            'BB+': 1.5, 'BB': 1.7, 'BB-': 2.0,
            'B+': 2.5, 'B': 3.0, 'B-': 3.5,
            'CCC': 4.0, 'CC': 4.5, 'C': 5.0
        }
        
        rating_multiplier = df['credit_rating'].map(rating_adjustments).fillna(1.0)
        
        # Adjust for sector volatility
        sector_adjustments = {
            'Energy': 1.3, 'Mining': 1.4, 'Real Estate': 1.2,
            'Financial': 1.1, 'Technology': 0.9, 'Healthcare': 0.95,
            'Consumer Goods': 0.9, 'Utilities': 0.8, 'Construction': 1.1,
            'Automotive': 1.0, 'Infrastructure': 1.2, 'Telecommunications': 1.1
        }
        
        sector_multiplier = df['sector'].map(sector_adjustments).fillna(1.0)
        
        # Final TTE calculation
        expected_tte = base_tte * rating_multiplier * sector_multiplier
        
        return expected_tte.clip(1, 365)  # 1 day to 1 year
    
    def recommend_exit_route(self, df: pd.DataFrame, liquidity_index: pd.Series, 
                           issue_size: pd.Series) -> List[ExitRouteRecommendation]:
        """
        Recommend exit routes based on liquidity and size constraints
        
        Args:
            df: DataFrame with bond data
            liquidity_index: Series with liquidity index values
            issue_size: Series with issue sizes
            
        Returns:
            List of exit route recommendations
        """
        recommendations = []
        
        for idx, row in df.iterrows():
            liq_idx = liquidity_index.iloc[idx]
            size = issue_size.iloc[idx]
            
            # Score each exit route
            route_scores = {}
            for route, params in self.exit_route_params.items():
                if liq_idx >= params['min_liquidity'] and size <= params['max_size']:
                    # Calculate score based on liquidity and size fit
                    liquidity_score = liq_idx / params['min_liquidity']
                    size_score = 1 - (size / params['max_size'])
                    route_scores[route] = (liquidity_score + size_score) / 2
                else:
                    route_scores[route] = 0
            
            if not route_scores:
                # Fallback to RFQ if no routes qualify
                primary_route = ExitRoute.RFQ
                secondary_route = ExitRoute.TOKENIZED
                confidence = 0.3
            else:
                # Sort routes by score
                sorted_routes = sorted(route_scores.items(), key=lambda x: x[1], reverse=True)
                primary_route = sorted_routes[0][0]
                secondary_route = sorted_routes[1][0] if len(sorted_routes) > 1 else ExitRoute.RFQ
                confidence = sorted_routes[0][1]
            
            # Get route parameters
            primary_params = self.exit_route_params[primary_route]
            
            # Generate reasoning
            reasoning = self._generate_exit_reasoning(row, primary_route, liq_idx, size)
            
            recommendation = ExitRouteRecommendation(
                primary_route=primary_route,
                secondary_route=secondary_route,
                confidence=confidence,
                reasoning=reasoning,
                expected_fill_time_hours=primary_params['fill_time_hours'],
                expected_cost_bps=primary_params['cost_bps']
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_exit_reasoning(self, row: pd.Series, route: ExitRoute, 
                               liquidity_index: float, size: float) -> str:
        """Generate human-readable reasoning for exit route recommendation"""
        
        if route == ExitRoute.MARKET_MAKER:
            return f"High liquidity ({liquidity_index:.2f}) and moderate size ({size/1e6:.1f}M) suitable for MM execution"
        elif route == ExitRoute.AUCTION:
            return f"Medium liquidity ({liquidity_index:.2f}) and larger size ({size/1e6:.1f}M) best served by auction process"
        elif route == ExitRoute.RFQ:
            return f"Lower liquidity ({liquidity_index:.2f}) requires RFQ to find natural buyers"
        else:  # Tokenized
            return f"Lowest liquidity ({liquidity_index:.2f}) may benefit from tokenization for fractional ownership"
    
    def rank_issuers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank issuers by liquidity metrics
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            DataFrame with issuer rankings
        """
        # Calculate liquidity metrics
        df = df.copy()
        df['liquidity_index'] = self.calculate_liquidity_index(df)
        df['expected_tte_days'] = self.calculate_expected_tte(df, df['liquidity_index'])
        
        # Get exit route recommendations
        exit_recommendations = self.recommend_exit_route(
            df, df['liquidity_index'], df['issue_size']
        )
        
        # Add exit route data
        df['primary_exit_route'] = [rec.primary_route.value for rec in exit_recommendations]
        df['exit_confidence'] = [rec.confidence for rec in exit_recommendations]
        df['expected_fill_time_hours'] = [rec.expected_fill_time_hours for rec in exit_recommendations]
        df['expected_cost_bps'] = [rec.expected_cost_bps for rec in exit_recommendations]
        
        # Group by issuer and aggregate
        issuer_summary = df.groupby('issuer_name').agg({
            'sector': 'first',
            'credit_rating': 'first',
            'liquidity_index': 'mean',
            'expected_tte_days': 'mean',
            'volume_traded': 'sum',
            'issue_size': 'sum',
            'spread_bps': 'mean',
            'bid_ask_spread': 'mean',
            'primary_exit_route': lambda x: x.mode().iloc[0] if not x.mode().empty else 'RFQ',
            'exit_confidence': 'mean',
            'expected_fill_time_hours': 'mean',
            'expected_cost_bps': 'mean'
        }).reset_index()
        
        # Calculate liquidity tier
        issuer_summary['liquidity_tier'] = pd.cut(
            issuer_summary['liquidity_index'],
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=['Illiquid', 'Low', 'Medium', 'High']
        )
        
        # Sort by liquidity index (descending)
        issuer_summary = issuer_summary.sort_values('liquidity_index', ascending=False)
        
        # Add rank
        issuer_summary['liquidity_rank'] = range(1, len(issuer_summary) + 1)
        
        return issuer_summary
    
    def generate_issuer_snapshots(self, df: pd.DataFrame) -> List[IssuerSnapshot]:
        """
        Generate comprehensive issuer snapshots
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            List of issuer snapshots
        """
        # Get issuer rankings
        issuer_rankings = self.rank_issuers(df)
        
        snapshots = []
        for _, row in issuer_rankings.iterrows():
            # Get issuer bonds
            issuer_bonds = df[df['issuer_name'] == row['issuer_name']]
            
            # Calculate portfolio weight
            total_portfolio = df['issue_size'].sum()
            portfolio_weight = row['issue_size'] / total_portfolio
            
            # Identify risk flags
            risk_flags = self._identify_risk_flags(row, issuer_bonds)
            
            # Calculate recent changes (simulated)
            recent_changes = self._calculate_recent_changes(row, issuer_bonds)
            
            # Create exit route recommendation
            exit_rec = ExitRouteRecommendation(
                primary_route=ExitRoute(row['primary_exit_route']),
                secondary_route=ExitRoute.RFQ,  # Default secondary
                confidence=row['exit_confidence'],
                reasoning=f"Based on liquidity index {row['liquidity_index']:.2f}",
                expected_fill_time_hours=row['expected_fill_time_hours'],
                expected_cost_bps=row['expected_cost_bps']
            )
            
            # Determine liquidity tier
            liquidity_tier = LiquidityTier(row['liquidity_tier'])
            
            snapshot = IssuerSnapshot(
                issuer_name=row['issuer_name'],
                sector=row['sector'],
                credit_rating=row['credit_rating'],
                liquidity_score=row['liquidity_index'],
                liquidity_tier=liquidity_tier,
                expected_tte_days=row['expected_tte_days'],
                exit_route_recommendation=exit_rec,
                risk_flags=risk_flags,
                recent_changes=recent_changes,
                portfolio_weight=portfolio_weight
            )
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def _identify_risk_flags(self, issuer_row: pd.Series, issuer_bonds: pd.DataFrame) -> List[str]:
        """Identify risk flags for an issuer"""
        flags = []
        
        if issuer_row['credit_rating'] in ['BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C']:
            flags.append("High yield/risky rating")
        
        if issuer_row['liquidity_index'] < 0.3:
            flags.append("Low liquidity")
        
        if issuer_row['expected_tte_days'] > 30:
            flags.append("Long expected exit time")
        
        if issuer_row['spread_bps'] > 300:
            flags.append("Wide credit spreads")
        
        if issuer_row['bid_ask_spread'] > 50:
            flags.append("Wide bid-ask spreads")
        
        return flags
    
    def _calculate_recent_changes(self, issuer_row: pd.Series, issuer_bonds: pd.DataFrame) -> Dict[str, Any]:
        """Calculate recent changes for an issuer (simulated)"""
        # Simulate recent changes based on current metrics
        changes = {}
        
        # Simulate spread changes
        spread_change = np.random.normal(0, 10)  # Random change in bps
        changes['spread_change_bps'] = spread_change
        
        # Simulate volume changes
        volume_change_pct = np.random.normal(0, 0.2)  # Random percentage change
        changes['volume_change_pct'] = volume_change_pct
        
        # Simulate liquidity index changes
        liq_change = np.random.normal(0, 0.05)  # Random change in liquidity index
        changes['liquidity_index_change'] = liq_change
        
        return changes
    
    def generate_portfolio_rollup(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate portfolio-level rollup statistics
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            Dictionary with portfolio rollup data
        """
        # Calculate liquidity metrics
        df = df.copy()
        df['liquidity_index'] = self.calculate_liquidity_index(df)
        df['expected_tte_days'] = self.calculate_expected_tte(df, df['liquidity_index'])
        
        # Portfolio totals
        total_par = df['issue_size'].sum()
        total_volume = df['volume_traded'].sum()
        
        # Weighted averages
        weighted_liquidity = (df['liquidity_index'] * df['issue_size']).sum() / total_par
        weighted_tte = (df['expected_tte_days'] * df['issue_size']).sum() / total_par
        weighted_spread = (df['spread_bps'] * df['issue_size']).sum() / total_par
        
        # Sector breakdown
        sector_breakdown = df.groupby('sector').agg({
            'issue_size': 'sum',
            'liquidity_index': 'mean',
            'expected_tte_days': 'mean',
            'spread_bps': 'mean'
        }).to_dict('index')
        
        # Rating breakdown
        rating_breakdown = df.groupby('credit_rating').agg({
            'issue_size': 'sum',
            'liquidity_index': 'mean',
            'expected_tte_days': 'mean',
            'spread_bps': 'mean'
        }).to_dict('index')
        
        # Liquidity tier breakdown
        liquidity_tiers = pd.cut(df['liquidity_index'], 
                               bins=[0, 0.2, 0.4, 0.6, 1.0],
                               labels=['Illiquid', 'Low', 'Medium', 'High'])
        tier_breakdown = df.groupby(liquidity_tiers).agg({
            'issue_size': 'sum',
            'count': 'size'
        }).to_dict('index')
        
        rollup = {
            'portfolio_summary': {
                'total_par_value': total_par,
                'total_volume_traded': total_volume,
                'weighted_liquidity_index': weighted_liquidity,
                'weighted_expected_tte_days': weighted_tte,
                'weighted_spread_bps': weighted_spread,
                'total_issuers': df['issuer_name'].nunique(),
                'total_bonds': len(df)
            },
            'sector_breakdown': sector_breakdown,
            'rating_breakdown': rating_breakdown,
            'liquidity_tier_breakdown': tier_breakdown,
            'risk_metrics': {
                'high_yield_exposure': df[df['credit_rating'].isin(['BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C'])]['issue_size'].sum() / total_par,
                'illiquid_exposure': df[df['liquidity_index'] < 0.3]['issue_size'].sum() / total_par,
                'long_tte_exposure': df[df['expected_tte_days'] > 30]['issue_size'].sum() / total_par
            }
        }
        
        return rollup
    
    def export_insights(self, df: pd.DataFrame, output_dir: str = "outputs") -> Dict[str, str]:
        """
        Export liquidity insights to various formats
        
        Args:
            df: DataFrame with bond data
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate insights
        issuer_rankings = self.rank_issuers(df)
        issuer_snapshots = self.generate_issuer_snapshots(df)
        portfolio_rollup = self.generate_portfolio_rollup(df)
        
        # Export CSV
        csv_path = output_path / "issuer_rankings.csv"
        issuer_rankings.to_csv(csv_path, index=False)
        
        # Export JSON
        json_path = output_path / "liquidity_insights.json"
        insights_data = {
            'issuer_rankings': issuer_rankings.to_dict('records'),
            'portfolio_rollup': portfolio_rollup,
            'generated_at': datetime.now().isoformat(),
            'seed': self.seed
        }
        
        with open(json_path, 'w') as f:
            json.dump(insights_data, f, indent=2, default=str)
        
        # Export dashboard-ready JSON
        dashboard_path = output_path / "dashboard_data.json"
        dashboard_data = {
            'issuer_snapshots': [
                {
                    'issuer_name': snap.issuer_name,
                    'sector': snap.sector,
                    'credit_rating': snap.credit_rating,
                    'liquidity_score': snap.liquidity_score,
                    'liquidity_tier': snap.liquidity_tier.value,
                    'expected_tte_days': snap.expected_tte_days,
                    'exit_route': snap.exit_route_recommendation.primary_route.value,
                    'exit_confidence': snap.exit_route_recommendation.confidence,
                    'risk_flags': snap.risk_flags,
                    'portfolio_weight': snap.portfolio_weight
                }
                for snap in issuer_snapshots
            ],
            'portfolio_rollup': portfolio_rollup,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return {
            'csv': str(csv_path),
            'json': str(json_path),
            'dashboard': str(dashboard_path)
        }
