"""
Sector Heatmaps Engine

This module provides comprehensive sector analysis including:
- ESG heatmaps by sector, rating, and tenor
- Yield spread heatmaps with drilldowns
- Systemic risk heatmaps with threshold monitoring
- Top movers and deterioration tracking
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
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class HeatmapType(Enum):
    """Types of heatmaps available"""
    ESG = "esg"
    YIELD_SPREAD = "yield_spread"
    SYSTEMIC_RISK = "systemic_risk"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"

class RiskLevel(Enum):
    """Risk levels for threshold monitoring"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class HeatmapData:
    """Structured heatmap data"""
    heatmap_type: HeatmapType
    data_matrix: pd.DataFrame
    row_labels: List[str]
    col_labels: List[str]
    color_scale: str = "RdYlGn_r"
    annotations: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThresholdAlert:
    """Threshold breach alert"""
    metric: str
    sector: str
    rating: str
    current_value: float
    threshold_value: float
    breach_type: str  # "above" or "below"
    severity: RiskLevel
    timestamp: datetime
    description: str

class SectorHeatmapEngine:
    """Engine for generating sector heatmaps and risk analysis"""
    
    def __init__(self, seed: int = 42):
        """Initialize the sector heatmap engine"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        
        # ESG scoring weights
        self.esg_weights = {
            'environmental': 0.4,
            'social': 0.3,
            'governance': 0.3
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'liquidity_index': {'low': 0.6, 'medium': 0.4, 'high': 0.2, 'critical': 0.1},
            'spread_bps': {'low': 100, 'medium': 200, 'high': 400, 'critical': 600},
            'bid_ask_spread': {'low': 20, 'medium': 40, 'high': 80, 'critical': 120},
            'esg_score': {'low': 0.7, 'medium': 0.5, 'high': 0.3, 'critical': 0.1}
        }
        
        # Sector ESG baseline scores (simulated)
        self.sector_esg_baselines = {
            'Technology': {'E': 0.8, 'S': 0.7, 'G': 0.8},
            'Healthcare': {'E': 0.7, 'S': 0.9, 'G': 0.7},
            'Consumer Goods': {'E': 0.6, 'S': 0.7, 'G': 0.8},
            'Utilities': {'E': 0.5, 'S': 0.8, 'G': 0.7},
            'Financial': {'E': 0.7, 'S': 0.6, 'G': 0.6},
            'Energy': {'E': 0.3, 'S': 0.5, 'G': 0.6},
            'Mining': {'E': 0.2, 'S': 0.4, 'G': 0.5},
            'Real Estate': {'E': 0.4, 'S': 0.6, 'G': 0.7},
            'Construction': {'E': 0.4, 'S': 0.6, 'G': 0.6},
            'Automotive': {'E': 0.5, 'S': 0.7, 'G': 0.7},
            'Infrastructure': {'E': 0.6, 'S': 0.7, 'G': 0.8},
            'Telecommunications': {'E': 0.7, 'S': 0.8, 'G': 0.7}
        }
    
    def calculate_esg_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ESG scores for bonds based on sector and rating
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            DataFrame with ESG scores added
        """
        df_esg = df.copy()
        
        # Initialize ESG columns
        df_esg['esg_environmental'] = 0.0
        df_esg['esg_social'] = 0.0
        df_esg['esg_governance'] = 0.0
        df_esg['esg_composite'] = 0.0
        
        for idx, row in df_esg.iterrows():
            sector = row['sector']
            rating = row['credit_rating']
            
            # Get sector baseline
            sector_baseline = self.sector_esg_baselines.get(sector, {'E': 0.5, 'S': 0.5, 'G': 0.5})
            
            # Apply rating adjustments
            rating_multiplier = self._get_rating_esg_multiplier(rating)
            
            # Calculate individual ESG scores
            df_esg.loc[idx, 'esg_environmental'] = sector_baseline['E'] * rating_multiplier
            df_esg.loc[idx, 'esg_social'] = sector_baseline['S'] * rating_multiplier
            df_esg.loc[idx, 'esg_governance'] = sector_baseline['G'] * rating_multiplier
            
            # Calculate composite score
            composite_score = (
                self.esg_weights['environmental'] * df_esg.loc[idx, 'esg_environmental'] +
                self.esg_weights['social'] * df_esg.loc[idx, 'esg_social'] +
                self.esg_weights['governance'] * df_esg.loc[idx, 'esg_governance']
            )
            
            df_esg.loc[idx, 'esg_composite'] = composite_score
        
        return df_esg
    
    def _get_rating_esg_multiplier(self, rating: str) -> float:
        """Get ESG multiplier based on credit rating"""
        rating_multipliers = {
            'AAA': 1.2, 'AA+': 1.15, 'AA': 1.1, 'AA-': 1.05,
            'A+': 1.0, 'A': 0.95, 'A-': 0.9, 'BBB+': 0.85,
            'BBB': 0.8, 'BBB-': 0.75, 'BB+': 0.7, 'BB': 0.65,
            'BB-': 0.6, 'B+': 0.55, 'B': 0.5, 'B-': 0.45,
            'CCC': 0.4, 'CC': 0.35, 'C': 0.3
        }
        
        return rating_multipliers.get(rating, 0.8)
    
    def generate_esg_heatmap(self, df: pd.DataFrame, 
                            group_by: str = 'sector') -> HeatmapData:
        """
        Generate ESG heatmap by sector or rating
        
        Args:
            df: DataFrame with bond data
            group_by: Grouping dimension ('sector' or 'credit_rating')
            
        Returns:
            HeatmapData object
        """
        # Calculate ESG scores
        df_esg = self.calculate_esg_scores(df)
        
        # Group by specified dimension
        if group_by == 'sector':
            grouped = df_esg.groupby('sector').agg({
                'esg_environmental': 'mean',
                'esg_social': 'mean',
                'esg_governance': 'mean',
                'esg_composite': 'mean'
            })
        else:  # credit_rating
            grouped = df_esg.groupby('credit_rating').agg({
                'esg_environmental': 'mean',
                'esg_social': 'mean',
                'esg_governance': 'mean',
                'esg_composite': 'mean'
            })
        
        # Create heatmap data
        heatmap_matrix = grouped[['esg_environmental', 'esg_social', 'esg_governance', 'esg_composite']]
        
        # Add annotations (count of bonds in each group)
        annotations = df_esg.groupby(group_by).size().to_dict()
        
        return HeatmapData(
            heatmap_type=HeatmapType.ESG,
            data_matrix=heatmap_matrix,
            row_labels=heatmap_matrix.index.tolist(),
            col_labels=heatmap_matrix.columns.tolist(),
            color_scale="RdYlGn",
            metadata={
                'group_by': group_by,
                'annotations': annotations,
                'total_bonds': len(df_esg)
            }
        )
    
    def generate_yield_spread_heatmap(self, df: pd.DataFrame, 
                                    group_by: str = 'sector') -> HeatmapData:
        """
        Generate yield spread heatmap by sector or rating
        
        Args:
            df: DataFrame with bond data
            group_by: Grouping dimension ('sector' or 'credit_rating')
            
        Returns:
            HeatmapData object
        """
        # Group by specified dimension
        if group_by == 'sector':
            grouped = df.groupby('sector').agg({
                'spread_bps': 'mean',
                'yield_to_maturity': 'mean',
                'bid_ask_spread': 'mean',
                'volume_traded': 'sum'
            })
        else:  # credit_rating
            grouped = df.groupby('credit_rating').agg({
                'spread_bps': 'mean',
                'yield_to_maturity': 'mean',
                'bid_ask_spread': 'mean',
                'volume_traded': 'sum'
            })
        
        # Normalize volume for better visualization
        grouped['volume_traded_normalized'] = grouped['volume_traded'] / grouped['volume_traded'].max()
        
        # Create heatmap data
        heatmap_matrix = grouped[['spread_bps', 'yield_to_maturity', 'bid_ask_spread', 'volume_traded_normalized']]
        
        # Add annotations
        annotations = df.groupby(group_by).size().to_dict()
        
        return HeatmapData(
            heatmap_type=HeatmapType.YIELD_SPREAD,
            data_matrix=heatmap_matrix,
            row_labels=heatmap_matrix.index.tolist(),
            col_labels=heatmap_matrix.columns.tolist(),
            color_scale="RdYlBu_r",
            metadata={
                'group_by': group_by,
                'annotations': annotations,
                'total_bonds': len(df)
            }
        )
    
    def generate_systemic_risk_heatmap(self, df: pd.DataFrame) -> HeatmapData:
        """
        Generate systemic risk heatmap combining multiple risk factors
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            HeatmapData object
        """
        # Calculate liquidity index if not present
        if 'liquidity_index' not in df.columns:
            from .liquidity_insights import LiquidityInsightsEngine
            liq_engine = LiquidityInsightsEngine(seed=self.seed)
            df = df.copy()
            df['liquidity_index'] = liq_engine.calculate_liquidity_index(df)
        
        # Group by sector and calculate risk metrics
        risk_metrics = df.groupby('sector').agg({
            'liquidity_index': 'mean',
            'spread_bps': 'mean',
            'bid_ask_spread': 'mean',
            'issue_size': 'sum'
        })
        
        # Calculate risk scores (0-1, higher = more risky)
        risk_metrics['liquidity_risk'] = 1 - risk_metrics['liquidity_index']
        risk_metrics['spread_risk'] = risk_metrics['spread_bps'] / risk_metrics['spread_bps'].max()
        risk_metrics['bid_ask_risk'] = risk_metrics['bid_ask_spread'] / risk_metrics['bid_ask_spread'].max()
        
        # Calculate composite risk score
        risk_metrics['composite_risk'] = (
            0.4 * risk_metrics['liquidity_risk'] +
            0.4 * risk_metrics['spread_risk'] +
            0.2 * risk_metrics['bid_ask_risk']
        )
        
        # Create heatmap data
        heatmap_matrix = risk_metrics[['liquidity_risk', 'spread_risk', 'bid_ask_risk', 'composite_risk']]
        
        # Add annotations
        annotations = df.groupby('sector').size().to_dict()
        
        return HeatmapData(
            heatmap_type=HeatmapType.SYSTEMIC_RISK,
            data_matrix=heatmap_matrix,
            row_labels=heatmap_matrix.index.tolist(),
            col_labels=heatmap_matrix.columns.tolist(),
            color_scale="Reds",
            metadata={
                'group_by': 'sector',
                'annotations': annotations,
                'total_bonds': len(df)
            }
        )
    
    def generate_liquidity_heatmap(self, df: pd.DataFrame) -> HeatmapData:
        """
        Generate liquidity heatmap by sector and rating
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            HeatmapData object
        """
        # Calculate liquidity index if not present
        if 'liquidity_index' not in df.columns:
            from .liquidity_insights import LiquidityInsightsEngine
            liq_engine = LiquidityInsightsEngine(seed=self.seed)
            df = df.copy()
            df['liquidity_index'] = liq_engine.calculate_liquidity_index(df)
        
        # Create pivot table: sector vs rating
        liquidity_pivot = df.pivot_table(
            values='liquidity_index',
            index='sector',
            columns='credit_rating',
            aggfunc='mean',
            fill_value=0
        )
        
        # Sort ratings in logical order
        rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
                       'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 
                       'B+', 'B', 'B-', 'CCC', 'CC', 'C']
        
        # Reorder columns to match rating order
        available_ratings = [r for r in rating_order if r in liquidity_pivot.columns]
        liquidity_pivot = liquidity_pivot[available_ratings]
        
        # Add annotations (count of bonds in each cell)
        count_pivot = df.pivot_table(
            values='issue_size',
            index='sector',
            columns='credit_rating',
            aggfunc='count',
            fill_value=0
        )
        
        return HeatmapData(
            heatmap_type=HeatmapType.LIQUIDITY,
            data_matrix=liquidity_pivot,
            row_labels=liquidity_pivot.index.tolist(),
            col_labels=liquidity_pivot.columns.tolist(),
            color_scale="RdYlGn",
            annotations=count_pivot.to_dict(),
            metadata={
                'group_by': 'sector_rating',
                'total_bonds': len(df)
            }
        )
    
    def identify_top_movers(self, df: pd.DataFrame, 
                           metric: str = 'spread_bps',
                           top_n: int = 10) -> pd.DataFrame:
        """
        Identify top movers for a given metric
        
        Args:
            df: DataFrame with bond data
            metric: Metric to analyze
            top_n: Number of top movers to return
            
        Returns:
            DataFrame with top movers
        """
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found in dataset")
        
        # Calculate percentage change (simulated for now)
        df_movers = df.copy()
        df_movers[f'{metric}_change_pct'] = np.random.normal(0, 0.15, len(df_movers))
        
        # Sort by absolute change
        df_movers['abs_change'] = abs(df_movers[f'{metric}_change_pct'])
        top_movers = df_movers.nlargest(top_n, 'abs_change')
        
        return top_movers[['issuer_name', 'sector', 'credit_rating', metric, 
                          f'{metric}_change_pct', 'abs_change']].sort_values('abs_change', ascending=False)
    
    def identify_deteriorations(self, df: pd.DataFrame, 
                              threshold_pct: float = -10.0) -> pd.DataFrame:
        """
        Identify bonds with significant deterioration
        
        Args:
            df: DataFrame with bond data
            threshold_pct: Threshold percentage for deterioration
            
        Returns:
            DataFrame with deteriorating bonds
        """
        # Calculate percentage changes (simulated)
        df_deteriorating = df.copy()
        df_deteriorating['spread_change_pct'] = np.random.normal(-5, 20, len(df_deteriorating))
        df_deteriorating['liquidity_change_pct'] = np.random.normal(-10, 15, len(df_deteriorating))
        
        # Filter for deteriorating bonds
        deteriorating_mask = (
            (df_deteriorating['spread_change_pct'] < threshold_pct) |
            (df_deteriorating['liquidity_change_pct'] < threshold_pct)
        )
        
        deteriorating_bonds = df_deteriorating[deteriorating_mask].copy()
        
        # Add severity classification
        deteriorating_bonds['severity'] = deteriorating_bonds.apply(
            lambda row: self._classify_deterioration_severity(row), axis=1
        )
        
        return deteriorating_bonds.sort_values('spread_change_pct')
    
    def _classify_deterioration_severity(self, row: pd.Series) -> str:
        """Classify deterioration severity"""
        spread_change = row.get('spread_change_pct', 0)
        liquidity_change = row.get('liquidity_change_pct', 0)
        
        if spread_change < -30 or liquidity_change < -30:
            return 'Critical'
        elif spread_change < -20 or liquidity_change < -20:
            return 'High'
        elif spread_change < -10 or liquidity_change < -10:
            return 'Medium'
        else:
            return 'Low'
    
    def check_thresholds(self, df: pd.DataFrame) -> List[ThresholdAlert]:
        """
        Check for threshold breaches across all metrics
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            List of threshold alerts
        """
        alerts = []
        
        # Calculate liquidity index if not present
        if 'liquidity_index' not in df.columns:
            from .liquidity_insights import LiquidityInsightsEngine
            liq_engine = LiquidityInsightsEngine(seed=self.seed)
            df = df.copy()
            df['liquidity_index'] = liq_engine.calculate_liquidity_index(df)
        
        # Calculate ESG scores if not present
        if 'esg_composite' not in df.columns:
            df = self.calculate_esg_scores(df)
        
        # Check each metric
        for metric, thresholds in self.risk_thresholds.items():
            if metric not in df.columns:
                continue
            
            for sector in df['sector'].unique():
                sector_data = df[df['sector'] == sector]
                
                for rating in sector_data['credit_rating'].unique():
                    rating_data = sector_data[sector_data['credit_rating'] == rating]
                    
                    if len(rating_data) == 0:
                        continue
                    
                    avg_value = rating_data[metric].mean()
                    
                    # Check thresholds
                    for level, threshold in thresholds.items():
                        if metric in ['liquidity_index', 'esg_score']:
                            # Lower values are worse for these metrics
                            if avg_value < threshold:
                                alert = ThresholdAlert(
                                    metric=metric,
                                    sector=sector,
                                    rating=rating,
                                    current_value=avg_value,
                                    threshold_value=threshold,
                                    breach_type="below",
                                    severity=RiskLevel(level.title()),
                                    timestamp=datetime.now(),
                                    description=f"{metric} below {level} threshold for {sector} {rating} bonds"
                                )
                                alerts.append(alert)
                        else:
                            # Higher values are worse for these metrics
                            if avg_value > threshold:
                                alert = ThresholdAlert(
                                    metric=metric,
                                    sector=sector,
                                    rating=rating,
                                    current_value=avg_value,
                                    threshold_value=threshold,
                                    breach_type="above",
                                    severity=RiskLevel(level.title()),
                                    timestamp=datetime.now(),
                                    description=f"{metric} above {level} threshold for {sector} {rating} bonds"
                                )
                                alerts.append(alert)
        
        return alerts
    
    def generate_heatmap_report(self, df: pd.DataFrame, 
                               output_dir: str = "outputs") -> Dict[str, str]:
        """
        Generate comprehensive heatmap report
        
        Args:
            df: DataFrame with bond data
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all heatmaps
        heatmaps = {
            'esg_sector': self.generate_esg_heatmap(df, 'sector'),
            'esg_rating': self.generate_esg_heatmap(df, 'credit_rating'),
            'yield_spread_sector': self.generate_yield_spread_heatmap(df, 'sector'),
            'yield_spread_rating': self.generate_yield_spread_heatmap(df, 'credit_rating'),
            'systemic_risk': self.generate_systemic_risk_heatmap(df),
            'liquidity': self.generate_liquidity_heatmap(df)
        }
        
        # Export heatmap data
        heatmap_data = {}
        for name, heatmap in heatmaps.items():
            heatmap_data[name] = {
                'type': heatmap.heatmap_type.value,
                'data': heatmap.data_matrix.to_dict(),
                'row_labels': heatmap.row_labels,
                'col_labels': heatmap.col_labels,
                'color_scale': heatmap.color_scale,
                'metadata': heatmap.metadata
            }
        
        # Export JSON
        json_path = output_path / "heatmap_data.json"
        with open(json_path, 'w') as f:
            json.dump(heatmap_data, f, indent=2, default=str)
        
        # Generate top movers and deteriorations
        top_movers = self.identify_top_movers(df)
        deteriorations = self.identify_deteriorations(df)
        
        # Export CSV files
        movers_path = output_path / "top_movers.csv"
        top_movers.to_csv(movers_path, index=False)
        
        deteriorations_path = output_path / "deteriorations.csv"
        deteriorations.to_csv(deteriorations_path, index=False)
        
        # Check thresholds
        threshold_alerts = self.check_thresholds(df)
        
        # Export alerts
        alerts_data = [
            {
                'metric': alert.metric,
                'sector': alert.sector,
                'rating': alert.rating,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'breach_type': alert.breach_type,
                'severity': alert.severity.value,
                'timestamp': alert.timestamp.isoformat(),
                'description': alert.description
            }
            for alert in threshold_alerts
        ]
        
        alerts_path = output_path / "threshold_alerts.json"
        with open(alerts_path, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
        
        # Generate summary report
        summary_data = {
            'heatmaps_generated': len(heatmaps),
            'top_movers_count': len(top_movers),
            'deteriorations_count': len(deteriorations),
            'threshold_alerts_count': len(threshold_alerts),
            'critical_alerts': len([a for a in threshold_alerts if a.severity == RiskLevel.CRITICAL]),
            'high_alerts': len([a for a in threshold_alerts if a.severity == RiskLevel.HIGH]),
            'generated_at': datetime.now().isoformat()
        }
        
        summary_path = output_path / "heatmap_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return {
            'heatmaps': str(json_path),
            'top_movers': str(movers_path),
            'deteriorations': str(deteriorations_path),
            'alerts': str(alerts_path),
            'summary': str(summary_path)
        }
    
    def plot_heatmap(self, heatmap_data: HeatmapData, 
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot a heatmap using matplotlib/seaborn
        
        Args:
            heatmap_data: HeatmapData object
            save_path: Optional path to save the plot
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data.data_matrix,
            annot=True,
            fmt='.3f',
            cmap=heatmap_data.color_scale,
            cbar_kws={'label': heatmap_data.heatmap_type.value.replace('_', ' ').title()},
            xticklabels=heatmap_data.col_labels,
            yticklabels=heatmap_data.row_labels
        )
        
        plt.title(f"{heatmap_data.heatmap_type.value.replace('_', ' ').title()} Heatmap")
        plt.xlabel("Metrics")
        plt.ylabel("Groups")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
