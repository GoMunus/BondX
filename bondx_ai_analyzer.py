#!/usr/bin/env python3
"""
BondX AI Liquidity Analysis System
==================================

AI-powered system for corporate bond market liquidity analysis and transparency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class LiquidityPulseScore:
    """Liquidity Pulse score with confidence bands"""
    overall_score: float  # 0-100
    liquidity_score: float  # 0-100
    credit_score: float  # 0-100
    market_score: float  # 0-100
    confidence_lower: float
    confidence_upper: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    trend: str  # IMPROVING, STABLE, DETERIORATING

@dataclass
class AnomalySignal:
    """Anomaly detection signal"""
    signal_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float  # 0-1
    description: str
    affected_bonds: List[str]
    recommended_action: str

class BondXAIliquidityAnalyzer:
    """Main BondX AI Liquidity Analysis System"""
    
    def __init__(self):
        """Initialize the BondX AI Liquidity Analyzer"""
        self.dataset = None
        self.liquidity_scores = {}
        self.anomaly_signals = []
        self.analysis_results = {}
        
        # Stakeholder configurations
        self.stakeholder_configs = {
            'regulator': {'include_technical': True, 'include_anomalies': True, 'include_confidence': True},
            'broker': {'include_technical': True, 'include_anomalies': True, 'include_confidence': False},
            'institutional_investor': {'include_technical': False, 'include_anomalies': True, 'include_confidence': True}
        }
        
        logger.info("BondX AI Liquidity Analyzer initialized")
    
    def load_dataset(self, dataset_path: str) -> bool:
        """Load and validate the corporate bond dataset"""
        try:
            # Try different file formats
            if dataset_path.endswith('.csv'):
                self.dataset = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.parquet'):
                self.dataset = pd.read_parquet(dataset_path)
            elif dataset_path.endswith('.xlsx'):
                self.dataset = pd.read_excel(dataset_path)
            else:
                logger.error(f"Unsupported file format: {dataset_path}")
                return False
            
            # Validate required columns
            required_columns = ['isin', 'spread_bps', 'yield_to_maturity', 'volume_traded']
            missing_columns = [col for col in required_columns if col not in self.dataset.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            logger.info(f"Dataset loaded successfully: {len(self.dataset)} records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return False
    
    def analyze_liquidity_patterns(self) -> Dict[str, Any]:
        """Analyze liquidity patterns in the dataset"""
        if self.dataset is None:
            logger.error("No dataset loaded")
            return {}
        
        try:
            analysis = {
                'summary': {},
                'sector_analysis': {},
                'rating_analysis': {},
                'correlation_analysis': {}
            }
            
            # Basic statistics
            analysis['summary'] = {
                'total_bonds': len(self.dataset),
                'avg_spread': self.dataset['spread_bps'].mean(),
                'avg_yield': self.dataset['yield_to_maturity'].mean(),
                'avg_volume': self.dataset['volume_traded'].mean(),
                'spread_volatility': self.dataset['spread_bps'].std(),
                'yield_volatility': self.dataset['yield_to_maturity'].std()
            }
            
            # Sector analysis
            if 'sector' in self.dataset.columns:
                sector_stats = self.dataset.groupby('sector').agg({
                    'spread_bps': ['mean', 'std', 'count'],
                    'yield_to_maturity': ['mean', 'std'],
                    'volume_traded': ['mean', 'sum']
                }).round(2)
                analysis['sector_analysis'] = sector_stats.to_dict()
            
            # Rating analysis
            if 'credit_rating' in self.dataset.columns:
                rating_stats = self.dataset.groupby('credit_rating').agg({
                    'spread_bps': ['mean', 'std', 'count'],
                    'yield_to_maturity': ['mean', 'std']
                }).round(2)
                analysis['rating_analysis'] = rating_stats.to_dict()
            
            # Correlation analysis
            numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                correlation_matrix = self.dataset[numeric_columns].corr().round(3)
                analysis['correlation_analysis'] = correlation_matrix.to_dict()
            
            self.analysis_results['liquidity_patterns'] = analysis
            logger.info("Liquidity pattern analysis completed")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze liquidity patterns: {str(e)}")
            return {}
    
    def detect_anomalies(self, threshold: float = 2.0) -> List[AnomalySignal]:
        """Detect anomalies in the dataset using statistical methods"""
        if self.dataset is None:
            logger.error("No dataset loaded")
            return []
        
        try:
            signals = []
            
            # Spread anomalies
            spread_mean = self.dataset['spread_bps'].mean()
            spread_std = self.dataset['spread_bps'].std()
            spread_z_scores = np.abs((self.dataset['spread_bps'] - spread_mean) / spread_std)
            
            spread_anomalies = self.dataset[spread_z_scores > threshold]
            if len(spread_anomalies) > 0:
                signal = AnomalySignal(
                    signal_type="SPREAD_WIDENING",
                    severity="HIGH" if threshold > 3.0 else "MEDIUM",
                    confidence=min(0.95, 1 - (1 / (threshold + 1))),
                    description=f"Detected {len(spread_anomalies)} bonds with unusually wide spreads",
                    affected_bonds=spread_anomalies['isin'].tolist(),
                    recommended_action="Review credit quality and market conditions for affected bonds"
                )
                signals.append(signal)
            
            # Volume anomalies
            if 'volume_traded' in self.dataset.columns:
                volume_mean = self.dataset['volume_traded'].mean()
                volume_std = self.dataset['volume_traded'].std()
                if volume_std > 0:
                    volume_z_scores = np.abs((self.dataset['volume_traded'] - volume_mean) / volume_std)
                    volume_anomalies = self.dataset[volume_z_scores > threshold]
                    
                    if len(volume_anomalies) > 0:
                        signal = AnomalySignal(
                            signal_type="VOLUME_SPIKE",
                            severity="MEDIUM",
                            confidence=min(0.90, 1 - (1 / (threshold + 1))),
                            description=f"Detected {len(volume_anomalies)} bonds with unusual trading volume",
                            affected_bonds=volume_anomalies['isin'].tolist(),
                            recommended_action="Monitor for potential market events or news affecting these bonds"
                        )
                        signals.append(signal)
            
            # Yield anomalies
            yield_mean = self.dataset['yield_to_maturity'].mean()
            yield_std = self.dataset['yield_to_maturity'].std()
            yield_z_scores = np.abs((self.dataset['yield_to_maturity'] - yield_mean) / yield_std)
            
            yield_anomalies = self.dataset[yield_z_scores > threshold]
            if len(yield_anomalies) > 0:
                signal = AnomalySignal(
                    signal_type="YIELD_ANOMALY",
                    severity="HIGH" if threshold > 3.0 else "MEDIUM",
                    confidence=min(0.92, 1 - (1 / (threshold + 1))),
                    description=f"Detected {len(yield_anomalies)} bonds with unusual yields",
                    affected_bonds=yield_anomalies['isin'].tolist(),
                    recommended_action="Investigate fundamental changes or market mispricing"
                )
                signals.append(signal)
            
            self.anomaly_signals = signals
            logger.info(f"Anomaly detection completed: {len(signals)} signals found")
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            return []
    
    def calculate_liquidity_pulse_scores(self) -> Dict[str, LiquidityPulseScore]:
        """Calculate Liquidity Pulse scores for all bonds"""
        if self.dataset is None:
            logger.error("No dataset loaded")
            return {}
        
        try:
            scores = {}
            
            for _, bond in self.dataset.iterrows():
                # Calculate component scores
                liquidity_score = self._calculate_liquidity_component(bond)
                credit_score = self._calculate_credit_component(bond)
                market_score = self._calculate_market_component(bond)
                
                # Overall score (weighted average)
                overall_score = (liquidity_score * 0.4 + 
                               credit_score * 0.35 + 
                               market_score * 0.25)
                
                # Confidence intervals
                confidence_range = self._calculate_confidence_interval(bond)
                
                # Risk level and trend
                risk_level = self._determine_risk_level(overall_score)
                trend = "STABLE"  # Simplified for now
                
                # Create score object
                pulse_score = LiquidityPulseScore(
                    overall_score=round(overall_score, 2),
                    liquidity_score=round(liquidity_score, 2),
                    credit_score=round(credit_score, 2),
                    market_score=round(market_score, 2),
                    confidence_lower=confidence_range[0],
                    confidence_upper=confidence_range[1],
                    risk_level=risk_level,
                    trend=trend
                )
                
                scores[bond['isin']] = pulse_score
            
            self.liquidity_scores = scores
            logger.info(f"Liquidity Pulse scores calculated for {len(scores)} bonds")
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate liquidity pulse scores: {str(e)}")
            return {}
    
    def _calculate_liquidity_component(self, bond: pd.Series) -> float:
        """Calculate liquidity component score (0-100)"""
        try:
            # Normalize spread (lower is better)
            max_spread = self.dataset['spread_bps'].max()
            spread_score = max(0, 100 - (bond['spread_bps'] / max_spread) * 100)
            
            # Normalize volume (higher is better)
            max_volume = self.dataset['volume_traded'].max()
            volume_score = (bond['volume_traded'] / max_volume) * 100 if max_volume > 0 else 0
            
            # Weighted average
            liquidity_score = (spread_score * 0.6 + volume_score * 0.4)
            return max(0, min(100, liquidity_score))
            
        except Exception:
            return 50.0
    
    def _calculate_credit_component(self, bond: pd.Series) -> float:
        """Calculate credit component score (0-100)"""
        try:
            # Rating-based scoring
            rating_scores = {
                'AAA': 95, 'AA+': 92, 'AA': 90, 'AA-': 88,
                'A+': 85, 'A': 82, 'A-': 80,
                'BBB+': 75, 'BBB': 72, 'BBB-': 70,
                'BB+': 65, 'BB': 62, 'BB-': 60,
                'B+': 55, 'B': 52, 'B-': 50,
                'CCC+': 45, 'CCC': 42, 'CCC-': 40,
                'CC': 35, 'C': 30, 'D': 10
            }
            
            if 'credit_rating' in bond and bond['credit_rating'] in rating_scores:
                base_score = rating_scores[bond['credit_rating']]
            else:
                base_score = 70
            
            # Yield adjustment
            yield_mean = self.dataset['yield_to_maturity'].mean()
            yield_std = self.dataset['yield_to_maturity'].std()
            
            if yield_std > 0:
                yield_z_score = (bond['yield_to_maturity'] - yield_mean) / yield_std
                yield_adjustment = max(-20, min(20, -yield_z_score * 5))
            else:
                yield_adjustment = 0
            
            credit_score = base_score + yield_adjustment
            return max(0, min(100, credit_score))
            
        except Exception:
            return 70.0
    
    def _calculate_market_component(self, bond: pd.Series) -> float:
        """Calculate market component score (0-100)"""
        try:
            # Sector component
            sector_score = 75  # Default
            if 'sector' in bond and bond['sector']:
                sector_scores = {
                    'GOVERNMENT': 90, 'FINANCIAL': 80, 'UTILITIES': 85,
                    'INDUSTRIAL': 75, 'TECHNOLOGY': 70, 'ENERGY': 80
                }
                sector_score = sector_scores.get(bond['sector'], 75)
            
            return sector_score
            
        except Exception:
            return 75.0
    
    def _calculate_confidence_interval(self, bond: pd.Series) -> Tuple[float, float]:
        """Calculate confidence interval for the score"""
        try:
            # Base confidence
            base_confidence = 0.85
            
            # Data quality adjustment
            data_quality = 1.0
            for field in ['spread_bps', 'yield_to_maturity', 'volume_traded']:
                if pd.isna(bond[field]):
                    data_quality -= 0.2
            
            final_confidence = base_confidence * data_quality
            final_confidence = max(0.5, min(0.95, final_confidence))
            
            # Calculate interval
            score = self._calculate_liquidity_component(bond)
            interval_width = (1 - final_confidence) * 20
            
            lower = max(0, score - interval_width)
            upper = min(100, score + interval_width)
            
            return round(lower, 1), round(upper, 1)
            
        except Exception:
            return 40.0, 60.0
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on score"""
        if score >= 80:
            return "LOW"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def generate_regulator_insights(self, stakeholder: str = 'regulator') -> Dict[str, Any]:
        """Generate regulator-friendly insights and recommendations"""
        try:
            config = self.stakeholder_configs.get(stakeholder, self.stakeholder_configs['regulator'])
            
            insights = {
                'summary': self._generate_executive_summary(),
                'key_metrics': self._generate_key_metrics_table(),
                'risk_signals': self._generate_risk_signals_summary(config),
                'recommendations': self._generate_recommendations(config),
                'stakeholder': stakeholder,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Regulator insights generated for {stakeholder}")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate regulator insights: {str(e)}")
            return {}
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary (2-3 lines)"""
        if not self.analysis_results.get('liquidity_patterns'):
            return "Analysis not yet performed. Please run liquidity pattern analysis first."
        
        summary = self.analysis_results['liquidity_patterns']['summary']
        
        summary_text = (
            f"BondX AI analysis of {summary['total_bonds']} corporate bonds reveals "
            f"average spreads of {summary['avg_spread']:.1f} bps with "
            f"{summary['spread_volatility']:.1f} bps volatility. "
            f"Market liquidity shows {len(self.anomaly_signals)} anomaly signals "
            f"requiring regulatory attention."
        )
        
        return summary_text
    
    def _generate_key_metrics_table(self) -> Dict[str, Any]:
        """Generate key metrics table"""
        if not self.analysis_results.get('liquidity_patterns'):
            return {}
        
        summary = self.analysis_results['liquidity_patterns']['summary']
        
        metrics = {
            'portfolio_metrics': {
                'Total Bonds': summary['total_bonds'],
                'Average Spread (bps)': f"{summary['avg_spread']:.1f}",
                'Average Yield (%)': f"{summary['avg_yield']:.2f}",
                'Average Volume': f"{summary['avg_volume']:,.0f}",
                'Spread Volatility (bps)': f"{summary['spread_volatility']:.1f}",
                'Yield Volatility (%)': f"{summary['yield_volatility']:.2f}"
            },
            'risk_metrics': {
                'High Risk Bonds': len([s for s in self.liquidity_scores.values() if s.risk_level in ['HIGH', 'CRITICAL']]),
                'Medium Risk Bonds': len([s for s in self.liquidity_scores.values() if s.risk_level == 'MEDIUM']),
                'Low Risk Bonds': len([s for s in self.liquidity_scores.values() if s.risk_level == 'LOW']),
                'Anomaly Signals': len(self.anomaly_signals)
            }
        }
        
        return metrics
    
    def _generate_risk_signals_summary(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk signals summary"""
        signals_summary = []
        
        for signal in self.anomaly_signals:
            if config['include_anomalies']:
                signal_info = {
                    'type': signal.signal_type,
                    'severity': signal.severity,
                    'description': signal.description,
                    'affected_bonds_count': len(signal.affected_bonds),
                    'recommended_action': signal.recommended_action,
                    'confidence': f"{signal.confidence:.1%}"
                }
            else:
                signal_info = {
                    'type': signal.signal_type,
                    'severity': signal.severity,
                    'affected_bonds_count': len(signal.affected_bonds)
                }
            
            signals_summary.append(signal_info)
        
        return signals_summary
    
    def _generate_recommendations(self, config: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Portfolio-level recommendations
        high_risk_count = len([s for s in self.liquidity_scores.values() if s.risk_level in ['HIGH', 'CRITICAL']])
        total_bonds = len(self.liquidity_scores)
        
        if high_risk_count / total_bonds > 0.2:
            recommendations.append("Consider portfolio rebalancing to reduce high-risk bond exposure")
        
        if len(self.anomaly_signals) > 5:
            recommendations.append("Implement enhanced monitoring for bonds with anomaly signals")
        
        # Sector-specific recommendations
        if self.analysis_results.get('liquidity_patterns', {}).get('sector_analysis'):
            sector_data = self.analysis_results['liquidity_patterns']['sector_analysis']
            for sector, stats in sector_data.items():
                if isinstance(stats, dict) and 'spread_bps' in stats:
                    avg_spread = stats['spread_bps']['mean']
                    if avg_spread > 200:  # High spread threshold
                        recommendations.append(f"Review {sector} sector bonds for potential credit quality issues")
        
        return recommendations
    
    def create_visualizations(self) -> Dict[str, Any]:
        """Create comprehensive visualizations for the analysis"""
        if self.dataset is None:
            logger.error("No dataset loaded")
            return {}
        
        try:
            plots = {}
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Liquidity Pulse Score Distribution
            if self.liquidity_scores:
                scores = list(self.liquidity_scores.values())
                overall_scores = [s.overall_score for s in scores]
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.hist(overall_scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                ax1.set_title('Distribution of Liquidity Pulse Scores', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Score (0-100)', fontsize=12)
                ax1.set_ylabel('Number of Bonds', fontsize=12)
                ax1.grid(True, alpha=0.3)
                plots['score_distribution'] = fig1
            
            # 2. Spread vs Yield Scatter Plot
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            scatter = ax2.scatter(
                self.dataset['yield_to_maturity'],
                self.dataset['spread_bps'],
                c=self.dataset['volume_traded'],
                s=50,
                alpha=0.6,
                cmap='viridis'
            )
            ax2.set_title('Yield vs Spread Analysis', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Yield to Maturity (%)', fontsize=12)
            ax2.set_ylabel('Spread (bps)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Volume', fontsize=12)
            plots['yield_spread_scatter'] = fig2
            
            # 3. Risk Level Distribution
            if self.liquidity_scores:
                risk_counts = {}
                for score in self.liquidity_scores.values():
                    risk_counts[score.risk_level] = risk_counts.get(score.risk_level, 0) + 1
                
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                colors = ['green', 'yellow', 'orange', 'red']
                wedges, texts, autotexts = ax3.pie(
                    list(risk_counts.values()),
                    labels=list(risk_counts.keys()),
                    autopct='%1.1f%%',
                    colors=colors[:len(risk_counts)],
                    startangle=90
                )
                ax3.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
                plots['risk_distribution'] = fig3
            
            logger.info(f"Visualizations created: {len(plots)} plots")
            return plots
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {str(e)}")
            return {}

def main():
    """Main function to demonstrate BondX AI Liquidity Analyzer"""
    print("=" * 60)
    print("BondX AI Liquidity Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = BondXAIliquidityAnalyzer()
    
    print("\n1. System Initialization")
    print("   ✓ BondX AI Liquidity Analyzer initialized")
    print("   ✓ Ready to analyze corporate bond datasets")
    
    print("\n2. Available Capabilities")
    print("   ✓ Liquidity pattern analysis")
    print("   ✓ Anomaly detection and early warning signals")
    print("   ✓ Liquidity Pulse scoring with confidence bands")
    print("   ✓ Regulator-friendly insights generation")
    print("   ✓ Multi-stakeholder reporting")
    print("   ✓ Comprehensive visualizations")
    
    print("\n3. Usage Instructions")
    print("   • Load dataset: analyzer.load_dataset('path/to/dataset')")
    print("   • Analyze patterns: analyzer.analyze_liquidity_patterns()")
    print("   • Detect anomalies: analyzer.detect_anomalies()")
    print("   • Calculate scores: analyzer.calculate_liquidity_pulse_scores()")
    print("   • Generate insights: analyzer.generate_regulator_insights('regulator')")
    print("   • Create visualizations: analyzer.create_visualizations()")
    
    print("\n4. Stakeholder Views")
    print("   • Regulator: Full technical details + anomaly details + confidence intervals")
    print("   • Broker: Technical details + anomaly details (no confidence intervals)")
    print("   • Institutional Investor: Anomaly details + confidence intervals (no technical details)")
    
    print("\n" + "=" * 60)
    print("Ready to enhance corporate bond market liquidity and transparency!")
    print("=" * 60)

if __name__ == "__main__":
    main()
