"""
Liquidity-Risk Orchestrator for BondX

This module orchestrates the integration of risk assessment and liquidity intelligence
to provide a unified narrative for retail investors and market makers.
"""

import hashlib
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from copy import deepcopy

from .liquidity_intelligence_service import (
    LiquidityIntelligenceService, LiquidityProfile, MarketMicrostructure,
    AuctionSignals, MarketMakerState
)
from .exit_recommender import (
    ExitRecommender, ExitAnalysis, ExitRecommendation
)
from .risk_scoring import RiskScoringEngine, RiskScore

logger = logging.getLogger(__name__)

class NarrativeMode(Enum):
    """Narrative generation modes."""
    RETAIL = "retail"          # Plain English for retail investors
    PROFESSIONAL = "professional"  # Technical details for professionals
    COMPLIANCE = "compliance"   # Compliance-focused narrative

class RiskCategory(Enum):
    """Risk categories for scoring."""
    LIQUIDITY = "liquidity"
    REFINANCING = "refinancing"
    LEVERAGE = "leverage"
    GOVERNANCE = "governance"
    LEGAL = "legal"
    ESG = "esg"

@dataclass
class RiskCategoryScore:
    """Risk score for a specific category."""
    name: str
    score_0_100: float
    level: str  # low, medium, high, critical
    probability_note: str
    citations: List[str]
    confidence: float

@dataclass
class RiskSummary:
    """Risk summary for a bond."""
    overall_score: float
    categories: List[RiskCategoryScore]
    confidence: float
    citations: List[str]
    last_updated: datetime
    methodology_version: str

@dataclass
class LiquidityRiskTranslation:
    """Complete liquidity-risk translation for a bond."""
    isin: str
    as_of: datetime
    
    # Risk assessment
    risk_summary: RiskSummary
    
    # Liquidity profile
    liquidity_profile: LiquidityProfile
    
    # Exit recommendations
    exit_recommendations: List[ExitRecommendation]
    
    # Narrative elements
    retail_narrative: str
    professional_summary: str
    risk_warnings: List[str]
    
    # Metadata
    confidence_overall: float
    data_freshness: str
    inputs_hash: str
    model_versions: Dict[str, str]
    caveats: List[str]

class LiquidityRiskOrchestrator:
    """
    Orchestrates the integration of risk and liquidity intelligence.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.liquidity_service = LiquidityIntelligenceService(config)
        self.exit_recommender = ExitRecommender(config)
        self.risk_engine = RiskScoringEngine(config)
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize all required services."""
        logger.info("Initializing Liquidity-Risk Orchestrator")
        
    def create_liquidity_risk_translation(self,
                                        isin: str,
                                        microstructure: MarketMicrostructure,
                                        risk_data: Dict[str, Any],
                                        auction_signals: Optional[AuctionSignals] = None,
                                        mm_state: Optional[MarketMakerState] = None,
                                        bond_metadata: Dict[str, Any] = None,
                                        trade_size: float = 100000,
                                        mode: NarrativeMode = NarrativeMode.RETAIL) -> LiquidityRiskTranslation:
        """
        Create comprehensive liquidity-risk translation for a bond.
        
        Args:
            isin: Bond ISIN
            microstructure: Market microstructure data
            risk_data: Risk assessment data
            auction_signals: Optional auction telemetry
            mm_state: Optional market maker state
            bond_metadata: Bond characteristics
            trade_size: Size of position to exit
            mode: Narrative generation mode
            
        Returns:
            Complete liquidity-risk translation
        """
        try:
            # Create liquidity profile
            liquidity_profile = self.liquidity_service.create_liquidity_profile(
                isin, microstructure, auction_signals, mm_state
            )
            
            # Create exit analysis
            exit_analysis = self.exit_recommender.recommend_exit_paths(
                isin, microstructure, auction_signals, mm_state, bond_metadata, trade_size
            )
            
            # Create risk summary
            risk_summary = self._create_risk_summary(risk_data, bond_metadata)
            
            # Generate narratives
            retail_narrative = self._generate_retail_narrative(
                risk_summary, liquidity_profile, exit_analysis, bond_metadata
            )
            
            professional_summary = self._generate_professional_summary(
                risk_summary, liquidity_profile, exit_analysis, bond_metadata
            )
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(
                risk_summary, liquidity_profile, exit_analysis, bond_metadata
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                risk_summary, liquidity_profile, exit_analysis
            )
            
            # Generate inputs hash for auditability
            inputs_hash = self._generate_inputs_hash(
                microstructure, risk_data, auction_signals, mm_state, bond_metadata
            )
            
            # Create translation
            translation = LiquidityRiskTranslation(
                isin=isin,
                as_of=datetime.now(),
                risk_summary=risk_summary,
                liquidity_profile=liquidity_profile,
                exit_recommendations=exit_analysis.recommendations,
                retail_narrative=retail_narrative,
                professional_summary=professional_summary,
                risk_warnings=risk_warnings,
                confidence_overall=overall_confidence,
                data_freshness=self._assess_overall_data_freshness(
                    microstructure.timestamp, risk_summary.last_updated
                ),
                inputs_hash=inputs_hash,
                model_versions=self._get_model_versions(),
                caveats=self._generate_caveats(risk_summary, liquidity_profile, exit_analysis)
            )
            
            logger.info(f"Created liquidity-risk translation for {isin}")
            return translation
            
        except Exception as e:
            logger.error(f"Error creating liquidity-risk translation for {isin}: {e}")
            raise
    
    def _create_risk_summary(self, risk_data: Dict[str, Any], 
                            bond_metadata: Dict[str, Any]) -> RiskSummary:
        """Create risk summary from risk data."""
        try:
            # Extract risk scores from data
            categories = []
            
            # Liquidity risk (from liquidity profile)
            liquidity_score = risk_data.get('liquidity_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Liquidity Risk",
                score_0_100=liquidity_score,
                level=self._classify_risk_level(liquidity_score),
                probability_note="Based on market depth, spread, and trading activity",
                citations=["Market microstructure analysis", "Liquidity intelligence service"],
                confidence=0.85
            ))
            
            # Credit risk
            credit_score = risk_data.get('credit_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Credit Risk",
                score_0_100=credit_score,
                level=self._classify_risk_level(credit_score),
                probability_note="Based on issuer rating and financial metrics",
                citations=["Credit rating agencies", "Financial statement analysis"],
                confidence=0.90
            ))
            
            # Interest rate risk
            ir_score = risk_data.get('interest_rate_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Interest Rate Risk",
                score_0_100=ir_score,
                level=self._classify_risk_level(ir_score),
                probability_note="Based on duration and yield curve sensitivity",
                citations=["Duration analysis", "Yield curve modeling"],
                confidence=0.80
            ))
            
            # Refinancing risk
            refi_score = risk_data.get('refinancing_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Refinancing Risk",
                score_0_100=refi_score,
                level=self._classify_risk_level(refi_score),
                probability_note="Based on maturity profile and market access",
                citations=["Maturity analysis", "Market access assessment"],
                confidence=0.75
            ))
            
            # Leverage risk
            leverage_score = risk_data.get('leverage_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Leverage Risk",
                score_0_100=leverage_score,
                level=self._classify_risk_level(leverage_score),
                probability_note="Based on debt-to-equity and coverage ratios",
                citations=["Financial ratios", "Balance sheet analysis"],
                confidence=0.80
            ))
            
            # Governance risk
            gov_score = risk_data.get('governance_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Governance Risk",
                score_0_100=gov_score,
                level=self._classify_risk_level(gov_score),
                probability_note="Based on board composition and policies",
                citations=["Corporate governance assessment", "Board analysis"],
                confidence=0.70
            ))
            
            # Legal risk
            legal_score = risk_data.get('legal_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="Legal Risk",
                score_0_100=legal_score,
                level=self._classify_risk_level(legal_score),
                probability_note="Based on regulatory compliance and litigation",
                citations=["Regulatory analysis", "Legal risk assessment"],
                confidence=0.75
            ))
            
            # ESG risk
            esg_score = risk_data.get('esg_risk_score', 50.0)
            categories.append(RiskCategoryScore(
                name="ESG Risk",
                score_0_100=esg_score,
                level=self._classify_risk_level(esg_score),
                probability_note="Based on environmental, social, and governance factors",
                citations=["ESG scoring", "Sustainability assessment"],
                confidence=0.70
            ))
            
            # Calculate overall score (weighted average)
            weights = [0.25, 0.25, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05]
            overall_score = sum(
                cat.score_0_100 * weight 
                for cat, weight in zip(categories, weights)
            )
            
            # Calculate overall confidence
            overall_confidence = sum(
                cat.confidence * weight 
                for cat, weight in zip(categories, weights)
            )
            
            return RiskSummary(
                overall_score=overall_score,
                categories=categories,
                confidence=overall_confidence,
                citations=["Comprehensive risk assessment", "Multi-factor analysis"],
                last_updated=datetime.now(),
                methodology_version="1.0"
            )
            
        except Exception as e:
            logger.error(f"Error creating risk summary: {e}")
            raise
    
    def _classify_risk_level(self, score: float) -> str:
        """Classify risk level based on score."""
        if score <= 25:
            return "low"
        elif score <= 50:
            return "medium"
        elif score <= 75:
            return "high"
        else:
            return "critical"
    
    def _generate_retail_narrative(self,
                                  risk_summary: RiskSummary,
                                  liquidity_profile: LiquidityProfile,
                                  exit_analysis: ExitAnalysis,
                                  bond_metadata: Dict[str, Any]) -> str:
        """Generate plain-English narrative for retail investors."""
        try:
            # Start with overall risk assessment
            risk_level = self._classify_risk_level(risk_summary.overall_score)
            risk_description = self._get_risk_description(risk_level)
            
            # Add liquidity context
            liquidity_level = liquidity_profile.liquidity_level.value
            liquidity_description = self._get_liquidity_description(liquidity_level)
            
            # Add exit pathway summary
            if exit_analysis.recommendations:
                primary_path = exit_analysis.recommendations[0]
                exit_summary = self._get_exit_path_summary(primary_path)
            else:
                exit_summary = "No immediate exit options available"
            
            # Construct narrative
            narrative = f"""
            This bond presents {risk_description} overall risk profile. 
            
            {risk_description.capitalize()} risk means {self._get_risk_explanation(risk_level)}.
            
            From a liquidity perspective, this bond has {liquidity_description} market activity. 
            {liquidity_description.capitalize()} liquidity means {self._get_liquidity_explanation(liquidity_level)}.
            
            For exiting your position, {exit_summary}
            
            Key considerations:
            • Credit quality: {self._get_rating_description(bond_metadata.get('rating', 'BBB'))}
            • Time to exit: {exit_analysis.expected_best_exit_time:.0f} minutes (estimated)
            • Expected price impact: {primary_path.expected_spread_bps:.1f} basis points
            
            Remember: These assessments are informational and not trading advice. 
            Market conditions can change rapidly, affecting both risk and liquidity.
            """
            
            return narrative.strip()
            
        except Exception as e:
            logger.error(f"Error generating retail narrative: {e}")
            return "Unable to generate narrative at this time."
    
    def _generate_professional_summary(self,
                                     risk_summary: RiskSummary,
                                     liquidity_profile: LiquidityProfile,
                                     exit_analysis: ExitAnalysis,
                                     bond_metadata: Dict[str, Any]) -> str:
        """Generate technical summary for professionals."""
        try:
            summary = f"""
            LIQUIDITY-RISK ANALYSIS SUMMARY
            
            RISK ASSESSMENT:
            Overall Score: {risk_summary.overall_score:.1f}/100 (Confidence: {risk_summary.confidence:.1%})
            
            Category Breakdown:
            {self._format_category_breakdown(risk_summary.categories)}
            
            LIQUIDITY PROFILE:
            Liquidity Index: {liquidity_profile.liquidity_index:.1f}/100
            Spread: {liquidity_profile.spread_bps:.1f} bps
            Depth Score: {liquidity_profile.depth_score:.1f}/100
            Expected TTE: {liquidity_profile.expected_time_to_exit_minutes:.1f} min
            
            EXIT PATHWAYS:
            {self._format_exit_pathways(exit_analysis.recommendations)}
            
            DATA QUALITY:
            Risk Data Freshness: {risk_summary.last_updated.strftime('%Y-%m-%d %H:%M:%S')}
            Market Data Freshness: {liquidity_profile.data_freshness}
            Overall Confidence: {exit_analysis.overall_confidence:.1%}
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating professional summary: {e}")
            return "Unable to generate professional summary."
    
    def _get_risk_description(self, risk_level: str) -> str:
        """Get plain-English risk description."""
        descriptions = {
            "low": "a low",
            "medium": "a moderate",
            "high": "an elevated",
            "critical": "a critical"
        }
        return descriptions.get(risk_level, "an unknown")
    
    def _get_risk_explanation(self, risk_level: str) -> str:
        """Get explanation of what the risk level means."""
        explanations = {
            "low": "the bond is considered relatively safe with minimal expected losses",
            "medium": "the bond has some risk factors that warrant attention",
            "high": "the bond has significant risk factors that require careful consideration",
            "critical": "the bond has severe risk factors that may lead to substantial losses"
        }
        return explanations.get(risk_level, "the risk level requires further assessment")
    
    def _get_liquidity_description(self, liquidity_level: str) -> str:
        """Get plain-English liquidity description."""
        descriptions = {
            "excellent": "excellent",
            "good": "good",
            "moderate": "moderate",
            "poor": "poor",
            "illiquid": "very limited"
        }
        return descriptions.get(liquidity_level, "unknown")
    
    def _get_liquidity_explanation(self, liquidity_level: str) -> str:
        """Get explanation of what the liquidity level means."""
        explanations = {
            "excellent": "you can easily buy or sell this bond with minimal price impact",
            "good": "you can trade this bond with reasonable ease and moderate price impact",
            "moderate": "trading this bond may take some time and have noticeable price impact",
            "poor": "trading this bond may be difficult with significant price impact",
            "illiquid": "trading this bond may be very difficult with substantial price impact"
        }
        return explanations.get(liquidity_level, "the liquidity level requires further assessment")
    
    def _get_exit_path_summary(self, recommendation: ExitRecommendation) -> str:
        """Get summary of exit pathway."""
        path_descriptions = {
            "market_maker": "Market Maker Quote",
            "auction": "Fractional Auction",
            "rfq_batch": "RFQ Batch Window",
            "tokenized_p2p": "Tokenized P2P"
        }
        
        path_name = path_descriptions.get(recommendation.path.value, recommendation.path.value)
        
        return f"the recommended approach is {path_name} with {recommendation.fill_probability:.1%} fill probability " \
               f"and expected completion in {recommendation.expected_time_to_exit_minutes:.0f} minutes"
    
    def _get_rating_description(self, rating: str) -> str:
        """Get plain-English rating description."""
        rating_descriptions = {
            "AAA": "Highest quality with minimal credit risk",
            "AA": "High quality with very low credit risk",
            "A": "Good quality with low credit risk",
            "BBB": "Adequate quality with moderate credit risk",
            "BB": "Speculative quality with significant credit risk",
            "B": "Highly speculative with high credit risk",
            "C": "Very high credit risk",
            "D": "Default or near default"
        }
        return rating_descriptions.get(rating, "Rating requires further assessment")
    
    def _format_category_breakdown(self, categories: List[RiskCategoryScore]) -> str:
        """Format risk category breakdown for professional summary."""
        lines = []
        for cat in categories:
            lines.append(f"  • {cat.name}: {cat.score_0_100:.1f}/100 ({cat.level}) - {cat.confidence:.1%} confidence")
        return "\n".join(lines)
    
    def _format_exit_pathways(self, recommendations: List[ExitRecommendation]) -> str:
        """Format exit pathway analysis for professional summary."""
        if not recommendations:
            return "No viable exit paths available"
        
        lines = []
        for i, rec in enumerate(recommendations, 1):
            priority = rec.priority.value.capitalize()
            path = rec.path.value.replace('_', ' ').title()
            lines.append(f"  {i}. {priority}: {path}")
            lines.append(f"     Fill Probability: {rec.fill_probability:.1%}")
            lines.append(f"     Expected TTE: {rec.expected_time_to_exit_minutes:.1f} min")
            lines.append(f"     Expected Spread: {rec.expected_spread_bps:.1f} bps")
            if rec.constraints:
                constraints = [c.value.replace('_', ' ').title() for c in rec.constraints]
                lines.append(f"     Constraints: {', '.join(constraints)}")
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def _generate_risk_warnings(self,
                               risk_summary: RiskSummary,
                               liquidity_profile: LiquidityProfile,
                               exit_analysis: ExitAnalysis,
                               bond_metadata: Dict[str, Any]) -> List[str]:
        """Generate comprehensive risk warnings."""
        warnings = []
        
        # Risk-based warnings
        if risk_summary.overall_score > 75:
            warnings.append(f"Critical risk level ({risk_summary.overall_score:.1f}/100) - immediate attention required")
        
        # Liquidity-based warnings
        if liquidity_profile.liquidity_level.value in ['poor', 'illiquid']:
            warnings.append(f"Limited liquidity - exit may be difficult and costly")
        
        # Exit path warnings
        if exit_analysis.risk_warnings:
            warnings.extend(exit_analysis.risk_warnings)
        
        # Rating-based warnings
        rating = bond_metadata.get('rating', 'BBB')
        if rating in ['BB', 'B', 'C', 'D']:
            warnings.append(f"High credit risk rating ({rating}) - potential for significant losses")
        
        # Data quality warnings
        if liquidity_profile.data_freshness in ['stale', 'outdated']:
            warnings.append(f"Market data may be outdated - current conditions may differ")
        
        return warnings
    
    def _calculate_overall_confidence(self,
                                    risk_summary: RiskSummary,
                                    liquidity_profile: LiquidityProfile,
                                    exit_analysis: ExitAnalysis) -> float:
        """Calculate overall confidence in the translation."""
        # Weight the confidence scores
        risk_weight = 0.4
        liquidity_weight = 0.3
        exit_weight = 0.3
        
        overall_confidence = (
            risk_summary.confidence * risk_weight +
            liquidity_profile.confidence * liquidity_weight +
            exit_analysis.overall_confidence * exit_weight
        )
        
        return min(1.0, overall_confidence)
    
    def _generate_inputs_hash(self,
                             microstructure: MarketMicrostructure,
                             risk_data: Dict[str, Any],
                             auction_signals: Optional[AuctionSignals],
                             mm_state: Optional[MarketMakerState],
                             bond_metadata: Dict[str, Any]) -> str:
        """Generate hash of inputs for auditability."""
        try:
            # Create a summary of key inputs
            inputs_summary = {
                'microstructure': {
                    'timestamp': microstructure.timestamp.isoformat(),
                    'bid': microstructure.bid,
                    'ask': microstructure.ask,
                    'spread_bps': microstructure.spread_bps
                },
                'risk_data_keys': list(risk_data.keys()),
                'auction_signals': auction_signals is not None,
                'mm_state': mm_state is not None,
                'bond_metadata_keys': list(bond_metadata.keys()) if bond_metadata else []
            }
            
            # Convert to JSON and hash
            inputs_json = json.dumps(inputs_summary, sort_keys=True)
            inputs_hash = hashlib.sha256(inputs_json.encode()).hexdigest()[:16]
            
            return inputs_hash
            
        except Exception as e:
            logger.error(f"Error generating inputs hash: {e}")
            return "hash_error"
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of models used in the analysis."""
        return {
            "liquidity_intelligence": "1.0.0",
            "exit_recommender": "1.0.0",
            "risk_scoring": "1.0.0",
            "orchestrator": "1.0.0"
        }
    
    def _generate_caveats(self,
                          risk_summary: RiskSummary,
                          liquidity_profile: LiquidityProfile,
                          exit_analysis: ExitAnalysis) -> List[str]:
        """Generate caveats and disclaimers."""
        caveats = [
            "This analysis is for informational purposes only and does not constitute investment advice",
            "Market conditions can change rapidly, affecting both risk and liquidity assessments",
            "Historical performance does not guarantee future results",
            "Exit pathway recommendations are subject to market conditions and availability",
            "Risk scores are estimates based on available data and may not capture all risk factors"
        ]
        
        # Add data-specific caveats
        if liquidity_profile.data_freshness in ['stale', 'outdated']:
            caveats.append("Market data may be outdated - verify current conditions before trading")
        
        if risk_summary.confidence < 0.7:
            caveats.append("Risk assessment has limited confidence - additional analysis recommended")
        
        if exit_analysis.overall_confidence < 0.6:
            caveats.append("Exit pathway analysis has limited confidence - consult with professionals")
        
        return caveats
    
    def _assess_overall_data_freshness(self,
                                      market_timestamp: datetime,
                                      risk_timestamp: datetime) -> str:
        """Assess overall data freshness across all sources."""
        market_age = (datetime.now() - market_timestamp).total_seconds() / 60
        risk_age = (datetime.now() - risk_timestamp).total_seconds() / 60
        
        max_age = max(market_age, risk_age)
        
        if max_age < 1:
            return "real_time"
        elif max_age < 5:
            return "fresh"
        elif max_age < 15:
            return "recent"
        elif max_age < 60:
            return "stale"
        else:
            return "outdated"
