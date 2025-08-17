"""
AI Risk Analytics Engine for BondX

This module provides comprehensive risk analytics, predictive modeling, and intelligent
advisory services for bond investments in the Indian market context.
"""

from .risk_scoring import RiskScoringEngine
from .yield_prediction import YieldPredictionEngine
from .ml_pipeline import MLPipeline
from .nlp_engine import NLPEngine
from .advisory_system import IntelligentAdvisorySystem
from .real_time_analytics import RealTimeAnalytics
from .model_governance import ModelGovernance

__version__ = "1.0.0"
__all__ = [
    "RiskScoringEngine",
    "YieldPredictionEngine", 
    "MLPipeline",
    "NLPEngine",
    "IntelligentAdvisorySystem",
    "RealTimeAnalytics",
    "ModelGovernance"
]
