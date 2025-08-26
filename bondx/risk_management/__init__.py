"""
Risk Management Package for BondX.

This package provides comprehensive risk management capabilities including:
- Portfolio risk analytics
- Real-time risk monitoring
- Regulatory compliance
- Risk limit management
- Advanced quantitative risk (Phase C)
"""

__version__ = "2.0.0"
__author__ = "BondX Team"

# Phase A & B Components
from .portfolio_risk import PortfolioRiskManager
from .risk_monitoring import RiskMonitoringSystem
from .compliance_engine import ComplianceEngine
from .regulatory_reporting import RegulatoryReportingEngine
from .risk_models import *

# Phase C Components - Advanced Quantitative Risk
from .correlation_matrix import CorrelationMatrixCalculator, CorrelationMatrix, CorrelationMatrixConfig
from .volatility_models import VolatilityModels, VolatilityResult, VolatilityConfig
from .liquidity_models import LiquidityModels, LiquidityScore, MarketImpact, LiquidityMetrics
from .scenario_generator import YieldCurveScenarioGenerator, ScenarioSet, YieldCurveScenario
from .portfolio_optimizer import PortfolioOptimizer, OptimizationResult, PortfolioConstraints

__all__ = [
    # Phase A & B
    "PortfolioRiskManager",
    "RiskMonitoringSystem", 
    "ComplianceEngine",
    "RegulatoryReportingEngine",
    
    # Phase C - Advanced Quantitative
    "CorrelationMatrixCalculator",
    "CorrelationMatrix", 
    "CorrelationMatrixConfig",
    "VolatilityModels",
    "VolatilityResult",
    "VolatilityConfig",
    "LiquidityModels",
    "LiquidityScore",
    "MarketImpact",
    "LiquidityMetrics",
    "YieldCurveScenarioGenerator",
    "ScenarioSet",
    "YieldCurveScenario",
    "PortfolioOptimizer",
    "OptimizationResult",
    "PortfolioConstraints",
]
