"""
AI Risk Analytics Engine for BondX

This module provides comprehensive risk analytics, predictive modeling, and intelligent
advisory services for bond investments in the Indian market context.

Phase D Components:
- Enhanced ML Pipeline with GPU acceleration and distributed computing
- Advanced volatility models (LSTM/GRU/Transformer/Neural-GARCH/HAR-RNN)
- Real-time streaming analytics with tick-level processing
- Ultra-low latency risk calculations for HFT applications
"""

from .risk_scoring import RiskScoringEngine
from .yield_prediction import YieldPredictionEngine
from .ml_pipeline import MLPipeline
from .nlp_engine import NLPEngine
from .advisory_system import IntelligentAdvisorySystem
from .real_time_analytics import RealTimeAnalytics
from .model_governance import ModelGovernance

# Phase D - Enhanced ML Pipeline
from .enhanced_ml_pipeline import (
    EnhancedMLPipeline,
    GPUAcceleratedPipeline,
    DistributedMLPipeline,
    PerformanceConfig,
    DistributedConfig,
    ComputeBackend,
    ModelArchitecture
)

__version__ = "2.0.0"
__all__ = [
    "RiskScoringEngine",
    "YieldPredictionEngine", 
    "MLPipeline",
    "NLPEngine",
    "IntelligentAdvisorySystem",
    "RealTimeAnalytics",
    "ModelGovernance",
    
    # Phase D Components
    "EnhancedMLPipeline",
    "GPUAcceleratedPipeline",
    "DistributedMLPipeline",
    "PerformanceConfig",
    "DistributedConfig",
    "ComputeBackend",
    "ModelArchitecture"
]
