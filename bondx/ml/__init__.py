"""
Machine Learning Module

This module provides ML capabilities for BondX including:
- Spread prediction models
- Downgrade risk models
- Liquidity shock prediction
- Anomaly detection
"""

from .spread_model import SpreadPredictionModel
from .downgrade_model import DowngradeRiskModel
from .liquidity_shock_model import LiquidityShockModel
from .anomaly_detector import AnomalyDetector

__all__ = [
    'SpreadPredictionModel',
    'DowngradeRiskModel',
    'LiquidityShockModel',
    'AnomalyDetector'
]
