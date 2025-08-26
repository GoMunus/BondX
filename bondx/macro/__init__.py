"""
Macro Analysis Module

This module provides macro-economic analysis capabilities for BondX including:
- Cross-market correlations
- Macro factor analysis
- Currency handling
- Factor contribution analysis
"""

from .correlations import CrossMarketCorrelationEngine
from .currency import CurrencyHandler

__all__ = [
    'CrossMarketCorrelationEngine',
    'CurrencyHandler'
]
