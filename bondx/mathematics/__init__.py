"""
Mathematics package for BondX Backend.

This package contains all mathematical functions and calculations
for bond pricing, yield calculations, duration, and risk metrics.
"""

# Phase A & B Components
from .bond_pricing import BondPricingEngine
from .day_count import DayCountCalculator
from .yield_calculations import YieldCalculator
from .duration_convexity import DurationCalculator
from .cash_flows import CashFlowEngine
from .option_adjusted_spread import OASCalculator
from .yield_curves import YieldCurveEngine

# Phase C Components - Advanced Pricing
from .advanced_pricing import AdvancedPricingEngine, PricingResult, PricingMethod

__all__ = [
    # Phase A & B
    "BondPricingEngine",
    "DayCountCalculator",
    "YieldCalculator",
    "DurationCalculator",
    "CashFlowEngine",
    "OASCalculator",
    "YieldCurveEngine",
    
    # Phase C - Advanced Pricing
    "AdvancedPricingEngine",
    "PricingResult",
    "PricingMethod",
]
