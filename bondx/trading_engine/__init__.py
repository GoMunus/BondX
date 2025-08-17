"""
Trading Engine Package for BondX.

This package provides comprehensive real-time trading capabilities including:
- Order management system
- Market making algorithms
- Real-time price discovery
- Trade execution engine
- Risk management integration
"""

__version__ = "1.0.0"
__author__ = "BondX Team"

from .order_manager import OrderManager
from .market_maker import MarketMaker
from .execution_engine import ExecutionEngine
from .price_discovery import PriceDiscoveryEngine
from .trading_models import *

__all__ = [
    "OrderManager",
    "MarketMaker", 
    "ExecutionEngine",
    "PriceDiscoveryEngine",
]
