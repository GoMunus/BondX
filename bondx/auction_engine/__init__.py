"""
BondX Auction Engine Package.

This package contains the sophisticated auction and settlement systems for Phase 3,
including auction mechanisms, fractional ownership, and real-time settlement infrastructure.
"""

__version__ = "1.0.0"
__author__ = "BondX Team"

from .auction_engine import AuctionEngine
from .auction_mechanisms import (
    DutchAuction,
    EnglishAuction,
    SealedBidAuction,
    MultiRoundAuction,
    HybridAuction
)
from .clearing_engine import ClearingEngine
from .allocation_engine import AllocationEngine
from .risk_engine import RiskEngine
from .settlement_engine import SettlementEngine

__all__ = [
    "AuctionEngine",
    "DutchAuction",
    "EnglishAuction", 
    "SealedBidAuction",
    "MultiRoundAuction",
    "HybridAuction",
    "ClearingEngine",
    "AllocationEngine",
    "RiskEngine",
    "SettlementEngine",
]
