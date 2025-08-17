"""
BondX Backend - AI-Powered Fractional Bond Marketplace

A sophisticated FastAPI backend designed specifically for the Indian debt capital market,
featuring advanced bond pricing engines, comprehensive financial mathematics,
and machine learning infrastructure.
"""

__version__ = "0.1.0"
__author__ = "BondX Team"
__email__ = "team@bondx.com"

from .core.config import settings
from .core.logging import setup_logging

# Initialize logging when package is imported
setup_logging()

__all__ = [
    "settings",
    "setup_logging",
]
