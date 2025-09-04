"""
Core Package for BondX.

This package provides core functionality including:
- Configuration management
- Logging and monitoring
- Database models and contracts
- Currency handling
- Advanced Redis clustering (Phase D)
- Real-time streaming analytics (Phase D)
"""

__version__ = "2.0.0"
__author__ = "BondX Team"

# Core Components
from .config import Settings, settings
from .logging import setup_logging
from .model_contracts import *
from .monitoring import *

# Phase D Components - Advanced Infrastructure
from .advanced_redis_cluster import (
    AdvancedRedisCluster,
    RedisTimeSeriesManager,
    RedisNodeConfig,
    ClusterConfig,
    CacheConfig,
    RedisMode,
    DataType
)

from .streaming_analytics import (
    StreamingAnalyticsEngine,
    KafkaStreamManager,
    TickData,
    RiskMetrics,
    LiquidityMetrics,
    SentimentMetrics,
    StreamType,
    ProcessingWindow,
    AggregationType
)

__all__ = [
    # Core Components
    "Settings",
    "settings",
    "setup_logging",
    
    # Phase D - Advanced Infrastructure
    "AdvancedRedisCluster",
    "RedisTimeSeriesManager",
    "RedisNodeConfig",
    "ClusterConfig",
    "CacheConfig",
    "RedisMode",
    "DataType",
    "StreamingAnalyticsEngine",
    "KafkaStreamManager",
    "TickData",
    "RiskMetrics",
    "LiquidityMetrics",
    "SentimentMetrics",
    "StreamType",
    "ProcessingWindow",
    "AggregationType",
]
