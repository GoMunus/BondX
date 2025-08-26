"""
Signal Adapters for Liquidity Pulse

This module provides adapters for different data sources including:
- Alternative data sources (traffic, utilities, ESG proxies)
- Market microstructure data
- Auction and market maker telemetry
- Sentiment and news data
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ...core.logging import get_logger
from ...api.v1.schemas_liquidity import (
    AltDataSignal, MicrostructureData, AuctionMMData, SentimentData,
    SignalQuality, DataFreshness
)

logger = get_logger(__name__)

class SignalType(str, Enum):
    """Types of signals that can be processed."""
    ALT_DATA = "alt_data"
    MICROSTRUCTURE = "microstructure"
    AUCTION_MM = "auction_mm"
    SENTIMENT = "sentiment"

class SignalStatus(str, Enum):
    """Status of signal processing."""
    ACTIVE = "active"
    STALE = "stale"
    MISSING = "missing"
    ERROR = "error"

@dataclass
class SignalMetadata:
    """Metadata for signal processing."""
    source_id: str
    signal_type: SignalType
    quality: SignalQuality
    freshness_s: float
    latency_ms: float
    provenance: str
    last_update: datetime
    status: SignalStatus
    error_count: int = 0
    success_count: int = 0

@dataclass
class ProcessedSignal:
    """Processed and normalized signal."""
    signal_id: str
    isin: str
    timestamp: datetime
    value: float
    normalized_value: float
    quality: SignalQuality
    freshness: DataFreshness
    metadata: SignalMetadata
    raw_data: Dict[str, Any]

class BaseSignalAdapter:
    """Base class for signal adapters."""
    
    def __init__(self, adapter_id: str, config: Dict[str, Any]):
        self.adapter_id = adapter_id
        self.config = config
        self.logger = get_logger(f"{__name__}.{adapter_id}")
        self.metadata = SignalMetadata(
            source_id=adapter_id,
            signal_type=self._get_signal_type(),
            quality=SignalQuality.MODERATE,
            freshness_s=0.0,
            latency_ms=0.0,
            provenance="unknown",
            last_update=datetime.now(),
            status=SignalStatus.ACTIVE
        )
    
    def _get_signal_type(self) -> SignalType:
        """Get the signal type for this adapter."""
        raise NotImplementedError
    
    async def fetch_signals(self, isins: List[str]) -> List[ProcessedSignal]:
        """Fetch signals for given ISINs."""
        raise NotImplementedError
    
    def normalize_signal(self, raw_signal: Any) -> ProcessedSignal:
        """Normalize raw signal to standard format."""
        raise NotImplementedError
    
    def calculate_freshness(self, timestamp: datetime) -> DataFreshness:
        """Calculate data freshness level."""
        age_seconds = (datetime.now() - timestamp).total_seconds()
        
        if age_seconds < 1:
            return DataFreshness.REAL_TIME
        elif age_seconds < 60:
            return DataFreshness.FRESH
        elif age_seconds < 3600:
            return DataFreshness.RECENT
        elif age_seconds < 86400:
            return DataFreshness.STALE
        else:
            return DataFreshness.OUTDATED
    
    def update_metadata(self, success: bool, latency_ms: float):
        """Update adapter metadata."""
        if success:
            self.metadata.success_count += 1
            self.metadata.status = SignalStatus.ACTIVE
        else:
            self.metadata.error_count += 1
            if self.metadata.error_count > 5:
                self.metadata.status = SignalStatus.ERROR
        
        self.metadata.latency_ms = latency_ms
        self.metadata.last_update = datetime.now()

class AltDataAdapter(BaseSignalAdapter):
    """Adapter for alternative data sources."""
    
    def _get_signal_type(self) -> SignalType:
        return SignalType.ALT_DATA
    
    async def fetch_signals(self, isins: List[str]) -> List[ProcessedSignal]:
        """Fetch alternative data signals."""
        start_time = datetime.now()
        try:
            # Mock implementation - replace with actual data source
            signals = []
            for isin in isins:
                # Simulate traffic/toll data, utilities, ESG proxies
                mock_signals = self._generate_mock_alt_data(isin)
                signals.extend(mock_signals)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(True, latency)
            return signals
            
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(False, latency)
            self.logger.error(f"Error fetching alt data: {e}")
            return []
    
    def _generate_mock_alt_data(self, isin: str) -> List[ProcessedSignal]:
        """Generate mock alternative data for testing."""
        signals = []
        now = datetime.now()
        
        # Traffic/toll data
        traffic_signal = ProcessedSignal(
            signal_id=f"traffic_{isin}_{now.timestamp()}",
            isin=isin,
            timestamp=now,
            value=np.random.normal(100, 20),
            normalized_value=np.random.uniform(0, 100),
            quality=SignalQuality.GOOD,
            freshness=self.calculate_freshness(now),
            metadata=self.metadata,
            raw_data={"source": "traffic", "value": 100}
        )
        signals.append(traffic_signal)
        
        # Utilities data
        utilities_signal = ProcessedSignal(
            signal_id=f"utilities_{isin}_{now.timestamp()}",
            isin=isin,
            timestamp=now,
            value=np.random.normal(0.8, 0.1),
            normalized_value=np.random.uniform(0, 100),
            quality=SignalQuality.MODERATE,
            freshness=self.calculate_freshness(now),
            metadata=self.metadata,
            raw_data={"source": "utilities", "value": 0.8}
        )
        signals.append(utilities_signal)
        
        return signals

class MicrostructureAdapter(BaseSignalAdapter):
    """Adapter for market microstructure data."""
    
    def _get_signal_type(self) -> SignalType:
        return SignalType.MICROSTRUCTURE
    
    async def fetch_signals(self, isins: List[str]) -> List[ProcessedSignal]:
        """Fetch microstructure data."""
        start_time = datetime.now()
        try:
            # Mock implementation - replace with actual market data
            signals = []
            for isin in isins:
                mock_signal = self._generate_mock_microstructure(isin)
                signals.append(mock_signal)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(True, latency)
            return signals
            
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(False, latency)
            self.logger.error(f"Error fetching microstructure data: {e}")
            return []
    
    def _generate_mock_microstructure(self, isin: str) -> ProcessedSignal:
        """Generate mock microstructure data for testing."""
        now = datetime.now()
        
        # Simulate bid-ask spread, depth, turnover
        spread_bps = np.random.exponential(5)  # Exponential distribution for spreads
        depth_qty = np.random.lognormal(10, 1)  # Log-normal for depth
        turnover = np.random.gamma(2, 1000)  # Gamma for turnover
        
        return ProcessedSignal(
            signal_id=f"microstructure_{isin}_{now.timestamp()}",
            isin=isin,
            timestamp=now,
            value=spread_bps,
            normalized_value=max(0, 100 - spread_bps * 2),  # Higher spread = lower liquidity
            quality=SignalQuality.EXCELLENT,
            freshness=self.calculate_freshness(now),
            metadata=self.metadata,
            raw_data={
                "spread_bps": spread_bps,
                "depth_qty": depth_qty,
                "turnover": turnover,
                "bid": 100.0,
                "ask": 100.0 + spread_bps / 10000
            }
        )

class AuctionMMAdapter(BaseSignalAdapter):
    """Adapter for auction and market maker data."""
    
    def _get_signal_type(self) -> SignalType:
        return SignalType.AUCTION_MM
    
    async def fetch_signals(self, isins: List[str]) -> List[ProcessedSignal]:
        """Fetch auction and MM data."""
        start_time = datetime.now()
        try:
            # Mock implementation - replace with actual auction/MM data
            signals = []
            for isin in isins:
                mock_signal = self._generate_mock_auction_mm(isin)
                signals.append(mock_signal)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(True, latency)
            return signals
            
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(False, latency)
            self.logger.error(f"Error fetching auction/MM data: {e}")
            return []
    
    def _generate_mock_auction_mm(self, isin: str) -> ProcessedSignal:
        """Generate mock auction/MM data for testing."""
        now = datetime.now()
        
        # Simulate auction demand and MM activity
        auction_demand = np.random.beta(2, 2)  # Beta distribution for demand
        mm_online = np.random.choice([True, False], p=[0.9, 0.1])
        mm_spread = np.random.exponential(3) if mm_online else 999
        
        return ProcessedSignal(
            signal_id=f"auction_mm_{isin}_{now.timestamp()}",
            isin=isin,
            timestamp=now,
            value=auction_demand,
            normalized_value=auction_demand * 100,  # Scale to 0-100
            quality=SignalQuality.GOOD,
            freshness=self.calculate_freshness(now),
            metadata=self.metadata,
            raw_data={
                "auction_demand_index": auction_demand,
                "mm_online": mm_online,
                "mm_spread_bps": mm_spread,
                "quotes_last_24h": np.random.poisson(50)
            }
        )

class SentimentAdapter(BaseSignalAdapter):
    """Adapter for sentiment and news data."""
    
    def _get_signal_type(self) -> SignalType:
        return SignalType.SENTIMENT
    
    async def fetch_signals(self, isins: List[str]) -> List[ProcessedSignal]:
        """Fetch sentiment data."""
        start_time = datetime.now()
        try:
            # Mock implementation - replace with actual sentiment data
            signals = []
            for isin in isins:
                mock_signal = self._generate_mock_sentiment(isin)
                signals.append(mock_signal)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(True, latency)
            return signals
            
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.update_metadata(False, latency)
            self.logger.error(f"Error fetching sentiment data: {e}")
            return []
    
    def _generate_mock_sentiment(self, isin: str) -> ProcessedSignal:
        """Generate mock sentiment data for testing."""
        now = datetime.now()
        
        # Simulate sentiment with some persistence and noise
        base_sentiment = np.random.normal(0, 0.3)  # Base sentiment
        noise = np.random.normal(0, 0.1)  # Daily noise
        sentiment_score = np.clip(base_sentiment + noise, -1, 1)
        
        return ProcessedSignal(
            signal_id=f"sentiment_{isin}_{now.timestamp()}",
            isin=isin,
            timestamp=now,
            value=sentiment_score,
            normalized_value=(sentiment_score + 1) * 50,  # Convert -1..1 to 0..100
            quality=SignalQuality.MODERATE,
            freshness=self.calculate_freshness(now),
            metadata=self.metadata,
            raw_data={
                "sentiment_score": sentiment_score,
                "buzz_volume": np.random.exponential(100),
                "topics": {
                    "earnings": np.random.uniform(0, 1),
                    "regulatory": np.random.uniform(0, 1),
                    "market": np.random.uniform(0, 1)
                }
            }
        )

class SignalAdapterManager:
    """Manages all signal adapters and coordinates data collection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.adapters: Dict[str, BaseSignalAdapter] = {}
        self.signal_cache: Dict[str, ProcessedSignal] = {}
        self.cache_ttl = config.get("cache_ttl_seconds", 300)  # 5 minutes
        
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize all signal adapters."""
        adapter_configs = self.config.get("adapters", {})
        
        # Initialize alt data adapter
        if adapter_configs.get("alt_data", {}).get("enabled", True):
            self.adapters["alt_data"] = AltDataAdapter("alt_data", adapter_configs.get("alt_data", {}))
        
        # Initialize microstructure adapter
        if adapter_configs.get("microstructure", {}).get("enabled", True):
            self.adapters["microstructure"] = MicrostructureAdapter("microstructure", adapter_configs.get("microstructure", {}))
        
        # Initialize auction/MM adapter
        if adapter_configs.get("auction_mm", {}).get("enabled", True):
            self.adapters["auction_mm"] = AuctionMMAdapter("auction_mm", adapter_configs.get("auction_mm", {}))
        
        # Initialize sentiment adapter
        if adapter_configs.get("sentiment", {}).get("enabled", True):
            self.adapters["sentiment"] = SentimentAdapter("sentiment", adapter_configs.get("sentiment", {}))
        
        self.logger.info(f"Initialized {len(self.adapters)} signal adapters")
    
    async def collect_all_signals(self, isins: List[str]) -> Dict[str, List[ProcessedSignal]]:
        """Collect signals from all adapters for given ISINs."""
        tasks = []
        for adapter_id, adapter in self.adapters.items():
            task = adapter.fetch_signals(isins)
            tasks.append((adapter_id, task))
        
        # Execute all adapters concurrently
        results = {}
        for adapter_id, task in tasks:
            try:
                signals = await task
                results[adapter_id] = signals
                self.logger.debug(f"Collected {len(signals)} signals from {adapter_id}")
            except Exception as e:
                self.logger.error(f"Error collecting signals from {adapter_id}: {e}")
                results[adapter_id] = []
        
        # Update cache
        for adapter_id, signals in results.items():
            for signal in signals:
                cache_key = f"{adapter_id}:{signal.isin}"
                self.signal_cache[cache_key] = signal
        
        return results
    
    def get_cached_signals(self, isin: str, signal_types: Optional[List[str]] = None) -> List[ProcessedSignal]:
        """Get cached signals for an ISIN."""
        if signal_types is None:
            signal_types = list(self.adapters.keys())
        
        cached_signals = []
        for signal_type in signal_types:
            cache_key = f"{signal_type}:{isin}"
            if cache_key in self.signal_cache:
                signal = self.signal_cache[cache_key]
                # Check if signal is still fresh
                age_seconds = (datetime.now() - signal.timestamp).total_seconds()
                if age_seconds < self.cache_ttl:
                    cached_signals.append(signal)
        
        return cached_signals
    
    def get_adapter_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all adapters."""
        health = {}
        for adapter_id, adapter in self.adapters.items():
            health[adapter_id] = {
                "status": adapter.metadata.status.value,
                "quality": adapter.metadata.quality.value,
                "freshness_s": adapter.metadata.freshness_s,
                "latency_ms": adapter.metadata.latency_ms,
                "last_update": adapter.metadata.last_update.isoformat(),
                "success_count": adapter.metadata.success_count,
                "error_count": adapter.metadata.error_count
            }
        return health
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        now = datetime.now()
        expired_keys = []
        
        for cache_key, signal in self.signal_cache.items():
            age_seconds = (now - signal.timestamp).total_seconds()
            if age_seconds > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.signal_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
