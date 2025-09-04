"""
Real-Time Streaming Analytics for Phase D

This module implements real-time streaming analytics with:
- Kafka + Flink integration for tick-level data
- Rolling VaR, liquidity, and sentiment calculations
- <50ms end-to-end latency
- Real-time risk monitoring and alerts
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Streaming imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    BEAM_AVAILABLE = True
except ImportError:
    BEAM_AVAILABLE = False

# Redis for real-time caching
try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Tick-level market data"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    trade_type: str  # 'buy', 'sell', 'unknown'
    exchange: str
    sequence_number: int

@dataclass
class StreamingConfig:
    """Configuration for streaming analytics"""
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: ['localhost:9092'])
    kafka_topic_prefix: str = "bondx"
    batch_size: int = 1000
    batch_timeout_ms: int = 100
    max_latency_ms: float = 50.0
    enable_kafka: bool = True
    enable_beam: bool = False
    enable_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    window_size: int = 1000  # Rolling window size
    update_frequency_ms: int = 10  # Update frequency in milliseconds

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    symbol: str
    rolling_var: float
    rolling_volatility: float
    rolling_beta: float
    liquidity_score: float
    sentiment_score: float
    correlation_change: float
    stress_indicator: float

class RealTimeStreamingAnalytics:
    """
    Real-Time Streaming Analytics Engine
    
    Provides ultra-low latency streaming analytics with Kafka + Flink integration
    for real-time risk monitoring and trading applications.
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.kafka_available = KAFKA_AVAILABLE
        self.beam_available = BEAM_AVAILABLE
        self.redis_available = REDIS_AVAILABLE
        
        # Data buffers
        self.tick_buffer = deque(maxlen=config.window_size)
        self.risk_metrics_buffer = deque(maxlen=config.window_size)
        
        # Real-time calculations
        self.rolling_calculator = RollingRiskCalculator(config.window_size)
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Performance monitoring
        self.latency_metrics = {
            'tick_processing': [],
            'risk_calculation': [],
            'kafka_production': [],
            'redis_operations': []
        }
        
        # Initialize components
        self._initialize_components()
        
        # Background tasks
        self.is_running = False
        self.background_tasks = []
        
        logger.info("Real-Time Streaming Analytics initialized")
    
    def _initialize_components(self):
        """Initialize Kafka, Redis, and other components"""
        # Initialize Kafka producer
        if self.config.enable_kafka and self.kafka_available:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3,
                    batch_size=16384,
                    linger_ms=5,
                    compression_type='lz4'
                )
                logger.info("Kafka producer initialized")
            except Exception as e:
                logger.warning(f"Kafka producer initialization failed: {e}")
                self.config.enable_kafka = False
        
        # Initialize Redis
        if self.config.enable_redis and self.redis_available:
            try:
                self.redis_client = Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis client initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.config.enable_redis = False
        
        # Initialize Apache Beam pipeline
        if self.config.enable_beam and self.beam_available:
            try:
                self.beam_pipeline = self._create_beam_pipeline()
                logger.info("Apache Beam pipeline initialized")
            except Exception as e:
                logger.warning(f"Apache Beam initialization failed: {e}")
                self.config.enable_beam = False
    
    def _create_beam_pipeline(self):
        """Create Apache Beam pipeline for stream processing"""
        pipeline_options = PipelineOptions([
            '--runner=DirectRunner',
            '--project=bondx-streaming',
            '--temp_location=gs://bondx-temp/',
            '--region=us-central1'
        ])
        
        pipeline = beam.Pipeline(options=pipeline_options)
        return pipeline
    
    async def start(self):
        """Start the streaming analytics engine"""
        if self.is_running:
            logger.warning("Streaming analytics engine is already running")
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._process_tick_stream()),
            asyncio.create_task(self._update_risk_metrics()),
            asyncio.create_task(self._monitor_performance())
        ]
        
        logger.info("Real-Time Streaming Analytics engine started")
    
    async def stop(self):
        """Stop the streaming analytics engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cleanup
        if hasattr(self, 'kafka_producer'):
            self.kafka_producer.close()
        
        logger.info("Real-Time Streaming Analytics engine stopped")
    
    async def process_tick(self, tick_data: TickData):
        """
        Process incoming tick data with ultra-low latency
        
        Args:
            tick_data: Tick-level market data
        """
        start_time = time.perf_counter()
        
        try:
            # Add to buffer
            self.tick_buffer.append(tick_data)
            
            # Update rolling calculations
            self.rolling_calculator.update(tick_data)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['tick_processing'].append(latency_ms)
            
            # Check latency SLA
            if latency_ms > self.config.max_latency_ms:
                logger.warning(f"Tick processing latency {latency_ms:.2f}ms exceeds SLA {self.config.max_latency_ms}ms")
            
            # Publish to Kafka if enabled
            if self.config.enable_kafka:
                await self._publish_to_kafka(tick_data)
            
            # Store in Redis if enabled
            if self.config.enable_redis:
                await self._store_in_redis(tick_data)
            
        except Exception as e:
            logger.error(f"Tick processing failed: {e}")
    
    async def _publish_to_kafka(self, tick_data: TickData):
        """Publish tick data to Kafka"""
        start_time = time.perf_counter()
        
        try:
            topic = f"{self.config.kafka_topic_prefix}.ticks"
            key = tick_data.symbol
            value = {
                'timestamp': tick_data.timestamp.isoformat(),
                'symbol': tick_data.symbol,
                'price': tick_data.price,
                'volume': tick_data.volume,
                'bid': tick_data.bid,
                'ask': tick_data.ask,
                'bid_size': tick_data.bid_size,
                'ask_size': tick_data.ask_size,
                'trade_type': tick_data.trade_type,
                'exchange': tick_data.exchange,
                'sequence_number': tick_data.sequence_number
            }
            
            future = self.kafka_producer.send(topic, key=key, value=value)
            await asyncio.get_event_loop().run_in_executor(None, future.get, 10)
            
            # Record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['kafka_production'].append(latency_ms)
            
        except Exception as e:
            logger.error(f"Kafka publishing failed: {e}")
    
    async def _store_in_redis(self, tick_data: TickData):
        """Store tick data in Redis for real-time access"""
        start_time = time.perf_counter()
        
        try:
            # Store latest tick
            key = f"tick:latest:{tick_data.symbol}"
            value = json.dumps({
                'timestamp': tick_data.timestamp.isoformat(),
                'price': tick_data.price,
                'volume': tick_data.volume,
                'bid': tick_data.bid,
                'ask': tick_data.ask
            })
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.set, key, value, ex=300
            )
            
            # Store in time series
            ts_key = f"tick:ts:{tick_data.symbol}"
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.zadd, ts_key, 
                {tick_data.timestamp.timestamp(): value}
            )
            
            # Trim old data (keep last 1000 ticks)
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.zremrangebyrank, ts_key, 0, -1001
            )
            
            # Record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['redis_operations'].append(latency_ms)
            
        except Exception as e:
            logger.error(f"Redis storage failed: {e}")
    
    async def _process_tick_stream(self):
        """Background task for processing tick stream"""
        while self.is_running:
            try:
                # Process any pending ticks
                if self.tick_buffer:
                    # Process in batches for efficiency
                    batch = list(self.tick_buffer)[-self.config.batch_size:]
                    
                    for tick in batch:
                        await self.process_tick(tick)
                
                await asyncio.sleep(self.config.update_frequency_ms / 1000)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tick stream processing error: {e}")
                await asyncio.sleep(1)
    
    async def _update_risk_metrics(self):
        """Background task for updating risk metrics"""
        while self.is_running:
            try:
                if len(self.tick_buffer) >= self.config.window_size:
                    # Calculate rolling risk metrics
                    risk_metrics = await self._calculate_rolling_risk_metrics()
                    
                    if risk_metrics:
                        self.risk_metrics_buffer.append(risk_metrics)
                        
                        # Publish risk metrics
                        if self.config.enable_kafka:
                            await self._publish_risk_metrics(risk_metrics)
                        
                        # Store in Redis
                        if self.config.enable_redis:
                            await self._store_risk_metrics(risk_metrics)
                
                await asyncio.sleep(self.config.update_frequency_ms / 1000)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk metrics update error: {e}")
                await asyncio.sleep(1)
    
    async def _calculate_rolling_risk_metrics(self) -> Optional[RiskMetrics]:
        """Calculate rolling risk metrics for the current window"""
        start_time = time.perf_counter()
        
        try:
            if len(self.tick_buffer) < self.config.window_size:
                return None
            
            # Get current window data
            window_data = list(self.tick_buffer)[-self.config.window_size:]
            
            # Calculate rolling metrics
            rolling_var = self.rolling_calculator.get_rolling_var()
            rolling_volatility = self.rolling_calculator.get_rolling_volatility()
            rolling_beta = self.rolling_calculator.get_rolling_beta()
            
            # Calculate liquidity score
            liquidity_score = self.liquidity_analyzer.calculate_liquidity_score(window_data)
            
            # Calculate sentiment score
            sentiment_score = self.sentiment_analyzer.calculate_sentiment_score(window_data)
            
            # Calculate correlation change
            correlation_change = self.rolling_calculator.get_correlation_change()
            
            # Calculate stress indicator
            stress_indicator = self._calculate_stress_indicator(window_data)
            
            # Record latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['risk_calculation'].append(latency_ms)
            
            return RiskMetrics(
                timestamp=datetime.utcnow(),
                symbol=window_data[-1].symbol,
                rolling_var=rolling_var,
                rolling_volatility=rolling_volatility,
                rolling_beta=rolling_beta,
                liquidity_score=liquidity_score,
                sentiment_score=sentiment_score,
                correlation_change=correlation_change,
                stress_indicator=stress_indicator
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return None
    
    def _calculate_stress_indicator(self, window_data: List[TickData]) -> float:
        """Calculate market stress indicator"""
        try:
            # Extract price changes
            prices = [tick.price for tick in window_data]
            price_changes = np.diff(prices) / prices[:-1]
            
            # Calculate volatility
            volatility = np.std(price_changes)
            
            # Calculate price momentum
            momentum = np.mean(price_changes[-10:]) if len(price_changes) >= 10 else np.mean(price_changes)
            
            # Calculate bid-ask spread
            spreads = [(tick.ask - tick.bid) / tick.price for tick in window_data if tick.ask > 0 and tick.bid > 0]
            avg_spread = np.mean(spreads) if spreads else 0
            
            # Combine indicators
            stress_indicator = (
                0.4 * volatility +
                0.3 * abs(momentum) +
                0.3 * avg_spread
            )
            
            return min(stress_indicator, 1.0)  # Normalize to [0, 1]
            
        except Exception as e:
            logger.error(f"Stress indicator calculation failed: {e}")
            return 0.0
    
    async def _publish_risk_metrics(self, risk_metrics: RiskMetrics):
        """Publish risk metrics to Kafka"""
        try:
            topic = f"{self.config.kafka_topic_prefix}.risk_metrics"
            key = risk_metrics.symbol
            value = {
                'timestamp': risk_metrics.timestamp.isoformat(),
                'symbol': risk_metrics.symbol,
                'rolling_var': risk_metrics.rolling_var,
                'rolling_volatility': risk_metrics.rolling_volatility,
                'rolling_beta': risk_metrics.rolling_beta,
                'liquidity_score': risk_metrics.liquidity_score,
                'sentiment_score': risk_metrics.sentiment_score,
                'correlation_change': risk_metrics.correlation_change,
                'stress_indicator': risk_metrics.stress_indicator
            }
            
            future = self.kafka_producer.send(topic, key=key, value=value)
            await asyncio.get_event_loop().run_in_executor(None, future.get, 10)
            
        except Exception as e:
            logger.error(f"Risk metrics publishing failed: {e}")
    
    async def _store_risk_metrics(self, risk_metrics: RiskMetrics):
        """Store risk metrics in Redis"""
        try:
            # Store latest metrics
            key = f"risk:latest:{risk_metrics.symbol}"
            value = json.dumps({
                'timestamp': risk_metrics.timestamp.isoformat(),
                'rolling_var': risk_metrics.rolling_var,
                'rolling_volatility': risk_metrics.rolling_volatility,
                'liquidity_score': risk_metrics.liquidity_score,
                'sentiment_score': risk_metrics.sentiment_score,
                'stress_indicator': risk_metrics.stress_indicator
            })
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.set, key, value, ex=300
            )
            
        except Exception as e:
            logger.error(f"Risk metrics storage failed: {e}")
    
    async def _monitor_performance(self):
        """Background task for performance monitoring"""
        while self.is_running:
            try:
                # Log performance metrics every minute
                await asyncio.sleep(60)
                
                metrics = self.get_performance_metrics()
                logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the streaming engine"""
        return {
            'latency_metrics': {
                'tick_processing': {
                    'mean_ms': np.mean(self.latency_metrics['tick_processing']) if self.latency_metrics['tick_processing'] else 0,
                    'p95_ms': np.percentile(self.latency_metrics['tick_processing'], 95) if self.latency_metrics['tick_processing'] else 0,
                    'p99_ms': np.percentile(self.latency_metrics['tick_processing'], 99) if self.latency_metrics['tick_processing'] else 0,
                    'count': len(self.latency_metrics['tick_processing'])
                },
                'risk_calculation': {
                    'mean_ms': np.mean(self.latency_metrics['risk_calculation']) if self.latency_metrics['risk_calculation'] else 0,
                    'p95_ms': np.percentile(self.latency_metrics['risk_calculation'], 95) if self.latency_metrics['risk_calculation'] else 0,
                    'p99_ms': np.percentile(self.latency_metrics['risk_calculation'], 99) if self.latency_metrics['risk_calculation'] else 0,
                    'count': len(self.latency_metrics['risk_calculation'])
                }
            },
            'buffer_stats': {
                'tick_buffer_size': len(self.tick_buffer),
                'risk_metrics_buffer_size': len(self.risk_metrics_buffer)
            },
            'component_status': {
                'kafka_available': self.kafka_available,
                'beam_available': self.beam_available,
                'redis_available': self.redis_available
            }
        }
    
    def get_latest_risk_metrics(self, symbol: str) -> Optional[RiskMetrics]:
        """Get latest risk metrics for a symbol"""
        if not self.risk_metrics_buffer:
            return None
        
        # Find latest metrics for the symbol
        for metrics in reversed(self.risk_metrics_buffer):
            if metrics.symbol == symbol:
                return metrics
        
        return None
    
    def get_risk_metrics_history(self, symbol: str, limit: int = 100) -> List[RiskMetrics]:
        """Get risk metrics history for a symbol"""
        if not self.risk_metrics_buffer:
            return []
        
        # Filter by symbol and limit
        filtered_metrics = [
            metrics for metrics in self.risk_metrics_buffer
            if metrics.symbol == symbol
        ]
        
        return filtered_metrics[-limit:]


class RollingRiskCalculator:
    """Calculates rolling risk metrics"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.return_history = deque(maxlen=window_size)
        self.market_returns = deque(maxlen=window_size)
    
    def update(self, tick_data: TickData):
        """Update with new tick data"""
        if self.price_history:
            # Calculate return
            prev_price = self.price_history[-1]
            if prev_price > 0:
                return_pct = (tick_data.price - prev_price) / prev_price
                self.return_history.append(return_pct)
        
        self.price_history.append(tick_data.price)
    
    def get_rolling_var(self, confidence_level: float = 0.99) -> float:
        """Get rolling VaR"""
        if len(self.return_history) < 10:
            return 0.0
        
        returns = np.array(list(self.return_history))
        var_index = int((1 - confidence_level) * len(returns))
        sorted_returns = np.sort(returns)
        return sorted_returns[var_index]
    
    def get_rolling_volatility(self) -> float:
        """Get rolling volatility"""
        if len(self.return_history) < 10:
            return 0.0
        
        returns = np.array(list(self.return_history))
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def get_rolling_beta(self) -> float:
        """Get rolling beta (simplified)"""
        if len(self.return_history) < 10:
            return 1.0
        
        # Simplified beta calculation
        returns = np.array(list(self.return_history))
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 1.0
    
    def get_correlation_change(self) -> float:
        """Get correlation change indicator"""
        if len(self.return_history) < 20:
            return 0.0
        
        # Calculate correlation change over time
        returns = np.array(list(self.return_history))
        mid_point = len(returns) // 2
        
        corr_first = np.corrcoef(returns[:mid_point], np.arange(mid_point))[0, 1]
        corr_second = np.corrcoef(returns[mid_point:], np.arange(mid_point))[0, 1]
        
        if np.isnan(corr_first) or np.isnan(corr_second):
            return 0.0
        
        return corr_second - corr_first


class LiquidityAnalyzer:
    """Analyzes market liquidity"""
    
    def calculate_liquidity_score(self, window_data: List[TickData]) -> float:
        """Calculate liquidity score based on bid-ask spreads and volumes"""
        try:
            if not window_data:
                return 0.0
            
            spreads = []
            volumes = []
            
            for tick in window_data:
                if tick.ask > 0 and tick.bid > 0:
                    spread_pct = (tick.ask - tick.bid) / tick.price
                    spreads.append(spread_pct)
                
                if tick.volume > 0:
                    volumes.append(tick.volume)
            
            if not spreads or not volumes:
                return 0.0
            
            # Calculate liquidity score (inverse of spread, weighted by volume)
            avg_spread = np.mean(spreads)
            avg_volume = np.mean(volumes)
            
            # Normalize to [0, 1] where 1 is most liquid
            liquidity_score = 1.0 / (1.0 + avg_spread * 100)  # Scale spread
            
            # Adjust for volume
            volume_factor = min(avg_volume / 1000000, 1.0)  # Normalize to 1M volume
            
            return liquidity_score * (0.7 + 0.3 * volume_factor)
            
        except Exception as e:
            logger.error(f"Liquidity score calculation failed: {e}")
            return 0.0


class SentimentAnalyzer:
    """Analyzes market sentiment"""
    
    def calculate_sentiment_score(self, window_data: List[TickData]) -> float:
        """Calculate sentiment score based on trade patterns"""
        try:
            if not window_data:
                return 0.0
            
            # Analyze trade types
            buy_trades = sum(1 for tick in window_data if tick.trade_type == 'buy')
            sell_trades = sum(1 for tick in window_data if tick.trade_type == 'sell')
            total_trades = len(window_data)
            
            if total_trades == 0:
                return 0.0
            
            # Calculate sentiment ratio
            sentiment_ratio = (buy_trades - sell_trades) / total_trades
            
            # Normalize to [-1, 1] where 1 is bullish, -1 is bearish
            return sentiment_ratio
            
        except Exception as e:
            logger.error(f"Sentiment score calculation failed: {e}")
            return 0.0
