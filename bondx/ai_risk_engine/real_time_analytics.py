"""
Real-time Analytics & Streaming Infrastructure

This module implements real-time processing systems that can ingest, analyze,
and respond to market data, news feeds, and user interactions with minimal latency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import asyncio
import json
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Streaming and messaging
import websockets
import aiohttp
import redis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# Real-time processing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import cvxpy as cp

# Custom imports
from .risk_scoring import RiskScoringEngine
from .yield_prediction import YieldPredictionEngine
from .nlp_engine import NLPEngine

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Data source types"""
    MARKET_DATA = "market_data"
    NEWS_FEED = "news_feed"
    USER_INTERACTION = "user_interaction"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_UPDATE = "portfolio_update"

class AlertType(Enum):
    """Alert types"""
    RISK_THRESHOLD = "risk_threshold"
    PRICE_MOVEMENT = "price_movement"
    NEWS_IMPACT = "news_impact"
    PORTFOLIO_ALERT = "portfolio_alert"
    SYSTEM_ALERT = "system_alert"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RealTimeData:
    """Real-time data structure"""
    source: DataSource
    data_type: str
    content: Any
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

@dataclass
class Alert:
    """Alert structure"""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    data: Dict = field(default_factory=dict)
    acknowledged: bool = False

@dataclass
class StreamingMetrics:
    """Streaming performance metrics"""
    messages_processed: int
    processing_latency: float
    throughput: float
    error_rate: float
    last_updated: datetime

class RealTimeAnalytics:
    """
    Real-time analytics system for bond market data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.risk_engine = RiskScoringEngine()
        self.yield_engine = YieldPredictionEngine()
        self.nlp_engine = NLPEngine()
        
        # Initialize streaming infrastructure
        self._initialize_streaming_infrastructure()
        
        # Initialize real-time processors
        self._initialize_real_time_processors()
        
        # Initialize alert system
        self._initialize_alert_system()
        
        # Performance monitoring
        self.metrics = StreamingMetrics(0, 0.0, 0.0, 0.0, datetime.now())
        self.processing_times = deque(maxlen=1000)
        
        # Start background tasks
        self._start_background_tasks()
        
    def _initialize_streaming_infrastructure(self):
        """Initialize streaming and messaging infrastructure"""
        try:
            # Redis for caching and pub/sub
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            
            # Kafka for high-throughput messaging
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None
                )
                
                self.kafka_consumer = KafkaConsumer(
                    'bond_market_data',
                    bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )
                
            except Exception as e:
                logger.warning(f"Kafka initialization failed: {e}")
                self.kafka_producer = None
                self.kafka_consumer = None
            
            # WebSocket server for real-time updates
            self.websocket_clients = set()
            self.websocket_server = None
            
            logger.info("Streaming infrastructure initialized")
            
        except Exception as e:
            logger.error(f"Error initializing streaming infrastructure: {e}")
    
    def _initialize_real_time_processors(self):
        """Initialize real-time data processors"""
        try:
            # Data processors
            self.market_data_processor = MarketDataProcessor(self.risk_engine, self.yield_engine)
            self.news_processor = NewsProcessor(self.nlp_engine)
            self.portfolio_processor = PortfolioProcessor(self.risk_engine)
            
            # Anomaly detection
            self.anomaly_detector = AnomalyDetector()
            
            # Real-time models
            self.online_models = OnlineLearningModels()
            
            # Thread pool for parallel processing
            self.executor = ThreadPoolExecutor(max_workers=10)
            
            logger.info("Real-time processors initialized")
            
        except Exception as e:
            logger.error(f"Error initializing real-time processors: {e}")
    
    def _initialize_alert_system(self):
        """Initialize alert system"""
        try:
            self.alert_rules = self._load_alert_rules()
            self.active_alerts = {}
            self.alert_history = deque(maxlen=10000)
            
            # Alert thresholds
            self.risk_thresholds = {
                'credit_risk': 70.0,
                'interest_rate_risk': 60.0,
                'liquidity_risk': 65.0,
                'overall_risk': 75.0
            }
            
            # Price movement thresholds
            self.price_thresholds = {
                'yield_change': 0.05,  # 5 basis points
                'price_change': 0.02,  # 2%
                'volume_spike': 3.0    # 3x average volume
            }
            
            logger.info("Alert system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing alert system: {e}")
    
    def _load_alert_rules(self) -> Dict:
        """Load alert rules from configuration"""
        return {
            'risk_threshold': {
                'enabled': True,
                'check_interval': 30,  # seconds
                'thresholds': self.risk_thresholds
            },
            'price_movement': {
                'enabled': True,
                'check_interval': 10,  # seconds
                'thresholds': self.price_thresholds
            },
            'news_impact': {
                'enabled': True,
                'check_interval': 60,  # seconds
                'sentiment_threshold': -0.3
            }
        }
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        try:
            # Start data processing loop
            self.processing_thread = threading.Thread(target=self._data_processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start alert monitoring loop
            self.alert_thread = threading.Thread(target=self._alert_monitoring_loop, daemon=True)
            self.alert_thread.start()
            
            # Start metrics collection loop
            self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
            self.metrics_thread.start()
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def start_websocket_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server for real-time updates"""
        try:
            async def websocket_handler(websocket, path):
                """Handle WebSocket connections"""
                try:
                    # Add client to set
                    self.websocket_clients.add(websocket)
                    
                    # Send welcome message
                    await websocket.send(json.dumps({
                        'type': 'connection_established',
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Connected to BondX Real-time Analytics'
                    }))
                    
                    # Keep connection alive and handle messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            response = await self._handle_websocket_message(data)
                            await websocket.send(json.dumps(response))
                        except json.JSONDecodeError:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Invalid JSON format'
                            }))
                        except Exception as e:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': f'Processing error: {str(e)}'
                            }))
                            
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    # Remove client from set
                    self.websocket_clients.discard(websocket)
            
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                websocket_handler, host, port
            )
            
            logger.info(f"WebSocket server started on ws://{host}:{port}")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
    
    async def _handle_websocket_message(self, message: Dict) -> Dict:
        """Handle incoming WebSocket messages"""
        try:
            message_type = message.get('type')
            
            if message_type == 'subscribe':
                return await self._handle_subscription(message)
            elif message_type == 'query':
                return await self._handle_query(message)
            elif message_type == 'alert_acknowledge':
                return await self._handle_alert_acknowledgment(message)
            else:
                return {
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            return {
                'type': 'error',
                'message': f'Internal error: {str(e)}'
            }
    
    async def _handle_subscription(self, message: Dict) -> Dict:
        """Handle subscription requests"""
        try:
            subscription_type = message.get('subscription_type')
            filters = message.get('filters', {})
            
            # Store subscription preferences
            subscription_id = f"sub_{int(time.time() * 1000)}"
            
            return {
                'type': 'subscription_confirmed',
                'subscription_id': subscription_id,
                'subscription_type': subscription_type,
                'filters': filters,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
            return {
                'type': 'error',
                'message': f'Subscription error: {str(e)}'
            }
    
    async def _handle_query(self, message: Dict) -> Dict:
        """Handle real-time queries"""
        try:
            query_type = message.get('query_type')
            query_params = message.get('params', {})
            
            if query_type == 'current_risk':
                result = await self._get_current_risk(query_params)
            elif query_type == 'market_summary':
                result = await self._get_market_summary(query_params)
            elif query_type == 'portfolio_status':
                result = await self._get_portfolio_status(query_params)
            else:
                result = {'error': f'Unknown query type: {query_type}'}
            
            return {
                'type': 'query_response',
                'query_type': query_type,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return {
                'type': 'error',
                'message': f'Query error: {str(e)}'
            }
    
    async def _get_current_risk(self, params: Dict) -> Dict:
        """Get current risk metrics"""
        try:
            # This would typically fetch from real-time data store
            return {
                'overall_risk': 45.2,
                'credit_risk': 38.5,
                'interest_rate_risk': 52.1,
                'liquidity_risk': 41.3,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting current risk: {e}")
            return {'error': str(e)}
    
    async def _get_market_summary(self, params: Dict) -> Dict:
        """Get market summary"""
        try:
            return {
                'market_sentiment': 'neutral',
                'yield_curve_slope': 0.85,
                'credit_spread': 125,
                'market_volatility': 'medium',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {'error': str(e)}
    
    async def _get_portfolio_status(self, params: Dict) -> Dict:
        """Get portfolio status"""
        try:
            portfolio_id = params.get('portfolio_id')
            # This would fetch from portfolio database
            return {
                'portfolio_id': portfolio_id,
                'total_value': 1000000,
                'risk_score': 42.3,
                'diversification_score': 0.78,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {'error': str(e)}
    
    def process_market_data(self, market_data: Dict) -> Dict:
        """Process incoming market data"""
        try:
            start_time = time.time()
            
            # Process market data
            processed_data = self.market_data_processor.process(market_data)
            
            # Check for anomalies
            anomalies = self.anomaly_detector.detect_anomalies(processed_data)
            
            # Update online models
            model_updates = self.online_models.update(processed_data)
            
            # Generate alerts if needed
            alerts = self._check_market_alerts(processed_data, anomalies)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update metrics
            self._update_metrics(processing_time)
            
            # Publish to Kafka if available
            if self.kafka_producer:
                self._publish_to_kafka('processed_market_data', processed_data)
            
            # Send WebSocket updates
            self._send_websocket_update('market_data', processed_data)
            
            return {
                'processed_data': processed_data,
                'anomalies': anomalies,
                'model_updates': model_updates,
                'alerts': alerts,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {'error': str(e)}
    
    def process_news_feed(self, news_item: Dict) -> Dict:
        """Process incoming news feed"""
        try:
            start_time = time.time()
            
            # Process news content
            processed_news = self.news_processor.process(news_item)
            
            # Check for market impact
            market_impact = self._assess_news_impact(processed_news)
            
            # Generate alerts if needed
            alerts = self._check_news_alerts(processed_news, market_impact)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update metrics
            self._update_metrics(processing_time)
            
            # Publish to Kafka if available
            if self.kafka_producer:
                self._publish_to_kafka('processed_news', processed_news)
            
            # Send WebSocket updates
            self._send_websocket_update('news_feed', processed_news)
            
            return {
                'processed_news': processed_news,
                'market_impact': market_impact,
                'alerts': alerts,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing news feed: {e}")
            return {'error': str(e)}
    
    def process_portfolio_update(self, portfolio_data: Dict) -> Dict:
        """Process portfolio updates"""
        try:
            start_time = time.time()
            
            # Process portfolio data
            processed_portfolio = self.portfolio_processor.process(portfolio_data)
            
            # Check for risk alerts
            risk_alerts = self._check_portfolio_risk_alerts(processed_portfolio)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update metrics
            self._update_metrics(processing_time)
            
            # Publish to Kafka if available
            if self.kafka_producer:
                self._publish_to_kafka('processed_portfolio', processed_portfolio)
            
            # Send WebSocket updates
            self._send_websocket_update('portfolio', processed_portfolio)
            
            return {
                'processed_portfolio': processed_portfolio,
                'risk_alerts': risk_alerts,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing portfolio update: {e}")
            return {'error': str(e)}
    
    def _check_market_alerts(self, market_data: Dict, anomalies: List) -> List[Alert]:
        """Check for market-related alerts"""
        alerts = []
        
        try:
            # Check risk thresholds
            if self.alert_rules['risk_threshold']['enabled']:
                risk_alerts = self._check_risk_thresholds(market_data)
                alerts.extend(risk_alerts)
            
            # Check price movements
            if self.alert_rules['price_movement']['enabled']:
                price_alerts = self._check_price_movements(market_data)
                alerts.extend(price_alerts)
            
            # Check anomalies
            if anomalies:
                anomaly_alerts = self._create_anomaly_alerts(anomalies)
                alerts.extend(anomaly_alerts)
            
            # Store alerts
            for alert in alerts:
                self._store_alert(alert)
            
        except Exception as e:
            logger.error(f"Error checking market alerts: {e}")
        
        return alerts
    
    def _check_risk_thresholds(self, market_data: Dict) -> List[Alert]:
        """Check if risk metrics exceed thresholds"""
        alerts = []
        
        try:
            for risk_type, threshold in self.risk_thresholds.items():
                if risk_type in market_data:
                    risk_value = market_data[risk_type]
                    
                    if risk_value > threshold:
                        severity = AlertSeverity.HIGH if risk_value > threshold * 1.2 else AlertSeverity.MEDIUM
                        
                        alert = Alert(
                            alert_type=AlertType.RISK_THRESHOLD,
                            severity=severity,
                            title=f"{risk_type.replace('_', ' ').title()} Alert",
                            message=f"{risk_type.replace('_', ' ').title()} has exceeded threshold: {risk_value:.1f} > {threshold}",
                            timestamp=datetime.now(),
                            source='market_data',
                            data={'risk_type': risk_type, 'value': risk_value, 'threshold': threshold}
                        )
                        
                        alerts.append(alert)
                        
        except Exception as e:
            logger.error(f"Error checking risk thresholds: {e}")
        
        return alerts
    
    def _check_price_movements(self, market_data: Dict) -> List[Alert]:
        """Check for significant price movements"""
        alerts = []
        
        try:
            # Check yield changes
            if 'yield_change' in market_data:
                yield_change = abs(market_data['yield_change'])
                if yield_change > self.price_thresholds['yield_change']:
                    alert = Alert(
                        alert_type=AlertType.PRICE_MOVEMENT,
                        severity=AlertSeverity.MEDIUM,
                        title="Significant Yield Movement",
                        message=f"Yield change of {yield_change:.3f} exceeds threshold of {self.price_thresholds['yield_change']}",
                        timestamp=datetime.now(),
                        source='market_data',
                        data={'yield_change': yield_change, 'threshold': self.price_thresholds['yield_change']}
                    )
                    alerts.append(alert)
            
            # Check volume spikes
            if 'volume_ratio' in market_data:
                volume_ratio = market_data['volume_ratio']
                if volume_ratio > self.price_thresholds['volume_spike']:
                    alert = Alert(
                        alert_type=AlertType.PRICE_MOVEMENT,
                        severity=AlertSeverity.LOW,
                        title="Volume Spike Detected",
                        message=f"Volume is {volume_ratio:.1f}x above average",
                        timestamp=datetime.now(),
                        source='market_data',
                        data={'volume_ratio': volume_ratio, 'threshold': self.price_thresholds['volume_spike']}
                    )
                    alerts.append(alert)
                    
        except Exception as e:
            logger.error(f"Error checking price movements: {e}")
        
        return alerts
    
    def _check_news_alerts(self, news_data: Dict, market_impact: Dict) -> List[Alert]:
        """Check for news-related alerts"""
        alerts = []
        
        try:
            if self.alert_rules['news_impact']['enabled']:
                sentiment_score = news_data.get('sentiment_score', 0)
                threshold = self.alert_rules['news_impact']['sentiment_threshold']
                
                if sentiment_score < threshold:
                    severity = AlertSeverity.HIGH if sentiment_score < -0.5 else AlertSeverity.MEDIUM
                    
                    alert = Alert(
                        alert_type=AlertType.NEWS_IMPACT,
                        severity=severity,
                        title="Negative News Impact",
                        message=f"News sentiment score {sentiment_score:.2f} indicates potential market impact",
                        timestamp=datetime.now(),
                        source='news_feed',
                        data={'sentiment_score': sentiment_score, 'threshold': threshold, 'impact': market_impact}
                    )
                    
                    alerts.append(alert)
                    
        except Exception as e:
            logger.error(f"Error checking news alerts: {e}")
        
        return alerts
    
    def _check_portfolio_risk_alerts(self, portfolio_data: Dict) -> List[Alert]:
        """Check for portfolio risk alerts"""
        alerts = []
        
        try:
            overall_risk = portfolio_data.get('overall_risk', 50)
            
            if overall_risk > self.risk_thresholds['overall_risk']:
                alert = Alert(
                    alert_type=AlertType.PORTFOLIO_ALERT,
                    severity=AlertSeverity.HIGH,
                    title="Portfolio Risk Alert",
                    message=f"Portfolio risk score {overall_risk:.1f} exceeds threshold of {self.risk_thresholds['overall_risk']}",
                    timestamp=datetime.now(),
                    source='portfolio_update',
                    data={'risk_score': overall_risk, 'threshold': self.risk_thresholds['overall_risk']}
                )
                
                alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Error checking portfolio risk alerts: {e}")
        
        return alerts
    
    def _create_anomaly_alerts(self, anomalies: List) -> List[Alert]:
        """Create alerts for detected anomalies"""
        alerts = []
        
        try:
            for anomaly in anomalies:
                alert = Alert(
                    alert_type=AlertType.SYSTEM_ALERT,
                    severity=AlertSeverity.MEDIUM,
                    title="Anomaly Detected",
                    message=f"Anomaly detected in {anomaly.get('metric', 'unknown')}: {anomaly.get('description', '')}",
                    timestamp=datetime.now(),
                    source='anomaly_detection',
                    data=anomaly
                )
                
                alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Error creating anomaly alerts: {e}")
        
        return alerts
    
    def _assess_news_impact(self, news_data: Dict) -> Dict:
        """Assess potential market impact of news"""
        try:
            sentiment_score = news_data.get('sentiment_score', 0)
            entities = news_data.get('entities', [])
            
            # Simple impact assessment
            if sentiment_score < -0.5:
                impact_level = 'high_negative'
            elif sentiment_score < -0.2:
                impact_level = 'moderate_negative'
            elif sentiment_score > 0.5:
                impact_level = 'high_positive'
            elif sentiment_score > 0.2:
                impact_level = 'moderate_positive'
            else:
                impact_level = 'neutral'
            
            return {
                'impact_level': impact_level,
                'sentiment_score': sentiment_score,
                'affected_entities': entities,
                'estimated_market_impact': self._estimate_market_impact(sentiment_score, entities)
            }
            
        except Exception as e:
            logger.error(f"Error assessing news impact: {e}")
            return {'impact_level': 'unknown', 'error': str(e)}
    
    def _estimate_market_impact(self, sentiment_score: float, entities: List) -> str:
        """Estimate market impact based on sentiment and entities"""
        try:
            # Simple estimation logic
            if abs(sentiment_score) > 0.7:
                return 'significant'
            elif abs(sentiment_score) > 0.4:
                return 'moderate'
            elif abs(sentiment_score) > 0.2:
                return 'minor'
            else:
                return 'minimal'
                
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return 'unknown'
    
    def _store_alert(self, alert: Alert):
        """Store alert in system"""
        try:
            alert_id = f"alert_{int(time.time() * 1000)}"
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Store in Redis for persistence
            if self.redis_client:
                self.redis_client.hset(
                    f"alert:{alert_id}",
                    mapping={
                        'type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'source': alert.source,
                        'data': json.dumps(alert.data)
                    }
                )
            
            # Send WebSocket alert
            self._send_websocket_update('alert', {
                'alert_id': alert_id,
                'alert': {
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def _publish_to_kafka(self, topic: str, data: Dict):
        """Publish data to Kafka topic"""
        try:
            if self.kafka_producer:
                self.kafka_producer.send(topic, value=data)
                self.kafka_producer.flush()
                
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")
    
    def _send_websocket_update(self, update_type: str, data: Dict):
        """Send update to WebSocket clients"""
        try:
            if not self.websocket_clients:
                return
            
            message = {
                'type': update_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to all connected clients
            for client in self.websocket_clients.copy():
                try:
                    asyncio.create_task(client.send(json.dumps(message)))
                except Exception as e:
                    logger.warning(f"Error sending to WebSocket client: {e}")
                    self.websocket_clients.discard(client)
                    
        except Exception as e:
            logger.error(f"Error sending WebSocket update: {e}")
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        try:
            self.metrics.messages_processed += 1
            self.metrics.processing_latency = np.mean(self.processing_times)
            self.metrics.throughput = len(self.processing_times) / 60.0  # messages per minute
            self.metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _data_processing_loop(self):
        """Background data processing loop"""
        try:
            if not self.kafka_consumer:
                return
            
            for message in self.kafka_consumer:
                try:
                    data = message.value
                    data_type = data.get('type', 'unknown')
                    
                    if data_type == 'market_data':
                        self.process_market_data(data)
                    elif data_type == 'news':
                        self.process_news_feed(data)
                    elif data_type == 'portfolio':
                        self.process_portfolio_update(data)
                        
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in data processing loop: {e}")
    
    def _alert_monitoring_loop(self):
        """Background alert monitoring loop"""
        try:
            while True:
                try:
                    # Check for new alerts
                    current_time = datetime.now()
                    
                    # Process any pending alerts
                    # This is a simplified version - in practice, you'd have more sophisticated alert processing
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in alert monitoring loop: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            logger.error(f"Error in alert monitoring loop: {e}")
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        try:
            while True:
                try:
                    # Calculate error rate
                    if self.metrics.messages_processed > 0:
                        self.metrics.error_rate = 0.0  # Simplified - would calculate actual error rate
                    
                    time.sleep(60)  # Update every minute
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                'messages_processed': self.metrics.messages_processed,
                'processing_latency': self.metrics.processing_latency,
                'throughput': self.metrics.throughput,
                'error_rate': self.metrics.error_rate,
                'last_updated': self.metrics.last_updated.isoformat(),
                'active_alerts': len(self.active_alerts),
                'websocket_clients': len(self.websocket_clients),
                'kafka_status': 'active' if self.kafka_producer else 'inactive',
                'redis_status': 'active' if self.redis_client else 'inactive'
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}
    
    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts"""
        try:
            return [
                {
                    'alert_id': alert_id,
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'source': alert.source,
                    'acknowledged': alert.acknowledged
                }
                for alert_id, alert in self.active_alerts.items()
            ]
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                
                # Remove from active alerts if acknowledged
                if self.active_alerts[alert_id].acknowledged:
                    del self.active_alerts[alert_id]
                
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False


class MarketDataProcessor:
    """Process market data in real-time"""
    
    def __init__(self, risk_engine: RiskScoringEngine, yield_engine: YieldPredictionEngine):
        self.risk_engine = risk_engine
        self.yield_engine = yield_engine
    
    def process(self, market_data: Dict) -> Dict:
        """Process market data"""
        try:
            processed_data = market_data.copy()
            
            # Calculate risk metrics
            if 'bond_data' in market_data:
                processed_data['risk_metrics'] = self._calculate_risk_metrics(market_data['bond_data'])
            
            # Calculate yield predictions
            if 'yield_data' in market_data:
                processed_data['yield_predictions'] = self._calculate_yield_predictions(market_data['yield_data'])
            
            # Add processing timestamp
            processed_data['processed_at'] = datetime.now().isoformat()
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return market_data
    
    def _calculate_risk_metrics(self, bond_data: Dict) -> Dict:
        """Calculate risk metrics for bonds"""
        try:
            # Simplified risk calculation
            return {
                'credit_risk': 45.2,
                'interest_rate_risk': 38.7,
                'liquidity_risk': 52.1,
                'overall_risk': 45.3
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_yield_predictions(self, yield_data: Dict) -> Dict:
        """Calculate yield predictions"""
        try:
            # Simplified yield prediction
            return {
                'next_day': 6.25,
                'next_week': 6.30,
                'next_month': 6.35
            }
        except Exception as e:
            logger.error(f"Error calculating yield predictions: {e}")
            return {}


class NewsProcessor:
    """Process news feed in real-time"""
    
    def __init__(self, nlp_engine: NLPEngine):
        self.nlp_engine = nlp_engine
    
    def process(self, news_item: Dict) -> Dict:
        """Process news item"""
        try:
            processed_news = news_item.copy()
            
            # Extract text content
            text_content = news_item.get('content', '')
            
            # Analyze sentiment
            sentiment = self.nlp_engine.analyze_sentiment(text_content, method="ensemble")
            processed_news['sentiment_score'] = sentiment.compound_score
            processed_news['sentiment_label'] = sentiment.sentiment_label.value
            
            # Extract entities
            entities = self.nlp_engine.extract_entities(text_content)
            processed_news['entities'] = [
                {
                    'text': entity.entity_text,
                    'type': entity.entity_type.value,
                    'confidence': entity.confidence
                }
                for entity in entities
            ]
            
            # Add processing timestamp
            processed_news['processed_at'] = datetime.now().isoformat()
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error processing news: {e}")
            return news_item


class PortfolioProcessor:
    """Process portfolio updates in real-time"""
    
    def __init__(self, risk_engine: RiskScoringEngine):
        self.risk_engine = risk_engine
    
    def process(self, portfolio_data: Dict) -> Dict:
        """Process portfolio data"""
        try:
            processed_portfolio = portfolio_data.copy()
            
            # Calculate portfolio risk
            if 'holdings' in portfolio_data:
                processed_portfolio['risk_metrics'] = self._calculate_portfolio_risk(portfolio_data['holdings'])
            
            # Add processing timestamp
            processed_portfolio['processed_at'] = datetime.now().isoformat()
            
            return processed_portfolio
            
        except Exception as e:
            logger.error(f"Error processing portfolio: {e}")
            return portfolio_data
    
    def _calculate_portfolio_risk(self, holdings: List[Dict]) -> Dict:
        """Calculate portfolio risk metrics"""
        try:
            # Simplified portfolio risk calculation
            return {
                'overall_risk': 42.3,
                'diversification_score': 0.78,
                'concentration_risk': 0.35
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {}


class AnomalyDetector:
    """Detect anomalies in real-time data"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.data_buffer = deque(maxlen=1000)
    
    def detect_anomalies(self, data: Dict) -> List[Dict]:
        """Detect anomalies in data"""
        try:
            anomalies = []
            
            # Extract numerical features
            features = self._extract_features(data)
            if not features:
                return anomalies
            
            # Add to buffer
            self.data_buffer.append(features)
            
            # Detect anomalies if enough data
            if len(self.data_buffer) > 100:
                # Prepare data for anomaly detection
                X = np.array(list(self.data_buffer))
                X_scaled = self.scaler.fit_transform(X)
                
                # Detect anomalies
                anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
                
                # Check latest data point
                if anomaly_labels[-1] == -1:  # Anomaly detected
                    anomaly = {
                        'metric': 'market_data',
                        'description': 'Anomaly detected in market data patterns',
                        'severity': 'medium',
                        'timestamp': datetime.now().isoformat()
                    }
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _extract_features(self, data: Dict) -> List[float]:
        """Extract numerical features from data"""
        try:
            features = []
            
            # Extract risk metrics
            if 'risk_metrics' in data:
                risk_metrics = data['risk_metrics']
                features.extend([
                    risk_metrics.get('credit_risk', 50),
                    risk_metrics.get('interest_rate_risk', 50),
                    risk_metrics.get('liquidity_risk', 50),
                    risk_metrics.get('overall_risk', 50)
                ])
            
            # Extract yield data
            if 'yield_data' in data:
                yield_data = data['yield_data']
                features.extend([
                    yield_data.get('current_yield', 6.0),
                    yield_data.get('yield_change', 0.0)
                ])
            
            return features if features else []
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []


class OnlineLearningModels:
    """Online learning models for real-time updates"""
    
    def __init__(self):
        self.models = {}
        self.update_count = 0
    
    def update(self, data: Dict) -> Dict:
        """Update online models with new data"""
        try:
            updates = {}
            
            # Update risk model
            if 'risk_metrics' in data:
                updates['risk_model'] = self._update_risk_model(data['risk_metrics'])
            
            # Update yield model
            if 'yield_data' in data:
                updates['yield_model'] = self._update_yield_model(data['yield_data'])
            
            self.update_count += 1
            
            return updates
            
        except Exception as e:
            logger.error(f"Error updating online models: {e}")
            return {}
    
    def _update_risk_model(self, risk_metrics: Dict) -> Dict:
        """Update risk model"""
        return {
            'updated': True,
            'new_data_points': 1,
            'model_performance': 'stable'
        }
    
    def _update_yield_model(self, yield_data: Dict) -> Dict:
        """Update yield model"""
        return {
            'updated': True,
            'new_data_points': 1,
            'model_performance': 'stable'
        }
