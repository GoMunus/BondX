"""
AI Service Layer - Unified Interface for All ML Components

This module provides a comprehensive service layer that integrates all AI components
including risk scoring, yield prediction, sentiment analysis, and advisory systems.
It handles model loading, caching, error handling, and provides production-ready
interfaces for the FastAPI application.
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings
from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd
from fastapi import HTTPException, BackgroundTasks
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

# AI Component imports
from .risk_scoring import RiskScoringEngine, RiskScore, RiskFactor
from .yield_prediction import YieldPredictionEngine, YieldPrediction, ModelType
from .nlp_engine import NLPEngine, DocumentType, EntityType
from .advisory_system import AdvisorySystem, InvestmentQuery, InvestmentAdvice

# Database imports
from ..database.models import Bond, Quote, Issuer, Rating
from ..database.base import get_db_session

logger = logging.getLogger(__name__)

# Prometheus metrics
AI_PREDICTION_COUNT = Counter(
    "ai_predictions_total",
    "Total AI predictions made",
    ["model_type", "endpoint"]
)

AI_PREDICTION_LATENCY = Histogram(
    "ai_prediction_duration_seconds",
    "AI prediction latency in seconds",
    ["model_type", "endpoint"]
)

AI_MODEL_ACCURACY = Gauge(
    "ai_model_accuracy",
    "AI model accuracy score",
    ["model_type", "model_version"]
)

AI_CACHE_HIT_RATIO = Gauge(
    "ai_cache_hit_ratio",
    "AI cache hit ratio",
    ["cache_type"]
)

@dataclass
class AIServiceConfig:
    """Configuration for AI services"""
    cache_ttl: int = 3600  # 1 hour
    max_batch_size: int = 100
    model_update_interval: int = 86400  # 24 hours
    enable_fallback_models: bool = True
    enable_model_ensemble: bool = True
    enable_real_time_updates: bool = True
    redis_url: str = "redis://localhost:6379"
    model_storage_path: str = "./models"
    enable_monitoring: bool = True

@dataclass
class RiskAnalysisRequest:
    """Risk analysis request structure"""
    isin: str
    include_historical: bool = True
    include_scenarios: bool = True
    confidence_level: float = 0.95
    user_profile: Optional[str] = None

@dataclass
class RiskAnalysisResponse:
    """Risk analysis response structure"""
    isin: str
    overall_risk_score: float
    risk_breakdown: Dict[str, float]
    confidence_interval: Tuple[float, float]
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    last_updated: datetime
    model_version: str

@dataclass
class YieldPredictionRequest:
    """Yield prediction request structure"""
    isin: str
    prediction_horizon: int = 30  # days
    include_scenarios: bool = True
    confidence_level: float = 0.95
    market_conditions: Optional[Dict[str, Any]] = None

@dataclass
class YieldPredictionResponse:
    """Yield prediction response structure"""
    isin: str
    predicted_yield: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: int
    model_confidence: float
    feature_importance: Dict[str, float]
    scenarios: List[Dict[str, Any]]
    last_updated: datetime
    model_version: str

@dataclass
class SentimentAnalysisRequest:
    """Sentiment analysis request structure"""
    text: str
    document_type: Optional[str] = None
    issuer_isin: Optional[str] = None
    include_entities: bool = True
    include_topics: bool = True

@dataclass
class SentimentAnalysisResponse:
    """Sentiment analysis response structure"""
    text_hash: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    entities: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]
    key_phrases: List[str]
    timestamp: datetime

@dataclass
class AdvisoryQueryRequest:
    """Advisory query request structure"""
    query: str
    user_profile: str
    risk_tolerance: str
    investment_horizon: str
    investment_amount: float
    current_portfolio: Optional[Dict[str, Any]] = None
    constraints: Optional[List[str]] = None

@dataclass
class AdvisoryQueryResponse:
    """Advisory query response structure"""
    query_id: str
    advice_type: str
    title: str
    summary: str
    detailed_explanation: str
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    supporting_data: Dict[str, Any]
    confidence_score: float
    sources: List[str]
    timestamp: datetime

class AIServiceLayer:
    """
    Comprehensive AI service layer that integrates all ML components
    """
    
    def __init__(self, config: AIServiceConfig = None):
        self.config = config or AIServiceConfig()
        self.redis_client = None
        self.is_initialized = False
        
        # Initialize AI components
        self.risk_engine = None
        self.yield_engine = None
        self.nlp_engine = None
        self.advisory_system = None
        
        # Model registry and cache
        self.model_registry = {}
        self.model_cache = {}
        self.feature_cache = {}
        
        # Health status
        self.health_status = {
            "risk_engine": False,
            "yield_engine": False,
            "nlp_engine": False,
            "advisory_system": False,
            "overall": False
        }
        
        # Background tasks
        self.background_tasks = []
        
    async def initialize(self):
        """Initialize all AI components and services"""
        try:
            logger.info("Initializing AI Service Layer...")
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize AI components
            await self._initialize_ai_components()
            
            # Load models
            await self._load_models()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            self.health_status["overall"] = True
            logger.info("AI Service Layer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Service Layer: {str(e)}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_client = None
    
    async def _initialize_ai_components(self):
        """Initialize all AI components"""
        try:
            # Initialize risk scoring engine
            self.risk_engine = RiskScoringEngine()
            self.health_status["risk_engine"] = True
            
            # Initialize yield prediction engine
            self.yield_engine = YieldPredictionEngine()
            self.health_status["yield_engine"] = True
            
            # Initialize NLP engine
            self.nlp_engine = NLPEngine()
            self.health_status["nlp_engine"] = True
            
            # Initialize advisory system
            self.advisory_system = AdvisorySystem()
            self.health_status["advisory_system"] = True
            
            logger.info("AI components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {str(e)}")
            raise
    
    async def _load_models(self):
        """Load trained models from storage"""
        try:
            model_path = Path(self.config.model_storage_path)
            if not model_path.exists():
                logger.warning(f"Model storage path does not exist: {model_path}")
                return
            
            # Load risk models
            risk_models = list(model_path.glob("risk_*.pkl"))
            for model_file in risk_models:
                await self._load_model("risk", model_file)
            
            # Load yield models
            yield_models = list(model_path.glob("yield_*.pkl"))
            for model_file in yield_models:
                await self._load_model("yield", model_file)
            
            logger.info(f"Loaded {len(self.model_registry)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
    
    async def _load_model(self, model_type: str, model_file: Path):
        """Load a specific model file"""
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            model_name = model_file.stem
            self.model_registry[model_name] = {
                "model": model,
                "type": model_type,
                "file_path": str(model_file),
                "loaded_at": datetime.now(),
                "version": getattr(model, 'version', 'unknown')
            }
            
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {str(e)}")
    
    async def _start_background_tasks(self):
        """Start background tasks for model updates and monitoring"""
        try:
            # Model update task
            asyncio.create_task(self._model_update_task())
            
            # Health monitoring task
            asyncio.create_task(self._health_monitoring_task())
            
            # Cache cleanup task
            asyncio.create_task(self._cache_cleanup_task())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {str(e)}")
    
    async def _model_update_task(self):
        """Background task for model updates"""
        while True:
            try:
                await asyncio.sleep(self.config.model_update_interval)
                await self._update_models()
            except Exception as e:
                logger.error(f"Model update task failed: {str(e)}")
    
    async def _health_monitoring_task(self):
        """Background task for health monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_health()
            except Exception as e:
                logger.error(f"Health monitoring task failed: {str(e)}")
    
    async def _cache_cleanup_task(self):
        """Background task for cache cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                await self._cleanup_cache()
            except Exception as e:
                logger.error(f"Cache cleanup task failed: {str(e)}")
    
    async def _update_models(self):
        """Update models with latest data"""
        try:
            logger.info("Starting model update process...")
            
            # Update risk models
            if self.risk_engine:
                await self._update_risk_models()
            
            # Update yield models
            if self.yield_engine:
                await self._update_yield_models()
            
            logger.info("Model update process completed")
            
        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")
    
    async def _check_health(self):
        """Check health of all AI components"""
        try:
            # Check risk engine
            if self.risk_engine:
                self.health_status["risk_engine"] = True
            
            # Check yield engine
            if self.yield_engine:
                self.health_status["yield_engine"] = True
            
            # Check NLP engine
            if self.nlp_engine:
                self.health_status["nlp_engine"] = True
            
            # Check advisory system
            if self.advisory_system:
                self.health_status["advisory_system"] = True
            
            # Overall health
            self.health_status["overall"] = all([
                self.health_status["risk_engine"],
                self.health_status["yield_engine"],
                self.health_status["nlp_engine"],
                self.health_status["advisory_system"]
            ])
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.health_status["overall"] = False
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            if self.redis_client:
                # Clean up feature cache
                await self._cleanup_feature_cache()
                
                # Clean up prediction cache
                await self._cleanup_prediction_cache()
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of AI services"""
        return {
            "status": "healthy" if self.health_status["overall"] else "degraded",
            "components": self.health_status,
            "timestamp": datetime.now(),
            "model_count": len(self.model_registry),
            "cache_status": "available" if self.redis_client else "unavailable"
        }

    async def analyze_risk(self, request: RiskAnalysisRequest) -> RiskAnalysisResponse:
        """Analyze risk for a specific bond"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"risk_analysis:{request.isin}:{hash(str(request))}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                return RiskAnalysisResponse(**cached_result)
            
            # Validate request
            if not request.isin:
                raise HTTPException(status_code=400, detail="ISIN is required")
            
            # Get bond data
            bond_data = await self._get_bond_data(request.isin)
            if not bond_data:
                raise HTTPException(status_code=404, detail="Bond not found")
            
            # Perform risk analysis
            risk_result = await self._perform_risk_analysis(bond_data, request)
            
            # Cache result
            await self._cache_result(cache_key, asdict(risk_result))
            
            # Update metrics
            duration = (datetime.now() - start_time).total_seconds()
            AI_PREDICTION_COUNT.labels(model_type="risk", endpoint="analyze").inc()
            AI_PREDICTION_LATENCY.labels(model_type="risk", endpoint="analyze").observe(duration)
            
            return risk_result
            
        except Exception as e:
            logger.error(f"Risk analysis failed for {request.isin}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")
    
    async def predict_yield(self, request: YieldPredictionRequest) -> YieldPredictionResponse:
        """Predict yield for a specific bond"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"yield_prediction:{request.isin}:{request.prediction_horizon}:{hash(str(request))}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                return YieldPredictionResponse(**cached_result)
            
            # Validate request
            if not request.isin:
                raise HTTPException(status_code=400, detail="ISIN is required")
            
            if request.prediction_horizon <= 0:
                raise HTTPException(status_code=400, detail="Prediction horizon must be positive")
            
            # Get bond data
            bond_data = await self._get_bond_data(request.isin)
            if not bond_data:
                raise HTTPException(status_code=404, detail="Bond not found")
            
            # Perform yield prediction
            prediction_result = await self._perform_yield_prediction(bond_data, request)
            
            # Cache result
            await self._cache_result(cache_key, asdict(prediction_result))
            
            # Update metrics
            duration = (datetime.now() - start_time).total_seconds()
            AI_PREDICTION_COUNT.labels(model_type="yield", endpoint="predict").inc()
            AI_PREDICTION_LATENCY.labels(model_type="yield", endpoint="predict").observe(duration)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Yield prediction failed for {request.isin}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")
    
    async def analyze_sentiment(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        """Analyze sentiment of text content"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            text_hash = hashlib.md5(request.text.encode()).hexdigest()
            cache_key = f"sentiment_analysis:{text_hash}:{hash(str(request))}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                return SentimentAnalysisResponse(**cached_result)
            
            # Validate request
            if not request.text:
                raise HTTPException(status_code=400, detail="Text content is required")
            
            # Perform sentiment analysis
            sentiment_result = await self._perform_sentiment_analysis(request)
            
            # Cache result
            await self._cache_result(cache_key, asdict(sentiment_result))
            
            # Update metrics
            duration = (datetime.now() - start_time).total_seconds()
            AI_PREDICTION_COUNT.labels(model_type="sentiment", endpoint="analyze").inc()
            AI_PREDICTION_LATENCY.labels(model_type="sentiment", endpoint="analyze").observe(duration)
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
    
    async def get_investment_advice(self, request: AdvisoryQueryRequest) -> AdvisoryQueryResponse:
        """Get investment advice based on user query"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            query_hash = hashlib.md5(request.query.encode()).hexdigest()
            cache_key = f"advisory_query:{query_hash}:{hash(str(request))}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                return AdvisoryQueryResponse(**cached_result)
            
            # Validate request
            if not request.query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            # Get investment advice
            advice_result = await self._perform_advisory_query(request)
            
            # Cache result
            await self._cache_result(cache_key, asdict(advice_result))
            
            # Update metrics
            duration = (datetime.now() - start_time).total_seconds()
            AI_PREDICTION_COUNT.labels(model_type="advisory", endpoint="query").inc()
            AI_PREDICTION_LATENCY.labels(model_type="advisory", endpoint="query").observe(duration)
            
            return advice_result
            
        except Exception as e:
            logger.error(f"Advisory query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Advisory query failed: {str(e)}")

    async def _get_bond_data(self, isin: str) -> Optional[Dict[str, Any]]:
        """Get bond data from database"""
        try:
            async with get_db_session() as session:
                # Get bond information
                bond = await session.get(Bond, isin)
                if not bond:
                    return None
                
                # Get latest quote
                quote = await session.query(Quote).filter(
                    Quote.isin == isin
                ).order_by(Quote.timestamp.desc()).first()
                
                # Get issuer information
                issuer = await session.get(Issuer, bond.issuer_id)
                
                # Get ratings
                ratings = await session.query(Rating).filter(
                    Rating.isin == isin
                ).all()
                
                return {
                    "bond": bond,
                    "quote": quote,
                    "issuer": issuer,
                    "ratings": ratings
                }
                
        except Exception as e:
            logger.error(f"Failed to get bond data for {isin}: {str(e)}")
            return None
    
    async def _perform_risk_analysis(self, bond_data: Dict[str, Any], request: RiskAnalysisRequest) -> RiskAnalysisResponse:
        """Perform comprehensive risk analysis"""
        try:
            # Extract bond information
            bond = bond_data["bond"]
            quote = bond_data["quote"]
            issuer = bond_data["issuer"]
            ratings = bond_data["ratings"]
            
            # Calculate risk scores using risk engine
            risk_scores = {}
            for factor in RiskFactor:
                score = await self._calculate_risk_factor_score(factor, bond, quote, issuer, ratings)
                risk_scores[factor.value] = score
            
            # Calculate overall risk score
            overall_score = np.mean(list(risk_scores.values()))
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(risk_scores, request.confidence_level)
            
            # Generate risk factors breakdown
            risk_factors = []
            for factor, score in risk_scores.items():
                risk_factors.append({
                    "factor": factor,
                    "score": score,
                    "description": self._get_risk_factor_description(factor, score),
                    "mitigation": self._get_risk_mitigation_strategies(factor, score)
                })
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_scores, overall_score)
            
            return RiskAnalysisResponse(
                isin=request.isin,
                overall_risk_score=overall_score,
                risk_breakdown=risk_scores,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                recommendations=recommendations,
                last_updated=datetime.now(),
                model_version=self._get_model_version("risk")
            )
            
        except Exception as e:
            logger.error(f"Risk analysis calculation failed: {str(e)}")
            raise
    
    async def _perform_yield_prediction(self, bond_data: Dict[str, Any], request: YieldPredictionRequest) -> YieldPredictionResponse:
        """Perform yield prediction"""
        try:
            # Extract bond information
            bond = bond_data["bond"]
            quote = bond_data["quote"]
            
            # Get market data
            market_data = await self._get_market_data()
            
            # Prepare features
            features = await self._prepare_yield_features(bond, quote, market_data)
            
            # Make prediction using yield engine
            prediction = await self.yield_engine.predict_yield(
                features=features,
                horizon=request.prediction_horizon,
                confidence_level=request.confidence_level
            )
            
            # Generate scenarios if requested
            scenarios = []
            if request.include_scenarios:
                scenarios = await self._generate_yield_scenarios(bond, quote, market_data, request.prediction_horizon)
            
            return YieldPredictionResponse(
                isin=request.isin,
                predicted_yield=prediction.predicted_yield,
                confidence_interval=prediction.confidence_interval,
                prediction_horizon=request.prediction_horizon,
                model_confidence=prediction.model_confidence,
                feature_importance=prediction.feature_importance,
                scenarios=scenarios,
                last_updated=datetime.now(),
                model_version=self._get_model_version("yield")
            )
            
        except Exception as e:
            logger.error(f"Yield prediction calculation failed: {str(e)}")
            raise
    
    async def _perform_sentiment_analysis(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        """Perform sentiment analysis"""
        try:
            # Analyze sentiment using NLP engine
            sentiment_result = await self.nlp_engine.analyze_sentiment(
                text=request.text,
                document_type=request.document_type
            )
            
            # Extract entities if requested
            entities = []
            if request.include_entities:
                entities = await self.nlp_engine.extract_entities(request.text)
            
            # Extract topics if requested
            topics = []
            if request.include_topics:
                topics = await self.nlp_engine.extract_topics(request.text)
            
            # Extract key phrases
            key_phrases = await self.nlp_engine.extract_key_phrases(request.text)
            
            return SentimentAnalysisResponse(
                text_hash=hashlib.md5(request.text.encode()).hexdigest(),
                sentiment_score=sentiment_result.compound_score,
                sentiment_label=sentiment_result.sentiment_label.value,
                confidence=sentiment_result.confidence,
                entities=entities,
                topics=topics,
                key_phrases=key_phrases,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis calculation failed: {str(e)}")
            raise
    
    async def _perform_advisory_query(self, request: AdvisoryQueryRequest) -> AdvisoryQueryResponse:
        """Perform advisory query"""
        try:
            # Create investment query
            investment_query = InvestmentQuery(
                query_text=request.query,
                user_profile=request.user_profile,
                risk_tolerance=request.risk_tolerance,
                investment_horizon=request.investment_horizon,
                investment_amount=request.investment_amount,
                current_portfolio=request.current_portfolio,
                constraints=request.constraints
            )
            
            # Get advice from advisory system
            advice = await self.advisory_system.get_investment_advice(investment_query)
            
            return AdvisoryQueryResponse(
                query_id=hashlib.md5(request.query.encode()).hexdigest(),
                advice_type=advice.advice_type.value,
                title=advice.title,
                summary=advice.summary,
                detailed_explanation=advice.detailed_explanation,
                recommendations=advice.recommendations,
                risk_assessment=advice.risk_assessment,
                supporting_data=advice.supporting_data,
                confidence_score=advice.confidence_score,
                sources=advice.sources,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Advisory query calculation failed: {str(e)}")
            raise
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        return None
    
    async def _cache_result(self, key: str, data: Dict[str, Any]):
        """Cache result"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    key,
                    self.config.cache_ttl,
                    json.dumps(data, default=str)
                )
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
    
    def _calculate_confidence_interval(self, scores: Dict[str, float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for risk scores"""
        values = list(scores.values())
        mean = np.mean(values)
        std = np.std(values)
        
        # Use t-distribution for small samples
        if len(values) < 30:
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence_level) / 2, len(values) - 1)
            margin = t_value * std / np.sqrt(len(values))
        else:
            # Use normal distribution for large samples
            from scipy import stats
            z_value = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_value * std / np.sqrt(len(values))
        
        return (max(0, mean - margin), min(1, mean + margin))
    
    def _get_risk_factor_description(self, factor: str, score: float) -> str:
        """Get description for risk factor"""
        descriptions = {
            "credit_risk": "Risk of issuer default or credit rating downgrade",
            "interest_rate_risk": "Risk of bond value changes due to interest rate fluctuations",
            "liquidity_risk": "Risk of inability to sell bond quickly at fair price",
            "concentration_risk": "Risk of over-exposure to specific issuer or sector",
            "esg_risk": "Environmental, social, and governance risk factors",
            "operational_risk": "Risk of operational failures or inefficiencies"
        }
        return descriptions.get(factor, "Unknown risk factor")
    
    def _get_risk_mitigation_strategies(self, factor: str, score: float) -> List[str]:
        """Get mitigation strategies for risk factor"""
        strategies = {
            "credit_risk": ["Diversify across issuers", "Monitor credit ratings", "Use credit default swaps"],
            "interest_rate_risk": ["Match duration to investment horizon", "Use floating rate bonds", "Ladder maturities"],
            "liquidity_risk": ["Focus on liquid bonds", "Maintain cash reserves", "Use bond ETFs"],
            "concentration_risk": ["Diversify across sectors", "Limit single issuer exposure", "Use index funds"],
            "esg_risk": ["ESG screening", "Engagement with issuers", "ESG-focused funds"],
            "operational_risk": ["Due diligence", "Regular monitoring", "Backup systems"]
        }
        return strategies.get(factor, [])
    
    def _generate_risk_recommendations(self, risk_scores: Dict[str, float], overall_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if overall_score > 0.7:
            recommendations.append("Consider reducing exposure to high-risk bonds")
            recommendations.append("Implement strict risk monitoring and alerts")
            recommendations.append("Diversify portfolio across different risk levels")
        
        elif overall_score > 0.4:
            recommendations.append("Monitor risk factors regularly")
            recommendations.append("Consider hedging strategies for major risk factors")
            recommendations.append("Maintain balanced portfolio allocation")
        
        else:
            recommendations.append("Current risk level is acceptable")
            recommendations.append("Continue monitoring for changes")
            recommendations.append("Consider opportunities for yield enhancement")
        
        return recommendations
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data"""
        # This would integrate with real-time market data feeds
        # For now, return mock data
        return {
            "repo_rate": 6.5,
            "gsec_10y": 7.2,
            "inflation": 5.8,
            "gdp_growth": 6.8,
            "fiscal_deficit": 6.4
        }
    
    async def _prepare_yield_features(self, bond: Any, quote: Any, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for yield prediction"""
        # This would create comprehensive feature vectors
        # For now, return basic features
        features = [
            bond.coupon_rate,
            bond.maturity_date.year - datetime.now().year,
            quote.yield_to_maturity if quote else 0,
            market_data["repo_rate"],
            market_data["gsec_10y"],
            market_data["inflation"]
        ]
        return np.array(features).reshape(1, -1)
    
    async def _generate_yield_scenarios(self, bond: Any, quote: Any, market_data: Dict[str, Any], horizon: int) -> List[Dict[str, Any]]:
        """Generate yield scenarios"""
        scenarios = []
        
        # Base scenario
        scenarios.append({
            "scenario": "base",
            "probability": 0.6,
            "yield_change": 0,
            "description": "Current market conditions persist"
        })
        
        # Bullish scenario
        scenarios.append({
            "scenario": "bullish",
            "probability": 0.2,
            "yield_change": -0.5,
            "description": "Improving economic conditions, lower yields"
        })
        
        # Bearish scenario
        scenarios.append({
            "scenario": "bearish",
            "probability": 0.2,
            "yield_change": 0.5,
            "description": "Economic challenges, higher yields"
        })
        
        return scenarios
    
    def _get_model_version(self, model_type: str) -> str:
        """Get version of specific model type"""
        for name, info in self.model_registry.items():
            if info["type"] == model_type:
                return info["version"]
        return "unknown"
    
    async def _calculate_risk_factor_score(self, factor: RiskFactor, bond: Any, quote: Any, issuer: Any, ratings: List[Any]) -> float:
        """Calculate score for specific risk factor"""
        # This would implement sophisticated risk scoring logic
        # For now, return mock scores
        base_scores = {
            RiskFactor.CREDIT_RISK: 0.3,
            RiskFactor.INTEREST_RATE_RISK: 0.4,
            RiskFactor.LIQUIDITY_RISK: 0.2,
            RiskFactor.CONCENTRATION_RISK: 0.1,
            RiskFactor.ESG_RISK: 0.2,
            RiskFactor.OPERATIONAL_RISK: 0.1
        }
        return base_scores.get(factor, 0.5)
    
    async def _update_risk_models(self):
        """Update risk models with latest data"""
        # Placeholder for model update logic
        pass
    
    async def _update_yield_models(self):
        """Update yield models with latest data"""
        # Placeholder for model update logic
        pass
    
    async def _cleanup_feature_cache(self):
        """Clean up feature cache"""
        # Placeholder for cache cleanup logic
        pass
    
    async def _cleanup_prediction_cache(self):
        """Clean up prediction cache"""
        # Placeholder for cache cleanup logic
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            logger.info("AI Service Layer cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

# Global AI service instance
ai_service = AIServiceLayer()

# Export for use in FastAPI endpoints
__all__ = ["ai_service", "AIServiceLayer", "RiskAnalysisRequest", "RiskAnalysisResponse"]
