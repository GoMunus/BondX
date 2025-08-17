"""
AI Endpoints Router

This module provides FastAPI endpoints for all AI services including
risk analysis, yield prediction, sentiment analysis, and investment advice.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from ...ai_risk_engine.ai_service_layer import (
    ai_service,
    RiskAnalysisRequest,
    RiskAnalysisResponse,
    YieldPredictionRequest,
    YieldPredictionResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    AdvisoryQueryRequest,
    AdvisoryQueryResponse
)

logger = logging.getLogger(__name__)

# Create AI router
router = APIRouter()

# Health check endpoint
@router.get("/health", tags=["ai-health"])
async def ai_health_check():
    """Check health status of AI services"""
    try:
        health_status = await ai_service.get_health_status()
        return {
            "status": "success",
            "data": health_status,
            "message": "AI services health check completed"
        }
    except Exception as e:
        logger.error(f"AI health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI health check failed: {str(e)}")

# Risk Analysis Endpoint
@router.post("/risk/analyze/{isin}", response_model=RiskAnalysisResponse, tags=["ai-risk"])
async def analyze_bond_risk(
    isin: str,
    request: RiskAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze risk for a specific bond
    
    This endpoint provides comprehensive risk analysis including:
    - Overall risk score
    - Risk factor breakdown
    - Confidence intervals
    - Risk mitigation recommendations
    """
    try:
        # Update request with ISIN from path
        request.isin = isin
        
        # Perform risk analysis
        result = await ai_service.analyze_risk(request)
        
        # Add background task for analytics
        background_tasks.add_task(_log_risk_analysis, isin, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk analysis failed for {isin}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

# Yield Prediction Endpoint
@router.post("/predictions/yield/{isin}", response_model=YieldPredictionResponse, tags=["ai-yield"])
async def predict_bond_yield(
    isin: str,
    request: YieldPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict yield for a specific bond
    
    This endpoint provides yield predictions including:
    - Predicted yield values
    - Confidence intervals
    - Scenario analysis
    - Feature importance
    """
    try:
        # Update request with ISIN from path
        request.isin = isin
        
        # Perform yield prediction
        result = await ai_service.predict_yield(request)
        
        # Add background task for analytics
        background_tasks.add_task(_log_yield_prediction, isin, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Yield prediction failed for {isin}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

# Sentiment Analysis Endpoint
@router.post("/sentiment/analyze", response_model=SentimentAnalysisResponse, tags=["ai-sentiment"])
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze sentiment of text content
    
    This endpoint provides sentiment analysis including:
    - Sentiment scores
    - Entity extraction
    - Topic modeling
    - Key phrase extraction
    """
    try:
        # Perform sentiment analysis
        result = await ai_service.analyze_sentiment(request)
        
        # Add background task for analytics
        background_tasks.add_task(_log_sentiment_analysis, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

# Advisory Query Endpoint
@router.post("/advisor/query", response_model=AdvisoryQueryResponse, tags=["ai-advisory"])
async def get_investment_advice(
    request: AdvisoryQueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Get investment advice based on user query
    
    This endpoint provides intelligent investment advice including:
    - Personalized recommendations
    - Risk assessment
    - Supporting data and sources
    - Confidence scores
    """
    try:
        # Get investment advice
        result = await ai_service.get_investment_advice(request)
        
        # Add background task for analytics
        background_tasks.add_task(_log_advisory_query, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advisory query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advisory query failed: {str(e)}")

# Batch Processing Endpoints
@router.post("/risk/analyze/batch", tags=["ai-risk"])
async def analyze_bond_risk_batch(
    isins: List[str] = Field(..., description="List of ISINs to analyze"),
    include_historical: bool = Query(True, description="Include historical analysis"),
    include_scenarios: bool = Query(True, description="Include scenario analysis"),
    confidence_level: float = Query(0.95, description="Confidence level for analysis")
):
    """
    Analyze risk for multiple bonds in batch
    
    This endpoint provides batch risk analysis for multiple bonds
    """
    try:
        results = []
        for isin in isins:
            try:
                request = RiskAnalysisRequest(
                    isin=isin,
                    include_historical=include_historical,
                    include_scenarios=include_scenarios,
                    confidence_level=confidence_level
                )
                result = await ai_service.analyze_risk(request)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze risk for {isin}: {str(e)}")
                results.append({
                    "isin": isin,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "status": "success",
            "data": results,
            "total_processed": len(isins),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r])
        }
        
    except Exception as e:
        logger.error(f"Batch risk analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch risk analysis failed: {str(e)}")

@router.post("/predictions/yield/batch", tags=["ai-yield"])
async def predict_bond_yield_batch(
    isins: List[str] = Field(..., description="List of ISINs to predict"),
    prediction_horizon: int = Query(30, description="Prediction horizon in days"),
    include_scenarios: bool = Query(True, description="Include scenario analysis"),
    confidence_level: float = Query(0.95, description="Confidence level for prediction")
):
    """
    Predict yield for multiple bonds in batch
    
    This endpoint provides batch yield predictions for multiple bonds
    """
    try:
        results = []
        for isin in isins:
            try:
                request = YieldPredictionRequest(
                    isin=isin,
                    prediction_horizon=prediction_horizon,
                    include_scenarios=include_scenarios,
                    confidence_level=confidence_level
                )
                result = await ai_service.predict_yield(request)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict yield for {isin}: {str(e)}")
                results.append({
                    "isin": isin,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "status": "success",
            "data": results,
            "total_processed": len(isins),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r])
        }
        
    except Exception as e:
        logger.error(f"Batch yield prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch yield prediction failed: {str(e)}")

# Model Information Endpoints
@router.get("/models", tags=["ai-models"])
async def get_model_info():
    """Get information about loaded AI models"""
    try:
        return {
            "status": "success",
            "data": {
                "model_registry": ai_service.model_registry,
                "health_status": ai_service.health_status,
                "cache_status": "available" if ai_service.redis_client else "unavailable"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/models/{model_type}", tags=["ai-models"])
async def get_model_type_info(model_type: str):
    """Get information about specific model type"""
    try:
        models = {
            name: info for name, info in ai_service.model_registry.items()
            if info["type"] == model_type
        }
        
        return {
            "status": "success",
            "data": {
                "model_type": model_type,
                "models": models,
                "count": len(models)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get model type info for {model_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model type info: {str(e)}")

# Cache Management Endpoints
@router.delete("/cache/clear", tags=["ai-cache"])
async def clear_ai_cache():
    """Clear all AI service caches"""
    try:
        if ai_service.redis_client:
            await ai_service.redis_client.flushdb()
            return {
                "status": "success",
                "message": "AI service cache cleared successfully"
            }
        else:
            return {
                "status": "warning",
                "message": "No Redis connection available for cache clearing"
            }
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/cache/status", tags=["ai-cache"])
async def get_cache_status():
    """Get AI service cache status"""
    try:
        if ai_service.redis_client:
            # Get cache info
            info = await ai_service.redis_client.info()
            return {
                "status": "success",
                "data": {
                    "cache_available": True,
                    "redis_info": info,
                    "cache_ttl": ai_service.config.cache_ttl
                }
            }
        else:
            return {
                "status": "warning",
                "data": {
                    "cache_available": False,
                    "message": "No Redis connection available"
                }
            }
    except Exception as e:
        logger.error(f"Failed to get cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

# Background task functions
async def _log_risk_analysis(isin: str, result: RiskAnalysisResponse):
    """Background task to log risk analysis results"""
    try:
        logger.info(f"Risk analysis completed for {isin}: score={result.overall_risk_score}")
        # Additional analytics and logging can be added here
    except Exception as e:
        logger.error(f"Failed to log risk analysis for {isin}: {str(e)}")

async def _log_yield_prediction(isin: str, result: YieldPredictionResponse):
    """Background task to log yield prediction results"""
    try:
        logger.info(f"Yield prediction completed for {isin}: yield={result.predicted_yield}")
        # Additional analytics and logging can be added here
    except Exception as e:
        logger.error(f"Failed to log yield prediction for {isin}: {str(e)}")

async def _log_sentiment_analysis(result: SentimentAnalysisResponse):
    """Background task to log sentiment analysis results"""
    try:
        logger.info(f"Sentiment analysis completed: score={result.sentiment_score}")
        # Additional analytics and logging can be added here
    except Exception as e:
        logger.error(f"Failed to log sentiment analysis: {str(e)}")

async def _log_advisory_query(result: AdvisoryQueryResponse):
    """Background task to log advisory query results"""
    try:
        logger.info(f"Advisory query completed: type={result.advice_type}, confidence={result.confidence_score}")
        # Additional analytics and logging can be added here
    except Exception as e:
        logger.error(f"Failed to log advisory query: {str(e)}")

# Export router
__all__ = ["router"]
