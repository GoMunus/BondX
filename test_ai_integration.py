"""
Comprehensive Testing Framework for AI Integration

This module provides comprehensive testing for all AI components including
the service layer, endpoints, monitoring, and integration scenarios.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

# FastAPI testing imports
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the application and components
from bondx.main import create_application
from bondx.ai_risk_engine.ai_service_layer import (
    AIServiceLayer, RiskAnalysisRequest, RiskAnalysisResponse,
    YieldPredictionRequest, YieldPredictionResponse,
    SentimentAnalysisRequest, SentimentAnalysisResponse,
    AdvisoryQueryRequest, AdvisoryQueryResponse
)
from bondx.core.monitoring import AIMonitoringSystem

# Test configuration
TEST_CONFIG = {
    "cache_ttl": 60,  # 1 minute for testing
    "max_batch_size": 10,
    "model_update_interval": 300,  # 5 minutes for testing
    "enable_fallback_models": True,
    "enable_model_ensemble": True,
    "enable_real_time_updates": False,  # Disable for testing
    "redis_url": "redis://localhost:6379",
    "model_storage_path": "./test_models",
    "enable_monitoring": True
}

class TestAIServiceLayer:
    """Test suite for AI Service Layer"""
    
    @pytest.fixture
    async def ai_service(self):
        """Create a test AI service instance"""
        service = AIServiceLayer(TEST_CONFIG)
        # Mock Redis connection
        service.redis_client = None
        await service.initialize()
        yield service
        await service.cleanup()
    
    @pytest.fixture
    def mock_bond_data(self):
        """Mock bond data for testing"""
        return {
            "bond": Mock(
                isin="TEST123456789",
                coupon_rate=7.5,
                maturity_date=datetime.now() + timedelta(days=365*5),
                face_value=1000
            ),
            "quote": Mock(
                yield_to_maturity=8.2,
                price=95.5,
                timestamp=datetime.now()
            ),
            "issuer": Mock(
                name="Test Corporation",
                sector="Technology",
                credit_rating="AA"
            ),
            "ratings": [
                Mock(agency="CRISIL", rating="AA", outlook="Stable"),
                Mock(agency="ICRA", rating="AA+", outlook="Positive")
            ]
        }
    
    @pytest.mark.asyncio
    async def test_ai_service_initialization(self, ai_service):
        """Test AI service initialization"""
        assert ai_service.is_initialized == True
        assert ai_service.risk_engine is not None
        assert ai_service.yield_engine is not None
        assert ai_service.nlp_engine is not None
        assert ai_service.advisory_system is not None
    
    @pytest.mark.asyncio
    async def test_risk_analysis(self, ai_service, mock_bond_data):
        """Test risk analysis functionality"""
        # Mock the bond data retrieval
        with patch.object(ai_service, '_get_bond_data', return_value=mock_bond_data):
            request = RiskAnalysisRequest(
                isin="TEST123456789",
                include_historical=True,
                include_scenarios=True,
                confidence_level=0.95
            )
            
            result = await ai_service.analyze_risk(request)
            
            assert isinstance(result, RiskAnalysisResponse)
            assert result.isin == "TEST123456789"
            assert 0 <= result.overall_risk_score <= 1
            assert len(result.risk_breakdown) > 0
            assert len(result.recommendations) > 0
            assert result.model_version is not None
    
    @pytest.mark.asyncio
    async def test_yield_prediction(self, ai_service, mock_bond_data):
        """Test yield prediction functionality"""
        # Mock the bond data retrieval
        with patch.object(ai_service, '_get_bond_data', return_value=mock_bond_data):
            request = YieldPredictionRequest(
                isin="TEST123456789",
                prediction_horizon=30,
                include_scenarios=True,
                confidence_level=0.95
            )
            
            result = await ai_service.predict_yield(request)
            
            assert isinstance(result, YieldPredictionResponse)
            assert result.isin == "TEST123456789"
            assert result.prediction_horizon == 30
            assert 0 <= result.model_confidence <= 1
            assert len(result.feature_importance) > 0
            assert len(result.scenarios) > 0
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, ai_service):
        """Test sentiment analysis functionality"""
        request = SentimentAnalysisRequest(
            text="This is a positive financial news article about bond markets.",
            document_type="news_article",
            include_entities=True,
            include_topics=True
        )
        
        result = await ai_service.analyze_sentiment(request)
        
        assert isinstance(result, SentimentAnalysisResponse)
        assert result.text_hash is not None
        assert -1 <= result.sentiment_score <= 1
        assert result.sentiment_label in ["very_positive", "positive", "neutral", "negative", "very_negative"]
        assert 0 <= result.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_advisory_query(self, ai_service):
        """Test advisory query functionality"""
        request = AdvisoryQueryRequest(
            query="What are the best bond investment strategies for conservative investors?",
            user_profile="retail_investor",
            risk_tolerance="conservative",
            investment_horizon="5-10 years",
            investment_amount=50000
        )
        
        result = await ai_service.get_investment_advice(request)
        
        assert isinstance(result, AdvisoryQueryResponse)
        assert result.query_id is not None
        assert result.advice_type is not None
        assert len(result.recommendations) > 0
        assert 0 <= result.confidence_score <= 1
        assert len(result.sources) > 0
    
    @pytest.mark.asyncio
    async def test_health_status(self, ai_service):
        """Test health status functionality"""
        health_status = await ai_service.get_health_status()
        
        assert "status" in health_status
        assert "components" in health_status
        assert "timestamp" in health_status
        assert "model_count" in health_status
        assert "cache_status" in health_status

class TestAIEndpoints:
    """Test suite for AI API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app = create_application()
        return TestClient(app)
    
    @pytest.fixture
    def mock_ai_service(self):
        """Mock AI service for testing"""
        with patch('bondx.ai_risk_engine.ai_service_layer.ai_service') as mock:
            # Mock health check
            mock.get_health_status.return_value = {
                "status": "healthy",
                "components": {
                    "risk_engine": True,
                    "yield_engine": True,
                    "nlp_engine": True,
                    "advisory_system": True,
                    "overall": True
                },
                "timestamp": datetime.now(),
                "model_count": 5,
                "cache_status": "unavailable"
            }
            
            # Mock risk analysis
            mock.analyze_risk.return_value = RiskAnalysisResponse(
                isin="TEST123456789",
                overall_risk_score=0.3,
                risk_breakdown={"credit_risk": 0.2, "liquidity_risk": 0.4},
                confidence_interval=(0.25, 0.35),
                risk_factors=[],
                recommendations=["Monitor regularly", "Consider diversification"],
                last_updated=datetime.now(),
                model_version="1.0.0"
            )
            
            # Mock yield prediction
            mock.predict_yield.return_value = YieldPredictionResponse(
                isin="TEST123456789",
                predicted_yield=7.8,
                confidence_interval=(7.5, 8.1),
                prediction_horizon=30,
                model_confidence=0.85,
                feature_importance={"coupon_rate": 0.3, "maturity": 0.4},
                scenarios=[],
                last_updated=datetime.now(),
                model_version="1.0.0"
            )
            
            yield mock
    
    def test_ai_health_check(self, client, mock_ai_service):
        """Test AI health check endpoint"""
        response = client.get("/api/v1/ai/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert data["data"]["status"] == "healthy"
    
    def test_risk_analysis_endpoint(self, client, mock_ai_service):
        """Test risk analysis endpoint"""
        request_data = {
            "include_historical": True,
            "include_scenarios": True,
            "confidence_level": 0.95
        }
        
        response = client.post("/api/v1/ai/risk/analyze/TEST123456789", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["isin"] == "TEST123456789"
        assert data["overall_risk_score"] == 0.3
        assert len(data["recommendations"]) > 0
    
    def test_yield_prediction_endpoint(self, client, mock_ai_service):
        """Test yield prediction endpoint"""
        request_data = {
            "prediction_horizon": 30,
            "include_scenarios": True,
            "confidence_level": 0.95
        }
        
        response = client.post("/api/v1/ai/predictions/yield/TEST123456789", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["isin"] == "TEST123456789"
        assert data["predicted_yield"] == 7.8
        assert data["prediction_horizon"] == 30
    
    def test_sentiment_analysis_endpoint(self, client, mock_ai_service):
        """Test sentiment analysis endpoint"""
        # Mock sentiment analysis result
        mock_ai_service.analyze_sentiment.return_value = SentimentAnalysisResponse(
            text_hash="abc123",
            sentiment_score=0.7,
            sentiment_label="positive",
            confidence=0.85,
            entities=[],
            topics=[],
            key_phrases=["bond markets", "positive outlook"],
            timestamp=datetime.now()
        )
        
        request_data = {
            "text": "Bond markets show positive outlook for next quarter",
            "document_type": "news_article",
            "include_entities": True,
            "include_topics": True
        }
        
        response = client.post("/api/v1/ai/sentiment/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["sentiment_score"] == 0.7
        assert data["sentiment_label"] == "positive"
        assert data["confidence"] == 0.85
    
    def test_advisory_query_endpoint(self, client, mock_ai_service):
        """Test advisory query endpoint"""
        # Mock advisory result
        mock_ai_service.get_investment_advice.return_value = AdvisoryQueryResponse(
            query_id="query123",
            advice_type="investment_recommendation",
            title="Conservative Bond Strategy",
            summary="Focus on high-quality government and corporate bonds",
            detailed_explanation="Conservative investors should focus on...",
            recommendations=["Invest in AAA-rated bonds", "Diversify across sectors"],
            risk_assessment={"overall_risk": "low", "volatility": "low"},
            supporting_data={"market_analysis": "stable outlook"},
            confidence_score=0.9,
            sources=["CRISIL", "ICRA"],
            timestamp=datetime.now()
        )
        
        request_data = {
            "query": "What are the best bond investment strategies for conservative investors?",
            "user_profile": "retail_investor",
            "risk_tolerance": "conservative",
            "investment_horizon": "5-10 years",
            "investment_amount": 50000
        }
        
        response = client.post("/api/v1/ai/advisor/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["advice_type"] == "investment_recommendation"
        assert data["confidence_score"] == 0.9
        assert len(data["recommendations"]) > 0

class TestMonitoringSystem:
    """Test suite for monitoring system"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a test monitoring system instance"""
        return AIMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_request_metrics_recording(self, monitoring_system):
        """Test request metrics recording"""
        await monitoring_system.record_request_metrics(
            service="test_service",
            endpoint="test_endpoint",
            status="success",
            duration=0.5
        )
        
        # Check if metrics were recorded
        assert "test_service" in monitoring_system.metrics_storage["performance_history"]
        assert "test_endpoint" in monitoring_system.metrics_storage["performance_history"]["test_service"]
        
        records = monitoring_system.metrics_storage["performance_history"]["test_service"]["test_endpoint"]
        assert len(records) == 1
        assert records[0]["status"] == "success"
        assert records[0]["duration"] == 0.5
    
    @pytest.mark.asyncio
    async def test_model_prediction_recording(self, monitoring_system):
        """Test model prediction metrics recording"""
        await monitoring_system.record_model_prediction(
            model_type="risk",
            model_version="1.0.0",
            endpoint="analyze",
            duration=1.2,
            confidence=0.85,
            accuracy=0.88
        )
        
        # Check if metrics were recorded
        assert "risk" in monitoring_system.metrics_storage["model_performance"]
        assert "1.0.0" in monitoring_system.metrics_storage["model_performance"]["risk"]
        
        records = monitoring_system.metrics_storage["model_performance"]["risk"]["1.0.0"]
        assert len(records) == 1
        assert records[0]["duration"] == 1.2
        assert records[0]["confidence"] == 0.85
        assert records[0]["accuracy"] == 0.88
    
    @pytest.mark.asyncio
    async def test_business_metrics_recording(self, monitoring_system):
        """Test business metrics recording"""
        await monitoring_system.record_business_metric(
            metric_type="user_engagement",
            value=150,
            unit="users",
            metadata={"model_type": "risk", "feature": "analysis"}
        )
        
        # Check if metrics were recorded
        assert "user_engagement" in monitoring_system.metrics_storage["business_metrics"]
        
        metrics = monitoring_system.metrics_storage["business_metrics"]["user_engagement"]
        assert len(metrics) == 1
        assert metrics[0].value == 150
        assert metrics[0].unit == "users"
        assert metrics[0].metadata["model_type"] == "risk"
    
    @pytest.mark.asyncio
    async def test_error_recording(self, monitoring_system):
        """Test error recording"""
        await monitoring_system.record_error(
            service="test_service",
            error_type="validation_error",
            severity="warning",
            error_message="Invalid input data",
            context={"field": "isin", "value": "invalid"}
        )
        
        # Check if error was recorded
        error_logs = monitoring_system.metrics_storage["error_logs"]
        assert len(error_logs) == 1
        assert error_logs[0]["service"] == "test_service"
        assert error_logs[0]["error_type"] == "validation_error"
        assert error_logs[0]["severity"] == "warning"
        assert error_logs[0]["error_message"] == "Invalid input data"
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, monitoring_system):
        """Test performance summary generation"""
        # Record some test metrics
        await monitoring_system.record_request_metrics("test_service", "endpoint1", "success", 0.5)
        await monitoring_system.record_request_metrics("test_service", "endpoint1", "success", 0.8)
        await monitoring_system.record_request_metrics("test_service", "endpoint1", "failed", 1.2)
        
        # Get performance summary
        summary = await monitoring_system.get_performance_summary("test_service", "1h")
        
        assert summary["service"] == "test_service"
        assert summary["total_requests"] == 3
        assert summary["successful_requests"] == 2
        assert summary["failed_requests"] == 1
        assert summary["success_rate"] == 2/3
        assert summary["avg_latency"] > 0
    
    @pytest.mark.asyncio
    async def test_alerts_checking(self, monitoring_system):
        """Test alerts checking functionality"""
        # Record metrics that should trigger alerts
        await monitoring_system.record_model_prediction(
            model_type="risk",
            model_version="1.0.0",
            endpoint="analyze",
            duration=3.0,  # Above threshold
            confidence=0.85,
            accuracy=0.75  # Below threshold
        )
        
        # Check for alerts
        alerts = await monitoring_system.check_alerts()
        
        # Should have alerts for high latency and low accuracy
        assert len(alerts) > 0
        
        # Check for specific alert types
        alert_types = [alert["type"] for alert in alerts]
        assert "latency_high" in alert_types or "model_accuracy_low" in alert_types

class TestIntegrationScenarios:
    """Test suite for integration scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        app = create_application()
        return TestClient(app)
    
    def test_end_to_end_risk_analysis_workflow(self, client):
        """Test end-to-end risk analysis workflow"""
        # This would test the complete workflow from request to response
        # including all middleware, validation, and processing
        pass
    
    def test_batch_processing_workflow(self, client):
        """Test batch processing workflow"""
        # This would test batch processing of multiple requests
        pass
    
    def test_error_handling_workflow(self, client):
        """Test error handling workflow"""
        # This would test various error scenarios and handling
        pass
    
    def test_monitoring_integration(self, client):
        """Test monitoring integration with endpoints"""
        # This would test that monitoring is properly integrated
        pass

# Performance testing
class TestPerformance:
    """Test suite for performance testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system performance under concurrent load"""
        # This would test the system's ability to handle concurrent requests
        pass
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test system performance with large batch requests"""
        # This would test the system's ability to handle large batches
        pass
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage under load"""
        # This would test memory usage patterns
        pass

# Load testing utilities
def create_load_test_scenario(num_requests: int, concurrent: int = 10):
    """Create a load test scenario"""
    async def run_load_test():
        # Implementation for load testing
        pass
    
    return run_load_test

# Main test runner
if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
