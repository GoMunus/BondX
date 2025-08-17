"""
Comprehensive Monitoring and Observability System

This module provides comprehensive monitoring, metrics collection, and observability
for the AI infrastructure including model performance, prediction accuracy, and
business metrics tracking.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Prometheus Metrics
# Request metrics
REQUEST_COUNT = Counter(
    "ai_requests_total",
    "Total AI service requests",
    ["service", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "ai_request_duration_seconds",
    "AI service request latency in seconds",
    ["service", "endpoint"]
)

# Model performance metrics
MODEL_PREDICTION_COUNT = Counter(
    "ai_model_predictions_total",
    "Total predictions made by AI models",
    ["model_type", "model_version", "endpoint"]
)

MODEL_PREDICTION_LATENCY = Histogram(
    "ai_model_prediction_duration_seconds",
    "AI model prediction latency in seconds",
    ["model_type", "model_version", "endpoint"]
)

MODEL_ACCURACY = Gauge(
    "ai_model_accuracy",
    "AI model accuracy score",
    ["model_type", "model_version"]
)

MODEL_CONFIDENCE = Histogram(
    "ai_model_confidence",
    "AI model prediction confidence scores",
    ["model_type", "model_version"]
)

# Business metrics
BUSINESS_IMPACT = Counter(
    "ai_business_impact_total",
    "Total business impact from AI predictions",
    ["metric_type", "model_type"]
)

USER_ENGAGEMENT = Counter(
    "ai_user_engagement_total",
    "Total user engagement with AI features",
    ["feature_type", "user_type"]
)

# Cache metrics
CACHE_HIT_RATIO = Gauge(
    "ai_cache_hit_ratio",
    "AI service cache hit ratio",
    ["cache_type"]
)

CACHE_SIZE = Gauge(
    "ai_cache_size",
    "AI service cache size in bytes",
    ["cache_type"]
)

# Error metrics
ERROR_COUNT = Counter(
    "ai_errors_total",
    "Total AI service errors",
    ["service", "error_type", "severity"]
)

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics data structure"""
    model_type: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    timestamp: datetime

@dataclass
class BusinessMetrics:
    """Business metrics data structure"""
    metric_type: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class UserInteractionMetrics:
    """User interaction metrics data structure"""
    user_id: str
    feature_type: str
    interaction_type: str
    duration: float
    success: bool
    timestamp: datetime

class AIMonitoringSystem:
    """
    Comprehensive monitoring system for AI infrastructure
    """
    
    def __init__(self):
        self.metrics_storage = {}
        self.alerting_rules = {}
        self.performance_baselines = {}
        self.is_initialized = False
        
        # Initialize metrics storage
        self._initialize_metrics_storage()
        
        # Initialize alerting rules
        self._initialize_alerting_rules()
        
        # Initialize performance baselines
        self._initialize_performance_baselines()
    
    def _initialize_metrics_storage(self):
        """Initialize metrics storage structures"""
        self.metrics_storage = {
            "model_performance": {},
            "business_metrics": {},
            "user_interactions": {},
            "error_logs": [],
            "performance_history": {}
        }
    
    def _initialize_alerting_rules(self):
        """Initialize alerting rules for monitoring"""
        self.alerting_rules = {
            "model_accuracy_threshold": 0.8,
            "latency_threshold_p95": 2.0,  # seconds
            "error_rate_threshold": 0.05,
            "cache_hit_ratio_threshold": 0.7,
            "business_impact_threshold": 0.1
        }
    
    def _initialize_performance_baselines(self):
        """Initialize performance baselines"""
        self.performance_baselines = {
            "risk_analysis": {
                "latency_p50": 0.5,
                "latency_p95": 1.5,
                "accuracy": 0.85
            },
            "yield_prediction": {
                "latency_p50": 1.0,
                "latency_p95": 3.0,
                "accuracy": 0.80
            },
            "sentiment_analysis": {
                "latency_p50": 0.3,
                "latency_p95": 1.0,
                "accuracy": 0.90
            },
            "advisory_system": {
                "latency_p50": 2.0,
                "latency_p95": 5.0,
                "accuracy": 0.85
            }
        }
    
    async def record_request_metrics(
        self,
        service: str,
        endpoint: str,
        status: str,
        duration: float
    ):
        """Record request metrics"""
        try:
            # Update Prometheus metrics
            REQUEST_COUNT.labels(
                service=service,
                endpoint=endpoint,
                status=status
            ).inc()
            
            REQUEST_LATENCY.labels(
                service=service,
                endpoint=endpoint
            ).observe(duration)
            
            # Store in local metrics
            if service not in self.metrics_storage["performance_history"]:
                self.metrics_storage["performance_history"][service] = {}
            
            if endpoint not in self.metrics_storage["performance_history"][service]:
                self.metrics_storage["performance_history"][service][endpoint] = []
            
            self.metrics_storage["performance_history"][service][endpoint].append({
                "timestamp": datetime.now(),
                "status": status,
                "duration": duration
            })
            
            # Keep only last 1000 records per endpoint
            if len(self.metrics_storage["performance_history"][service][endpoint]) > 1000:
                self.metrics_storage["performance_history"][service][endpoint] = \
                    self.metrics_storage["performance_history"][service][endpoint][-1000:]
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {str(e)}")
    
    async def record_model_prediction(
        self,
        model_type: str,
        model_version: str,
        endpoint: str,
        duration: float,
        confidence: float,
        accuracy: Optional[float] = None
    ):
        """Record model prediction metrics"""
        try:
            # Update Prometheus metrics
            MODEL_PREDICTION_COUNT.labels(
                model_type=model_type,
                model_version=model_version,
                endpoint=endpoint
            ).inc()
            
            MODEL_PREDICTION_LATENCY.labels(
                model_type=model_type,
                model_version=model_version,
                endpoint=endpoint
            ).observe(duration)
            
            MODEL_CONFIDENCE.labels(
                model_type=model_type,
                model_version=model_version
            ).observe(confidence)
            
            if accuracy is not None:
                MODEL_ACCURACY.labels(
                    model_type=model_type,
                    model_version=model_version
                ).set(accuracy)
            
            # Store in local metrics
            if model_type not in self.metrics_storage["model_performance"]:
                self.metrics_storage["model_performance"][model_type] = {}
            
            if model_version not in self.metrics_storage["model_performance"][model_type]:
                self.metrics_storage["model_performance"][model_type][model_version] = []
            
            self.metrics_storage["model_performance"][model_type][model_version].append({
                "timestamp": datetime.now(),
                "endpoint": endpoint,
                "duration": duration,
                "confidence": confidence,
                "accuracy": accuracy
            })
            
        except Exception as e:
            logger.error(f"Failed to record model prediction metrics: {str(e)}")
    
    async def record_business_metric(
        self,
        metric_type: str,
        value: float,
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record business metrics"""
        try:
            # Update Prometheus metrics
            BUSINESS_IMPACT.labels(
                metric_type=metric_type,
                model_type=metadata.get("model_type", "unknown") if metadata else "unknown"
            ).inc(value)
            
            # Store in local metrics
            business_metric = BusinessMetrics(
                metric_type=metric_type,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            if metric_type not in self.metrics_storage["business_metrics"]:
                self.metrics_storage["business_metrics"][metric_type] = []
            
            self.metrics_storage["business_metrics"][metric_type].append(business_metric)
            
            # Keep only last 1000 records per metric type
            if len(self.metrics_storage["business_metrics"][metric_type]) > 1000:
                self.metrics_storage["business_metrics"][metric_type] = \
                    self.metrics_storage["business_metrics"][metric_type][-1000:]
            
        except Exception as e:
            logger.error(f"Failed to record business metric: {str(e)}")
    
    async def record_user_interaction(
        self,
        user_id: str,
        feature_type: str,
        interaction_type: str,
        duration: float,
        success: bool
    ):
        """Record user interaction metrics"""
        try:
            # Update Prometheus metrics
            USER_ENGAGEMENT.labels(
                feature_type=feature_type,
                user_type="user"  # Could be enhanced with user classification
            ).inc()
            
            # Store in local metrics
            user_interaction = UserInteractionMetrics(
                user_id=user_id,
                feature_type=feature_type,
                interaction_type=interaction_type,
                duration=duration,
                success=success,
                timestamp=datetime.now()
            )
            
            if feature_type not in self.metrics_storage["user_interactions"]:
                self.metrics_storage["user_interactions"][feature_type] = []
            
            self.metrics_storage["user_interactions"][feature_type].append(user_interaction)
            
        except Exception as e:
            logger.error(f"Failed to record user interaction: {str(e)}")
    
    async def record_error(
        self,
        service: str,
        error_type: str,
        severity: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record error metrics"""
        try:
            # Update Prometheus metrics
            ERROR_COUNT.labels(
                service=service,
                error_type=error_type,
                severity=severity
            ).inc()
            
            # Store in local metrics
            error_log = {
                "timestamp": datetime.now(),
                "service": service,
                "error_type": error_type,
                "severity": severity,
                "error_message": error_message,
                "context": context or {}
            }
            
            self.metrics_storage["error_logs"].append(error_log)
            
            # Keep only last 1000 error logs
            if len(self.metrics_storage["error_logs"]) > 1000:
                self.metrics_storage["error_logs"] = \
                    self.metrics_storage["error_logs"][-1000:]
            
        except Exception as e:
            logger.error(f"Failed to record error: {str(e)}")
    
    async def update_cache_metrics(
        self,
        cache_type: str,
        hit_ratio: float,
        size_bytes: int
    ):
        """Update cache metrics"""
        try:
            # Update Prometheus metrics
            CACHE_HIT_RATIO.labels(cache_type=cache_type).set(hit_ratio)
            CACHE_SIZE.labels(cache_type=cache_type).set(size_bytes)
            
        except Exception as e:
            logger.error(f"Failed to update cache metrics: {str(e)}")
    
    async def get_performance_summary(self, service: str, time_window: str = "1h") -> Dict[str, Any]:
        """Get performance summary for a service"""
        try:
            if service not in self.metrics_storage["performance_history"]:
                return {"error": f"No performance data for service: {service}"}
            
            # Calculate time window
            now = datetime.now()
            if time_window == "1h":
                start_time = now - timedelta(hours=1)
            elif time_window == "24h":
                start_time = now - timedelta(days=1)
            elif time_window == "7d":
                start_time = now - timedelta(days=7)
            else:
                start_time = now - timedelta(hours=1)
            
            # Aggregate metrics
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            durations = []
            
            for endpoint, records in self.metrics_storage["performance_history"][service].items():
                for record in records:
                    if record["timestamp"] >= start_time:
                        total_requests += 1
                        durations.append(record["duration"])
                        
                        if record["status"] == "success":
                            successful_requests += 1
                        else:
                            failed_requests += 1
            
            if not durations:
                return {
                    "service": service,
                    "time_window": time_window,
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0
                }
            
            # Calculate statistics
            success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
            avg_latency = np.mean(durations)
            p95_latency = np.percentile(durations, 95)
            
            return {
                "service": service,
                "time_window": time_window,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "min_latency": np.min(durations),
                "max_latency": np.max(durations)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {str(e)}")
            return {"error": str(e)}
    
    async def get_model_performance_summary(self, model_type: str) -> Dict[str, Any]:
        """Get performance summary for a model type"""
        try:
            if model_type not in self.metrics_storage["model_performance"]:
                return {"error": f"No performance data for model: {model_type}"}
            
            model_data = self.metrics_storage["model_performance"][model_type]
            
            summary = {
                "model_type": model_type,
                "versions": {},
                "overall": {
                    "total_predictions": 0,
                    "avg_confidence": 0.0,
                    "avg_latency": 0.0,
                    "avg_accuracy": 0.0
                }
            }
            
            all_durations = []
            all_confidences = []
            all_accuracies = []
            
            for version, records in model_data.items():
                if not records:
                    continue
                
                durations = [r["duration"] for r in records]
                confidences = [r["confidence"] for r in records]
                accuracies = [r["accuracy"] for r in records if r["accuracy"] is not None]
                
                summary["versions"][version] = {
                    "total_predictions": len(records),
                    "avg_confidence": np.mean(confidences) if confidences else 0.0,
                    "avg_latency": np.mean(durations) if durations else 0.0,
                    "avg_accuracy": np.mean(accuracies) if accuracies else 0.0,
                    "p95_latency": np.percentile(durations, 95) if durations else 0.0
                }
                
                all_durations.extend(durations)
                all_confidences.extend(confidences)
                all_accuracies.extend(accuracies)
                
                summary["overall"]["total_predictions"] += len(records)
            
            if all_durations:
                summary["overall"]["avg_latency"] = np.mean(all_durations)
                summary["overall"]["p95_latency"] = np.percentile(all_durations, 95)
            
            if all_confidences:
                summary["overall"]["avg_confidence"] = np.mean(all_confidences)
            
            if all_accuracies:
                summary["overall"]["avg_accuracy"] = np.mean(all_accuracies)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get model performance summary: {str(e)}")
            return {"error": str(e)}
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alerting conditions"""
        try:
            alerts = []
            
            # Check model accuracy alerts
            for model_type in self.metrics_storage["model_performance"]:
                for version in self.metrics_storage["model_performance"][model_type]:
                    records = self.metrics_storage["model_performance"][model_type][version]
                    if records:
                        recent_records = [r for r in records if 
                                        (datetime.now() - r["timestamp"]).total_seconds() < 3600]
                        if recent_records:
                            avg_accuracy = np.mean([r["accuracy"] for r in recent_records if r["accuracy"] is not None])
                            if avg_accuracy < self.alerting_rules["model_accuracy_threshold"]:
                                alerts.append({
                                    "type": "model_accuracy_low",
                                    "severity": "warning",
                                    "message": f"Model {model_type} v{version} accuracy below threshold: {avg_accuracy:.3f}",
                                    "timestamp": datetime.now(),
                                    "details": {
                                        "model_type": model_type,
                                        "version": version,
                                        "current_accuracy": avg_accuracy,
                                        "threshold": self.alerting_rules["model_accuracy_threshold"]
                                    }
                                })
            
            # Check latency alerts
            for service in self.metrics_storage["performance_history"]:
                for endpoint in self.metrics_storage["performance_history"][service]:
                    records = self.metrics_storage["performance_history"][service][endpoint]
                    if records:
                        recent_records = [r for r in records if 
                                        (datetime.now() - r["timestamp"]).total_seconds() < 3600]
                        if recent_records:
                            p95_latency = np.percentile([r["duration"] for r in recent_records], 95)
                            if p95_latency > self.alerting_rules["latency_threshold_p95"]:
                                alerts.append({
                                    "type": "latency_high",
                                    "severity": "warning",
                                    "message": f"Service {service} endpoint {endpoint} P95 latency above threshold: {p95_latency:.3f}s",
                                    "timestamp": datetime.now(),
                                    "details": {
                                        "service": service,
                                        "endpoint": endpoint,
                                        "current_p95_latency": p95_latency,
                                        "threshold": self.alerting_rules["latency_threshold_p95"]
                                    }
                                })
            
            # Check error rate alerts
            recent_errors = [e for e in self.metrics_storage["error_logs"] if 
                           (datetime.now() - e["timestamp"]).total_seconds() < 3600]
            
            if recent_errors:
                total_requests = sum(len(records) for service in self.metrics_storage["performance_history"].values()
                                   for records in service.values())
                
                if total_requests > 0:
                    error_rate = len(recent_errors) / total_requests
                    if error_rate > self.alerting_rules["error_rate_threshold"]:
                        alerts.append({
                            "type": "error_rate_high",
                            "severity": "critical",
                            "message": f"Error rate above threshold: {error_rate:.3f}",
                            "timestamp": datetime.now(),
                            "details": {
                                "current_error_rate": error_rate,
                                "threshold": self.alerting_rules["error_rate_threshold"],
                                "total_requests": total_requests,
                                "total_errors": len(recent_errors)
                            }
                        })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {str(e)}")
            return []
    
    async def export_metrics(self) -> str:
        """Export Prometheus metrics"""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return ""
    
    async def get_metrics_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive metrics data for dashboard"""
        try:
            dashboard_data = {
                "overview": {
                    "total_requests": 0,
                    "total_predictions": 0,
                    "total_errors": 0,
                    "avg_response_time": 0.0
                },
                "services": {},
                "models": {},
                "alerts": await self.check_alerts(),
                "timestamp": datetime.now()
            }
            
            # Calculate overview metrics
            for service in self.metrics_storage["performance_history"]:
                for endpoint in self.metrics_storage["performance_history"][service]:
                    records = self.metrics_storage["performance_history"][service][endpoint]
                    dashboard_data["overview"]["total_requests"] += len(records)
                    
                    if records:
                        durations = [r["duration"] for r in records]
                        dashboard_data["overview"]["avg_response_time"] += np.mean(durations)
            
            # Calculate service metrics
            for service in self.metrics_storage["performance_history"]:
                dashboard_data["services"][service] = await self.get_performance_summary(service, "24h")
            
            # Calculate model metrics
            for model_type in self.metrics_storage["model_performance"]:
                dashboard_data["models"][model_type] = await self.get_model_performance_summary(model_type)
                dashboard_data["overview"]["total_predictions"] += dashboard_data["models"][model_type].get("overall", {}).get("total_predictions", 0)
            
            # Calculate error count
            dashboard_data["overview"]["total_errors"] = len(self.metrics_storage["error_logs"])
            
            # Calculate average response time
            if dashboard_data["overview"]["total_requests"] > 0:
                dashboard_data["overview"]["avg_response_time"] /= dashboard_data["overview"]["total_requests"]
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {str(e)}")
            return {"error": str(e)}

# Global monitoring instance
monitoring_system = AIMonitoringSystem()

# Export for use in other modules
__all__ = ["monitoring_system", "AIMonitoringSystem"]
