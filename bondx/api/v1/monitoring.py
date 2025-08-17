"""
Monitoring and Observability Endpoints

This module provides FastAPI endpoints for monitoring, metrics, and observability
of the AI infrastructure and overall system performance.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
import logging
from datetime import datetime

from ...core.monitoring import monitoring_system

logger = logging.getLogger(__name__)

# Create monitoring router
router = APIRouter()

# Metrics export endpoint
@router.get("/metrics", tags=["monitoring"])
async def export_metrics():
    """Export Prometheus metrics"""
    try:
        metrics = await monitoring_system.export_metrics()
        return StreamingResponse(
            iter([metrics]),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to export metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")

# Performance summary endpoints
@router.get("/performance/{service}", tags=["monitoring"])
async def get_service_performance(
    service: str,
    time_window: str = Query("24h", description="Time window for analysis (1h, 24h, 7d)")
):
    """Get performance summary for a specific service"""
    try:
        if time_window not in ["1h", "24h", "7d"]:
            raise HTTPException(status_code=400, detail="Invalid time window. Use 1h, 24h, or 7d")
        
        performance_data = await monitoring_system.get_performance_summary(service, time_window)
        
        if "error" in performance_data:
            raise HTTPException(status_code=404, detail=performance_data["error"])
        
        return {
            "status": "success",
            "data": performance_data,
            "message": f"Performance summary for {service} over {time_window}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance summary for {service}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

@router.get("/performance", tags=["monitoring"])
async def get_all_services_performance(
    time_window: str = Query("24h", description="Time window for analysis (1h, 24h, 7d)")
):
    """Get performance summary for all services"""
    try:
        if time_window not in ["1h", "24h", "7d"]:
            raise HTTPException(status_code=400, detail="Invalid time window. Use 1h, 24h, or 7d")
        
        # Get all available services
        services = list(monitoring_system.metrics_storage["performance_history"].keys())
        
        performance_data = {}
        for service in services:
            try:
                service_performance = await monitoring_system.get_performance_summary(service, time_window)
                if "error" not in service_performance:
                    performance_data[service] = service_performance
            except Exception as e:
                logger.warning(f"Failed to get performance for {service}: {str(e)}")
                performance_data[service] = {"error": str(e)}
        
        return {
            "status": "success",
            "data": {
                "time_window": time_window,
                "services": performance_data,
                "total_services": len(services)
            },
            "message": f"Performance summary for all services over {time_window}"
        }
        
    except Exception as e:
        logger.error(f"Failed to get all services performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

# Model performance endpoints
@router.get("/models/{model_type}/performance", tags=["monitoring"])
async def get_model_performance(model_type: str):
    """Get performance summary for a specific model type"""
    try:
        performance_data = await monitoring_system.get_model_performance_summary(model_type)
        
        if "error" in performance_data:
            raise HTTPException(status_code=404, detail=performance_data["error"])
        
        return {
            "status": "success",
            "data": performance_data,
            "message": f"Performance summary for {model_type} model"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model performance for {model_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@router.get("/models/performance", tags=["monitoring"])
async def get_all_models_performance():
    """Get performance summary for all models"""
    try:
        # Get all available model types
        model_types = list(monitoring_system.metrics_storage["model_performance"].keys())
        
        performance_data = {}
        for model_type in model_types:
            try:
                model_performance = await monitoring_system.get_model_performance_summary(model_type)
                if "error" not in model_performance:
                    performance_data[model_type] = model_performance
            except Exception as e:
                logger.warning(f"Failed to get performance for {model_type}: {str(e)}")
                performance_data[model_type] = {"error": str(e)}
        
        return {
            "status": "success",
            "data": {
                "models": performance_data,
                "total_models": len(model_types)
            },
            "message": "Performance summary for all models"
        }
        
    except Exception as e:
        logger.error(f"Failed to get all models performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models performance: {str(e)}")

# Business metrics endpoints
@router.get("/business-metrics", tags=["monitoring"])
async def get_business_metrics(
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    limit: int = Query(100, description="Maximum number of records to return")
):
    """Get business metrics data"""
    try:
        business_metrics = monitoring_system.metrics_storage["business_metrics"]
        
        if metric_type:
            if metric_type not in business_metrics:
                raise HTTPException(status_code=404, detail=f"Metric type {metric_type} not found")
            
            metrics = business_metrics[metric_type][-limit:] if limit > 0 else business_metrics[metric_type]
            
            return {
                "status": "success",
                "data": {
                    "metric_type": metric_type,
                    "metrics": metrics,
                    "total_records": len(metrics)
                },
                "message": f"Business metrics for {metric_type}"
            }
        else:
            # Return summary of all metric types
            summary = {}
            for mt, metrics in business_metrics.items():
                summary[mt] = {
                    "total_records": len(metrics),
                    "latest_value": metrics[-1].value if metrics else 0,
                    "latest_timestamp": metrics[-1].timestamp if metrics else None
                }
            
            return {
                "status": "success",
                "data": {
                    "metric_types": summary,
                    "total_metric_types": len(summary)
                },
                "message": "Business metrics summary"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get business metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get business metrics: {str(e)}")

# User interaction metrics endpoints
@router.get("/user-interactions", tags=["monitoring"])
async def get_user_interactions(
    feature_type: Optional[str] = Query(None, description="Filter by feature type"),
    limit: int = Query(100, description="Maximum number of records to return")
):
    """Get user interaction metrics data"""
    try:
        user_interactions = monitoring_system.metrics_storage["user_interactions"]
        
        if feature_type:
            if feature_type not in user_interactions:
                raise HTTPException(status_code=404, detail=f"Feature type {feature_type} not found")
            
            interactions = user_interactions[feature_type][-limit:] if limit > 0 else user_interactions[feature_type]
            
            return {
                "status": "success",
                "data": {
                    "feature_type": feature_type,
                    "interactions": interactions,
                    "total_records": len(interactions)
                },
                "message": f"User interactions for {feature_type}"
            }
        else:
            # Return summary of all feature types
            summary = {}
            for ft, interactions in user_interactions.items():
                summary[ft] = {
                    "total_interactions": len(interactions),
                    "successful_interactions": len([i for i in interactions if i.success]),
                    "failed_interactions": len([i for i in interactions if not i.success]),
                    "avg_duration": sum(i.duration for i in interactions) / len(interactions) if interactions else 0
                }
            
            return {
                "status": "success",
                "data": {
                    "feature_types": summary,
                    "total_feature_types": len(summary)
                },
                "message": "User interactions summary"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user interactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user interactions: {str(e)}")

# Error logs endpoints
@router.get("/errors", tags=["monitoring"])
async def get_error_logs(
    service: Optional[str] = Query(None, description="Filter by service"),
    error_type: Optional[str] = Query(None, description="Filter by error type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, description="Maximum number of records to return")
):
    """Get error logs"""
    try:
        error_logs = monitoring_system.metrics_storage["error_logs"]
        
        # Apply filters
        filtered_logs = error_logs
        
        if service:
            filtered_logs = [log for log in filtered_logs if log["service"] == service]
        
        if error_type:
            filtered_logs = [log for log in filtered_logs if log["error_type"] == error_type]
        
        if severity:
            filtered_logs = [log for log in filtered_logs if log["severity"] == severity]
        
        # Apply limit
        if limit > 0:
            filtered_logs = filtered_logs[-limit:]
        
        # Calculate summary statistics
        total_errors = len(filtered_logs)
        errors_by_service = {}
        errors_by_type = {}
        errors_by_severity = {}
        
        for log in filtered_logs:
            # Count by service
            service_name = log["service"]
            errors_by_service[service_name] = errors_by_service.get(service_name, 0) + 1
            
            # Count by error type
            type_name = log["error_type"]
            errors_by_type[type_name] = errors_by_type.get(type_name, 0) + 1
            
            # Count by severity
            severity_name = log["severity"]
            errors_by_severity[severity_name] = errors_by_severity.get(severity_name, 0) + 1
        
        return {
            "status": "success",
            "data": {
                "errors": filtered_logs,
                "summary": {
                    "total_errors": total_errors,
                    "errors_by_service": errors_by_service,
                    "errors_by_type": errors_by_type,
                    "errors_by_severity": errors_by_severity
                },
                "filters": {
                    "service": service,
                    "error_type": error_type,
                    "severity": severity,
                    "limit": limit
                }
            },
            "message": "Error logs retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get error logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get error logs: {str(e)}")

# Alerts endpoint
@router.get("/alerts", tags=["monitoring"])
async def get_current_alerts():
    """Get current system alerts"""
    try:
        alerts = await monitoring_system.check_alerts()
        
        return {
            "status": "success",
            "data": {
                "alerts": alerts,
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
                "warning_alerts": len([a for a in alerts if a["severity"] == "warning"]),
                "timestamp": monitoring_system.metrics_storage.get("timestamp", None)
            },
            "message": "Current alerts retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

# Dashboard data endpoint
@router.get("/dashboard", tags=["monitoring"])
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        dashboard_data = await monitoring_system.get_metrics_dashboard_data()
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=500, detail=dashboard_data["error"])
        
        return {
            "status": "success",
            "data": dashboard_data,
            "message": "Dashboard data retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

# System health endpoint
@router.get("/health", tags=["monitoring"])
async def get_system_health():
    """Get overall system health status"""
    try:
        # Check various system components
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": monitoring_system.metrics_storage.get("timestamp", None)
        }
        
        # Check performance history
        if monitoring_system.metrics_storage["performance_history"]:
            health_status["components"]["performance_monitoring"] = "healthy"
        else:
            health_status["components"]["performance_monitoring"] = "degraded"
        
        # Check model performance
        if monitoring_system.metrics_storage["model_performance"]:
            health_status["components"]["model_monitoring"] = "healthy"
        else:
            health_status["components"]["model_monitoring"] = "degraded"
        
        # Check error logs
        recent_errors = [e for e in monitoring_system.metrics_storage["error_logs"] if 
                        (monitoring_system.metrics_storage.get("timestamp", datetime.now()) - e["timestamp"]).total_seconds() < 3600]
        
        if len(recent_errors) == 0:
            health_status["components"]["error_monitoring"] = "healthy"
        elif len(recent_errors) < 10:
            health_status["components"]["error_monitoring"] = "warning"
        else:
            health_status["components"]["error_monitoring"] = "critical"
        
        # Overall health
        component_statuses = list(health_status["components"].values())
        if "critical" in component_statuses:
            health_status["status"] = "critical"
        elif "warning" in component_statuses:
            health_status["status"] = "warning"
        elif "degraded" in component_statuses:
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "healthy"
        
        return {
            "status": "success",
            "data": health_status,
            "message": "System health status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

# Export router
__all__ = ["router"]
