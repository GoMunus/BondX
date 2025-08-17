"""
Risk Management API endpoints for BondX.

This module provides comprehensive REST API endpoints for risk management operations,
including portfolio risk analysis, stress testing, and compliance monitoring.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ...database.base import get_db
from ...risk_management.portfolio_risk import PortfolioRiskManager, VaRMethod, StressTestType
from ...risk_management.risk_models import (
    RiskMetrics, Portfolio, PortfolioPosition, RiskLevel,
    StressTestScenario, StressTestResult, ComplianceRule, ComplianceCheck
)
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/risk", tags=["risk_management"])

# Global instances (in production, these would be properly managed)
portfolio_risk_manager: Optional[PortfolioRiskManager] = None


def get_portfolio_risk_manager(db: Session = Depends(get_db)) -> PortfolioRiskManager:
    """Get portfolio risk manager instance."""
    global portfolio_risk_manager
    if portfolio_risk_manager is None:
        portfolio_risk_manager = PortfolioRiskManager(db)
    return portfolio_risk_manager


# Portfolio Risk Analysis Endpoints

@router.post("/portfolios/{portfolio_id}/risk", response_model=Dict[str, Any])
async def calculate_portfolio_risk(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    method: VaRMethod = Body(VaRMethod.HISTORICAL_SIMULATION, description="VaR calculation method"),
    confidence_level: float = Body(0.95, ge=0.5, le=0.999, description="Confidence level for VaR"),
    time_horizon: int = Body(1, ge=1, le=30, description="Time horizon in days"),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    This endpoint calculates Value at Risk (VaR), volatility, duration, and other risk metrics.
    """
    try:
        logger.info(f"Calculating portfolio risk for {portfolio_id}")
        
        # Calculate risk metrics
        risk_metrics = await portfolio_risk_manager.calculate_portfolio_risk(
            portfolio_id, method, confidence_level, time_horizon
        )
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "calculation_time": risk_metrics.calculation_time.isoformat(),
            "method": method.value,
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon,
            "risk_metrics": {
                "var_95_1d": risk_metrics.var_95_1d,
                "var_99_1d": risk_metrics.var_99_1d,
                "var_95_10d": risk_metrics.var_95_10d,
                "var_99_10d": risk_metrics.var_99_10d,
                "cvar_95_1d": risk_metrics.cvar_95_1d,
                "cvar_99_1d": risk_metrics.cvar_99_1d,
                "portfolio_volatility": risk_metrics.portfolio_volatility,
                "portfolio_beta": risk_metrics.portfolio_beta,
                "modified_duration": risk_metrics.modified_duration,
                "effective_duration": risk_metrics.effective_duration,
                "convexity": risk_metrics.convexity,
                "yield_to_maturity": risk_metrics.yield_to_maturity,
                "liquidity_score": risk_metrics.liquidity_score,
                "concentration_risk": risk_metrics.concentration_risk,
                "sector_concentration": risk_metrics.sector_concentration,
                "issuer_concentration": risk_metrics.issuer_concentration,
                "leverage_ratio": risk_metrics.leverage_ratio,
                "gross_exposure": risk_metrics.gross_exposure,
                "net_exposure": risk_metrics.net_exposure,
                "max_drawdown": risk_metrics.max_drawdown,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio,
                "calmar_ratio": risk_metrics.calmar_ratio
            },
            "metadata": {
                "calculation_method": risk_metrics.calculation_method,
                "data_points": risk_metrics.data_points,
                "last_update": risk_metrics.last_update.isoformat()
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error calculating portfolio risk: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating portfolio risk for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolios/{portfolio_id}/risk", response_model=Dict[str, Any])
async def get_portfolio_risk(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Get current risk metrics for a portfolio.
    
    This endpoint returns the most recently calculated risk metrics.
    """
    try:
        # Get risk summary
        risk_summary = await portfolio_risk_manager.get_portfolio_risk_summary(portfolio_id)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "risk_summary": risk_summary
        }
        
    except ValueError as e:
        logger.error(f"Validation error getting portfolio risk: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting portfolio risk for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolios/{portfolio_id}/risk/decomposition", response_model=Dict[str, Any])
async def get_risk_decomposition(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Get risk decomposition for a portfolio.
    
    This endpoint returns risk attribution analysis showing how risk is distributed.
    """
    try:
        # Calculate risk decomposition
        decomposition = await portfolio_risk_manager.calculate_risk_decomposition(portfolio_id)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "calculation_time": decomposition.calculation_time.isoformat(),
            "risk_decomposition": {
                "total_risk": decomposition.total_risk,
                "systematic_risk": decomposition.systematic_risk,
                "idiosyncratic_risk": decomposition.idiosyncratic_risk,
                "position_contributions": decomposition.position_contributions,
                "factor_contributions": decomposition.factor_contributions,
                "sector_contributions": decomposition.sector_contributions,
                "duration_contributions": decomposition.duration_contributions,
                "credit_contributions": decomposition.credit_contributions
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error getting risk decomposition: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting risk decomposition for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Stress Testing Endpoints

@router.post("/portfolios/{portfolio_id}/stress-test", response_model=Dict[str, Any])
async def perform_stress_test(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    scenario: StressTestScenario = Body(..., description="Stress test scenario"),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Perform stress test on a portfolio.
    
    This endpoint executes a stress test scenario and returns the results.
    """
    try:
        logger.info(f"Performing stress test for portfolio {portfolio_id}")
        
        # Perform stress test
        result = await portfolio_risk_manager.perform_stress_test(portfolio_id, scenario)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.scenario_name,
                "scenario_type": scenario.scenario_type,
                "stress_level": scenario.stress_level
            },
            "stress_test_result": {
                "result_id": result.result_id,
                "execution_time": result.execution_time.isoformat(),
                "portfolio_value_before": float(result.portfolio_value_before),
                "portfolio_value_after": float(result.portfolio_value_after),
                "portfolio_value_change": float(result.portfolio_value_change),
                "portfolio_value_change_percent": result.portfolio_value_change_percent,
                "var_change": result.var_change,
                "duration_change": result.duration_change,
                "convexity_change": result.convexity_change,
                "is_passed": result.is_passed,
                "failure_reason": result.failure_reason,
                "recommendations": result.recommendations,
                "execution_duration_seconds": result.execution_duration_seconds,
                "data_points_used": result.data_points_used
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error performing stress test: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error performing stress test for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/portfolios/{portfolio_id}/stress-test/default", response_model=Dict[str, Any])
async def perform_default_stress_test(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    scenario_type: StressTestType = Body(..., description="Type of stress test scenario"),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Perform default stress test on a portfolio.
    
    This endpoint uses predefined stress test scenarios.
    """
    try:
        logger.info(f"Performing default stress test for portfolio {portfolio_id}")
        
        # Get default scenario
        if scenario_type not in portfolio_risk_manager.default_scenarios:
            raise HTTPException(status_code=400, detail=f"Default scenario not found for type {scenario_type}")
        
        scenario = portfolio_risk_manager.default_scenarios[scenario_type]
        
        # Perform stress test
        result = await portfolio_risk_manager.perform_stress_test(portfolio_id, scenario)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "scenario_type": scenario_type.value,
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.scenario_name,
                "description": scenario.description,
                "stress_level": scenario.stress_level,
                "parameters": scenario.parameters
            },
            "stress_test_result": {
                "result_id": result.result_id,
                "execution_time": result.execution_time.isoformat(),
                "portfolio_value_change_percent": result.portfolio_value_change_percent,
                "is_passed": result.is_passed,
                "failure_reason": result.failure_reason,
                "execution_duration_seconds": result.execution_duration_seconds
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error performing default stress test: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error performing default stress test for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolios/{portfolio_id}/stress-test/results", response_model=List[Dict[str, Any]])
async def get_stress_test_results(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Get stress test results for a portfolio.
    
    This endpoint returns a paginated list of stress test results.
    """
    try:
        # Get recent stress test results
        stress_results = await portfolio_risk_manager._get_recent_stress_test_results(portfolio_id)
        
        # Apply pagination
        total_count = len(stress_results)
        paginated_results = stress_results[offset:offset + limit]
        
        # Format results
        result_list = []
        for result in paginated_results:
            result_data = {
                "result_id": result.result_id,
                "scenario_id": result.scenario_id,
                "execution_time": result.execution_time.isoformat(),
                "portfolio_value_change_percent": result.portfolio_value_change_percent,
                "is_passed": result.is_passed,
                "failure_reason": result.failure_reason,
                "execution_duration_seconds": result.execution_duration_seconds,
                "data_points_used": result.data_points_used
            }
            result_list.append(result_data)
        
        return {
            "stress_test_results": result_list,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stress test results for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Portfolio Management Endpoints

@router.post("/portfolios", response_model=Dict[str, Any])
async def create_portfolio(
    portfolio_data: Dict[str, Any] = Body(...),
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Create a new portfolio for risk management.
    
    This endpoint creates a portfolio with risk profile settings.
    """
    try:
        logger.info(f"Creating portfolio: {portfolio_data.get('portfolio_name', 'Unknown')}")
        
        # Validate required fields
        required_fields = ['portfolio_name', 'participant_id', 'portfolio_type']
        for field in required_fields:
            if field not in portfolio_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create portfolio (this would typically save to database)
        portfolio = Portfolio(
            portfolio_id=portfolio_data.get('portfolio_id', f"PORT_{datetime.utcnow().timestamp()}"),
            portfolio_name=portfolio_data['portfolio_name'],
            participant_id=portfolio_data['participant_id'],
            portfolio_type=portfolio_data['portfolio_type'],
            risk_tolerance=RiskLevel(portfolio_data.get('risk_tolerance', 'MEDIUM')),
            investment_horizon=portfolio_data.get('investment_horizon', 'MEDIUM_TERM'),
            target_return=portfolio_data.get('target_return'),
            max_drawdown=portfolio_data.get('max_drawdown'),
            description=portfolio_data.get('description')
        )
        
        return {
            "success": True,
            "message": "Portfolio created successfully",
            "portfolio": {
                "portfolio_id": portfolio.portfolio_id,
                "portfolio_name": portfolio.portfolio_name,
                "participant_id": portfolio.participant_id,
                "portfolio_type": portfolio.portfolio_type,
                "risk_tolerance": portfolio.risk_tolerance.value,
                "investment_horizon": portfolio.investment_horizon,
                "target_return": portfolio.target_return,
                "max_drawdown": portfolio.max_drawdown,
                "created_at": portfolio.created_at.isoformat()
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error creating portfolio: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolios", response_model=List[Dict[str, Any]])
async def get_portfolios(
    participant_id: Optional[int] = Query(None, description="Filter by participant ID"),
    portfolio_type: Optional[str] = Query(None, description="Filter by portfolio type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of portfolios to return"),
    offset: int = Query(0, ge=0, description="Number of portfolios to skip")
):
    """
    Get list of portfolios with optional filtering.
    
    This endpoint returns a paginated list of portfolios.
    """
    try:
        # This would typically query the database
        # For now, return mock data
        portfolios = [
            {
                "portfolio_id": "PORT_001",
                "portfolio_name": "Trading Portfolio 1",
                "participant_id": 1,
                "portfolio_type": "TRADING",
                "total_value": 1000000,
                "risk_tolerance": "MEDIUM",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "portfolio_id": "PORT_002",
                "portfolio_name": "Investment Portfolio 1",
                "participant_id": 1,
                "portfolio_type": "INVESTMENT",
                "total_value": 5000000,
                "risk_tolerance": "LOW",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        # Apply filters
        if participant_id:
            portfolios = [p for p in portfolios if p['participant_id'] == participant_id]
        if portfolio_type:
            portfolios = [p for p in portfolios if p['portfolio_type'] == portfolio_type]
        
        # Apply pagination
        total_count = len(portfolios)
        paginated_portfolios = portfolios[offset:offset + limit]
        
        return {
            "portfolios": paginated_portfolios,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolios: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolios/{portfolio_id}", response_model=Dict[str, Any])
async def get_portfolio(
    portfolio_id: str = Path(..., description="ID of the portfolio")
):
    """
    Get detailed information about a specific portfolio.
    
    This endpoint returns comprehensive portfolio information.
    """
    try:
        # This would typically query the database
        # For now, return mock data
        portfolio = {
            "portfolio_id": portfolio_id,
            "portfolio_name": f"Portfolio {portfolio_id}",
            "participant_id": 1,
            "portfolio_type": "TRADING",
            "total_value": 1000000,
            "total_face_value": 1050000,
            "total_cost_basis": 980000,
            "risk_tolerance": "MEDIUM",
            "investment_horizon": "MEDIUM_TERM",
            "target_return": 8.5,
            "max_drawdown": 15.0,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error getting portfolio {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Risk Limits and Monitoring Endpoints

@router.post("/portfolios/{portfolio_id}/limits", response_model=Dict[str, Any])
async def create_risk_limit(
    portfolio_id: str = Path(..., description="ID of the portfolio"),
    limit_data: Dict[str, Any] = Body(...)
):
    """
    Create a new risk limit for a portfolio.
    
    This endpoint creates risk limits for monitoring and control.
    """
    try:
        logger.info(f"Creating risk limit for portfolio {portfolio_id}")
        
        # Validate required fields
        required_fields = ['limit_type', 'limit_name', 'limit_value']
        for field in required_fields:
            if field not in limit_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create risk limit (this would typically save to database)
        risk_limit = {
            "limit_id": f"LIMIT_{datetime.utcnow().timestamp()}",
            "portfolio_id": portfolio_id,
            "limit_type": limit_data['limit_type'],
            "limit_name": limit_data['limit_name'],
            "limit_value": limit_data['limit_value'],
            "limit_currency": limit_data.get('limit_currency', 'INR'),
            "limit_unit": limit_data.get('limit_unit', 'ABSOLUTE'),
            "warning_threshold": limit_data.get('warning_threshold', 0.8),
            "critical_threshold": limit_data.get('critical_threshold', 0.95),
            "is_active": True,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "message": "Risk limit created successfully",
            "risk_limit": risk_limit
        }
        
    except ValueError as e:
        logger.error(f"Validation error creating risk limit: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating risk limit: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolios/{portfolio_id}/limits", response_model=List[Dict[str, Any]])
async def get_risk_limits(
    portfolio_id: str = Path(..., description="ID of the portfolio")
):
    """
    Get risk limits for a portfolio.
    
    This endpoint returns all risk limits configured for a portfolio.
    """
    try:
        # This would typically query the database
        # For now, return mock data
        risk_limits = [
            {
                "limit_id": "LIMIT_001",
                "portfolio_id": portfolio_id,
                "limit_type": "VAR_LIMIT",
                "limit_name": "Daily VaR Limit",
                "limit_value": 100000,
                "limit_currency": "INR",
                "limit_unit": "ABSOLUTE",
                "warning_threshold": 0.8,
                "critical_threshold": 0.95,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "limit_id": "LIMIT_002",
                "portfolio_id": portfolio_id,
                "limit_type": "CONCENTRATION_LIMIT",
                "limit_name": "Sector Concentration Limit",
                "limit_value": 0.25,
                "limit_currency": "INR",
                "limit_unit": "PERCENTAGE",
                "warning_threshold": 0.8,
                "critical_threshold": 0.95,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return risk_limits
        
    except Exception as e:
        logger.error(f"Error getting risk limits for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# System Status and Health Endpoints

@router.get("/status", response_model=Dict[str, Any])
async def get_risk_management_status(
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Get risk management system status.
    
    This endpoint returns comprehensive system status and health information.
    """
    try:
        # Get system stats
        system_stats = portfolio_risk_manager.get_system_stats()
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "system_stats": system_stats,
            "components": {
                "portfolio_risk_manager": "available",
                "stress_testing": "available",
                "risk_decomposition": "available"
            },
            "performance_metrics": {
                "total_calculations": system_stats["total_calculations"],
                "average_calculation_time": system_stats["average_calculation_time"],
                "default_scenarios_count": system_stats["default_scenarios_count"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting risk management status: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check for risk management system.
    
    This endpoint provides a basic health check for the risk management components.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "portfolio_risk_manager": "available",
                "stress_testing": "available",
                "risk_decomposition": "available"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# Risk Analytics and Reporting Endpoints

@router.get("/analytics/var-methods", response_model=List[Dict[str, Any]])
async def get_var_methods():
    """
    Get available VaR calculation methods.
    
    This endpoint returns information about different VaR calculation approaches.
    """
    try:
        var_methods = [
            {
                "method": "HISTORICAL_SIMULATION",
                "name": "Historical Simulation",
                "description": "Uses historical price data to simulate portfolio returns",
                "advantages": ["No distribution assumptions", "Captures fat tails", "Easy to understand"],
                "disadvantages": ["Requires large historical dataset", "May not capture regime changes"],
                "suitable_for": ["Portfolios with sufficient historical data", "Non-normal return distributions"]
            },
            {
                "method": "PARAMETRIC",
                "name": "Parametric (Variance-Covariance)",
                "description": "Assumes normal distribution and uses portfolio variance-covariance matrix",
                "advantages": ["Fast computation", "Easy to implement", "Good for large portfolios"],
                "disadvantages": ["Assumes normal distribution", "May underestimate tail risk"],
                "suitable_for": ["Portfolios with normal returns", "Quick risk assessments"]
            },
            {
                "method": "MONTE_CARLO",
                "name": "Monte Carlo Simulation",
                "description": "Generates random scenarios based on statistical models",
                "advantages": ["Flexible modeling", "Can capture complex dependencies", "No distribution assumptions"],
                "disadvantages": ["Computationally intensive", "Model risk", "Requires good models"],
                "suitable_for": ["Complex portfolios", "Custom risk scenarios", "Stress testing"]
            }
        ]
        
        return var_methods
        
    except Exception as e:
        logger.error(f"Error getting VaR methods: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/stress-test-scenarios", response_model=List[Dict[str, Any]])
async def get_stress_test_scenarios(
    portfolio_risk_manager: PortfolioRiskManager = Depends(get_portfolio_risk_manager)
):
    """
    Get available stress test scenarios.
    
    This endpoint returns information about predefined stress test scenarios.
    """
    try:
        scenarios = []
        for scenario_type, scenario in portfolio_risk_manager.default_scenarios.items():
            scenario_info = {
                "scenario_type": scenario_type.value,
                "scenario_name": scenario.scenario_name,
                "description": scenario.description,
                "stress_level": scenario.stress_level,
                "parameters": scenario.parameters,
                "execution_frequency": scenario.execution_frequency,
                "is_active": scenario.is_active
            }
            scenarios.append(scenario_info)
        
        return scenarios
        
    except Exception as e:
        logger.error(f"Error getting stress test scenarios: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/risk-metrics", response_model=Dict[str, Any])
async def get_risk_metrics_info():
    """
    Get information about risk metrics.
    
    This endpoint returns detailed information about various risk metrics and their interpretation.
    """
    try:
        risk_metrics_info = {
            "var": {
                "name": "Value at Risk (VaR)",
                "description": "Maximum expected loss over a given time horizon at a specified confidence level",
                "interpretation": "Lower VaR indicates lower risk",
                "units": "Currency (e.g., INR)",
                "calculation_methods": ["Historical Simulation", "Parametric", "Monte Carlo"]
            },
            "volatility": {
                "name": "Portfolio Volatility",
                "description": "Standard deviation of portfolio returns",
                "interpretation": "Higher volatility indicates higher risk",
                "units": "Percentage per time period",
                "calculation_methods": ["Historical", "Implied", "GARCH"]
            },
            "duration": {
                "name": "Modified Duration",
                "description": "Sensitivity of bond price to changes in yield",
                "interpretation": "Higher duration means higher interest rate risk",
                "units": "Years",
                "calculation_methods": ["Analytical", "Numerical"]
            },
            "convexity": {
                "name": "Convexity",
                "description": "Second-order sensitivity of bond price to yield changes",
                "interpretation": "Higher convexity provides better price protection",
                "units": "Years squared",
                "calculation_methods": ["Analytical", "Numerical"]
            },
            "concentration": {
                "name": "Concentration Risk",
                "description": "Risk from over-concentration in specific assets or sectors",
                "interpretation": "Lower concentration indicates better diversification",
                "units": "Herfindahl Index (0-1)",
                "calculation_methods": ["Position-based", "Sector-based", "Issuer-based"]
            }
        }
        
        return risk_metrics_info
        
    except Exception as e:
        logger.error(f"Error getting risk metrics info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
