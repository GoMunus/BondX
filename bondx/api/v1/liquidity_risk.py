"""
Liquidity-Risk Translator API for BondX

This module provides REST API endpoints for the Liquidity-Risk Translator service,
delivering integrated risk assessment and liquidity intelligence with exit pathway recommendations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from ...ai_risk_engine.liquidity_risk_orchestrator import (
    LiquidityRiskOrchestrator, LiquidityRiskTranslation, NarrativeMode
)
from ...ai_risk_engine.liquidity_intelligence_service import (
    MarketMicrostructure, AuctionSignals, MarketMakerState
)
from ..schemas import BaseResponse, ErrorResponse
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/liquidity-risk", tags=["Liquidity-Risk Translator"])

# Initialize the orchestrator
orchestrator = LiquidityRiskOrchestrator()

# Mock data for demonstration - in production, this would come from real data sources
def get_mock_microstructure(isin: str) -> MarketMicrostructure:
    """Get mock market microstructure data for testing."""
    from ...ai_risk_engine.liquidity_intelligence_service import MarketMicrostructure
    
    return MarketMicrostructure(
        timestamp=datetime.now(),
        isin=isin,
        bid=100.50,
        ask=100.75,
        bid_size=1000000,
        ask_size=1000000,
        l2_depth_qty=5000000,
        l2_levels=5,
        trades_count=25,
        vwap=100.625,
        volume_face=10000000,
        time_since_last_trade_s=300
    )

def get_mock_auction_signals(isin: str) -> AuctionSignals:
    """Get mock auction signals for testing."""
    from ...ai_risk_engine.liquidity_intelligence_service import AuctionSignals
    
    return AuctionSignals(
        timestamp=datetime.now(),
        isin=isin,
        auction_id=f"AUCTION_{isin}_{datetime.now().strftime('%Y%m%d')}",
        lots_offered=10,
        bids_count=8,
        demand_curve_points=[(100.0, 1000000), (100.25, 2000000), (100.5, 3000000)],
        clearing_price_estimate=100.25,
        next_window=datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
    )

def get_mock_mm_state(isin: str) -> MarketMakerState:
    """Get mock market maker state for testing."""
    from ...ai_risk_engine.liquidity_intelligence_service import MarketMakerState
    
    return MarketMakerState(
        timestamp=datetime.now(),
        isin=isin,
        mm_online=True,
        mm_inventory_band=(500000, 1000000, 2000000),
        mm_min_spread_bps=10.0,
        last_quote_spread_bps=25.0,
        quotes_last_24h=45
    )

def get_mock_risk_data(isin: str) -> Dict[str, Any]:
    """Get mock risk data for testing."""
    return {
        'liquidity_risk_score': 35.0,
        'credit_risk_score': 45.0,
        'interest_rate_risk_score': 60.0,
        'refinancing_risk_score': 30.0,
        'leverage_risk_score': 40.0,
        'governance_risk_score': 25.0,
        'legal_risk_score': 20.0,
        'esg_risk_score': 35.0
    }

def get_mock_bond_metadata(isin: str) -> Dict[str, Any]:
    """Get mock bond metadata for testing."""
    return {
        'rating': 'AA',
        'tenor': '5-10Y',
        'issuer_class': 'corporate',
        'coupon_rate': 7.5,
        'maturity_date': '2029-03-15',
        'issue_size': 500000000
    }

@router.get("/{isin}", response_model=Dict[str, Any])
async def get_liquidity_risk_translation(
    isin: str,
    mode: str = Query("fast", description="Mode: 'fast' for cached, 'accurate' for recompute"),
    detail: str = Query("summary", description="Detail level: 'summary' or 'full'"),
    trade_size: float = Query(100000, description="Trade size in currency units")
):
    """
    Get comprehensive liquidity-risk translation for a bond.
    
    Args:
        isin: Bond ISIN identifier
        mode: 'fast' for cached results, 'accurate' for recompute
        detail: 'summary' or 'full' detail level
        trade_size: Size of position to exit
        
    Returns:
        Integrated liquidity-risk analysis with exit recommendations
    """
    try:
        logger.info(f"Processing liquidity-risk translation request for {isin}")
        
        # Get market data (mock for now)
        microstructure = get_mock_microstructure(isin)
        auction_signals = get_mock_auction_signals(isin)
        mm_state = get_mock_mm_state(isin)
        risk_data = get_mock_risk_data(isin)
        bond_metadata = get_mock_bond_metadata(isin)
        
        # Create translation
        translation = orchestrator.create_liquidity_risk_translation(
            isin=isin,
            microstructure=microstructure,
            risk_data=risk_data,
            auction_signals=auction_signals,
            mm_state=mm_state,
            bond_metadata=bond_metadata,
            trade_size=trade_size,
            mode=NarrativeMode.RETAIL
        )
        
        # Format response based on detail level
        if detail == "summary":
            response = {
                "isin": translation.isin,
                "as_of": translation.as_of.isoformat(),
                "risk_summary": {
                    "overall_score": translation.risk_summary.overall_score,
                    "confidence": translation.risk_summary.confidence
                },
                "liquidity_profile": {
                    "liquidity_index": translation.liquidity_profile.liquidity_index,
                    "spread_bps": translation.liquidity_profile.spread_bps,
                    "liquidity_level": translation.liquidity_profile.liquidity_level.value,
                    "expected_time_to_exit_minutes": translation.liquidity_profile.expected_time_to_exit_minutes
                },
                "exit_recommendations": [
                    {
                        "path": rec.path.value,
                        "priority": rec.priority.value,
                        "fill_probability": rec.fill_probability,
                        "expected_time_to_exit_minutes": rec.expected_time_to_exit_minutes,
                        "expected_spread_bps": rec.expected_spread_bps
                    }
                    for rec in translation.exit_recommendations[:2]  # Top 2 only
                ],
                "retail_narrative": translation.retail_narrative,
                "confidence_overall": translation.confidence_overall,
                "data_freshness": translation.data_freshness
            }
        else:  # full detail
            response = {
                "isin": translation.isin,
                "as_of": translation.as_of.isoformat(),
                "risk_summary": {
                    "overall_score": translation.risk_summary.overall_score,
                    "categories": [
                        {
                            "name": cat.name,
                            "score_0_100": cat.score_0_100,
                            "level": cat.level,
                            "probability_note": cat.probability_note,
                            "citations": cat.citations,
                            "confidence": cat.confidence
                        }
                        for cat in translation.risk_summary.categories
                    ],
                    "confidence": translation.risk_summary.confidence,
                    "citations": translation.risk_summary.citations,
                    "last_updated": translation.risk_summary.last_updated.isoformat(),
                    "methodology_version": translation.risk_summary.methodology_version
                },
                "liquidity_profile": {
                    "liquidity_index": translation.liquidity_profile.liquidity_index,
                    "spread_bps": translation.liquidity_profile.spread_bps,
                    "depth_score": translation.liquidity_profile.depth_score,
                    "turnover_rank": translation.liquidity_profile.turnover_rank,
                    "time_since_last_trade_s": translation.liquidity_profile.time_since_last_trade_s,
                    "expected_time_to_exit_minutes": translation.liquidity_profile.expected_time_to_exit_minutes,
                    "liquidity_level": translation.liquidity_profile.liquidity_level.value,
                    "confidence": translation.liquidity_profile.confidence,
                    "data_freshness": translation.liquidity_profile.data_freshness,
                    "metadata": translation.liquidity_profile.metadata
                },
                "exit_recommendations": [
                    {
                        "path": rec.path.value,
                        "priority": rec.priority.value,
                        "expected_price": rec.expected_price,
                        "expected_spread_bps": rec.expected_spread_bps,
                        "fill_probability": rec.fill_probability,
                        "expected_time_to_exit_minutes": rec.expected_time_to_exit_minutes,
                        "rationale": rec.rationale,
                        "constraints": [c.value for c in rec.constraints],
                        "confidence": rec.confidence,
                        "metadata": rec.metadata
                    }
                    for rec in translation.exit_recommendations
                ],
                "retail_narrative": translation.retail_narrative,
                "professional_summary": translation.professional_summary,
                "risk_warnings": translation.risk_warnings,
                "confidence_overall": translation.confidence_overall,
                "data_freshness": translation.data_freshness,
                "inputs_hash": translation.inputs_hash,
                "model_versions": translation.model_versions,
                "caveats": translation.caveats
            }
        
        logger.info(f"Successfully generated liquidity-risk translation for {isin}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating liquidity-risk translation for {isin}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate liquidity-risk translation: {str(e)}"
        )

@router.post("/recompute")
async def recompute_liquidity_risk(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Trigger recomputation of liquidity-risk analysis for specified bonds.
    
    Args:
        request: Dictionary containing 'isins' list and 'mode' string
        
    Returns:
        Confirmation of recomputation request
    """
    try:
        isins = request.get('isins', [])
        mode = request.get('mode', 'accurate')
        
        if not isins:
            raise HTTPException(status_code=400, detail="No ISINs provided")
        
        logger.info(f"Recomputation requested for {len(isins)} bonds: {isins}")
        
        # In a real implementation, this would queue background tasks
        # For now, we'll just log the request
        background_tasks.add_task(
            _process_recomputation_request,
            isins,
            mode
        )
        
        return {
            "success": True,
            "message": f"Recomputation queued for {len(isins)} bonds",
            "timestamp": datetime.now().isoformat(),
            "requested_isins": isins,
            "mode": mode
        }
        
    except Exception as e:
        logger.error(f"Error processing recomputation request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process recomputation request: {str(e)}"
        )

async def _process_recomputation_request(isins: List[str], mode: str):
    """Background task to process recomputation requests."""
    logger.info(f"Processing recomputation for {len(isins)} bonds in {mode} mode")
    
    # This would typically:
    # 1. Fetch fresh data for each ISIN
    # 2. Recompute all analyses
    # 3. Update cache/database
    # 4. Trigger WebSocket notifications
    
    for isin in isins:
        try:
            logger.info(f"Recomputing analysis for {isin}")
            # Placeholder for actual recomputation logic
            await asyncio.sleep(0.1)  # Simulate processing time
            
        except Exception as e:
            logger.error(f"Error recomputing {isin}: {e}")

@router.get("/audit/{isin}")
async def get_liquidity_risk_audit(
    isin: str,
    as_of: Optional[str] = Query(None, description="Audit timestamp (ISO format)")
):
    """
    Get audit trail for liquidity-risk analysis of a bond.
    
    Args:
        isin: Bond ISIN identifier
        as_of: Optional timestamp for historical audit
        
    Returns:
        Audit trail and data lineage information
    """
    try:
        logger.info(f"Audit request for {isin} as_of={as_of}")
        
        # In a real implementation, this would fetch from audit logs
        # For now, return mock audit data
        audit_data = {
            "isin": isin,
            "audit_timestamp": datetime.now().isoformat(),
            "requested_as_of": as_of,
            "data_lineage": {
                "risk_data_source": "Risk Scoring Engine v1.0",
                "liquidity_data_source": "Market Data Feed",
                "auction_data_source": "Auction Engine",
                "mm_data_source": "Market Maker Telemetry",
                "last_updated": datetime.now().isoformat()
            },
            "model_versions": {
                "liquidity_intelligence": "1.0.0",
                "exit_recommender": "1.0.0",
                "risk_scoring": "1.0.0",
                "orchestrator": "1.0.0"
            },
            "data_quality_metrics": {
                "risk_data_freshness": "real_time",
                "market_data_freshness": "real_time",
                "auction_data_freshness": "fresh",
                "mm_data_freshness": "real_time"
            },
            "computation_metadata": {
                "processing_time_ms": 45,
                "cache_hit": False,
                "data_sources_used": 4,
                "confidence_threshold_met": True
            }
        }
        
        return audit_data
        
    except Exception as e:
        logger.error(f"Error generating audit for {isin}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate audit: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for the Liquidity-Risk Translator service."""
    try:
        # Check service health
        health_status = {
            "service": "Liquidity-Risk Translator",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "liquidity_intelligence": "operational",
                "exit_recommender": "operational",
                "orchestrator": "operational"
            },
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "service": "Liquidity-Risk Translator",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Import asyncio for background tasks
import asyncio
