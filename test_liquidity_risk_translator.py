#!/usr/bin/env python3
"""
Test Liquidity-Risk Translator System for BondX

This script demonstrates the functionality of the Liquidity-Risk Translator system,
including liquidity intelligence, exit recommendations, and narrative generation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import the Liquidity-Risk Translator components
from bondx.ai_risk_engine.liquidity_intelligence_service import (
    LiquidityIntelligenceService,
    MarketMicrostructure,
    AuctionSignals,
    MarketMakerState
)
from bondx.ai_risk_engine.exit_recommender import (
    ExitRecommender,
    ExitPath
)
from bondx.ai_risk_engine.liquidity_risk_orchestrator import (
    LiquidityRiskOrchestrator,
    NarrativeMode
)

def create_sample_data():
    """Create sample data for testing the system."""
    
    # Sample market microstructure data
    microstructure = MarketMicrostructure(
        timestamp=datetime.now(),
        isin="IN0012345678",
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
    
    # Sample auction signals
    auction_signals = AuctionSignals(
        timestamp=datetime.now(),
        isin="IN0012345678",
        auction_id="AUCTION_IN0012345678_20241201",
        lots_offered=10,
        bids_count=8,
        demand_curve_points=[(100.0, 1000000), (100.25, 2000000), (100.5, 3000000)],
        clearing_price_estimate=100.25,
        next_window=datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
    )
    
    # Sample market maker state
    mm_state = MarketMakerState(
        timestamp=datetime.now(),
        isin="IN0012345678",
        mm_online=True,
        mm_inventory_band=(500000, 1000000, 2000000),
        mm_min_spread_bps=10.0,
        last_quote_spread_bps=25.0,
        quotes_last_24h=45
    )
    
    # Sample risk data
    risk_data = {
        'liquidity_risk_score': 35.0,
        'credit_risk_score': 45.0,
        'interest_rate_risk_score': 60.0,
        'refinancing_risk_score': 30.0,
        'leverage_risk_score': 40.0,
        'governance_risk_score': 25.0,
        'legal_risk_score': 20.0,
        'esg_risk_score': 35.0
    }
    
    # Sample bond metadata
    bond_metadata = {
        'rating': 'AA',
        'tenor': '5-10Y',
        'issuer_class': 'corporate',
        'coupon_rate': 7.5,
        'maturity_date': '2029-03-15',
        'issue_size': 500000000
    }
    
    return {
        'microstructure': microstructure,
        'auction_signals': auction_signals,
        'mm_state': mm_state,
        'risk_data': risk_data,
        'bond_metadata': bond_metadata
    }

def test_liquidity_intelligence():
    """Test the Liquidity Intelligence Service."""
    print("=" * 60)
    print("TESTING LIQUIDITY INTELLIGENCE SERVICE")
    print("=" * 60)
    
    service = LiquidityIntelligenceService()
    sample_data = create_sample_data()
    
    # Test liquidity index computation
    liquidity_index = service.compute_liquidity_index(
        sample_data['microstructure'],
        sample_data['auction_signals'],
        sample_data['mm_state']
    )
    print(f"Liquidity Index: {liquidity_index:.2f}/100")
    
    # Test liquidity profile creation
    profile = service.create_liquidity_profile(
        "IN0012345678",
        sample_data['microstructure'],
        sample_data['auction_signals'],
        sample_data['mm_state']
    )
    
    print(f"Liquidity Profile:")
    print(f"  - ISIN: {profile.isin}")
    print(f"  - Liquidity Index: {profile.liquidity_index:.2f}")
    print(f"  - Spread: {profile.spread_bps:.1f} bps")
    print(f"  - Depth Score: {profile.depth_score:.1f}/100")
    print(f"  - Liquidity Level: {profile.liquidity_level.value}")
    print(f"  - Expected TTE: {profile.expected_time_to_exit_minutes:.1f} minutes")
    print(f"  - Confidence: {profile.confidence:.1%}")
    
    # Test time-to-exit estimation for different paths
    paths = [ExitPath.MARKET_MAKER, ExitPath.AUCTION, ExitPath.RFQ_BATCH, ExitPath.TOKENIZED_P2P]
    market_conditions = {'market_open': True, 'volatility': 0.1}
    
    print(f"\nTime-to-Exit Estimates:")
    for path in paths:
        tte = service.estimate_time_to_exit(
            liquidity_index, path, 1000000, market_conditions
        )
        print(f"  - {path.value.replace('_', ' ').title()}: {tte:.1f} minutes")
    
    return profile

def test_exit_recommender():
    """Test the Exit Recommender Service."""
    print("\n" + "=" * 60)
    print("TESTING EXIT RECOMMENDER SERVICE")
    print("=" * 60)
    
    service = ExitRecommender()
    sample_data = create_sample_data()
    
    # Test exit path recommendations
    exit_analysis = service.recommend_exit_paths(
        "IN0012345678",
        sample_data['microstructure'],
        sample_data['auction_signals'],
        sample_data['mm_state'],
        sample_data['bond_metadata'],
        trade_size=1000000
    )
    
    print(f"Exit Analysis for IN0012345678:")
    print(f"  - Overall Confidence: {exit_analysis.overall_confidence:.1%}")
    print(f"  - Best Path: {exit_analysis.best_path.value if exit_analysis.best_path else 'None'}")
    print(f"  - Expected Best Exit Time: {exit_analysis.expected_best_exit_time:.1f} minutes")
    print(f"  - Risk Warnings: {len(exit_analysis.risk_warnings)}")
    
    print(f"\nExit Recommendations:")
    for i, rec in enumerate(exit_analysis.recommendations, 1):
        print(f"  {i}. {rec.path.value.replace('_', ' ').title()}")
        print(f"     Priority: {rec.priority.value}")
        print(f"     Fill Probability: {rec.fill_probability:.1%}")
        print(f"     Expected TTE: {rec.expected_time_to_exit_minutes:.1f} minutes")
        print(f"     Expected Spread: {rec.expected_spread_bps:.1f} bps")
        print(f"     Confidence: {rec.confidence:.1%}")
        if rec.constraints:
            constraints = [c.value.replace('_', ' ').title() for c in rec.constraints]
            print(f"     Constraints: {', '.join(constraints)}")
        print()
    
    if exit_analysis.risk_warnings:
        print(f"Risk Warnings:")
        for warning in exit_analysis.risk_warnings:
            print(f"  - {warning}")
    
    return exit_analysis

def test_orchestrator():
    """Test the Liquidity-Risk Orchestrator."""
    print("\n" + "=" * 60)
    print("TESTING LIQUIDITY-RISK ORCHESTRATOR")
    print("=" * 60)
    
    orchestrator = LiquidityRiskOrchestrator()
    sample_data = create_sample_data()
    
    # Test complete translation creation
    translation = orchestrator.create_liquidity_risk_translation(
        "IN0012345678",
        sample_data['microstructure'],
        sample_data['risk_data'],
        sample_data['auction_signals'],
        sample_data['mm_state'],
        sample_data['bond_metadata'],
        trade_size=1000000,
        mode=NarrativeMode.RETAIL
    )
    
    print(f"Liquidity-Risk Translation:")
    print(f"  - ISIN: {translation.isin}")
    print(f"  - As of: {translation.as_of}")
    print(f"  - Overall Confidence: {translation.confidence_overall:.1%}")
    print(f"  - Data Freshness: {translation.data_freshness}")
    print(f"  - Inputs Hash: {translation.inputs_hash}")
    
    print(f"\nRisk Summary:")
    print(f"  - Overall Score: {translation.risk_summary.overall_score:.1f}/100")
    print(f"  - Confidence: {translation.risk_summary.confidence:.1%}")
    print(f"  - Categories: {len(translation.risk_summary.categories)}")
    
    print(f"\nLiquidity Profile:")
    print(f"  - Liquidity Index: {translation.liquidity_profile.liquidity_index:.1f}/100")
    print(f"  - Spread: {translation.liquidity_profile.spread_bps:.1f} bps")
    print(f"  - Level: {translation.liquidity_profile.liquidity_level.value}")
    
    print(f"\nExit Recommendations: {len(translation.exit_recommendations)}")
    
    print(f"\nRetail Narrative:")
    print("-" * 40)
    print(translation.retail_narrative)
    print("-" * 40)
    
    print(f"\nProfessional Summary:")
    print("-" * 40)
    print(translation.professional_summary)
    print("-" * 40)
    
    if translation.risk_warnings:
        print(f"\nRisk Warnings:")
        for warning in translation.risk_warnings:
            print(f"  - {warning}")
    
    if translation.caveats:
        print(f"\nCaveats:")
        for caveat in translation.caveats:
            print(f"  - {caveat}")
    
    return translation

def test_api_endpoints():
    """Test the API endpoints (simulated)."""
    print("\n" + "=" * 60)
    print("TESTING API ENDPOINTS (SIMULATED)")
    print("=" * 60)
    
    # Simulate API calls
    print("GET /api/v1/liquidity-risk/IN0012345678?mode=fast&detail=summary")
    print("  Response: Summary liquidity-risk translation")
    
    print("\nGET /api/v1/liquidity-risk/IN0012345678?mode=accurate&detail=full")
    print("  Response: Full liquidity-risk translation with all details")
    
    print("\nPOST /api/v1/liquidity-risk/recompute")
    print("  Request: {'isins': ['IN0012345678'], 'mode': 'accurate'}")
    print("  Response: Recomputation queued successfully")
    
    print("\nGET /api/v1/liquidity-risk/audit/IN0012345678")
    print("  Response: Audit trail and data lineage")
    
    print("\nGET /api/v1/liquidity-risk/health")
    print("  Response: Service health status")

def test_websocket_events():
    """Test the WebSocket events (simulated)."""
    print("\n" + "=" * 60)
    print("TESTING WEBSOCKET EVENTS (SIMULATED)")
    print("=" * 60)
    
    print("WebSocket Topic: lr.IN0012345678")
    print("Event Types:")
    print("  - snapshot: Full liquidity-risk translation")
    print("  - risk_update: Risk category score changes")
    print("  - liquidity_update: Liquidity profile changes")
    print("  - exit_path_update: Exit pathway updates")
    print("  - alert: Risk or liquidity alerts")
    print("  - heartbeat: Keep-alive messages")
    
    print("\nSample WebSocket Message:")
    sample_message = {
        "type": "snapshot",
        "topic": "lr.IN0012345678",
        "seq": 1,
        "ts": datetime.now().isoformat(),
        "payload": {
            "isin": "IN0012345678",
            "risk_summary": {"overall_score": 42.5, "confidence": 0.85},
            "liquidity_profile": {"liquidity_index": 78.2, "spread_bps": 25.0},
            "exit_recommendations": [
                {"path": "market_maker", "fill_probability": 0.82, "tte_minutes": 15}
            ]
        },
        "meta": {"data_freshness": "real_time", "confidence": 0.85}
    }
    print(json.dumps(sample_message, indent=2))

def main():
    """Main test function."""
    print("LIQUIDITY-RISK TRANSLATOR SYSTEM TEST")
    print("=" * 60)
    print("This test demonstrates the complete functionality of the")
    print("Liquidity-Risk Translator system for BondX.")
    print()
    
    try:
        # Test individual components
        liquidity_profile = test_liquidity_intelligence()
        exit_analysis = test_exit_recommender()
        translation = test_orchestrator()
        
        # Test API and WebSocket
        test_api_endpoints()
        test_websocket_events()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        print(f"\nSummary:")
        print(f"  - Liquidity Index: {liquidity_profile.liquidity_index:.1f}/100 ({liquidity_profile.liquidity_level.value})")
        print(f"  - Risk Score: {translation.risk_summary.overall_score:.1f}/100")
        print(f"  - Best Exit Path: {exit_analysis.best_path.value if exit_analysis.best_path else 'None'}")
        print(f"  - Expected Exit Time: {exit_analysis.expected_best_exit_time:.1f} minutes")
        print(f"  - Overall Confidence: {translation.confidence_overall:.1%}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
