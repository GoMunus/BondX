#!/usr/bin/env python3
"""
Test script for AI components
"""

import sys
import traceback
from pathlib import Path

# Add the bondx directory to the path
sys.path.insert(0, str(Path(__file__).parent / "bondx"))

def test_imports():
    """Test importing all AI components"""
    print("Testing AI component imports...")
    
    try:
        from bondx.ai_risk_engine import (
            RiskScoringEngine,
            YieldPredictionEngine,
            MLPipeline,
            NLPEngine,
            IntelligentAdvisorySystem,
            RealTimeAnalytics,
            ModelGovernance
        )
        print("✅ All AI components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_initialization():
    """Test initializing AI components"""
    print("\nTesting AI component initialization...")
    
    try:
        # Test Risk Scoring Engine
        print("Testing RiskScoringEngine...")
        risk_engine = RiskScoringEngine()
        print("✅ RiskScoringEngine initialized")
        
        # Test Yield Prediction Engine
        print("Testing YieldPredictionEngine...")
        yield_engine = YieldPredictionEngine()
        print("✅ YieldPredictionEngine initialized")
        
        # Test ML Pipeline
        print("Testing MLPipeline...")
        ml_pipeline = MLPipeline()
        print("✅ MLPipeline initialized")
        
        # Test NLP Engine
        print("Testing NLPEngine...")
        nlp_engine = NLPEngine()
        print("✅ NLPEngine initialized")
        
        # Test Advisory System
        print("Testing IntelligentAdvisorySystem...")
        advisory_system = IntelligentAdvisorySystem()
        print("✅ IntelligentAdvisorySystem initialized")
        
        # Test Real-time Analytics
        print("Testing RealTimeAnalytics...")
        realtime_analytics = RealTimeAnalytics()
        print("✅ RealTimeAnalytics initialized")
        
        # Test Model Governance
        print("Testing ModelGovernance...")
        model_governance = ModelGovernance()
        print("✅ ModelGovernance initialized")
        
        print("\n✅ All AI components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of AI components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test Risk Scoring
        risk_engine = RiskScoringEngine()
        credit_score = risk_engine.calculate_credit_risk_score(
            rating="AAA", 
            rating_agency=risk_engine.rating_mappings.keys().__iter__().__next__()
        )
        print(f"✅ Credit risk score calculated: {credit_score}")
        
        # Test ML Pipeline
        ml_pipeline = MLPipeline()
        summary = ml_pipeline.get_pipeline_summary()
        print(f"✅ ML Pipeline summary: {summary}")
        
        # Test Model Governance
        model_governance = ModelGovernance()
        status = model_governance.get_system_status()
        print(f"✅ Model Governance status: {status}")
        
        print("\n✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 BondX AI Components Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Exiting.")
        return False
    
    # Test initialization
    if not test_initialization():
        print("\n❌ Initialization tests failed. Exiting.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed. Exiting.")
        return False
    
    print("\n🎉 All tests passed! AI components are working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
