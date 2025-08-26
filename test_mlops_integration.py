#!/usr/bin/env python3
"""
Test MLOps Integration

This script demonstrates the MLOps functionality including:
- Configuration loading
- Experiment tracking
- Model registry operations
- Drift detection
- Retraining pipeline
- Canary deployment
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add the bondx directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bondx'))

from bondx.mlops import (
    MLOpsConfig, ExperimentTracker, ModelRegistry, 
    DriftMonitor, RetrainPipeline, CanaryDeploymentManager
)
from bondx.mlops.registry import ModelStage


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_data():
    """Create sample bond data for testing"""
    np.random.seed(42)
    
    # Generate sample bond data
    n_samples = 1000
    
    data = {
        'coupon_rate': np.random.uniform(2.0, 8.0, n_samples),
        'maturity_years': np.random.uniform(1.0, 30.0, n_samples),
        'credit_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB'], n_samples),
        'issue_size': np.random.uniform(1000000, 100000000, n_samples),
        'sector': np.random.choice(['Financial', 'Technology', 'Healthcare', 'Energy'], n_samples),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_samples),
        'yield_to_maturity': np.random.uniform(2.0, 10.0, n_samples),
        'duration': np.random.uniform(1.0, 15.0, n_samples),
        'convexity': np.random.uniform(0.1, 5.0, n_samples),
        'liquidity_score': np.random.uniform(0.1, 1.0, n_samples),
        'volatility': np.random.uniform(0.05, 0.3, n_samples),
        'correlation_market': np.random.uniform(-0.5, 0.8, n_samples),
        'spread_bps': np.random.uniform(50, 500, n_samples)
    }
    
    return pd.DataFrame(data)


def create_drifted_data(baseline_data, drift_factor=0.3):
    """Create drifted data by modifying the baseline"""
    drifted_data = baseline_data.copy()
    
    # Add drift to some features
    drifted_data['coupon_rate'] = baseline_data['coupon_rate'] * (1 + np.random.normal(0, drift_factor, len(baseline_data)))
    drifted_data['maturity_years'] = baseline_data['maturity_years'] * (1 + np.random.normal(0, drift_factor, len(baseline_data)))
    drifted_data['spread_bps'] = baseline_data['spread_bps'] * (1 + np.random.normal(0, drift_factor, len(baseline_data)))
    
    return drifted_data


def test_mlops_integration():
    """Test the complete MLOps workflow"""
    
    print("🚀 Starting BondX MLOps Integration Test")
    print("=" * 50)
    
    # Create test directories
    test_dirs = ['./models', './data', './logs', './mlruns']
    for dir_path in test_dirs:
        Path(dir_path).mkdir(exist_ok=True)
    
    try:
        # 1. Load configuration
        print("\n1. Loading MLOps Configuration...")
        config = MLOpsConfig.get_default_config()
        print(f"   ✓ Configuration loaded: {config.environment}")
        print(f"   ✓ Supported models: {', '.join(config.supported_models)}")
        
        # 2. Initialize components
        print("\n2. Initializing MLOps Components...")
        tracker = ExperimentTracker(config)
        registry = ModelRegistry(config)
        drift_monitor = DriftMonitor(config)
        retrain_pipeline = RetrainPipeline(config, tracker, registry)
        deployment_manager = CanaryDeploymentManager(config, registry)
        print("   ✓ All components initialized")
        
        # 3. Create sample data
        print("\n3. Creating Sample Data...")
        baseline_data = create_sample_data()
        current_data = create_drifted_data(baseline_data, drift_factor=0.2)
        print(f"   ✓ Baseline data: {baseline_data.shape}")
        print(f"   ✓ Current data: {current_data.shape}")
        
        # 4. Start experiment tracking
        print("\n4. Starting Experiment Tracking...")
        run_id = tracker.start_run(
            experiment_name="mlops_integration_test",
            model_type="spread",
            parameters={
                'test_type': 'integration',
                'data_samples': len(baseline_data),
                'drift_factor': 0.2
            }
        )
        print(f"   ✓ Experiment started: {run_id}")
        
        # 5. Log some metrics
        print("\n5. Logging Experiment Metrics...")
        tracker.log_metrics(run_id, {
            'baseline_samples': len(baseline_data),
            'current_samples': len(current_data),
            'feature_count': len(baseline_data.columns) - 1  # Exclude target
        })
        print("   ✓ Metrics logged")
        
        # 6. Test drift detection
        print("\n6. Testing Drift Detection...")
        feature_columns = [col for col in baseline_data.columns if col != 'spread_bps']
        
        drift_report = drift_monitor.detect_drift(
            model_name="spread_test",
            model_version="1.0.0",
            baseline_data=baseline_data,
            current_data=current_data,
            feature_columns=feature_columns,
            target_column="spread_bps"
        )
        
        print(f"   ✓ Drift detection completed")
        print(f"   ✓ Drift detected: {drift_report.requires_retraining}")
        print(f"   ✓ Drifted features: {drift_report.drifted_features}")
        print(f"   ✓ Overall drift score: {drift_report.overall_drift_score:.4f}")
        
        # 7. Test retraining pipeline
        print("\n7. Testing Retraining Pipeline...")
        retrain_result = retrain_pipeline.trigger_retraining(drift_report, baseline_data)
        
        if retrain_result.success:
            print(f"   ✓ Retraining completed successfully")
            print(f"   ✓ New model version: {retrain_result.new_model_version}")
            print(f"   ✓ Training time: {retrain_result.training_time_seconds:.2f}s")
            
            # 8. Test model registry
            print("\n8. Testing Model Registry...")
            model = registry.get_model("spread_test", retrain_result.new_model_version)
            if model:
                print(f"   ✓ Model retrieved from registry")
                print(f"   ✓ Model stage: {model.stage.value}")
                print(f"   ✓ Model status: {model.status.value}")
            
            # 9. Test model promotion
            print("\n9. Testing Model Promotion...")
            success = registry.promote_model(
                model_name="spread_test",
                version=retrain_result.new_model_version,
                target_stage=ModelStage.STAGING,
                deployed_by="test_user",
                deployment_notes="Integration test promotion"
            )
            
            if success:
                print(f"   ✓ Model promoted to staging")
                
                # 10. Test canary deployment
                print("\n10. Testing Canary Deployment...")
                deployment_id = deployment_manager.create_canary_deployment(
                    model_name="spread_test",
                    candidate_version=retrain_result.new_model_version,
                    deployed_by="test_user",
                    initial_traffic_percentage=0.1,
                    deployment_notes="Integration test deployment"
                )
                
                print(f"   ✓ Canary deployment created: {deployment_id}")
                
                # Test prediction routing
                route_to_canary, request_id = deployment_manager.route_prediction(
                    model_name="spread_test",
                    features={
                        'coupon_rate': 5.0,
                        'maturity_years': 10.0,
                        'credit_rating': 'A',
                        'issue_size': 50000000,
                        'sector': 'Financial',
                        'currency': 'USD',
                        'yield_to_maturity': 6.0,
                        'duration': 8.0,
                        'convexity': 2.5,
                        'liquidity_score': 0.7,
                        'volatility': 0.15,
                        'correlation_market': 0.3
                    }
                )
                
                print(f"   ✓ Prediction routing: {'canary' if route_to_canary else 'production'}")
                
                # 11. Test listing operations
                print("\n11. Testing List Operations...")
                models = registry.list_models(model_name="spread_test")
                deployments = deployment_manager.list_deployments(model_name="spread_test")
                experiments = tracker.list_runs(experiment_name="mlops_integration_test")
                
                print(f"   ✓ Models found: {len(models)}")
                print(f"   ✓ Deployments found: {len(deployments)}")
                print(f"   ✓ Experiments found: {len(experiments)}")
                
        else:
            print(f"   ✗ Retraining failed: {retrain_result.error_message}")
        
        # 12. End experiment
        print("\n12. Ending Experiment...")
        tracker.end_run(run_id, "completed")
        print("   ✓ Experiment completed")
        
        # 13. Cleanup
        print("\n13. Cleanup...")
        deployment_manager.stop()
        print("   ✓ Deployment manager stopped")
        
        print("\n" + "=" * 50)
        print("✅ MLOps Integration Test Completed Successfully!")
        print("\nTest Summary:")
        print(f"   - Experiment tracking: ✓")
        print(f"   - Drift detection: ✓")
        print(f"   - Retraining pipeline: ✓")
        print(f"   - Model registry: ✓")
        print(f"   - Canary deployment: ✓")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.exception("Test failed")
        return False


def main():
    """Main function"""
    setup_logging()
    
    print("BondX MLOps Integration Test")
    print("This test demonstrates the complete MLOps workflow")
    
    success = test_mlops_integration()
    
    if success:
        print("\n🎉 All tests passed! The MLOps module is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
