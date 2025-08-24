"""
MLOps CLI Module

This module provides command-line interface for MLOps operations including:
- Model training and evaluation
- Model registration and promotion
- Drift detection and monitoring
- Canary deployment management
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

from .config import MLOpsConfig
from .tracking import ExperimentTracker
from .registry import ModelRegistry, ModelStage
from .drift import DriftMonitor
from .retrain import RetrainPipeline
from .deploy import CanaryDeploymentManager


class MLOpsCLI:
    """Command-line interface for MLOps operations"""
    
    def __init__(self):
        """Initialize MLOps CLI"""
        self.config = MLOpsConfig.get_default_config()
        self.tracker = ExperimentTracker(self.config)
        self.registry = ModelRegistry(self.config)
        self.drift_monitor = DriftMonitor(self.config)
        self.retrain_pipeline = RetrainPipeline(self.config, self.tracker, self.registry)
        self.deployment_manager = CanaryDeploymentManager(self.config, self.registry)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.log_path + '/mlops_cli.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments"""
        
        parser = self._create_parser()
        
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = parser.parse_args(args)
        
        try:
            if parsed_args.command == 'train':
                return self._handle_train(parsed_args)
            elif parsed_args.command == 'evaluate':
                return self._handle_evaluate(parsed_args)
            elif parsed_args.command == 'register':
                return self._handle_register(parsed_args)
            elif parsed_args.command == 'promote':
                return self._handle_promote(parsed_args)
            elif parsed_args.command == 'rollback':
                return self._handle_rollback(parsed_args)
            elif parsed_args.command == 'drift-check':
                return self._handle_drift_check(parsed_args)
            elif parsed_args.command == 'deploy':
                return self._handle_deploy(parsed_args)
            elif parsed_args.command == 'list':
                return self._handle_list(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        
        parser = argparse.ArgumentParser(
            description='BondX MLOps Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  bondx-mlops train --model spread --data data.csv --config config.yaml
  bondx-mlops evaluate --model spread --version 1.0.0 --test-data test.csv
  bondx-mlops register --model spread --version 1.0.0 --path ./models/spread_v1
  bondx-mlops promote --model spread --version 1.0.0 --stage production --user admin
  bondx-mlops drift-check --model spread --baseline baseline.csv --current current.csv
  bondx-mlops deploy --model spread --version 1.0.1 --user admin --traffic 0.1
  bondx-mlops list --models --stage production
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train a new model')
        train_parser.add_argument('--model', required=True, help='Model name (spread, downgrade, liquidity_shock, anomaly)')
        train_parser.add_argument('--data', required=True, help='Path to training data CSV file')
        train_parser.add_argument('--config', help='Path to training configuration file')
        train_parser.add_argument('--output', help='Output directory for trained model')
        train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        
        # Evaluate command
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
        eval_parser.add_argument('--model', required=True, help='Model name')
        eval_parser.add_argument('--version', required=True, help='Model version')
        eval_parser.add_argument('--test-data', required=True, help='Path to test data CSV file')
        eval_parser.add_argument('--output', help='Output file for evaluation results')
        
        # Register command
        register_parser = subparsers.add_parser('register', help='Register a trained model')
        register_parser.add_argument('--model', required=True, help='Model name')
        register_parser.add_argument('--version', required=True, help='Model version')
        register_parser.add_argument('--path', required=True, help='Path to model directory')
        register_parser.add_argument('--experiment', help='Experiment name')
        register_parser.add_argument('--run-id', help='Experiment run ID')
        
        # Promote command
        promote_parser = subparsers.add_parser('promote', help='Promote a model to a stage')
        promote_parser.add_argument('--model', required=True, help='Model name')
        promote_parser.add_argument('--version', required=True, help='Model version')
        promote_parser.add_argument('--stage', required=True, choices=['staging', 'production'], help='Target stage')
        promote_parser.add_argument('--user', required=True, help='User performing the promotion')
        promote_parser.add_argument('--notes', help='Promotion notes')
        
        # Rollback command
        rollback_parser = subparsers.add_parser('rollback', help='Rollback a model')
        rollback_parser.add_argument('--model', required=True, help='Model name')
        rollback_parser.add_argument('--version', required=True, help='Model version')
        rollback_parser.add_argument('--reason', required=True, help='Rollback reason')
        rollback_parser.add_argument('--user', required=True, help='User performing the rollback')
        
        # Drift check command
        drift_parser = subparsers.add_parser('drift-check', help='Check for data drift')
        drift_parser.add_argument('--model', required=True, help='Model name')
        drift_parser.add_argument('--version', required=True, help='Model version')
        drift_parser.add_argument('--baseline', required=True, help='Path to baseline data CSV file')
        drift_parser.add_argument('--current', required=True, help='Path to current data CSV file')
        drift_parser.add_argument('--output', help='Output file for drift report')
        
        # Deploy command
        deploy_parser = subparsers.add_parser('deploy', help='Create canary deployment')
        deploy_parser.add_argument('--model', required=True, help='Model name')
        deploy_parser.add_argument('--version', required=True, help='Model version to deploy')
        deploy_parser.add_argument('--user', required=True, help='User creating the deployment')
        deploy_parser.add_argument('--traffic', type=float, help='Initial traffic percentage (0.0-1.0)')
        deploy_parser.add_argument('--notes', help='Deployment notes')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List models, deployments, or experiments')
        list_parser.add_argument('--models', action='store_true', help='List models')
        list_parser.add_argument('--deployments', action='store_true', help='List deployments')
        list_parser.add_argument('--experiments', action='store_true', help='List experiments')
        list_parser.add_argument('--stage', choices=['development', 'staging', 'production', 'archived'], help='Filter by stage')
        list_parser.add_argument('--model', help='Filter by model name')
        list_parser.add_argument('--output', help='Output file for results')
        
        return parser
    
    def _handle_train(self, args: argparse.Namespace) -> int:
        """Handle train command"""
        
        try:
            self.logger.info(f"Starting training for model: {args.model}")
            
            # Load training data
            if not os.path.exists(args.data):
                self.logger.error(f"Training data file not found: {args.data}")
                return 1
            
            training_data = pd.read_csv(args.data)
            self.logger.info(f"Loaded training data: {training_data.shape}")
            
            # Execute retraining
            result = self.retrain_pipeline.scheduled_retraining(
                model_name=args.model,
                training_data=training_data
            )
            
            if result.success:
                self.logger.info(f"Training completed successfully. New version: {result.new_model_version}")
                self.logger.info(f"Training time: {result.training_time_seconds:.2f} seconds")
                self.logger.info(f"Model size: {result.model_size_mb:.2f} MB")
                
                # Print metrics
                print(f"\nTraining Results:")
                print(f"  New Version: {result.new_model_version}")
                print(f"  Run ID: {result.run_id}")
                print(f"  Training Time: {result.training_time_seconds:.2f}s")
                print(f"  Model Size: {result.model_size_mb:.2f} MB")
                
                print(f"\nTest Metrics:")
                for metric, value in result.test_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                return 0
            else:
                self.logger.error(f"Training failed: {result.error_message}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Training command failed: {e}")
            return 1
    
    def _handle_evaluate(self, args: argparse.Namespace) -> int:
        """Handle evaluate command"""
        
        try:
            self.logger.info(f"Evaluating model: {args.model}:{args.version}")
            
            # Get model from registry
            model = self.registry.get_model(args.model, args.version)
            if not model:
                self.logger.error(f"Model not found: {args.model}:{args.version}")
                return 1
            
            # Load test data
            if not os.path.exists(args.test_data):
                self.logger.error(f"Test data file not found: {args.test_data}")
                return 1
            
            test_data = pd.read_csv(args.test_data)
            self.logger.info(f"Loaded test data: {test_data.shape}")
            
            # Load model
            import joblib
            trained_model = joblib.load(model.model_path)
            
            # Prepare features
            feature_columns = model.feature_columns
            target_column = model.target_column
            
            X_test = test_data[feature_columns].copy()
            y_test = test_data[target_column].copy()
            
            # Handle missing values
            X_test = X_test.fillna(X_test.mean())
            y_test = y_test.fillna(y_test.mean())
            
            # Convert categorical features
            for col in X_test.columns:
                if X_test[col].dtype == 'object':
                    X_test[col] = pd.Categorical(X_test[col]).codes
            
            # Make predictions
            y_pred = trained_model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
            
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'r2_score': r2_score(y_test, y_pred),
                'explained_variance': explained_variance_score(y_test, y_pred)
            }
            
            # Print results
            print(f"\nEvaluation Results for {args.model}:{args.version}")
            print(f"  Test Data Size: {len(X_test)}")
            print(f"  Features: {len(feature_columns)}")
            
            print(f"\nMetrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save results if output specified
            if args.output:
                results = {
                    'model_name': args.model,
                    'model_version': args.version,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'test_data_size': len(X_test),
                    'feature_count': len(feature_columns),
                    'metrics': metrics
                }
                
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                self.logger.info(f"Evaluation results saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Evaluation command failed: {e}")
            return 1
    
    def _handle_register(self, args: argparse.Narsespace) -> int:
        """Handle register command"""
        
        try:
            self.logger.info(f"Registering model: {args.model}:{args.version}")
            
            # Check if model path exists
            if not os.path.exists(args.path):
                self.logger.error(f"Model path not found: {args.path}")
                return 1
            
            # Find model files
            model_path = os.path.join(args.path, "model.pkl")
            metadata_path = os.path.join(args.path, "metadata.json")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return 1
            
            if not os.path.exists(metadata_path):
                self.logger.error(f"Metadata file not found: {metadata_path}")
                return 1
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Register model
            model_id = self.registry.register_model(
                model_name=args.model,
                version=args.version,
                run_id=args.run_id or "cli_register",
                experiment_name=args.experiment or "cli_experiment",
                model_type=metadata.get('model_type', 'unknown'),
                feature_columns=metadata.get('feature_columns', []),
                target_column=metadata.get('target_column', ''),
                performance_metrics=metadata.get('test_metrics', {}),
                hyperparameters=metadata.get('hyperparameters', {}),
                model_path=model_path,
                metadata_path=metadata_path
            )
            
            self.logger.info(f"Model registered successfully: {model_id}")
            print(f"Model registered: {model_id}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Register command failed: {e}")
            return 1
    
    def _handle_promote(self, args: argparse.Namespace) -> int:
        """Handle promote command"""
        
        try:
            self.logger.info(f"Promoting model: {args.model}:{args.version} to {args.stage}")
            
            # Determine target stage
            if args.stage == 'staging':
                target_stage = ModelStage.STAGING
            elif args.stage == 'production':
                target_stage = ModelStage.PRODUCTION
            else:
                self.logger.error(f"Invalid stage: {args.stage}")
                return 1
            
            # Promote model
            success = self.registry.promote_model(
                model_name=args.model,
                version=args.version,
                target_stage=target_stage,
                deployed_by=args.user,
                deployment_notes=args.notes
            )
            
            if success:
                self.logger.info(f"Model promoted successfully to {args.stage}")
                print(f"Model {args.model}:{args.version} promoted to {args.stage}")
                return 0
            else:
                self.logger.error("Model promotion failed")
                return 1
                
        except Exception as e:
            self.logger.error(f"Promote command failed: {e}")
            return 1
    
    def _handle_rollback(self, args: argparse.Namespace) -> int:
        """Handle rollback command"""
        
        try:
            self.logger.info(f"Rolling back model: {args.model}:{args.version}")
            
            # Rollback model
            success = self.registry.demote_model(
                model_name=args.model,
                version=args.version,
                reason=args.reason
            )
            
            if success:
                self.logger.info("Model rolled back successfully")
                print(f"Model {args.model}:{args.version} rolled back")
                return 0
            else:
                self.logger.error("Model rollback failed")
                return 1
                
        except Exception as e:
            self.logger.error(f"Rollback command failed: {e}")
            return 1
    
    def _handle_drift_check(self, args: argparse.Namespace) -> int:
        """Handle drift-check command"""
        
        try:
            self.logger.info(f"Checking drift for model: {args.model}:{args.version}")
            
            # Load data files
            if not os.path.exists(args.baseline):
                self.logger.error(f"Baseline data file not found: {args.baseline}")
                return 1
            
            if not os.path.exists(args.current):
                self.logger.error(f"Current data file not found: {args.current}")
                return 1
            
            baseline_data = pd.read_csv(args.baseline)
            current_data = pd.read_csv(args.current)
            
            self.logger.info(f"Loaded baseline data: {baseline_data.shape}")
            self.logger.info(f"Loaded current data: {current_data.shape}")
            
            # Get model from registry
            model = self.registry.get_model(args.model, args.version)
            if not model:
                self.logger.error(f"Model not found: {args.model}:{args.version}")
                return 1
            
            # Detect drift
            drift_report = self.drift_monitor.detect_drift(
                model_name=args.model,
                model_version=args.version,
                baseline_data=baseline_data,
                current_data=current_data,
                feature_columns=model.feature_columns,
                target_column=model.target_column
            )
            
            # Print results
            print(f"\nDrift Detection Results for {args.model}:{args.version}")
            print(f"  Analysis Timestamp: {drift_report.analysis_timestamp}")
            print(f"  Total Features: {drift_report.total_features}")
            print(f"  Drifted Features: {drift_report.drifted_features}")
            print(f"  Drift Percentage: {drift_report.drift_percentage:.2f}%")
            print(f"  Overall Drift Score: {drift_report.overall_drift_score:.4f}")
            print(f"  Requires Retraining: {drift_report.requires_retraining}")
            
            print(f"\nRecommendations:")
            for rec in drift_report.recommendations:
                print(f"  - {rec}")
            
            # Save results if output specified
            if args.output:
                # Convert drift report to serializable format
                report_data = {
                    'model_name': drift_report.model_name,
                    'model_version': drift_report.model_version,
                    'analysis_timestamp': drift_report.analysis_timestamp.isoformat(),
                    'total_features': drift_report.total_features,
                    'drifted_features': drift_report.drifted_features,
                    'drift_percentage': drift_report.drift_percentage,
                    'overall_drift_score': drift_report.overall_drift_score,
                    'requires_retraining': drift_report.requires_retraining,
                    'recommendations': drift_report.recommendations
                }
                
                with open(args.output, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                
                self.logger.info(f"Drift report saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Drift-check command failed: {e}")
            return 1
    
    def _handle_deploy(self, args: argparse.Namespace) -> int:
        """Handle deploy command"""
        
        try:
            self.logger.info(f"Creating canary deployment for model: {args.model}:{args.version}")
            
            # Create canary deployment
            deployment_id = self.deployment_manager.create_canary_deployment(
                model_name=args.model,
                candidate_version=args.version,
                deployed_by=args.user,
                initial_traffic_percentage=args.traffic,
                deployment_notes=args.notes
            )
            
            self.logger.info(f"Canary deployment created: {deployment_id}")
            print(f"Canary deployment created: {deployment_id}")
            
            # Get deployment status
            deployment = self.deployment_manager.get_deployment_status(deployment_id)
            if deployment:
                print(f"  Status: {deployment.status.value}")
                print(f"  Traffic: {deployment.current_traffic_percentage:.1%}")
                print(f"  Production: {deployment.production_version}")
                print(f"  Candidate: {deployment.candidate_version}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Deploy command failed: {e}")
            return 1
    
    def _handle_list(self, args: argparse.Namespace) -> int:
        """Handle list command"""
        
        try:
            results = {}
            
            if args.models:
                self.logger.info("Listing models")
                models = self.registry.list_models(
                    model_name=args.model,
                    stage=ModelStage(args.stage) if args.stage else None
                )
                results['models'] = [
                    {
                        'name': m.model_name,
                        'version': m.version,
                        'stage': m.stage.value,
                        'status': m.status.value,
                        'created_at': m.created_at.isoformat(),
                        'deployed_at': m.deployed_at.isoformat() if m.deployed_at else None
                    }
                    for m in models
                ]
            
            if args.deployments:
                self.logger.info("Listing deployments")
                deployments = self.deployment_manager.list_deployments(
                    model_name=args.model
                )
                results['deployments'] = [
                    {
                        'id': d.deployment_id,
                        'model_name': d.model_name,
                        'status': d.status.value,
                        'production_version': d.production_version,
                        'candidate_version': d.candidate_version,
                        'traffic_percentage': d.current_traffic_percentage,
                        'created_at': d.created_at.isoformat()
                    }
                    for d in deployments
                ]
            
            if args.experiments:
                self.logger.info("Listing experiments")
                experiments = self.tracker.list_runs(
                    experiment_name=args.model
                )
                results['experiments'] = [
                    {
                        'run_id': e.run_id,
                        'experiment_name': e.experiment_name,
                        'model_type': e.model_type,
                        'status': e.status,
                        'start_time': e.start_time.isoformat(),
                        'end_time': e.end_time.isoformat() if e.end_time else None
                    }
                    for e in experiments
                ]
            
            # Print results
            if results:
                for category, items in results.items():
                    print(f"\n{category.upper()}:")
                    if not items:
                        print("  No items found")
                    else:
                        for item in items:
                            if category == 'models':
                                print(f"  {item['name']}:{item['version']} ({item['stage']}) - {item['status']}")
                            elif category == 'deployments':
                                print(f"  {item['id']} - {item['model_name']} ({item['status']}) - {item['traffic_percentage']:.1%}")
                            elif category == 'experiments':
                                print(f"  {item['run_id']} - {item['experiment_name']} ({item['status']})")
            
            # Save results if output specified
            if args.output and results:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                self.logger.info(f"List results saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"List command failed: {e}")
            return 1


def main():
    """Main entry point for MLOps CLI"""
    
    cli = MLOpsCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
