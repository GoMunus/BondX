"""
BondX CLI Interface

This module provides a command-line interface for orchestrating the entire BondX analysis pipeline:
- Data validation and quality checks
- Liquidity insights generation
- ML model training and evaluation
- Stress testing and scenario analysis
- Report generation and export
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
from datetime import datetime
import pandas as pd

# BondX imports
from ..analysis.liquidity_insights import LiquidityInsightsEngine
from ..analysis.stress_engine import StressTestingEngine
from ..analysis.heatmaps import SectorHeatmapEngine
from ..ml.spread_model import SpreadPredictionModel, ModelType
from ..ml.downgrade_model import DowngradeRiskModel
from ..ml.liquidity_shock_model import LiquidityShockModel
from ..ml.anomaly_detector import AnomalyDetector
from ..reporting.generate_reports import ReportGenerator
from ..quality.validators import DataValidator
from ..quality.quality_gates import QualityGateManager
from ..quality.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class BondXCLI:
    """Main CLI interface for BondX analysis pipeline"""
    
    def __init__(self):
        """Initialize the CLI interface"""
        self.setup_logging()
        self.config = self.load_config()
        self.metrics_collector = MetricsCollector()
        
        # Initialize engines
        self.liq_engine = LiquidityInsightsEngine(seed=self.config.get('seed', 42))
        self.stress_engine = StressTestingEngine(seed=self.config.get('seed', 42))
        self.heatmap_engine = SectorHeatmapEngine(seed=self.config.get('seed', 42))
        
        # Initialize ML models
        self.spread_model = SpreadPredictionModel(
            seed=self.config.get('seed', 42),
            model_type=ModelType.XGBOOST
        )
        self.downgrade_model = DowngradeRiskModel(seed=self.config.get('seed', 42))
        self.liquidity_shock_model = LiquidityShockModel(seed=self.config.get('seed', 42))
        self.anomaly_detector = AnomalyDetector(seed=self.config.get('seed', 42))
        
        # Initialize reporting
        self.report_generator = ReportGenerator()
        
        # Initialize quality components
        self.data_validator = DataValidator()
        self.quality_gates = QualityGateManager()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('bondx_cli.log')
            ]
        )
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path("config/bondx_config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'seed': 42,
                'output_dir': 'outputs',
                'data_path': 'sample_bond_dataset.csv',
                'model_dir': 'models',
                'report_dir': 'reports',
                'quality_thresholds': {
                    'min_data_quality': 0.8,
                    'min_model_performance': 0.7,
                    'max_processing_time': 300
                }
            }
    
    def validate_data(self, data_path: str) -> bool:
        """Validate input data quality"""
        logger.info(f"Validating data from {data_path}")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Run validation
            validation_result = self.data_validator.validate_dataset(df)
            
            # Check quality gates
            quality_result = self.quality_gates.run_quality_checks(validation_result)
            
            if quality_result['passed']:
                logger.info("Data validation passed quality gates")
                return True
            else:
                logger.error(f"Data validation failed: {quality_result['issues']}")
                return False
                
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def generate_insights(self, data_path: str, output_dir: str) -> Dict[str, str]:
        """Generate liquidity insights"""
        logger.info("Generating liquidity insights")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Generate insights
            output_files = self.liq_engine.export_insights(df, output_dir)
            
            logger.info(f"Insights generated: {output_files}")
            return output_files
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}
    
    def run_stress_tests(self, data_path: str, output_dir: str) -> Dict[str, str]:
        """Run stress testing scenarios"""
        logger.info("Running stress testing scenarios")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Get preset scenarios
            scenarios = self.stress_engine.create_preset_scenarios()
            
            # Run scenarios
            results = self.stress_engine.run_multiple_scenarios(df, scenarios)
            
            # Generate report
            output_files = self.stress_engine.generate_scenario_report(results, output_dir)
            
            logger.info(f"Stress testing completed: {output_files}")
            return output_files
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}
    
    def generate_heatmaps(self, data_path: str, output_dir: str) -> Dict[str, str]:
        """Generate sector heatmaps"""
        logger.info("Generating sector heatmaps")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Generate heatmaps
            output_files = self.heatmap_engine.generate_heatmap_report(df, output_dir)
            
            logger.info(f"Heatmaps generated: {output_files}")
            return output_files
            
        except Exception as e:
            logger.error(f"Error generating heatmaps: {e}")
            return {}
    
    def train_ml_models(self, data_path: str, output_dir: str) -> Dict[str, str]:
        """Train ML models"""
        logger.info("Training ML models")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Train spread prediction model
            spread_results = self.spread_model.train(df)
            
            # Train downgrade risk model
            downgrade_results = self.downgrade_model.train(df)
            
            # Train liquidity shock model
            shock_results = self.liquidity_shock_model.train(df)
            
            # Train anomaly detector
            anomaly_results = self.anomaly_detector.train(df)
            
            # Save models
            model_dir = Path(output_dir) / "models"
            model_dir.mkdir(exist_ok=True)
            
            self.spread_model.save_model(str(model_dir / "spread_model"))
            self.downgrade_model.save_model(str(model_dir / "downgrade_model"))
            self.liquidity_shock_model.save_model(str(model_dir / "liquidity_shock_model"))
            self.anomaly_detector.save_model(str(model_dir / "anomaly_dector"))
            
            # Compile training results
            training_summary = {
                'spread_model': spread_results,
                'downgrade_model': downgrade_results,
                'liquidity_shock_model': shock_results,
                'anomaly_detector': anomaly_results,
                'training_date': datetime.now().isoformat()
            }
            
            # Save training summary
            summary_path = model_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2, default=str)
            
            logger.info("ML model training completed")
            return {'models': str(model_dir), 'summary': str(summary_path)}
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return {}
    
    def generate_reports(self, data_path: str, output_dir: str) -> Dict[str, str]:
        """Generate comprehensive reports"""
        logger.info("Generating reports")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Generate reports
            output_files = self.report_generator.generate_all_reports(df, output_dir)
            
            logger.info(f"Reports generated: {output_files}")
            return output_files
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            return {}
    
    def run_full_pipeline(self, data_path: str, output_dir: str, profile: str = "stage") -> bool:
        """Run the complete analysis pipeline"""
        logger.info(f"Running full pipeline with profile: {profile}")
        
        start_time = datetime.now()
        
        try:
            # 1. Data validation
            if not self.validate_data(data_path):
                logger.error("Data validation failed. Pipeline stopped.")
                return False
            
            # 2. Generate insights
            insights_output = self.generate_insights(data_path, output_dir)
            
            # 3. Run stress tests
            stress_output = self.run_stress_tests(data_path, output_dir)
            
            # 4. Generate heatmaps
            heatmap_output = self.generate_heatmaps(data_path, output_dir)
            
            # 5. Train ML models (skip for dev profile)
            ml_output = {}
            if profile in ["stage", "demo"]:
                ml_output = self.train_ml_models(data_path, output_dir)
            
            # 6. Generate reports
            report_output = self.generate_reports(data_path, output_dir)
            
            # 7. Collect metrics
            pipeline_metrics = {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'profile': profile,
                'outputs': {
                    'insights': insights_output,
                    'stress_tests': stress_output,
                    'heatmaps': heatmap_output,
                    'ml_models': ml_output,
                    'reports': report_output
                }
            }
            
            # Save pipeline metrics
            metrics_path = Path(output_dir) / "pipeline_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(pipeline_metrics, f, indent=2, default=str)
            
            logger.info("Full pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def export_heatmaps(self, data_path: str, output_dir: str) -> Dict[str, str]:
        """Export heatmaps for external use"""
        logger.info("Exporting heatmaps")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Generate heatmaps
            output_files = self.heatmap_engine.generate_heatmap_report(df, output_dir)
            
            # Create dashboard-ready export
            dashboard_export = {
                'heatmaps': output_files,
                'export_date': datetime.now().isoformat(),
                'format': 'dashboard_ready'
            }
            
            export_path = Path(output_dir) / "dashboard_export.json"
            with open(export_path, 'w') as f:
                json.dump(dashboard_export, f, indent=2, default=str)
            
            logger.info("Heatmaps exported successfully")
            return {'dashboard_export': str(export_path)}
            
        except Exception as e:
            logger.error(f"Error exporting heatmaps: {e}")
            return {}
    
    def run_command(self, command: str, **kwargs) -> bool:
        """Run a specific command"""
        try:
            if command == "validate":
                return self.validate_data(kwargs.get('data_path', self.config['data_path']))
            
            elif command == "insights":
                output_files = self.generate_insights(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir'])
                )
                return len(output_files) > 0
            
            elif command == "stress":
                output_files = self.run_stress_tests(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir'])
                )
                return len(output_files) > 0
            
            elif command == "heatmaps":
                output_files = self.generate_heatmaps(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir'])
                )
                return len(output_files) > 0
            
            elif command == "train":
                output_files = self.train_ml_models(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir'])
                )
                return len(output_files) > 0
            
            elif command == "report":
                output_files = self.generate_reports(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir'])
                )
                return len(output_files) > 0
            
            elif command == "export-heatmaps":
                output_files = self.export_heatmaps(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir'])
                )
                return len(output_files) > 0
            
            elif command == "pipeline":
                return self.run_full_pipeline(
                    kwargs.get('data_path', self.config['data_path']),
                    kwargs.get('output_dir', self.config['output_dir']),
                    kwargs.get('profile', 'stage')
                )
            
            else:
                logger.error(f"Unknown command: {command}")
                return False
                
        except Exception as e:
            logger.error(f"Error running command {command}: {e}")
            return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="BondX Analysis Pipeline CLI")
    
    # Global options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data-path', type=str, help='Input data file path')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate input data')
    
    # Insights command
    insights_parser = subparsers.add_parser('insights', help='Generate liquidity insights')
    
    # Stress testing command
    stress_parser = subparsers.add_parser('stress', help='Run stress testing scenarios')
    
    # Heatmaps command
    heatmaps_parser = subparsers.add_parser('heatmaps', help='Generate sector heatmaps')
    
    # ML training command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    
    # Reporting command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    
    # Export heatmaps command
    export_parser = subparsers.add_parser('export-heatmaps', help='Export heatmaps for dashboards')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete analysis pipeline')
    pipeline_parser.add_argument('--profile', choices=['dev', 'stage', 'demo'], 
                               default='stage', help='Pipeline profile')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    cli = BondXCLI()
    
    # Override config with command line arguments
    if args.config:
        # Load custom config
        pass
    
    if args.data_path:
        cli.config['data_path'] = args.data_path
    
    if args.output_dir:
        cli.config['output_dir'] = args.output_dir
    
    if args.seed:
        cli.config['seed'] = args.seed
    
    # Run command
    success = cli.run_command(
        args.command,
        data_path=args.data_path,
        output_dir=args.output_dir,
        profile=getattr(args, 'profile', 'stage')
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
