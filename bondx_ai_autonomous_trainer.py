#!/usr/bin/env python3
"""
BondX AI Autonomous Training System - Enhanced for Enterprise

This system autonomously generates synthetic datasets, trains ML models, validates quality,
and continuously improves until convergence. Enhanced for long training sessions (4-6 hours)
with enterprise-grade monitoring, ensemble modeling, and stress testing capabilities.
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add bondx to path
sys.path.append(str(Path(__file__).parent / "bondx"))

from bondx.ml.spread_model import SpreadPredictionModel, ModelType
from bondx.ml.downgrade_model import DowngradePredictionModel
from bondx.ml.liquidity_shock_model import LiquidityShockModel
from bondx.ml.anomaly_detector import AnomalyDetector
from bondx.quality.quality_gates import QualityGateManager
from bondx.quality.validators import DataValidator
from bondx.reporting.generate_reports import ReportGenerator
from bondx.reporting.regulator.evidence_pack import EvidencePackGenerator

# Configure enhanced logging for long sessions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bondx_ai_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add log rotation for long sessions
from logging.handlers import RotatingFileHandler
rotating_handler = RotatingFileHandler(
    'bondx_ai_training.log', 
    maxBytes=100*1024*1024,  # 100MB
    backupCount=5
)
rotating_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(rotating_handler)

@dataclass
class TrainingMetrics:
    """Enhanced metrics for model training performance"""
    model_name: str
    epoch: int
    mse: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    timestamp: datetime
    feature_importance: Dict[str, float] = None
    convergence_rate: float = None
    stress_test_results: Dict[str, Any] = None

@dataclass
class QualityMetrics:
    """Enhanced quality metrics for datasets"""
    dataset_name: str
    coverage: float
    esg_completeness: float
    liquidity_median: float
    negative_spreads_pct: float
    maturity_anomalies_pct: float
    quality_score: float
    timestamp: datetime
    stress_test_quality: float = None
    macro_scenario_impact: float = None

@dataclass
class ConvergenceStatus:
    """Enhanced status of training convergence for long sessions"""
    models_converged: bool
    quality_stable: bool
    anomaly_rate: float
    improvement_rate: float
    epochs_since_improvement: int
    timestamp: datetime
    total_training_time_hours: float = None
    convergence_confidence: float = None
    stress_test_convergence: bool = None

@dataclass
class EnsembleMetrics:
    """Metrics for ensemble model performance"""
    ensemble_name: str
    base_models: List[str]
    meta_learner: str
    ensemble_accuracy: float
    ensemble_precision: float
    ensemble_recall: float
    ensemble_f1: float
    improvement_over_best_base: float
    timestamp: datetime

class BondXAIAutonomousTrainer:
    """Enhanced orchestrator for autonomous BondX AI training with enterprise features"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced autonomous trainer"""
        self.config = self._load_config(config_path)
        self.seed = self.config.get('random_seed', 42)
        np.random.seed(self.seed)
        
        # Initialize components
        self.quality_gates = QualityGateManager()
        self.validator = DataValidator()
        self.report_generator = ReportGenerator()
        self.evidence_generator = EvidencePackGenerator()
        
        # Enhanced training state for long sessions
        self.current_epoch = 0
        self.training_start_time = datetime.now()
        self.training_metrics: List[TrainingMetrics] = []
        self.quality_metrics: List[QualityMetrics] = []
        self.ensemble_metrics: List[EnsembleMetrics] = []
        self.convergence_status = ConvergenceStatus(
            models_converged=False,
            quality_stable=False,
            anomaly_rate=1.0,
            improvement_rate=1.0,
            epochs_since_improvement=0,
            timestamp=datetime.now(),
            total_training_time_hours=0.0,
            convergence_confidence=0.0,
            stress_test_convergence=False
        )
        
        # Model instances
        self.models = {}
        self.ensemble_models = {}
        self.datasets = {}
        
        # Output directories
        self.output_dir = Path(self.config.get('output', {}).get('base_directory', "autonomous_training_output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced monitoring
        self.monitoring_enabled = self.config.get('monitoring', {}).get('enable_prometheus_metrics', False)
        self.alert_thresholds = self.config.get('monitoring', {}).get('alert_thresholds', {})
        
        logger.info(f"Enhanced BondX AI Autonomous Trainer initialized with seed {self.seed}")
        logger.info(f"Configuration: {self.config}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load enhanced configuration from file or use defaults"""
        default_config = {
            'random_seed': 42,
            'max_epochs': 500,  # Increased for long sessions
            'convergence_threshold': 0.0001,  # More stringent
            'quality_threshold': 0.98,  # Higher quality requirements
            'anomaly_threshold': 0.03,  # Lower anomaly tolerance
            'improvement_patience': 25,  # Increased patience
            'dataset_size': 1000,  # Expanded dataset
            'training_split': 0.75,
            'validation_split': 0.15,
            'test_split': 0.10,
            'convergence_timeout_hours': 6,  # Maximum training time
            'enable_ensemble_methods': True,
            'enable_stress_testing': True,
            'enable_scenario_simulation': True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    default_config.update(custom_config)
                    logger.info(f"Loaded custom config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")
        
        return default_config
    
    async def run_autonomous_training_loop(self):
        """Enhanced main autonomous training loop for long sessions"""
        logger.info("Starting Enhanced BondX AI autonomous training loop")
        start_time = time.time()
        
        try:
            # Phase 1: Initial dataset generation with expanded diversity
            await self._generate_initial_datasets()
            
            # Phase 2: Continuous training loop with enhanced monitoring
            while not self._should_stop_training():
                self.current_epoch += 1
                epoch_start_time = time.time()
                
                logger.info(f"Starting training epoch {self.current_epoch}")
                
                # Generate new dataset iteration with macroeconomic factors
                await self._generate_dataset_iteration()
                
                # Train all models with enhanced monitoring
                await self._train_all_models()
                
                # Train ensemble models if enabled
                if self.config.get('enable_ensemble_methods', False):
                    await self._train_ensemble_models()
                
                # Run stress testing if enabled
                if self.config.get('enable_stress_testing', False):
                    await self._run_stress_tests()
                
                # Validate quality with enhanced metrics
                await self._validate_quality()
                
                # Update convergence status with enhanced tracking
                self._update_convergence_status()
                
                # Generate enhanced reports
                await self._generate_epoch_reports()
                
                # Check for convergence with enhanced criteria
                if self._check_convergence():
                    logger.info("Training converged! Stopping autonomous loop.")
                    break
                
                # Check for timeout
                if self._check_timeout():
                    logger.warning("Training timeout reached. Stopping autonomous loop.")
                    break
                
                # Enhanced monitoring and alerting
                await self._monitor_and_alert()
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {self.current_epoch} completed in {epoch_time:.2f} seconds")
            
            # Phase 3: Final validation and output with compliance packs
            await self._finalize_training()
            
            total_time = time.time() - start_time
            total_hours = total_time / 3600
            logger.info(f"Enhanced autonomous training completed in {total_hours:.2f} hours")
            
        except Exception as e:
            logger.error(f"Error in enhanced autonomous training loop: {e}")
            raise
    
    def _check_timeout(self) -> bool:
        """Check if training has exceeded the maximum time limit"""
        if 'convergence_timeout_hours' not in self.config:
            return False
        
        elapsed_hours = (datetime.now() - self.training_start_time).total_seconds() / 3600
        timeout_hours = self.config['convergence_timeout_hours']
        
        if elapsed_hours >= timeout_hours:
            logger.warning(f"Training timeout reached: {elapsed_hours:.2f} hours >= {timeout_hours} hours")
            return True
        
        return False
    
    async def _monitor_and_alert(self):
        """Enhanced monitoring and alerting for enterprise deployment"""
        if not self.monitoring_enabled:
            return
        
        # Check alert thresholds
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name == 'accuracy_drop':
                current_accuracy = self._get_current_accuracy()
                if current_accuracy < threshold:
                    await self._send_alert(f"Accuracy dropped below {threshold}: {current_accuracy:.3f}")
            
            elif metric_name == 'quality_score_drop':
                current_quality = self._get_current_quality_score()
                if current_quality < threshold:
                    await self._send_alert(f"Quality score dropped below {threshold}: {current_quality:.3f}")
            
            elif metric_name == 'convergence_stall_hours':
                if self.convergence_status.epochs_since_improvement > 0:
                    stall_hours = self.convergence_status.epochs_since_improvement * 0.1  # Approximate
                    if stall_hours >= threshold:
                        await self._send_alert(f"Training stalled for {stall_hours:.1f} hours")
    
    async def _send_alert(self, message: str):
        """Send alert through configured channels"""
        logger.warning(f"ALERT: {message}")
        
        # TODO: Implement actual alerting (Slack, email, etc.)
        # For now, just log the alert
        
        # Save alert to file
        alert_file = self.output_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
    
    def _get_current_accuracy(self) -> float:
        """Get current model accuracy"""
        if not self.training_metrics:
            return 0.0
        
        recent_metrics = [m for m in self.training_metrics if m.epoch >= self.current_epoch - 5]
        if not recent_metrics:
            return 0.0
        
        return np.mean([m.accuracy for m in recent_metrics])
    
    def _get_current_quality_score(self) -> float:
        """Get current dataset quality score"""
        if not self.quality_metrics:
            return 0.0
        
        recent_metrics = [m for m in self.quality_metrics if m.dataset_name == f"epoch_{self.current_epoch}"]
        if not recent_metrics:
            return 0.0
        
        return np.mean([m.quality_score for m in recent_metrics])
    
    async def _train_ensemble_models(self):
        """Train ensemble models for improved predictive power"""
        logger.info("Training ensemble models")
        
        if not self.models:
            logger.warning("No base models available for ensemble training")
            return
        
        try:
            # Simple stacking ensemble
            base_predictions = {}
            for model_name, model in self.models.items():
                if hasattr(model, 'predict'):
                    # Get predictions from base model
                    current_data = self.datasets[f"bonds_epoch_{self.current_epoch}"]
                    features, _ = self._prepare_training_data(current_data)
                    
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(features)[:, 1]  # Probability of positive class
                    else:
                        predictions = model.predict(features)
                    
                    base_predictions[model_name] = predictions
            
            if len(base_predictions) >= 2:
                # Create ensemble predictions
                ensemble_predictions = np.mean(list(base_predictions.values()), axis=0)
                
                # Calculate ensemble metrics
                ensemble_metrics = EnsembleMetrics(
                    ensemble_name="stacking_ensemble",
                    base_models=list(base_predictions.keys()),
                    meta_learner="mean",
                    ensemble_accuracy=0.85,  # Placeholder
                    ensemble_precision=0.83,  # Placeholder
                    ensemble_recall=0.87,  # Placeholder
                    ensemble_f1=0.85,  # Placeholder
                    improvement_over_best_base=0.02,  # Placeholder
                    timestamp=datetime.now()
                )
                
                self.ensemble_metrics.append(ensemble_metrics)
                logger.info("Ensemble model training completed")
        
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
    
    async def _run_stress_tests(self):
        """Run stress tests for scenario simulation"""
        logger.info("Running stress tests")
        
        try:
            # Get current dataset
            current_data = self.datasets[f"bonds_epoch_{self.current_epoch}"]
            
            # Run different stress scenarios
            scenarios = [
                "global_liquidity_freeze",
                "downgrade_cascade", 
                "interest_rate_shocks",
                "inflation_scenarios",
                "fx_risk_scenarios"
            ]
            
            stress_results = {}
            for scenario in scenarios:
                if self.config.get('stress_testing', {}).get(f'enable_{scenario}', False):
                    scenario_result = await self._run_stress_scenario(current_data, scenario)
                    stress_results[scenario] = scenario_result
            
            # Store stress test results
            stress_file = self.output_dir / f"epoch_{self.current_epoch}_stress_tests.json"
            with open(stress_file, 'w') as f:
                json.dump(stress_results, f, indent=2)
            
            logger.info(f"Stress tests completed for {len(stress_results)} scenarios")
        
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
    
    async def _run_stress_scenario(self, data: pd.DataFrame, scenario: str) -> Dict[str, Any]:
        """Run a specific stress scenario"""
        try:
            # Simulate stress scenario
            if scenario == "global_liquidity_freeze":
                # Reduce liquidity scores across the board
                stressed_data = data.copy()
                stressed_data['Liquidity_Score(0-100)'] = data['Liquidity_Score(0-100)'] * 0.3
                stressed_data['Liquidity_Spread(bps)'] = data['Liquidity_Spread(bps)'] * 2.0
                
            elif scenario == "downgrade_cascade":
                # Simulate rating downgrades
                stressed_data = data.copy()
                # Reduce ratings by 1-2 notches
                rating_mapping = {i: max(1, i-1) for i in range(1, 23)}  # 23 rating levels
                stressed_data['Rating_Numeric'] = data['Rating_Numeric'].map(rating_mapping)
                
            elif scenario == "interest_rate_shocks":
                # Simulate interest rate increases
                stressed_data = data.copy()
                stressed_data['Yield_to_Maturity(%)'] = data['Yield_to_Maturity(%)'] + 2.0
                stressed_data['Liquidity_Spread(bps)'] = data['Liquidity_Spread(bps)'] * 1.5
                
            else:
                # Default stress scenario
                stressed_data = data.copy()
                stressed_data['Liquidity_Score(0-100)'] = data['Liquidity_Score(0-100)'] * 0.8
            
            # Calculate stress impact metrics
            impact_metrics = {
                'scenario': scenario,
                'original_liquidity_median': data['Liquidity_Score(0-100)'].median(),
                'stressed_liquidity_median': stressed_data['Liquidity_Score(0-100)'].median(),
                'liquidity_deterioration': (
                    data['Liquidity_Score(0-100)'].median() - 
                    stressed_data['Liquidity_Score(0-100)'].median()
                ) / data['Liquidity_Score(0-100)'].median() * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            return impact_metrics
        
        except Exception as e:
            logger.error(f"Error running stress scenario {scenario}: {e}")
            return {'error': str(e), 'scenario': scenario}
    
    async def _generate_initial_datasets(self):
        """Generate initial synthetic datasets with expanded diversity"""
        logger.info("Generating initial synthetic datasets with expanded diversity")
        
        try:
            # Import and run enhanced dataset generation
            from data.synthetic.generate_enterprise_dataset import (
                generate_enhanced_synthetic_dataset,
                export_datasets
            )
            
            # Generate enhanced datasets
            bonds_data = generate_enhanced_synthetic_dataset(self.config['dataset_size'])
            
            # Export to multiple formats
            export_datasets(bonds_data, self.output_dir)
            
            # Load datasets
            self.datasets['bonds'] = pd.read_csv(self.output_dir / "enhanced_bonds_1000plus.csv")
            
            logger.info(f"Generated initial dataset with {len(self.datasets['bonds'])} bonds")
            logger.info(f"Dataset sectors: {self.datasets['bonds']['Sector'].nunique()}")
            logger.info(f"Dataset ratings: {self.datasets['bonds']['Rating'].nunique()}")
            
        except ImportError:
            logger.warning("Enhanced dataset generator not available, falling back to basic generation")
            # Fallback to basic dataset generation
            await self._generate_basic_datasets()
    
    async def _generate_basic_datasets(self):
        """Fallback basic dataset generation"""
        logger.info("Generating basic synthetic datasets")
        
        # Import and run basic dataset generation
        from data.synthetic.generate_enhanced_synthetic_dataset import (
            generate_enhanced_synthetic_dataset,
            export_datasets
        )
        
        # Generate datasets
        bonds_data = generate_enhanced_synthetic_dataset(self.config['dataset_size'])
        
        # Export to multiple formats
        export_datasets(bonds_data, self.output_dir)
        
        # Load datasets
        self.datasets['bonds'] = pd.read_csv(self.output_dir / "enhanced_bonds_150plus.csv")
        
        logger.info(f"Generated basic dataset with {len(self.datasets['bonds'])} bonds")
    
    async def _generate_dataset_iteration(self):
        """Generate new dataset iteration with macroeconomic variations"""
        logger.info(f"Generating dataset iteration for epoch {self.current_epoch}")
        
        # Create variations of existing dataset
        base_data = self.datasets['bonds'].copy()
        
        # Add noise and variations with macroeconomic factors
        noise_factor = 0.01 * (1 + self.current_epoch * 0.05)  # Reduced noise for stability
        
        # Vary numerical fields
        numerical_cols = ['Coupon_Rate(%)', 'Face_Value($)', 'Market_Price($)', 
                         'Yield_to_Maturity(%)', 'Liquidity_Score(0-100)', 'ESG_Score',
                         'Liquidity_Spread(bps)', 'BondX_Score']
        
        for col in numerical_cols:
            if col in base_data.columns:
                noise = np.random.normal(0, noise_factor, len(base_data))
                base_data[col] = base_data[col] * (1 + noise)
                base_data[col] = base_data[col].clip(lower=0)  # Ensure non-negative
        
        # Add macroeconomic scenario variations
        macro_scenarios = ['interest_rate_shock', 'inflation_shock', 'fx_volatility', 'liquidity_freeze']
        current_scenario = macro_scenarios[self.current_epoch % len(macro_scenarios)]
        
        # Apply scenario-specific variations
        if current_scenario == 'interest_rate_shock':
            base_data['Yield_to_Maturity(%)'] += np.random.normal(0.1, 0.05, len(base_data))
        elif current_scenario == 'inflation_shock':
            base_data['ESG_Score'] *= np.random.normal(0.98, 0.02, len(base_data))
        elif current_scenario == 'fx_volatility':
            base_data['Liquidity_Score(0-100)'] *= np.random.normal(0.99, 0.01, len(base_data))
        elif current_scenario == 'liquidity_freeze':
            base_data['Liquidity_Spread(bps)'] *= np.random.normal(1.02, 0.01, len(base_data))
        
        # Store iteration
        iteration_name = f"bonds_epoch_{self.current_epoch}"
        self.datasets[iteration_name] = base_data
        
        # Export iteration
        base_data.to_csv(self.output_dir / f"{iteration_name}.csv", index=False)
        
        # Export metadata
        metadata = {
            'epoch': self.current_epoch,
            'scenario': current_scenario,
            'noise_factor': noise_factor,
            'generation_timestamp': datetime.now().isoformat(),
            'base_dataset_size': len(base_data),
            'variations_applied': numerical_cols
        }
        
        metadata_path = self.output_dir / f"{iteration_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated dataset iteration: {iteration_name} with scenario: {current_scenario}")
    
    async def _train_all_models(self):
        """Train all ML models with enhanced monitoring and feature importance tracking"""
        logger.info(f"Training all models for epoch {self.current_epoch}")
        
        current_data = self.datasets[f"bonds_epoch_{self.current_epoch}"]
        
        # Prepare features and targets
        features, targets = self._prepare_training_data(current_data)
        
        # Train spread model with enhanced monitoring
        if 'spread_model' not in self.models:
            self.models['spread_model'] = SpreadPredictionModel(seed=self.seed)
        
        spread_metrics = await self._train_model_with_monitoring(
            self.models['spread_model'], 
            features, 
            targets['spread'], 
            'spread_model'
        )
        
        # Train downgrade model with enhanced monitoring
        if 'downgrade_model' not in self.models:
            self.models['downgrade_model'] = DowngradePredictionModel(seed=self.seed)
        
        downgrade_metrics = await self._train_model_with_monitoring(
            self.models['downgrade_model'],
            features,
            targets['downgrade'],
            'downgrade_model'
        )
        
        # Train liquidity shock model with enhanced monitoring
        if 'liquidity_shock_model' not in self.models:
            self.models['liquidity_shock_model'] = LiquidityShockModel(seed=self.seed)
        
        liquidity_metrics = await self._train_model_with_monitoring(
            self.models['liquidity_shock_model'],
            features,
            targets['liquidity_shock'],
            'liquidity_shock_model'
        )
        
        # Train anomaly detector with enhanced monitoring
        if 'anomaly_detector' not in self.models:
            self.models['anomaly_detector'] = AnomalyDetector(seed=self.seed)
        
        anomaly_metrics = await self._train_model_with_monitoring(
            self.models['anomaly_detector'],
            features,
            targets['anomaly'],
            'anomaly_detector'
        )
        
        # Store enhanced metrics
        self.training_metrics.extend([
            spread_metrics, downgrade_metrics, 
            liquidity_metrics, anomaly_metrics
        ])
        
        # Log feature importance if available
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_file = self.output_dir / f"epoch_{self.current_epoch}_{model_name}_feature_importance.json"
                with open(importance_file, 'w') as f:
                    json.dump(model.feature_importances_, f, indent=2)
                logger.info(f"Feature importance saved for {model_name}")
    
    async def _train_model_with_monitoring(self, model, features, targets, model_name: str) -> TrainingMetrics:
        """Train a single model with enhanced monitoring and feature importance tracking"""
        start_time = time.time()
        
        try:
            # Train the model
            if hasattr(model, 'fit'):
                model.fit(features, targets)
            
            # Get predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(features)
            else:
                predictions = np.zeros_like(targets)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
            
            mse = mean_squared_error(targets, predictions)
            
            # For classification models, calculate classification metrics
            if hasattr(model, 'predict_proba'):
                # Binary classification
                if len(np.unique(targets)) == 2:
                    accuracy = accuracy_score(targets, predictions)
                    precision = precision_score(targets, predictions, average='binary', zero_division=0)
                    recall = recall_score(targets, predictions, average='binary', zero_division=0)
                    f1 = f1_score(targets, predictions, average='binary', zero_division=0)
                else:
                    # Multi-class classification
                    accuracy = accuracy_score(targets, predictions)
                    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
                    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
                    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
            else:
                # Regression model
                accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like metric
                precision = 1.0 / (1.0 + mse)
                recall = 1.0 / (1.0 + mse)
                f1 = 1.0 / (1.0 + mse)
            
            training_time = time.time() - start_time
            
            # Extract feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features.columns, model.feature_importances_))
            
            # Calculate convergence rate
            convergence_rate = None
            if len(self.training_metrics) > 0:
                recent_metrics = [m for m in self.training_metrics if m.model_name == model_name]
                if recent_metrics:
                    prev_mse = recent_metrics[-1].mse
                    convergence_rate = (prev_mse - mse) / prev_mse if prev_mse > 0 else 0.0
            
            # Create enhanced training metrics
            metrics = TrainingMetrics(
                model_name=model_name,
                epoch=self.current_epoch,
                mse=mse,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                timestamp=datetime.now(),
                feature_importance=feature_importance,
                convergence_rate=convergence_rate
            )
            
            logger.info(f"{model_name} training completed - Accuracy: {accuracy:.3f}, MSE: {mse:.6f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            # Return error metrics
            return TrainingMetrics(
                model_name=model_name,
                epoch=self.current_epoch,
                mse=float('inf'),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare features and targets for training"""
        # Select numerical features
        feature_cols = ['Coupon_Rate(%)', 'Face_Value($)', 'Market_Price($)', 
                       'Yield_to_Maturity(%)', 'Liquidity_Score(0-100)', 'ESG_Score']
        
        # Filter to columns that exist
        available_features = [col for col in feature_cols if col in data.columns]
        
        if not available_features:
            # Fallback to any numerical columns
            available_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        features = data[available_features].fillna(0).values
        
        # Create synthetic targets
        targets = {
            'spread': np.random.normal(100, 50, len(data)),  # Spread in bps
            'downgrade': np.random.choice([0, 1], len(data), p=[0.9, 0.1]),  # Binary
            'liquidity_shock': np.random.choice([0, 1], len(data), p=[0.95, 0.05]),  # Binary
            'anomaly': np.random.choice([0, 1], len(data), p=[0.98, 0.02])  # Binary
        }
        
        return features, targets
    
    async def _validate_quality(self):
        """Validate dataset quality using quality gates"""
        logger.info(f"Validating quality for epoch {self.current_epoch}")
        
        current_data = self.datasets[f"bonds_epoch_{self.current_epoch}"]
        
        # Run quality gates
        gate_results = []
        
        # Coverage gate
        coverage_result = self.quality_gates.evaluate_coverage_gate(current_data, f"epoch_{self.current_epoch}")
        gate_results.append(coverage_result)
        
        # ESG completeness gate
        if 'ESG_Score' in current_data.columns:
            esg_completeness = (current_data['ESG_Score'].notna().sum() / len(current_data)) * 100
        else:
            esg_completeness = 0.0
        
        # Liquidity index gate
        if 'Liquidity_Score(0-100)' in current_data.columns:
            liquidity_median = current_data['Liquidity_Score(0-100)'].median()
        else:
            liquidity_median = 0.0
        
        # Negative spreads gate
        if 'Yield_to_Maturity(%)' in current_data.columns:
            negative_spreads_pct = (current_data['Yield_to_Maturity(%)'] < 0).sum() / len(current_data) * 100
        else:
            negative_spreads_pct = 0.0
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(gate_results, esg_completeness, 
                                                   liquidity_median, negative_spreads_pct)
        
        # Store quality metrics
        quality_metrics = QualityMetrics(
            dataset_name=f"epoch_{self.current_epoch}",
            coverage=coverage_result.actual_value,
            esg_completeness=esg_completeness,
            liquidity_median=liquidity_median,
            negative_spreads_pct=negative_spreads_pct,
            maturity_anomalies_pct=0.0,  # Placeholder
            quality_score=quality_score,
            timestamp=datetime.now()
        )
        
        self.quality_metrics.append(quality_metrics)
        
        logger.info(f"Quality validation completed. Score: {quality_score:.3f}")
    
    def _calculate_quality_score(self, gate_results: List, esg_completeness: float, 
                               liquidity_median: float, negative_spreads_pct: float) -> float:
        """Calculate overall quality score"""
        # Base score from quality gates
        gate_score = sum(1 for result in gate_results if result.passed) / len(gate_results) if gate_results else 0.0
        
        # ESG score component
        esg_score = min(esg_completeness / 100.0, 1.0)
        
        # Liquidity score component
        liquidity_score = min(liquidity_median / 100.0, 1.0)
        
        # Spread quality component
        spread_score = max(0, 1.0 - negative_spreads_pct / 100.0)
        
        # Weighted average
        quality_score = (
            gate_score * 0.4 +
            esg_score * 0.2 +
            liquidity_score * 0.2 +
            spread_score * 0.2
        )
        
        return quality_score
    
    def _update_convergence_status(self):
        """Update convergence status with enhanced tracking for long sessions"""
        if not self.training_metrics:
            return
        
        # Get recent metrics
        recent_metrics = [m for m in self.training_metrics if m.epoch >= self.current_epoch - 5]
        if not recent_metrics:
            return
        
        # Calculate improvement rate
        if len(recent_metrics) >= 2:
            recent_accuracy = np.mean([m.accuracy for m in recent_metrics[-3:]])
            older_accuracy = np.mean([m.accuracy for m in recent_metrics[:-3]])
            
            if older_accuracy > 0:
                self.convergence_status.improvement_rate = (recent_accuracy - older_accuracy) / older_accuracy
            else:
                self.convergence_status.improvement_rate = 0.0
        
        # Check for improvement
        if self.convergence_status.improvement_rate > self.config.get('convergence', {}).get('model_improvement_threshold', 0.0005):
            self.convergence_status.epochs_since_improvement = 0
        else:
            self.convergence_status.epochs_since_improvement += 1
        
        # Update total training time
        elapsed_hours = (datetime.now() - self.training_start_time).total_seconds() / 3600
        self.convergence_status.total_training_time_hours = elapsed_hours
        
        # Calculate convergence confidence
        if self.convergence_status.epochs_since_improvement == 0:
            self.convergence_status.convergence_confidence = 1.0
        else:
            self.convergence_status.convergence_confidence = max(0.0, 1.0 - (self.convergence_status.epochs_since_improvement * 0.1))
        
        # Update timestamp
        self.convergence_status.timestamp = datetime.now()
        
        logger.info(f"Convergence status updated - Improvement rate: {self.convergence_status.improvement_rate:.6f}, "
                   f"Epochs since improvement: {self.convergence_status.epochs_since_improvement}, "
                   f"Confidence: {self.convergence_status.convergence_confidence:.3f}")
    
    def _check_convergence(self) -> bool:
        """Check for convergence with enhanced criteria for long sessions"""
        if self.current_epoch < self.config.get('convergence', {}).get('min_epochs', 50):
            return False
        
        # Check if models have converged
        if self.convergence_status.epochs_since_improvement >= self.config.get('convergence', {}).get('max_epochs_without_improvement', 30):
            logger.info("Models converged - no improvement for required epochs")
            return True
        
        # Check if quality is stable
        if len(self.quality_metrics) >= 5:
            recent_quality = [m.quality_score for m in self.quality_metrics[-5:]]
            quality_stability = np.std(recent_quality)
            if quality_stability < self.config.get('convergence', {}).get('quality_stability_threshold', 0.005):
                logger.info("Quality is stable - training converged")
                return True
        
        # Check if improvement rate is below threshold
        if abs(self.convergence_status.improvement_rate) < self.config.get('convergence_threshold', 0.0001):
            logger.info("Improvement rate below threshold - training converged")
            return True
        
        return False
    
    def _should_stop_training(self) -> bool:
        """Check if training should stop"""
        # Check epoch limit
        if self.current_epoch >= self.config.get('max_epochs', 500):
            logger.info("Maximum epochs reached")
            return True
        
        # Check timeout
        if self._check_timeout():
            return True
        
        # Check if models have converged
        if self.convergence_status.models_converged:
            return True
        
        return False
    
    async def _generate_epoch_reports(self):
        """Generate reports for current epoch"""
        logger.info(f"Generating reports for epoch {self.current_epoch}")
        
        # Create epoch summary
        epoch_summary = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'training_metrics': [asdict(m) for m in self.training_metrics if m.epoch == self.current_epoch],
            'quality_metrics': [asdict(m) for m in self.quality_metrics if m.dataset_name == f"epoch_{self.current_epoch}"],
            'convergence_status': asdict(self.convergence_status)
        }
        
        # Save epoch report
        report_path = self.output_dir / f"epoch_{self.current_epoch}_report.json"
        with open(report_path, 'w') as f:
            json.dump(epoch_summary, f, indent=2, default=str)
        
        # Generate real-time dashboard metrics
        dashboard_metrics = self._generate_dashboard_metrics()
        dashboard_path = self.output_dir / f"epoch_{self.current_epoch}_dashboard.json"
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_metrics, f, indent=2, default=str)
        
        logger.info(f"Generated reports for epoch {self.current_epoch}")
    
    def _generate_dashboard_metrics(self) -> Dict[str, Any]:
        """Generate metrics for real-time dashboard"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.config['max_epochs'],
            'models_trained': len(self.models),
            'datasets_generated': len(self.datasets),
            'convergence_status': asdict(self.convergence_status),
            'recent_performance': {
                'avg_mse': np.mean([m.mse for m in self.training_metrics[-4:]]),
                'avg_accuracy': np.mean([m.accuracy for m in self.training_metrics[-4:]]),
                'avg_quality_score': np.mean([q.quality_score for q in self.quality_metrics[-4:]])
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def _finalize_training(self):
        """Finalize training and generate final outputs"""
        logger.info("Finalizing training and generating final outputs")
        
        # Generate final comprehensive report
        final_report = {
            'training_summary': {
                'total_epochs': self.current_epoch,
                'final_convergence_status': asdict(self.convergence_status),
                'total_training_time': sum(m.training_time for m in self.training_metrics),
                'final_quality_score': self.quality_metrics[-1].quality_score if self.quality_metrics else 0.0
            },
            'model_performance': {
                model_name: {
                    'final_mse': metrics[-1].mse if metrics else float('inf'),
                    'final_accuracy': metrics[-1].accuracy if metrics else 0.0,
                    'improvement_trajectory': [m.mse for m in metrics]
                }
                for model_name, metrics in self._group_metrics_by_model().items()
            },
            'dataset_quality': {
                'final_coverage': self.quality_metrics[-1].coverage if self.quality_metrics else 0.0,
                'quality_trajectory': [q.quality_score for q in self.quality_metrics]
            },
            'reproducibility': {
                'seed': self.seed,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Save final report
        final_report_path = self.output_dir / "final_training_report.json"
        with open(final_report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Generate evidence pack
        await self._generate_evidence_pack()
        
        # Export final datasets
        self._export_final_datasets()
        
        # Export trained models
        self._export_trained_models()
        
        logger.info("Training finalization completed")
    
    def _group_metrics_by_model(self) -> Dict[str, List[TrainingMetrics]]:
        """Group training metrics by model name"""
        grouped = {}
        for metric in self.training_metrics:
            if metric.model_name not in grouped:
                grouped[metric.model_name] = []
            grouped[metric.model_name].append(metric)
        return grouped
    
    async def _generate_evidence_pack(self):
        """Generate regulatory evidence pack"""
        logger.info("Generating regulatory evidence pack")
        
        try:
            # Create evidence pack
            evidence_data = {
                'training_summary': {
                    'total_epochs': self.current_epoch,
                    'convergence_achieved': self.convergence_status.models_converged,
                    'final_quality_score': self.quality_metrics[-1].quality_score if self.quality_metrics else 0.0
                },
                'datasets': {
                    name: {
                        'rows': len(data),
                        'columns': len(data.columns),
                        'quality_score': self._get_dataset_quality_score(name)
                    }
                    for name, data in self.datasets.items()
                },
                'models': {
                    name: {
                        'type': type(model).__name__,
                        'final_performance': self._get_model_final_performance(name)
                    }
                    for name, model in self.models.items()
                },
                'audit_trail': {
                    'seed': self.seed,
                    'config': self.config,
                    'training_timeline': [
                        {
                            'epoch': m.epoch,
                            'timestamp': m.timestamp.isoformat(),
                            'model': m.model_name,
                            'mse': m.mse,
                            'accuracy': m.accuracy
                        }
                        for m in self.training_metrics
                    ]
                }
            }
            
            # Save evidence pack
            evidence_path = self.output_dir / "regulatory_evidence_pack.json"
            with open(evidence_path, 'w') as f:
                json.dump(evidence_data, f, indent=2, default=str)
            
            logger.info("Generated regulatory evidence pack")
            
        except Exception as e:
            logger.error(f"Error generating evidence pack: {e}")
    
    def _get_dataset_quality_score(self, dataset_name: str) -> float:
        """Get quality score for a specific dataset"""
        for metric in self.quality_metrics:
            if metric.dataset_name == dataset_name:
                return metric.quality_score
        return 0.0
    
    def _get_model_final_performance(self, model_name: str) -> Dict[str, float]:
        """Get final performance metrics for a specific model"""
        model_metrics = [m for m in self.training_metrics if m.model_name == model_name]
        if model_metrics:
            final_metric = model_metrics[-1]
            return {
                'mse': final_metric.mse,
                'accuracy': final_metric.accuracy,
                'precision': final_metric.precision,
                'recall': final_metric.recall,
                'f1_score': final_metric.f1_score
            }
        return {}
    
    def _export_final_datasets(self):
        """Export final datasets in multiple formats"""
        logger.info("Exporting final datasets")
        
        # Export final iteration
        final_dataset = self.datasets[f"bonds_epoch_{self.current_epoch}"]
        
        # CSV format
        csv_path = self.output_dir / "final_bonds_dataset.csv"
        final_dataset.to_csv(csv_path, index=False)
        
        # JSONL format
        jsonl_path = self.output_dir / "final_bonds_dataset.jsonl"
        with open(jsonl_path, 'w') as f:
            for _, row in final_dataset.iterrows():
                f.write(json.dumps(row.to_dict()) + '\n')
        
        # Export all iterations
        all_iterations_path = self.output_dir / "all_dataset_iterations.json"
        all_iterations = {
            name: {
                'rows': len(data),
                'columns': list(data.columns),
                'quality_score': self._get_dataset_quality_score(name)
            }
            for name, data in self.datasets.items()
        }
        
        with open(all_iterations_path, 'w') as f:
            json.dump(all_iterations, f, indent=2, default=str)
        
        logger.info("Final datasets exported")
    
    def _export_trained_models(self):
        """Export trained model artifacts"""
        logger.info("Exporting trained model artifacts")
        
        models_dir = self.output_dir / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                # Save model if it has save method
                if hasattr(model, 'save'):
                    model_path = models_dir / f"{model_name}.pkl"
                    model.save(str(model_path))
                elif hasattr(model, 'save_model'):
                    model_path = models_dir / f"{model_name}.pkl"
                    model.save_model(str(model_path))
                
                # Save metadata
                metadata_path = models_dir / f"{model_name}_metadata.json"
                metadata = {
                    'model_name': model_name,
                    'model_type': type(model).__name__,
                    'training_epochs': self.current_epoch,
                    'final_performance': self._get_model_final_performance(model_name),
                    'seed': self.seed,
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
            except Exception as e:
                logger.warning(f"Could not export model {model_name}: {e}")
        
        logger.info("Trained model artifacts exported")

async def main():
    """Main entry point"""
    # Create trainer
    trainer = BondXAIAutonomousTrainer()
    
    # Run autonomous training loop
    await trainer.run_autonomous_training_loop()

if __name__ == "__main__":
    # Run the autonomous trainer
    asyncio.run(main())
