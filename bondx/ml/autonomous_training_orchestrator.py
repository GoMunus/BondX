#!/usr/bin/env python3
"""
Autonomous Training Orchestrator for BondX AI
Implements continuous training loop with quality validation and convergence monitoring.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from .spread_model import SpreadPredictionModel, ModelType
from .downgrade_model import DowngradePredictionModel
from .liquidity_shock_model import LiquidityShockModel
from .anomaly_detector import AnomalyDetector

class AutonomousTrainingOrchestrator:
    """
    Orchestrates continuous training of all ML models with quality validation
    until convergence is achieved.
    """
    
    def __init__(self, 
                 data_path: str = "data/synthetic/enhanced_bonds_150plus.csv",
                 output_dir: str = "bondx/ml/trained_models",
                 seed: int = 42,
                 max_iterations: int = 100,
                 convergence_threshold: float = 0.001,
                 quality_threshold: float = 0.90):
        """
        Initialize the autonomous training orchestrator.
        
        Args:
            data_path: Path to the synthetic dataset
            output_dir: Directory to save trained models
            seed: Random seed for reproducibility
            max_iterations: Maximum training iterations
            convergence_threshold: Threshold for convergence detection
            quality_threshold: Minimum quality gate pass rate
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        np.random.seed(seed)
        
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.quality_threshold = quality_threshold
        
        # Initialize models
        self.spread_model = SpreadPredictionModel(seed=seed, model_type=ModelType.XGBOOST)
        self.downgrade_model = DowngradePredictionModel(seed=seed)
        self.liquidity_shock_model = LiquidityShockModel(seed=seed)
        self.anomaly_detector = AnomalyDetector(seed=seed)
        
        # Training state
        self.current_iteration = 0
        self.training_history = []
        self.convergence_metrics = {}
        self.quality_gate_results = []
        
        # Setup logging
        self._setup_logging()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        logger.info(f"Autonomous Training Orchestrator initialized with {len(self.dataset)} bonds")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        global logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load and preprocess the synthetic dataset."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset: {df.shape}")
            
            # Basic preprocessing
            df = df.dropna()
            
            # Convert dates
            df['Issue_Date'] = pd.to_datetime(df['Issue_Date'])
            df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'])
            
            # Calculate time to maturity
            df['Time_to_Maturity'] = (df['Maturity_Date'] - df['Issue_Date']).dt.days / 365.25
            
            # Create target variables
            df['spread_bps'] = ((df['Face_Value'] - df['Market_Price']) / df['Face_Value']) * 10000
            df['downgrade_risk'] = np.where(df['Credit_Rating'].isin(['BB+', 'BB', 'BB-']), 1, 0)
            df['liquidity_shock_risk'] = np.where(df['Liquidity_Score'] < 40, 1, 0)
            
            logger.info(f"Preprocessed dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Prepare features for all models."""
        feature_sets = {}
        
        # Common features
        common_features = [
            'Coupon_Rate', 'Face_Value', 'Market_Price', 'Yield_to_Maturity',
            'Liquidity_Score', 'ESG_Score', 'Trading_Volume', 'Bid_Ask_Spread',
            'Market_Cap', 'Debt_to_Equity', 'Time_to_Maturity'
        ]
        
        # Rating encoding
        rating_mapping = {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10, 'BB+': 11, 'BB': 12, 'BB-': 13
        }
        df['rating_encoded'] = df['Credit_Rating'].map(rating_mapping)
        
        # Sector encoding
        sector_mapping = {
            'Technology': 1, 'Finance': 2, 'Energy': 3, 'Industrial': 4,
            'Consumer Goods': 5, 'Healthcare': 6
        }
        df['sector_encoded'] = df['Sector'].map(sector_mapping)
        
        # Add encoded features
        common_features.extend(['rating_encoded', 'sector_encoded'])
        
        # Create feature sets for each model
        feature_sets['spread'] = common_features
        feature_sets['downgrade'] = common_features + ['spread_bps']
        feature_sets['liquidity_shock'] = common_features + ['spread_bps']
        feature_sets['anomaly'] = common_features + ['spread_bps']
        
        # Ensure all features exist
        for model_name, features in feature_sets.items():
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features for {model_name}: {missing_features}")
                feature_sets[model_name] = [f for f in features if f in df.columns]
        
        return df, feature_sets
    
    def _train_spread_model(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Train the spread prediction model."""
        logger.info("Training spread prediction model...")
        
        X = df[features].values
        y = df['spread_bps'].values
        
        # Split data
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.spread_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.spread_model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
        
        logger.info(f"Spread model metrics: {metrics}")
        return metrics
    
    def _train_downgrade_model(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Train the downgrade prediction model."""
        logger.info("Training downgrade prediction model...")
        
        X = df[features].values
        y = df['downgrade_risk'].values
        
        # Split data
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.downgrade_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.downgrade_model.predict(X_test)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Downgrade model metrics: {metrics}")
        return metrics
    
    def _train_liquidity_shock_model(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Train the liquidity shock prediction model."""
        logger.info("Training liquidity shock prediction model...")
        
        X = df[features].values
        y = df['liquidity_shock_risk'].values
        
        # Split data
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.liquidity_shock_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.liquidity_shock_model.predict(X_test)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Liquidity shock model metrics: {metrics}")
        return metrics
    
    def _train_anomaly_detector(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Train the anomaly detection model."""
        logger.info("Training anomaly detection model...")
        
        X = df[features].values
        
        # Split data
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        # Train model
        self.anomaly_detector.fit(X_train)
        
        # Evaluate
        anomalies = self.anomaly_detector.predict(X_test)
        anomaly_rate = np.mean(anomalies)
        
        metrics = {
            'anomaly_rate': anomaly_rate,
            'anomaly_count': int(np.sum(anomalies))
        }
        
        logger.info(f"Anomaly detector metrics: {metrics}")
        return metrics
    
    def _validate_quality_gates(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that all models meet quality gate thresholds."""
        quality_results = {}
        
        # Spread model quality gates
        spread_metrics = metrics.get('spread', {})
        spread_quality = {
            'r2_threshold': spread_metrics.get('r2', 0) >= 0.7,
            'rmse_threshold': spread_metrics.get('rmse', float('inf')) <= 100,
            'overall': spread_metrics.get('r2', 0) >= 0.7 and spread_metrics.get('rmse', float('inf')) <= 100
        }
        quality_results['spread'] = spread_quality
        
        # Downgrade model quality gates
        downgrade_metrics = metrics.get('downgrade', {})
        downgrade_quality = {
            'accuracy_threshold': downgrade_metrics.get('accuracy', 0) >= 0.8,
            'f1_threshold': downgrade_metrics.get('f1', 0) >= 0.7,
            'overall': downgrade_metrics.get('accuracy', 0) >= 0.8 and downgrade_metrics.get('f1', 0) >= 0.7
        }
        quality_results['downgrade'] = downgrade_quality
        
        # Liquidity shock model quality gates
        liquidity_metrics = metrics.get('liquidity_shock', {})
        liquidity_quality = {
            'accuracy_threshold': liquidity_metrics.get('accuracy', 0) >= 0.8,
            'f1_threshold': liquidity_metrics.get('f1', 0) >= 0.7,
            'overall': liquidity_metrics.get('accuracy', 0) >= 0.8 and liquidity_metrics.get('f1', 0) >= 0.7
        }
        quality_results['liquidity_shock'] = liquidity_quality
        
        # Overall quality
        overall_quality = all([
            spread_quality['overall'],
            downgrade_quality['overall'],
            liquidity_quality['overall']
        ])
        
        quality_results['overall'] = overall_quality
        quality_results['pass_rate'] = sum([
            spread_quality['overall'],
            downgrade_quality['overall'],
            liquidity_quality['overall']
        ]) / 3.0
        
        return quality_results
    
    def _check_convergence(self, current_metrics: Dict[str, Dict[str, Any]], 
                          previous_metrics: Optional[Dict[str, Dict[str, Any]]]) -> bool:
        """Check if models have converged."""
        if previous_metrics is None:
            return False
        
        # Calculate improvement in key metrics
        improvements = {}
        
        for model_name in current_metrics:
            if model_name in previous_metrics:
                if model_name == 'spread':
                    current_r2 = current_metrics[model_name].get('r2', 0)
                    previous_r2 = previous_metrics[model_name].get('r2', 0)
                    improvements[model_name] = current_r2 - previous_r2
                elif model_name in ['downgrade', 'liquidity_shock']:
                    current_f1 = current_metrics[model_name].get('f1', 0)
                    previous_f1 = previous_metrics[model_name].get('f1', 0)
                    improvements[model_name] = current_f1 - previous_f1
        
        # Check if improvements are below threshold
        max_improvement = max(improvements.values()) if improvements else 0
        converged = max_improvement < self.convergence_threshold
        
        logger.info(f"Convergence check: max improvement = {max_improvement:.6f}, converged = {converged}")
        return converged
    
    def _save_models(self):
        """Save all trained models and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        model_files = {}
        
        # Save spread model
        spread_path = self.output_dir / f"spread_model_{timestamp}.pkl"
        self.spread_model.save_model(str(spread_path))
        model_files['spread'] = str(spread_path)
        
        # Save downgrade model
        downgrade_path = self.output_dir / f"downgrade_model_{timestamp}.pkl"
        self.downgrade_model.save_model(str(downgrade_path))
        model_files['downgrade'] = str(downgrade_path)
        
        # Save liquidity shock model
        liquidity_path = self.output_dir / f"liquidity_shock_model_{timestamp}.pkl"
        self.liquidity_shock_model.save_model(str(liquidity_path))
        model_files['liquidity_shock'] = str(liquidity_path)
        
        # Save anomaly detector
        anomaly_path = self.output_dir / f"anomaly_detector_{timestamp}.pkl"
        self.anomaly_detector.save_model(str(anomaly_path))
        model_files['anomaly'] = str(anomaly_path)
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'seed': self.seed,
            'iterations': self.current_iteration,
            'convergence_threshold': self.convergence_threshold,
            'quality_threshold': self.quality_threshold,
            'training_history': self.training_history,
            'convergence_metrics': self.convergence_metrics,
            'quality_gate_results': self.quality_gate_results,
            'model_files': model_files
        }
        
        metadata_path = self.output_dir / f"training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Models and metadata saved to {self.output_dir}")
        return model_files, metadata
    
    def _generate_interim_report(self, iteration: int, metrics: Dict[str, Dict[str, Any]], 
                                quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interim report for the current iteration."""
        report = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'model_performance': metrics,
            'quality_gates': quality_results,
            'overall_status': 'PASS' if quality_results['overall'] else 'FAIL',
            'convergence_status': 'CONVERGED' if iteration > 0 and self._check_convergence(
                metrics, self.training_history[-1]['metrics'] if self.training_history else None
            ) else 'TRAINING'
        }
        
        return report
    
    def run_autonomous_training_loop(self) -> Dict[str, Any]:
        """
        Run the autonomous training loop until convergence or max iterations.
        
        Returns:
            Dictionary containing final results and model paths
        """
        logger.info("Starting autonomous training loop...")
        start_time = time.time()
        
        # Prepare features
        df_features, feature_sets = self._prepare_features(self.dataset)
        
        previous_metrics = None
        
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            iteration_start = time.time()
            
            logger.info(f"\n=== Training Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Train all models
            metrics = {}
            
            try:
                metrics['spread'] = self._train_spread_model(df_features, feature_sets['spread'])
                metrics['downgrade'] = self._train_downgrade_model(df_features, feature_sets['downgrade'])
                metrics['liquidity_shock'] = self._train_liquidity_shock_model(df_features, feature_sets['liquidity_shock'])
                metrics['anomaly'] = self._train_anomaly_detector(df_features, feature_sets['anomaly'])
                
            except Exception as e:
                logger.error(f"Error in training iteration {iteration}: {e}")
                continue
            
            # Validate quality gates
            quality_results = self._validate_quality_gates(metrics)
            self.quality_gate_results.append(quality_results)
            
            # Generate interim report
            interim_report = self._generate_interim_report(iteration, metrics, quality_results)
            
            # Check convergence
            converged = self._check_convergence(metrics, previous_metrics)
            
            # Store training history
            training_record = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'quality_results': quality_results,
                'converged': converged,
                'duration': time.time() - iteration_start
            }
            self.training_history.append(training_record)
            
            # Log progress
            logger.info(f"Iteration {iteration + 1} completed:")
            logger.info(f"  Quality gates passed: {quality_results['pass_rate']:.2%}")
            logger.info(f"  Overall quality: {'PASS' if quality_results['overall'] else 'FAIL'}")
            logger.info(f"  Converged: {converged}")
            
            # Check if quality threshold is met
            if quality_results['pass_rate'] >= self.quality_threshold:
                logger.info(f"Quality threshold met: {quality_results['pass_rate']:.2%} >= {self.quality_threshold}")
                
                # Check convergence
                if converged:
                    logger.info("Models have converged! Training complete.")
                    break
            else:
                logger.warning(f"Quality threshold not met: {quality_results['pass_rate']:.2%} < {self.quality_threshold}")
            
            # Store metrics for next iteration
            previous_metrics = metrics
            
            # Save interim report
            report_path = self.output_dir / f"interim_report_{iteration:03d}.json"
            with open(report_path, 'w') as f:
                json.dump(interim_report, f, indent=2, default=str)
            
            # Hourly checkpoint
            if (iteration + 1) % 10 == 0:  # Every 10 iterations (approximately hourly)
                logger.info("Hourly checkpoint - saving interim models...")
                self._save_models()
        
        # Training loop complete
        total_duration = time.time() - start_time
        
        logger.info(f"\n=== Training Loop Complete ===")
        logger.info(f"Total iterations: {self.current_iteration + 1}")
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Final quality pass rate: {quality_results['pass_rate']:.2%}")
        logger.info(f"Converged: {converged}")
        
        # Save final models
        model_files, metadata = self._save_models()
        
        # Generate final report
        final_results = {
            'training_complete': True,
            'iterations': self.current_iteration + 1,
            'converged': converged,
            'quality_pass_rate': quality_results['pass_rate'],
            'total_duration': total_duration,
            'model_files': model_files,
            'metadata': metadata,
            'final_metrics': metrics,
            'final_quality': quality_results
        }
        
        # Save final results
        results_path = self.output_dir / "final_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_path}")
        
        return final_results

def main():
    """Main function to run the autonomous training loop."""
    # Initialize orchestrator
    orchestrator = AutonomousTrainingOrchestrator(
        data_path="data/synthetic/enhanced_bonds_150plus.csv",
        output_dir="bondx/ml/trained_models",
        seed=42,
        max_iterations=100,
        convergence_threshold=0.001,
        quality_threshold=0.90
    )
    
    # Run autonomous training loop
    results = orchestrator.run_autonomous_training_loop()
    
    print(f"\nAutonomous Training Complete!")
    print(f"Models saved to: {results['model_files']}")
    print(f"Final quality: {results['quality_pass_rate']:.2%}")
    print(f"Converged: {results['converged']}")

if __name__ == "__main__":
    main()
