"""
Anomaly Detector

This module provides anomaly detection capabilities including:
- Isolation Forest for outlier detection
- Robust z-scores for statistical anomaly detection
- Multi-dimensional anomaly scoring
- Confidence scoring and ranking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score

# Statistical imports
from scipy import stats
from scipy.stats import zscore, iqr

logger = logging.getLogger(__name__)

class AnomalyMethod(Enum):
    """Available anomaly detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    DBSCAN = "dbscan"
    ZSCORE = "zscore"
    IQR = "iqr"
    MAHALANOBIS = "mahalanobis"

class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    issuer_name: str
    sector: str
    credit_rating: str
    anomaly_score: float
    is_anomaly: bool
    severity: AnomalySeverity
    confidence: float
    contributing_factors: List[str]
    method: str
    model_metadata: Dict[str, Any]

class AnomalyDetector:
    """Multi-method anomaly detector for bond data"""
    
    def __init__(self, seed: int = 42, method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST):
        """Initialize the anomaly detector"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.method = method
        
        # Models
        self.isolation_forest = None
        self.local_outlier_factor = None
        self.elliptic_envelope = None
        self.dbscan = None
        
        # Preprocessing
        self.scaler = None
        self.pca = None
        
        # Performance tracking
        self.detection_history = []
        self.feature_importance = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the anomaly detection model based on method"""
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=self.seed,
                n_estimators=100
            )
        elif self.method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
            self.local_outlier_factor = LocalOutlierFactor(
                contamination=0.1,
                n_neighbors=20,
                metric='manhattan'
            )
        elif self.method == AnomalyMethod.ELLIPTIC_ENVELOPE:
            self.elliptic_envelope = EllipticEnvelope(
                contamination=0.1,
                random_state=self.seed
            )
        elif self.method == AnomalyMethod.DBSCAN:
            self.dbscan = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='manhattan'
            )
        elif self.method == AnomalyMethod.ZSCORE:
            # Z-score is statistical, no model needed
            pass
        elif self.method == AnomalyMethod.IQR:
            # IQR is statistical, no model needed
            pass
        elif self.method == AnomalyMethod.MAHALANOBIS:
            # Mahalanobis is statistical, no model needed
            pass
        else:
            raise ValueError(f"Unsupported anomaly detection method: {self.method}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for anomaly detection
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Ensure liquidity index exists
        if 'liquidity_index' not in df_features.columns:
            from ..analysis.liquidity_insights import LiquidityInsightsEngine
            liq_engine = LiquidityInsightsEngine(seed=self.seed)
            df_features['liquidity_index'] = liq_engine.calculate_liquidity_index(df_features)
        
        # Ensure ESG scores exist
        if 'esg_composite' not in df_features.columns:
            from ..analysis.heatmaps import SectorHeatmapEngine
            heatmap_engine = SectorHeatmapEngine(seed=self.seed)
            df_features = heatmap_engine.calculate_esg_scores(df_features)
        
        # Basic numerical features
        numerical_features = [
            'yield_to_maturity', 'volume_traded', 'bid_ask_spread',
            'time_to_maturity', 'coupon_rate', 'issue_size',
            'liquidity_index', 'esg_composite', 'spread_bps'
        ]
        
        # Create derived features
        df_features['spread_to_yield_ratio'] = df_features['spread_bps'] / (df_features['yield_to_maturity'] + 0.01)
        df_features['volume_to_size_ratio'] = df_features['volume_traded'] / (df_features['issue_size'] + 0.01)
        df_features['liquidity_spread_ratio'] = df_features['liquidity_index'] / (df_features['spread_bps'] + 0.01)
        
        # Create market stress indicators
        df_features['market_stress'] = (
            (df_features['spread_bps'] > df_features['spread_bps'].quantile(0.8)).astype(int) +
            (df_features['bid_ask_spread'] > df_features['bid_ask_spread'].quantile(0.8)).astype(int) +
            (df_features['liquidity_index'] < df_features['liquidity_index'].quantile(0.2)).astype(int)
        )
        
        # Create sector-specific features
        sector_volatility_map = {
            'Energy': 0.8, 'Mining': 0.9, 'Real Estate': 0.7,
            'Financial': 0.6, 'Technology': 0.4, 'Healthcare': 0.3,
            'Consumer Goods': 0.3, 'Utilities': 0.2, 'Construction': 0.6,
            'Automotive': 0.5, 'Infrastructure': 0.5, 'Telecommunications': 0.4
        }
        
        df_features['sector_volatility'] = df_features['sector'].map(sector_volatility_map).fillna(0.5)
        
        # Create rating-specific features
        rating_volatility_map = {
            'AAA': 0.1, 'AA+': 0.15, 'AA': 0.2, 'AA-': 0.25,
            'A+': 0.3, 'A': 0.35, 'A-': 0.4, 'BBB+': 0.45,
            'BBB': 0.5, 'BBB-': 0.55, 'BB+': 0.6, 'BB': 0.65,
            'BB-': 0.7, 'B+': 0.75, 'B': 0.8, 'B-': 0.85,
            'CCC': 0.9, 'CC': 0.95, 'C': 1.0
        }
        
        df_features['rating_volatility'] = df_features['credit_rating'].map(rating_volatility_map).fillna(0.5)
        
        # Create interaction features
        df_features['liquidity_sector_volatility'] = df_features['liquidity_index'] * df_features['sector_volatility']
        df_features['spread_sector_volatility'] = df_features['spread_bps'] * df_features['sector_volatility']
        df_features['rating_sector_volatility'] = df_features['rating_volatility'] * df_features['sector_volatility']
        
        # Create polynomial features
        df_features['liquidity_squared'] = df_features['liquidity_index'] ** 2
        df_features['spread_squared'] = df_features['spread_bps'] ** 2
        df_features['volume_squared'] = df_features['volume_traded'] ** 2
        
        # Create market depth proxy
        df_features['market_depth'] = df_features['volume_traded'] / (df_features['bid_ask_spread'] + 0.01)
        df_features['market_depth'] = df_features['market_depth'].replace([np.inf, -np.inf], 0)
        
        # Create credit quality composite
        df_features['credit_quality'] = (
            df_features['esg_composite'] * 0.4 +
            df_features['liquidity_index'] * 0.3 +
            (1 - df_features['spread_bps'] / df_features['spread_bps'].max()) * 0.3
        )
        
        # Handle missing values
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numerical_cols] = df_features[numerical_cols].fillna(df_features[numerical_cols].median())
        
        return df_features
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for anomaly detection
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (feature DataFrame, feature column names)
        """
        # Select numerical features for modeling
        feature_candidates = [
            'yield_to_maturity', 'volume_traded', 'bid_ask_spread',
            'time_to_maturity', 'coupon_rate', 'issue_size',
            'liquidity_index', 'esg_composite', 'spread_bps',
            'spread_to_yield_ratio', 'volume_to_size_ratio', 'liquidity_spread_ratio',
            'market_stress', 'sector_volatility', 'rating_volatility',
            'liquidity_sector_volatility', 'spread_sector_volatility',
            'rating_sector_volatility', 'liquidity_squared', 'spread_squared',
            'volume_squared', 'market_depth', 'credit_quality'
        ]
        
        # Filter to available features
        available_features = [col for col in feature_candidates if col in df.columns]
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Remove rows with missing values
        valid_mask = ~X.isnull().any(axis=1)
        X = X[valid_mask]
        
        return X, available_features
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[AnomalyResult]:
        """
        Detect anomalies in bond data
        
        Args:
            df: DataFrame with bond data
            
        Returns:
            List of anomaly detection results
        """
        self.logger.info(f"Starting anomaly detection with {self.method.value}")
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Prepare features
        X, feature_names = self.prepare_features(df_features)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        
        # Apply anomaly detection
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            results = self._detect_with_isolation_forest(X, df_features, feature_names)
        elif self.method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
            results = self._detect_with_lof(X, df_features, feature_names)
        elif self.method == AnomalyMethod.ELLIPTIC_ENVELOPE:
            results = self._detect_with_elliptic_envelope(X, df_features, feature_names)
        elif self.method == AnomalyMethod.DBSCAN:
            results = self._detect_with_dbscan(X, df_features, feature_names)
        elif self.method == AnomalyMethod.ZSCORE:
            results = self._detect_with_zscore(X, df_features, feature_names)
        elif self.method == AnomalyMethod.IQR:
            results = self._detect_with_iqr(X, df_features, feature_names)
        elif self.method == AnomalyMethod.MAHALANOBIS:
            results = self._detect_with_mahalanobis(X, df_features, feature_names)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Store detection history
        detection_summary = {
            'method': self.method.value,
            'total_bonds': len(df),
            'anomalies_detected': len([r for r in results if r.is_anomaly]),
            'detection_date': datetime.now().isoformat()
        }
        
        self.detection_history.append(detection_summary)
        
        self.logger.info(f"Anomaly detection completed. Found {detection_summary['anomalies_detected']} anomalies")
        
        return results
    
    def _detect_with_isolation_forest(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                                    feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using Isolation Forest"""
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and predict
        self.isolation_forest.fit(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)
        
        # Convert predictions: -1 for anomaly, 1 for normal
        is_anomaly = predictions == -1
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "Isolation Forest"
        )
    
    def _detect_with_lof(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                         feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using Local Outlier Factor"""
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and predict
        lof_scores = self.local_outlier_factor.fit_predict(X_scaled)
        anomaly_scores = self.local_outlier_factor.negative_outlier_factor_
        
        # Convert predictions: -1 for anomaly, 1 for normal
        is_anomaly = lof_scores == -1
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "Local Outlier Factor"
        )
    
    def _detect_with_elliptic_envelope(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                                      feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using Elliptic Envelope"""
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and predict
        self.elliptic_envelope.fit(X_scaled)
        anomaly_scores = self.elliptic_envelope.score_samples(X_scaled)
        predictions = self.elliptic_envelope.predict(X_scaled)
        
        # Convert predictions: -1 for anomaly, 1 for normal
        is_anomaly = predictions == -1
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "Elliptic Envelope"
        )
    
    def _detect_with_dbscan(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                           feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using DBSCAN"""
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and predict
        cluster_labels = self.dbscan.fit_predict(X_scaled)
        
        # DBSCAN: -1 indicates noise (anomaly), other values are cluster labels
        is_anomaly = cluster_labels == -1
        
        # Calculate anomaly scores based on distance to nearest cluster
        anomaly_scores = np.zeros(len(X))
        for i in range(len(X)):
            if cluster_labels[i] == -1:
                # Calculate distance to nearest cluster
                distances = []
                for j in range(len(X)):
                    if cluster_labels[j] != -1:
                        dist = np.linalg.norm(X_scaled[i] - X_scaled[j])
                        distances.append(dist)
                
                if distances:
                    anomaly_scores[i] = -min(distances)  # Negative for anomalies
                else:
                    anomaly_scores[i] = -1.0
            else:
                anomaly_scores[i] = 0.0
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "DBSCAN"
        )
    
    def _detect_with_zscore(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                           feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using Z-score method"""
        # Calculate z-scores for each feature
        z_scores = np.zeros_like(X.values)
        for i, col in enumerate(X.columns):
            z_scores[:, i] = np.abs(zscore(X[col]))
        
        # Calculate composite anomaly score (max z-score across features)
        anomaly_scores = np.max(z_scores, axis=1)
        
        # Define threshold (e.g., 3 standard deviations)
        threshold = 3.0
        is_anomaly = anomaly_scores > threshold
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "Z-Score"
        )
    
    def _detect_with_iqr(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                         feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using IQR method"""
        # Calculate IQR-based anomaly scores for each feature
        iqr_scores = np.zeros_like(X.values)
        for i, col in enumerate(X.columns):
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate distance from Q1 and Q3 in IQR units
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Calculate anomaly score
            for j in range(len(X)):
                value = X.iloc[j, i]
                if value < lower_bound:
                    iqr_scores[j, i] = (lower_bound - value) / IQR
                elif value > upper_bound:
                    iqr_scores[j, i] = (value - upper_bound) / IQR
                else:
                    iqr_scores[j, i] = 0
        
        # Calculate composite anomaly score (max IQR score across features)
        anomaly_scores = np.max(iqr_scores, axis=1)
        
        # Define threshold
        threshold = 1.5
        is_anomaly = anomaly_scores > threshold
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "IQR"
        )
    
    def _detect_with_mahalanobis(self, X: pd.DataFrame, df_features: pd.DataFrame, 
                                 feature_names: List[str]) -> List[AnomalyResult]:
        """Detect anomalies using Mahalanobis distance"""
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_scaled.T)
        
        # Calculate Mahalanobis distance for each point
        anomaly_scores = np.zeros(len(X))
        for i in range(len(X)):
            try:
                # Calculate Mahalanobis distance
                diff = X_scaled[i] - np.mean(X_scaled, axis=0)
                inv_cov = np.linalg.inv(cov_matrix)
                mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                anomaly_scores[i] = mahal_dist
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                anomaly_scores[i] = 0
        
        # Define threshold (e.g., 95th percentile)
        threshold = np.percentile(anomaly_scores, 95)
        is_anomaly = anomaly_scores > threshold
        
        return self._create_anomaly_results(
            df_features, feature_names, anomaly_scores, is_anomaly, "Mahalanobis"
        )
    
    def _create_anomaly_results(self, df_features: pd.DataFrame, feature_names: List[str],
                               anomaly_scores: np.ndarray, is_anomaly: np.ndarray,
                               method_name: str) -> List[AnomalyResult]:
        """Create anomaly result objects"""
        results = []
        
        for i, (idx, row) in enumerate(df_features.iterrows()):
            # Get original data row
            orig_row = df_features.loc[idx]
            
            # Determine severity
            score = anomaly_scores[i]
            if score > np.percentile(anomaly_scores, 95):
                severity = AnomalySeverity.CRITICAL
            elif score > np.percentile(anomaly_scores, 90):
                severity = AnomalySeverity.HIGH
            elif score > np.percentile(anomaly_scores, 80):
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            # Calculate confidence based on score magnitude
            confidence = min(0.95, max(0.1, abs(score) / np.max(np.abs(anomaly_scores))))
            
            # Get contributing factors
            contributing_factors = self._get_contributing_factors(
                df_features.iloc[i], feature_names, score
            )
            
            result = AnomalyResult(
                issuer_name=orig_row['issuer_name'],
                sector=orig_row['sector'],
                credit_rating=orig_row['credit_rating'],
                anomaly_score=float(score),
                is_anomaly=bool(is_anomaly[i]),
                severity=severity,
                confidence=confidence,
                contributing_factors=contributing_factors,
                method=method_name,
                model_metadata={
                    'method': method_name,
                    'detection_date': datetime.now().isoformat(),
                    'total_features': len(feature_names)
                }
            )
            
            results.append(result)
        
        return results
    
    def _get_contributing_factors(self, features: pd.Series, feature_names: List[str], 
                                 anomaly_score: float) -> List[str]:
        """Get contributing factors for an anomaly"""
        # Simple approach: identify features with extreme values
        contributing_factors = []
        
        for feature in feature_names:
            if feature in features.index:
                value = features[feature]
                
                # Check if value is extreme (simplified logic)
                if abs(value) > 1000:  # High absolute values
                    contributing_factors.append(f"{feature}: {value:.2f}")
                elif feature in ['liquidity_index', 'esg_composite'] and value < 0.3:
                    contributing_factors.append(f"{feature}: {value:.2f}")
                elif feature == 'spread_bps' and value > 400:
                    contributing_factors.append(f"{feature}: {value:.2f}")
        
        # Limit to top factors
        return contributing_factors[:5]
    
    def save_model(self, filepath: str) -> None:
        """Save the trained anomaly detection model"""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save models based on method
        if self.method == AnomalyMethod.ISOLATION_FOREST and self.isolation_forest:
            model_path = f"{filepath}_isolation_forest.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.isolation_forest, f)
        
        elif self.method == AnomalyMethod.LOCAL_OUTLIER_FACTOR and self.local_outlier_factor:
            model_path = f"{filepath}_lof.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.local_outlier_factor, f)
        
        elif self.method == AnomalyMethod.ELLIPTIC_ENVELOPE and self.elliptic_envelope:
            model_path = f"{filepath}_elliptic_envelope.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.elliptic_envelope, f)
        
        elif self.method == AnomalyMethod.DBSCAN and self.dbscan:
            model_path = f"{filepath}_dbscan.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.dbscan, f)
        
        # Save metadata
        metadata_path = f"{filepath}_metadata.json"
        metadata = {
            'method': self.method.value,
            'detection_history': self.detection_history,
            'seed': self.seed
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained anomaly detection model"""
        # Load models based on method
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            model_path = f"{filepath}_isolation_forest.pkl"
            with open(model_path, 'rb') as f:
                self.isolation_forest = pickle.load(f)
        
        elif self.method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
            model_path = f"{filepath}_lof.pkl"
            with open(model_path, 'rb') as f:
                self.local_outlier_factor = pickle.load(f)
        
        elif self.method == AnomalyMethod.ELLIPTIC_ENVELOPE:
            model_path = f"{filepath}_elliptic_envelope.pkl"
            with open(model_path, 'rb') as f:
                self.elliptic_envelope = pickle.load(f)
        
        elif self.method == AnomalyMethod.DBSCAN:
            model_path = f"{filepath}_dbscan.pkl"
            with open(model_path, 'rb') as f:
                self.dbscan = pickle.load(f)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Update instance variables
        self.detection_history = metadata_dict.get('detection_history', [])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def evaluate_detection(self, df: pd.DataFrame, known_anomalies: Optional[List[int]] = None) -> Dict[str, Any]:
        """Evaluate anomaly detection performance"""
        # Run detection
        results = self.detect_anomalies(df)
        
        # If we have known anomalies, calculate metrics
        if known_anomalies:
            # Create ground truth labels
            y_true = np.zeros(len(df))
            y_true[known_anomalies] = 1
            
            # Create predicted labels
            y_pred = np.array([1 if r.is_anomaly else 0 for r in results])
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_anomalies': len(known_anomalies),
                'detected_anomalies': np.sum(y_pred),
                'true_positives': np.sum((y_true == 1) & (y_pred == 1)),
                'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
                'false_negatives': np.sum((y_true == 1) & (y_pred == 0))
            }
        else:
            metrics = {
                'total_bonds': len(df),
                'anomalies_detected': len([r for r in results if r.is_anomaly]),
                'anomaly_rate': len([r for r in results if r.is_anomaly]) / len(df)
            }
        
        # Add severity breakdown
        severity_counts = {}
        for severity in AnomalySeverity:
            count = len([r for r in results if r.severity == severity])
            severity_counts[severity.value] = count
        
        metrics['severity_breakdown'] = severity_counts
        
        # Add method information
        metrics['method'] = self.method.value
        metrics['evaluation_date'] = datetime.now().isoformat()
        
        return metrics
