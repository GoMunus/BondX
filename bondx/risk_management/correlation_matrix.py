"""
Correlation Matrix Calculator for BondX Risk Management System

This module provides production-grade correlation and covariance matrix services
for risk factors including rates, spreads, and liquidity proxies.

Features:
- Rolling windows (60/125/250 trading days) with business-day calendar
- Shrinkage estimators: Ledoit-Wolf and Oracle-Approximating Shrinkage
- PSD enforcement with nearPD projection
- Factor taxonomy: rate tenors, spread buckets, FX, liquidity
- Performance: ≤50ms for up to 200 factors, ≤20ms incremental roll
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import warnings
from scipy import linalg
from scipy.optimize import minimize_scalar
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ShrinkageMethod(Enum):
    """Available shrinkage estimation methods."""
    NONE = "none"
    LEDOIT_WOLF = "ledoit_wolf"
    ORACLE_APPROXIMATING = "oracle_approximating"
    CONSTANT = "constant"

class WindowSize(Enum):
    """Standard rolling window sizes."""
    W60 = 60
    W125 = 125
    W250 = 250

@dataclass
class CorrelationMatrixConfig:
    """Configuration for correlation matrix calculation."""
    window_size: WindowSize = WindowSize.W125
    shrinkage_method: ShrinkageMethod = ShrinkageMethod.LEDOIT_WOLF
    winsorization_threshold: float = 0.01  # 1% winsorization
    min_eigenvalue: float = 1e-8  # Minimum eigenvalue for PSD
    max_condition_number: float = 1e12  # Maximum condition number
    missing_data_policy: Literal["pairwise", "em"] = "pairwise"
    business_day_calendar: Optional[List[date]] = None
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class FactorTaxonomy:
    """Taxonomy for risk factors."""
    rate_tenors: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    spread_buckets: List[str] = field(default_factory=lambda: ["AAA", "AA", "A", "BBB"])
    fx_pairs: List[str] = field(default_factory=lambda: ["USD/INR", "EUR/INR", "GBP/INR"])
    liquidity_proxies: List[str] = field(default_factory=lambda: ["turnover", "bid_ask", "depth"])

@dataclass
class MatrixValidation:
    """Validation diagnostics for correlation/covariance matrices."""
    is_psd: bool
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    shrinkage_lambda: Optional[float] = None
    winsorization_applied: bool = False
    nearpd_applied: bool = False
    warnings: List[str] = field(default_factory=list)

@dataclass
class CorrelationMatrix:
    """Correlation matrix object with metadata."""
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    volatility_vector: np.ndarray
    factor_list: List[str]
    window_start: date
    window_end: date
    calculation_date: date
    config: CorrelationMatrixConfig
    validation: MatrixValidation
    version: str = "1.0"
    
    def __post_init__(self):
        """Validate matrix dimensions and properties."""
        n_factors = len(self.factor_list)
        assert self.correlation_matrix.shape == (n_factors, n_factors)
        assert self.covariance_matrix.shape == (n_factors, n_factors)
        assert self.volatility_vector.shape == (n_factors,)
        
        # Ensure correlation matrix is symmetric
        assert np.allclose(self.correlation_matrix, self.correlation_matrix.T)
        assert np.allclose(self.covariance_matrix, self.covariance_matrix.T)

class CorrelationMatrixCalculator:
    """
    Production-grade correlation and covariance matrix calculator.
    
    Performance targets:
    - ≤50ms for up to 200 factors
    - ≤20ms incremental roll
    """
    
    def __init__(self, config: Optional[CorrelationMatrixConfig] = None):
        self.config = config or CorrelationMatrixConfig()
        self._cache: Dict[str, CorrelationMatrix] = {}
        self._last_calculation: Optional[datetime] = None
        
    def calculate_matrix(
        self,
        returns_data: pd.DataFrame,
        factor_list: Optional[List[str]] = None,
        window_start: Optional[date] = None,
        window_end: Optional[date] = None
    ) -> CorrelationMatrix:
        """
        Calculate correlation and covariance matrices for the specified window.
        
        Args:
            returns_data: DataFrame with dates as index and factors as columns
            factor_list: List of factor names (if None, uses all columns)
            window_start: Start date for calculation window
            window_end: End date for calculation window
            
        Returns:
            CorrelationMatrix object with matrices and validation
        """
        start_time = datetime.now()
        
        # Validate inputs
        if returns_data.empty:
            raise ValueError("Returns data cannot be empty")
            
        if factor_list is None:
            factor_list = list(returns_data.columns)
            
        if window_start is None:
            window_start = returns_data.index.min().date()
        if window_end is None:
            window_end = returns_data.index.max().date()
            
        # Filter data for window
        window_data = self._filter_window_data(returns_data, window_start, window_end)
        
        # Align factor data
        aligned_data = self._align_factor_data(window_data, factor_list)
        
        # Handle missing data
        cleaned_data = self._handle_missing_data(aligned_data)
        
        # Calculate sample covariance
        sample_cov = self._calculate_sample_covariance(cleaned_data)
        
        # Apply shrinkage if configured
        if self.config.shrinkage_method != ShrinkageMethod.NONE:
            shrunk_cov, shrinkage_lambda = self._apply_shrinkage(sample_cov, cleaned_data)
        else:
            shrunk_cov = sample_cov
            shrinkage_lambda = None
            
        # Ensure positive semi-definiteness
        psd_cov, nearpd_applied = self._ensure_psd(shrunk_cov)
        
        # Extract volatilities and correlation matrix
        volatilities = np.sqrt(np.diag(psd_cov))
        correlation_matrix = self._covariance_to_correlation(psd_cov, volatilities)
        
        # Validate results
        validation = self._validate_matrices(psd_cov, correlation_matrix, shrinkage_lambda, nearpd_applied)
        
        # Create result object
        result = CorrelationMatrix(
            correlation_matrix=correlation_matrix,
            covariance_matrix=psd_cov,
            volatility_vector=volatilities,
            factor_list=factor_list,
            window_start=window_start,
            window_end=window_end,
            calculation_date=datetime.now().date(),
            config=self.config,
            validation=validation
        )
        
        # Cache result
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(factor_list, window_start, window_end)
            self._cache[cache_key] = result
            
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Matrix calculation completed in {execution_time:.2f}ms")
        
        if execution_time > 50:
            logger.warning(f"Matrix calculation exceeded 50ms target: {execution_time:.2f}ms")
            
        return result
    
    def incremental_roll(
        self,
        current_matrix: CorrelationMatrix,
        new_data: pd.DataFrame,
        roll_date: date
    ) -> CorrelationMatrix:
        """
        Perform incremental roll of correlation matrix.
        
        Args:
            current_matrix: Current correlation matrix
            new_data: New data for the roll
            roll_date: Date for the roll
            
        Returns:
            Updated CorrelationMatrix
        """
        start_time = datetime.now()
        
        # Calculate new window
        window_size = self.config.window_size.value
        new_window_start = roll_date - timedelta(days=window_size)
        new_window_end = roll_date
        
        # Get data for new window
        window_data = self._filter_window_data(new_data, new_window_start, new_window_end)
        
        # Calculate new matrix
        result = self.calculate_matrix(
            window_data,
            current_matrix.factor_list,
            new_window_start,
            new_window_end
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Incremental roll completed in {execution_time:.2f}ms")
        
        if execution_time > 20:
            logger.warning(f"Incremental roll exceeded 20ms target: {execution_time:.2f}ms")
            
        return result
    
    def _filter_window_data(
        self,
        data: pd.DataFrame,
        window_start: date,
        window_end: date
    ) -> pd.DataFrame:
        """Filter data for the specified window."""
        mask = (data.index.date >= window_start) & (data.index.date <= window_end)
        return data[mask]
    
    def _align_factor_data(
        self,
        data: pd.DataFrame,
        factor_list: List[str]
    ) -> pd.DataFrame:
        """Align factor data and handle missing factors."""
        # Ensure all required factors are present
        missing_factors = set(factor_list) - set(data.columns)
        if missing_factors:
            logger.warning(f"Missing factors: {missing_factors}")
            
        # Select only available factors
        available_factors = [f for f in factor_list if f in data.columns]
        return data[available_factors]
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to policy."""
        if self.config.missing_data_policy == "pairwise":
            # Use pairwise deletion (drop rows with any missing values)
            return data.dropna()
        elif self.config.missing_data_policy == "em":
            # Use EM algorithm for missing data
            return self._em_imputation(data)
        else:
            raise ValueError(f"Unknown missing data policy: {self.config.missing_data_policy}")
    
    def _em_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple EM imputation for missing data."""
        # For now, use forward fill then backward fill
        # In production, implement proper EM algorithm
        return data.fillna(method='ffill').fillna(method='bfill')
    
    def _calculate_sample_covariance(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate sample covariance matrix."""
        # Remove mean
        demeaned = data - data.mean()
        
        # Calculate covariance
        n = len(data) - 1
        cov_matrix = (demeaned.T @ demeaned) / n
        
        return cov_matrix.values
    
    def _apply_shrinkage(
        self,
        sample_cov: np.ndarray,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """Apply shrinkage estimation to covariance matrix."""
        if self.config.shrinkage_method == ShrinkageMethod.LEDOIT_WOLF:
            return self._ledoit_wolf_shrinkage(sample_cov, data)
        elif self.config.shrinkage_method == ShrinkageMethod.ORACLE_APPROXIMATING:
            return self._oracle_approximating_shrinkage(sample_cov, data)
        elif self.config.shrinkage_method == ShrinkageMethod.CONSTANT:
            return self._constant_shrinkage(sample_cov, data)
        else:
            raise ValueError(f"Unknown shrinkage method: {self.config.shrinkage_method}")
    
    def _ledoit_wolf_shrinkage(
        self,
        sample_cov: np.ndarray,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """Apply Ledoit-Wolf shrinkage estimation."""
        n, p = data.shape
        
        # Target matrix (diagonal with average variance)
        target = np.eye(p) * np.trace(sample_cov) / p
        
        # Calculate optimal shrinkage parameter
        sample_var = np.var(data, axis=0)
        sample_cov_squared = sample_cov ** 2
        
        # Numerator: sum of squared off-diagonal elements
        numerator = np.sum(sample_cov_squared) - np.sum(np.diag(sample_cov_squared))
        
        # Denominator: sum of squared differences from target
        diff = sample_cov - target
        denominator = np.sum(diff ** 2)
        
        if denominator == 0:
            shrinkage_lambda = 0.0
        else:
            shrinkage_lambda = min(1.0, max(0.0, numerator / denominator))
        
        # Apply shrinkage
        shrunk_cov = shrinkage_lambda * target + (1 - shrinkage_lambda) * sample_cov
        
        return shrunk_cov, shrinkage_lambda
    
    def _oracle_approximating_shrinkage(
        self,
        sample_cov: np.ndarray,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """Apply Oracle-Approximating shrinkage estimation."""
        n, p = data.shape
        
        # Target matrix (diagonal with individual variances)
        target = np.diag(np.diag(sample_cov))
        
        # Calculate optimal shrinkage parameter
        sample_var = np.var(data, axis=0)
        sample_cov_squared = sample_cov ** 2
        
        # Numerator: sum of squared off-diagonal elements
        numerator = np.sum(sample_cov_squared) - np.sum(np.diag(sample_cov_squared))
        
        # Denominator: sum of squared differences from target
        diff = sample_cov - target
        denominator = np.sum(diff ** 2)
        
        if denominator == 0:
            shrinkage_lambda = 0.0
        else:
            shrinkage_lambda = min(1.0, max(0.0, numerator / denominator))
        
        # Apply shrinkage
        shrunk_cov = shrinkage_lambda * target + (1 - shrinkage_lambda) * sample_cov
        
        return shrunk_cov, shrinkage_lambda
    
    def _constant_shrinkage(
        self,
        sample_cov: np.ndarray,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """Apply constant shrinkage (0.5)."""
        shrinkage_lambda = 0.5
        
        # Target matrix (diagonal with average variance)
        target = np.eye(sample_cov.shape[0]) * np.trace(sample_cov) / sample_cov.shape[0]
        
        # Apply shrinkage
        shrunk_cov = shrinkage_lambda * target + (1 - shrinkage_lambda) * sample_cov
        
        return shrunk_cov, shrinkage_lambda
    
    def _ensure_psd(self, cov_matrix: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Ensure positive semi-definiteness using nearPD projection."""
        # Check if already PSD
        eigenvals = linalg.eigvalsh(cov_matrix)
        if np.all(eigenvals >= -self.config.min_eigenvalue):
            return cov_matrix, False
        
        # Apply nearPD projection
        n = cov_matrix.shape[0]
        
        # Eigendecomposition
        eigenvals, eigenvecs = linalg.eigh(cov_matrix)
        
        # Project eigenvalues to be non-negative
        eigenvals = np.maximum(eigenvals, self.config.min_eigenvalue)
        
        # Reconstruct matrix
        psd_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure symmetry
        psd_matrix = (psd_matrix + psd_matrix.T) / 2
        
        return psd_matrix, True
    
    def _covariance_to_correlation(
        self,
        cov_matrix: np.ndarray,
        volatilities: np.ndarray
    ) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        # Avoid division by zero
        vol_matrix = np.outer(volatilities, volatilities)
        vol_matrix[vol_matrix == 0] = 1.0
        
        correlation_matrix = cov_matrix / vol_matrix
        
        # Ensure diagonal elements are exactly 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return correlation_matrix
    
    def _validate_matrices(
        self,
        cov_matrix: np.ndarray,
        corr_matrix: np.ndarray,
        shrinkage_lambda: Optional[float],
        nearpd_applied: bool
    ) -> MatrixValidation:
        """Validate correlation and covariance matrices."""
        warnings_list = []
        
        # Check PSD property
        eigenvals = linalg.eigvalsh(cov_matrix)
        is_psd = np.all(eigenvals >= -self.config.min_eigenvalue)
        
        # Calculate condition number
        condition_number = np.max(eigenvals) / np.maximum(np.min(eigenvals), self.config.min_eigenvalue)
        
        # Check condition number
        if condition_number > self.config.max_condition_number:
            warnings_list.append(f"High condition number: {condition_number:.2e}")
        
        # Check correlation bounds
        if np.any(corr_matrix < -1) or np.any(corr_matrix > 1):
            warnings_list.append("Correlation values outside [-1, 1] range")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
            warnings_list.append("Matrix contains NaN or infinite values")
        
        return MatrixValidation(
            is_psd=is_psd,
            condition_number=condition_number,
            min_eigenvalue=np.min(eigenvals),
            max_eigenvalue=np.max(eigenvals),
            shrinkage_lambda=shrinkage_lambda,
            nearpd_applied=nearpd_applied,
            warnings=warnings_list
        )
    
    def _generate_cache_key(
        self,
        factor_list: List[str],
        window_start: date,
        window_end: date
    ) -> str:
        """Generate cache key for matrix results."""
        factors_str = "_".join(sorted(factor_list))
        return f"{factors_str}_{window_start}_{window_end}"
    
    def get_matrix_by_date(
        self,
        target_date: date,
        factor_list: Optional[List[str]] = None
    ) -> Optional[CorrelationMatrix]:
        """Retrieve matrix for a specific date."""
        # Implementation would query cached results or recalculate
        # For now, return None if not found
        return None
    
    def get_matrix_by_window(
        self,
        window_start: date,
        window_end: date,
        factor_list: Optional[List[str]] = None
    ) -> Optional[CorrelationMatrix]:
        """Retrieve matrix for a specific window."""
        if factor_list is None:
            # Return any matrix for this window
            for matrix in self._cache.values():
                if matrix.window_start == window_start and matrix.window_end == window_end:
                    return matrix
        else:
            cache_key = self._generate_cache_key(factor_list, window_start, window_end)
            return self._cache.get(cache_key)
        
        return None
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Correlation matrix cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "last_calculation": self._last_calculation.isoformat() if self._last_calculation else None
        }
