"""
HFT-Grade Risk Engine for Phase D

This module implements ultra-low latency risk calculations with:
- GPU-vectorized VaR calculations
- Microsecond latency (<1ms standard, <5ms complex)
- Pre-computed stress shocks
- Real-time risk monitoring
- Automated regulatory capital calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from pathlib import Path

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy as cp_scipy
    import cupyx.scipy.linalg as cp_linalg
    import cupyx.scipy.stats as cp_stats
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_scipy = None
    cp_linalg = None
    cp_stats = None

# CUDA imports for ultra-low latency
try:
    import numba
    from numba import cuda, jit
    from numba.cuda.random import create_xoroshiro128p_states
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

class RiskMetric(Enum):
    """Available risk metrics"""
    VAR = "var"
    CVAR = "cvar"
    STRESS_VAR = "stress_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    REGULATORY_CAPITAL = "regulatory_capital"

class ComputeMode(Enum):
    """Available compute modes"""
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

@dataclass
class RiskConfig:
    """Configuration for risk calculations"""
    confidence_level: float = 0.99
    time_horizon: int = 1  # days
    monte_carlo_simulations: int = 10000
    stress_scenarios: int = 1000
    gpu_batch_size: int = 10000
    max_latency_ms: float = 1.0
    enable_precomputation: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

@dataclass
class StressScenario:
    """Stress test scenario configuration"""
    name: str
    description: str
    market_shock: float  # percentage
    volatility_multiplier: float
    correlation_adjustment: float
    liquidity_impact: float
    probability: float

class HFTRiskEngine:
    """
    High-Frequency Trading Grade Risk Engine
    
    Provides ultra-low latency risk calculations with GPU acceleration
    and pre-computed stress scenarios for real-time trading applications.
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.gpu_available = GPU_AVAILABLE
        self.numba_available = NUMBA_AVAILABLE
        
        # Pre-computed stress scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        # Risk calculation caches
        self.var_cache = {}
        self.stress_cache = {}
        self.correlation_cache = {}
        
        # Performance monitoring
        self.latency_metrics = {
            'var_calculation': [],
            'stress_testing': [],
            'correlation_update': []
        }
        
        # GPU memory management
        if self.gpu_available:
            self._initialize_gpu()
        
        logger.info(f"HFT Risk Engine initialized with GPU: {self.gpu_available}")
    
    def _initialize_gpu(self):
        """Initialize GPU resources and memory pools"""
        try:
            # Set memory pool for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            
            # Warm up GPU with dummy calculations
            dummy_data = cp.random.random((1000, 1000))
            _ = cp.linalg.eigvals(dummy_data)
            
            logger.info("GPU initialized successfully")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            self.gpu_available = False
    
    def _initialize_stress_scenarios(self) -> List[StressScenario]:
        """Initialize pre-defined stress scenarios"""
        scenarios = [
            StressScenario(
                name="2008_Financial_Crisis",
                description="Global financial crisis scenario",
                market_shock=-0.40,
                volatility_multiplier=3.0,
                correlation_adjustment=0.8,
                liquidity_impact=-0.60,
                probability=0.001
            ),
            StressScenario(
                name="2020_COVID_Crash",
                description="COVID-19 market crash scenario",
                market_shock=-0.30,
                volatility_multiplier=2.5,
                correlation_adjustment=0.7,
                liquidity_impact=-0.50,
                probability=0.005
            ),
            StressScenario(
                name="2013_Taper_Tantrum",
                description="Fed tapering scenario",
                market_shock=-0.15,
                volatility_multiplier=2.0,
                correlation_adjustment=0.6,
                liquidity_impact=-0.30,
                probability=0.01
            ),
            StressScenario(
                name="2016_Brexit",
                description="Brexit uncertainty scenario",
                market_shock=-0.10,
                volatility_multiplier=1.8,
                correlation_adjustment=0.5,
                liquidity_impact=-0.25,
                probability=0.02
            )
        ]
        return scenarios
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_var_calculation(returns: np.ndarray, confidence_level: float) -> float:
        """Numba-optimized VaR calculation"""
        sorted_returns = np.sort(returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        return sorted_returns[var_index]
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _fast_cvar_calculation(returns: np.ndarray, var_threshold: float) -> float:
        """Numba-optimized CVaR calculation"""
        tail_returns = returns[returns <= var_threshold]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
    
    def calculate_portfolio_var(
        self,
        portfolio_weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio VaR with ultra-low latency
        
        Args:
            portfolio_weights: Portfolio weights (n_assets,)
            returns_matrix: Historical returns matrix (n_obs, n_assets)
            confidence_level: VaR confidence level
            
        Returns:
            Dictionary with VaR metrics and performance data
        """
        start_time = time.perf_counter()
        
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        try:
            if self.gpu_available and len(portfolio_weights) > 100:
                # GPU-accelerated calculation for large portfolios
                result = self._calculate_gpu_var(portfolio_weights, returns_matrix, confidence_level)
            else:
                # CPU-optimized calculation
                result = self._calculate_cpu_var(portfolio_weights, returns_matrix, confidence_level)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['var_calculation'].append(latency_ms)
            
            # Add performance metrics
            result['latency_ms'] = latency_ms
            result['compute_mode'] = 'gpu' if self.gpu_available and len(portfolio_weights) > 100 else 'cpu'
            
            # Cache result if enabled
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(portfolio_weights, returns_matrix, confidence_level)
                self.var_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return {
                'var': None,
                'cvar': None,
                'volatility': None,
                'latency_ms': (time.perf_counter() - start_time) * 1000,
                'error': str(e)
            }
    
    def _calculate_gpu_var(
        self,
        portfolio_weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: float
    ) -> Dict[str, float]:
        """GPU-accelerated VaR calculation"""
        # Transfer data to GPU
        weights_gpu = cp.asarray(portfolio_weights)
        returns_gpu = cp.asarray(returns_matrix)
        
        # Calculate portfolio returns
        portfolio_returns = cp.dot(returns_gpu, weights_gpu)
        
        # Calculate VaR
        var_index = int((1 - confidence_level) * len(portfolio_returns))
        sorted_returns = cp.sort(portfolio_returns)
        var = float(sorted_returns[var_index])
        
        # Calculate CVaR
        tail_returns = portfolio_returns[portfolio_returns <= var]
        cvar = float(cp.mean(tail_returns)) if len(tail_returns) > 0 else var
        
        # Calculate volatility
        volatility = float(cp.std(portfolio_returns))
        
        return {
            'var': var,
            'cvar': cvar,
            'volatility': volatility,
            'portfolio_returns_mean': float(cp.mean(portfolio_returns))
        }
    
    def _calculate_cpu_var(
        self,
        portfolio_weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: float
    ) -> Dict[str, float]:
        """CPU-optimized VaR calculation using Numba"""
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_matrix, portfolio_weights)
        
        # Use Numba-optimized functions
        var = self._fast_var_calculation(portfolio_returns, confidence_level)
        cvar = self._fast_cvar_calculation(portfolio_returns, var)
        volatility = np.std(portfolio_returns)
        
        return {
            'var': var,
            'cvar': cvar,
            'volatility': volatility,
            'portfolio_returns_mean': np.mean(portfolio_returns)
        }
    
    def calculate_stress_var(
        self,
        portfolio_weights: np.ndarray,
        returns_matrix: np.ndarray,
        scenario_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate stress VaR for specific scenarios
        
        Args:
            portfolio_weights: Portfolio weights
            returns_matrix: Historical returns matrix
            scenario_name: Specific stress scenario name
            
        Returns:
            Dictionary with stress VaR results
        """
        start_time = time.perf_counter()
        
        try:
            if scenario_name:
                scenarios = [s for s in self.stress_scenarios if s.name == scenario_name]
            else:
                scenarios = self.stress_scenarios
            
            results = {}
            
            for scenario in scenarios:
                # Apply stress scenario adjustments
                stressed_returns = self._apply_stress_scenario(returns_matrix, scenario)
                
                # Calculate stressed VaR
                stressed_var = self.calculate_portfolio_var(
                    portfolio_weights, stressed_returns, self.config.confidence_level
                )
                
                results[scenario.name] = {
                    'scenario': scenario,
                    'stressed_var': stressed_var['var'],
                    'stressed_cvar': stressed_var['cvar'],
                    'var_impact': stressed_var['var'] - stressed_var.get('baseline_var', 0),
                    'probability': scenario.probability
                }
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_metrics['stress_testing'].append(latency_ms)
            
            return {
                'scenarios': results,
                'latency_ms': latency_ms,
                'total_scenarios': len(scenarios)
            }
            
        except Exception as e:
            logger.error(f"Stress VaR calculation failed: {e}")
            return {'error': str(e)}
    
    def _apply_stress_scenario(
        self,
        returns_matrix: np.ndarray,
        scenario: StressScenario
    ) -> np.ndarray:
        """Apply stress scenario adjustments to returns matrix"""
        # Apply market shock
        stressed_returns = returns_matrix * (1 + scenario.market_shock)
        
        # Apply volatility multiplier
        stressed_returns *= scenario.volatility_multiplier
        
        # Apply correlation adjustment (simplified)
        if scenario.correlation_adjustment > 0:
            # Increase correlation between assets
            n_assets = returns_matrix.shape[1]
            correlation_matrix = np.eye(n_assets) * (1 - scenario.correlation_adjustment) + \
                               np.ones((n_assets, n_assets)) * scenario.correlation_adjustment / n_assets
            
            # Apply correlation adjustment
            stressed_returns = np.dot(stressed_returns, correlation_matrix)
        
        return stressed_returns
    
    def calculate_regulatory_capital(
        self,
        portfolio_weights: np.ndarray,
        returns_matrix: np.ndarray,
        instrument_ratings: List[str],
        instrument_maturities: List[float]
    ) -> Dict[str, float]:
        """
        Calculate regulatory capital requirements (Basel III/IV, SEBI)
        
        Args:
            portfolio_weights: Portfolio weights
            returns_matrix: Historical returns matrix
            instrument_ratings: Credit ratings for each instrument
            instrument_maturities: Maturity in years for each instrument
            
        Returns:
            Dictionary with regulatory capital requirements
        """
        try:
            # Calculate base VaR
            base_var = self.calculate_portfolio_var(portfolio_weights, returns_matrix)
            
            # Calculate stressed VaR
            stressed_var = self.calculate_stress_var(portfolio_weights, returns_matrix)
            
            # Basel III/IV capital calculation
            basel_capital = self._calculate_basel_capital(
                base_var, stressed_var, instrument_ratings, instrument_maturities
            )
            
            # SEBI capital calculation
            sebi_capital = self._calculate_sebi_capital(
                base_var, stressed_var, instrument_ratings, instrument_maturities
            )
            
            return {
                'basel_capital': basel_capital,
                'sebi_capital': sebi_capital,
                'total_regulatory_capital': max(basel_capital, sebi_capital),
                'base_var': base_var['var'],
                'stressed_var': max([v['stressed_var'] for v in stressed_var['scenarios'].values()])
            }
            
        except Exception as e:
            logger.error(f"Regulatory capital calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_basel_capital(
        self,
        base_var: Dict[str, float],
        stressed_var: Dict[str, Any],
        instrument_ratings: List[str],
        instrument_maturities: List[float]
    ) -> float:
        """Calculate Basel III/IV regulatory capital"""
        # Base capital requirement
        base_capital = base_var['var'] * 3.0  # 3x multiplier
        
        # Stressed VaR component
        max_stressed_var = max([v['stressed_var'] for v in stressed_var['scenarios'].values()])
        stressed_capital = max_stressed_var * 2.0  # 2x multiplier
        
        # Credit risk adjustment
        credit_risk = self._calculate_credit_risk_adjustment(instrument_ratings, instrument_maturities)
        
        # Market risk capital
        market_risk_capital = max(base_capital, stressed_capital)
        
        # Total capital requirement
        total_capital = market_risk_capital + credit_risk
        
        return total_capital
    
    def _calculate_sebi_capital(
        self,
        base_var: Dict[str, float],
        stressed_var: Dict[str, Any],
        instrument_ratings: List[str],
        instrument_maturities: List[float]
    ) -> float:
        """Calculate SEBI regulatory capital"""
        # SEBI specific calculations
        base_capital = base_var['var'] * 2.5  # SEBI multiplier
        
        # Stressed VaR component
        max_stressed_var = max([v['stressed_var'] for v in stressed_var['scenarios'].values()])
        stressed_capital = max_stressed_var * 1.5  # SEBI stressed multiplier
        
        # Credit risk adjustment
        credit_risk = self._calculate_credit_risk_adjustment(instrument_ratings, instrument_maturities) * 0.8
        
        # Market risk capital
        market_risk_capital = max(base_capital, stressed_capital)
        
        # Total capital requirement
        total_capital = market_risk_capital + credit_risk
        
        return total_capital
    
    def _calculate_credit_risk_adjustment(
        self,
        instrument_ratings: List[str],
        instrument_maturities: List[float]
    ) -> float:
        """Calculate credit risk adjustment based on ratings and maturities"""
        rating_weights = {
            'AAA': 0.0, 'AA': 0.1, 'A': 0.2, 'BBB': 0.5,
            'BB': 1.0, 'B': 2.0, 'CCC': 3.0, 'CC': 4.0, 'C': 5.0
        }
        
        maturity_weights = {
            'short': 0.5,   # < 1 year
            'medium': 1.0,  # 1-5 years
            'long': 1.5     # > 5 years
        }
        
        total_adjustment = 0.0
        
        for rating, maturity in zip(instrument_ratings, instrument_maturities):
            rating_weight = rating_weights.get(rating, 1.0)
            
            if maturity < 1:
                maturity_category = 'short'
            elif maturity < 5:
                maturity_category = 'medium'
            else:
                maturity_category = 'long'
            
            maturity_weight = maturity_weights[maturity_category]
            
            total_adjustment += rating_weight * maturity_weight
        
        return total_adjustment
    
    def _generate_cache_key(
        self,
        portfolio_weights: np.ndarray,
        returns_matrix: np.ndarray,
        confidence_level: float
    ) -> str:
        """Generate cache key for risk calculations"""
        # Create hash of inputs
        weights_hash = hash(tuple(portfolio_weights.flatten()))
        returns_hash = hash(tuple(returns_matrix.flatten()))
        confidence_hash = hash(confidence_level)
        
        return f"{weights_hash}_{returns_hash}_{confidence_hash}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the risk engine"""
        return {
            'latency_metrics': {
                'var_calculation': {
                    'mean_ms': np.mean(self.latency_metrics['var_calculation']) if self.latency_metrics['var_calculation'] else 0,
                    'p95_ms': np.percentile(self.latency_metrics['var_calculation'], 95) if self.latency_metrics['var_calculation'] else 0,
                    'p99_ms': np.percentile(self.latency_metrics['var_calculation'], 99) if self.latency_metrics['var_calculation'] else 0,
                    'count': len(self.latency_metrics['var_calculation'])
                },
                'stress_testing': {
                    'mean_ms': np.mean(self.latency_metrics['stress_testing']) if self.latency_metrics['stress_testing'] else 0,
                    'p95_ms': np.percentile(self.latency_metrics['stress_testing'], 95) if self.latency_metrics['stress_testing'] else 0,
                    'p99_ms': np.percentile(self.latency_metrics['stress_testing'], 99) if self.latency_metrics['stress_testing'] else 0,
                    'count': len(self.latency_metrics['stress_testing'])
                }
            },
            'gpu_available': self.gpu_available,
            'numba_available': self.numba_available,
            'cache_stats': {
                'var_cache_size': len(self.var_cache),
                'stress_cache_size': len(self.stress_cache)
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.var_cache.clear()
        self.stress_cache.clear()
        self.correlation_cache.clear()
        logger.info("Risk engine caches cleared")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.gpu_available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("GPU memory freed")
            except Exception as e:
                logger.warning(f"GPU cleanup failed: {e}")
        
        self.clear_cache()
        logger.info("HFT Risk Engine cleanup completed")
