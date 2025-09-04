"""
GPU Acceleration Module for BondX

This module provides GPU-accelerated mathematical operations:
- CUDA/cuPy integration for matrix operations
- GPU-accelerated yield curve bootstrapping
- PCA (SVD decomposition) on GPU
- Monte Carlo simulations with GPU parallelization
- Fallback to CPU for environments without GPUs
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.linalg as cp_linalg
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_linalg = None
    cp_sparse = None

# PyTorch for GPU operations
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..core.logging import get_logger
from ..core.monitoring import MetricsCollector

logger = get_logger(__name__)

class GPUDeviceType(Enum):
    """GPU device types"""
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"

@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""
    device_type: GPUDeviceType = GPUDeviceType.AUTO
    memory_pool_size: Optional[int] = None  # MB
    enable_memory_pool: bool = True
    fallback_to_cpu: bool = True
    batch_size: int = 1000
    max_matrix_size: int = 10000  # Maximum matrix size for GPU processing

@dataclass
class MatrixOperationResult:
    """Result of matrix operation"""
    result: Union[np.ndarray, cp.ndarray]
    device_used: str
    computation_time: float
    memory_used: float
    metadata: Dict[str, Any]

class GPUAccelerator:
    """GPU acceleration for mathematical operations"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics = MetricsCollector()
        
        # Device setup
        self.device = self._setup_device()
        
        # Memory pool setup
        if self.config.enable_memory_pool and GPU_AVAILABLE:
            self._setup_memory_pool()
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'average_gpu_time': 0.0,
            'average_cpu_time': 0.0,
            'memory_usage': 0.0
        }
    
    def _setup_device(self) -> str:
        """Setup GPU device"""
        if self.config.device_type == GPUDeviceType.CPU:
            self.logger.info("GPU acceleration disabled, using CPU")
            return 'cpu'
        
        if self.config.device_type == GPUDeviceType.CUDA and GPU_AVAILABLE:
            try:
                # Test GPU availability
                test_array = cp.array([1, 2, 3])
                del test_array
                self.logger.info("GPU acceleration enabled with CUDA")
                return 'cuda'
            except Exception as e:
                self.logger.warning(f"GPU setup failed: {e}, falling back to CPU")
                return 'cpu'
        
        # Auto-detect
        if GPU_AVAILABLE:
            try:
                test_array = cp.array([1, 2, 3])
                del test_array
                self.logger.info("GPU acceleration auto-enabled with CUDA")
                return 'cuda'
            except Exception as e:
                self.logger.warning(f"GPU auto-detection failed: {e}, using CPU")
                return 'cpu'
        
        self.logger.info("GPU not available, using CPU")
        return 'cpu'
    
    def _setup_memory_pool(self):
        """Setup CUDA memory pool"""
        if self.device == 'cuda' and GPU_AVAILABLE:
            try:
                if self.config.memory_pool_size:
                    pool_size = self.config.memory_pool_size * 1024 * 1024  # Convert MB to bytes
                    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                    self.logger.info(f"CUDA memory pool enabled with {self.config.memory_pool_size}MB")
                else:
                    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                    self.logger.info("CUDA memory pool enabled with default size")
            except Exception as e:
                self.logger.warning(f"Memory pool setup failed: {e}")
    
    def should_use_gpu(self, matrix_size: int) -> bool:
        """Determine if GPU should be used for given matrix size"""
        if self.device != 'cuda':
            return False
        
        if matrix_size > self.config.max_matrix_size:
            self.logger.warning(f"Matrix size {matrix_size} exceeds GPU limit {self.config.max_matrix_size}")
            return False
        
        return True
    
    def to_gpu(self, array: np.ndarray) -> cp.ndarray:
        """Convert numpy array to GPU array"""
        if self.device == 'cuda' and GPU_AVAILABLE:
            return cp.asarray(array)
        else:
            return array
    
    def to_cpu(self, array: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """Convert GPU array to CPU array"""
        if hasattr(array, 'get'):  # GPU array
            return cp.asnumpy(array)
        else:
            return array
    
    async def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> MatrixOperationResult:
        """GPU-accelerated matrix multiplication"""
        start_time = time.time()
        
        try:
            if self.should_use_gpu(A.shape[0] * A.shape[1]):
                result = await self._gpu_matrix_multiply(A, B)
                device_used = 'cuda'
                self.performance_metrics['gpu_operations'] += 1
            else:
                result = await self._cpu_matrix_multiply(A, B)
                device_used = 'cpu'
                self.performance_metrics['cpu_operations'] += 1
            
            computation_time = time.time() - start_time
            memory_used = result.nbytes / (1024 * 1024)  # MB
            
            self._update_performance_metrics(computation_time, device_used)
            
            return MatrixOperationResult(
                result=result,
                device_used=device_used,
                computation_time=computation_time,
                memory_used=memory_used,
                metadata={
                    "matrix_shapes": [A.shape, B.shape],
                    "operation": "matrix_multiply"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in matrix multiplication: {e}")
            raise
    
    async def _gpu_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> cp.ndarray:
        """GPU matrix multiplication using cuBLAS"""
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        
        result = cp.dot(A_gpu, B_gpu)
        
        # Clean up GPU memory
        del A_gpu, B_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return result
    
    async def _cpu_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """CPU matrix multiplication"""
        return np.dot(A, B)
    
    async def svd_decomposition(self, matrix: np.ndarray, full_matrices: bool = True) -> MatrixOperationResult:
        """GPU-accelerated SVD decomposition"""
        start_time = time.time()
        
        try:
            if self.should_use_gpu(matrix.shape[0] * matrix.shape[1]):
                U, S, Vt = await self._gpu_svd(matrix, full_matrices)
                device_used = 'cuda'
                self.performance_metrics['gpu_operations'] += 1
            else:
                U, S, Vt = await self._cpu_svd(matrix, full_matrices)
                device_used = 'cpu'
                self.performance_metrics['cpu_operations'] += 1
            
            computation_time = time.time() - start_time
            memory_used = sum(arr.nbytes for arr in [U, S, Vt]) / (1024 * 1024)  # MB
            
            self._update_performance_metrics(computation_time, device_used)
            
            return MatrixOperationResult(
                result=(U, S, Vt),
                device_used=device_used,
                computation_time=computation_time,
                memory_used=memory_used,
                metadata={
                    "matrix_shape": matrix.shape,
                    "full_matrices": full_matrices,
                    "operation": "svd_decomposition"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in SVD decomposition: {e}")
            raise
    
    async def _gpu_svd(self, matrix: np.ndarray, full_matrices: bool) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """GPU SVD decomposition using cuSOLVER"""
        matrix_gpu = cp.asarray(matrix)
        
        U, S, Vt = cp_linalg.svd(matrix_gpu, full_matrices=full_matrices)
        
        # Clean up GPU memory
        del matrix_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return U, S, Vt
    
    async def _cpu_svd(self, matrix: np.ndarray, full_matrices: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU SVD decomposition"""
        return np.linalg.svd(matrix, full_matrices=full_matrices)
    
    async def pca_decomposition(self, data: np.ndarray, n_components: Optional[int] = None) -> MatrixOperationResult:
        """GPU-accelerated PCA decomposition"""
        start_time = time.time()
        
        try:
            if self.should_use_gpu(data.shape[0] * data.shape[1]):
                components, explained_variance = await self._gpu_pca(data, n_components)
                device_used = 'cuda'
                self.performance_metrics['gpu_operations'] += 1
            else:
                components, explained_variance = await self._cpu_pca(data, n_components)
                device_used = 'cpu'
                self.performance_metrics['cpu_operations'] += 1
            
            computation_time = time.time() - start_time
            memory_used = sum(arr.nbytes for arr in [components, explained_variance]) / (1024 * 1024)  # MB
            
            self._update_performance_metrics(computation_time, device_used)
            
            return MatrixOperationResult(
                result=(components, explained_variance),
                device_used=device_used,
                computation_time=computation_time,
                memory_used=memory_used,
                metadata={
                    "data_shape": data.shape,
                    "n_components": n_components,
                    "operation": "pca_decomposition"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in PCA decomposition: {e}")
            raise
    
    async def _gpu_pca(self, data: np.ndarray, n_components: Optional[int]) -> Tuple[cp.ndarray, cp.ndarray]:
        """GPU PCA decomposition"""
        # Center the data
        data_gpu = cp.asarray(data)
        mean_gpu = cp.mean(data_gpu, axis=0)
        data_centered_gpu = data_gpu - mean_gpu
        
        # Compute covariance matrix
        cov_matrix = cp.dot(data_centered_gpu.T, data_centered_gpu) / (data_gpu.shape[0] - 1)
        
        # SVD decomposition
        U, S, Vt = cp_linalg.svd(cov_matrix, full_matrices=False)
        
        # Select components
        if n_components is None:
            n_components = min(data_gpu.shape)
        
        components = Vt[:n_components].T
        explained_variance = S[:n_components]
        
        # Clean up GPU memory
        del data_gpu, mean_gpu, data_centered_gpu, cov_matrix, U, S, Vt
        cp.get_default_memory_pool().free_all_blocks()
        
        return components, explained_variance
    
    async def _cpu_pca(self, data: np.ndarray, n_components: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """CPU PCA decomposition"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        pca.fit(data)
        
        return pca.components_.T, pca.explained_variance_
    
    async def monte_carlo_simulation(self, n_paths: int, n_steps: int, 
                                   drift: float, volatility: float, 
                                   initial_value: float = 1.0) -> MatrixOperationResult:
        """GPU-accelerated Monte Carlo simulation"""
        start_time = time.time()
        
        try:
            if self.should_use_gpu(n_paths * n_steps):
                paths = await self._gpu_monte_carlo(n_paths, n_steps, drift, volatility, initial_value)
                device_used = 'cuda'
                self.performance_metrics['gpu_operations'] += 1
            else:
                paths = await self._cpu_monte_carlo(n_paths, n_steps, drift, volatility, initial_value)
                device_used = 'cpu'
                self.performance_metrics['cpu_operations'] += 1
            
            computation_time = time.time() - start_time
            memory_used = paths.nbytes / (1024 * 1024)  # MB
            
            self._update_performance_metrics(computation_time, device_used)
            
            return MatrixOperationResult(
                result=paths,
                device_used=device_used,
                computation_time=computation_time,
                memory_used=memory_used,
                metadata={
                    "n_paths": n_paths,
                    "n_steps": n_steps,
                    "drift": drift,
                    "volatility": volatility,
                    "initial_value": initial_value,
                    "operation": "monte_carlo_simulation"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            raise
    
    async def _gpu_monte_carlo(self, n_paths: int, n_steps: int, 
                              drift: float, volatility: float, 
                              initial_value: float) -> cp.ndarray:
        """GPU Monte Carlo simulation"""
        # Generate random numbers on GPU
        rng = cp.random.RandomState(42)
        random_numbers = rng.normal(0, 1, (n_paths, n_steps))
        
        # Calculate time steps
        dt = 1.0 / n_steps
        sqrt_dt = cp.sqrt(dt)
        
        # Initialize paths
        paths = cp.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_value
        
        # Simulate paths
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * cp.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * sqrt_dt * random_numbers[:, i]
            )
        
        # Clean up GPU memory
        del random_numbers
        cp.get_default_memory_pool().free_all_blocks()
        
        return paths
    
    async def _cpu_monte_carlo(self, n_paths: int, n_steps: int, 
                              drift: float, volatility: float, 
                              initial_value: float) -> np.ndarray:
        """CPU Monte Carlo simulation"""
        rng = np.random.RandomState(42)
        random_numbers = rng.normal(0, 1, (n_paths, n_steps))
        
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_value
        
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * sqrt_dt * random_numbers[:, i]
            )
        
        return paths
    
    async def yield_curve_bootstrap(self, cash_flows: np.ndarray, 
                                  market_prices: np.ndarray, 
                                  maturities: np.ndarray) -> MatrixOperationResult:
        """GPU-accelerated yield curve bootstrapping"""
        start_time = time.time()
        
        try:
            if self.should_use_gpu(len(cash_flows) * len(maturities)):
                spot_rates = await self._gpu_yield_bootstrap(cash_flows, market_prices, maturities)
                device_used = 'cuda'
                self.performance_metrics['gpu_operations'] += 1
            else:
                spot_rates = await self._cpu_yield_bootstrap(cash_flows, market_prices, maturities)
                device_used = 'cpu'
                self.performance_metrics['cpu_operations'] += 1
            
            computation_time = time.time() - start_time
            memory_used = spot_rates.nbytes / (1024 * 1024)  # MB
            
            self._update_performance_metrics(computation_time, device_used)
            
            return MatrixOperationResult(
                result=spot_rates,
                device_used=device_used,
                computation_time=computation_time,
                memory_used=memory_used,
                metadata={
                    "cash_flows_shape": cash_flows.shape,
                    "n_maturities": len(maturities),
                    "operation": "yield_curve_bootstrap"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in yield curve bootstrapping: {e}")
            raise
    
    async def _gpu_yield_bootstrap(self, cash_flows: np.ndarray, 
                                 market_prices: np.ndarray, 
                                 maturities: np.ndarray) -> cp.ndarray:
        """GPU yield curve bootstrapping"""
        # Convert to GPU arrays
        cf_gpu = cp.asarray(cash_flows)
        prices_gpu = cp.asarray(market_prices)
        mat_gpu = cp.asarray(maturities)
        
        # Initialize spot rates
        n_maturities = len(maturities)
        spot_rates = cp.zeros(n_maturities)
        
        # Bootstrap spot rates sequentially
        for i in range(n_maturities):
            if i == 0:
                # First rate from first bond
                spot_rates[i] = -cp.log(prices_gpu[i] / cf_gpu[i, i]) / mat_gpu[i]
            else:
                # Solve for subsequent rates using Newton-Raphson
                spot_rates[i] = await self._solve_spot_rate_gpu(
                    cf_gpu, prices_gpu, mat_gpu, spot_rates, i
                )
        
        # Clean up GPU memory
        del cf_gpu, prices_gpu, mat_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return spot_rates
    
    async def _cpu_yield_bootstrap(self, cash_flows: np.ndarray, 
                                 market_prices: np.ndarray, 
                                 maturities: np.ndarray) -> np.ndarray:
        """CPU yield curve bootstrapping"""
        n_maturities = len(maturities)
        spot_rates = np.zeros(n_maturities)
        
        for i in range(n_maturities):
            if i == 0:
                spot_rates[i] = -np.log(market_prices[i] / cash_flows[i, i]) / maturities[i]
            else:
                spot_rates[i] = self._solve_spot_rate_cpu(
                    cash_flows, market_prices, maturities, spot_rates, i
                )
        
        return spot_rates
    
    async def _solve_spot_rate_gpu(self, cf_gpu: cp.ndarray, prices_gpu: cp.ndarray, 
                                  mat_gpu: cp.ndarray, spot_rates: cp.ndarray, 
                                  index: int) -> float:
        """Solve for spot rate using Newton-Raphson on GPU"""
        # Simplified Newton-Raphson implementation
        # In production, use more sophisticated solvers
        
        rate = 0.05  # Initial guess
        tolerance = 1e-6
        max_iterations = 100
        
        for _ in range(max_iterations):
            # Calculate bond price with current rate
            price_calc = self._calculate_bond_price_gpu(cf_gpu, mat_gpu, spot_rates, rate, index)
            
            # Calculate derivative
            derivative = self._calculate_bond_price_derivative_gpu(cf_gpu, mat_gpu, spot_rates, rate, index)
            
            # Update rate
            rate_new = rate - (price_calc - prices_gpu[index]) / derivative
            
            if abs(rate_new - rate) < tolerance:
                break
            
            rate = rate_new
        
        return float(rate)
    
    def _solve_spot_rate_cpu(self, cash_flows: np.ndarray, market_prices: np.ndarray, 
                            maturities: np.ndarray, spot_rates: np.ndarray, 
                            index: int) -> float:
        """Solve for spot rate using Newton-Raphson on CPU"""
        rate = 0.05
        tolerance = 1e-6
        max_iterations = 100
        
        for _ in range(max_iterations):
            price_calc = self._calculate_bond_price_cpu(cash_flows, maturities, spot_rates, rate, index)
            derivative = self._calculate_bond_price_derivative_cpu(cash_flows, maturities, spot_rates, rate, index)
            
            rate_new = rate - (price_calc - market_prices[index]) / derivative
            
            if abs(rate_new - rate) < tolerance:
                break
            
            rate = rate_new
        
        return rate
    
    def _calculate_bond_price_gpu(self, cf_gpu: cp.ndarray, mat_gpu: cp.ndarray, 
                                 spot_rates: cp.ndarray, rate: float, 
                                 index: int) -> float:
        """Calculate bond price on GPU"""
        price = 0.0
        
        for j in range(index + 1):
            if cf_gpu[index, j] > 0:
                if j == index:
                    discount = cp.exp(-rate * mat_gpu[j])
                else:
                    discount = cp.exp(-spot_rates[j] * mat_gpu[j])
                
                price += cf_gpu[index, j] * discount
        
        return float(price)
    
    def _calculate_bond_price_cpu(self, cash_flows: np.ndarray, maturities: np.ndarray, 
                                 spot_rates: np.ndarray, rate: float, 
                                 index: int) -> float:
        """Calculate bond price on CPU"""
        price = 0.0
        
        for j in range(index + 1):
            if cash_flows[index, j] > 0:
                if j == index:
                    discount = np.exp(-rate * maturities[j])
                else:
                    discount = np.exp(-spot_rates[j] * maturities[j])
                
                price += cash_flows[index, j] * discount
        
        return price
    
    def _calculate_bond_price_derivative_gpu(self, cf_gpu: cp.ndarray, mat_gpu: cp.ndarray, 
                                           spot_rates: cp.ndarray, rate: float, 
                                           index: int) -> float:
        """Calculate bond price derivative on GPU"""
        derivative = 0.0
        
        for j in range(index + 1):
            if cf_gpu[index, j] > 0 and j == index:
                derivative -= cf_gpu[index, j] * mat_gpu[j] * cp.exp(-rate * mat_gpu[j])
        
        return float(derivative)
    
    def _calculate_bond_price_derivative_cpu(self, cash_flows: np.ndarray, maturities: np.ndarray, 
                                           spot_rates: np.ndarray, rate: float, 
                                           index: int) -> float:
        """Calculate bond price derivative on CPU"""
        derivative = 0.0
        
        for j in range(index + 1):
            if cash_flows[index, j] > 0 and j == index:
                derivative -= cash_flows[index, j] * maturities[j] * np.exp(-rate * maturities[j])
        
        return derivative
    
    def _update_performance_metrics(self, computation_time: float, device: str):
        """Update performance tracking metrics"""
        self.performance_metrics['total_operations'] += 1
        
        if device == 'cuda':
            current_avg = self.performance_metrics['average_gpu_time']
            total_ops = self.performance_metrics['gpu_operations']
            
            self.performance_metrics['average_gpu_time'] = (
                (current_avg * (total_ops - 1) + computation_time) / total_ops
            )
        else:
            current_avg = self.performance_metrics['average_cpu_time']
            total_ops = self.performance_metrics['cpu_operations']
            
            self.performance_metrics['average_cpu_time'] = (
                (current_avg * (total_ops - 1) + computation_time) / total_ops
            )
        
        if self.performance_metrics['total_operations'] % 100 == 0:
            self.logger.info(f"GPU accelerator performance: "
                           f"GPU ops: {self.performance_metrics['gpu_operations']}, "
                           f"CPU ops: {self.performance_metrics['cpu_operations']}, "
                           f"Avg GPU time: {self.performance_metrics['average_gpu_time']:.4f}s, "
                           f"Avg CPU time: {self.performance_metrics['average_cpu_time']:.4f}s")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def clear_performance_metrics(self):
        """Clear performance metrics"""
        self.performance_metrics = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'average_gpu_time': 0.0,
            'average_cpu_time': 0.0,
            'memory_usage': 0.0
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {
            "device_type": self.device,
            "gpu_available": GPU_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "memory_pool_enabled": self.config.enable_memory_pool
        }
        
        if self.device == 'cuda' and GPU_AVAILABLE:
            try:
                info["gpu_memory_total"] = cp.cuda.runtime.memGetInfo()[1] / (1024**3)  # GB
                info["gpu_memory_free"] = cp.cuda.runtime.memGetInfo()[0] / (1024**3)  # GB
            except Exception as e:
                info["gpu_memory_error"] = str(e)
        
        return info
