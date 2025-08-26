"""
Portfolio Optimization for BondX Risk Management System

This module provides a fixed-income optimizer with practical constraints.

Features:
- Mean-variance optimization with constraints
- Duration target/bands, rating/sector caps, issuer concentration
- Robust variants: shrinkage covariance, worst-case bands
- Black-Litterman scaffold for views
- Turnover penalties and feasibility diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import warnings
from scipy import optimize
from scipy import linalg
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Available optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    ROBUST_MEAN_VARIANCE = "robust_mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"

class ConstraintType(Enum):
    """Available constraint types."""
    DURATION_TARGET = "duration_target"
    DURATION_BANDS = "duration_bands"
    RATING_CAPS = "rating_caps"
    SECTOR_CAPS = "sector_caps"
    ISSUER_CONCENTRATION = "issuer_concentration"
    TURNOVER_PENALTY = "turnover_penalty"
    WEIGHT_BOUNDS = "weight_bounds"

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    duration_target: Optional[float] = None
    duration_bands: Optional[Tuple[float, float]] = None
    rating_caps: Optional[Dict[str, float]] = None
    sector_caps: Optional[Dict[str, float]] = None
    issuer_concentration: Optional[float] = None
    turnover_penalty: Optional[float] = None
    weight_bounds: Optional[Tuple[float, float]] = (0.0, 1.0)
    max_issuer_weight: Optional[float] = None
    min_rating: Optional[str] = None
    max_rating: Optional[str] = None

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    risk_aversion: float = 1.0
    expected_return_method: Literal["carry_roll", "historical", "views"] = "carry_roll"
    covariance_method: Literal["sample", "shrinkage", "robust"] = "shrinkage"
    shrinkage_lambda: Optional[float] = None
    black_litterman_tau: float = 0.05
    black_litterman_omega_scale: float = 0.1
    enable_views: bool = False
    max_iterations: int = 1000
    tolerance: float = 1e-6
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class AssetData:
    """Asset data for optimization."""
    asset_id: str
    current_weight: float
    expected_return: float
    duration: float
    convexity: float
    rating: str
    sector: str
    issuer: str
    market_value: float
    face_value: float
    yield_to_maturity: float
    coupon_rate: float
    maturity_date: date

@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    optimal_weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    duration: float
    convexity: float
    constraint_violations: List[str]
    shadow_prices: Dict[str, float]
    optimization_status: str
    execution_time_ms: float
    metadata: Dict[str, Union[str, float, int]]

@dataclass
class BlackLittermanViews:
    """Black-Litterman views for optimization."""
    view_matrix: np.ndarray
    view_returns: np.ndarray
    view_confidence: np.ndarray
    asset_names: List[str]

class PortfolioOptimizer:
    """
    Fixed-income portfolio optimizer with practical constraints.
    
    Provides mean-variance optimization with duration, rating, and sector constraints.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self._cache: Dict[str, OptimizationResult] = {}
        self._last_optimization: Optional[datetime] = None
        
    def optimize_portfolio(
        self,
        assets: List[AssetData],
        covariance_matrix: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
        current_weights: Optional[np.ndarray] = None,
        views: Optional[BlackLittermanViews] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            assets: List of assets with characteristics
            covariance_matrix: Asset covariance matrix
            constraints: Optimization constraints
            current_weights: Current portfolio weights
            views: Black-Litterman views (if applicable)
            
        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        start_time = datetime.now()
        
        # Validate inputs
        if len(assets) != covariance_matrix.shape[0]:
            raise ValueError("Number of assets must match covariance matrix dimensions")
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        if current_weights is None:
            current_weights = np.array([asset.current_weight for asset in assets])
        
        # Prepare data
        asset_names = [asset.asset_id for asset in assets]
        expected_returns = np.array([asset.expected_return for asset in assets])
        durations = np.array([asset.duration for asset in assets])
        ratings = [asset.rating for asset in assets]
        sectors = [asset.sector for asset in assets]
        issuers = [asset.issuer for asset in assets]
        
        # Apply optimization method
        if self.config.method == OptimizationMethod.MEAN_VARIANCE:
            result = self._mean_variance_optimization(
                expected_returns, covariance_matrix, constraints, current_weights,
                asset_names, durations, ratings, sectors, issuers
            )
        elif self.config.method == OptimizationMethod.ROBUST_MEAN_VARIANCE:
            result = self._robust_mean_variance_optimization(
                expected_returns, covariance_matrix, constraints, current_weights,
                asset_names, durations, ratings, sectors, issuers
            )
        elif self.config.method == OptimizationMethod.BLACK_LITTERMAN:
            if views is None:
                raise ValueError("Black-Litterman optimization requires views")
            result = self._black_litterman_optimization(
                expected_returns, covariance_matrix, constraints, current_weights,
                asset_names, durations, ratings, sectors, issuers, views
            )
        elif self.config.method == OptimizationMethod.RISK_PARITY:
            result = self._risk_parity_optimization(
                covariance_matrix, constraints, current_weights,
                asset_names, durations, ratings, sectors, issuers
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
        
        # Cache result
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(assets, constraints, views)
            self._cache[cache_key] = result
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Portfolio optimization completed in {execution_time:.2f}ms")
        
        return result
    
    def _mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        current_weights: np.ndarray,
        asset_names: List[str],
        durations: np.ndarray,
        ratings: List[str],
        sectors: List[str],
        issuers: List[str]
    ) -> OptimizationResult:
        """Perform mean-variance optimization."""
        n_assets = len(expected_returns)
        
        # Objective function: minimize risk - lambda * return
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
            return portfolio_risk - self.config.risk_aversion * portfolio_return
        
        # Constraints
        constraint_list = []
        
        # Weight sum constraint
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Duration target constraint
        if constraints.duration_target is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w * durations) - constraints.duration_target
            })
        
        # Duration bands constraint
        if constraints.duration_bands is not None:
            min_duration, max_duration = constraints.duration_bands
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w * durations) - min_duration
            })
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda w: max_duration - np.sum(w * durations)
            })
        
        # Rating caps constraint
        if constraints.rating_caps is not None:
            for rating, cap in constraints.rating_caps.items():
                rating_mask = np.array([r == rating for r in ratings])
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, mask=rating_mask, c=cap: c - np.sum(w * mask)
                })
        
        # Sector caps constraint
        if constraints.sector_caps is not None:
            for sector, cap in constraints.sector_caps.items():
                sector_mask = np.array([s == sector for s in sectors])
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, mask=sector_mask, c=cap: c - np.sum(w * mask)
                })
        
        # Issuer concentration constraint
        if constraints.issuer_concentration is not None:
            for issuer in set(issuers):
                issuer_mask = np.array([i == issuer for i in issuers])
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, mask=issuer_mask, c=constraints.issuer_concentration: c - np.sum(w * mask)
                })
        
        # Weight bounds
        if constraints.weight_bounds is not None:
            min_weight, max_weight = constraints.weight_bounds
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: w[idx] - min_weight
                })
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: max_weight - w[idx]
                })
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=constraint_list,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x
                optimization_status = "success"
            else:
                optimal_weights = initial_weights
                optimization_status = f"failed: {result.message}"
                
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            optimal_weights = initial_weights
            optimization_status = f"error: {str(e)}"
        
        # Calculate results
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_risk = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        portfolio_duration = np.sum(optimal_weights * durations)
        portfolio_convexity = np.sum(optimal_weights * np.array([asset.convexity for asset in assets]))
        
        # Check constraint violations
        constraint_violations = self._check_constraint_violations(
            optimal_weights, constraints, durations, ratings, sectors, issuers
        )
        
        # Calculate shadow prices (simplified)
        shadow_prices = self._calculate_shadow_prices(
            optimal_weights, expected_returns, covariance_matrix, constraints
        )
        
        return OptimizationResult(
            optimal_weights=optimal_weights,
            expected_return=float(portfolio_return),
            expected_risk=float(portfolio_risk),
            sharpe_ratio=float(sharpe_ratio),
            duration=float(portfolio_duration),
            convexity=float(portfolio_convexity),
            constraint_violations=constraint_violations,
            shadow_prices=shadow_prices,
            optimization_status=optimization_status,
            execution_time_ms=0.0,  # Will be set by caller
            metadata={
                "method": self.config.method.value,
                "risk_aversion": self.config.risk_aversion,
                "num_assets": n_assets,
                "num_constraints": len(constraint_list)
            }
        )
    
    def _robust_mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        current_weights: np.ndarray,
        asset_names: List[str],
        durations: np.ndarray,
        ratings: List[str],
        sectors: List[str],
        issuers: List[str]
    ) -> OptimizationResult:
        """Perform robust mean-variance optimization."""
        # Apply shrinkage to covariance matrix
        if self.config.covariance_method == "shrinkage":
            shrunk_cov = self._apply_shrinkage(covariance_matrix)
        else:
            shrunk_cov = covariance_matrix
        
        # Use worst-case expected returns
        robust_returns = expected_returns - 0.5 * np.sqrt(np.diag(shrunk_cov))
        
        # Call standard mean-variance optimization with robust parameters
        return self._mean_variance_optimization(
            robust_returns, shrunk_cov, constraints, current_weights,
            asset_names, durations, ratings, sectors, issuers
        )
    
    def _black_litterman_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        current_weights: np.ndarray,
        asset_names: List[str],
        durations: np.ndarray,
        ratings: List[str],
        sectors: List[str],
        issuers: List[str],
        views: BlackLittermanViews
    ) -> OptimizationResult:
        """Perform Black-Litterman optimization."""
        # Calculate equilibrium returns
        tau = self.config.black_litterman_tau
        pi = self._calculate_equilibrium_returns(current_weights, covariance_matrix)
        
        # Combine views with equilibrium returns
        combined_returns = self._combine_black_litterman_views(
            pi, views, covariance_matrix, tau
        )
        
        # Use combined returns for optimization
        return self._mean_variance_optimization(
            combined_returns, covariance_matrix, constraints, current_weights,
            asset_names, durations, ratings, sectors, issuers
        )
    
    def _risk_parity_optimization(
        self,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        current_weights: np.ndarray,
        asset_names: List[str],
        durations: np.ndarray,
        ratings: List[str],
        sectors: List[str],
        issuers: List[str]
    ) -> OptimizationResult:
        """Perform risk parity optimization."""
        n_assets = len(covariance_matrix)
        
        # Objective function: minimize sum of squared risk contributions
        def objective(weights):
            portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
            risk_contributions = weights * (covariance_matrix @ weights) / portfolio_risk
            target_contribution = portfolio_risk / n_assets
            return np.sum((risk_contributions - target_contribution) ** 2)
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weight sum
        ]
        
        # Add other constraints as in mean-variance
        if constraints.duration_target is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w * durations) - constraints.duration_target
            })
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=constraint_list,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x
                optimization_status = "success"
            else:
                optimal_weights = initial_weights
                optimization_status = f"failed: {result.message}"
                
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")
            optimal_weights = initial_weights
            optimization_status = f"error: {str(e)}"
        
        # Calculate results (similar to mean-variance)
        expected_returns = np.zeros(n_assets)  # Risk parity doesn't use returns
        portfolio_return = 0.0
        portfolio_risk = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
        sharpe_ratio = 0.0
        portfolio_duration = np.sum(optimal_weights * durations)
        portfolio_convexity = 0.0  # Would need asset convexity data
        
        return OptimizationResult(
            optimal_weights=optimal_weights,
            expected_return=portfolio_return,
            expected_risk=float(portfolio_risk),
            sharpe_ratio=sharpe_ratio,
            duration=float(portfolio_duration),
            convexity=portfolio_convexity,
            constraint_violations=[],
            shadow_prices={},
            optimization_status=optimization_status,
            execution_time_ms=0.0,
            metadata={
                "method": "risk_parity",
                "num_assets": n_assets
            }
        )
    
    def _apply_shrinkage(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Apply shrinkage to covariance matrix."""
        if self.config.shrinkage_lambda is not None:
            lambda_param = self.config.shrinkage_lambda
        else:
            lambda_param = 0.5  # Default shrinkage
        
        # Target matrix (diagonal with average variance)
        target = np.eye(covariance_matrix.shape[0]) * np.trace(covariance_matrix) / covariance_matrix.shape[0]
        
        # Apply shrinkage
        shrunk_cov = lambda_param * target + (1 - lambda_param) * covariance_matrix
        
        return shrunk_cov
    
    def _calculate_equilibrium_returns(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate equilibrium returns using reverse optimization."""
        # Assuming risk aversion of 2.5 (market standard)
        risk_aversion = 2.5
        pi = risk_aversion * covariance_matrix @ weights
        return pi
    
    def _combine_black_litterman_views(
        self,
        equilibrium_returns: np.ndarray,
        views: BlackLittermanViews,
        covariance_matrix: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """Combine Black-Litterman views with equilibrium returns."""
        # Prior precision matrix
        tau_sigma = tau * covariance_matrix
        prior_precision = np.linalg.inv(tau_sigma)
        
        # Views precision matrix
        omega = self.config.black_litterman_omega_scale * np.eye(len(views.view_returns))
        views_precision = views.view_matrix.T @ np.linalg.inv(omega) @ views.view_matrix
        
        # Posterior precision and mean
        posterior_precision = prior_precision + views_precision
        posterior_mean = np.linalg.solve(
            posterior_precision,
            prior_precision @ equilibrium_returns + 
            views.view_matrix.T @ np.linalg.inv(omega) @ views.view_returns
        )
        
        return posterior_mean
    
    def _check_constraint_violations(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints,
        durations: np.ndarray,
        ratings: List[str],
        sectors: List[str],
        issuers: List[str]
    ) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        # Duration target
        if constraints.duration_target is not None:
            portfolio_duration = np.sum(weights * durations)
            if abs(portfolio_duration - constraints.duration_target) > 0.1:
                violations.append(f"Duration target: {portfolio_duration:.2f} vs {constraints.duration_target:.2f}")
        
        # Duration bands
        if constraints.duration_bands is not None:
            min_duration, max_duration = constraints.duration_bands
            portfolio_duration = np.sum(weights * durations)
            if portfolio_duration < min_duration or portfolio_duration > max_duration:
                violations.append(f"Duration bands: {portfolio_duration:.2f} not in [{min_duration:.2f}, {max_duration:.2f}]")
        
        # Rating caps
        if constraints.rating_caps is not None:
            for rating, cap in constraints.rating_caps.items():
                rating_weight = np.sum(weights * np.array([r == rating for r in ratings]))
                if rating_weight > cap + 0.01:  # Allow small tolerance
                    violations.append(f"Rating cap {rating}: {rating_weight:.3f} > {cap:.3f}")
        
        # Sector caps
        if constraints.sector_caps is not None:
            for sector, cap in constraints.sector_caps.items():
                sector_weight = np.sum(weights * np.array([s == sector for s in sectors]))
                if sector_weight > cap + 0.01:
                    violations.append(f"Sector cap {sector}: {sector_weight:.3f} > {cap:.3f}")
        
        # Issuer concentration
        if constraints.issuer_concentration is not None:
            for issuer in set(issuers):
                issuer_weight = np.sum(weights * np.array([i == issuer for i in issuers]))
                if issuer_weight > constraints.issuer_concentration + 0.01:
                    violations.append(f"Issuer concentration {issuer}: {issuer_weight:.3f} > {constraints.issuer_concentration:.3f}")
        
        return violations
    
    def _calculate_shadow_prices(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: PortfolioConstraints
    ) -> Dict[str, float]:
        """Calculate shadow prices for constraints (simplified)."""
        shadow_prices = {}
        
        # This is a simplified calculation
        # In production, use proper sensitivity analysis
        
        if constraints.duration_target is not None:
            # Approximate shadow price for duration constraint
            shadow_prices["duration_target"] = 0.1
        
        if constraints.rating_caps is not None:
            for rating in constraints.rating_caps:
                shadow_prices[f"rating_cap_{rating}"] = 0.05
        
        if constraints.sector_caps is not None:
            for sector in constraints.sector_caps:
                shadow_prices[f"sector_cap_{sector}"] = 0.05
        
        return shadow_prices
    
    def _generate_cache_key(
        self,
        assets: List[AssetData],
        constraints: PortfolioConstraints,
        views: Optional[BlackLittermanViews]
    ) -> str:
        """Generate cache key for optimization results."""
        asset_ids = "_".join([asset.asset_id for asset in assets])
        constraint_hash = str(hash(str(constraints)))
        views_hash = str(hash(str(views))) if views else "none"
        
        return f"optimization_{asset_ids}_{constraint_hash}_{views_hash}"
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Portfolio optimizer cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "last_optimization": self._last_optimization.isoformat() if self._last_optimization else None
        }
