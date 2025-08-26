"""
Yield Curve Scenario Generator for BondX Risk Management System

This module provides curve/spread scenario generation for stress, VaR, and simulation.

Features:
- PCA-driven (level/slope/curvature) scenarios
- Regime-aware parameters (low/high vol, tightening/easing)
- Random draws from covariance
- Deterministic shocks
- Path simulators for multi-step scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import warnings
from scipy import linalg, stats
from scipy.optimize import minimize
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Available scenario types."""
    PCA_DRIVEN = "pca_driven"
    REGIME_AWARE = "regime_aware"
    DETERMINISTIC = "deterministic"
    RANDOM_DRAWS = "random_draws"
    PATH_SIMULATION = "path_simulation"

class RegimeType(Enum):
    """Available market regimes."""
    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"
    TIGHTENING = "tightening"
    EASING = "easing"
    CRISIS = "crisis"
    NORMAL = "normal"

@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    scenario_type: ScenarioType = ScenarioType.PCA_DRIVEN
    regime_type: RegimeType = RegimeType.NORMAL
    num_scenarios: int = 1000
    confidence_level: float = 0.95
    time_horizon_days: int = 1
    num_steps: int = 1
    pca_components: int = 3
    random_seed: Optional[int] = None
    enable_regime_switching: bool = True
    regime_transition_matrix: Optional[np.ndarray] = None
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class PCALoadings:
    """PCA loadings for yield curve factors."""
    level_loading: np.ndarray
    slope_loading: np.ndarray
    curvature_loading: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    factor_names: List[str]

@dataclass
class YieldCurveScenario:
    """Generated yield curve scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    regime_type: RegimeType
    base_curve: np.ndarray
    shocked_curve: np.ndarray
    tenors: np.ndarray
    shock_magnitude: float
    pca_factors: Optional[Dict[str, float]] = None
    probability: float = 1.0
    generation_date: date = field(default_factory=lambda: datetime.now().date())
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)

@dataclass
class SpreadScenario:
    """Generated credit spread scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    regime_type: RegimeType
    base_spreads: Dict[str, float]
    shocked_spreads: Dict[str, float]
    rating_buckets: List[str]
    shock_magnitude: float
    probability: float = 1.0
    generation_date: date = field(default_factory=lambda: datetime.now().date())
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)

@dataclass
class ScenarioSet:
    """Collection of generated scenarios."""
    scenarios: List[Union[YieldCurveScenario, SpreadScenario]]
    config: ScenarioConfig
    generation_date: date
    metadata: Dict[str, Union[str, float, int]]

class YieldCurveScenarioGenerator:
    """
    Yield curve and spread scenario generator for stress testing and simulation.
    
    Provides PCA-driven and regime-aware scenario generation with full audit trail.
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.config = config or ScenarioConfig()
        self._cache: Dict[str, ScenarioSet] = {}
        self._last_generation: Optional[datetime] = None
        
        # Set random seed if provided
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def generate_yield_curve_scenarios(
        self,
        base_curve: np.ndarray,
        tenors: np.ndarray,
        covariance_matrix: np.ndarray,
        pca_loadings: Optional[PCALoadings] = None
    ) -> ScenarioSet:
        """
        Generate yield curve scenarios.
        
        Args:
            base_curve: Base yield curve (rates by tenor)
            tenors: Tenor points for the curve
            covariance_matrix: Factor covariance matrix
            pca_loadings: PCA loadings for factor decomposition
            
        Returns:
            ScenarioSet with generated scenarios
        """
        start_time = datetime.now()
        
        # Validate inputs
        if len(base_curve) != len(tenors):
            raise ValueError("Base curve and tenors must have same length")
        
        if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")
        
        # Generate scenarios based on type
        if self.config.scenario_type == ScenarioType.PCA_DRIVEN:
            scenarios = self._generate_pca_scenarios(
                base_curve, tenors, covariance_matrix, pca_loadings
            )
        elif self.config.scenario_type == ScenarioType.REGIME_AWARE:
            scenarios = self._generate_regime_scenarios(
                base_curve, tenors, covariance_matrix
            )
        elif self.config.scenario_type == ScenarioType.DETERMINISTIC:
            scenarios = self._generate_deterministic_scenarios(
                base_curve, tenors
            )
        elif self.config.scenario_type == ScenarioType.RANDOM_DRAWS:
            scenarios = self._generate_random_scenarios(
                base_curve, tenors, covariance_matrix
            )
        elif self.config.scenario_type == ScenarioType.PATH_SIMULATION:
            scenarios = self._generate_path_scenarios(
                base_curve, tenors, covariance_matrix
            )
        else:
            raise ValueError(f"Unknown scenario type: {self.config.scenario_type}")
        
        # Create result set
        result = ScenarioSet(
            scenarios=scenarios,
            config=self.config,
            generation_date=datetime.now().date(),
            metadata={
                "num_scenarios": len(scenarios),
                "scenario_type": self.config.scenario_type.value,
                "regime_type": self.config.regime_type.value,
                "num_tenors": len(tenors),
                "time_horizon_days": self.config.time_horizon_days
            }
        )
        
        # Cache result
        if self.config.enable_caching:
            cache_key = self._generate_cache_key("yield_curve", base_curve, tenors)
            self._cache[cache_key] = result
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Yield curve scenario generation completed in {execution_time:.2f}ms")
        
        return result
    
    def generate_spread_scenarios(
        self,
        base_spreads: Dict[str, float],
        covariance_matrix: np.ndarray,
        rating_buckets: Optional[List[str]] = None
    ) -> ScenarioSet:
        """
        Generate credit spread scenarios.
        
        Args:
            base_spreads: Base spreads by rating bucket
            covariance_matrix: Spread factor covariance matrix
            rating_buckets: Rating buckets for spreads
            
        Returns:
            ScenarioSet with generated scenarios
        """
        start_time = datetime.now()
        
        if rating_buckets is None:
            rating_buckets = list(base_spreads.keys())
        
        # Validate inputs
        if len(base_spreads) != covariance_matrix.shape[0]:
            raise ValueError("Base spreads and covariance matrix dimensions must match")
        
        # Generate scenarios based on type
        if self.config.scenario_type == ScenarioType.PCA_DRIVEN:
            scenarios = self._generate_spread_pca_scenarios(
                base_spreads, covariance_matrix, rating_buckets
            )
        elif self.config.scenario_type == ScenarioType.REGIME_AWARE:
            scenarios = self._generate_spread_regime_scenarios(
                base_spreads, covariance_matrix, rating_buckets
            )
        elif self.config.scenario_type == ScenarioType.DETERMINISTIC:
            scenarios = self._generate_spread_deterministic_scenarios(
                base_spreads, rating_buckets
            )
        elif self.config.scenario_type == ScenarioType.RANDOM_DRAWS:
            scenarios = self._generate_spread_random_scenarios(
                base_spreads, covariance_matrix, rating_buckets
            )
        else:
            raise ValueError(f"Unsupported scenario type for spreads: {self.config.scenario_type}")
        
        # Create result set
        result = ScenarioSet(
            scenarios=scenarios,
            config=self.config,
            generation_date=datetime.now().date(),
            metadata={
                "num_scenarios": len(scenarios),
                "scenario_type": self.config.scenario_type.value,
                "regime_type": self.config.regime_type.value,
                "num_rating_buckets": len(rating_buckets),
                "time_horizon_days": self.config.time_horizon_days
            }
        )
        
        # Cache result
        if self.config.enable_caching:
            cache_key = self._generate_cache_key("spreads", base_spreads, rating_buckets)
            self._cache[cache_key] = result
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Spread scenario generation completed in {execution_time:.2f}ms")
        
        return result
    
    def _generate_pca_scenarios(
        self,
        base_curve: np.ndarray,
        tenors: np.ndarray,
        covariance_matrix: np.ndarray,
        pca_loadings: Optional[PCALoadings]
    ) -> List[YieldCurveScenario]:
        """Generate PCA-driven yield curve scenarios."""
        scenarios = []
        
        # Use provided PCA loadings or estimate from covariance
        if pca_loadings is None:
            pca_loadings = self._estimate_pca_loadings(covariance_matrix, tenors)
        
        # Generate scenarios using PCA factors
        for i in range(self.config.num_scenarios):
            # Generate random PCA factor shocks
            factor_shocks = np.random.multivariate_normal(
                mean=np.zeros(self.config.pca_components),
                cov=covariance_matrix[:self.config.pca_components, :self.config.pca_components]
            )
            
            # Apply shocks to curve
            shocked_curve = self._apply_pca_shocks(
                base_curve, factor_shocks, pca_loadings
            )
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean((shocked_curve - base_curve) ** 2))
            
            # Create scenario
            scenario = YieldCurveScenario(
                scenario_id=f"PCA_{i:04d}",
                scenario_type=ScenarioType.PCA_DRIVEN,
                regime_type=self.config.regime_type,
                base_curve=base_curve.copy(),
                shocked_curve=shocked_curve,
                tenors=tenors.copy(),
                shock_magnitude=shock_magnitude,
                pca_factors={
                    "level": float(factor_shocks[0]),
                    "slope": float(factor_shocks[1]),
                    "curvature": float(factor_shocks[2])
                },
                probability=1.0 / self.config.num_scenarios
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_regime_scenarios(
        self,
        base_curve: np.ndarray,
        tenors: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> List[YieldCurveScenario]:
        """Generate regime-aware yield curve scenarios."""
        scenarios = []
        
        # Define regime-specific parameters
        regime_params = self._get_regime_parameters()
        
        for i in range(self.config.num_scenarios):
            # Select regime (with regime switching if enabled)
            if self.config.enable_regime_switching and self.config.regime_transition_matrix is not None:
                regime = self._select_regime_with_transition()
            else:
                regime = self.config.regime_type
            
            # Get regime parameters
            params = regime_params[regime]
            
            # Generate shocks with regime-specific volatility
            shocks = np.random.multivariate_normal(
                mean=params["mean_shift"],
                cov=covariance_matrix * params["volatility_multiplier"]
            )
            
            # Apply shocks to curve
            shocked_curve = base_curve + shocks
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean(shocks ** 2))
            
            # Create scenario
            scenario = YieldCurveScenario(
                scenario_id=f"REGIME_{regime.value}_{i:04d}",
                scenario_type=ScenarioType.REGIME_AWARE,
                regime_type=regime,
                base_curve=base_curve.copy(),
                shocked_curve=shocked_curve,
                tenors=tenors.copy(),
                shock_magnitude=shock_magnitude,
                probability=params["probability"]
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_deterministic_scenarios(
        self,
        base_curve: np.ndarray,
        tenors: np.ndarray
    ) -> List[YieldCurveScenario]:
        """Generate deterministic yield curve scenarios."""
        scenarios = []
        
        # Define standard deterministic scenarios
        deterministic_shifts = [
            ("parallel_up", np.ones_like(base_curve) * 0.01),      # +100 bps parallel
            ("parallel_down", np.ones_like(base_curve) * -0.01),   # -100 bps parallel
            ("steepening", np.linspace(0, 0.02, len(base_curve))), # Steepening
            ("flattening", np.linspace(0.02, 0, len(base_curve))), # Flattening
            ("hump", np.array([0.01, 0.02, 0.03, 0.02, 0.01])),  # Hump shape
            ("butterfly", np.array([0.01, 0.005, 0, -0.005, -0.01]))  # Butterfly
        ]
        
        for scenario_name, shift in deterministic_shifts:
            # Ensure shift has correct length
            if len(shift) != len(base_curve):
                # Interpolate to match tenor points
                shift = np.interp(tenors, np.linspace(0, 1, len(shift)), shift)
            
            # Apply shift
            shocked_curve = base_curve + shift
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean(shift ** 2))
            
            # Create scenario
            scenario = YieldCurveScenario(
                scenario_id=f"DET_{scenario_name}",
                scenario_type=ScenarioType.DETERMINISTIC,
                regime_type=self.config.regime_type,
                base_curve=base_curve.copy(),
                shocked_curve=shocked_curve,
                tenors=tenors.copy(),
                shock_magnitude=shock_magnitude,
                probability=1.0 / len(deterministic_shifts)
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_random_scenarios(
        self,
        base_curve: np.ndarray,
        tenors: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> List[YieldCurveScenario]:
        """Generate random yield curve scenarios."""
        scenarios = []
        
        for i in range(self.config.num_scenarios):
            # Generate random shocks
            shocks = np.random.multivariate_normal(
                mean=np.zeros(len(base_curve)),
                cov=covariance_matrix
            )
            
            # Apply shocks to curve
            shocked_curve = base_curve + shocks
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean(shocks ** 2))
            
            # Create scenario
            scenario = YieldCurveScenario(
                scenario_id=f"RANDOM_{i:04d}",
                scenario_type=ScenarioType.RANDOM_DRAWS,
                regime_type=self.config.regime_type,
                base_curve=base_curve.copy(),
                shocked_curve=shocked_curve,
                tenors=tenors.copy(),
                shock_magnitude=shock_magnitude,
                probability=1.0 / self.config.num_scenarios
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_path_scenarios(
        self,
        base_curve: np.ndarray,
        tenors: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> List[YieldCurveScenario]:
        """Generate path simulation scenarios."""
        scenarios = []
        
        for i in range(self.config.num_scenarios):
            # Simulate path over multiple steps
            current_curve = base_curve.copy()
            
            for step in range(self.config.num_steps):
                # Generate step shocks
                step_shocks = np.random.multivariate_normal(
                    mean=np.zeros(len(current_curve)),
                    cov=covariance_matrix / self.config.num_steps
                )
                
                # Apply step shocks
                current_curve += step_shocks
            
            # Calculate total shock magnitude
            total_shock = current_curve - base_curve
            shock_magnitude = np.sqrt(np.mean(total_shock ** 2))
            
            # Create scenario
            scenario = YieldCurveScenario(
                scenario_id=f"PATH_{i:04d}",
                scenario_type=ScenarioType.PATH_SIMULATION,
                regime_type=self.config.regime_type,
                base_curve=base_curve.copy(),
                shocked_curve=current_curve,
                tenors=tenors.copy(),
                shock_magnitude=shock_magnitude,
                probability=1.0 / self.config.num_scenarios,
                metadata={"num_steps": self.config.num_steps}
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_spread_pca_scenarios(
        self,
        base_spreads: Dict[str, float],
        covariance_matrix: np.ndarray,
        rating_buckets: List[str]
    ) -> List[SpreadScenario]:
        """Generate PCA-driven spread scenarios."""
        scenarios = []
        
        # Convert to arrays
        base_spreads_array = np.array([base_spreads[bucket] for bucket in rating_buckets])
        
        for i in range(self.config.num_scenarios):
            # Generate random shocks
            shocks = np.random.multivariate_normal(
                mean=np.zeros(len(rating_buckets)),
                cov=covariance_matrix
            )
            
            # Apply shocks to spreads
            shocked_spreads_array = base_spreads_array + shocks
            
            # Convert back to dictionary
            shocked_spreads = {
                bucket: float(shocked_spreads_array[j])
                for j, bucket in enumerate(rating_buckets)
            }
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean(shocks ** 2))
            
            # Create scenario
            scenario = SpreadScenario(
                scenario_id=f"SPREAD_PCA_{i:04d}",
                scenario_type=ScenarioType.PCA_DRIVEN,
                regime_type=self.config.regime_type,
                base_spreads=base_spreads.copy(),
                shocked_spreads=shocked_spreads,
                rating_buckets=rating_buckets.copy(),
                shock_magnitude=shock_magnitude,
                probability=1.0 / self.config.num_scenarios
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_spread_regime_scenarios(
        self,
        base_spreads: Dict[str, float],
        covariance_matrix: np.ndarray,
        rating_buckets: List[str]
    ) -> List[SpreadScenario]:
        """Generate regime-aware spread scenarios."""
        scenarios = []
        
        # Get regime parameters
        regime_params = self._get_regime_parameters()
        
        # Convert to arrays
        base_spreads_array = np.array([base_spreads[bucket] for bucket in rating_buckets])
        
        for i in range(self.config.num_scenarios):
            # Select regime
            regime = self.config.regime_type
            params = regime_params[regime]
            
            # Generate shocks with regime-specific parameters
            shocks = np.random.multivariate_normal(
                mean=params["mean_shift"][:len(rating_buckets)],
                cov=covariance_matrix * params["volatility_multiplier"]
            )
            
            # Apply shocks to spreads
            shocked_spreads_array = base_spreads_array + shocks
            
            # Convert back to dictionary
            shocked_spreads = {
                bucket: float(shocked_spreads_array[j])
                for j, bucket in enumerate(rating_buckets)
            }
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean(shocks ** 2))
            
            # Create scenario
            scenario = SpreadScenario(
                scenario_id=f"SPREAD_REGIME_{regime.value}_{i:04d}",
                scenario_type=ScenarioType.REGIME_AWARE,
                regime_type=regime,
                base_spreads=base_spreads.copy(),
                shocked_spreads=shocked_spreads,
                rating_buckets=rating_buckets.copy(),
                shock_magnitude=shock_magnitude,
                probability=params["probability"]
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_spread_deterministic_scenarios(
        self,
        base_spreads: Dict[str, float],
        rating_buckets: List[str]
    ) -> List[SpreadScenario]:
        """Generate deterministic spread scenarios."""
        scenarios = []
        
        # Define standard spread scenarios
        spread_shifts = [
            ("widening", {bucket: 0.01 for bucket in rating_buckets}),      # +100 bps
            ("tightening", {bucket: -0.01 for bucket in rating_buckets}),   # -100 bps
            ("flight_to_quality", {  # Flight to quality
                "AAA": -0.005, "AA": -0.002, "A": 0.005, "BBB": 0.02
            }),
            ("credit_crisis", {  # Credit crisis
                "AAA": 0.01, "AA": 0.02, "A": 0.05, "BBB": 0.10
            })
        ]
        
        for scenario_name, shift in spread_shifts:
            # Apply shift to base spreads
            shocked_spreads = {}
            for bucket in rating_buckets:
                base_spread = base_spreads.get(bucket, 0.0)
                shift_amount = shift.get(bucket, 0.0)
                shocked_spreads[bucket] = base_spread + shift_amount
            
            # Calculate shock magnitude
            shift_values = list(shift.values())
            shock_magnitude = np.sqrt(np.mean(np.array(shift_values) ** 2))
            
            # Create scenario
            scenario = SpreadScenario(
                scenario_id=f"SPREAD_DET_{scenario_name}",
                scenario_type=ScenarioType.DETERMINISTIC,
                regime_type=self.config.regime_type,
                base_spreads=base_spreads.copy(),
                shocked_spreads=shocked_spreads,
                rating_buckets=rating_buckets.copy(),
                shock_magnitude=shock_magnitude,
                probability=1.0 / len(spread_shifts)
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_spread_random_scenarios(
        self,
        base_spreads: Dict[str, float],
        covariance_matrix: np.ndarray,
        rating_buckets: List[str]
    ) -> List[SpreadScenario]:
        """Generate random spread scenarios."""
        scenarios = []
        
        # Convert to arrays
        base_spreads_array = np.array([base_spreads[bucket] for bucket in rating_buckets])
        
        for i in range(self.config.num_scenarios):
            # Generate random shocks
            shocks = np.random.multivariate_normal(
                mean=np.zeros(len(rating_buckets)),
                cov=covariance_matrix
            )
            
            # Apply shocks to spreads
            shocked_spreads_array = base_spreads_array + shocks
            
            # Convert back to dictionary
            shocked_spreads = {
                bucket: float(shocked_spreads_array[j])
                for j, bucket in enumerate(rating_buckets)
            }
            
            # Calculate shock magnitude
            shock_magnitude = np.sqrt(np.mean(shocks ** 2))
            
            # Create scenario
            scenario = SpreadScenario(
                scenario_id=f"SPREAD_RANDOM_{i:04d}",
                scenario_type=ScenarioType.RANDOM_DRAWS,
                regime_type=self.config.regime_type,
                base_spreads=base_spreads.copy(),
                shocked_spreads=shocked_spreads,
                rating_buckets=rating_buckets.copy(),
                shock_magnitude=shock_magnitude,
                probability=1.0 / self.config.num_scenarios
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _estimate_pca_loadings(
        self,
        covariance_matrix: np.ndarray,
        tenors: np.ndarray
    ) -> PCALoadings:
        """Estimate PCA loadings from covariance matrix."""
        # Perform eigendecomposition
        eigenvals, eigenvecs = linalg.eigh(covariance_matrix)
        
        # Sort by eigenvalues (descending)
        sort_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_idx]
        eigenvecs = eigenvecs[:, sort_idx]
        
        # Calculate explained variance ratios
        total_variance = np.sum(eigenvals)
        explained_variance_ratio = eigenvals / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Extract first three components
        level_loading = eigenvecs[:, 0]
        slope_loading = eigenvecs[:, 1]
        curvature_loading = eigenvecs[:, 2]
        
        return PCALoadings(
            level_loading=level_loading,
            slope_loading=slope_loading,
            curvature_loading=curvature_loading,
            explained_variance_ratio=explained_variance_ratio[:3],
            cumulative_variance_ratio=cumulative_variance_ratio[:3],
            factor_names=["level", "slope", "curvature"]
        )
    
    def _apply_pca_shocks(
        self,
        base_curve: np.ndarray,
        factor_shocks: np.ndarray,
        pca_loadings: PCALoadings
    ) -> np.ndarray:
        """Apply PCA factor shocks to yield curve."""
        shocked_curve = base_curve.copy()
        
        # Apply level shock
        shocked_curve += factor_shocks[0] * pca_loadings.level_loading
        
        # Apply slope shock
        if len(factor_shocks) > 1:
            shocked_curve += factor_shocks[1] * pca_loadings.slope_loading
        
        # Apply curvature shock
        if len(factor_shocks) > 2:
            shocked_curve += factor_shocks[2] * pca_loadings.curvature_loading
        
        return shocked_curve
    
    def _get_regime_parameters(self) -> Dict[RegimeType, Dict[str, Union[np.ndarray, float]]]:
        """Get regime-specific parameters."""
        return {
            RegimeType.LOW_VOL: {
                "mean_shift": np.zeros(10),
                "volatility_multiplier": 0.5,
                "probability": 0.3
            },
            RegimeType.HIGH_VOL: {
                "mean_shift": np.zeros(10),
                "volatility_multiplier": 2.0,
                "probability": 0.2
            },
            RegimeType.TIGHTENING: {
                "mean_shift": np.ones(10) * 0.01,
                "volatility_multiplier": 1.5,
                "probability": 0.2
            },
            RegimeType.EASING: {
                "mean_shift": np.ones(10) * -0.01,
                "volatility_multiplier": 1.5,
                "probability": 0.2
            },
            RegimeType.CRISIS: {
                "mean_shift": np.ones(10) * 0.02,
                "volatility_multiplier": 3.0,
                "probability": 0.1
            },
            RegimeType.NORMAL: {
                "mean_shift": np.zeros(10),
                "volatility_multiplier": 1.0,
                "probability": 0.3
            }
        }
    
    def _select_regime_with_transition(self) -> RegimeType:
        """Select regime using transition matrix."""
        if self.config.regime_transition_matrix is None:
            return self.config.regime_type
        
        # Simple regime selection (in production, use proper Markov chain)
        current_regime_idx = list(RegimeType).index(self.config.regime_type)
        transition_probs = self.config.regime_transition_matrix[current_regime_idx, :]
        
        # Select new regime
        new_regime_idx = np.random.choice(len(RegimeType), p=transition_probs)
        return list(RegimeType)[new_regime_idx]
    
    def _generate_cache_key(
        self,
        scenario_type: str,
        base_data: Union[np.ndarray, Dict],
        additional_data: Union[np.ndarray, List]
    ) -> str:
        """Generate cache key for scenario results."""
        if isinstance(base_data, np.ndarray):
            base_hash = str(hash(base_data.tobytes()))
        else:
            base_hash = str(hash(str(base_data)))
        
        if isinstance(additional_data, np.ndarray):
            additional_hash = str(hash(additional_data.tobytes()))
        else:
            additional_hash = str(hash(str(additional_data)))
        
        return f"{scenario_type}_{base_hash}_{additional_hash}"
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Scenario generator cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "last_generation": self._last_generation.isoformat() if self._last_generation else None
        }
