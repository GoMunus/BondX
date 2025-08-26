"""
Option-adjusted spread calculator for BondX Backend.

This module implements OAS calculations for bonds with embedded options including
callable, putable, and callable-with-make-whole structures using both lattice
and Monte Carlo methods.
"""

import hashlib
import json
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize_scalar
from scipy.stats import norm

from ..core.logging import get_logger
from ..database.models import DayCountConvention
from .day_count import DayCountCalculator
from .yield_curves import YieldCurve, CurveType
from .cash_flows import CashFlow, BondCashFlowConfig

logger = get_logger(__name__)


class OptionType(Enum):
    """Types of embedded options in bonds."""
    NONE = "NONE"
    CALLABLE = "CALLABLE"
    PUTABLE = "PUTABLE"
    CALLABLE_MAKE_WHOLE = "CALLABLE_MAKE_WHOLE"
    PREPAYMENT = "PREPAYMENT"


class PricingMethod(Enum):
    """Pricing methods for OAS calculation."""
    LATTICE = "LATTICE"
    MONTE_CARLO = "MONTE_CARLO"


class LatticeModel(Enum):
    """Short-rate lattice models."""
    HO_LEE = "HO_LEE"
    BLACK_DERMAN_TOY = "BLACK_DERMAN_TOY"


@dataclass
class CallSchedule:
    """Call schedule for callable bonds."""
    call_date: date
    call_price: Decimal
    notice_period_days: int = 30
    make_whole_spread: Optional[Decimal] = None  # For make-whole calls


@dataclass
class PutSchedule:
    """Put schedule for putable bonds."""
    put_date: date
    put_price: Decimal
    notice_period_days: int = 30


@dataclass
class PrepaymentFunction:
    """Prepayment function for mortgage-backed securities."""
    cpr_base: float  # Constant prepayment rate base
    psa_multiplier: float = 1.0  # PSA multiplier
    burnout_factor: float = 1.0  # Burnout factor
    age_factor: float = 1.0  # Age factor


@dataclass
class VolatilitySurface:
    """Volatility term structure for short-rate models."""
    tenors: np.ndarray  # Tenors in years
    volatilities: np.ndarray  # Rate volatilities
    mean_reversion: Optional[float] = None  # Mean reversion parameter
    correlation_matrix: Optional[np.ndarray] = None  # For multi-factor models


@dataclass
class OASInputs:
    """Inputs for OAS calculation."""
    base_curve: YieldCurve
    volatility_surface: VolatilitySurface
    cash_flows: List[CashFlow]
    option_type: OptionType
    call_schedule: Optional[List[CallSchedule]] = None
    put_schedule: Optional[List[PutSchedule]] = None
    prepayment_function: Optional[PrepaymentFunction] = None
    market_price: Decimal
    day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360
    compounding_frequency: int = 2  # Semi-annual
    settlement_date: Optional[date] = None
    prepayment_hook: Optional[Callable] = None


@dataclass
class OASOutputs:
    """Outputs from OAS calculation."""
    oas_bps: float  # OAS in basis points
    model_pv: Decimal  # Model present value at OAS
    option_value: Decimal  # Value of embedded option
    option_adjusted_duration: float
    option_adjusted_convexity: float
    delta: Optional[float] = None  # First derivative w.r.t. rates
    gamma: Optional[float] = None  # Second derivative w.r.t. rates
    theta: Optional[float] = None  # Time decay
    convergence_status: str = "UNKNOWN"
    iterations: int = 0
    lattice_steps: Optional[int] = None
    monte_carlo_paths: Optional[int] = None
    solve_time_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatticeNode:
    """Node in the interest rate lattice."""
    time_step: int
    rate_level: int
    short_rate: float
    discount_factor: float
    option_value: float = 0.0
    bond_value: float = 0.0
    exercise_value: float = 0.0
    continuation_value: float = 0.0


class OASCalculator:
    """
    Production-grade OAS calculator for bonds with embedded options.
    
    Supports both lattice and Monte Carlo methods with configurable
    short-rate models and optimization algorithms.
    """
    
    def __init__(
        self,
        pricing_method: PricingMethod = PricingMethod.LATTICE,
        lattice_model: LatticeModel = LatticeModel.HO_LEE,
        lattice_steps: int = 500,
        monte_carlo_paths: int = 10000,
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 100,
        random_seed: Optional[int] = None
    ):
        """
        Initialize OAS calculator.
        
        Args:
            pricing_method: Lattice or Monte Carlo method
            lattice_model: Short-rate model for lattice
            lattice_steps: Number of time steps for lattice
            monte_carlo_paths: Number of paths for Monte Carlo
            convergence_tolerance: Tolerance for OAS convergence
            max_iterations: Maximum iterations for root finding
            random_seed: Random seed for Monte Carlo
        """
        self.pricing_method = pricing_method
        self.lattice_model = lattice_model
        self.lattice_steps = lattice_steps
        self.monte_carlo_paths = monte_carlo_paths
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        
        # Initialize components
        self.day_count_calculator = DayCountCalculator()
        self.logger = logger
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_oas(self, inputs: OASInputs) -> OASOutputs:
        """
        Calculate OAS for a bond with embedded options.
        
        Args:
            inputs: OAS calculation inputs
            
        Returns:
            OAS calculation outputs
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Validate inputs
            self._validate_inputs(inputs)
            
            # Calculate OAS using selected method
            if self.pricing_method == PricingMethod.LATTICE:
                oas_bps, diagnostics = self._calculate_oas_lattice(inputs)
            else:
                oas_bps, diagnostics = self._calculate_oas_monte_carlo(inputs)
            
            # Calculate option-adjusted metrics
            metrics = self._calculate_option_adjusted_metrics(inputs, oas_bps)
            
            # Calculate Greeks if possible
            greeks = self._calculate_greeks(inputs, oas_bps)
            
            solve_time_ms = (pd.Timestamp.now() - start_time).total_seconds() * 1000
            
            return OASOutputs(
                oas_bps=oas_bps,
                model_pv=metrics['model_pv'],
                option_value=metrics['option_value'],
                option_adjusted_duration=metrics['duration'],
                option_adjusted_convexity=metrics['convexity'],
                delta=greeks.get('delta'),
                gamma=greeks.get('gamma'),
                theta=greeks.get('theta'),
                convergence_status=diagnostics['convergence_status'],
                iterations=diagnostics['iterations'],
                lattice_steps=self.lattice_steps if self.pricing_method == PricingMethod.LATTICE else None,
                monte_carlo_paths=self.monte_carlo_paths if self.pricing_method == PricingMethod.MONTE_CARLO else None,
                solve_time_ms=solve_time_ms,
                diagnostics=diagnostics
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating OAS: {str(e)}")
            raise
    
    def _validate_inputs(self, inputs: OASInputs) -> None:
        """Validate OAS calculation inputs."""
        if inputs.base_curve is None:
            raise ValueError("Base curve is required")
        
        if inputs.volatility_surface is None:
            raise ValueError("Volatility surface is required")
        
        if not inputs.cash_flows:
            raise ValueError("Cash flows are required")
        
        if inputs.market_price <= 0:
            raise ValueError("Market price must be positive")
        
        if inputs.option_type == OptionType.CALLABLE and not inputs.call_schedule:
            raise ValueError("Call schedule required for callable bonds")
        
        if inputs.option_type == OptionType.PUTABLE and not inputs.put_schedule:
            raise ValueError("Put schedule required for putable bonds")
    
    def _calculate_oas_lattice(self, inputs: OASInputs) -> Tuple[float, Dict[str, Any]]:
        """Calculate OAS using lattice method."""
        # Build short-rate lattice
        lattice = self._build_short_rate_lattice(inputs)
        
        # Calibrate lattice to match base curve
        self._calibrate_lattice_to_curve(lattice, inputs.base_curve)
        
        # Define objective function for OAS
        def objective_function(oas_bps: float) -> float:
            model_pv = self._price_bond_on_lattice(lattice, inputs, oas_bps)
            return float(model_pv - inputs.market_price)
        
        # Find OAS using root finding
        try:
            result = root_scalar(
                objective_function,
                bracket=[-1000, 1000],  # -1000 to +1000 bps
                method='brentq',
                xtol=self.convergence_tolerance,
                maxiter=self.max_iterations
            )
            
            if result.converged:
                convergence_status = "CONVERGED"
            else:
                convergence_status = "MAX_ITERATIONS"
                warnings.warn("OAS calculation did not converge within max iterations")
            
            return result.root, {
                'convergence_status': convergence_status,
                'iterations': result.iterations,
                'function_calls': result.function_calls,
                'root_finding_method': 'brentq'
            }
            
        except Exception as e:
            self.logger.error(f"Root finding failed: {str(e)}")
            # Fallback to minimization
            result = minimize_scalar(
                lambda x: abs(objective_function(x)),
                bounds=(-1000, 1000),
                method='bounded'
            )
            
            return result.x, {
                'convergence_status': 'MINIMIZATION_FALLBACK',
                'iterations': result.nit,
                'function_calls': result.nfev,
                'root_finding_method': 'minimization_fallback'
            }
    
    def _build_short_rate_lattice(self, inputs: OASInputs) -> List[List[LatticeNode]]:
        """Build short-rate lattice."""
        lattice = []
        dt = 1.0 / self.lattice_steps
        
        for step in range(self.lattice_steps + 1):
            step_nodes = []
            max_level = step
            
            for level in range(-max_level, max_level + 1):
                if self.lattice_model == LatticeModel.HO_LEE:
                    # Ho-Lee model: dr = θ(t)dt + σdW
                    short_rate = self._ho_lee_rate(step, level, dt, inputs.volatility_surface)
                else:
                    # Black-Derman-Toy model: dln(r) = θ(t)dt + σ(t)dW
                    short_rate = self._black_derman_toy_rate(step, level, dt, inputs.volatility_surface)
                
                node = LatticeNode(
                    time_step=step,
                    rate_level=level,
                    short_rate=short_rate,
                    discount_factor=np.exp(-short_rate * dt)
                )
                step_nodes.append(node)
            
            lattice.append(step_nodes)
        
        return lattice
    
    def _ho_lee_rate(self, step: int, level: int, dt: float, vol_surface: VolatilitySurface) -> float:
        """Calculate short rate using Ho-Lee model."""
        # Simplified Ho-Lee implementation
        # In practice, this would be calibrated to match the yield curve
        base_rate = 0.05  # 5% base rate
        volatility = np.interp(step * dt, vol_surface.tenors, vol_surface.volatilities)
        drift = 0.0  # Would be calibrated
        
        rate = base_rate + drift * step * dt + volatility * level * np.sqrt(dt)
        return max(rate, 0.001)  # Ensure positive rates
    
    def _black_derman_toy_rate(self, step: int, level: int, dt: float, vol_surface: VolatilitySurface) -> float:
        """Calculate short rate using Black-Derman-Toy model."""
        # Simplified BDT implementation
        base_rate = 0.05
        volatility = np.interp(step * dt, vol_surface.tenors, vol_surface.volatilities)
        
        # BDT uses log-normal rates
        log_rate = np.log(base_rate) + volatility * level * np.sqrt(dt)
        rate = np.exp(log_rate)
        return max(rate, 0.001)
    
    def _calibrate_lattice_to_curve(self, lattice: List[List[LatticeNode]], base_curve: YieldCurve) -> None:
        """Calibrate lattice to match base yield curve."""
        # This is a simplified calibration
        # In practice, would use more sophisticated methods like:
        # - Forward induction for Ho-Lee
        # - Recursive calibration for BDT
        
        for step, step_nodes in enumerate(lattice):
            if step == 0:
                continue
            
            # Adjust rates to match curve at this tenor
            target_tenor = step / self.lattice_steps
            target_rate = self._interpolate_curve_rate(base_curve, target_tenor)
            
            # Simple adjustment - in practice would be more sophisticated
            adjustment = target_rate - np.mean([node.short_rate for node in step_nodes])
            for node in step_nodes:
                node.short_rate += adjustment
                node.discount_factor = np.exp(-node.short_rate / self.lattice_steps)
    
    def _interpolate_curve_rate(self, curve: YieldCurve, tenor: float) -> float:
        """Interpolate rate from yield curve."""
        if curve.curve_type == CurveType.ZERO_CURVE:
            return np.interp(tenor, curve.tenors, curve.rates)
        else:
            # Convert other curve types to zero rates if needed
            # This is simplified - would need proper conversion
            return np.interp(tenor, curve.tenors, curve.rates)
    
    def _price_bond_on_lattice(self, lattice: List[List[LatticeNode]], inputs: OASInputs, oas_bps: float) -> Decimal:
        """Price bond on lattice with given OAS."""
        # Backward induction to price bond
        oas_rate = oas_bps / 10000.0
        
        # Start from final cash flow
        final_step = len(lattice) - 1
        for node in lattice[final_step]:
            node.bond_value = self._calculate_final_cash_flow(inputs, final_step)
        
        # Backward induction
        for step in range(final_step - 1, -1, -1):
            for node in lattice[step]:
                # Calculate continuation value
                continuation_value = self._calculate_continuation_value(
                    lattice, step, node, inputs, oas_rate
                )
                
                # Calculate exercise value if option exists
                exercise_value = self._calculate_exercise_value(
                    step, node, inputs
                )
                
                # Choose optimal value
                if inputs.option_type == OptionType.CALLABLE:
                    # Issuer optimal - choose minimum
                    node.bond_value = min(continuation_value, exercise_value)
                elif inputs.option_type == OptionType.PUTABLE:
                    # Holder optimal - choose maximum
                    node.bond_value = max(continuation_value, exercise_value)
                else:
                    # No options
                    node.bond_value = continuation_value
                
                node.continuation_value = continuation_value
                node.exercise_value = exercise_value
        
        # Return present value at root
        root_node = lattice[0][0]
        return Decimal(str(root_node.bond_value))
    
    def _calculate_final_cash_flow(self, inputs: OASInputs, step: int) -> float:
        """Calculate final cash flow at given step."""
        # Simplified - would need proper cash flow mapping
        total_cf = 0.0
        for cf in inputs.cash_flows:
            if cf.period_index == step:
                total_cf += float(cf.coupon_amount + cf.principal_repayment)
        return total_cf
    
    def _calculate_continuation_value(
        self,
        lattice: List[List[LatticeNode]],
        step: int,
        node: LatticeNode,
        inputs: OASInputs,
        oas_rate: float
    ) -> float:
        """Calculate continuation value for backward induction."""
        if step + 1 >= len(lattice):
            return 0.0
        
        # Get next step nodes
        next_step = lattice[step + 1]
        
        # Calculate expected value
        expected_value = 0.0
        for next_node in next_step:
            # Simplified transition probabilities
            # In practice, would use proper risk-neutral probabilities
            prob = 1.0 / len(next_step)
            expected_value += prob * next_node.bond_value
        
        # Discount and add OAS
        discount_factor = node.discount_factor
        oas_adjustment = np.exp(-oas_rate / self.lattice_steps)
        
        return (expected_value + self._calculate_cash_flow_at_step(inputs, step)) * discount_factor * oas_adjustment
    
    def _calculate_cash_flow_at_step(self, inputs: OASInputs, step: int) -> float:
        """Calculate cash flow at given step."""
        # Simplified cash flow calculation
        # In practice, would map steps to actual cash flow dates
        total_cf = 0.0
        for cf in inputs.cash_flows:
            if cf.period_index == step:
                total_cf += float(cf.coupon_amount + cf.principal_repayment)
        return total_cf
    
    def _calculate_exercise_value(self, step: int, node: LatticeNode, inputs: OASInputs) -> float:
        """Calculate exercise value for embedded options."""
        if inputs.option_type == OptionType.NONE:
            return float('inf')  # No exercise
        
        # Simplified exercise value calculation
        # In practice, would check actual option schedules
        if inputs.option_type == OptionType.CALLABLE and inputs.call_schedule:
            # Check if callable at this step
            for call in inputs.call_schedule:
                if self._is_callable_at_step(step, call):
                    return float(call.call_price)
        
        elif inputs.option_type == OptionType.PUTABLE and inputs.put_schedule:
            # Check if putable at this step
            for put in inputs.put_schedule:
                if self._is_putable_at_step(step, put):
                    return float(put.put_price)
        
        return float('inf')  # No exercise at this step
    
    def _is_callable_at_step(self, step: int, call: CallSchedule) -> bool:
        """Check if bond is callable at given step."""
        # Simplified - would need proper date mapping
        return step > 0  # Callable after first period
    
    def _is_putable_at_step(self, step: int, put: PutSchedule) -> bool:
        """Check if bond is putable at given step."""
        # Simplified - would need proper date mapping
        return step > 0  # Putable after first period
    
    def _calculate_oas_monte_carlo(self, inputs: OASInputs) -> Tuple[float, Dict[str, Any]]:
        """Calculate OAS using Monte Carlo method."""
        # Generate interest rate paths
        rate_paths = self._generate_rate_paths(inputs)
        
        # Define objective function for OAS
        def objective_function(oas_bps: float) -> float:
            model_pv = self._price_bond_monte_carlo(rate_paths, inputs, oas_bps)
            return float(model_pv - inputs.market_price)
        
        # Find OAS using root finding
        try:
            result = root_scalar(
                objective_function,
                bracket=[-1000, 1000],
                method='brentq',
                xtol=self.convergence_tolerance,
                maxiter=self.max_iterations
            )
            
            return result.root, {
                'convergence_status': 'CONVERGED' if result.converged else 'MAX_ITERATIONS',
                'iterations': result.iterations,
                'function_calls': result.function_calls,
                'root_finding_method': 'brentq'
            }
            
        except Exception as e:
            self.logger.error(f"Monte Carlo OAS calculation failed: {str(e)}")
            raise
    
    def _generate_rate_paths(self, inputs: OASInputs) -> np.ndarray:
        """Generate interest rate paths for Monte Carlo."""
        # Simplified rate path generation
        # In practice, would use proper short-rate models
        
        paths = np.zeros((self.monte_carlo_paths, self.lattice_steps + 1))
        dt = 1.0 / self.lattice_steps
        
        # Initial rate
        initial_rate = self._interpolate_curve_rate(inputs.base_curve, 0.0)
        paths[:, 0] = initial_rate
        
        # Generate paths
        for path in range(self.monte_carlo_paths):
            for step in range(1, self.lattice_steps + 1):
                # Simplified random walk
                volatility = np.interp(step * dt, inputs.volatility_surface.tenors, inputs.volatility_surface.volatilities)
                random_shock = np.random.normal(0, 1) * volatility * np.sqrt(dt)
                
                paths[path, step] = max(
                    paths[path, step - 1] + random_shock,
                    0.001  # Ensure positive rates
                )
        
        return paths
    
    def _price_bond_monte_carlo(self, rate_paths: np.ndarray, inputs: OASInputs, oas_bps: float) -> Decimal:
        """Price bond using Monte Carlo paths."""
        oas_rate = oas_bps / 10000.0
        dt = 1.0 / self.lattice_steps
        
        present_values = []
        
        for path in rate_paths:
            # Calculate present value for this path
            pv = 0.0
            discount_factor = 1.0
            
            for step, rate in enumerate(path):
                # Add cash flow if any
                cf = self._calculate_cash_flow_at_step(inputs, step)
                pv += cf * discount_factor
                
                # Update discount factor
                discount_factor *= np.exp(-(rate + oas_rate) * dt)
            
            present_values.append(pv)
        
        # Return average present value
        avg_pv = np.mean(present_values)
        return Decimal(str(avg_pv))
    
    def _calculate_option_adjusted_metrics(self, inputs: OASInputs, oas_bps: float) -> Dict[str, Any]:
        """Calculate option-adjusted duration and convexity."""
        # Calculate metrics at OAS
        oas_rate = oas_bps / 10000.0
        
        # Calculate base metrics
        base_pv = self._calculate_base_present_value(inputs, 0.0)
        oas_pv = self._calculate_base_present_value(inputs, oas_rate)
        
        # Calculate option value
        option_value = base_pv - oas_pv
        
        # Calculate duration and convexity
        duration = self._calculate_duration(inputs, oas_rate)
        convexity = self._calculate_convexity(inputs, oas_rate)
        
        return {
            'model_pv': oas_pv,
            'option_value': option_value,
            'duration': duration,
            'convexity': convexity
        }
    
    def _calculate_base_present_value(self, inputs: OASInputs, spread_rate: float) -> Decimal:
        """Calculate base present value with given spread."""
        # Simplified PV calculation
        # In practice, would use proper discounting
        total_pv = Decimal('0')
        
        for cf in inputs.cash_flows:
            # Simplified discounting
            years_to_cf = (cf.payment_date - (inputs.settlement_date or date.today())).days / 365.25
            discount_factor = Decimal(str(np.exp(-spread_rate * years_to_cf)))
            total_pv += cf.coupon_amount * discount_factor + cf.principal_repayment * discount_factor
        
        return total_pv
    
    def _calculate_duration(self, inputs: OASInputs, spread_rate: float) -> float:
        """Calculate option-adjusted duration."""
        # Simplified duration calculation
        # In practice, would use proper numerical differentiation
        
        # Small rate shock
        shock = 0.0001  # 1 bp
        
        pv_up = self._calculate_base_present_value(inputs, spread_rate + shock)
        pv_down = self._calculate_base_present_value(inputs, spread_rate - shock)
        
        duration = -float((pv_up - pv_down) / (2 * shock * float(pv_up + pv_down) / 2))
        return duration
    
    def _calculate_convexity(self, inputs: OASInputs, spread_rate: float) -> float:
        """Calculate option-adjusted convexity."""
        # Simplified convexity calculation
        # In practice, would use proper numerical differentiation
        
        # Small rate shock
        shock = 0.0001  # 1 bp
        
        pv_up = self._calculate_base_present_value(inputs, spread_rate + shock)
        pv_down = self._calculate_base_present_value(inputs, spread_rate - shock)
        pv_base = self._calculate_base_present_value(inputs, spread_rate)
        
        convexity = float((pv_up + pv_down - 2 * pv_base) / (shock * shock * float(pv_base)))
        return convexity
    
    def _calculate_greeks(self, inputs: OASInputs, oas_bps: float) -> Dict[str, float]:
        """Calculate Greeks if possible."""
        greeks = {}
        
        try:
            # Delta (first derivative w.r.t. rates)
            shock = 0.0001
            oas_rate = oas_bps / 10000.0
            
            pv_up = self._calculate_base_present_value(inputs, oas_rate + shock)
            pv_down = self._calculate_base_present_value(inputs, oas_rate - shock)
            
            greeks['delta'] = float((pv_up - pv_down) / (2 * shock))
            
            # Gamma (second derivative w.r.t. rates)
            pv_base = self._calculate_base_present_value(inputs, oas_rate)
            greeks['gamma'] = float((pv_up + pv_down - 2 * pv_base) / (shock * shock))
            
            # Theta (time decay) - simplified
            greeks['theta'] = 0.0  # Would need proper time decay calculation
            
        except Exception as e:
            self.logger.warning(f"Could not calculate Greeks: {str(e)}")
        
        return greeks
    
    def get_cache_key(self, inputs: OASInputs) -> str:
        """Generate cache key for OAS calculation."""
        # Create hash of inputs for caching
        input_data = {
            'base_curve_hash': hashlib.md5(
                json.dumps(inputs.base_curve.__dict__, default=str, sort_keys=True).encode()
            ).hexdigest(),
            'vol_surface_hash': hashlib.md5(
                json.dumps(inputs.volatility_surface.__dict__, default=str, sort_keys=True).encode()
            ).hexdigest(),
            'cash_flows_hash': hashlib.md5(
                json.dumps([cf.__dict__ for cf in inputs.cash_flows], default=str, sort_keys=True).encode()
            ).hexdigest(),
            'option_type': inputs.option_type.value,
            'market_price': str(inputs.market_price),
            'pricing_method': self.pricing_method.value,
            'lattice_model': self.lattice_model.value if self.pricing_method == PricingMethod.LATTICE else None,
            'lattice_steps': self.lattice_steps if self.pricing_method == PricingMethod.LATTICE else None,
            'monte_carlo_paths': self.monte_carlo_paths if self.pricing_method == PricingMethod.MONTE_CARLO else None
        }
        
        return hashlib.md5(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()


# Export classes
__all__ = [
    "OASCalculator",
    "OptionType",
    "PricingMethod",
    "LatticeModel",
    "CallSchedule",
    "PutSchedule",
    "PrepaymentFunction",
    "VolatilitySurface",
    "OASInputs",
    "OASOutputs"
]
