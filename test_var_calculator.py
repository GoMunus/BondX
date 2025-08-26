#!/usr/bin/env python3
"""
Comprehensive test suite for VaRCalculator.

Tests both parametric and historical VaR methods, validation, and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from decimal import Decimal

from bondx.risk_management.var_calculator import (
    VaRCalculator, VaRResult, Position, RiskFactor,
    VaRMethod, ConfidenceLevel
)


class TestVaRCalculator(unittest.TestCase):
    """Test cases for VaRCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = VaRCalculator()
        self.today = date.today()
        
        # Sample positions for testing
        self.positions = [
            Position(
                instrument_id="BOND_001",
                face_value=Decimal('1000000'),
                market_value=Decimal('980000'),
                duration=4.5,
                convexity=25.0,
                dv01=450.0,
                credit_spread=50.0,
                issuer_rating="AA",
                sector="FINANCIAL",
                maturity_date=self.today + timedelta(days=365*5),
                coupon_rate=0.05,
                yield_to_maturity=0.055
            ),
            Position(
                instrument_id="BOND_002",
                face_value=Decimal('500000'),
                market_value=Decimal('495000'),
                duration=2.8,
                convexity=12.0,
                dv01=280.0,
                credit_spread=75.0,
                issuer_rating="A",
                sector="CORPORATE",
                maturity_date=self.today + timedelta(days=365*3),
                coupon_rate=0.04,
                yield_to_maturity=0.047
            ),
            Position(
                instrument_id="BOND_003",
                face_value=Decimal('750000'),
                market_value=Decimal('760000'),
                duration=7.2,
                convexity=45.0,
                dv01=720.0,
                credit_spread=100.0,
                issuer_rating="BBB",
                sector="UTILITIES",
                maturity_date=self.today + timedelta(days=365*10),
                coupon_rate=0.06,
                yield_to_maturity=0.065
            )
        ]
        
        # Sample risk factors
        self.risk_factors = [
            RiskFactor(
                factor_id="RATE_2Y",
                factor_type="rate",
                tenor=2.0,
                current_value=0.05,
                volatility=0.002  # 20 bps daily volatility
            ),
            RiskFactor(
                factor_id="RATE_5Y",
                factor_type="rate",
                tenor=5.0,
                current_value=0.055,
                volatility=0.0025  # 25 bps daily volatility
            ),
            RiskFactor(
                factor_id="RATE_10Y",
                factor_type="rate",
                tenor=10.0,
                current_value=0.06,
                volatility=0.003  # 30 bps daily volatility
            ),
            RiskFactor(
                factor_id="SPREAD_AA",
                factor_type="spread",
                rating_bucket="AA",
                current_value=0.005,
                volatility=0.001  # 10 bps daily volatility
            ),
            RiskFactor(
                factor_id="SPREAD_A",
                factor_type="spread",
                rating_bucket="A",
                current_value=0.0075,
                volatility=0.0012  # 12 bps daily volatility
            ),
            RiskFactor(
                factor_id="CREDIT_BBB",
                factor_type="credit",
                rating_bucket="BBB",
                sector="UTILITIES",
                current_value=0.01,
                volatility=0.002  # 20 bps daily volatility
            )
        ]
        
        # Sample historical data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range(start=self.today - timedelta(days=252), end=self.today, freq='D')
        self.historical_data = pd.DataFrame(
            index=dates,
            data={
                "RATE_2Y": np.random.normal(0.05, 0.002, len(dates)),
                "RATE_5Y": np.random.normal(0.055, 0.0025, len(dates)),
                "RATE_10Y": np.random.normal(0.06, 0.003, len(dates)),
                "SPREAD_AA": np.random.normal(0.005, 0.001, len(dates)),
                "SPREAD_A": np.random.normal(0.0075, 0.0012, len(dates)),
                "CREDIT_BBB": np.random.normal(0.01, 0.002, len(dates))
            }
        )
    
    def test_parametric_var_basic(self):
        """Test basic parametric VaR calculation."""
        result = self.calculator.calculate_var(
            positions=self.positions,
            risk_factors=self.risk_factors,
            method=VaRMethod.PARAMETRIC,
            confidence_level=ConfidenceLevel.P95,
            time_horizon=1.0
        )
        
        # Basic validation
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.PARAMETRIC)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.time_horizon, 1.0)
        self.assertEqual(result.calculation_date, self.today)
        
        # VaR should be positive
        self.assertGreater(result.var_value, 0)
        
        # Portfolio value should match
        expected_portfolio_value = sum(float(pos.market_value) for pos in self.positions)
        self.assertAlmostEqual(result.portfolio_value, expected_portfolio_value, places=2)
        
        # Should have VaR contribution for each position
        self.assertEqual(len(result.var_contribution), len(self.positions))
        for pos in self.positions:
            self.assertIn(pos.instrument_id, result.var_contribution)
    
    def test_historical_var_basic(self):
        """Test basic historical VaR calculation."""
        result = self.calculator.calculate_var(
            positions=self.positions,
            risk_factors=self.risk_factors,
            method=VaRMethod.HISTORICAL,
            confidence_level=ConfidenceLevel.P95,
            time_horizon=1.0,
            historical_data=self.historical_data
        )
        
        # Basic validation
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.HISTORICAL)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.time_horizon, 1.0)
        
        # VaR should be positive
        self.assertGreater(result.var_value, 0)
        
        # Should have VaR contribution for each position
        self.assertEqual(len(result.var_contribution), len(self.positions))
    
    def test_different_confidence_levels(self):
        """Test VaR calculation with different confidence levels."""
        confidence_levels = [ConfidenceLevel.P95, ConfidenceLevel.P99, ConfidenceLevel.P99_9]
        
        for conf_level in confidence_levels:
            result = self.calculator.calculate_var(
                positions=self.positions,
                risk_factors=self.risk_factors,
                method=VaRMethod.PARAMETRIC,
                confidence_level=conf_level,
                time_horizon=1.0
            )
            
            # Higher confidence should give higher VaR
            self.assertEqual(result.confidence_level, conf_level.value)
            self.assertGreater(result.var_value, 0)
    
    def test_different_time_horizons(self):
        """Test VaR calculation with different time horizons."""
        time_horizons = [1.0, 5.0, 10.0]  # 1 day, 5 days, 10 days
        
        for horizon in time_horizons:
            result = self.calculator.calculate_var(
                positions=self.positions,
                risk_factors=self.risk_factors,
                method=VaRMethod.PARAMETRIC,
                confidence_level=ConfidenceLevel.P95,
                time_horizon=horizon
            )
            
            # Longer horizon should give higher VaR (approximately sqrt scaling)
            self.assertEqual(result.time_horizon, horizon)
            self.assertGreater(result.var_value, 0)
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Empty positions
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=[],
                risk_factors=self.risk_factors,
                method=VaRMethod.PARAMETRIC
            )
        
        # Empty risk factors
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=self.positions,
                risk_factors=[],
                method=VaRMethod.PARAMETRIC
            )
        
        # Duplicate instrument IDs
        duplicate_positions = self.positions + [self.positions[0]]
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=duplicate_positions,
                risk_factors=self.risk_factors,
                method=VaRMethod.PARAMETRIC
            )
        
        # Duplicate factor IDs
        duplicate_factors = self.risk_factors + [self.risk_factors[0]]
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=self.positions,
                risk_factors=duplicate_factors,
                method=VaRMethod.PARAMETRIC
            )
        
        # Invalid position data
        invalid_position = Position(
            instrument_id="INVALID",
            face_value=Decimal('0'),  # Invalid
            market_value=Decimal('100000'),
            duration=4.5,
            convexity=25.0,
            dv01=450.0,
            credit_spread=50.0,
            issuer_rating="AA",
            sector="FINANCIAL",
            maturity_date=self.today + timedelta(days=365*5),
            coupon_rate=0.05,
            yield_to_maturity=0.055
        )
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=[invalid_position],
                risk_factors=self.risk_factors,
                method=VaRMethod.PARAMETRIC
            )
        
        # Invalid risk factor data
        invalid_factor = RiskFactor(
            factor_id="INVALID",
            factor_type="rate",
            tenor=2.0,
            current_value=0.05,
            volatility=-0.002  # Invalid negative volatility
        )
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=self.positions,
                risk_factors=[invalid_factor],
                method=VaRMethod.PARAMETRIC
            )
        
        # Historical VaR without historical data
        with self.assertRaises(ValueError):
            self.calculator.calculate_var(
                positions=self.positions,
                risk_factors=self.risk_factors,
                method=VaRMethod.HISTORICAL
            )
    
    def test_correlation_estimation(self):
        """Test correlation estimation between risk factors."""
        # Test rate factor correlation
        rate_factor1 = RiskFactor(
            factor_id="RATE_1",
            factor_type="rate",
            tenor=2.0,
            current_value=0.05,
            volatility=0.002
        )
        rate_factor2 = RiskFactor(
            factor_id="RATE_2",
            factor_type="rate",
            tenor=5.0,
            current_value=0.055,
            volatility=0.0025
        )
        
        correlation = self.calculator._estimate_correlation(rate_factor1, rate_factor2)
        self.assertAlmostEqual(correlation, 0.8, places=1)  # High correlation for rates
        
        # Test different factor types
        spread_factor = RiskFactor(
            factor_id="SPREAD_1",
            factor_type="spread",
            rating_bucket="AA",
            current_value=0.005,
            volatility=0.001
        )
        
        correlation = self.calculator._estimate_correlation(rate_factor1, spread_factor)
        self.assertAlmostEqual(correlation, 0.3, places=1)  # Lower correlation for different types
    
    def test_credit_correlation_estimation(self):
        """Test credit correlation estimation."""
        # Same rating
        factor1 = RiskFactor(
            factor_id="CREDIT_1",
            factor_type="credit",
            rating_bucket="AA",
            current_value=0.005,
            volatility=0.001
        )
        factor2 = RiskFactor(
            factor_id="CREDIT_2",
            factor_type="credit",
            rating_bucket="AA",
            current_value=0.005,
            volatility=0.001
        )
        
        correlation = self.calculator._estimate_credit_correlation(factor1, factor2)
        self.assertAlmostEqual(correlation, 0.9, places=1)  # High correlation for same rating
        
        # Adjacent ratings
        factor3 = RiskFactor(
            factor_id="CREDIT_3",
            factor_type="credit",
            rating_bucket="A",
            current_value=0.0075,
            volatility=0.0012
        )
        
        correlation = self.calculator._estimate_credit_correlation(factor1, factor3)
        self.assertAlmostEqual(correlation, 0.7, places=1)  # Moderate correlation for adjacent ratings
        
        # Distant ratings
        factor4 = RiskFactor(
            factor_id="CREDIT_4",
            factor_type="credit",
            rating_bucket="BBB",
            current_value=0.01,
            volatility=0.002
        )
        
        correlation = self.calculator._estimate_credit_correlation(factor1, factor4)
        self.assertAlmostEqual(correlation, 0.3, places=1)  # Low correlation for distant ratings
    
    def test_portfolio_sensitivities(self):
        """Test portfolio sensitivity calculations."""
        sensitivities = self.calculator._calculate_portfolio_sensitivities(
            self.positions, self.risk_factors
        )
        
        # Should have sensitivity for each risk factor
        self.assertEqual(len(sensitivities), len(self.risk_factors))
        
        # Sensitivities should be non-negative
        for sensitivity in sensitivities:
            self.assertGreaterEqual(sensitivity, 0)
    
    def test_covariance_matrix_construction(self):
        """Test covariance matrix construction."""
        cov_matrix = self.calculator._build_covariance_matrix(self.risk_factors)
        
        # Should be square matrix
        n_factors = len(self.risk_factors)
        self.assertEqual(cov_matrix.shape, (n_factors, n_factors))
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)
        
        # Diagonal elements should be variances (positive)
        for i in range(n_factors):
            self.assertGreater(cov_matrix[i, i], 0)
    
    def test_z_score_calculation(self):
        """Test z-score calculation for different confidence levels."""
        # Test 95% confidence
        z_score = self.calculator._get_z_score(ConfidenceLevel.P95)
        self.assertAlmostEqual(z_score, 1.645, places=3)
        
        # Test 99% confidence
        z_score = self.calculator._get_z_score(ConfidenceLevel.P99)
        self.assertAlmostEqual(z_score, 2.326, places=3)
        
        # Test 99.9% confidence
        z_score = self.calculator._get_z_score(ConfidenceLevel.P99_9)
        self.assertAlmostEqual(z_score, 3.090, places=3)
        
        # Test default case
        z_score = self.calculator._get_z_score(ConfidenceLevel.P95)
        self.assertAlmostEqual(z_score, 1.645, places=3)
    
    def test_historical_factor_returns(self):
        """Test historical factor returns calculation."""
        factor_returns = self.calculator._get_historical_factor_returns(
            self.risk_factors, self.historical_data
        )
        
        # Should have returns for each factor
        self.assertGreater(len(factor_returns.columns), 0)
        
        # Returns should be numeric
        for col in factor_returns.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(factor_returns[col]))
    
    def test_historical_pnl_scenarios(self):
        """Test historical P&L scenario calculation."""
        factor_returns = self.calculator._get_historical_factor_returns(
            self.risk_factors, self.historical_data
        )
        
        # Test approximate P&L
        pnl_scenarios = self.calculator._calculate_historical_pnl_scenarios(
            self.positions, self.risk_factors, factor_returns, use_full_repricing=False
        )
        
        # Should have P&L for each scenario
        self.assertEqual(len(pnl_scenarios), len(factor_returns))
        
        # Test full repricing P&L
        pnl_scenarios_full = self.calculator._calculate_historical_pnl_scenarios(
            self.positions, self.risk_factors, factor_returns, use_full_repricing=True
        )
        
        # Should have same number of scenarios
        self.assertEqual(len(pnl_scenarios_full), len(pnl_scenarios))
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation."""
        # Create sample P&L scenarios
        pnl_scenarios = np.random.normal(0, 1000, 1000)  # Normal distribution
        
        # Calculate CVaR at 95% confidence
        var_percentile = 5.0  # 5th percentile for 95% confidence
        cvar = self.calculator.calculate_cvar(pnl_scenarios, var_percentile)
        
        # CVaR should be less than or equal to VaR
        var_value = np.percentile(pnl_scenarios, var_percentile)
        self.assertLessEqual(cvar, var_value)
    
    def test_var_backtesting(self):
        """Test VaR backtesting functionality."""
        # Create sample VaR predictions and actual P&L
        np.random.seed(42)
        var_predictions = [1000.0] * 100  # Constant VaR prediction
        actual_pnl = np.random.normal(0, 800, 100)  # Actual P&L
        
        # Perform backtest
        backtest_result = self.calculator.backtest_var(
            var_predictions, actual_pnl.tolist(), ConfidenceLevel.P95
        )
        
        # Check backtest result structure
        required_fields = [
            "expected_violations", "actual_violations", "violation_rate",
            "expected_rate", "test_statistic", "critical_value", "reject_null"
        ]
        
        for field in required_fields:
            self.assertIn(field, backtest_result)
        
        # Check violation rate calculation
        expected_violations = (1 - 0.95) * 100  # 5% of 100
        self.assertAlmostEqual(backtest_result["expected_violations"], expected_violations)
        
        # Check test statistic calculation
        self.assertGreaterEqual(backtest_result["test_statistic"], 0)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single position
        single_position = [self.positions[0]]
        result = self.calculator.calculate_var(
            positions=single_position,
            risk_factors=self.risk_factors,
            method=VaRMethod.PARAMETRIC,
            confidence_level=ConfidenceLevel.P95
        )
        
        # Should still work
        self.assertIsInstance(result, VaRResult)
        self.assertGreater(result.var_value, 0)
        
        # Single risk factor
        single_factor = [self.risk_factors[0]]
        result = self.calculator.calculate_var(
            positions=self.positions,
            risk_factors=single_factor,
            method=VaRMethod.PARAMETRIC,
            confidence_level=ConfidenceLevel.P95
        )
        
        # Should still work
        self.assertIsInstance(result, VaRResult)
        self.assertGreater(result.var_value, 0)
    
    def test_performance_requirements(self):
        """Test performance requirements."""
        # Create larger portfolio
        large_positions = []
        for i in range(100):
            position = Position(
                instrument_id=f"BOND_{i:03d}",
                face_value=Decimal('100000'),
                market_value=Decimal('98000'),
                duration=4.0 + (i % 5),
                convexity=20.0 + (i % 10),
                dv01=400.0 + (i % 100),
                credit_spread=50.0 + (i % 50),
                issuer_rating="AA" if i % 3 == 0 else "A" if i % 3 == 1 else "BBB",
                sector="FINANCIAL" if i % 2 == 0 else "CORPORATE",
                maturity_date=self.today + timedelta(days=365*(3 + i % 7)),
                coupon_rate=0.04 + (i % 20) / 1000,
                yield_to_maturity=0.045 + (i % 20) / 1000
            )
            large_positions.append(position)
        
        # Should calculate VaR efficiently
        import time
        start_time = time.time()
        
        result = self.calculator.calculate_var(
            positions=large_positions,
            risk_factors=self.risk_factors,
            method=VaRMethod.PARAMETRIC,
            confidence_level=ConfidenceLevel.P95
        )
        
        end_time = time.time()
        calculation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete under 100ms
        self.assertLess(calculation_time, 100, f"Large portfolio VaR took {calculation_time:.2f}ms")
        
        # Should have correct result
        self.assertIsInstance(result, VaRResult)
        self.assertGreater(result.var_value, 0)


if __name__ == '__main__':
    unittest.main()
