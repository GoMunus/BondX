"""
Comprehensive test suite for BondX Phase B components.

This module tests the integration of OAS Calculator, Stress Testing Engine,
and Portfolio Analytics components.
"""

import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
import numpy as np

from bondx.mathematics.option_adjusted_spread import (
    OASCalculator, OASInputs, OptionType, PricingMethod, LatticeModel,
    CallSchedule, PutSchedule, VolatilitySurface
)
from bondx.risk_management.stress_testing import (
    StressTestingEngine, Position, StressScenario, CalculationMode,
    RatingBucket, SectorBucket, ScenarioType
)
from bondx.risk_management.portfolio_analytics import (
    PortfolioAnalytics, PortfolioMetrics, AttributionResult, TurnoverMetrics
)
from bondx.core.model_contracts import (
    ModelResultStore, ModelValidator, ModelType, ModelStatus
)
from bondx.mathematics.yield_curves import (
    YieldCurve, CurveType, CurveConstructionConfig
)


class TestPhaseBIntegration(unittest.TestCase):
    """Test suite for Phase B component integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test yield curve
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        rates = np.array([0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085])
        
        config = CurveConstructionConfig()
        self.test_curve = YieldCurve(
            curve_type=CurveType.ZERO_CURVE,
            tenors=tenors,
            rates=rates,
            construction_date=date.today(),
            config=config
        )
        
        # Create test volatility surface
        self.test_vol_surface = VolatilitySurface(
            tenors=tenors,
            volatilities=np.array([0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.27])
        )
        
        # Create test positions
        self.test_positions = [
            Position(
                instrument_id="BOND_001",
                face_value=Decimal("1000000"),
                book_value=Decimal("980000"),
                market_value=Decimal("985000"),
                coupon_rate=Decimal("0.06"),
                maturity_date=date.today() + timedelta(days=1825),  # 5 years
                duration=4.5,
                convexity=25.0,
                spread_dv01=0.8,
                liquidity_score=0.8,
                issuer_id="ISSUER_001",
                sector=SectorBucket.CORPORATE,
                rating=RatingBucket.A,
                tenor_bucket="3-5Y",
                oas_sensitive=False
            ),
            Position(
                instrument_id="BOND_002",
                face_value=Decimal("2000000"),
                book_value=Decimal("1950000"),
                market_value=Decimal("1980000"),
                coupon_rate=Decimal("0.07"),
                maturity_date=date.today() + timedelta(days=3650),  # 10 years
                duration=8.2,
                convexity=75.0,
                spread_dv01=1.2,
                liquidity_score=0.6,
                issuer_id="ISSUER_002",
                sector=SectorBucket.FINANCIAL,
                rating=RatingBucket.BBB,
                tenor_bucket="5-10Y",
                oas_sensitive=True
            )
        ]
        
        # Initialize components
        self.oas_calculator = OASCalculator(
            pricing_method=PricingMethod.LATTICE,
            lattice_model=LatticeModel.HO_LEE,
            lattice_steps=100,  # Reduced for testing
            convergence_tolerance=1e-4
        )
        
        self.stress_engine = StressTestingEngine(
            calculation_mode=CalculationMode.FAST_APPROXIMATION,
            parallel_processing=False,  # Disable for testing
            cache_results=True
        )
        
        self.portfolio_analytics = PortfolioAnalytics(
            enable_pca=True,
            curve_factors=3,
            attribution_method="FACTOR_MODEL"
        )
        
        self.model_store = ModelResultStore(
            enable_caching=True,
            max_cache_size=100
        )
        
        self.model_validator = ModelValidator(strict_mode=False)
    
    def test_oas_calculator_basic_functionality(self):
        """Test basic OAS calculator functionality."""
        # Create test cash flows (simplified)
        cash_flows = []  # Would be populated with actual cash flows
        
        # Create OAS inputs
        oas_inputs = OASInputs(
            base_curve=self.test_curve,
            volatility_surface=self.test_vol_surface,
            cash_flows=cash_flows,
            option_type=OptionType.NONE,  # Start with option-free
            market_price=Decimal("98.50")
        )
        
        # Calculate OAS
        result = self.oas_calculator.calculate_oas(oas_inputs)
        
        # Basic validation
        self.assertIsNotNone(result)
        self.assertIsInstance(result.oas_bps, float)
        self.assertIsInstance(result.model_pv, Decimal)
        self.assertEqual(result.convergence_status, "CONVERGED")
        self.assertGreater(result.iterations, 0)
    
    def test_oas_calculator_callable_bond(self):
        """Test OAS calculation for callable bond."""
        # Create call schedule
        call_schedule = [
            CallSchedule(
                call_date=date.today() + timedelta(days=1095),  # 3 years
                call_price=Decimal("100.0")
            )
        ]
        
        # Create OAS inputs for callable bond
        oas_inputs = OASInputs(
            base_curve=self.test_curve,
            volatility_surface=self.test_vol_surface,
            cash_flows=[],  # Simplified
            option_type=OptionType.CALLABLE,
            call_schedule=call_schedule,
            market_price=Decimal("99.50")
        )
        
        # Calculate OAS
        result = self.oas_calculator.calculate_oas(oas_inputs)
        
        # Validation for callable bond
        self.assertIsNotNone(result)
        self.assertIsInstance(result.option_value, Decimal)
        self.assertGreater(result.option_adjusted_duration, 0)
        self.assertGreater(result.option_adjusted_convexity, 0)
    
    def test_stress_testing_basic_scenarios(self):
        """Test basic stress testing scenarios."""
        # Create test scenarios
        scenarios = [
            StressScenario(
                scenario_id="TEST_PARALLEL_50",
                scenario_type=ScenarioType.PARLEL_SHIFT,
                name="Test Parallel +50bps",
                description="Test parallel rate shift",
                parallel_shift_bps=50
            ),
            StressScenario(
                scenario_id="TEST_CREDIT_100",
                scenario_type=ScenarioType.CREDIT_SPREAD_BLOWOUT,
                name="Test Credit +100bps",
                description="Test credit spread blowout",
                credit_spread_shocks={
                    RatingBucket.A: 100,
                    RatingBucket.BBB: 150
                }
            )
        ]
        
        # Get base curves and spread surfaces
        base_curves = {"INR": self.test_curve}
        spread_surfaces = {
            rating: {"3-5Y": 0.002, "5-10Y": 0.003}
            for rating in [RatingBucket.A, RatingBucket.BBB]
        }
        
        # Run stress tests
        results = self.stress_engine.run_multiple_scenarios(
            portfolio=self.test_positions,
            base_curves=base_curves,
            spread_surfaces=spread_surfaces,
            scenarios=scenarios
        )
        
        # Validation
        self.assertEqual(len(results), len(scenarios))
        for result in results:
            self.assertIsNotNone(result.total_pnl)
            self.assertIsNotNone(result.total_pnl_bps)
            self.assertIsNotNone(result.delta_dv01)
    
    def test_stress_testing_performance_targets(self):
        """Test stress testing performance targets."""
        # Create larger portfolio for performance testing
        large_portfolio = self.test_positions * 50  # 100 positions
        
        # Create simple scenario
        scenario = StressScenario(
            scenario_id="PERF_TEST",
            scenario_type=ScenarioType.PARLEL_SHIFT,
            name="Performance Test",
            description="Performance test scenario",
            parallel_shift_bps=100
        )
        
        # Measure performance
        start_time = datetime.now()
        
        results = self.stress_engine.run_multiple_scenarios(
            portfolio=large_portfolio,
            base_curves={"INR": self.test_curve},
            spread_surfaces={},
            scenarios=[scenario]
        )
        
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance validation (fast mode target: ≤100ms for 10,000 positions)
        # For 100 positions, should be much faster
        self.assertLess(execution_time_ms, 50)  # Should be < 50ms for 100 positions
        
        # Results validation
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].total_pnl)
    
    def test_portfolio_analytics_metrics(self):
        """Test portfolio analytics metrics calculation."""
        # Calculate portfolio metrics
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            positions=self.test_positions
        )
        
        # Basic validation
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.positions_count, len(self.test_positions))
        self.assertIsInstance(metrics.total_market_value, Decimal)
        self.assertIsInstance(metrics.portfolio_duration, float)
        self.assertIsInstance(metrics.portfolio_convexity, float)
        
        # Value validation
        expected_market_value = sum(pos.market_value for pos in self.test_positions)
        self.assertEqual(metrics.total_market_value, expected_market_value)
        
        # Duration validation (weighted average)
        total_value = float(expected_market_value)
        expected_duration = sum(
            float(pos.market_value) * pos.duration for pos in self.test_positions
        ) / total_value
        self.assertAlmostEqual(metrics.portfolio_duration, expected_duration, places=2)
        
        # Concentration validation
        self.assertIsInstance(metrics.issuer_concentration, dict)
        self.assertIsInstance(metrics.sector_concentration, dict)
        self.assertIsInstance(metrics.rating_concentration, dict)
        self.assertIsInstance(metrics.tenor_concentration, dict)
    
    def test_portfolio_analytics_attribution(self):
        """Test portfolio analytics performance attribution."""
        # Create end-of-period positions (with some changes)
        positions_end = []
        for pos in self.test_positions:
            # Simulate some changes
            new_pos = Position(
                instrument_id=pos.instrument_id,
                face_value=pos.face_value,
                book_value=pos.book_value,
                market_value=pos.market_value * Decimal("1.02"),  # 2% increase
                coupon_rate=pos.coupon_rate,
                maturity_date=pos.maturity_date,
                duration=pos.duration,
                convexity=pos.convexity,
                spread_dv01=pos.spread_dv01,
                liquidity_score=pos.liquidity_score,
                issuer_id=pos.issuer_id,
                sector=pos.sector,
                rating=pos.rating,
                tenor_bucket=pos.tenor_bucket,
                oas_sensitive=pos.oas_sensitive
            )
            positions_end.append(new_pos)
        
        # Create yield curves for start and end
        yield_curves_start = {"INR": self.test_curve}
        yield_curves_end = {"INR": self.test_curve}  # Same for testing
        
        # Calculate attribution
        attribution = self.portfolio_analytics.calculate_performance_attribution(
            positions_start=self.test_positions,
            positions_end=positions_end,
            yield_curves_start=yield_curves_start,
            yield_curves_end=yield_curves_end,
            period_start=date.today() - timedelta(days=30),
            period_end=date.today()
        )
        
        # Basic validation
        self.assertIsNotNone(attribution)
        self.assertIsInstance(attribution.total_return, float)
        self.assertIsInstance(attribution.factor_contributions, dict)
        self.assertIsInstance(attribution.curve_level_contribution, float)
        self.assertIsInstance(attribution.credit_spread_contribution, float)
        
        # Return validation
        self.assertGreater(attribution.total_return, 0)  # Should be positive due to 2% increase
    
    def test_portfolio_analytics_turnover(self):
        """Test portfolio analytics turnover metrics."""
        # Create end-of-period positions with some changes
        positions_end = self.test_positions.copy()
        
        # Modify one position
        positions_end[0] = Position(
            instrument_id="BOND_001_MODIFIED",
            face_value=Decimal("1100000"),  # Increased
            book_value=Decimal("1080000"),
            market_value=Decimal("1085000"),
            coupon_rate=Decimal("0.065"),  # Changed
            maturity_date=positions_end[0].maturity_date,
            duration=4.8,  # Changed
            convexity=26.0,  # Changed
            spread_dv01=0.85,  # Changed
            liquidity_score=0.8,
            issuer_id="ISSUER_001",
            sector=SectorBucket.CORPORATE,
            rating=RatingBucket.A,
            tenor_bucket="3-5Y",
            oas_sensitive=False
        )
        
        # Calculate turnover metrics
        turnover = self.portfolio_analytics.calculate_turnover_metrics(
            positions_start=self.test_positions,
            positions_end=positions_end,
            period_start=date.today() - timedelta(days=30),
            period_end=date.today()
        )
        
        # Basic validation
        self.assertIsNotNone(turnover)
        self.assertIsInstance(turnover.gross_turnover, float)
        self.assertIsInstance(turnover.net_turnover, float)
        self.assertIsInstance(turnover.positions_modified, int)
        
        # Turnover validation
        self.assertGreater(turnover.gross_turnover, 0)
        self.assertGreater(turnover.positions_modified, 0)
        self.assertEqual(turnover.positions_modified, 1)  # One position modified
    
    def test_model_result_store_functionality(self):
        """Test model result store functionality."""
        # Create test model result
        from bondx.core.model_contracts import ModelInputs, ModelOutputs
        
        inputs = ModelInputs(
            model_type=ModelType.OAS_CALCULATION,
            input_hash="test_hash",
            config_version="1.0"
        )
        
        outputs = ModelOutputs(
            model_type=ModelType.OAS_CALCULATION,
            outputs={"oas_bps": 150.5},
            diagnostics={"iterations": 25},
            execution_time_ms=45.2,
            status=ModelStatus.COMPLETED
        )
        
        from bondx.core.model_contracts import ModelResult
        
        result = ModelResult(
            model_type=ModelType.OAS_CALCULATION,
            inputs=inputs,
            outputs=outputs,
            model_id="TEST_MODEL_001",
            execution_id="TEST_EXEC_001"
        )
        
        # Store result
        cache_key = self.model_store.store_result(result)
        self.assertIsNotNone(cache_key)
        
        # Retrieve result
        retrieved_result = self.model_store.retrieve_result(cache_key)
        self.assertIsNotNone(retrieved_result)
        self.assertEqual(retrieved_result.model_id, "TEST_MODEL_001")
        
        # Search results
        search_results = self.model_store.search_results(
            model_type=ModelType.OAS_CALCULATION
        )
        self.assertGreater(len(search_results), 0)
        
        # Get cache stats
        cache_stats = self.model_store.get_cache_stats()
        self.assertIsInstance(cache_stats, dict)
        self.assertIn('cache_size', cache_stats)
        self.assertIn('hit_rate', cache_stats)
    
    def test_model_validator_functionality(self):
        """Test model validator functionality."""
        # Create test inputs
        from bondx.core.model_contracts import ModelInputs
        
        inputs = ModelInputs(
            model_type=ModelType.OAS_CALCULATION,
            input_hash="test_hash",
            config_version="1.0"
        )
        
        # Validate inputs
        warnings = self.model_validator.validate_inputs(inputs)
        self.assertIsInstance(warnings, list)
        
        # Create test outputs
        from bondx.core.model_contracts import ModelOutputs
        
        outputs = ModelOutputs(
            model_type=ModelType.OAS_CALCULATION,
            outputs={"test": "data"},
            diagnostics={},
            execution_time_ms=100.0,
            status=ModelStatus.COMPLETED
        )
        
        # Validate outputs
        warnings = self.model_validator.validate_outputs(outputs)
        self.assertIsInstance(warnings, list)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Calculate OAS for a callable bond
        call_schedule = [
            CallSchedule(
                call_date=date.today() + timedelta(days=1095),
                call_price=Decimal("100.0")
            )
        ]
        
        oas_inputs = OASInputs(
            base_curve=self.test_curve,
            volatility_surface=self.test_vol_surface,
            cash_flows=[],
            option_type=OptionType.CALLABLE,
            call_schedule=call_schedule,
            market_price=Decimal("99.50")
        )
        
        oas_result = self.oas_calculator.calculate_oas(oas_inputs)
        self.assertIsNotNone(oas_result)
        
        # 2. Run stress test on portfolio
        stress_scenario = StressScenario(
            scenario_id="E2E_TEST",
            scenario_type=ScenarioType.PARLEL_SHIFT,
            name="End-to-End Test",
            description="End-to-end workflow test",
            parallel_shift_bps=100
        )
        
        stress_results = self.stress_engine.run_multiple_scenarios(
            portfolio=self.test_positions,
            base_curves={"INR": self.test_curve},
            spread_surfaces={},
            scenarios=[stress_scenario]
        )
        
        self.assertEqual(len(stress_results), 1)
        self.assertIsNotNone(stress_results[0].total_pnl)
        
        # 3. Calculate portfolio analytics
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            positions=self.test_positions
        )
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.positions_count, len(self.test_positions))
        
        # 4. Store results in model store
        from bondx.core.model_contracts import ModelInputs, ModelOutputs, ModelResult
        
        # Store OAS result
        oas_inputs_model = ModelInputs(
            model_type=ModelType.OAS_CALCULATION,
            input_hash="e2e_test_hash",
            config_version="1.0"
        )
        
        oas_outputs_model = ModelOutputs(
            model_type=ModelType.OAS_CALCULATION,
            outputs=oas_result,
            diagnostics={"test": "e2e"},
            execution_time_ms=oas_result.solve_time_ms,
            status=ModelStatus.COMPLETED
        )
        
        oas_model_result = ModelResult(
            model_type=ModelType.OAS_CALCULATION,
            inputs=oas_inputs_model,
            outputs=oas_outputs_model,
            model_id="E2E_OAS_001",
            execution_id="E2E_EXEC_001"
        )
        
        cache_key = self.model_store.store_result(oas_model_result)
        self.assertIsNotNone(cache_key)
        
        # Verify workflow completion
        retrieved_result = self.model_store.retrieve_result(cache_key)
        self.assertIsNotNone(retrieved_result)
        self.assertEqual(retrieved_result.model_id, "E2E_OAS_001")
    
    def test_error_handling_and_validation(self):
        """Test error handling and validation."""
        # Test invalid OAS inputs
        with self.assertRaises(Exception):
            invalid_inputs = OASInputs(
                base_curve=None,  # Invalid
                volatility_surface=self.test_vol_surface,
                cash_flows=[],
                option_type=OptionType.CALLABLE,
                call_schedule=[],
                market_price=Decimal("99.50")
            )
            self.oas_calculator.calculate_oas(invalid_inputs)
        
        # Test invalid stress test inputs
        with self.assertRaises(Exception):
            invalid_scenario = StressScenario(
                scenario_id="INVALID",
                scenario_type=ScenarioType.PARLEL_SHIFT,
                name="Invalid",
                description="Invalid scenario",
                parallel_shift_bps=10000  # Too large
            )
            # This should fail validation
        
        # Test invalid portfolio analytics inputs
        with self.assertRaises(Exception):
            self.portfolio_analytics.calculate_portfolio_metrics(positions=[])
    
    def test_performance_under_load(self):
        """Test performance under load conditions."""
        # Create larger test portfolio
        large_portfolio = []
        for i in range(100):  # 100 positions
            pos = Position(
                instrument_id=f"BOND_{i:03d}",
                face_value=Decimal("1000000"),
                book_value=Decimal("980000"),
                market_value=Decimal("985000"),
                coupon_rate=Decimal("0.06"),
                maturity_date=date.today() + timedelta(days=1825 + i * 365),
                duration=4.5 + (i % 10) * 0.5,
                convexity=25.0 + (i % 10) * 5.0,
                spread_dv01=0.8 + (i % 5) * 0.1,
                liquidity_score=0.5 + (i % 10) * 0.05,
                issuer_id=f"ISSUER_{i % 20:02d}",
                sector=list(SectorBucket)[i % len(SectorBucket)],
                rating=list(RatingBucket)[i % len(RatingBucket)],
                tenor_bucket="3-5Y" if i % 2 == 0 else "5-10Y",
                oas_sensitive=i % 3 == 0
            )
            large_portfolio.append(pos)
        
        # Test portfolio metrics performance
        start_time = datetime.now()
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            positions=large_portfolio
        )
        end_time = datetime.now()
        
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance validation: should be < 50ms for 100 positions
        self.assertLess(execution_time_ms, 50)
        self.assertEqual(metrics.positions_count, 100)
        
        # Test stress testing performance
        stress_scenario = StressScenario(
            scenario_id="PERF_LOAD_TEST",
            scenario_type=ScenarioType.PARLEL_SHIFT,
            name="Performance Load Test",
            description="Performance test under load",
            parallel_shift_bps=100
        )
        
        start_time = datetime.now()
        stress_results = self.stress_engine.run_multiple_scenarios(
            portfolio=large_portfolio,
            base_curves={"INR": self.test_curve},
            spread_surfaces={},
            scenarios=[stress_scenario]
        )
        end_time = datetime.now()
        
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance validation: should be < 100ms for 100 positions in fast mode
        self.assertLess(execution_time_ms, 100)
        self.assertEqual(len(stress_results), 1)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
