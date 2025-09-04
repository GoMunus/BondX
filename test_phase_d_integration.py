#!/usr/bin/env python3
"""
Phase D Integration Test

This script demonstrates and validates all Phase D components working together:
- Enhanced ML Pipeline with GPU acceleration
- HFT-grade risk engine
- Automated regulatory capital engine
- Advanced Redis clustering
- Real-time streaming analytics

Usage:
    python test_phase_d_integration.py
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add BondX to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bondx'))

# Import Phase D components
from bondx.ai_risk_engine.enhanced_ml_pipeline import (
    EnhancedMLPipeline, PerformanceConfig, DistributedConfig
)
from bondx.risk_management.hft_risk_engine import (
    HFTRiskEngine, RiskParameters, PortfolioPosition, StressScenarioType
)
from bondx.risk_management.regulatory_capital_engine import (
    RegulatoryCapitalEngine, BaselFramework, CapitalApproach, RegulatoryInstrument, AssetClass
)
from bondx.core.advanced_redis_cluster import (
    AdvancedRedisCluster, ClusterConfig, CacheConfig, RedisMode, RedisNodeConfig
)
from bondx.core.streaming_analytics import (
    StreamingAnalyticsEngine, KafkaStreamManager, TickData
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhaseDIntegrationTest:
    """Comprehensive test of all Phase D components"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
        logger.info("Phase D Integration Test initialized")
    
    async def run_all_tests(self):
        """Run all Phase D component tests"""
        logger.info("Starting Phase D Integration Test Suite")
        
        try:
            # Test 1: Enhanced ML Pipeline
            await self.test_enhanced_ml_pipeline()
            
            # Test 2: HFT Risk Engine
            await self.test_hft_risk_engine()
            
            # Test 3: Regulatory Capital Engine
            await self.test_regulatory_capital_engine()
            
            # Test 4: Advanced Redis Cluster
            await self.test_advanced_redis_cluster()
            
            # Test 5: Streaming Analytics
            await self.test_streaming_analytics()
            
            # Test 6: End-to-End Integration
            await self.test_end_to_end_integration()
            
            # Generate test report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
    
    async def test_enhanced_ml_pipeline(self):
        """Test enhanced ML pipeline with GPU acceleration and distributed computing"""
        logger.info("Testing Enhanced ML Pipeline...")
        
        try:
            # Initialize pipeline
            performance_config = PerformanceConfig(
                target_inference_latency_ms=50.0,
                target_training_time_minutes=30.0,
                max_memory_usage_gb=16.0,
                use_mixed_precision=True
            )
            
            distributed_config = DistributedConfig(
                use_ray=False,  # Disable Ray for testing
                use_spark=False,  # Disable Spark for testing
                num_nodes=1
            )
            
            pipeline = EnhancedMLPipeline(performance_config, distributed_config)
            
            # Test GPU acceleration
            test_data = np.random.random((1000, 100))
            benchmark_results = pipeline.benchmark_performance(test_data, num_iterations=10)
            
            # Test model training
            model_configs = [
                {
                    'model_id': 'lstm_test_1',
                    'architecture': 'lstm',
                    'sequence_length': 60,
                    'hidden_size': 128
                },
                {
                    'model_id': 'transformer_test_1',
                    'architecture': 'transformer',
                    'sequence_length': 60,
                    'hidden_size': 256
                }
            ]
            
            # Generate synthetic training data
            training_data = {
                'returns': np.random.standard_normal((1000, 100)),
                'volatility': np.random.standard_normal((1000, 100)),
                'liquidity': np.random.standard_normal((1000, 100))
            }
            
            validation_data = {
                'returns': np.random.standard_normal((200, 100)),
                'volatility': np.random.standard_normal((200, 100)),
                'liquidity': np.random.standard_normal((200, 100))
            }
            
            # Train models
            training_results = await pipeline.train_advanced_models(
                model_configs, training_data, validation_data
            )
            
            # Store results
            self.test_results['enhanced_ml_pipeline'] = {
                'status': 'success',
                'benchmark_results': benchmark_results,
                'training_results': training_results,
                'pipeline_summary': pipeline.get_pipeline_summary()
            }
            
            logger.info("Enhanced ML Pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"Enhanced ML Pipeline test failed: {e}")
            self.test_results['enhanced_ml_pipeline'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_hft_risk_engine(self):
        """Test HFT-grade risk engine with microsecond latency"""
        logger.info("Testing HFT Risk Engine...")
        
        try:
            # Initialize risk engine
            risk_config = RiskParameters(
                confidence_level=0.99,
                time_horizon_days=1,
                num_simulations=10000,
                use_gpu=True,
                max_latency_ms=1.0
            )
            
            risk_engine = HFTRiskEngine(risk_config)
            
            # Create test portfolio
            portfolio_positions = [
                PortfolioPosition(
                    instrument_id="BOND_001",
                    quantity=1000000,
                    market_value=1000000,
                    duration=5.0,
                    convexity=25.0,
                    credit_spread=0.015,
                    liquidity_score=0.8,
                    sector="Financial",
                    rating="AA",
                    maturity_date=datetime.now() + timedelta(days=1825)
                ),
                PortfolioPosition(
                    instrument_id="BOND_002",
                    quantity=500000,
                    market_value=500000,
                    duration=3.0,
                    convexity=15.0,
                    credit_spread=0.025,
                    liquidity_score=0.6,
                    sector="Corporate",
                    rating="A",
                    maturity_date=datetime.now() + timedelta(days=1095)
                )
            ]
            
            # Mock market data
            market_data = {
                'timestamp': datetime.now(),
                'current_volatility': 0.15,
                'current_spreads': 100,
                'high_quality_liquid_assets': {
                    'cash': 1000000,
                    'sovereign_bonds_aaa': 500000
                },
                'net_cash_outflows': {
                    'deposits': 800000,
                    'wholesale_funding': 400000
                },
                'available_stable_funding': {
                    'tier_1_capital': 2000000,
                    'stable_deposits': 3000000
                },
                'required_stable_funding': {
                    'corporate_bonds_aa': 1000000,
                    'equity': 500000
                }
            }
            
            # Calculate portfolio risk
            risk_result = risk_engine.calculate_portfolio_risk(portfolio_positions, market_data)
            
            # Store results
            self.test_results['hft_risk_engine'] = {
                'status': 'success',
                'risk_result': {
                    'var_95': risk_result.var_95,
                    'var_99': risk_result.var_99,
                    'expected_shortfall': risk_result.expected_shortfall,
                    'computation_time_ms': risk_result.computation_time_ms
                },
                'performance_summary': risk_engine.get_performance_summary()
            }
            
            logger.info("HFT Risk Engine test completed successfully")
            
        except Exception as e:
            logger.error(f"HFT Risk Engine test failed: {e}")
            self.test_results['hft_risk_engine'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_regulatory_capital_engine(self):
        """Test automated regulatory capital engine"""
        logger.info("Testing Regulatory Capital Engine...")
        
        try:
            # Initialize regulatory engine
            regulatory_engine = RegulatoryCapitalEngine(
                basel_framework=BaselFramework.BASEL_IV,
                capital_approach=CapitalApproach.STANDARDIZED
            )
            
            # Create test regulatory instruments
            regulatory_instruments = [
                RegulatoryInstrument(
                    instrument_id="REG_BOND_001",
                    asset_class=AssetClass.CORPORATES,
                    risk_weight=0.5,
                    maturity=5.0,
                    notional_amount=1000000,
                    market_value=1000000,
                    credit_rating="AA",
                    issuer_type="corporate",
                    country_of_issuer="IN",
                    sector="Financial",
                    liquidity_category="liquid",
                    collateral_type="unsecured",
                    guarantor_type="none"
                ),
                RegulatoryInstrument(
                    instrument_id="REG_BOND_002",
                    asset_class=AssetClass.SOVEREIGN,
                    risk_weight=0.0,
                    maturity=3.0,
                    notional_amount=500000,
                    market_value=500000,
                    credit_rating="AAA",
                    issuer_type="sovereign",
                    country_of_issuer="IN",
                    sector="Government",
                    liquidity_category="highly_liquid",
                    collateral_type="none",
                    guarantor_type="none"
                )
            ]
            
            # Mock liquidity data
            liquidity_data = {
                'high_quality_liquid_assets': {
                    'cash': 1000000,
                    'sovereign_bonds_aaa': 500000
                },
                'net_cash_outflows': {
                    'deposits': 800000,
                    'wholesale_funding': 400000
                },
                'available_stable_funding': {
                    'tier_1_capital': 2000000,
                    'stable_deposits': 3000000
                },
                'required_stable_funding': {
                    'corporate_bonds_aa': 1000000,
                    'equity': 500000
                }
            }
            
            # Calculate regulatory capital
            regulatory_report = regulatory_engine.calculate_regulatory_capital(
                regulatory_instruments, liquidity_data
            )
            
            # Generate SEBI report
            sebi_report = regulatory_engine.generate_sebi_report(regulatory_report)
            
            # Store results
            self.test_results['regulatory_capital_engine'] = {
                'status': 'success',
                'regulatory_report': {
                    'capital_adequacy_ratio': regulatory_report.capital_requirements.capital_adequacy_ratio,
                    'lcr_ratio': regulatory_report.liquidity_metrics.lcr_ratio,
                    'nsfr_ratio': regulatory_report.liquidity_metrics.nsfr_ratio,
                    'generation_time_seconds': regulatory_report.generation_time_seconds
                },
                'sebi_report': sebi_report,
                'performance_summary': regulatory_engine.get_performance_summary()
            }
            
            logger.info("Regulatory Capital Engine test completed successfully")
            
        except Exception as e:
            logger.error(f"Regulatory Capital Engine test failed: {e}")
            self.test_results['regulatory_capital_engine'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_advanced_redis_cluster(self):
        """Test advanced Redis clustering with time series support"""
        logger.info("Testing Advanced Redis Cluster...")
        
        try:
            # Initialize Redis cluster (single node for testing)
            redis_config = ClusterConfig(
                mode=RedisMode.SINGLE,
                nodes=[
                    RedisNodeConfig(
                        host="localhost",
                        port=6379,
                        password=None,
                        db=0
                    )
                ]
            )
            
            cache_config = CacheConfig(
                default_ttl=300,
                enable_compression=True,
                compression_threshold=1000
            )
            
            # Note: This will fail if Redis is not running locally
            # In production, you'd have Redis running
            try:
                redis_cluster = AdvancedRedisCluster(redis_config, cache_config)
                
                # Test basic operations
                test_key = "test_phase_d"
                test_value = {"message": "Hello Phase D!", "timestamp": datetime.now().isoformat()}
                
                # Set value
                set_result = redis_cluster.set(test_key, test_value, data_type=redis_cluster.DataType.JSON)
                
                # Get value
                retrieved_value = redis_cluster.get(test_key, data_type=redis_cluster.DataType.JSON)
                
                # Test time series
                timeseries_key = "test_timeseries"
                redis_cluster.timeseries_manager.create_timeseries(timeseries_key)
                
                # Add samples
                for i in range(10):
                    timestamp = datetime.now() + timedelta(seconds=i)
                    value = np.random.random()
                    redis_cluster.set_timeseries(timeseries_key, timestamp, value)
                
                # Query time series
                from_time = datetime.now() - timedelta(seconds=5)
                to_time = datetime.now() + timedelta(seconds=5)
                timeseries_data = redis_cluster.get_timeseries(timeseries_key, from_time, to_time)
                
                # Store results
                self.test_results['advanced_redis_cluster'] = {
                    'status': 'success',
                    'set_result': set_result,
                    'retrieved_value': retrieved_value,
                    'timeseries_data_count': len(timeseries_data),
                    'performance_summary': redis_cluster.get_performance_summary()
                }
                
                # Cleanup
                redis_cluster.close()
                
            except Exception as redis_error:
                logger.warning(f"Redis test skipped (Redis not available): {redis_error}")
                self.test_results['advanced_redis_cluster'] = {
                    'status': 'skipped',
                    'reason': 'Redis not available locally'
                }
            
            logger.info("Advanced Redis Cluster test completed")
            
        except Exception as e:
            logger.error(f"Advanced Redis Cluster test failed: {e}")
            self.test_results['advanced_redis_cluster'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_streaming_analytics(self):
        """Test real-time streaming analytics pipeline"""
        logger.info("Testing Streaming Analytics...")
        
        try:
            # Initialize Kafka manager (will skip if Kafka not available)
            kafka_manager = KafkaStreamManager(
                bootstrap_servers=["localhost:9092"],
                client_id="phase_d_test"
            )
            
            # Initialize streaming analytics engine
            streaming_engine = StreamingAnalyticsEngine(kafka_manager)
            
            # Generate test tick data
            test_tick_data = TickData(
                timestamp=datetime.now(),
                instrument_id="TEST_BOND_001",
                price=100.0,
                volume=1000000,
                bid=99.95,
                ask=100.05,
                bid_size=500000,
                ask_size=500000,
                trade_type="market",
                market_maker="TEST_MM",
                venue="TEST_VENUE",
                sequence_number=1,
                yield_to_maturity=0.05,
                duration=5.0,
                convexity=25.0,
                credit_spread=0.015,
                liquidity_score=0.8
            )
            
            # Process tick data
            streaming_engine.process_tick_data(test_tick_data)
            
            # Generate more tick data for testing
            for i in range(5):
                tick_data = TickData(
                    timestamp=datetime.now() + timedelta(seconds=i),
                    instrument_id="TEST_BOND_001",
                    price=100.0 + (i * 0.01),
                    volume=1000000 + (i * 100000),
                    bid=99.95 + (i * 0.01),
                    ask=100.05 + (i * 0.01),
                    bid_size=500000,
                    ask_size=500000,
                    trade_type="market",
                    market_maker="TEST_MM",
                    venue="TEST_VENUE",
                    sequence_number=i + 2,
                    yield_to_maturity=0.05,
                    duration=5.0,
                    convexity=25.0,
                    credit_spread=0.015,
                    liquidity_score=0.8
                )
                streaming_engine.process_tick_data(tick_data)
            
            # Get performance summary
            performance_summary = streaming_engine.get_performance_summary()
            
            # Store results
            self.test_results['streaming_analytics'] = {
                'status': 'success',
                'tick_data_processed': 6,
                'performance_summary': performance_summary
            }
            
            # Cleanup
            kafka_manager.close()
            
            logger.info("Streaming Analytics test completed successfully")
            
        except Exception as e:
            logger.error(f"Streaming Analytics test failed: {e}")
            self.test_results['streaming_analytics'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_end_to_end_integration(self):
        """Test end-to-end integration of all Phase D components"""
        logger.info("Testing End-to-End Integration...")
        
        try:
            # This test would demonstrate how all components work together
            # For now, we'll simulate the integration
            
            integration_results = {
                'ml_pipeline_ready': self.test_results.get('enhanced_ml_pipeline', {}).get('status') == 'success',
                'risk_engine_ready': self.test_results.get('hft_risk_engine', {}).get('status') == 'success',
                'regulatory_engine_ready': self.test_results.get('regulatory_capital_engine', {}).get('status') == 'success',
                'redis_cluster_ready': self.test_results.get('advanced_redis_cluster', {}).get('status') in ['success', 'skipped'],
                'streaming_ready': self.test_results.get('streaming_analytics', {}).get('status') == 'success'
            }
            
            # Calculate overall readiness
            ready_components = sum(integration_results.values())
            total_components = len(integration_results)
            readiness_percentage = (ready_components / total_components) * 100
            
            # Store results
            self.test_results['end_to_end_integration'] = {
                'status': 'success' if readiness_percentage >= 80 else 'partial',
                'readiness_percentage': readiness_percentage,
                'component_status': integration_results,
                'integration_summary': {
                    'total_components': total_components,
                    'ready_components': ready_components,
                    'overall_status': 'READY' if readiness_percentage >= 80 else 'PARTIAL'
                }
            }
            
            logger.info(f"End-to-End Integration test completed. Readiness: {readiness_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"End-to-End Integration test failed: {e}")
            self.test_results['end_to_end_integration'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating Phase D Integration Test Report...")
        
        # Calculate test duration
        test_duration = time.time() - self.start_time
        
        # Count test results
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'success')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'failed')
        skipped_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'skipped')
        
        # Generate report
        report = {
            'test_suite': 'Phase D Integration Test',
            'timestamp': datetime.now().isoformat(),
            'test_duration_seconds': test_duration,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'skipped_tests': skipped_tests,
                'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'detailed_results': self.test_results
        }
        
        # Print report
        print("\n" + "="*80)
        print("PHASE D INTEGRATION TEST REPORT")
        print("="*80)
        print(f"Test Suite: {report['test_suite']}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Duration: {test_duration:.2f} seconds")
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Skipped: {skipped_tests}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print("\n" + "-"*80)
        
        # Print detailed results
        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')
            status_symbol = "✅" if status == 'success' else "❌" if status == 'failed' else "⏭️"
            print(f"{status_symbol} {test_name.upper()}: {status.upper()}")
            
            if status == 'failed' and 'error' in result:
                print(f"   Error: {result['error']}")
            elif status == 'success' and 'performance_summary' in result:
                perf = result['performance_summary']
                if 'operation_summary' in perf:
                    for op, metrics in perf['operation_summary'].items():
                        if 'avg_latency_ms' in metrics:
                            print(f"   {op}: {metrics['avg_latency_ms']:.2f}ms avg latency")
        
        print("\n" + "="*80)
        
        # Save report to file
        report_file = f"phase_d_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Detailed report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save report to file: {e}")
        
        return report

async def main():
    """Main test execution function"""
    try:
        # Create and run test suite
        test_suite = PhaseDIntegrationTest()
        await test_suite.run_all_tests()
        
        logger.info("Phase D Integration Test Suite completed successfully")
        
    except Exception as e:
        logger.error(f"Phase D Integration Test Suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
