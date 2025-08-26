"""
Main Liquidity Pulse Engine

This module orchestrates all components to generate the complete liquidity pulse output
including real-time indices, forecasts, and driver analysis.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from ...core.logging import get_logger
from ...core.model_contracts import ModelResult, ModelType, ModelStatus
from ...api.v1.schemas_liquidity import (
    LiquidityPulse, ForecastPoint, Driver, DataFreshness, SignalQuality
)
from .signal_adapters import SignalAdapterManager, ProcessedSignal
from .feature_engine import FeatureEngine
from .forecast_engine import ForecastEngine
from .models import PulseModelRegistry

logger = get_logger(__name__)

@dataclass
class PulseCalculationResult:
    """Result of a pulse calculation."""
    isin: str
    timestamp: datetime
    liquidity_index: float
    repayment_support: float
    bondx_score: float
    forecast: List[ForecastPoint]
    drivers: List[Driver]
    missing_signals: List[str]
    freshness: DataFreshness
    uncertainty: float
    inputs_hash: str
    model_versions: Dict[str, str]
    calculation_time_ms: float
    success: bool
    error_message: Optional[str] = None

class LiquidityPulseEngine:
    """Main engine for generating liquidity pulse calculations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.signal_manager = SignalAdapterManager(config.get("signal_adapters", {}))
        self.feature_engine = FeatureEngine(config.get("feature_engine", {}))
        self.forecast_engine = ForecastEngine(config.get("forecast_engine", {}))
        self.model_registry = PulseModelRegistry(config.get("models", {}))
        
        # Performance configuration
        self.max_calculation_time_ms = config.get("max_calculation_time_ms", 50)
        self.enable_forecasting = config.get("enable_forecasting", True)
        self.enable_driver_analysis = config.get("enable_driver_analysis", True)
        
        # Historical data storage for forecasting
        self.historical_liquidity: Dict[str, List[float]] = {}
        self.max_history_length = config.get("max_history_length", 100)
        
        # Model version tracking
        self.model_versions = {
            "liquidity_model": "1.0.0",
            "repayment_model": "1.0.0",
            "forecast_model": "1.0.0",
            "feature_engine": "1.0.0"
        }
        
        self.logger.info("Liquidity Pulse Engine initialized")
    
    async def calculate_pulse(self, isin: str, mode: str = "fast", 
                            include_forecast: bool = True, include_drivers: bool = True,
                            sector: str = "INDUSTRIAL", rating: str = "BBB", tenor: str = "MEDIUM") -> PulseCalculationResult:
        """Calculate complete liquidity pulse for an ISIN."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting pulse calculation for {isin} in {mode} mode")
            
            # Collect signals
            signals = await self._collect_signals([isin])
            if not signals:
                return self._create_error_result(isin, "No signals available")
            
            # Compute features
            feature_set = self.feature_engine.compute_features(signals, isin)
            if not feature_set.features:
                return self._create_error_result(isin, "Failed to compute features")
            
            # Calculate scores
            scores = self.model_registry.calculate_all_scores(
                feature_set, signals, sector, rating, tenor
            )
            
            # Generate forecasts if requested
            forecast = []
            if include_forecast and self.enable_forecasting:
                historical_liquidity = self.historical_liquidity.get(isin, [])
                forecast = self.forecast_engine.generate_forecasts(feature_set, historical_liquidity)
                
                # Update historical data
                self._update_historical_liquidity(isin, scores["liquidity_index"])
            
            # Prepare drivers
            drivers = []
            if include_drivers and self.enable_driver_analysis:
                drivers = scores["top_drivers"]
            
            # Calculate data freshness
            freshness = self._calculate_overall_freshness(signals)
            
            # Identify missing signals
            missing_signals = self._identify_missing_signals(signals)
            
            # Calculate inputs hash
            inputs_hash = self._calculate_inputs_hash(signals, feature_set)
            
            # Calculate uncertainty
            uncertainty = scores["uncertainty"]
            
            # Calculate execution time
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check performance requirements
            if calculation_time > self.max_calculation_time_ms:
                self.logger.warning(f"Pulse calculation for {isin} exceeded performance target: {calculation_time:.2f}ms")
            
            # Create result
            result = PulseCalculationResult(
                isin=isin,
                timestamp=datetime.now(),
                liquidity_index=scores["liquidity_index"],
                repayment_support=scores["repayment_support"],
                bondx_score=scores["bondx_score"],
                forecast=forecast,
                drivers=drivers,
                missing_signals=missing_signals,
                freshness=freshness,
                uncertainty=uncertainty,
                inputs_hash=inputs_hash,
                model_versions=self.model_versions,
                calculation_time_ms=calculation_time,
                success=True
            )
            
            self.logger.info(f"Pulse calculation completed for {isin} in {calculation_time:.2f}ms")
            return result
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Error calculating pulse for {isin}: {e}")
            return self._create_error_result(isin, str(e), calculation_time)
    
    async def calculate_batch_pulse(self, isins: List[str], mode: str = "fast",
                                  include_forecast: bool = True, include_drivers: bool = True) -> List[PulseCalculationResult]:
        """Calculate liquidity pulse for multiple ISINs in batch."""
        try:
            self.logger.info(f"Starting batch pulse calculation for {len(isins)} ISINs")
            
            # Collect signals for all ISINs
            all_signals = await self.signal_manager.collect_all_signals(isins)
            
            # Group signals by ISIN
            signals_by_isin = {}
            for adapter_id, signals in all_signals.items():
                for signal in signals:
                    if signal.isin not in signals_by_isin:
                        signals_by_isin[signal.isin] = []
                    signals_by_isin[signal.isin].append(signal)
            
            # Calculate pulse for each ISIN
            results = []
            for isin in isins:
                signals = signals_by_isin.get(isin, [])
                if signals:
                    result = await self.calculate_pulse(
                        isin, mode, include_forecast, include_drivers
                    )
                    results.append(result)
                else:
                    error_result = self._create_error_result(isin, "No signals available")
                    results.append(error_result)
            
            self.logger.info(f"Batch pulse calculation completed for {len(results)} ISINs")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch pulse calculation: {e}")
            # Return error results for all ISINs
            return [self._create_error_result(isin, str(e)) for isin in isins]
    
    async def _collect_signals(self, isins: List[str]) -> List[ProcessedSignal]:
        """Collect signals for given ISINs."""
        try:
            # Collect from all adapters
            all_signals = await self.signal_manager.collect_all_signals(isins)
            
            # Flatten and combine signals
            combined_signals = []
            for adapter_id, signals in all_signals.items():
                combined_signals.extend(signals)
            
            # Group by ISIN and take the most recent signal of each type per ISIN
            signals_by_isin = {}
            for signal in combined_signals:
                if signal.isin not in signals_by_isin:
                    signals_by_isin[signal.isin] = {}
                
                signal_type = signal.metadata.signal_type.value
                if signal_type not in signals_by_isin[signal.isin] or \
                   signal.timestamp > signals_by_isin[signal.isin][signal_type].timestamp:
                    signals_by_isin[signal.isin][signal_type] = signal
            
            # Flatten back to list
            final_signals = []
            for isin_signals in signals_by_isin.values():
                final_signals.extend(isin_signals.values())
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error collecting signals: {e}")
            return []
    
    def _calculate_overall_freshness(self, signals: List[ProcessedSignal]) -> DataFreshness:
        """Calculate overall data freshness across all signals."""
        try:
            if not signals:
                return DataFreshness.OUTDATED
            
            # Calculate average age of signals
            now = datetime.now()
            ages = []
            
            for signal in signals:
                age_seconds = (now - signal.timestamp).total_seconds()
                ages.append(age_seconds)
            
            avg_age = np.mean(ages)
            
            # Map to freshness level
            if avg_age < 1:
                return DataFreshness.REAL_TIME
            elif avg_age < 60:
                return DataFreshness.FRESH
            elif avg_age < 3600:
                return DataFreshness.RECENT
            elif avg_age < 86400:
                return DataFreshness.STALE
            else:
                return DataFreshness.OUTDATED
                
        except Exception as e:
            self.logger.debug(f"Error calculating overall freshness: {e}")
            return DataFreshness.RECENT
    
    def _identify_missing_signals(self, signals: List[ProcessedSignal]) -> List[str]:
        """Identify missing signal sources."""
        try:
            available_sources = {s.metadata.source_id for s in signals}
            expected_sources = {"alt_data", "microstructure", "auction_mm", "sentiment"}
            
            missing = expected_sources - available_sources
            return list(missing)
            
        except Exception as e:
            self.logger.debug(f"Error identifying missing signals: {e}")
            return []
    
    def _calculate_inputs_hash(self, signals: List[ProcessedSignal], feature_set: FeatureSet) -> str:
        """Calculate hash of input data for reproducibility."""
        try:
            # Create input summary
            input_summary = {
                "signal_count": len(signals),
                "signal_sources": [s.metadata.source_id for s in signals],
                "signal_timestamps": [s.timestamp.isoformat() for s in signals],
                "feature_count": len(feature_set.features),
                "feature_names": list(feature_set.features.keys())
            }
            
            # Convert to JSON and hash
            input_json = json.dumps(input_summary, sort_keys=True)
            return hashlib.md5(input_json.encode()).hexdigest()
            
        except Exception as e:
            self.logger.debug(f"Error calculating inputs hash: {e}")
            return "unknown"
    
    def _update_historical_liquidity(self, isin: str, liquidity_index: float):
        """Update historical liquidity data for forecasting."""
        try:
            if isin not in self.historical_liquidity:
                self.historical_liquidity[isin] = []
            
            # Add new value
            self.historical_liquidity[isin].append(liquidity_index)
            
            # Limit history length
            if len(self.historical_liquidity[isin]) > self.max_history_length:
                self.historical_liquidity[isin] = self.historical_liquidity[isin][-self.max_history_length:]
                
        except Exception as e:
            self.logger.debug(f"Error updating historical liquidity for {isin}: {e}")
    
    def _create_error_result(self, isin: str, error_message: str, calculation_time_ms: float = 0.0) -> PulseCalculationResult:
        """Create error result when calculation fails."""
        return PulseCalculationResult(
            isin=isin,
            timestamp=datetime.now(),
            liquidity_index=50.0,
            repayment_support=60.0,
            bondx_score=55.0,
            forecast=[],
            drivers=[],
            missing_signals=["all"],
            freshness=DataFreshness.OUTDATED,
            uncertainty=0.9,
            inputs_hash="error",
            model_versions=self.model_versions,
            calculation_time_ms=calculation_time_ms,
            success=False,
            error_message=error_message
        )
    
    def convert_to_liquidity_pulse(self, result: PulseCalculationResult) -> LiquidityPulse:
        """Convert calculation result to API schema."""
        try:
            return LiquidityPulse(
                isin=result.isin,
                as_of=result.timestamp,
                liquidity_index=result.liquidity_index,
                repayment_support=result.repayment_support,
                bondx_score=result.bondx_score,
                forecast=result.forecast,
                drivers=result.drivers,
                missing_signals=result.missing_signals,
                freshness=result.freshness,
                uncertainty=result.uncertainty,
                inputs_hash=result.inputs_hash,
                model_versions=result.model_versions
            )
        except Exception as e:
            self.logger.error(f"Error converting to LiquidityPulse: {e}")
            # Return minimal pulse on error
            return LiquidityPulse(
                isin=result.isin,
                as_of=result.timestamp,
                liquidity_index=50.0,
                repayment_support=60.0,
                bondx_score=55.0,
                forecast=[],
                drivers=[],
                missing_signals=["all"],
                freshness=DataFreshness.OUTDATED,
                uncertainty=0.9,
                inputs_hash="error",
                model_versions=result.model_versions
            )
    
    def get_engine_health(self) -> Dict[str, Any]:
        """Get health status of the pulse engine."""
        try:
            # Get component health
            signal_health = self.signal_manager.get_adapter_health()
            forecast_performance = self.forecast_engine.get_model_performance()
            
            # Calculate overall health
            signal_statuses = [h["status"] for h in signal_health.values()]
            active_signals = signal_statuses.count("active")
            total_signals = len(signal_statuses)
            
            overall_health = "healthy" if active_signals / total_signals > 0.7 else "degraded"
            
            return {
                "status": overall_health,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "signal_adapters": signal_health,
                    "forecast_engine": forecast_performance,
                    "feature_engine": "healthy",
                    "model_registry": "healthy"
                },
                "performance": {
                    "max_calculation_time_ms": self.max_calculation_time_ms,
                    "historical_liquidity_count": len(self.historical_liquidity),
                    "model_versions": self.model_versions
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting engine health: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def train_forecast_models(self, training_data: Dict[str, List[Tuple[Any, float]]]):
        """Train forecast models using historical data."""
        try:
            self.forecast_engine.train_models(training_data)
            self.logger.info("Forecast models training completed")
        except Exception as e:
            self.logger.error(f"Error training forecast models: {e}")
    
    def save_forecast_models(self, filepath: str):
        """Save trained forecast models to disk."""
        try:
            self.forecast_engine.save_models(filepath)
        except Exception as e:
            self.logger.error(f"Error saving forecast models: {e}")
    
    def load_forecast_models(self, filepath: str):
        """Load trained forecast models from disk."""
        try:
            self.forecast_engine.load_models(filepath)
        except Exception as e:
            self.logger.error(f"Error loading forecast models: {e}")
    
    def cleanup_old_data(self):
        """Clean up old historical data."""
        try:
            # Clean up feature engine data
            self.feature_engine.cleanup_old_data()
            
            # Clean up forecast engine data
            self.forecast_engine.cleanup_old_data()
            
            # Clean up signal manager cache
            self.signal_manager.cleanup_cache()
            
            self.logger.debug("Old data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the engine."""
        try:
            return {
                "max_calculation_time_ms": self.max_calculation_time_ms,
                "signal_adapters": len(self.signal_manager.adapters),
                "forecast_horizons": len(self.forecast_engine.models),
                "historical_isins": len(self.historical_liquidity),
                "model_versions": self.model_versions,
                "component_status": {
                    "signal_manager": "active",
                    "feature_engine": "active",
                    "forecast_engine": "active",
                    "model_registry": "active"
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
