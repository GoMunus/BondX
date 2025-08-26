"""
Common contracts and storage for BondX Phase B models.

This module provides common data structures, validation, and persistence
for OAS calculations, stress testing, and portfolio analytics.
"""

import hashlib
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, TypeVar, Generic
from dataclasses import dataclass, field, asdict
import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator, root_validator

from .logging import get_logger
from ..database.models import DayCountConvention

logger = get_logger(__name__)

# Generic type for model outputs
T = TypeVar('T')


class ModelType(Enum):
    """Types of financial models."""
    OAS_CALCULATION = "OAS_CALCULATION"
    STRESS_TEST = "STRESS_TEST"
    PORTFOLIO_ANALYTICS = "PORTFOLIO_ANALYTICS"
    YIELD_CURVE_CONSTRUCTION = "YIELD_CURVE_CONSTRUCTION"
    CASH_FLOW_PROJECTION = "CASH_FLOW_PROJECTION"
    RISK_METRICS = "RISK_METRICS"


class ValidationSeverity(Enum):
    """Severity levels for validation warnings."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelStatus(Enum):
    """Status of model execution."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ValidationWarning:
    """Validation warning or error."""
    field: str
    message: str
    severity: ValidationSeverity
    suggested_fix: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelInputs:
    """Base class for model inputs."""
    model_type: ModelType
    input_hash: str
    config_version: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate input hash if not provided."""
        if not self.input_hash:
            self.input_hash = self._generate_input_hash()
    
    def _generate_input_hash(self) -> str:
        """Generate hash of input data."""
        input_data = asdict(self)
        # Remove hash and timestamp for consistent hashing
        input_data.pop('input_hash', None)
        input_data.pop('timestamp', None)
        
        return hashlib.md5(
            json.dumps(input_data, default=str, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class ModelOutputs(Generic[T]):
    """Base class for model outputs."""
    model_type: ModelType
    outputs: T
    diagnostics: Dict[str, Any]
    execution_time_ms: float
    status: ModelStatus
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[ValidationWarning] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ModelResult:
    """Complete model result with inputs, outputs, and metadata."""
    model_type: ModelType
    inputs: ModelInputs
    outputs: ModelOutputs
    model_id: str
    execution_id: str
    created_date: datetime
    updated_date: datetime
    curve_id: Optional[str] = None
    vol_id: Optional[str] = None
    reproducibility_seed: Optional[int] = None
    cache_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set timestamps if not provided."""
        if not self.created_date:
            self.created_date = datetime.now()
        if not self.updated_date:
            self.updated_date = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'model_type': self.model_type.value,
            'model_id': self.model_id,
            'execution_id': self.execution_id,
            'inputs': asdict(self.inputs),
            'outputs': asdict(self.outputs),
            'created_date': self.created_date.isoformat(),
            'updated_date': self.updated_date.isoformat(),
            'curve_id': self.curve_id,
            'vol_id': self.vol_id,
            'reproducibility_seed': self.reproducibility_seed,
            'cache_key': self.cache_key,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelResult':
        """Create from dictionary."""
        # Convert string dates back to datetime
        if 'created_date' in data and isinstance(data['created_date'], str):
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        if 'updated_date' in data and isinstance(data['updated_date'], str):
            data['updated_date'] = datetime.fromisoformat(data['updated_date'])
        
        # Convert string model type back to enum
        if 'model_type' in data and isinstance(data['model_type'], str):
            data['model_type'] = ModelType(data['model_type'])
        
        return cls(**data)


class ModelResultStore:
    """
    Storage for model results with caching and persistence.
    
    Provides in-memory caching and optional persistence to database
    for financial model results.
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        max_cache_size: int = 1000,
        enable_persistence: bool = False,
        persistence_backend: Optional[str] = None
    ):
        """
        Initialize model result store.
        
        Args:
            enable_caching: Enable in-memory caching
            max_cache_size: Maximum number of cached results
            enable_persistence: Enable persistence to database
            persistence_backend: Backend for persistence
        """
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self.enable_persistence = enable_persistence
        self.persistence_backend = persistence_backend
        self.logger = logger
        
        # In-memory cache
        self._cache: Dict[str, ModelResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
    
    def store_result(self, result: ModelResult) -> str:
        """
        Store a model result.
        
        Args:
            result: Model result to store
            
        Returns:
            Cache key for the stored result
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(result)
            result.cache_key = cache_key
            
            # Store in cache
            if self.enable_caching:
                self._store_in_cache(cache_key, result)
            
            # Store in persistence layer if enabled
            if self.enable_persistence:
                self._store_in_persistence(result)
            
            self.logger.info(f"Stored result for model {result.model_id}")
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Error storing model result: {str(e)}")
            raise
    
    def retrieve_result(
        self,
        cache_key: str,
        model_type: Optional[ModelType] = None
    ) -> Optional[ModelResult]:
        """
        Retrieve a model result by cache key.
        
        Args:
            cache_key: Cache key for the result
            model_type: Expected model type for validation
            
        Returns:
            Model result if found, None otherwise
        """
        try:
            # Try cache first
            if self.enable_caching and cache_key in self._cache:
                result = self._cache[cache_key]
                self._cache_hits += 1
                
                # Validate model type if specified
                if model_type and result.model_type != model_type:
                    self.logger.warning(
                        f"Model type mismatch: expected {model_type}, got {result.model_type}"
                    )
                    return None
                
                return result
            
            # Try persistence layer
            if self.enable_persistence:
                result = self._retrieve_from_persistence(cache_key)
                if result:
                    # Store in cache for future access
                    if self.enable_caching:
                        self._store_in_cache(cache_key, result)
                    return result
            
            self._cache_misses += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving model result: {str(e)}")
            return None
    
    def search_results(
        self,
        model_type: Optional[ModelType] = None,
        curve_id: Optional[str] = None,
        vol_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ModelResult]:
        """
        Search for model results by criteria.
        
        Args:
            model_type: Filter by model type
            curve_id: Filter by curve ID
            vol_id: Filter by volatility surface ID
            date_from: Filter by start date
            date_to: Filter by end date
            limit: Maximum number of results
            
        Returns:
            List of matching model results
        """
        try:
            results = []
            
            # Search cache
            if self.enable_caching:
                cache_results = self._search_cache(
                    model_type, curve_id, vol_id, date_from, date_to
                )
                results.extend(cache_results)
            
            # Search persistence layer
            if self.enable_persistence:
                persistence_results = self._search_persistence(
                    model_type, curve_id, vol_id, date_from, date_to, limit
                )
                results.extend(persistence_results)
            
            # Remove duplicates and sort by date
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(
                unique_results,
                key=lambda x: x.created_date,
                reverse=True
            )
            
            return sorted_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching model results: {str(e)}")
            return []
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_evictions': self._cache_evictions,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0
        }
    
    def _generate_cache_key(self, result: ModelResult) -> str:
        """Generate cache key for model result."""
        key_data = {
            'model_type': result.model_type.value,
            'input_hash': result.inputs.input_hash,
            'config_version': result.inputs.config_version,
            'curve_id': result.curve_id,
            'vol_id': result.vol_id
        }
        
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
    
    def _store_in_cache(self, cache_key: str, result: ModelResult) -> None:
        """Store result in cache."""
        # Check cache size limit
        if len(self._cache) >= self.max_cache_size:
            self._evict_oldest_cache_entry()
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _evict_oldest_cache_entry(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache_timestamps:
            return
        
        oldest_key = min(
            self._cache_timestamps.keys(),
            key=lambda k: self._cache_timestamps[k]
        )
        
        del self._cache[oldest_key]
        del self._cache_timestamps[oldest_key]
        self._cache_evictions += 1
    
    def _store_in_persistence(self, result: ModelResult) -> None:
        """Store result in persistence layer."""
        # This would implement actual persistence
        # For now, just log the action
        self.logger.debug(f"Would store result in persistence: {result.model_id}")
    
    def _retrieve_from_persistence(self, cache_key: str) -> Optional[ModelResult]:
        """Retrieve result from persistence layer."""
        # This would implement actual persistence retrieval
        # For now, return None
        return None
    
    def _search_cache(
        self,
        model_type: Optional[ModelType],
        curve_id: Optional[str],
        vol_id: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> List[ModelResult]:
        """Search cache for matching results."""
        results = []
        
        for result in self._cache.values():
            if self._result_matches_criteria(
                result, model_type, curve_id, vol_id, date_from, date_to
            ):
                results.append(result)
        
        return results
    
    def _search_persistence(
        self,
        model_type: Optional[ModelType],
        curve_id: Optional[str],
        vol_id: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        limit: int
    ) -> List[ModelResult]:
        """Search persistence layer for matching results."""
        # This would implement actual persistence search
        # For now, return empty list
        return []
    
    def _result_matches_criteria(
        self,
        result: ModelResult,
        model_type: Optional[ModelType],
        curve_id: Optional[str],
        vol_id: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> bool:
        """Check if result matches search criteria."""
        if model_type and result.model_type != model_type:
            return False
        
        if curve_id and result.curve_id != curve_id:
            return False
        
        if vol_id and result.vol_id != vol_id:
            return False
        
        if date_from and result.created_date < date_from:
            return False
        
        if date_to and result.created_date > date_to:
            return False
        
        return True
    
    def _deduplicate_results(self, results: List[ModelResult]) -> List[ModelResult]:
        """Remove duplicate results based on model_id and execution_id."""
        seen = set()
        unique_results = []
        
        for result in results:
            identifier = (result.model_id, result.execution_id)
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)
        
        return unique_results


class ModelValidator:
    """
    Validator for model inputs and outputs.
    
    Provides validation rules and checks for financial model inputs
    and outputs to ensure data quality and consistency.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize model validator.
        
        Args:
            strict_mode: Enable strict validation (fail on warnings)
        """
        self.strict_mode = strict_mode
        self.logger = logger
    
    def validate_inputs(self, inputs: ModelInputs) -> List[ValidationWarning]:
        """
        Validate model inputs.
        
        Args:
            inputs: Model inputs to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        try:
            # Validate required fields
            if not inputs.model_type:
                warnings.append(ValidationWarning(
                    field="model_type",
                    message="Model type is required",
                    severity=ValidationSeverity.ERROR
                ))
            
            if not inputs.config_version:
                warnings.append(ValidationWarning(
                    field="config_version",
                    message="Config version is required",
                    severity=ValidationSeverity.ERROR
                ))
            
            # Model-specific validation
            if inputs.model_type == ModelType.OAS_CALCULATION:
                warnings.extend(self._validate_oas_inputs(inputs))
            elif inputs.model_type == ModelType.STRESS_TEST:
                warnings.extend(self._validate_stress_test_inputs(inputs))
            elif inputs.model_type == ModelType.PORTFOLIO_ANALYTICS:
                warnings.append(ValidationWarning(
                    field="validation",
                    message="Portfolio analytics validation not yet implemented",
                    severity=ValidationSeverity.INFO
                ))
            
            # Check for critical errors
            critical_errors = [w for w in warnings if w.severity == ValidationSeverity.ERROR]
            if critical_errors and self.strict_mode:
                error_messages = [f"{w.field}: {w.message}" for w in critical_errors]
                raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
            
        except Exception as e:
            self.logger.error(f"Error during input validation: {str(e)}")
            warnings.append(ValidationWarning(
                field="validation",
                message=f"Validation error: {str(e)}",
                severity=ValidationSeverity.CRITICAL
            ))
        
        return warnings
    
    def validate_outputs(self, outputs: ModelOutputs) -> List[ValidationWarning]:
        """
        Validate model outputs.
        
        Args:
            outputs: Model outputs to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        try:
            # Validate required fields
            if not outputs.model_type:
                warnings.append(ValidationWarning(
                    field="model_type",
                    message="Model type is required",
                    severity=ValidationSeverity.ERROR
                ))
            
            if outputs.execution_time_ms < 0:
                warnings.append(ValidationWarning(
                    field="execution_time_ms",
                    message="Execution time cannot be negative",
                    severity=ValidationSeverity.ERROR
                ))
            
            if not outputs.status:
                warnings.append(ValidationWarning(
                    field="status",
                    message="Status is required",
                    severity=ValidationSeverity.ERROR
                ))
            
            # Check for failed status
            if outputs.status == ModelStatus.FAILED and not outputs.errors:
                warnings.append(ValidationWarning(
                    field="status",
                    message="Failed status should have error messages",
                    severity=ValidationSeverity.WARNING
                ))
            
        except Exception as e:
            self.logger.error(f"Error during output validation: {str(e)}")
            warnings.append(ValidationWarning(
                field="validation",
                message=f"Validation error: {str(e)}",
                severity=ValidationSeverity.CRITICAL
            ))
        
        return warnings
    
    def _validate_oas_inputs(self, inputs: ModelInputs) -> List[ValidationWarning]:
        """Validate OAS calculation inputs."""
        warnings = []
        
        # This would implement OAS-specific validation
        # For now, return empty list
        return warnings
    
    def _validate_stress_test_inputs(self, inputs: ModelInputs) -> List[ValidationWarning]:
        """Validate stress test inputs."""
        warnings = []
        
        # This would implement stress test-specific validation
        # For now, return empty list
        return warnings


# Export classes
__all__ = [
    "ModelType",
    "ValidationSeverity",
    "ModelStatus",
    "ValidationWarning",
    "ModelInputs",
    "ModelOutputs",
    "ModelResult",
    "ModelResultStore",
    "ModelValidator"
]
