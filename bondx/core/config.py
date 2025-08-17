"""
Core configuration module for BondX Backend.

This module handles all environment variables, application settings,
and configuration management for the bond pricing engine.
"""

import os
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(20, description="Database connection pool size")
    max_overflow: int = Field(30, description="Maximum overflow connections")
    pool_timeout: int = Field(30, description="Connection pool timeout in seconds")
    pool_recycle: int = Field(3600, description="Connection pool recycle time in seconds")
    
    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("Database URL must be a valid PostgreSQL connection string")
        return v


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    pool_size: int = Field(20, description="Redis connection pool size")
    max_connections: int = Field(50, description="Maximum Redis connections")
    
    @field_validator("url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with 'redis://'")
        return v


class CelerySettings(BaseSettings):
    """Celery configuration settings."""
    
    broker_url: str = Field("redis://localhost:6379/1", description="Celery broker URL")
    result_backend: str = Field("redis://localhost:6379/2", description="Celery result backend URL")
    task_serializer: str = Field("json", description="Task serialization format")
    result_serializer: str = Field("json", description="Result serialization format")
    accept_content: List[str] = Field(["json"], description="Accepted content types")
    timezone: str = Field("Asia/Kolkata", description="Celery timezone")
    enable_utc: bool = Field(True, description="Enable UTC timezone")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(..., description="JWT secret key")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, description="Access token expiration time")
    refresh_token_expire_days: int = Field(7, description="Refresh token expiration time")
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key length."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class MarketDataSettings(BaseSettings):
    """Market data configuration settings."""
    
    update_interval: int = Field(300, description="Market data update interval in seconds")
    retention_days: int = Field(365, description="Market data retention period in days")
    batch_size: int = Field(1000, description="Batch size for data processing")
    
    # External API keys
    nse_api_key: Optional[str] = Field(None, description="NSE API key")
    bse_api_key: Optional[str] = Field(None, description="BSE API key")
    rbi_api_key: Optional[str] = Field(None, description="RBI API key")
    crisil_api_key: Optional[str] = Field(None, description="CRISIL API key")
    icra_api_key: Optional[str] = Field(None, description="ICRA API key")
    care_api_key: Optional[str] = Field(None, description="CARE API key")
    bloomberg_api_key: Optional[str] = Field(None, description="Bloomberg API key")
    
    # External service URLs
    nse_base_url: str = Field("https://www.nseindia.com", description="NSE base URL")
    bse_base_url: str = Field("https://www.bseindia.com", description="BSE base URL")
    rbi_base_url: str = Field("https://www.rbi.org.in", description="RBI base URL")
    crisil_base_url: str = Field("https://www.crisil.com", description="CRISIL base URL")
    icra_base_url: str = Field("https://www.icra.in", description="ICRA base URL")
    care_base_url: str = Field("https://www.careratings.com", description="CARE base URL")


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    ttl: int = Field(300, description="Cache TTL in seconds")
    max_size: int = Field(10000, description="Maximum cache size")
    stale_while_revalidate: int = Field(60, description="Stale while revalidate time in seconds")


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration settings."""
    
    requests: int = Field(100, description="Maximum requests per window")
    window: int = Field(60, description="Rate limit window in seconds")
    burst: int = Field(200, description="Maximum burst requests")


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field("json", description="Log format")
    file_path: Optional[str] = Field(None, description="Log file path")
    max_size: str = Field("100MB", description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup log files")


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""
    
    prometheus_enabled: bool = Field(True, description="Enable Prometheus metrics")
    sentry_dsn: Optional[str] = Field(None, description="Sentry DSN for error tracking")
    health_check_interval: int = Field(30, description="Health check interval in seconds")


class IndianMarketSettings(BaseSettings):
    """Indian market specific settings."""
    
    timezone: str = Field("Asia/Kolkata", description="Market timezone")
    holidays_file: str = Field("data/market_holidays.json", description="Market holidays file path")
    settlement_cycles_file: str = Field("data/settlement_cycles.json", description="Settlement cycles file path")


class DataQualitySettings(BaseSettings):
    """Data quality configuration settings."""
    
    validation_enabled: bool = Field(True, description="Enable data validation")
    outlier_detection_threshold: float = Field(3.0, description="Outlier detection threshold")
    data_freshness_threshold: int = Field(3600, description="Data freshness threshold in seconds")


class ComplianceSettings(BaseSettings):
    """Regulatory compliance settings."""
    
    audit_log_enabled: bool = Field(True, description="Enable audit logging")
    data_retention_days: int = Field(2555, description="Data retention period (7 years)")
    encryption_enabled: bool = Field(True, description="Enable data encryption")


class PerformanceSettings(BaseSettings):
    """Performance configuration settings."""
    
    query_timeout: int = Field(30, description="Query timeout in seconds")
    max_concurrent_queries: int = Field(100, description="Maximum concurrent queries")
    batch_processing_size: int = Field(5000, description="Batch processing size")


class WebSocketSettings(BaseSettings):
    """WebSocket configuration settings."""
    
    max_connections: int = Field(1000, description="Maximum WebSocket connections")
    ping_interval: int = Field(30, description="Ping interval in seconds")
    ping_timeout: int = Field(10, description="Ping timeout in seconds")


class BackgroundTaskSettings(BaseSettings):
    """Background task configuration settings."""
    
    max_tasks: int = Field(50, description="Maximum background tasks")
    task_timeout: int = Field(300, description="Task timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay: int = Field(60, description="Retry delay in seconds")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field("BondX Backend", description="Application name")
    app_version: str = Field("0.1.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("production", description="Environment")
    
    # Server settings
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, description="Server port")
    workers: int = Field(4, description="Number of workers")
    reload: bool = Field(False, description="Auto-reload on code changes")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    indian_market: IndianMarketSettings = Field(default_factory=IndianMarketSettings)
    data_quality: DataQualitySettings = Field(default_factory=DataQualitySettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)
    background_tasks: BackgroundTaskSettings = Field(default_factory=BackgroundTaskSettings)
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_environments = ["development", "staging", "production", "testing"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @field_validator("debug")
    @classmethod
    def validate_debug(cls, v: bool, info: Any) -> bool:
        """Validate debug mode based on environment."""
        if info.data.get("environment") == "production" and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting."""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL with proper formatting."""
        return self.redis.url
    
    def get_celery_broker_url(self) -> str:
        """Get Celery broker URL with proper formatting."""
        return self.celery.broker_url
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"


# Global settings instance
settings = Settings()

# Export commonly used settings for convenience
__all__ = [
    "Settings",
    "settings",
    "DatabaseSettings",
    "RedisSettings",
    "CelerySettings",
    "SecuritySettings",
    "MarketDataSettings",
    "CacheSettings",
    "RateLimitSettings",
    "LoggingSettings",
    "MonitoringSettings",
    "IndianMarketSettings",
    "DataQualitySettings",
    "ComplianceSettings",
    "PerformanceSettings",
    "WebSocketSettings",
    "BackgroundTaskSettings",
]
