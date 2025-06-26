"""
Environment configuration management for Financial Simulator.

This module handles environment-specific configurations for development,
testing, staging, and production environments.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str
    port: int
    name: str
    user: str
    password: str = field(repr=False)
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    backend: str
    host: str
    port: int
    ttl: int = 3600
    password: Optional[str] = field(default=None, repr=False)


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    format: str
    output: str
    rotation: str = "daily"
    retention: int = 30
    max_size: str = "100MB"


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str = field(repr=False)
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    cors_origins: list = field(default_factory=list)
    rate_limit: str = "100/hour"
    ssl_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    max_workers: int
    batch_size: int
    cache_enabled: bool
    compression_enabled: bool
    profiling_enabled: bool


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""
    name: Environment
    debug: bool
    database: DatabaseConfig
    cache: CacheConfig
    logging: LoggingConfig
    security: SecurityConfig
    performance: PerformanceConfig
    api_base_url: str
    data_storage_path: str
    model_registry_path: str
    feature_flags: Dict[str, bool] = field(default_factory=dict)


class EnvironmentManager:
    """Manages environment-specific configurations."""
    
    def __init__(self):
        self.current_env = self._detect_environment()
        self.config = self._load_config()
        self._setup_logging()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            logging.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_config(self) -> EnvironmentConfig:
        """Load configuration for current environment."""
        configs = {
            Environment.DEVELOPMENT: self._get_development_config(),
            Environment.TESTING: self._get_testing_config(),
            Environment.STAGING: self._get_staging_config(),
            Environment.PRODUCTION: self._get_production_config()
        }
        return configs[self.current_env]
    
    def _get_development_config(self) -> EnvironmentConfig:
        """Development environment configuration."""
        return EnvironmentConfig(
            name=Environment.DEVELOPMENT,
            debug=True,
            database=DatabaseConfig(
                host="localhost",
                port=5432,
                name="financial_simulator_dev",
                user="dev_user",
                password=os.getenv("DB_PASSWORD_DEV", "dev_password"),
                echo=True
            ),
            cache=CacheConfig(
                backend="redis",
                host="localhost",
                port=6379,
                ttl=300
            ),
            logging=LoggingConfig(
                level="DEBUG",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                output="console",
                rotation="daily",
                retention=7
            ),
            security=SecurityConfig(
                secret_key=os.getenv("SECRET_KEY_DEV", "dev-secret-key-change-in-production"),
                cors_origins=["http://localhost:3000", "http://localhost:8000"],
                ssl_enabled=False,
                rate_limit="1000/hour"
            ),
            performance=PerformanceConfig(
                max_workers=4,
                batch_size=100,
                cache_enabled=True,
                compression_enabled=False,
                profiling_enabled=True
            ),
            api_base_url="http://localhost:8000",
            data_storage_path="./data/dev",
            model_registry_path="./models/dev",
            feature_flags={
                "enable_experimental_features": True,
                "enable_debug_endpoints": True,
                "enable_mock_data": True
            }
        )
    
    def _get_testing_config(self) -> EnvironmentConfig:
        """Testing environment configuration."""
        return EnvironmentConfig(
            name=Environment.TESTING,
            debug=True,
            database=DatabaseConfig(
                host="localhost",
                port=5433,
                name="financial_simulator_test",
                user="test_user",
                password=os.getenv("DB_PASSWORD_TEST", "test_password"),
                echo=False
            ),
            cache=CacheConfig(
                backend="memory",
                host="localhost",
                port=6380,
                ttl=60
            ),
            logging=LoggingConfig(
                level="INFO",
                format="%(asctime)s - %(levelname)s - %(message)s",
                output="file",
                rotation="test",
                retention=1
            ),
            security=SecurityConfig(
                secret_key=os.getenv("SECRET_KEY_TEST", "test-secret-key"),
                cors_origins=["http://localhost:3001"],
                ssl_enabled=False,
                rate_limit="10000/hour"
            ),
            performance=PerformanceConfig(
                max_workers=2,
                batch_size=50,
                cache_enabled=False,
                compression_enabled=False,
                profiling_enabled=False
            ),
            api_base_url="http://localhost:8001",
            data_storage_path="./data/test",
            model_registry_path="./models/test",
            feature_flags={
                "enable_experimental_features": False,
                "enable_debug_endpoints": True,
                "enable_mock_data": True
            }
        )
    
    def _get_staging_config(self) -> EnvironmentConfig:
        """Staging environment configuration."""
        return EnvironmentConfig(
            name=Environment.STAGING,
            debug=False,
            database=DatabaseConfig(
                host=os.getenv("DB_HOST_STAGING", "staging-db.example.com"),
                port=5432,
                name="financial_simulator_staging",
                user=os.getenv("DB_USER_STAGING", "staging_user"),
                password=os.getenv("DB_PASSWORD_STAGING", ""),
                pool_size=10,
                max_overflow=20
            ),
            cache=CacheConfig(
                backend="redis",
                host=os.getenv("REDIS_HOST_STAGING", "staging-redis.example.com"),
                port=6379,
                ttl=1800,
                password=os.getenv("REDIS_PASSWORD_STAGING", "")
            ),
            logging=LoggingConfig(
                level="INFO",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                output="file",
                rotation="daily",
                retention=14
            ),
            security=SecurityConfig(
                secret_key=os.getenv("SECRET_KEY_STAGING", ""),
                cors_origins=["https://staging.example.com"],
                ssl_enabled=True,
                rate_limit="500/hour"
            ),
            performance=PerformanceConfig(
                max_workers=8,
                batch_size=500,
                cache_enabled=True,
                compression_enabled=True,
                profiling_enabled=False
            ),
            api_base_url="https://api-staging.example.com",
            data_storage_path="/data/staging",
            model_registry_path="/models/staging",
            feature_flags={
                "enable_experimental_features": True,
                "enable_debug_endpoints": False,
                "enable_mock_data": False
            }
        )
    
    def _get_production_config(self) -> EnvironmentConfig:
        """Production environment configuration."""
        return EnvironmentConfig(
            name=Environment.PRODUCTION,
            debug=False,
            database=DatabaseConfig(
                host=os.getenv("DB_HOST_PROD", "prod-db.example.com"),
                port=5432,
                name="financial_simulator_prod",
                user=os.getenv("DB_USER_PROD", "prod_user"),
                password=os.getenv("DB_PASSWORD_PROD", ""),
                pool_size=20,
                max_overflow=40,
                echo=False
            ),
            cache=CacheConfig(
                backend="redis",
                host=os.getenv("REDIS_HOST_PROD", "prod-redis.example.com"),
                port=6379,
                ttl=3600,
                password=os.getenv("REDIS_PASSWORD_PROD", "")
            ),
            logging=LoggingConfig(
                level="WARNING",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                output="file",
                rotation="daily",
                retention=90,
                max_size="500MB"
            ),
            security=SecurityConfig(
                secret_key=os.getenv("SECRET_KEY_PROD", ""),
                jwt_expiration=7200,
                cors_origins=["https://app.example.com", "https://www.example.com"],
                ssl_enabled=True,
                rate_limit="100/hour"
            ),
            performance=PerformanceConfig(
                max_workers=16,
                batch_size=1000,
                cache_enabled=True,
                compression_enabled=True,
                profiling_enabled=False
            ),
            api_base_url="https://api.example.com",
            data_storage_path="/data/prod",
            model_registry_path="/models/prod",
            feature_flags={
                "enable_experimental_features": False,
                "enable_debug_endpoints": False,
                "enable_mock_data": False
            }
        )
    
    def _setup_logging(self):
        """Configure logging based on environment settings."""
        config = self.config.logging
        
        if config.output == "console":
            handler = logging.StreamHandler()
        else:
            from logging.handlers import RotatingFileHandler
            log_dir = Path("logs") / self.current_env.value
            log_dir.mkdir(parents=True, exist_ok=True)
            
            handler = RotatingFileHandler(
                log_dir / "app.log",
                maxBytes=self._parse_size(config.max_size),
                backupCount=config.retention
            )
        
        handler.setFormatter(logging.Formatter(config.format))
        
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, config.level))
        logger.addHandler(handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(size_str[:-2]) * multiplier
        return int(size_str)
    
    def get_config(self) -> EnvironmentConfig:
        """Get current environment configuration."""
        return self.config
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        db = self.config.database
        return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        cache = self.config.cache
        if cache.password:
            return f"redis://:{cache.password}@{cache.host}:{cache.port}/0"
        return f"redis://{cache.host}:{cache.port}/0"
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.config.feature_flags.get(feature, False)
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        required_env_vars = {
            Environment.PRODUCTION: [
                "DB_HOST_PROD", "DB_USER_PROD", "DB_PASSWORD_PROD",
                "REDIS_HOST_PROD", "REDIS_PASSWORD_PROD", "SECRET_KEY_PROD"
            ],
            Environment.STAGING: [
                "DB_HOST_STAGING", "DB_USER_STAGING", "DB_PASSWORD_STAGING",
                "REDIS_HOST_STAGING", "SECRET_KEY_STAGING"
            ]
        }
        
        if self.current_env in required_env_vars:
            missing = [var for var in required_env_vars[self.current_env] 
                      if not os.getenv(var)]
            if missing:
                logging.error(f"Missing required environment variables: {missing}")
                return False
        
        return True


# Singleton instance
env_manager = EnvironmentManager()
config = env_manager.get_config()


# Convenience functions
def get_config() -> EnvironmentConfig:
    """Get current environment configuration."""
    return config


def get_db_url() -> str:
    """Get database connection URL."""
    return env_manager.get_database_url()


def get_redis_url() -> str:
    """Get Redis connection URL."""
    return env_manager.get_redis_url()


def is_production() -> bool:
    """Check if running in production environment."""
    return config.name == Environment.PRODUCTION


def is_development() -> bool:
    """Check if running in development environment."""
    return config.name == Environment.DEVELOPMENT