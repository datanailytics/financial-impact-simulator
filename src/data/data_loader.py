"""
Data Loader Module - Multi-Source Data Integration
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides comprehensive data loading capabilities from multiple
sources including APIs, databases, files, and web scraping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import json
import sqlite3
import psycopg2
import pymongo
from sqlalchemy import create_engine
import yfinance as yf
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum
import os
from pathlib import Path
import pickle
import redis
from functools import lru_cache
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"
    MONGODB = "mongodb"
    API = "api"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    BLOOMBERG = "bloomberg"
    REDIS = "redis"
    WEB_SCRAPING = "web_scraping"


class DataFrequency(Enum):
    """Data frequency types."""
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"
    YEARLY = "1y"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    source: DataSource
    connection_params: Dict[str, Any]
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    retry_attempts: int = 3
    timeout: int = 30
    batch_size: int = 1000
    async_enabled: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts must be non-negative")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class DataRequest:
    """Represents a data request."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    frequency: DataFrequency
    fields: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")


class DataCache:
    """Caching layer for data operations."""
    
    def __init__(self, backend: str = "memory", **kwargs):
        """
        Initialize cache.
        
        Args:
            backend: Cache backend ('memory', 'redis', 'disk')
            **kwargs: Backend-specific parameters
        """
        self.backend = backend
        self._cache = {}
        
        if backend == "redis":
            self.redis_client = redis.Redis(
                host=kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 6379),
                db=kwargs.get('db', 0)
            )
        elif backend == "disk":
            self.cache_dir = Path(kwargs.get('cache_dir', './cache'))
            self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        if self.backend == "memory":
            return self._cache.get(key)
        elif self.backend == "redis":
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        elif self.backend == "disk":
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def set(self, key: str, data: pd.DataFrame, ttl: int = 3600):
        """Set data in cache."""
        if self.backend == "memory":
            self._cache[key] = data
        elif self.backend == "redis":
            self.redis_client.setex(key, ttl, pickle.dumps(data))
        elif self.backend == "disk":
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
    
    def clear(self):
        """Clear cache."""
        if self.backend == "memory":
            self._cache.clear()
        elif self.backend == "redis":
            self.redis_client.flushdb()
        elif self.backend == "disk":
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


class BaseDataLoader:
    """Base class for all data loaders."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data loader.
        
        Args:
            config: Data loader configuration
        """
        self.config = config
        self.cache = DataCache() if config.cache_enabled else None
        
    def load(self, request: DataRequest) -> pd.DataFrame:
        """
        Load data based on request.
        
        Args:
            request: Data request parameters
            
        Returns:
            DataFrame with requested data
        """
        raise NotImplementedError("Subclasses must implement load method")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate loaded data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        if data.empty:
            logger.warning("Empty DataFrame loaded")
            return False
        
        if data.isnull().any().any():
            logger.warning("Data contains null values")
        
        return True
    
    def _get_cache_key(self, request: DataRequest) -> str:
        """Generate cache key from request."""
        key_parts = [
            self.config.source.value,
            '_'.join(request.symbols),
            request.start_date.strftime('%Y%m%d'),
            request.end_date.strftime('%Y%m%d'),
            request.frequency.value
        ]
        return '_'.join(key_parts)


class FileDataLoader(BaseDataLoader):
    """Loader for file-based data sources."""
    
    def load(self, request: DataRequest) -> pd.DataFrame:
        """Load data from files."""
        file_path = self.config.connection_params.get('file_path')
        if not file_path:
            raise ValueError("File path not specified in config")
        
        # Check cache first
        if self.cache:
            cache_key = self._get_cache_key(request)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Data loaded from cache: {cache_key}")
                return cached_data
        
        # Load based on file type
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            data = self._load_csv(file_path, request)
        elif file_extension in ['.xlsx', '.xls']:
            data = self._load_excel(file_path, request)
        elif file_extension == '.json':
            data = self._load_json(file_path, request)
        elif file_extension == '.parquet':
            data = self._load_parquet(file_path, request)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Filter data based on request
        data = self._filter_data(data, request)
        
        # Cache if enabled
        if self.cache and not data.empty:
            self.cache.set(cache_key, data, self.config.cache_ttl)
        
        return data
    
    def _load_csv(self, file_path: str, request: DataRequest) -> pd.DataFrame:
        """Load CSV file."""
        try:
            data = pd.read_csv(
                file_path,
                index_col=0,
                parse_dates=True,
                infer_datetime_format=True
            )
            logger.info(f"Loaded {len(data)} rows from CSV: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def _load_excel(self, file_path: str, request: DataRequest) -> pd.DataFrame:
        """Load Excel file."""
        try:
            sheet_name = self.config.connection_params.get('sheet_name', 0)
            data = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                index_col=0,
                parse_dates=True
            )
            logger.info(f"Loaded {len(data)} rows from Excel: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    def _load_json(self, file_path: str, request: DataRequest) -> pd.DataFrame:
        """Load JSON file."""
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            data = pd.DataFrame(json_data)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            logger.info(f"Loaded {len(data)} rows from JSON: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def _load_parquet(self, file_path: str, request: DataRequest) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            data = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(data)} rows from Parquet: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise
    
    def _filter_data(self, data: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        """Filter data based on request parameters."""
        # Filter by date range
        if isinstance(data.index, pd.DatetimeIndex):
            mask = (data.index >= request.start_date) & (data.index <= request.end_date)
            data = data.loc[mask]
        
        # Filter by symbols
        if request.symbols and all(symbol in data.columns for symbol in request.symbols):
            data = data[request.symbols]
        
        # Apply custom filters
        if request.filters:
            for column, filter_value in request.filters.items():
                if column in data.columns:
                    data = data[data[column] == filter_value]
        
        return data


class DatabaseDataLoader(BaseDataLoader):
    """Loader for database sources."""
    
    def __init__(self, config: DataConfig):
        """Initialize database loader."""
        super().__init__(config)
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        if self.config.source == DataSource.SQL:
            db_type = self.config.connection_params.get('db_type', 'sqlite')
            
            if db_type == 'sqlite':
                self.connection = sqlite3.connect(
                    self.config.connection_params.get('database', ':memory:')
                )
            elif db_type == 'postgresql':
                self.connection = psycopg2.connect(
                    host=self.config.connection_params.get('host'),
                    port=self.config.connection_params.get('port', 5432),
                    database=self.config.connection_params.get('database'),
                    user=self.config.connection_params.get('user'),
                    password=self.config.connection_params.get('password')
                )
            else:
                # Use SQLAlchemy for other databases
                connection_string = self.config.connection_params.get('connection_string')
                self.engine = create_engine(connection_string)
        
        elif self.config.source == DataSource.MONGODB:
            client = pymongo.MongoClient(
                self.config.connection_params.get('connection_string', 'mongodb://localhost:27017/')
            )
            self.db = client[self.config.connection_params.get('database', 'financial_data')]
    
    def load(self, request: DataRequest) -> pd.DataFrame:
        """Load data from database."""
        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(request)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        if self.config.source == DataSource.SQL:
            data = self._load_sql(request)
        elif self.config.source == DataSource.MONGODB:
            data = self._load_mongodb(request)
        else:
            raise ValueError(f"Unsupported database source: {self.config.source}")
        
        # Cache if enabled
        if self.cache and not data.empty:
            self.cache.set(cache_key, data, self.config.cache_ttl)
        
        return data
    
    def _load_sql(self, request: DataRequest) -> pd.DataFrame:
        """Load data from SQL database."""
        table_name = self.config.connection_params.get('table_name', 'market_data')
        
        # Build query
        query = f"""
        SELECT * FROM {table_name}
        WHERE symbol IN ({','.join(['?' for _ in request.symbols])})
        AND date >= ? AND date <= ?
        """
        
        params = request.symbols + [request.start_date, request.end_date]
        
        try:
            if hasattr(self, 'engine'):
                data = pd.read_sql_query(query, self.engine, params=params)
            else:
                data = pd.read_sql_query(query, self.connection, params=params)
            
            # Set date as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            return data
        except Exception as e:
            logger.error(f"Error loading from SQL: {e}")
            raise
    
    def _load_mongodb(self, request: DataRequest) -> pd.DataFrame:
        """Load data from MongoDB."""
        collection_name = self.config.connection_params.get('collection', 'market_data')
        collection = self.db[collection_name]
        
        # Build query
        query = {
            'symbol': {'$in': request.symbols},
            'date': {
                '$gte': request.start_date,
                '$lte': request.end_date
            }
        }
        
        # Execute query
        cursor = collection.find(query)
        data = pd.DataFrame(list(cursor))
        
        if not data.empty:
            # Remove MongoDB _id field
            if '_id' in data.columns:
                data.drop('_id', axis=1, inplace=True)
            
            # Set date as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
        
        return data
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class APIDataLoader(BaseDataLoader):
    """Loader for API-based data sources."""
    
    def __init__(self, config: DataConfig):
        """Initialize API loader."""
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DataNailytics/1.0'
        })
    
    def load(self, request: DataRequest) -> pd.DataFrame:
        """Load data from API."""
        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(request)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Route to appropriate API loader
        if self.config.source == DataSource.YAHOO_FINANCE:
            data = self._load_yahoo_finance(request)
        elif self.config.source == DataSource.ALPHA_VANTAGE:
            data = self._load_alpha_vantage(request)
        elif self.config.source == DataSource.QUANDL:
            data = self._load_quandl(request)
        elif self.config.source == DataSource.API:
            data = self._load_generic_api(request)
        else:
            raise ValueError(f"Unsupported API source: {self.config.source}")
        
        # Cache if enabled
        if self.cache and not data.empty:
            self.cache.set(cache_key, data, self.config.cache_ttl)
        
        return data
    
    def _load_yahoo_finance(self, request: DataRequest) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            # Map frequency to yfinance interval
            interval_map = {
                DataFrequency.MINUTE_1: "1m",
                DataFrequency.MINUTE_5: "5m",
                DataFrequency.MINUTE_15: "15m",
                DataFrequency.MINUTE_30: "30m",
                DataFrequency.HOURLY: "1h",
                DataFrequency.DAILY: "1d",
                DataFrequency.WEEKLY: "1wk",
                DataFrequency.MONTHLY: "1mo"
            }
            
            interval = interval_map.get(request.frequency, "1d")
            
            # Download data for all symbols
            data_frames = []
            
            for symbol in request.symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=request.start_date,
                    end=request.end_date,
                    interval=interval
                )
                
                if not hist.empty:
                    # Add symbol column
                    hist['symbol'] = symbol
                    data_frames.append(hist)
            
            if data_frames:
                # Combine all data
                data = pd.concat(data_frames)
                return data
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading from Yahoo Finance: {e}")
            if self.config.retry_attempts > 0:
                logger.info("Retrying...")
                return self._retry_load(request, self._load_yahoo_finance)
            raise
    
    def _load_alpha_vantage(self, request: DataRequest) -> pd.DataFrame:
        """Load data from Alpha Vantage."""
        api_key = self.config.connection_params.get('api_key')
        if not api_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        base_url = "https://www.alphavantage.co/query"
        
        # Map frequency to Alpha Vantage function
        function_map = {
            DataFrequency.MINUTE_1: "TIME_SERIES_INTRADAY",
            DataFrequency.MINUTE_5: "TIME_SERIES_INTRADAY",
            DataFrequency.MINUTE_15: "TIME_SERIES_INTRADAY",
            DataFrequency.MINUTE_30: "TIME_SERIES_INTRADAY",
            DataFrequency.MINUTE_60: "TIME_SERIES_INTRADAY",
            DataFrequency.DAILY: "TIME_SERIES_DAILY",
            DataFrequency.WEEKLY: "TIME_SERIES_WEEKLY",
            DataFrequency.MONTHLY: "TIME_SERIES_MONTHLY"
        }
        
        function = function_map.get(request.frequency, "TIME_SERIES_DAILY")
        
        data_frames = []
        
        for symbol in request.symbols:
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            if function == "TIME_SERIES_INTRADAY":
                params['interval'] = request.frequency.value
            
            try:
                response = self.session.get(base_url, params=params, timeout=self.config.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract time series data
                time_series_key = [k for k in data.keys() if 'Time Series' in k][0]
                time_series = data[time_series_key]
                
                # Convert to DataFrame
                df = pd.DataFrame(time_series).T
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df['symbol'] = symbol
                
                # Rename columns
                df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
                
                data_frames.append(df)
                
            except Exception as e:
                logger.error(f"Error loading {symbol} from Alpha Vantage: {e}")
                continue
        
        if data_frames:
            return pd.concat(data_frames)
        return pd.DataFrame()
    
    def _load_quandl(self, request: DataRequest) -> pd.DataFrame:
        """Load data from Quandl."""
        api_key = self.config.connection_params.get('api_key')
        
        try:
            import quandl
            quandl.ApiConfig.api_key = api_key
            
            data_frames = []
            
            for symbol in request.symbols:
                # Quandl uses database/dataset format
                quandl_code = self.config.connection_params.get('dataset_prefix', 'WIKI') + '/' + symbol
                
                data = quandl.get(
                    quandl_code,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                data['symbol'] = symbol
                data_frames.append(data)
            
            if data_frames:
                return pd.concat(data_frames)
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading from Quandl: {e}")
            raise
    
    def _load_generic_api(self, request: DataRequest) -> pd.DataFrame:
        """Load data from generic API."""
        base_url = self.config.connection_params.get('base_url')
        endpoint = self.config.connection_params.get('endpoint', '/data')
        
        headers = self.config.connection_params.get('headers', {})
        auth = self.config.connection_params.get('auth')
        
        # Build request parameters
        params = {
            'symbols': ','.join(request.symbols),
            'start_date': request.start_date.isoformat(),
            'end_date': request.end_date.isoformat(),
            'frequency': request.frequency.value
        }
        
        # Add custom parameters
        custom_params = self.config.connection_params.get('params', {})
        params.update(custom_params)
        
        try:
            response = self.session.get(
                f"{base_url}{endpoint}",
                params=params,
                headers=headers,
                auth=auth,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame (assumes specific structure)
            df = pd.DataFrame(data['data'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading from API: {e}")
            raise
    
    def _retry_load(self, request: DataRequest, load_func: Callable) -> pd.DataFrame:
        """Retry failed load operation."""
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Retry attempt {attempt + 1}/{self.config.retry_attempts}")
                return load_func(request)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff


class AsyncAPIDataLoader(APIDataLoader):
    """Asynchronous API data loader for improved performance."""
    
    def __init__(self, config: DataConfig):
        """Initialize async loader."""
        super().__init__(config)
        self.config.async_enabled = True
    
    async def load_async(self, request: DataRequest) -> pd.DataFrame:
        """Load data asynchronously."""
        # Check cache
        if self.cache:
            cache_key = self._get_cache_key(request)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Create async session
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for symbol in request.symbols:
                task = self._fetch_symbol_async(session, symbol, request)
                tasks.append(task)
            
            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            data_frames = []
            for result in results:
                if isinstance(result, pd.DataFrame):
                    data_frames.append(result)
                else:
                    logger.error(f"Error in async fetch: {result}")
            
            if data_frames:
                data = pd.concat(data_frames)
                
                # Cache if enabled
                if self.cache:
                    self.cache.set(cache_key, data, self.config.cache_ttl)
                
                return data
            
            return pd.DataFrame()
    
    async def _fetch_symbol_async(self, session: aiohttp.ClientSession,
                                 symbol: str, request: DataRequest) -> pd.DataFrame:
        """Fetch data for a single symbol asynchronously."""
        # Implementation depends on specific API
        # This is a placeholder for the pattern
        url = f"{self.config.connection_params.get('base_url')}/quote/{symbol}"
        
        params = {
            'start': request.start_date.isoformat(),
            'end': request.end_date.isoformat(),
            'interval': request.frequency.value
        }
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise


class DataLoaderFactory:
    """Factory for creating appropriate data loaders."""
    
    @staticmethod
    def create_loader(config: DataConfig) -> BaseDataLoader:
        """
        Create data loader based on configuration.
        
        Args:
            config: Data loader configuration
            
        Returns:
            Appropriate data loader instance
        """
        if config.source in [DataSource.CSV, DataSource.EXCEL, DataSource.JSON, DataSource.PARQUET]:
            return FileDataLoader(config)
        
        elif config.source in [DataSource.SQL, DataSource.MONGODB]:
            return DatabaseDataLoader(config)
        
        elif config.source in [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE, 
                              DataSource.QUANDL, DataSource.API]:
            if config.async_enabled:
                return AsyncAPIDataLoader(config)
            return APIDataLoader(config)
        
        else:
            raise ValueError(f"Unsupported data source: {config.source}")


class DataManager:
    """High-level data management interface."""
    
    def __init__(self):
        """Initialize data manager."""
        self.loaders = {}
        self.configs = {}
    
    def register_source(self, name: str, config: DataConfig):
        """
        Register a data source.
        
        Args:
            name: Unique name for the source
            config: Source configuration
        """
        loader = DataLoaderFactory.create_loader(config)
        self.loaders[name] = loader
        self.configs[name] = config
    
    def load_data(self, source_name: str, request: DataRequest) -> pd.DataFrame:
        """
        Load data from a registered source.
        
        Args:
            source_name: Name of the registered source
            request: Data request
            
        Returns:
            Loaded data
        """
        if source_name not in self.loaders:
            raise ValueError(f"Unknown data source: {source_name}")
        
        loader = self.loaders[source_name]
        
        # Use async loading if available
        if hasattr(loader, 'load_async') and loader.config.async_enabled:
            return asyncio.run(loader.load_async(request))
        
        return loader.load(request)
    
    def load_multi_source(self, requests: Dict[str, DataRequest]) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple sources.
        
        Args:
            requests: Dictionary mapping source names to requests
            
        Returns:
            Dictionary mapping source names to data
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.load_data, source, request): source
                for source, request in requests.items()
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    data = future.result()
                    results[source] = data
                except Exception as e:
                    logger.error(f"Error loading from {source}: {e}")
                    results[source] = pd.DataFrame()
        
        return results
    
    def combine_data(self, data_dict: Dict[str, pd.DataFrame],
                    join_method: str = 'outer') -> pd.DataFrame:
        """
        Combine data from multiple sources.
        
        Args:
            data_dict: Dictionary of DataFrames
            join_method: How to join data ('outer', 'inner', 'left', 'right')
            
        Returns:
            Combined DataFrame
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Start with first non-empty DataFrame
        combined = None
        for name, data in data_dict.items():
            if not data.empty:
                if combined is None:
                    combined = data.copy()
                else:
                    # Join on index (assumed to be date)
                    combined = combined.join(data, how=join_method, rsuffix=f'_{name}')
        
        return combined if combined is not None else pd.DataFrame()


def main():
    """Example usage of the data loader module."""
    # Create data manager
    manager = DataManager()
    
    # Register Yahoo Finance source
    yf_config = DataConfig(
        source=DataSource.YAHOO_FINANCE,
        connection_params={},
        cache_enabled=True
    )
    manager.register_source('yahoo', yf_config)
    
    # Register CSV source
    csv_config = DataConfig(
        source=DataSource.CSV,
        connection_params={
            'file_path': 'data/market_data.csv'
        }
    )
    manager.register_source('csv', csv_config)
    
    # Create data request
    request = DataRequest(
        symbols=['AAPL', 'GOOGL', 'MSFT'],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        frequency=DataFrequency.DAILY
    )
    
    # Load data
    print("Loading data from Yahoo Finance...")
    yf_data = manager.load_data('yahoo', request)
    print(f"Loaded {len(yf_data)} rows")
    
    # Load from multiple sources
    multi_requests = {
        'yahoo': request,
        'csv': request
    }
    
    print("\nLoading from multiple sources...")
    multi_data = manager.load_multi_source(multi_requests)
    
    # Combine data
    combined = manager.combine_data(multi_data)
    print(f"Combined data shape: {combined.shape}")


if __name__ == "__main__":
    main()
