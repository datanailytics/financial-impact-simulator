"""
Helpers Module - Utility Functions for Financial Analytics
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides utility functions and helpers for the financial
analytics platform, including data manipulation, calculations, and formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta, date
import json
import yaml
import pickle
import logging
from functools import wraps, lru_cache
import time
import hashlib
import re
from pathlib import Path
import warnings
from decimal import Decimal, ROUND_HALF_UP
import pytz
from dateutil import parser
import holidays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Decorators
def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
        return wrapper
    return decorator


def validate_input(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Validation functions for each parameter
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Validation failed for parameter '{param_name}' with value: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(ttl: int = 3600):
    """
    Decorator to cache function results with TTL.
    
    Args:
        ttl: Time to live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_time = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                f"{func.__name__}_{args}_{kwargs}".encode()
            ).hexdigest()
            
            # Check cache
            if key in cache and (time.time() - cache_time[key]) < ttl:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Update cache
            cache[key] = result
            cache_time[key] = time.time()
            
            return result
        
        return wrapper
    return decorator


# Date and Time Utilities
def get_business_days(start_date: Union[str, datetime, date], 
                     end_date: Union[str, datetime, date],
                     country: str = 'US') -> List[datetime]:
    """
    Get list of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        country: Country code for holidays
        
    Returns:
        List of business days
    """
    # Convert to datetime
    if isinstance(start_date, str):
        start_date = parser.parse(start_date)
    if isinstance(end_date, str):
        end_date = parser.parse(end_date)
    
    # Get country holidays
    country_holidays = holidays.get_country(country)
    
    # Generate business days
    business_days = []
    current_date = start_date
    
    while current_date <= end_date:
        # Check if it's a business day
        if current_date.weekday() < 5 and current_date not in country_holidays:
            business_days.append(current_date)
        current_date += timedelta(days=1)
    
    return business_days


def get_trading_days(start_date: Union[str, datetime], 
                    end_date: Union[str, datetime],
                    exchange: str = 'NYSE') -> pd.DatetimeIndex:
    """
    Get trading days for a specific exchange.
    
    Args:
        start_date: Start date
        end_date: End date
        exchange: Exchange name
        
    Returns:
        DatetimeIndex of trading days
    """
    # Get business days
    business_days = get_business_days(start_date, end_date, 'US')
    
    # Additional market holidays (simplified)
    market_holidays = {
        'NYSE': [
            'Good Friday',
            'Day after Thanksgiving'
        ]
    }
    
    # Filter out market-specific holidays
    trading_days = [d for d in business_days]  # Simplified implementation
    
    return pd.DatetimeIndex(trading_days)


def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """
    Convert datetime between timezones.
    
    Args:
        dt: Datetime to convert
        from_tz: Source timezone
        to_tz: Target timezone
        
    Returns:
        Converted datetime
    """
    from_timezone = pytz.timezone(from_tz)
    to_timezone = pytz.timezone(to_tz)
    
    # Localize if naive
    if dt.tzinfo is None:
        dt = from_timezone.localize(dt)
    else:
        dt = dt.astimezone(from_timezone)
    
    # Convert to target timezone
    return dt.astimezone(to_timezone)


def get_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
    """
    Get start and end dates for a quarter.
    
    Args:
        year: Year
        quarter: Quarter (1-4)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if quarter not in range(1, 5):
        raise ValueError("Quarter must be between 1 and 4")
    
    quarter_months = {
        1: (1, 3),
        2: (4, 6),
        3: (7, 9),
        4: (10, 12)
    }
    
    start_month, end_month = quarter_months[quarter]
    start_date = datetime(year, start_month, 1)
    
    # Get last day of end month
    if end_month == 12:
        end_date = datetime(year, 12, 31)
    else:
        end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
    
    return start_date, end_date


# Financial Calculations
def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
        
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown returns method: {method}")


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        returns: Returns series
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized return
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    if years > 0:
        annualized = (1 + total_return) ** (1 / years) - 1
    else:
        annualized = 0
    
    return annualized


def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Returns series
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                         periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if returns.std() > 0:
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    else:
        return 0.0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                          target_return: float = 0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        target_return: Target return for downside deviation
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) > 0:
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        if downside_deviation > 0:
            return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    
    return np.inf


def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    Calculate maximum drawdown and dates.
    
    Args:
        prices: Price series
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_drawdown = drawdown.min()
    trough_date = drawdown.idxmin()
    
    # Find peak date before trough
    peak_date = cumulative[:trough_date].idxmax()
    
    return max_drawdown, peak_date, trough_date


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        returns: Returns series
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    annual_return = calculate_annualized_return(returns, periods_per_year)
    cumulative = (1 + returns).cumprod()
    max_dd, _, _ = calculate_max_drawdown(cumulative)
    
    if max_dd != 0:
        return annual_return / abs(max_dd)
    else:
        return np.inf


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                              periods_per_year: int = 252) -> float:
    """
    Calculate information ratio.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error > 0:
        return active_returns.mean() * periods_per_year / tracking_error
    else:
        return 0.0


# Data Manipulation
def resample_data(data: pd.DataFrame, source_freq: str, target_freq: str,
                 method: str = 'last') -> pd.DataFrame:
    """
    Resample time series data to different frequency.
    
    Args:
        data: Input data
        source_freq: Source frequency ('D', 'W', 'M', etc.)
        target_freq: Target frequency
        method: Aggregation method ('last', 'mean', 'sum', etc.)
        
    Returns:
        Resampled data
    """
    resampling_map = {
        'last': lambda x: x.last(),
        'first': lambda x: x.first(),
        'mean': lambda x: x.mean(),
        'sum': lambda x: x.sum(),
        'min': lambda x: x.min(),
        'max': lambda x: x.max(),
        'median': lambda x: x.median()
    }
    
    if method not in resampling_map:
        raise ValueError(f"Unknown resampling method: {method}")
    
    return resampling_map[method](data.resample(target_freq))


def align_dataframes(dfs: List[pd.DataFrame], join: str = 'inner') -> List[pd.DataFrame]:
    """
    Align multiple DataFrames by index.
    
    Args:
        dfs: List of DataFrames
        join: Join method ('inner', 'outer', 'left', 'right')
        
    Returns:
        List of aligned DataFrames
    """
    if not dfs:
        return []
    
    if len(dfs) == 1:
        return dfs
    
    # Get common index
    if join == 'inner':
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
    elif join == 'outer':
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.union(df.index)
    else:
        raise ValueError(f"Unsupported join method: {join}")
    
    # Align all DataFrames
    aligned_dfs = []
    for df in dfs:
        aligned_dfs.append(df.reindex(common_index))
    
    return aligned_dfs


def winsorize_data(data: pd.Series, lower_quantile: float = 0.01,
                  upper_quantile: float = 0.99) -> pd.Series:
    """
    Winsorize data by capping extreme values.
    
    Args:
        data: Input series
        lower_quantile: Lower quantile threshold
        upper_quantile: Upper quantile threshold
        
    Returns:
        Winsorized series
    """
    lower_bound = data.quantile(lower_quantile)
    upper_bound = data.quantile(upper_quantile)
    
    return data.clip(lower=lower_bound, upper=upper_bound)


def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data
    """
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    elif method == 'robust':
        return (data - data.median()) / (data.quantile(0.75) - data.quantile(0.25))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Formatting Utilities
def format_number(value: Union[int, float], decimal_places: int = 2,
                 prefix: str = '', suffix: str = '', 
                 thousands_sep: bool = True) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
        prefix: Prefix string
        suffix: Suffix string
        thousands_sep: Use thousands separator
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    # Format with decimal places
    if isinstance(value, int) and decimal_places == 0:
        formatted = str(value)
    else:
        formatted = f"{value:.{decimal_places}f}"
    
    # Add thousands separator
    if thousands_sep and '.' in formatted:
        integer_part, decimal_part = formatted.split('.')
        integer_part = '{:,}'.format(int(integer_part))
        formatted = f"{integer_part}.{decimal_part}"
    elif thousands_sep:
        formatted = '{:,}'.format(int(formatted))
    
    return f"{prefix}{formatted}{suffix}"


def format_percentage(value: float, decimal_places: int = 2,
                     include_sign: bool = True) -> str:
    """
    Format percentage for display.
    
    Args:
        value: Percentage value (0.05 = 5%)
        decimal_places: Number of decimal places
        include_sign: Include + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    
    percentage = value * 100
    formatted = f"{percentage:.{decimal_places}f}%"
    
    if include_sign and percentage > 0:
        formatted = f"+{formatted}"
    
    return formatted


def format_currency(value: Union[int, float], currency: str = 'USD',
                   decimal_places: int = 2) -> str:
    """
    Format currency value.
    
    Args:
        value: Currency value
        currency: Currency code
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CNY': '¥',
        'INR': '₹'
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    # Handle negative values
    if value < 0:
        return f"-{symbol}{format_number(abs(value), decimal_places)}"
    else:
        return f"{symbol}{format_number(value, decimal_places)}"


def format_date(dt: Union[datetime, date, str], format_string: str = '%Y-%m-%d') -> str:
    """
    Format date for display.
    
    Args:
        dt: Date to format
        format_string: strftime format string
        
    Returns:
        Formatted date string
    """
    if isinstance(dt, str):
        dt = parser.parse(dt)
    
    return dt.strftime(format_string)


# Validation Utilities
def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                      required_index_type: Optional[type] = None,
                      min_rows: int = 1) -> bool:
    """
    Validate DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: Required column names
        required_index_type: Required index type
        min_rows: Minimum number of rows
        
    Returns:
        True if valid
    """
    # Check if DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a DataFrame")
        return False
    
    # Check minimum rows
    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        return False
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
    
    # Check index type
    if required_index_type and not isinstance(df.index, required_index_type):
        logger.error(f"Index type {type(df.index)} does not match required type {required_index_type}")
        return False
    
    return True


def validate_returns(returns: pd.Series, max_return: float = 1.0,
                    min_return: float = -1.0) -> bool:
    """
    Validate returns series.
    
    Args:
        returns: Returns series
        max_return: Maximum acceptable return
        min_return: Minimum acceptable return
        
    Returns:
        True if valid
    """
    # Check for extreme values
    if returns.max() > max_return:
        logger.warning(f"Maximum return {returns.max():.2%} exceeds threshold {max_return:.2%}")
        return False
    
    if returns.min() < min_return:
        logger.warning(f"Minimum return {returns.min():.2%} below threshold {min_return:.2%}")
        return False
    
    # Check for NaN values
    if returns.isna().any():
        logger.warning(f"Returns contain {returns.isna().sum()} NaN values")
        return False
    
    return True


# File I/O Utilities
def save_data(data: Any, filepath: str, format: str = 'auto'):
    """
    Save data to file.
    
    Args:
        data: Data to save
        filepath: Output filepath
        format: File format ('auto', 'csv', 'excel', 'json', 'pickle', 'parquet')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = path.suffix[1:] if path.suffix else 'csv'
    
    if format == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filepath)
    elif format == 'excel' and isinstance(data, pd.DataFrame):
        data.to_excel(filepath)
    elif format == 'json':
        with open(filepath, 'w') as f:
            if isinstance(data, pd.DataFrame):
                data.to_json(f, orient='index')
            else:
                json.dump(data, f, indent=2, default=str)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'parquet' and isinstance(data, pd.DataFrame):
        data.to_parquet(filepath)
    elif format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data saved to {filepath}")


def load_data(filepath: str, format: str = 'auto') -> Any:
    """
    Load data from file.
    
    Args:
        filepath: Input filepath
        format: File format ('auto', 'csv', 'excel', 'json', 'pickle', 'parquet')
        
    Returns:
        Loaded data
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'auto':
        format = path.suffix[1:] if path.suffix else 'csv'
    
    if format == 'csv':
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == 'excel':
        return pd.read_excel(filepath, index_col=0, parse_dates=True)
    elif format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
                # Try to convert to DataFrame
                try:
                    return pd.DataFrame(data)
                except:
                    return data
            return data
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'yaml':
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Statistical Utilities
def calculate_correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix.
    
    Args:
        data: Input data
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    return data.corr(method=method)


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series,
                                window: int = 60) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series1: First series
        series2: Second series
        window: Rolling window size
        
    Returns:
        Rolling correlation series
    """
    return series1.rolling(window).corr(series2)


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data.
    
    Args:
        data: Input series
        method: Detection method ('iqr', 'zscore', 'isolation')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# Portfolio Utilities
def calculate_portfolio_weights(values: pd.Series) -> pd.Series:
    """
    Calculate portfolio weights from values.
    
    Args:
        values: Series of asset values
        
    Returns:
        Series of weights
    """
    total_value = values.sum()
    if total_value > 0:
        return values / total_value
    else:
        return pd.Series(0, index=values.index)


def rebalance_portfolio(current_weights: pd.Series, target_weights: pd.Series,
                       threshold: float = 0.05) -> pd.Series:
    """
    Calculate rebalancing trades.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Rebalancing threshold
        
    Returns:
        Series of required trades (positive = buy, negative = sell)
    """
    weight_diff = target_weights - current_weights
    
    # Only rebalance if difference exceeds threshold
    trades = weight_diff.copy()
    trades[abs(weight_diff) < threshold] = 0
    
    return trades


def calculate_tracking_error(portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series) -> float:
    """
    Calculate tracking error.
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Tracking error (annualized)
    """
    active_returns = portfolio_returns - benchmark_returns
    return active_returns.std() * np.sqrt(252)


# Utility Functions
def generate_unique_id(prefix: str = '') -> str:
    """
    Generate unique identifier.
    
    Args:
        prefix: Optional prefix
        
    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_suffix}"
    else:
        return f"{timestamp}_{random_suffix}"


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary.
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def safe_divide(numerator: Union[int, float], denominator: Union[int, float],
                default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safe division with default value for division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def round_to_decimals(value: float, decimals: int = 2) -> float:
    """
    Round to specified decimal places using banker's rounding.
    
    Args:
        value: Value to round
        decimals: Number of decimal places
        
    Returns:
        Rounded value
    """
    d = Decimal(str(value))
    return float(d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP))


def get_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive summary of DataFrame.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_values': data.isna().sum().to_dict(),
        'duplicates': data.duplicated().sum()
    }
    
    # Add numeric column statistics
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = data[numeric_cols].describe().to_dict()
    
    # Add date range if datetime index
    if isinstance(data.index, pd.DatetimeIndex):
        summary['date_range'] = {
            'start': data.index.min(),
            'end': data.index.max(),
            'periods': len(data.index),
            'frequency': pd.infer_freq(data.index)
        }
    
    return summary


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    
    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('[^a-zA-Z0-9_]', '', regex=True)
    )
    
    return df


def main():
    """Example usage of helper functions."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates)))), index=dates)
    returns = calculate_returns(prices)
    
    # Financial calculations
    print("Financial Metrics:")
    print(f"  Annualized Return: {format_percentage(calculate_annualized_return(returns))}")
    print(f"  Annualized Volatility: {format_percentage(calculate_annualized_volatility(returns))}")
    print(f"  Sharpe Ratio: {format_number(calculate_sharpe_ratio(returns))}")
    print(f"  Max Drawdown: {format_percentage(calculate_max_drawdown(prices)[0])}")
    
    # Date utilities
    print("\nBusiness Days in Q1 2023:")
    q1_start, q1_end = get_quarter_dates(2023, 1)
    business_days = get_business_days(q1_start, q1_end)
    print(f"  Total: {len(business_days)} days")
    
    # Data validation
    print("\nData Validation:")
    sample_df = pd.DataFrame({
        'price': prices[:100],
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    is_valid = validate_dataframe(
        sample_df,
        required_columns=['price', 'volume'],
        min_rows=50
    )
    print(f"  DataFrame valid: {is_valid}")
    
    # Formatting examples
    print("\nFormatting Examples:")
    print(f"  Currency: {format_currency(1234567.89)}")
    print(f"  Number: {format_number(1234567.89, thousands_sep=True)}")
    print(f"  Percentage: {format_percentage(0.1234)}")
    print(f"  Date: {format_date(datetime.now(), '%B %d, %Y')}")


if __name__ == "__main__":
    main()
