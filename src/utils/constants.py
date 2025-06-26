"""
Constants Module - System-wide Constants and Configuration
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module contains all constants, configurations, and static values
used throughout the financial analytics platform.
"""

from enum import Enum
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path

# ============================================================================
# SYSTEM INFORMATION
# ============================================================================

# Application metadata
APP_NAME = "DataNailytics Financial Analytics Platform"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Naiara Rodríguez Solano"
APP_EMAIL = "datanailytics@outlook.com"
APP_GITHUB = "https://github.com/datanailytics"
APP_PORTFOLIO = "https://datanailytics.github.io"
APP_LINKEDIN = "https://linkedin.com/in/naiara-rsolano"

# Copyright and licensing
COPYRIGHT = f"© 2024 {APP_AUTHOR}. All rights reserved."
LICENSE = "MIT License"

# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
CHARTS_DIR = OUTPUT_DIR / "charts"
LOGS_DIR = OUTPUT_DIR / "logs"
TEMP_DIR = OUTPUT_DIR / "temp"

# Configuration directories
CONFIG_DIR = BASE_DIR / "config"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR,
                  OUTPUT_DIR, REPORTS_DIR, CHARTS_DIR, LOGS_DIR, TEMP_DIR,
                  CONFIG_DIR, TEMPLATES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FINANCIAL CONSTANTS
# ============================================================================

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5
TRADING_HOURS_PER_DAY = 6.5  # NYSE regular trading hours

# Calendar days
DAYS_PER_YEAR = 365
DAYS_PER_MONTH = 30
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

# Market hours (Eastern Time)
MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "16:00"
PRE_MARKET_OPEN = "04:00"
AFTER_MARKET_CLOSE = "20:00"

# Risk-free rate defaults
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual
RISK_FREE_RATE_DAILY = DEFAULT_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

# Transaction costs
DEFAULT_COMMISSION = 0.0005  # 0.05% per trade
DEFAULT_SLIPPAGE = 0.0001   # 0.01% slippage
DEFAULT_SPREAD = 0.0002     # 0.02% bid-ask spread

# Tax rates
CAPITAL_GAINS_TAX_SHORT = 0.35  # Short-term capital gains
CAPITAL_GAINS_TAX_LONG = 0.15   # Long-term capital gains
DIVIDEND_TAX_RATE = 0.15         # Qualified dividends

# ============================================================================
# MARKET DATA CONSTANTS
# ============================================================================

# Major market indices
MARKET_INDICES = {
    'US': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI'],
    'INTERNATIONAL': ['EFA', 'EEM', 'VEA', 'VWO', 'VXUS'],
    'SECTORS': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE'],
    'BONDS': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'HYG', 'LQD'],
    'COMMODITIES': ['GLD', 'SLV', 'USO', 'DBA', 'DBC'],
    'VOLATILITY': ['VIX', 'VXX', 'UVXY', 'SVXY']
}

# Currency pairs
CURRENCY_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
    'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
]

# Cryptocurrencies
CRYPTO_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
    'SOL-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD'
]

# Exchange codes
EXCHANGES = {
    'US': ['NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS'],
    'EUROPE': ['LSE', 'EURONEXT', 'XETRA', 'SIX', 'BME'],
    'ASIA': ['TSE', 'HKEX', 'SSE', 'SZSE', 'NSE', 'KRX'],
    'CRYPTO': ['BINANCE', 'COINBASE', 'KRAKEN', 'GEMINI', 'FTX']
}

# Asset classes
ASSET_CLASSES = [
    'EQUITY', 'FIXED_INCOME', 'COMMODITY', 'CURRENCY', 
    'REAL_ESTATE', 'CRYPTO', 'ALTERNATIVE', 'CASH'
]

# Sectors (GICS)
SECTORS = [
    'Information Technology',
    'Health Care',
    'Financials',
    'Consumer Discretionary',
    'Communication Services',
    'Industrials',
    'Consumer Staples',
    'Energy',
    'Utilities',
    'Real Estate',
    'Materials'
]

# ============================================================================
# RISK MANAGEMENT CONSTANTS
# ============================================================================

# VaR confidence levels
VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
DEFAULT_VAR_CONFIDENCE = 0.95

# Risk limits
MAX_POSITION_SIZE = 0.10      # 10% max position size
MAX_SECTOR_EXPOSURE = 0.30    # 30% max sector exposure
MAX_LEVERAGE = 2.0            # 2x maximum leverage
MIN_LIQUIDITY_RATIO = 0.10    # 10% minimum cash

# Drawdown limits
MAX_DRAWDOWN_WARNING = 0.10   # 10% drawdown warning
MAX_DRAWDOWN_STOP = 0.20      # 20% drawdown stop

# Volatility targets
TARGET_VOLATILITY = 0.15      # 15% annual volatility target
MIN_VOLATILITY = 0.05         # 5% minimum volatility
MAX_VOLATILITY = 0.30         # 30% maximum volatility

# Risk metrics thresholds
MIN_SHARPE_RATIO = 0.5
MIN_SORTINO_RATIO = 0.7
MAX_BETA = 1.5
MIN_ALPHA = 0.0

# ============================================================================
# PORTFOLIO OPTIMIZATION CONSTANTS
# ============================================================================

# Optimization constraints
MIN_WEIGHT = 0.0              # Minimum asset weight
MAX_WEIGHT = 1.0              # Maximum asset weight
MIN_ASSETS = 5                # Minimum number of assets
MAX_ASSETS = 50               # Maximum number of assets

# Rebalancing parameters
REBALANCE_THRESHOLD = 0.05    # 5% deviation triggers rebalance
MIN_REBALANCE_INTERVAL = 30   # Minimum days between rebalances
REBALANCE_FREQUENCIES = ['DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUALLY']

# Portfolio types
PORTFOLIO_TYPES = [
    'CONSERVATIVE',
    'MODERATE',
    'AGGRESSIVE',
    'INCOME',
    'GROWTH',
    'BALANCED',
    'CUSTOM'
]

# Benchmark portfolios
BENCHMARK_ALLOCATIONS = {
    'CONSERVATIVE': {'EQUITY': 0.30, 'FIXED_INCOME': 0.60, 'CASH': 0.10},
    'MODERATE': {'EQUITY': 0.50, 'FIXED_INCOME': 0.40, 'CASH': 0.10},
    'AGGRESSIVE': {'EQUITY': 0.80, 'FIXED_INCOME': 0.15, 'CASH': 0.05},
    'INCOME': {'EQUITY': 0.20, 'FIXED_INCOME': 0.70, 'REAL_ESTATE': 0.10},
    'GROWTH': {'EQUITY': 0.90, 'FIXED_INCOME': 0.05, 'CASH': 0.05},
    'BALANCED': {'EQUITY': 0.60, 'FIXED_INCOME': 0.40}
}

# ============================================================================
# DATA PROCESSING CONSTANTS
# ============================================================================

# Data quality thresholds
MAX_MISSING_DATA_PCT = 0.10   # 10% maximum missing data
MIN_DATA_POINTS = 30          # Minimum data points for analysis
MAX_OUTLIER_PCT = 0.05        # 5% maximum outliers

# Time series parameters
DEFAULT_LOOKBACK_DAYS = 252   # 1 year default lookback
MIN_LOOKBACK_DAYS = 20        # Minimum lookback period
MAX_LOOKBACK_DAYS = 2520      # 10 years maximum lookback

# Rolling window sizes
ROLLING_WINDOWS = {
    'SHORT': 20,              # 1 month
    'MEDIUM': 60,             # 3 months
    'LONG': 252               # 1 year
}

# Data frequencies
DATA_FREQUENCIES = {
    'TICK': 'tick',
    'MINUTE': '1Min',
    'MINUTE_5': '5Min',
    'MINUTE_15': '15Min',
    'MINUTE_30': '30Min',
    'HOURLY': '1H',
    'DAILY': '1D',
    'WEEKLY': '1W',
    'MONTHLY': '1M',
    'QUARTERLY': '3M',
    'YEARLY': '1Y'
}

# ============================================================================
# TECHNICAL INDICATORS CONSTANTS
# ============================================================================

# Moving averages
MA_PERIODS = {
    'FAST': [5, 10, 20],
    'SLOW': [50, 100, 200]
}

# RSI parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2

# ATR period
ATR_PERIOD = 14

# Stochastic parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# ============================================================================
# MONTE CARLO SIMULATION CONSTANTS
# ============================================================================

# Simulation parameters
DEFAULT_NUM_SIMULATIONS = 10000
MIN_SIMULATIONS = 1000
MAX_SIMULATIONS = 100000

# Random seeds
DEFAULT_RANDOM_SEED = 42

# Confidence intervals
CONFIDENCE_INTERVALS = [0.05, 0.25, 0.50, 0.75, 0.95]

# Distribution types
DISTRIBUTION_TYPES = [
    'NORMAL',
    'LOGNORMAL',
    'STUDENT_T',
    'SKEW_NORMAL',
    'GARCH',
    'JUMP_DIFFUSION'
]

# ============================================================================
# API CONFIGURATION
# ============================================================================

# API rate limits
API_RATE_LIMITS = {
    'YAHOO_FINANCE': 2000,    # Requests per hour
    'ALPHA_VANTAGE': 500,     # Requests per day
    'QUANDL': 50000,          # Requests per day
    'IEX': 50000,             # Requests per month
    'POLYGON': 200            # Requests per minute
}

# API endpoints
API_ENDPOINTS = {
    'YAHOO_FINANCE': 'https://query1.finance.yahoo.com',
    'ALPHA_VANTAGE': 'https://www.alphavantage.co/query',
    'QUANDL': 'https://www.quandl.com/api/v3',
    'IEX': 'https://cloud.iexapis.com/stable',
    'POLYGON': 'https://api.polygon.io'
}

# Retry configuration
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0         # Seconds
API_TIMEOUT = 30              # Seconds

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Database defaults
DEFAULT_DB_TYPE = 'sqlite'
DEFAULT_DB_NAME = 'financial_data.db'
DEFAULT_DB_PATH = str(DATA_DIR / DEFAULT_DB_NAME)

# Connection pool settings
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_TIMEOUT = 30

# Table names
DB_TABLES = {
    'PRICES': 'market_prices',
    'FUNDAMENTALS': 'fundamentals',
    'PORTFOLIOS': 'portfolios',
    'TRANSACTIONS': 'transactions',
    'RISK_METRICS': 'risk_metrics',
    'REPORTS': 'reports'
}

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Color schemes
COLOR_SCHEMES = {
    'DEFAULT': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'PROFESSIONAL': ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#f5f5f5'],
    'COLORFUL': ['#ff006e', '#fb5607', '#ffbe0b', '#8338ec', '#3a86ff'],
    'MONOCHROME': ['#000000', '#333333', '#666666', '#999999', '#cccccc'],
    'GREEN_RED': ['#26a69a', '#ef5350', '#42a5f5', '#66bb6a', '#ec407a']
}

# Chart dimensions
CHART_WIDTH = 1200
CHART_HEIGHT = 600
THUMBNAIL_WIDTH = 400
THUMBNAIL_HEIGHT = 300

# Font sizes
FONT_SIZES = {
    'TITLE': 20,
    'SUBTITLE': 16,
    'LABEL': 12,
    'TICK': 10,
    'LEGEND': 10
}

# ============================================================================
# REPORTING CONSTANTS
# ============================================================================

# Report formats
REPORT_FORMATS = ['PDF', 'EXCEL', 'HTML', 'POWERPOINT', 'MARKDOWN', 'JSON']

# Report sections
STANDARD_REPORT_SECTIONS = [
    'EXECUTIVE_SUMMARY',
    'PORTFOLIO_OVERVIEW',
    'PERFORMANCE_ANALYSIS',
    'RISK_ANALYSIS',
    'HOLDINGS_DETAIL',
    'TRANSACTION_HISTORY',
    'COMPLIANCE_CHECK',
    'RECOMMENDATIONS'
]

# Report frequencies
REPORT_FREQUENCIES = ['DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUALLY', 'ON_DEMAND']

# Page settings
PAGE_MARGIN = 72  # Points (1 inch)
PAGE_WIDTH = 612  # Letter size width in points
PAGE_HEIGHT = 792  # Letter size height in points

# ============================================================================
# COMPLIANCE AND REGULATORY
# ============================================================================

# Regulatory bodies
REGULATORY_BODIES = {
    'US': ['SEC', 'FINRA', 'CFTC', 'OCC'],
    'EU': ['ESMA', 'ECB', 'FCA'],
    'GLOBAL': ['IOSCO', 'BIS', 'FSB']
}

# Compliance rules
COMPLIANCE_RULES = {
    'WASH_SALE_DAYS': 30,
    'PATTERN_DAY_TRADER_MIN': 25000,
    'MAX_MARGIN_RATIO': 0.5,
    'QUALIFIED_DIVIDEND_DAYS': 60
}

# Reporting requirements
REGULATORY_REPORTS = [
    'FORM_ADV',
    'FORM_13F',
    'SCHEDULE_D',
    'BLUE_SHEETS'
]

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'INVALID_DATA': "Invalid data provided",
    'MISSING_REQUIRED': "Missing required parameter: {param}",
    'CONNECTION_ERROR': "Unable to connect to {service}",
    'RATE_LIMIT': "Rate limit exceeded for {service}",
    'INSUFFICIENT_DATA': "Insufficient data for analysis",
    'CALCULATION_ERROR': "Error in calculation: {details}",
    'FILE_NOT_FOUND': "File not found: {filepath}",
    'PERMISSION_DENIED': "Permission denied: {resource}",
    'INVALID_FORMAT': "Invalid format: {format}",
    'TIMEOUT': "Operation timed out after {seconds} seconds"
}

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

SUCCESS_MESSAGES = {
    'DATA_LOADED': "Data loaded successfully",
    'REPORT_GENERATED': "Report generated: {filepath}",
    'ANALYSIS_COMPLETE': "Analysis completed successfully",
    'PORTFOLIO_OPTIMIZED': "Portfolio optimization complete",
    'DATA_SAVED': "Data saved to {filepath}",
    'CONNECTION_ESTABLISHED': "Connected to {service}",
    'CALCULATION_COMPLETE': "Calculation completed",
    'VALIDATION_PASSED': "Validation passed"
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_NAME = 'financial_analytics.log'
LOG_FILE_PATH = str(LOGS_DIR / LOG_FILE_NAME)
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_TTL = {
    'MARKET_DATA': 300,        # 5 minutes
    'FUNDAMENTALS': 86400,     # 1 day
    'REPORTS': 3600,           # 1 hour
    'CALCULATIONS': 1800,      # 30 minutes
    'STATIC_DATA': 604800      # 1 week
}

CACHE_MAX_SIZE = 1024 * 1024 * 100  # 100 MB

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

PERFORMANCE_THRESHOLDS = {
    'QUERY_TIMEOUT': 30,       # Seconds
    'MAX_MEMORY_MB': 4096,     # 4 GB
    'MAX_CPU_PERCENT': 80,     # 80% CPU usage
    'MAX_DATAFRAME_ROWS': 1000000,  # 1 million rows
    'CHUNK_SIZE': 10000        # Process in chunks
}

# ============================================================================
# ENVIRONMENT CONFIGURATIONS
# ============================================================================

ENVIRONMENTS = {
    'DEVELOPMENT': {
        'DEBUG': True,
        'LOG_LEVEL': 'DEBUG',
        'CACHE_ENABLED': False,
        'RATE_LIMIT_ENABLED': False
    },
    'TESTING': {
        'DEBUG': True,
        'LOG_LEVEL': 'INFO',
        'CACHE_ENABLED': True,
        'RATE_LIMIT_ENABLED': False
    },
    'STAGING': {
        'DEBUG': False,
        'LOG_LEVEL': 'INFO',
        'CACHE_ENABLED': True,
        'RATE_LIMIT_ENABLED': True
    },
    'PRODUCTION': {
        'DEBUG': False,
        'LOG_LEVEL': 'WARNING',
        'CACHE_ENABLED': True,
        'RATE_LIMIT_ENABLED': True
    }
}

# Default environment
DEFAULT_ENVIRONMENT = 'DEVELOPMENT'


def get_constant(key: str, default: Any = None) -> Any:
    """
    Get constant value by key.
    
    Args:
        key: Constant key
        default: Default value if not found
        
    Returns:
        Constant value
    """
    return globals().get(key, default)


def list_constants() -> List[str]:
    """
    List all available constants.
    
    Returns:
        List of constant names
    """
    return [key for key in globals().keys() 
            if key.isupper() and not key.startswith('_')]


def validate_constants() -> bool:
    """
    Validate all constants are properly defined.
    
    Returns:
        True if all constants are valid
    """
    required_constants = [
        'APP_NAME', 'APP_VERSION', 'BASE_DIR', 
        'TRADING_DAYS_PER_YEAR', 'DEFAULT_RISK_FREE_RATE'
    ]
    
    for const in required_constants:
        if const not in globals():
            print(f"Missing required constant: {const}")
            return False
    
    return True


def main():
    """Example usage of constants module."""
    print(f"{APP_NAME} v{APP_VERSION}")
    print(f"Author: {APP_AUTHOR}")
    print(f"Email: {APP_EMAIL}")
    print("-" * 50)
    
    print("\nSystem Paths:")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Reports Directory: {REPORTS_DIR}")
    
    print("\nFinancial Constants:")
    print(f"  Trading Days per Year: {TRADING_DAYS_PER_YEAR}")
    print(f"  Risk-Free Rate: {DEFAULT_RISK_FREE_RATE:.2%}")
    print(f"  Default Commission: {DEFAULT_COMMISSION:.3%}")
    
    print("\nRisk Limits:")
    print(f"  Max Position Size: {MAX_POSITION_SIZE:.1%}")
    print(f"  Max Drawdown Stop: {MAX_DRAWDOWN_STOP:.1%}")
    print(f"  Target Volatility: {TARGET_VOLATILITY:.1%}")
    
    print("\nAvailable Constants:")
    constants = list_constants()
    print(f"  Total: {len(constants)} constants defined")
    
    # Validate constants
    if validate_constants():
        print("\n✓ All required constants are properly defined")
    else:
        print("\n✗ Some required constants are missing")


if __name__ == "__main__":
    main()
