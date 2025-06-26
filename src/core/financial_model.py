"""
Financial Model Module - Core Analytics Engine
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module implements a comprehensive financial modeling framework for
enterprise-level data analytics, including DCF models, portfolio optimization,
risk metrics, and advanced statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of available financial model types."""
    DCF = "discounted_cash_flow"
    PORTFOLIO = "portfolio_optimization"
    MONTE_CARLO = "monte_carlo_simulation"
    BLACK_SCHOLES = "black_scholes_option"
    CREDIT_RISK = "credit_risk_model"
    VAR = "value_at_risk"
    FACTOR_MODEL = "factor_model"
    REGRESSION = "regression_analysis"


class RiskMetric(Enum):
    """Enumeration of risk metrics."""
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    MAX_DRAWDOWN = "maximum_drawdown"
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"


@dataclass
class FinancialAsset:
    """Represents a financial asset with its properties."""
    ticker: str
    name: str
    sector: str
    market_cap: float
    currency: str = "USD"
    exchange: str = "NYSE"
    asset_class: str = "equity"
    country: str = "US"
    isin: Optional[str] = None
    cusip: Optional[str] = None
    
    def __post_init__(self):
        """Validate asset data after initialization."""
        if self.market_cap <= 0:
            raise ValueError(f"Market cap must be positive for {self.ticker}")
        if not self.ticker:
            raise ValueError("Ticker symbol cannot be empty")


@dataclass
class ModelParameters:
    """Configuration parameters for financial models."""
    risk_free_rate: float = 0.02
    market_return: float = 0.08
    confidence_level: float = 0.95
    time_horizon: int = 252  # Trading days
    simulation_runs: int = 10000
    rebalancing_frequency: str = "quarterly"
    transaction_costs: float = 0.001
    tax_rate: float = 0.21
    inflation_rate: float = 0.025
    
    def validate(self) -> None:
        """Validate model parameters."""
        if not 0 <= self.risk_free_rate <= 1:
            raise ValueError("Risk-free rate must be between 0 and 1")
        if not 0 <= self.confidence_level <= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.simulation_runs <= 0:
            raise ValueError("Number of simulations must be positive")


class BaseFinancialModel(ABC):
    """Abstract base class for all financial models."""
    
    def __init__(self, parameters: ModelParameters):
        """
        Initialize base financial model.
        
        Args:
            parameters: Model configuration parameters
        """
        self.parameters = parameters
        self.parameters.validate()
        self.results = {}
        self.metadata = {
            "created_at": datetime.now(),
            "model_version": "1.0.0",
            "author": "Naiara Rodríguez Solano"
        }
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Abstract method for model calculations.
        
        Args:
            data: Input data for the model
            
        Returns:
            Dictionary containing calculation results
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for the model.
        
        Args:
            data: Input data to validate
            
        Returns:
            Boolean indicating if data is valid
        """
        pass
    
    def get_summary_statistics(self, data: pd.Series) -> Dict[str, float]:
        """
        Calculate summary statistics for a data series.
        
        Args:
            data: Pandas series with numerical data
            
        Returns:
            Dictionary of statistical measures
        """
        return {
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "variance": float(data.var()),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "min": float(data.min()),
            "max": float(data.max()),
            "q1": float(data.quantile(0.25)),
            "q3": float(data.quantile(0.75)),
            "iqr": float(data.quantile(0.75) - data.quantile(0.25))
        }


class DiscountedCashFlowModel(BaseFinancialModel):
    """Implementation of Discounted Cash Flow valuation model."""
    
    def __init__(self, parameters: ModelParameters, growth_rate: float = 0.03):
        """
        Initialize DCF model.
        
        Args:
            parameters: Model configuration
            growth_rate: Perpetual growth rate for terminal value
        """
        super().__init__(parameters)
        self.growth_rate = growth_rate
        self.discount_rate = 0.10  # WACC
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate DCF valuation.
        
        Args:
            data: DataFrame with cash flow projections
            
        Returns:
            Dictionary with valuation results
        """
        if not self.validate_inputs(data):
            raise ValueError("Invalid input data for DCF model")
        
        # Extract cash flows
        cash_flows = data['free_cash_flow'].values
        years = np.arange(1, len(cash_flows) + 1)
        
        # Calculate present value of cash flows
        discount_factors = (1 + self.discount_rate) ** years
        pv_cash_flows = cash_flows / discount_factors
        
        # Calculate terminal value
        terminal_growth_rate = min(self.growth_rate, self.parameters.inflation_rate)
        terminal_cash_flow = cash_flows[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_cash_flow / (self.discount_rate - terminal_growth_rate)
        pv_terminal_value = terminal_value / discount_factors[-1]
        
        # Calculate enterprise value
        enterprise_value = np.sum(pv_cash_flows) + pv_terminal_value
        
        # Sensitivity analysis
        sensitivity_results = self._sensitivity_analysis(
            cash_flows, terminal_growth_rate
        )
        
        results = {
            "enterprise_value": enterprise_value,
            "pv_cash_flows": np.sum(pv_cash_flows),
            "pv_terminal_value": pv_terminal_value,
            "terminal_value": terminal_value,
            "discount_rate": self.discount_rate,
            "growth_rate": self.growth_rate,
            "sensitivity_analysis": sensitivity_results,
            "cash_flow_breakdown": {
                "years": years.tolist(),
                "cash_flows": cash_flows.tolist(),
                "pv_cash_flows": pv_cash_flows.tolist(),
                "discount_factors": discount_factors.tolist()
            }
        }
        
        self.results = results
        return results
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate DCF input data."""
        required_columns = ['free_cash_flow']
        if not all(col in data.columns for col in required_columns):
            logger.error("Missing required columns for DCF model")
            return False
        
        if len(data) < 3:
            logger.error("Insufficient data points for DCF analysis")
            return False
        
        if data['free_cash_flow'].isna().any():
            logger.error("Missing values in cash flow data")
            return False
        
        return True
    
    def _sensitivity_analysis(self, cash_flows: np.ndarray, 
                            terminal_growth: float) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on key parameters.
        
        Args:
            cash_flows: Array of projected cash flows
            terminal_growth: Terminal growth rate
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        # Discount rate sensitivity
        discount_rates = np.linspace(
            self.discount_rate - 0.02, 
            self.discount_rate + 0.02, 
            5
        )
        
        # Growth rate sensitivity
        growth_rates = np.linspace(
            terminal_growth - 0.01,
            terminal_growth + 0.01,
            5
        )
        
        sensitivity_matrix = np.zeros((len(discount_rates), len(growth_rates)))
        
        for i, dr in enumerate(discount_rates):
            for j, gr in enumerate(growth_rates):
                # Recalculate enterprise value
                years = np.arange(1, len(cash_flows) + 1)
                discount_factors = (1 + dr) ** years
                pv_cf = cash_flows / discount_factors
                
                terminal_cf = cash_flows[-1] * (1 + gr)
                tv = terminal_cf / (dr - gr) if dr > gr else np.inf
                pv_tv = tv / discount_factors[-1] if tv != np.inf else 0
                
                sensitivity_matrix[i, j] = np.sum(pv_cf) + pv_tv
        
        return {
            "discount_rates": discount_rates.tolist(),
            "growth_rates": growth_rates.tolist(),
            "valuation_matrix": sensitivity_matrix.tolist(),
            "base_case": {
                "discount_rate": self.discount_rate,
                "growth_rate": terminal_growth
            }
        }
    
    def calculate_wacc(self, equity_value: float, debt_value: float,
                      cost_of_equity: float, cost_of_debt: float) -> float:
        """
        Calculate Weighted Average Cost of Capital.
        
        Args:
            equity_value: Market value of equity
            debt_value: Market value of debt
            cost_of_equity: Cost of equity capital
            cost_of_debt: Cost of debt capital
            
        Returns:
            WACC value
        """
        total_value = equity_value + debt_value
        equity_weight = equity_value / total_value
        debt_weight = debt_value / total_value
        
        after_tax_cost_of_debt = cost_of_debt * (1 - self.parameters.tax_rate)
        wacc = (equity_weight * cost_of_equity + 
                debt_weight * after_tax_cost_of_debt)
        
        return wacc


class PortfolioOptimizationModel(BaseFinancialModel):
    """Modern Portfolio Theory implementation with advanced optimization."""
    
    def __init__(self, parameters: ModelParameters):
        """Initialize portfolio optimization model."""
        super().__init__(parameters)
        self.optimization_method = 'SLSQP'
        self.constraints = []
        self.bounds = []
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform portfolio optimization.
        
        Args:
            data: DataFrame with asset returns
            
        Returns:
            Dictionary with optimization results
        """
        if not self.validate_inputs(data):
            raise ValueError("Invalid input data for portfolio optimization")
        
        # Calculate returns and covariance matrix
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        n_assets = len(data.columns)
        
        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
        ]
        
        # Asset bounds (0 to 1 for each asset)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array(n_assets * [1. / n_assets])
        
        # Optimize for different objectives
        min_variance_weights = self._minimize_variance(
            mean_returns, cov_matrix, constraints, bounds, initial_weights
        )
        
        max_sharpe_weights = self._maximize_sharpe_ratio(
            mean_returns, cov_matrix, constraints, bounds, initial_weights
        )
        
        # Calculate efficient frontier
        efficient_frontier = self._calculate_efficient_frontier(
            mean_returns, cov_matrix, constraints, bounds
        )
        
        # Risk metrics for optimal portfolios
        min_var_metrics = self._calculate_portfolio_metrics(
            min_variance_weights, mean_returns, cov_matrix
        )
        
        max_sharpe_metrics = self._calculate_portfolio_metrics(
            max_sharpe_weights, mean_returns, cov_matrix
        )
        
        results = {
            "assets": data.columns.tolist(),
            "min_variance_portfolio": {
                "weights": min_variance_weights.tolist(),
                "metrics": min_var_metrics
            },
            "max_sharpe_portfolio": {
                "weights": max_sharpe_weights.tolist(),
                "metrics": max_sharpe_metrics
            },
            "efficient_frontier": efficient_frontier,
            "correlation_matrix": returns.corr().to_dict(),
            "asset_statistics": {
                col: self.get_summary_statistics(returns[col])
                for col in returns.columns
            }
        }
        
        self.results = results
        return results
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate portfolio input data."""
        if data.empty:
            logger.error("Empty DataFrame provided")
            return False
        
        if len(data) < 30:
            logger.error("Insufficient data points for reliable optimization")
            return False
        
        if data.isna().any().any():
            logger.error("Missing values in return data")
            return False
        
        return True
    
    def _minimize_variance(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                          constraints: List, bounds: Tuple, initial_weights: np.ndarray) -> np.ndarray:
        """Minimize portfolio variance."""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method=self.optimization_method,
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def _maximize_sharpe_ratio(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                              constraints: List, bounds: Tuple, initial_weights: np.ndarray) -> np.ndarray:
        """Maximize Sharpe ratio."""
        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.parameters.risk_free_rate) / portfolio_std
            return -sharpe
        
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            method=self.optimization_method,
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, 
                                   mean_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        sharpe_ratio = (portfolio_return - self.parameters.risk_free_rate) / portfolio_std
        
        # Downside deviation for Sortino ratio
        downside_returns = mean_returns[mean_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns**2))
            sortino_ratio = (portfolio_return - self.parameters.risk_free_rate) / downside_deviation
        else:
            sortino_ratio = np.inf
        
        return {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_std),
            "variance": float(portfolio_variance),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "risk_adjusted_return": float(portfolio_return / portfolio_std)
        }
    
    def _calculate_efficient_frontier(self, mean_returns: pd.Series, 
                                    cov_matrix: pd.DataFrame,
                                    constraints: List, bounds: Tuple) -> Dict[str, List[float]]:
        """Calculate the efficient frontier."""
        n_portfolios = 50
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_portfolios)
        
        frontier_volatility = []
        frontier_weights = []
        
        for target_return in target_returns:
            # Add return constraint
            return_constraint = {
                'type': 'eq',
                'fun': lambda x, r=target_return: np.dot(x, mean_returns) - r
            }
            
            all_constraints = constraints + [return_constraint]
            
            # Minimize variance for target return
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            result = minimize(
                portfolio_variance,
                np.array(len(mean_returns) * [1. / len(mean_returns)]),
                method=self.optimization_method,
                bounds=bounds,
                constraints=all_constraints
            )
            
            if result.success:
                frontier_volatility.append(np.sqrt(result.fun))
                frontier_weights.append(result.x.tolist())
        
        return {
            "returns": target_returns.tolist(),
            "volatilities": frontier_volatility,
            "weights": frontier_weights
        }


class MonteCarloSimulation(BaseFinancialModel):
    """Monte Carlo simulation for financial modeling and risk analysis."""
    
    def __init__(self, parameters: ModelParameters):
        """Initialize Monte Carlo simulation model."""
        super().__init__(parameters)
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            data: Historical price data
            
        Returns:
            Dictionary with simulation results
        """
        if not self.validate_inputs(data):
            raise ValueError("Invalid input data for Monte Carlo simulation")
        
        # Calculate historical parameters
        returns = data.pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Run simulations for each asset
        simulation_results = {}
        
        for asset in data.columns:
            asset_simulations = self._simulate_asset_paths(
                data[asset].iloc[-1],  # Last price
                mean_return[asset],
                std_return[asset],
                self.parameters.time_horizon,
                self.parameters.simulation_runs
            )
            
            # Calculate statistics
            final_prices = asset_simulations[:, -1]
            
            simulation_results[asset] = {
                "current_price": float(data[asset].iloc[-1]),
                "expected_price": float(np.mean(final_prices)),
                "median_price": float(np.median(final_prices)),
                "std_price": float(np.std(final_prices)),
                "percentiles": {
                    "5%": float(np.percentile(final_prices, 5)),
                    "25%": float(np.percentile(final_prices, 25)),
                    "75%": float(np.percentile(final_prices, 75)),
                    "95%": float(np.percentile(final_prices, 95))
                },
                "probability_of_profit": float(np.mean(final_prices > data[asset].iloc[-1])),
                "var_95": float(np.percentile(final_prices - data[asset].iloc[-1], 5)),
                "cvar_95": float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])),
                "max_simulated": float(np.max(final_prices)),
                "min_simulated": float(np.min(final_prices))
            }
        
        # Portfolio simulation if multiple assets
        if len(data.columns) > 1:
            portfolio_results = self._simulate_portfolio(
                data, mean_return, std_return, returns.corr()
            )
            simulation_results["portfolio"] = portfolio_results
        
        results = {
            "simulation_parameters": {
                "runs": self.parameters.simulation_runs,
                "time_horizon": self.parameters.time_horizon,
                "random_seed": self.random_seed
            },
            "asset_simulations": simulation_results,
            "model_assumptions": {
                "distribution": "log-normal",
                "constant_parameters": True,
                "no_dividends": True
            }
        }
        
        self.results = results
        return results
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate Monte Carlo input data."""
        if data.empty:
            logger.error("Empty DataFrame provided")
            return False
        
        if len(data) < 60:
            logger.error("Insufficient data for reliable parameter estimation")
            return False
        
        return True
    
    def _simulate_asset_paths(self, initial_price: float, mean_return: float,
                            std_return: float, time_steps: int, 
                            n_simulations: int) -> np.ndarray:
        """
        Simulate asset price paths using geometric Brownian motion.
        
        Args:
            initial_price: Starting price
            mean_return: Expected return
            std_return: Return volatility
            time_steps: Number of time periods
            n_simulations: Number of simulation runs
            
        Returns:
            Array of simulated price paths
        """
        dt = 1 / 252  # Daily time step
        
        # Generate random shocks
        random_shocks = np.random.normal(
            0, 1, size=(n_simulations, time_steps)
        )
        
        # Initialize price paths
        price_paths = np.zeros((n_simulations, time_steps + 1))
        price_paths[:, 0] = initial_price
        
        # Simulate paths
        for t in range(1, time_steps + 1):
            drift = (mean_return - 0.5 * std_return**2) * dt
            diffusion = std_return * np.sqrt(dt) * random_shocks[:, t-1]
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion)
        
        return price_paths
    
    def _simulate_portfolio(self, data: pd.DataFrame, mean_returns: pd.Series,
                          std_returns: pd.Series, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Simulate portfolio performance."""
        n_assets = len(data.columns)
        weights = np.array(n_assets * [1. / n_assets])  # Equal weights
        
        # Portfolio parameters
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(correlation_matrix * np.outer(std_returns, std_returns), weights))
        )
        
        # Simulate portfolio
        initial_value = np.dot(weights, data.iloc[-1])
        portfolio_paths = self._simulate_asset_paths(
            initial_value, portfolio_return, portfolio_std,
            self.parameters.time_horizon, self.parameters.simulation_runs
        )
        
        final_values = portfolio_paths[:, -1]
        
        return {
            "initial_value": float(initial_value),
            "expected_value": float(np.mean(final_values)),
            "median_value": float(np.median(final_values)),
            "std_value": float(np.std(final_values)),
            "var_95": float(np.percentile(final_values - initial_value, 5)),
            "cvar_95": float(np.mean(final_values[final_values <= np.percentile(final_values, 5)])),
            "probability_of_loss": float(np.mean(final_values < initial_value)),
            "expected_return": float(portfolio_return),
            "portfolio_volatility": float(portfolio_std),
            "weights": weights.tolist()
        }


class RiskAnalysisModel(BaseFinancialModel):
    """Comprehensive risk analysis and metrics calculation."""
    
    def __init__(self, parameters: ModelParameters):
        """Initialize risk analysis model."""
        super().__init__(parameters)
        self.risk_metrics = {}
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            data: Price or return data
            
        Returns:
            Dictionary with risk analysis results
        """
        if not self.validate_inputs(data):
            raise ValueError("Invalid input data for risk analysis")
        
        # Calculate returns if price data provided
        if all(data.iloc[0] > 1):  # Assume price data
            returns = data.pct_change().dropna()
        else:
            returns = data
        
        # Individual asset risk metrics
        asset_risk_metrics = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset]
            
            # Basic risk metrics
            volatility = self._calculate_volatility(asset_returns)
            var = self._calculate_var(asset_returns, self.parameters.confidence_level)
            cvar = self._calculate_cvar(asset_returns, self.parameters.confidence_level)
            max_drawdown = self._calculate_max_drawdown(data[asset] if asset in data.columns else asset_returns)
            
            # Advanced metrics
            downside_deviation = self._calculate_downside_deviation(asset_returns)
            omega_ratio = self._calculate_omega_ratio(asset_returns)
            calmar_ratio = self._calculate_calmar_ratio(asset_returns, max_drawdown)
            
            # Higher moments
            skewness = float(asset_returns.skew())
            kurtosis = float(asset_returns.kurtosis())
            
            asset_risk_metrics[asset] = {
                "volatility": volatility,
                "annualized_volatility": volatility * np.sqrt(252),
                "var_95": var,
                "cvar_95": cvar,
                "max_drawdown": max_drawdown,
                "downside_deviation": downside_deviation,
                "omega_ratio": omega_ratio,
                "calmar_ratio": calmar_ratio,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "risk_adjusted_metrics": {
                    "sharpe_ratio": self._calculate_sharpe_ratio(asset_returns, volatility),
                    "sortino_ratio": self._calculate_sortino_ratio(asset_returns, downside_deviation),
                    "information_ratio": self._calculate_information_ratio(asset_returns)
                }
            }
        
        # Portfolio risk analysis if multiple assets
        if len(returns.columns) > 1:
            portfolio_risk = self._analyze_portfolio_risk(returns)
        else:
            portfolio_risk = None
        
        # Stress testing
        stress_test_results = self._perform_stress_tests(returns)
        
        results = {
            "asset_risk_metrics": asset_risk_metrics,
            "portfolio_risk": portfolio_risk,
            "stress_test_results": stress_test_results,
            "risk_summary": self._generate_risk_summary(asset_risk_metrics),
            "correlation_analysis": {
                "correlation_matrix": returns.corr().to_dict(),
                "covariance_matrix": returns.cov().to_dict()
            }
        }
        
        self.results = results
        return results
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate risk analysis input data."""
        if data.empty:
            logger.error("Empty DataFrame provided")
            return False
        
        if len(data) < 20:
            logger.error("Insufficient data for risk analysis")
            return False
        
        return True
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate return volatility."""
        return float(returns.std())
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(returns, confidence_level)
        return float(returns[returns <= var].mean())
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min())
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            return float(np.sqrt(np.mean(negative_returns**2)))
        return 0.0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) > 0 and losses.sum() > 0:
            return float(gains.sum() / losses.sum())
        return np.inf
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        annual_return = returns.mean() * 252
        if max_drawdown != 0:
            return float(annual_return / abs(max_drawdown))
        return np.inf
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        excess_return = returns.mean() - self.parameters.risk_free_rate / 252
        if volatility > 0:
            return float(excess_return / volatility * np.sqrt(252))
        return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, downside_deviation: float) -> float:
        """Calculate Sortino ratio."""
        excess_return = returns.mean() - self.parameters.risk_free_rate / 252
        if downside_deviation > 0:
            return float(excess_return / downside_deviation * np.sqrt(252))
        return np.inf
    
    def _calculate_information_ratio(self, returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series] = None) -> float:
        """Calculate Information ratio."""
        if benchmark_returns is None:
            # Use risk-free rate as benchmark
            benchmark_returns = pd.Series(
                self.parameters.risk_free_rate / 252, 
                index=returns.index
            )
        
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error > 0:
            return float(active_returns.mean() / tracking_error * np.sqrt(252))
        return 0.0
    
    def _analyze_portfolio_risk(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio-level risk metrics."""
        # Equal weight portfolio
        n_assets = len(returns.columns)
        weights = np.array(n_assets * [1. / n_assets])
        
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Portfolio metrics
        portfolio_volatility = portfolio_returns.std()
        portfolio_var = self._calculate_var(portfolio_returns, self.parameters.confidence_level)
        portfolio_cvar = self._calculate_cvar(portfolio_returns, self.parameters.confidence_level)
        
        # Diversification ratio
        weighted_avg_volatility = np.sum(weights * returns.std())
        diversification_ratio = weighted_avg_volatility / portfolio_volatility
        
        # Marginal VaR
        marginal_var = self._calculate_marginal_var(returns, weights)
        
        # Component VaR
        component_var = self._calculate_component_var(returns, weights, portfolio_var)
        
        return {
            "portfolio_volatility": float(portfolio_volatility * np.sqrt(252)),
            "portfolio_var": float(portfolio_var),
            "portfolio_cvar": float(portfolio_cvar),
            "diversification_ratio": float(diversification_ratio),
            "effective_number_of_assets": float(1 / np.sum(weights**2)),
            "marginal_var": marginal_var,
            "component_var": component_var,
            "risk_contribution": {
                asset: float(component_var[asset] / portfolio_var)
                for asset in returns.columns
            }
        }
    
    def _calculate_marginal_var(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate marginal VaR for each asset."""
        marginal_var = {}
        base_portfolio_returns = returns @ weights
        base_var = self._calculate_var(base_portfolio_returns, self.parameters.confidence_level)
        
        for i, asset in enumerate(returns.columns):
            # Increase weight by 1%
            marginal_weights = weights.copy()
            marginal_weights[i] += 0.01
            marginal_weights = marginal_weights / marginal_weights.sum()
            
            marginal_portfolio_returns = returns @ marginal_weights
            marginal_portfolio_var = self._calculate_var(
                marginal_portfolio_returns, self.parameters.confidence_level
            )
            
            marginal_var[asset] = float((marginal_portfolio_var - base_var) / 0.01)
        
        return marginal_var
    
    def _calculate_component_var(self, returns: pd.DataFrame, weights: np.ndarray, 
                               portfolio_var: float) -> Dict[str, float]:
        """Calculate component VaR for each asset."""
        component_var = {}
        
        for i, asset in enumerate(returns.columns):
            asset_weight = weights[i]
            asset_returns = returns[asset]
            portfolio_returns = returns @ weights
            
            # Calculate beta of asset to portfolio
            covariance = np.cov(asset_returns, portfolio_returns)[0, 1]
            portfolio_variance = portfolio_returns.var()
            beta = covariance / portfolio_variance if portfolio_variance > 0 else 0
            
            component_var[asset] = float(asset_weight * beta * portfolio_var)
        
        return component_var
    
    def _perform_stress_tests(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Perform stress testing scenarios."""
        stress_scenarios = {
            "market_crash": -0.20,  # 20% market decline
            "volatility_spike": 2.0,  # Double volatility
            "correlation_breakdown": 0.9,  # High correlation
            "black_swan": -0.40  # 40% decline
        }
        
        stress_results = {}
        
        for scenario, shock in stress_scenarios.items():
            if scenario == "market_crash" or scenario == "black_swan":
                # Apply market shock
                stressed_returns = returns + shock / 252
                portfolio_returns = stressed_returns.mean(axis=1)
                
                stress_results[scenario] = {
                    "portfolio_loss": float(portfolio_returns.sum()),
                    "worst_asset": returns.sum().idxmin(),
                    "worst_asset_loss": float(returns.sum().min() + shock)
                }
                
            elif scenario == "volatility_spike":
                # Increase volatility
                stressed_vol = returns.std() * shock
                stress_results[scenario] = {
                    "new_var_95": float(stats.norm.ppf(0.05) * stressed_vol.mean()),
                    "volatility_increase": f"{(shock - 1) * 100:.0f}%"
                }
                
            elif scenario == "correlation_breakdown":
                # High correlation scenario
                n_assets = len(returns.columns)
                high_corr_matrix = np.full((n_assets, n_assets), shock)
                np.fill_diagonal(high_corr_matrix, 1.0)
                
                stress_results[scenario] = {
                    "diversification_loss": float(1 - 1/shock),
                    "portfolio_volatility_increase": f"{shock * 100:.0f}%"
                }
        
        return stress_results
    
    def _generate_risk_summary(self, asset_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive risk summary."""
        # Identify highest risk assets
        volatilities = {
            asset: metrics["annualized_volatility"] 
            for asset, metrics in asset_metrics.items()
        }
        
        max_drawdowns = {
            asset: metrics["max_drawdown"] 
            for asset, metrics in asset_metrics.items()
        }
        
        return {
            "highest_volatility_asset": max(volatilities, key=volatilities.get),
            "lowest_volatility_asset": min(volatilities, key=volatilities.get),
            "highest_drawdown_asset": min(max_drawdowns, key=max_drawdowns.get),
            "average_volatility": float(np.mean(list(volatilities.values()))),
            "risk_rating": self._calculate_risk_rating(volatilities, max_drawdowns)
        }
    
    def _calculate_risk_rating(self, volatilities: Dict[str, float], 
                             drawdowns: Dict[str, float]) -> str:
        """Calculate overall portfolio risk rating."""
        avg_volatility = np.mean(list(volatilities.values()))
        avg_drawdown = np.mean(list(drawdowns.values()))
        
        if avg_volatility < 0.15 and avg_drawdown > -0.10:
            return "Low Risk"
        elif avg_volatility < 0.25 and avg_drawdown > -0.20:
            return "Medium Risk"
        elif avg_volatility < 0.35 and avg_drawdown > -0.30:
            return "High Risk"
        else:
            return "Very High Risk"


class FinancialModelFactory:
    """Factory class for creating financial models."""
    
    @staticmethod
    def create_model(model_type: ModelType, parameters: ModelParameters) -> BaseFinancialModel:
        """
        Create a financial model instance.
        
        Args:
            model_type: Type of model to create
            parameters: Model configuration parameters
            
        Returns:
            Instance of the requested model
        """
        model_mapping = {
            ModelType.DCF: DiscountedCashFlowModel,
            ModelType.PORTFOLIO: PortfolioOptimizationModel,
            ModelType.MONTE_CARLO: MonteCarloSimulation,
            ModelType.VAR: RiskAnalysisModel
        }
        
        model_class = model_mapping.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_class(parameters)


class FinancialAnalyzer:
    """High-level financial analysis orchestrator."""
    
    def __init__(self):
        """Initialize financial analyzer."""
        self.models = {}
        self.results = {}
        self.parameters = ModelParameters()
        
    def add_model(self, name: str, model_type: ModelType, 
                 custom_parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a model to the analyzer.
        
        Args:
            name: Unique name for the model
            model_type: Type of model to add
            custom_parameters: Optional custom parameters
        """
        params = ModelParameters()
        if custom_parameters:
            for key, value in custom_parameters.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        
        model = FinancialModelFactory.create_model(model_type, params)
        self.models[name] = model
        
    def run_analysis(self, data: pd.DataFrame, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run financial analysis using specified models.
        
        Args:
            data: Input data for analysis
            model_names: List of model names to run (None = run all)
            
        Returns:
            Dictionary with all analysis results
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        for name in model_names:
            if name not in self.models:
                logger.warning(f"Model '{name}' not found, skipping")
                continue
            
            try:
                model = self.models[name]
                results = model.calculate(data)
                self.results[name] = {
                    "success": True,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error running model '{name}': {str(e)}")
                self.results[name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "author": "Naiara Rodríguez Solano",
                "models_used": list(self.models.keys()),
                "success_rate": self._calculate_success_rate()
            },
            "executive_summary": self._generate_executive_summary(),
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of model runs."""
        if not self.results:
            return 0.0
        
        successful = sum(1 for r in self.results.values() if r.get("success", False))
        return successful / len(self.results)
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of results."""
        summary = {
            "models_executed": len(self.results),
            "successful_runs": sum(1 for r in self.results.values() if r.get("success", False)),
            "key_findings": []
        }
        
        # Extract key findings from successful models
        for name, result in self.results.items():
            if result.get("success") and "results" in result:
                if name.startswith("portfolio"):
                    sharpe = result["results"].get("max_sharpe_portfolio", {}).get("metrics", {}).get("sharpe_ratio")
                    if sharpe:
                        summary["key_findings"].append(
                            f"Optimal portfolio achieved Sharpe ratio of {sharpe:.2f}"
                        )
                elif name.startswith("risk"):
                    risk_summary = result["results"].get("risk_summary", {})
                    if risk_summary:
                        summary["key_findings"].append(
                            f"Portfolio risk rating: {risk_summary.get('risk_rating', 'Unknown')}"
                        )
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate investment recommendations based on analysis."""
        recommendations = []
        
        # Analyze results and generate recommendations
        for name, result in self.results.items():
            if not result.get("success"):
                continue
            
            if name.startswith("portfolio") and "results" in result:
                max_sharpe = result["results"].get("max_sharpe_portfolio", {})
                if max_sharpe:
                    recommendations.append(
                        "Consider rebalancing portfolio according to maximum Sharpe ratio weights"
                    )
            
            if name.startswith("risk") and "results" in result:
                risk_summary = result["results"].get("risk_summary", {})
                if risk_summary.get("risk_rating") in ["High Risk", "Very High Risk"]:
                    recommendations.append(
                        "Portfolio shows elevated risk levels - consider diversification"
                    )
        
        return recommendations


def main():
    """Example usage of the financial model module."""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Simulate stock prices
    np.random.seed(42)
    stocks = {
        'AAPL': 100 * np.exp(np.cumsum(np.random.normal(0.0008, 0.02, len(dates)))),
        'GOOGL': 1000 * np.exp(np.cumsum(np.random.normal(0.0007, 0.025, len(dates)))),
        'MSFT': 200 * np.exp(np.cumsum(np.random.normal(0.0009, 0.018, len(dates)))),
        'AMZN': 3000 * np.exp(np.cumsum(np.random.normal(0.0006, 0.03, len(dates))))
    }
    
    price_data = pd.DataFrame(stocks, index=dates)
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer()
    
    # Add models
    analyzer.add_model("portfolio_optimization", ModelType.PORTFOLIO)
    analyzer.add_model("risk_analysis", ModelType.VAR)
    analyzer.add_model("monte_carlo", ModelType.MONTE_CARLO, 
                      {"simulation_runs": 1000})
    
    # Run analysis
    results = analyzer.run_analysis(price_data)
    
    # Generate report
    report = analyzer.generate_report()
    
    print("Financial Analysis Complete")
    print(f"Models executed: {report['report_metadata']['models_used']}")
    print(f"Success rate: {report['report_metadata']['success_rate']:.1%}")
    print("\nKey Findings:")
    for finding in report['executive_summary']['key_findings']:
        print(f"- {finding}")


if __name__ == "__main__":
    main()
