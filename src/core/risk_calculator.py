"""
Risk Calculator Module - Advanced Risk Analytics
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides comprehensive risk calculation and analysis tools
for financial portfolios, including VaR, CVaR, stress testing, and
advanced risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMeasure(Enum):
    """Types of risk measures available."""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    VOLATILITY = "volatility"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    DOWNSIDE_RISK = "downside_risk"
    MAX_DRAWDOWN = "maximum_drawdown"
    EXPECTED_SHORTFALL = "expected_shortfall"
    TAIL_RISK = "tail_risk"
    SYSTEMIC_RISK = "systemic_risk"


class RiskMethod(Enum):
    """Methods for calculating risk measures."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EXTREME_VALUE = "extreme_value_theory"
    GARCH = "garch"


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    annual_volatility: float
    downside_deviation: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    treynor_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    kurtosis: Optional[float] = None
    skewness: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class RiskCalculator:
    """Main class for calculating financial risk metrics."""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize risk calculator.
        
        Args:
            confidence_levels: Confidence levels for VaR/CVaR calculations
        """
        self.confidence_levels = confidence_levels
        self.risk_free_rate = 0.02  # Annual risk-free rate
        
    def calculate_risk_metrics(self, returns: pd.DataFrame,
                             benchmark_returns: Optional[pd.DataFrame] = None,
                             method: RiskMethod = RiskMethod.HISTORICAL) -> Dict[str, RiskMetrics]:
        """
        Calculate comprehensive risk metrics for assets.
        
        Args:
            returns: DataFrame of asset returns
            benchmark_returns: Optional benchmark returns
            method: Method for risk calculation
            
        Returns:
            Dictionary mapping asset names to RiskMetrics
        """
        results = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            # Basic risk measures
            var_95 = self.calculate_var(asset_returns, 0.95, method)
            var_99 = self.calculate_var(asset_returns, 0.99, method)
            cvar_95 = self.calculate_cvar(asset_returns, 0.95, method)
            cvar_99 = self.calculate_cvar(asset_returns, 0.99, method)
            
            # Volatility measures
            volatility = asset_returns.std()
            annual_volatility = volatility * np.sqrt(252)
            downside_deviation = self.calculate_downside_deviation(asset_returns)
            
            # Drawdown
            max_drawdown = self.calculate_max_drawdown(asset_returns)
            
            # Risk-adjusted returns
            sharpe_ratio = self.calculate_sharpe_ratio(asset_returns)
            sortino_ratio = self.calculate_sortino_ratio(asset_returns, downside_deviation)
            calmar_ratio = self.calculate_calmar_ratio(asset_returns, max_drawdown)
            
            # Market-related metrics
            if benchmark_returns is not None and asset in benchmark_returns.columns:
                beta = self.calculate_beta(asset_returns, benchmark_returns[asset])
                alpha = self.calculate_alpha(asset_returns, benchmark_returns[asset], beta)
                tracking_error = self.calculate_tracking_error(asset_returns, benchmark_returns[asset])
                information_ratio = self.calculate_information_ratio(
                    asset_returns, benchmark_returns[asset], tracking_error
                )
                treynor_ratio = self.calculate_treynor_ratio(asset_returns, beta)
            else:
                beta = alpha = tracking_error = information_ratio = treynor_ratio = None
            
            # Additional metrics
            omega_ratio = self.calculate_omega_ratio(asset_returns)
            kurtosis = asset_returns.kurtosis()
            skewness = asset_returns.skew()
            
            # Create RiskMetrics object
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                annual_volatility=annual_volatility,
                downside_deviation=downside_deviation,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio or 0.0,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                treynor_ratio=treynor_ratio,
                omega_ratio=omega_ratio,
                kurtosis=kurtosis,
                skewness=skewness
            )
            
            results[asset] = metrics
        
        return results
    
    def calculate_var(self, returns: pd.Series, confidence_level: float,
                     method: RiskMethod = RiskMethod.HISTORICAL) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95)
            method: Calculation method
            
        Returns:
            VaR value
        """
        if method == RiskMethod.HISTORICAL:
            return self._historical_var(returns, confidence_level)
        elif method == RiskMethod.PARAMETRIC:
            return self._parametric_var(returns, confidence_level)
        elif method == RiskMethod.CORNISH_FISHER:
            return self._cornish_fisher_var(returns, confidence_level)
        elif method == RiskMethod.MONTE_CARLO:
            return self._monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float,
                      method: RiskMethod = RiskMethod.HISTORICAL) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence_level, method)
        
        if method == RiskMethod.HISTORICAL:
            return returns[returns <= var].mean()
        elif method == RiskMethod.PARAMETRIC:
            # For normal distribution
            alpha = 1 - confidence_level
            pdf_alpha = stats.norm.pdf(stats.norm.ppf(alpha))
            return returns.mean() - returns.std() * pdf_alpha / alpha
        else:
            # For other methods, use historical approach on tail
            return returns[returns <= var].mean()
    
    def calculate_portfolio_var(self, returns: pd.DataFrame, weights: np.ndarray,
                              confidence_level: float = 0.95,
                              method: RiskMethod = RiskMethod.HISTORICAL) -> float:
        """
        Calculate portfolio VaR.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            Portfolio VaR
        """
        # Calculate portfolio returns
        portfolio_returns = returns @ weights
        
        return self.calculate_var(portfolio_returns, confidence_level, method)
    
    def calculate_marginal_var(self, returns: pd.DataFrame, weights: np.ndarray,
                             confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate marginal VaR for each asset.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence_level: Confidence level
            
        Returns:
            Dictionary of marginal VaRs
        """
        base_var = self.calculate_portfolio_var(returns, weights, confidence_level)
        marginal_vars = {}
        delta = 0.01  # 1% change in weight
        
        for i, asset in enumerate(returns.columns):
            # Increase weight slightly
            new_weights = weights.copy()
            new_weights[i] += delta
            new_weights = new_weights / new_weights.sum()  # Renormalize
            
            # Calculate new VaR
            new_var = self.calculate_portfolio_var(returns, new_weights, confidence_level)
            
            # Marginal VaR
            marginal_vars[asset] = (new_var - base_var) / delta
        
        return marginal_vars
    
    def calculate_component_var(self, returns: pd.DataFrame, weights: np.ndarray,
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate component VaR for each asset.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence_level: Confidence level
            
        Returns:
            Dictionary of component VaRs
        """
        portfolio_returns = returns @ weights
        portfolio_var = self.calculate_var(portfolio_returns, confidence_level)
        
        component_vars = {}
        
        for i, asset in enumerate(returns.columns):
            # Calculate beta of asset to portfolio
            covariance = np.cov(returns[asset], portfolio_returns)[0, 1]
            portfolio_variance = portfolio_returns.var()
            
            if portfolio_variance > 0:
                beta = covariance / portfolio_variance
            else:
                beta = 0
            
            # Component VaR
            component_vars[asset] = weights[i] * beta * portfolio_var
        
        return component_vars
    
    def calculate_incremental_var(self, returns: pd.DataFrame, weights: np.ndarray,
                                position_change: Dict[str, float],
                                confidence_level: float = 0.95) -> float:
        """
        Calculate incremental VaR for a position change.
        
        Args:
            returns: DataFrame of asset returns
            weights: Current portfolio weights
            position_change: Dictionary of position changes
            confidence_level: Confidence level
            
        Returns:
            Incremental VaR
        """
        # Current VaR
        current_var = self.calculate_portfolio_var(returns, weights, confidence_level)
        
        # New weights
        new_weights = weights.copy()
        for i, asset in enumerate(returns.columns):
            if asset in position_change:
                new_weights[i] += position_change[asset]
        
        # Renormalize if needed
        if new_weights.sum() != 1.0:
            new_weights = new_weights / new_weights.sum()
        
        # New VaR
        new_var = self.calculate_portfolio_var(returns, new_weights, confidence_level)
        
        return new_var - current_var
    
    def stress_test_portfolio(self, returns: pd.DataFrame, weights: np.ndarray,
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            scenarios: Dictionary of stress scenarios
            
        Returns:
            Stress test results
        """
        results = {}
        
        # Current portfolio metrics
        portfolio_returns = returns @ weights
        base_metrics = {
            "return": portfolio_returns.mean() * 252,
            "volatility": portfolio_returns.std() * np.sqrt(252),
            "var_95": self.calculate_var(portfolio_returns, 0.95),
            "sharpe": self.calculate_sharpe_ratio(portfolio_returns)
        }
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks to returns
            stressed_returns = returns.copy()
            
            for asset, shock in shocks.items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] + shock
            
            # Calculate stressed portfolio returns
            stressed_portfolio = stressed_returns @ weights
            
            # Calculate stressed metrics
            stressed_metrics = {
                "return": stressed_portfolio.mean() * 252,
                "volatility": stressed_portfolio.std() * np.sqrt(252),
                "var_95": self.calculate_var(stressed_portfolio, 0.95),
                "max_loss": stressed_portfolio.min(),
                "probability_of_loss": (stressed_portfolio < 0).mean()
            }
            
            # Calculate impact
            impact = {
                "return_impact": stressed_metrics["return"] - base_metrics["return"],
                "volatility_impact": stressed_metrics["volatility"] - base_metrics["volatility"],
                "var_impact": stressed_metrics["var_95"] - base_metrics["var_95"]
            }
            
            results[scenario_name] = {
                "stressed_metrics": stressed_metrics,
                "impact": impact,
                "shocks_applied": shocks
            }
        
        return {
            "base_metrics": base_metrics,
            "scenario_results": results
        }
    
    def calculate_risk_contribution(self, returns: pd.DataFrame, 
                                  weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset to portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            
        Returns:
            Dictionary of risk contributions
        """
        # Portfolio volatility
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)
        
        # Marginal contributions to risk
        marginal_contrib = (cov_matrix @ weights) / portfolio_vol
        
        # Risk contributions
        risk_contrib = weights * marginal_contrib
        
        # Convert to percentages
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        
        return dict(zip(returns.columns, risk_contrib_pct))
    
    def calculate_tail_risk_measures(self, returns: pd.Series,
                                   tail_probability: float = 0.05) -> Dict[str, float]:
        """
        Calculate tail risk measures.
        
        Args:
            returns: Series of returns
            tail_probability: Probability threshold for tail
            
        Returns:
            Dictionary of tail risk measures
        """
        # Identify tail returns
        threshold = np.percentile(returns, tail_probability * 100)
        tail_returns = returns[returns <= threshold]
        
        if len(tail_returns) == 0:
            return {"error": "No tail returns found"}
        
        # Tail risk measures
        measures = {
            "tail_mean": tail_returns.mean(),
            "tail_std": tail_returns.std(),
            "tail_skewness": tail_returns.skew(),
            "tail_kurtosis": tail_returns.kurtosis(),
            "expected_tail_loss": -tail_returns.mean(),
            "worst_return": returns.min(),
            "tail_probability": len(tail_returns) / len(returns)
        }
        
        # Extreme value theory - fit GPD to tail
        try:
            from scipy.stats import genpareto
            shape, loc, scale = genpareto.fit(-tail_returns)
            measures["gpd_shape"] = shape
            measures["gpd_scale"] = scale
            measures["tail_index"] = 1 / shape if shape > 0 else np.inf
        except:
            logger.warning("Could not fit GPD to tail returns")
        
        return measures
    
    def calculate_liquidity_risk(self, returns: pd.DataFrame, 
                               volumes: pd.DataFrame,
                               position_sizes: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate liquidity risk metrics.
        
        Args:
            returns: DataFrame of returns
            volumes: DataFrame of trading volumes
            position_sizes: Dictionary of position sizes
            
        Returns:
            Liquidity risk metrics
        """
        liquidity_metrics = {}
        
        for asset in returns.columns:
            if asset not in volumes.columns:
                continue
            
            # Average daily volume
            avg_volume = volumes[asset].mean()
            
            # Position size
            position = position_sizes.get(asset, 0)
            
            # Days to liquidate (assuming can trade 10% of daily volume)
            days_to_liquidate = position / (0.1 * avg_volume) if avg_volume > 0 else np.inf
            
            # Liquidity-adjusted VaR (Bangia et al.)
            spread_volatility = returns[asset].std()
            liquidity_cost = 0.5 * spread_volatility * np.sqrt(days_to_liquidate)
            
            # Amihud illiquidity measure
            amihud = (returns[asset].abs() / volumes[asset]).mean()
            
            liquidity_metrics[asset] = {
                "avg_daily_volume": avg_volume,
                "days_to_liquidate": days_to_liquidate,
                "liquidity_cost": liquidity_cost,
                "amihud_illiquidity": amihud,
                "position_size": position,
                "position_as_pct_of_adv": position / avg_volume if avg_volume > 0 else np.inf
            }
        
        # Portfolio-level metrics
        total_days = max(m["days_to_liquidate"] for m in liquidity_metrics.values())
        total_cost = sum(m["liquidity_cost"] * position_sizes.get(asset, 0) 
                        for asset, m in liquidity_metrics.items())
        
        return {
            "asset_liquidity": liquidity_metrics,
            "portfolio_metrics": {
                "max_days_to_liquidate": total_days,
                "total_liquidity_cost": total_cost,
                "illiquid_assets": [
                    asset for asset, m in liquidity_metrics.items() 
                    if m["days_to_liquidate"] > 5
                ]
            }
        }
    
    def calculate_correlation_risk(self, returns: pd.DataFrame,
                                 rolling_window: int = 60) -> Dict[str, Any]:
        """
        Analyze correlation risk and stability.
        
        Args:
            returns: DataFrame of returns
            rolling_window: Window for rolling correlation
            
        Returns:
            Correlation risk analysis
        """
        # Static correlation
        static_corr = returns.corr()
        
        # Rolling correlation
        rolling_corr_list = []
        for i in range(rolling_window, len(returns)):
            window_returns = returns.iloc[i-rolling_window:i]
            rolling_corr_list.append(window_returns.corr())
        
        # Correlation stability metrics
        corr_changes = []
        for i in range(1, len(rolling_corr_list)):
            change = rolling_corr_list[i] - rolling_corr_list[i-1]
            corr_changes.append(change.abs().mean().mean())
        
        # Identify correlation breaks
        correlation_breaks = []
        threshold = np.std(corr_changes) * 2
        
        for i, change in enumerate(corr_changes):
            if change > threshold:
                correlation_breaks.append({
                    "date_index": i + rolling_window,
                    "magnitude": change
                })
        
        # Calculate correlation risk metrics
        avg_correlation = static_corr.values[np.triu_indices_from(static_corr.values, k=1)].mean()
        max_correlation = static_corr.values[np.triu_indices_from(static_corr.values, k=1)].max()
        
        # Correlation concentration
        eigenvalues = np.linalg.eigvals(static_corr)
        concentration = eigenvalues[0] / eigenvalues.sum()
        
        return {
            "static_correlation": static_corr.to_dict(),
            "average_correlation": avg_correlation,
            "max_correlation": max_correlation,
            "correlation_concentration": concentration,
            "correlation_stability": 1 - np.std(corr_changes),
            "correlation_breaks": correlation_breaks,
            "rolling_window": rolling_window
        }
    
    # Helper methods for different VaR calculations
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR."""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean = returns.mean()
        std = returns.std()
        return mean + std * stats.norm.ppf(1 - confidence_level)
    
    def _cornish_fisher_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Cornish-Fisher VaR adjusting for skewness and kurtosis."""
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        cf_z = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        return mean + std * cf_z
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float,
                        n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR."""
        mean = returns.mean()
        std = returns.std()
        
        # Generate simulations
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        return np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    # Risk-adjusted return metrics
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                              downside_deviation: float) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation > 0 else np.inf
    
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) > 0 and losses.sum() > 0:
            return gains.sum() / losses.sum()
        return np.inf
    
    def calculate_downside_deviation(self, returns: pd.Series, 
                                   threshold: float = 0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < threshold] - threshold
        if len(downside_returns) > 0:
            return np.sqrt(np.mean(downside_returns**2))
        return 0.0
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta."""
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance > 0 else 0
    
    def calculate_alpha(self, returns: pd.Series, market_returns: pd.Series, 
                       beta: float) -> float:
        """Calculate alpha (Jensen's alpha)."""
        return (returns.mean() - beta * market_returns.mean()) * 252
    
    def calculate_tracking_error(self, returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """Calculate tracking error."""
        active_returns = returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)
    
    def calculate_information_ratio(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series,
                                  tracking_error: float) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark_returns
        return active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    def calculate_treynor_ratio(self, returns: pd.Series, beta: float) -> float:
        """Calculate Treynor ratio."""
        excess_returns = returns.mean() - self.risk_free_rate / 252
        return excess_returns * 252 / beta if beta > 0 else 0


def main():
    """Example usage of the risk calculator."""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate returns
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0008, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.0007, 0.025, len(dates)),
        'MSFT': np.random.normal(0.0009, 0.018, len(dates)),
        'AMZN': np.random.normal(0.0006, 0.03, len(dates))
    }, index=dates)
    
    # Simulate market returns for beta calculation
    market_returns = pd.DataFrame({
        'SPY': np.random.normal(0.0007, 0.015, len(dates))
    }, index=dates)
    
    # Initialize calculator
    calculator = RiskCalculator()
    
    # Calculate individual asset risk metrics
    print("Calculating risk metrics...")
    risk_metrics = calculator.calculate_risk_metrics(returns, market_returns)
    
    # Display results for one asset
    print(f"\nRisk Metrics for AAPL:")
    aapl_metrics = risk_metrics['AAPL']
    for metric, value in aapl_metrics.to_dict().items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
    # Portfolio risk analysis
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    portfolio_var = calculator.calculate_portfolio_var(returns, weights)
    print(f"\nPortfolio VaR (95%): {portfolio_var:.4f}")
    
    # Risk contributions
    risk_contrib = calculator.calculate_risk_contribution(returns, weights)
    print("\nRisk Contributions:")
    for asset, contrib in risk_contrib.items():
        print(f"  {asset}: {contrib:.2%}")
    
    # Stress testing
    stress_scenarios = {
        "Market Crash": {"AAPL": -0.20, "GOOGL": -0.25, "MSFT": -0.18, "AMZN": -0.30},
        "Tech Selloff": {"AAPL": -0.15, "GOOGL": -0.20, "MSFT": -0.15, "AMZN": -0.25}
    }
    
    stress_results = calculator.stress_test_portfolio(returns, weights, stress_scenarios)
    print("\nStress Test Results:")
    for scenario, results in stress_results["scenario_results"].items():
        print(f"  {scenario}:")
        print(f"    Return Impact: {results['impact']['return_impact']:.2%}")
        print(f"    VaR Impact: {results['impact']['var_impact']:.4f}")


if __name__ == "__main__":
    main()
