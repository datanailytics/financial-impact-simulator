"""
Scenario Generator Module - Advanced Financial Scenario Analysis
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides sophisticated scenario generation capabilities for
financial analysis, including stress testing, what-if analysis, and
Monte Carlo scenario generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import warnings
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios that can be generated."""
    HISTORICAL = "historical_scenario"
    STATISTICAL = "statistical_scenario"
    STRESS_TEST = "stress_test_scenario"
    MONTE_CARLO = "monte_carlo_scenario"
    FACTOR_BASED = "factor_based_scenario"
    REGIME_SWITCHING = "regime_switching_scenario"
    BOOTSTRAP = "bootstrap_scenario"
    CUSTOM = "custom_scenario"


class DistributionType(Enum):
    """Statistical distributions for scenario generation."""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    SKEW_NORMAL = "skew_normal"
    GARCH = "garch"
    JUMP_DIFFUSION = "jump_diffusion"
    LEVY = "levy"
    EMPIRICAL = "empirical"


@dataclass
class Scenario:
    """Represents a single scenario with its parameters and outcomes."""
    scenario_id: str
    scenario_type: ScenarioType
    description: str
    probability: float
    parameters: Dict[str, Any]
    outcomes: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not 0 <= self.probability <= 1:
            raise ValueError("Scenario probability must be between 0 and 1")


@dataclass
class ScenarioSet:
    """Collection of related scenarios."""
    set_id: str
    name: str
    scenarios: List[Scenario]
    base_case: Optional[Scenario] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.validate_probabilities()
    
    def validate_probabilities(self):
        """Ensure scenario probabilities are valid."""
        total_prob = sum(s.probability for s in self.scenarios)
        if abs(total_prob - 1.0) > 0.01:
            logger.warning(f"Scenario probabilities sum to {total_prob}, normalizing...")
            for scenario in self.scenarios:
                scenario.probability /= total_prob


class ScenarioGenerator:
    """Main class for generating financial scenarios."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize scenario generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.generated_scenarios = []
        
    def generate_scenarios(self, data: pd.DataFrame, 
                         scenario_type: ScenarioType,
                         n_scenarios: int = 100,
                         **kwargs) -> ScenarioSet:
        """
        Generate scenarios based on specified type.
        
        Args:
            data: Historical data for scenario generation
            scenario_type: Type of scenarios to generate
            n_scenarios: Number of scenarios to generate
            **kwargs: Additional parameters for specific scenario types
            
        Returns:
            ScenarioSet containing generated scenarios
        """
        generator_map = {
            ScenarioType.HISTORICAL: self._generate_historical_scenarios,
            ScenarioType.STATISTICAL: self._generate_statistical_scenarios,
            ScenarioType.STRESS_TEST: self._generate_stress_test_scenarios,
            ScenarioType.MONTE_CARLO: self._generate_monte_carlo_scenarios,
            ScenarioType.FACTOR_BASED: self._generate_factor_based_scenarios,
            ScenarioType.REGIME_SWITCHING: self._generate_regime_switching_scenarios,
            ScenarioType.BOOTSTRAP: self._generate_bootstrap_scenarios
        }
        
        generator_func = generator_map.get(scenario_type)
        if not generator_func:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        return generator_func(data, n_scenarios, **kwargs)
    
    def _generate_historical_scenarios(self, data: pd.DataFrame, 
                                     n_scenarios: int,
                                     lookback_window: int = 252,
                                     crisis_periods: Optional[List[Tuple[str, str]]] = None) -> ScenarioSet:
        """
        Generate scenarios based on historical periods.
        
        Args:
            data: Historical price/return data
            n_scenarios: Number of scenarios to generate
            lookback_window: Days to look back for each scenario
            crisis_periods: List of (start_date, end_date) tuples for crisis periods
            
        Returns:
            ScenarioSet with historical scenarios
        """
        scenarios = []
        
        # Define historical crisis periods if not provided
        if crisis_periods is None:
            crisis_periods = [
                ("2007-07-01", "2009-03-31"),  # Financial Crisis
                ("2020-02-01", "2020-04-30"),  # COVID-19 Crash
                ("2000-03-01", "2002-10-31"),  # Dot-com Bubble
                ("1997-07-01", "1998-10-31"),  # Asian Financial Crisis
                ("2011-08-01", "2011-10-31"),  # European Debt Crisis
            ]
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Generate scenarios from crisis periods
        for i, (start_date, end_date) in enumerate(crisis_periods[:n_scenarios//2]):
            try:
                crisis_returns = returns.loc[start_date:end_date]
                if len(crisis_returns) > 0:
                    scenario = Scenario(
                        scenario_id=f"hist_crisis_{i}",
                        scenario_type=ScenarioType.HISTORICAL,
                        description=f"Crisis period: {start_date} to {end_date}",
                        probability=1.0 / n_scenarios,
                        parameters={
                            "start_date": start_date,
                            "end_date": end_date,
                            "period_type": "crisis"
                        },
                        outcomes={
                            "returns": crisis_returns.to_dict(),
                            "cumulative_return": (1 + crisis_returns).prod() - 1,
                            "volatility": crisis_returns.std(),
                            "max_drawdown": self._calculate_max_drawdown(crisis_returns)
                        }
                    )
                    scenarios.append(scenario)
            except KeyError:
                logger.warning(f"Crisis period {start_date} to {end_date} not found in data")
        
        # Generate random historical scenarios
        valid_indices = returns.index[lookback_window:]
        n_random = n_scenarios - len(scenarios)
        
        for i in range(n_random):
            random_end = np.random.choice(valid_indices)
            random_start = random_end - timedelta(days=lookback_window)
            
            period_returns = returns.loc[random_start:random_end]
            
            scenario = Scenario(
                scenario_id=f"hist_random_{i}",
                scenario_type=ScenarioType.HISTORICAL,
                description=f"Random historical period ending {random_end}",
                probability=1.0 / n_scenarios,
                parameters={
                    "start_date": str(random_start),
                    "end_date": str(random_end),
                    "period_type": "random"
                },
                outcomes={
                    "returns": period_returns.mean().to_dict(),
                    "cumulative_return": (1 + period_returns).prod().mean() - 1,
                    "volatility": period_returns.std().mean(),
                    "correlation": period_returns.corr().to_dict()
                }
            )
            scenarios.append(scenario)
        
        # Create base case (recent history)
        base_returns = returns.tail(lookback_window)
        base_case = Scenario(
            scenario_id="base_case",
            scenario_type=ScenarioType.HISTORICAL,
            description="Base case - recent history",
            probability=0.0,  # Base case probability handled separately
            parameters={
                "lookback_days": lookback_window
            },
            outcomes={
                "returns": base_returns.mean().to_dict(),
                "volatility": base_returns.std().to_dict()
            }
        )
        
        return ScenarioSet(
            set_id=f"historical_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Historical Scenarios",
            scenarios=scenarios,
            base_case=base_case,
            metadata={
                "generation_method": "historical",
                "n_scenarios": len(scenarios),
                "lookback_window": lookback_window
            }
        )
    
    def _generate_statistical_scenarios(self, data: pd.DataFrame,
                                      n_scenarios: int,
                                      distribution: DistributionType = DistributionType.NORMAL,
                                      time_horizon: int = 252) -> ScenarioSet:
        """
        Generate scenarios using statistical distributions.
        
        Args:
            data: Historical data
            n_scenarios: Number of scenarios
            distribution: Type of distribution to use
            time_horizon: Time horizon for scenarios
            
        Returns:
            ScenarioSet with statistical scenarios
        """
        returns = data.pct_change().dropna()
        scenarios = []
        
        # Fit distribution parameters
        dist_params = self._fit_distribution(returns, distribution)
        
        # Generate scenarios
        for i in range(n_scenarios):
            # Generate returns based on distribution
            simulated_returns = self._sample_from_distribution(
                distribution, dist_params, 
                size=(time_horizon, len(returns.columns))
            )
            
            # Create scenario
            scenario_returns = pd.DataFrame(
                simulated_returns,
                columns=returns.columns
            )
            
            scenario = Scenario(
                scenario_id=f"stat_{distribution.value}_{i}",
                scenario_type=ScenarioType.STATISTICAL,
                description=f"{distribution.value} distribution scenario {i}",
                probability=1.0 / n_scenarios,
                parameters={
                    "distribution": distribution.value,
                    "distribution_params": dist_params,
                    "time_horizon": time_horizon
                },
                outcomes={
                    "expected_returns": scenario_returns.mean().to_dict(),
                    "volatility": scenario_returns.std().to_dict(),
                    "cumulative_return": (1 + scenario_returns).prod() - 1,
                    "percentiles": {
                        "5%": scenario_returns.quantile(0.05).to_dict(),
                        "95%": scenario_returns.quantile(0.95).to_dict()
                    }
                }
            )
            scenarios.append(scenario)
        
        return ScenarioSet(
            set_id=f"statistical_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"Statistical Scenarios ({distribution.value})",
            scenarios=scenarios,
            metadata={
                "generation_method": "statistical",
                "distribution": distribution.value,
                "time_horizon": time_horizon
            }
        )
    
    def _generate_stress_test_scenarios(self, data: pd.DataFrame,
                                      n_scenarios: int,
                                      stress_factors: Optional[Dict[str, float]] = None) -> ScenarioSet:
        """
        Generate stress test scenarios.
        
        Args:
            data: Historical data
            n_scenarios: Number of scenarios
            stress_factors: Custom stress factors
            
        Returns:
            ScenarioSet with stress test scenarios
        """
        returns = data.pct_change().dropna()
        scenarios = []
        
        # Default stress test scenarios
        if stress_factors is None:
            stress_tests = [
                {"name": "Market Crash", "factor": -0.30, "volatility_mult": 3.0},
                {"name": "Interest Rate Shock", "factor": -0.15, "volatility_mult": 2.0},
                {"name": "Currency Crisis", "factor": -0.20, "volatility_mult": 2.5},
                {"name": "Liquidity Crisis", "factor": -0.25, "volatility_mult": 4.0},
                {"name": "Sector Rotation", "factor": 0.0, "volatility_mult": 1.5},
                {"name": "Inflation Shock", "factor": -0.10, "volatility_mult": 2.0},
                {"name": "Geopolitical Crisis", "factor": -0.35, "volatility_mult": 3.5},
                {"name": "Tech Bubble Burst", "factor": -0.40, "volatility_mult": 3.0},
            ]
        else:
            stress_tests = [{"name": k, "factor": v, "volatility_mult": 2.0} 
                          for k, v in stress_factors.items()]
        
        # Generate stress scenarios
        for i, stress_test in enumerate(stress_tests[:n_scenarios]):
            # Apply stress factor
            stressed_returns = returns.copy()
            stressed_returns = stressed_returns + stress_test["factor"] / 252
            
            # Increase volatility
            vol_mult = stress_test["volatility_mult"]
            noise = np.random.normal(0, returns.std() * (vol_mult - 1), size=returns.shape)
            stressed_returns += pd.DataFrame(noise, index=returns.index, columns=returns.columns)
            
            # Calculate stressed outcomes
            scenario = Scenario(
                scenario_id=f"stress_{i}_{stress_test['name'].replace(' ', '_')}",
                scenario_type=ScenarioType.STRESS_TEST,
                description=f"Stress Test: {stress_test['name']}",
                probability=1.0 / n_scenarios,  # Equal probability for now
                parameters={
                    "stress_factor": stress_test["factor"],
                    "volatility_multiplier": vol_mult,
                    "stress_type": stress_test["name"]
                },
                outcomes={
                    "immediate_impact": stress_test["factor"],
                    "volatility_impact": (returns.std() * vol_mult).to_dict(),
                    "expected_loss": (stressed_returns.mean() * 252).to_dict(),
                    "max_loss": stressed_returns.min().to_dict(),
                    "recovery_time_estimate": self._estimate_recovery_time(stress_test["factor"])
                }
            )
            scenarios.append(scenario)
        
        return ScenarioSet(
            set_id=f"stress_test_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Stress Test Scenarios",
            scenarios=scenarios,
            metadata={
                "generation_method": "stress_test",
                "stress_factors": stress_factors
            }
        )
    
    def _generate_monte_carlo_scenarios(self, data: pd.DataFrame,
                                      n_scenarios: int,
                                      time_horizon: int = 252,
                                      use_correlation: bool = True) -> ScenarioSet:
        """
        Generate Monte Carlo scenarios.
        
        Args:
            data: Historical data
            n_scenarios: Number of scenarios
            time_horizon: Time horizon for simulation
            use_correlation: Whether to preserve correlations
            
        Returns:
            ScenarioSet with Monte Carlo scenarios
        """
        returns = data.pct_change().dropna()
        scenarios = []
        
        # Calculate parameters
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Use Ledoit-Wolf shrinkage for more stable covariance
        if use_correlation:
            lw = LedoitWolf()
            cov_matrix = pd.DataFrame(
                lw.fit(returns).covariance_,
                index=returns.columns,
                columns=returns.columns
            )
        
        # Generate scenarios
        for i in range(n_scenarios):
            if use_correlation:
                # Multivariate normal with correlation
                simulated_returns = np.random.multivariate_normal(
                    mean_returns, cov_matrix, size=time_horizon
                )
            else:
                # Independent normal
                simulated_returns = np.random.normal(
                    mean_returns, returns.std(), 
                    size=(time_horizon, len(returns.columns))
                )
            
            # Calculate scenario outcomes
            scenario_df = pd.DataFrame(simulated_returns, columns=returns.columns)
            cumulative_returns = (1 + scenario_df).prod() - 1
            
            scenario = Scenario(
                scenario_id=f"mc_{i}",
                scenario_type=ScenarioType.MONTE_CARLO,
                description=f"Monte Carlo scenario {i}",
                probability=1.0 / n_scenarios,
                parameters={
                    "time_horizon": time_horizon,
                    "use_correlation": use_correlation,
                    "random_seed": self.random_seed + i
                },
                outcomes={
                    "terminal_values": cumulative_returns.to_dict(),
                    "expected_return": scenario_df.mean().to_dict(),
                    "volatility": scenario_df.std().to_dict(),
                    "path_statistics": {
                        "max_return": scenario_df.max().to_dict(),
                        "min_return": scenario_df.min().to_dict(),
                        "skewness": scenario_df.skew().to_dict(),
                        "kurtosis": scenario_df.kurtosis().to_dict()
                    }
                }
            )
            scenarios.append(scenario)
        
        return ScenarioSet(
            set_id=f"monte_carlo_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Monte Carlo Scenarios",
            scenarios=scenarios,
            metadata={
                "generation_method": "monte_carlo",
                "time_horizon": time_horizon,
                "use_correlation": use_correlation
            }
        )
    
    def _generate_factor_based_scenarios(self, data: pd.DataFrame,
                                       n_scenarios: int,
                                       n_factors: int = 3) -> ScenarioSet:
        """
        Generate factor-based scenarios using PCA.
        
        Args:
            data: Historical data
            n_scenarios: Number of scenarios
            n_factors: Number of principal factors to use
            
        Returns:
            ScenarioSet with factor-based scenarios
        """
        returns = data.pct_change().dropna()
        scenarios = []
        
        # Perform PCA
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(returns)
        loadings = pca.components_
        explained_variance = pca.explained_variance_ratio_
        
        # Generate factor scenarios
        factor_shocks = [
            {"name": "Factor 1 Positive Shock", "shocks": [2, 0, 0]},
            {"name": "Factor 1 Negative Shock", "shocks": [-2, 0, 0]},
            {"name": "Factor 2 Positive Shock", "shocks": [0, 2, 0]},
            {"name": "Factor 2 Negative Shock", "shocks": [0, -2, 0]},
            {"name": "Combined Positive", "shocks": [1.5, 1.5, 1]},
            {"name": "Combined Negative", "shocks": [-1.5, -1.5, -1]},
            {"name": "Rotation", "shocks": [2, -2, 0]},
        ]
        
        for i, shock_config in enumerate(factor_shocks[:n_scenarios]):
            # Apply factor shocks
            factor_shock = np.array(shock_config["shocks"][:n_factors])
            
            # Transform back to asset space
            asset_shock = factor_shock @ loadings
            
            # Generate scenario returns
            shocked_returns = returns.mean() + asset_shock * returns.std()
            
            scenario = Scenario(
                scenario_id=f"factor_{i}_{shock_config['name'].replace(' ', '_')}",
                scenario_type=ScenarioType.FACTOR_BASED,
                description=f"Factor scenario: {shock_config['name']}",
                probability=1.0 / n_scenarios,
                parameters={
                    "n_factors": n_factors,
                    "factor_shocks": shock_config["shocks"],
                    "explained_variance": explained_variance.tolist()
                },
                outcomes={
                    "asset_impacts": dict(zip(returns.columns, asset_shock)),
                    "expected_returns": shocked_returns.to_dict(),
                    "factor_loadings": {
                        f"factor_{j}": dict(zip(returns.columns, loadings[j]))
                        for j in range(n_factors)
                    }
                }
            )
            scenarios.append(scenario)
        
        return ScenarioSet(
            set_id=f"factor_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Factor-Based Scenarios",
            scenarios=scenarios,
            metadata={
                "generation_method": "factor_based",
                "n_factors": n_factors,
                "explained_variance": explained_variance.tolist()
            }
        )
    
    def _generate_regime_switching_scenarios(self, data: pd.DataFrame,
                                           n_scenarios: int,
                                           n_regimes: int = 3) -> ScenarioSet:
        """
        Generate regime-switching scenarios.
        
        Args:
            data: Historical data
            n_scenarios: Number of scenarios
            n_regimes: Number of market regimes
            
        Returns:
            ScenarioSet with regime-switching scenarios
        """
        returns = data.pct_change().dropna()
        scenarios = []
        
        # Define market regimes
        regimes = {
            "bull_market": {
                "mean_mult": 1.5,
                "vol_mult": 0.8,
                "description": "Bull market - high returns, low volatility"
            },
            "bear_market": {
                "mean_mult": -1.5,
                "vol_mult": 1.5,
                "description": "Bear market - negative returns, high volatility"
            },
            "sideways_market": {
                "mean_mult": 0.2,
                "vol_mult": 1.0,
                "description": "Sideways market - low returns, normal volatility"
            },
            "volatile_growth": {
                "mean_mult": 1.2,
                "vol_mult": 2.0,
                "description": "Volatile growth - positive returns, high volatility"
            },
            "crisis": {
                "mean_mult": -3.0,
                "vol_mult": 3.0,
                "description": "Crisis - extreme negative returns, extreme volatility"
            }
        }
        
        # Generate transition matrix
        transition_matrix = self._generate_transition_matrix(len(regimes))
        
        regime_names = list(regimes.keys())
        
        for i in range(n_scenarios):
            # Sample regime sequence
            regime_sequence = self._sample_regime_sequence(
                regime_names, transition_matrix, 252
            )
            
            # Calculate scenario outcomes
            scenario_returns = []
            for regime in regime_sequence:
                regime_config = regimes[regime]
                daily_return = (returns.mean() * regime_config["mean_mult"] + 
                              np.random.normal(0, returns.std() * regime_config["vol_mult"]))
                scenario_returns.append(daily_return)
            
            scenario_returns = pd.DataFrame(scenario_returns)
            
            # Count regime occurrences
            regime_counts = pd.Series(regime_sequence).value_counts()
            
            scenario = Scenario(
                scenario_id=f"regime_switch_{i}",
                scenario_type=ScenarioType.REGIME_SWITCHING,
                description=f"Regime-switching scenario {i}",
                probability=1.0 / n_scenarios,
                parameters={
                    "n_regimes": n_regimes,
                    "transition_matrix": transition_matrix.tolist(),
                    "regime_sequence_summary": regime_counts.to_dict()
                },
                outcomes={
                    "expected_return": scenario_returns.mean().to_dict(),
                    "volatility": scenario_returns.std().to_dict(),
                    "regime_durations": self._calculate_regime_durations(regime_sequence),
                    "dominant_regime": regime_counts.idxmax()
                }
            )
            scenarios.append(scenario)
        
        return ScenarioSet(
            set_id=f"regime_switching_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Regime-Switching Scenarios",
            scenarios=scenarios,
            metadata={
                "generation_method": "regime_switching",
                "regimes": list(regimes.keys()),
                "regime_descriptions": {k: v["description"] for k, v in regimes.items()}
            }
        )
    
    def _generate_bootstrap_scenarios(self, data: pd.DataFrame,
                                    n_scenarios: int,
                                    block_size: int = 20) -> ScenarioSet:
        """
        Generate bootstrap scenarios.
        
        Args:
            data: Historical data
            n_scenarios: Number of scenarios
            block_size: Size of blocks for block bootstrap
            
        Returns:
            ScenarioSet with bootstrap scenarios
        """
        returns = data.pct_change().dropna()
        scenarios = []
        
        n_obs = len(returns)
        n_blocks = n_obs // block_size
        
        for i in range(n_scenarios):
            # Block bootstrap
            bootstrapped_returns = []
            
            for _ in range(n_blocks):
                # Sample a random block
                start_idx = np.random.randint(0, n_obs - block_size)
                block = returns.iloc[start_idx:start_idx + block_size]
                bootstrapped_returns.append(block)
            
            # Concatenate blocks
            bootstrap_sample = pd.concat(bootstrapped_returns, ignore_index=True)
            
            # Calculate statistics
            scenario = Scenario(
                scenario_id=f"bootstrap_{i}",
                scenario_type=ScenarioType.BOOTSTRAP,
                description=f"Bootstrap scenario {i}",
                probability=1.0 / n_scenarios,
                parameters={
                    "block_size": block_size,
                    "n_blocks": n_blocks,
                    "sample_size": len(bootstrap_sample)
                },
                outcomes={
                    "expected_return": bootstrap_sample.mean().to_dict(),
                    "volatility": bootstrap_sample.std().to_dict(),
                    "skewness": bootstrap_sample.skew().to_dict(),
                    "kurtosis": bootstrap_sample.kurtosis().to_dict(),
                    "confidence_intervals": {
                        "lower_95": bootstrap_sample.quantile(0.025).to_dict(),
                        "upper_95": bootstrap_sample.quantile(0.975).to_dict()
                    }
                }
            )
            scenarios.append(scenario)
        
        return ScenarioSet(
            set_id=f"bootstrap_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Bootstrap Scenarios",
            scenarios=scenarios,
            metadata={
                "generation_method": "bootstrap",
                "block_size": block_size,
                "original_data_size": n_obs
            }
        )
    
    def combine_scenario_sets(self, scenario_sets: List[ScenarioSet],
                            weights: Optional[List[float]] = None) -> ScenarioSet:
        """
        Combine multiple scenario sets with optional weighting.
        
        Args:
            scenario_sets: List of scenario sets to combine
            weights: Optional weights for each set
            
        Returns:
            Combined ScenarioSet
        """
        if weights is None:
            weights = [1.0 / len(scenario_sets)] * len(scenario_sets)
        
        all_scenarios = []
        
        for scenario_set, weight in zip(scenario_sets, weights):
            for scenario in scenario_set.scenarios:
                # Adjust probability based on weight
                scenario.probability *= weight
                all_scenarios.append(scenario)
        
        combined_set = ScenarioSet(
            set_id=f"combined_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Combined Scenarios",
            scenarios=all_scenarios,
            metadata={
                "source_sets": [s.set_id for s in scenario_sets],
                "weights": weights,
                "n_total_scenarios": len(all_scenarios)
            }
        )
        
        return combined_set
    
    def analyze_scenario_impacts(self, scenario_set: ScenarioSet,
                               portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze the impact of scenarios on a portfolio.
        
        Args:
            scenario_set: Set of scenarios to analyze
            portfolio_weights: Portfolio weights
            
        Returns:
            Dictionary with impact analysis
        """
        impacts = []
        
        for scenario in scenario_set.scenarios:
            if "expected_returns" in scenario.outcomes:
                # Calculate portfolio impact
                portfolio_return = sum(
                    portfolio_weights.get(asset, 0) * scenario.outcomes["expected_returns"].get(asset, 0)
                    for asset in portfolio_weights
                )
                
                impact = {
                    "scenario_id": scenario.scenario_id,
                    "description": scenario.description,
                    "probability": scenario.probability,
                    "portfolio_return": portfolio_return,
                    "weighted_impact": portfolio_return * scenario.probability
                }
                impacts.append(impact)
        
        # Calculate summary statistics
        portfolio_returns = [i["portfolio_return"] for i in impacts]
        probabilities = [i["probability"] for i in impacts]
        
        expected_return = sum(r * p for r, p in zip(portfolio_returns, probabilities))
        
        # Calculate VaR and CVaR
        sorted_impacts = sorted(impacts, key=lambda x: x["portfolio_return"])
        cumulative_prob = 0
        var_95 = None
        cvar_returns = []
        
        for impact in sorted_impacts:
            cumulative_prob += impact["probability"]
            if cumulative_prob >= 0.05 and var_95 is None:
                var_95 = impact["portfolio_return"]
            if cumulative_prob <= 0.05:
                cvar_returns.append(impact["portfolio_return"])
        
        cvar_95 = np.mean(cvar_returns) if cvar_returns else var_95
        
        return {
            "scenario_impacts": impacts,
            "summary": {
                "expected_return": expected_return,
                "best_scenario": max(impacts, key=lambda x: x["portfolio_return"]),
                "worst_scenario": min(impacts, key=lambda x: x["portfolio_return"]),
                "var_95": var_95,
                "cvar_95": cvar_95,
                "return_distribution": {
                    "mean": np.mean(portfolio_returns),
                    "std": np.std(portfolio_returns),
                    "skewness": stats.skew(portfolio_returns),
                    "kurtosis": stats.kurtosis(portfolio_returns)
                }
            }
        }
    
    # Helper methods
    def _fit_distribution(self, returns: pd.DataFrame, 
                         distribution: DistributionType) -> Dict[str, Any]:
        """Fit distribution parameters to returns data."""
        params = {}
        
        if distribution == DistributionType.NORMAL:
            params["mean"] = returns.mean().to_dict()
            params["std"] = returns.std().to_dict()
        
        elif distribution == DistributionType.STUDENT_T:
            for col in returns.columns:
                fit_params = stats.t.fit(returns[col])
                params[col] = {"df": fit_params[0], "loc": fit_params[1], "scale": fit_params[2]}
        
        elif distribution == DistributionType.SKEW_NORMAL:
            for col in returns.columns:
                fit_params = stats.skewnorm.fit(returns[col])
                params[col] = {"a": fit_params[0], "loc": fit_params[1], "scale": fit_params[2]}
        
        return params
    
    def _sample_from_distribution(self, distribution: DistributionType,
                                params: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        """Sample from specified distribution."""
        samples = np.zeros(size)
        
        if distribution == DistributionType.NORMAL:
            for i, (asset, asset_params) in enumerate(params["mean"].items()):
                samples[:, i] = np.random.normal(
                    asset_params, params["std"][asset], size[0]
                )
        
        elif distribution == DistributionType.STUDENT_T:
            for i, (asset, asset_params) in enumerate(params.items()):
                samples[:, i] = stats.t.rvs(
                    asset_params["df"], 
                    loc=asset_params["loc"],
                    scale=asset_params["scale"],
                    size=size[0]
                )
        
        return samples
    
    def _calculate_max_drawdown(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate maximum drawdown for each asset."""
        drawdowns = {}
        
        for col in returns.columns:
            cumulative = (1 + returns[col]).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            drawdowns[col] = float(drawdown.min())
        
        return drawdowns
    
    def _estimate_recovery_time(self, stress_factor: float) -> int:
        """Estimate recovery time based on stress factor."""
        # Simple heuristic: deeper stress takes longer to recover
        base_recovery = 60  # days
        recovery_time = int(base_recovery * (1 + abs(stress_factor) * 2))
        return recovery_time
    
    def _generate_transition_matrix(self, n_states: int) -> np.ndarray:
        """Generate a random transition matrix for regime switching."""
        # Create a diagonally dominant matrix (regimes tend to persist)
        matrix = np.random.dirichlet(np.ones(n_states), size=n_states)
        
        # Increase diagonal elements
        for i in range(n_states):
            matrix[i, i] += 0.5
            matrix[i, :] /= matrix[i, :].sum()
        
        return matrix
    
    def _sample_regime_sequence(self, regimes: List[str], 
                              transition_matrix: np.ndarray, 
                              length: int) -> List[str]:
        """Sample a sequence of regimes based on transition matrix."""
        sequence = []
        current_regime = np.random.choice(len(regimes))
        
        for _ in range(length):
            sequence.append(regimes[current_regime])
            # Sample next regime
            current_regime = np.random.choice(
                len(regimes), 
                p=transition_matrix[current_regime]
            )
        
        return sequence
    
    def _calculate_regime_durations(self, regime_sequence: List[str]) -> Dict[str, float]:
        """Calculate average duration for each regime."""
        durations = {}
        current_regime = regime_sequence[0]
        current_duration = 1
        regime_durations = {regime: [] for regime in set(regime_sequence)}
        
        for regime in regime_sequence[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                regime_durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1
        
        # Add last duration
        regime_durations[current_regime].append(current_duration)
        
        # Calculate averages
        for regime, duration_list in regime_durations.items():
            if duration_list:
                durations[regime] = np.mean(duration_list)
            else:
                durations[regime] = 0
        
        return durations


def main():
    """Example usage of the scenario generator."""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate asset prices
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    data = pd.DataFrame(
        {
            asset: 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
            for asset in assets
        },
        index=dates
    )
    
    # Initialize generator
    generator = ScenarioGenerator()
    
    # Generate different types of scenarios
    print("Generating Historical Scenarios...")
    historical_scenarios = generator.generate_scenarios(
        data, ScenarioType.HISTORICAL, n_scenarios=10
    )
    
    print("Generating Stress Test Scenarios...")
    stress_scenarios = generator.generate_scenarios(
        data, ScenarioType.STRESS_TEST, n_scenarios=8
    )
    
    print("Generating Monte Carlo Scenarios...")
    mc_scenarios = generator.generate_scenarios(
        data, ScenarioType.MONTE_CARLO, n_scenarios=100
    )
    
    # Combine scenarios
    combined = generator.combine_scenario_sets(
        [historical_scenarios, stress_scenarios],
        weights=[0.6, 0.4]
    )
    
    # Analyze impact on a portfolio
    portfolio_weights = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.2, 'AMZN': 0.2}
    impact_analysis = generator.analyze_scenario_impacts(combined, portfolio_weights)
    
    print(f"\nScenario Analysis Complete")
    print(f"Expected Portfolio Return: {impact_analysis['summary']['expected_return']:.2%}")
    print(f"95% VaR: {impact_analysis['summary']['var_95']:.2%}")
    print(f"95% CVaR: {impact_analysis['summary']['cvar_95']:.2%}")


if __name__ == "__main__":
    main()
