"""
Charts Module - Advanced Financial Data Visualization
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides comprehensive charting capabilities for financial
data visualization using multiple plotting libraries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
from enum import Enum
import logging

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pio.templates.default = "plotly_dark"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Available chart types."""
    LINE = "line"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    AREA = "area"
    WATERFALL = "waterfall"
    FUNNEL = "funnel"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    GAUGE = "gauge"
    INDICATOR = "indicator"
    TABLE = "table"


class ChartTheme(Enum):
    """Available chart themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    PROFESSIONAL = "professional"
    COLORFUL = "colorful"
    MINIMALIST = "minimalist"


class FinancialChartBuilder:
    """Main class for building financial charts."""
    
    def __init__(self, theme: ChartTheme = ChartTheme.PROFESSIONAL):
        """
        Initialize chart builder.
        
        Args:
            theme: Chart theme to use
        """
        self.theme = theme
        self._apply_theme()
        self.color_palette = self._get_color_palette()
    
    def _apply_theme(self):
        """Apply the selected theme."""
        if self.theme == ChartTheme.DARK:
            plt.style.use('dark_background')
            pio.templates.default = "plotly_dark"
        elif self.theme == ChartTheme.LIGHT:
            plt.style.use('seaborn-v0_8-whitegrid')
            pio.templates.default = "plotly_white"
        elif self.theme == ChartTheme.PROFESSIONAL:
            plt.style.use('seaborn-v0_8-darkgrid')
            pio.templates.default = "seaborn"
        elif self.theme == ChartTheme.MINIMALIST:
            plt.style.use('classic')
            pio.templates.default = "simple_white"
    
    def _get_color_palette(self) -> List[str]:
        """Get color palette based on theme."""
        palettes = {
            ChartTheme.DEFAULT: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            ChartTheme.DARK: ['#00d9ff', '#00ff41', '#ff0080', '#ffaa00', '#9d02d7'],
            ChartTheme.LIGHT: ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099'],
            ChartTheme.PROFESSIONAL: ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#f5f5f5'],
            ChartTheme.COLORFUL: ['#ff006e', '#fb5607', '#ffbe0b', '#8338ec', '#3a86ff'],
            ChartTheme.MINIMALIST: ['#000000', '#666666', '#999999', '#cccccc', '#eeeeee']
        }
        return palettes.get(self.theme, palettes[ChartTheme.DEFAULT])
    
    def create_line_chart(self, data: pd.DataFrame, 
                         title: str = "Line Chart",
                         x_col: Optional[str] = None,
                         y_cols: Optional[List[str]] = None,
                         interactive: bool = True) -> Union[go.Figure, plt.Figure]:
        """
        Create a line chart.
        
        Args:
            data: DataFrame with data
            title: Chart title
            x_col: X-axis column (default: index)
            y_cols: Y-axis columns (default: all numeric)
            interactive: Use Plotly (True) or Matplotlib (False)
            
        Returns:
            Chart figure
        """
        if interactive:
            return self._create_plotly_line(data, title, x_col, y_cols)
        else:
            return self._create_matplotlib_line(data, title, x_col, y_cols)
    
    def _create_plotly_line(self, data: pd.DataFrame, title: str,
                           x_col: Optional[str], y_cols: Optional[List[str]]) -> go.Figure:
        """Create interactive line chart with Plotly."""
        fig = go.Figure()
        
        # Determine x values
        x_values = data[x_col] if x_col else data.index
        
        # Determine y columns
        if y_cols is None:
            y_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add traces
        for i, col in enumerate(y_cols):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=self.color_palette[i % len(self.color_palette)], width=2),
                hovertemplate='%{fullData.name}<br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(title=x_col or 'Date', showgrid=True),
            yaxis=dict(title='Value', showgrid=True),
            hovermode='x unified',
            template=pio.templates.default,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_matplotlib_line(self, data: pd.DataFrame, title: str,
                              x_col: Optional[str], y_cols: Optional[List[str]]) -> plt.Figure:
        """Create static line chart with Matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Determine x values
        x_values = data[x_col] if x_col else data.index
        
        # Determine y columns
        if y_cols is None:
            y_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Plot lines
        for i, col in enumerate(y_cols):
            ax.plot(x_values, data[col], label=col, 
                   color=self.color_palette[i % len(self.color_palette)], linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col or 'Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_candlestick_chart(self, data: pd.DataFrame,
                               title: str = "Candlestick Chart",
                               volume: bool = True) -> go.Figure:
        """
        Create a candlestick chart for OHLC data.
        
        Args:
            data: DataFrame with OHLC columns
            title: Chart title
            volume: Include volume subplot
            
        Returns:
            Plotly figure
        """
        # Validate OHLC data
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError("Data must contain open, high, low, close columns")
        
        # Create subplots
        if volume and 'volume' in data.columns:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.02
            )
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            # Add volume bars
            colors = ['#26a69a' if close >= open else '#ef5350' 
                     for close, open in zip(data['close'], data['open'])]
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ), row=2, col=1)
            
            # Update layout
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            )])
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_rangeslider_visible=False,
            template=pio.templates.default
        )
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame,
                      title: str = "Correlation Heatmap",
                      annotate: bool = True) -> Union[go.Figure, plt.Figure]:
        """
        Create a heatmap.
        
        Args:
            data: DataFrame with data (will compute correlation if not already)
            title: Chart title
            annotate: Show values on heatmap
            
        Returns:
            Chart figure
        """
        # Compute correlation if needed
        if data.shape[0] != data.shape[1]:
            corr_data = data.corr()
        else:
            corr_data = data
        
        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(2) if annotate else None,
            texttemplate='%{text}' if annotate else None,
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=pio.templates.default,
            width=800,
            height=800
        )
        
        return fig
    
    def create_distribution_plot(self, data: pd.Series,
                               title: str = "Distribution Plot",
                               bins: int = 50,
                               kde: bool = True) -> go.Figure:
        """
        Create a distribution plot.
        
        Args:
            data: Series with data
            title: Chart title
            bins: Number of histogram bins
            kde: Include KDE overlay
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name='Histogram',
            histnorm='probability density',
            marker_color=self.color_palette[0],
            opacity=0.7
        ))
        
        # Add KDE if requested
        if kde:
            from scipy import stats
            kde_x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data.dropna())
            kde_y = kde(kde_x)
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color=self.color_palette[1], width=3)
            ))
        
        # Add normal distribution overlay
        mean = data.mean()
        std = data.std()
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mean, std)
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal',
            line=dict(color=self.color_palette[2], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Value',
            yaxis_title='Density',
            template=pio.templates.default,
            showlegend=True
        )
        
        return fig
    
    def create_portfolio_performance_chart(self, returns: pd.DataFrame,
                                         benchmark: Optional[pd.Series] = None,
                                         title: str = "Portfolio Performance") -> go.Figure:
        """
        Create a comprehensive portfolio performance chart.
        
        Args:
            returns: DataFrame with portfolio returns
            benchmark: Optional benchmark returns
            title: Chart title
            
        Returns:
            Plotly figure with multiple subplots
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Cumulative Returns', 'Daily Returns', 'Rolling Volatility'),
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.05
        )
        
        # Plot cumulative returns
        for i, col in enumerate(returns.columns):
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                mode='lines',
                name=col,
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ), row=1, col=1)
        
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            fig.add_trace(go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', dash='dash')
            ), row=1, col=1)
        
        # Plot daily returns
        for i, col in enumerate(returns.columns):
            fig.add_trace(go.Bar(
                x=returns.index,
                y=returns[col],
                name=col,
                marker_color=self.color_palette[i % len(self.color_palette)],
                showlegend=False
            ), row=2, col=1)
        
        # Plot rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        for i, col in enumerate(returns.columns):
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[col],
                mode='lines',
                name=col,
                line=dict(color=self.color_palette[i % len(self.color_palette)]),
                showlegend=False
            ), row=3, col=1)
        
        # Update layout
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return", row=2, col=1)
        fig.update_yaxes(title_text="Ann. Volatility", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=pio.templates.default,
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_risk_return_scatter(self, returns: pd.DataFrame,
                                 title: str = "Risk-Return Profile") -> go.Figure:
        """
        Create risk-return scatter plot.
        
        Args:
            returns: DataFrame with asset returns
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Calculate annualized metrics
        annual_returns = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratios = annual_returns / annual_vol
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add assets
        for i, asset in enumerate(returns.columns):
            fig.add_trace(go.Scatter(
                x=[annual_vol[asset]],
                y=[annual_returns[asset]],
                mode='markers+text',
                name=asset,
                text=[asset],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=self.color_palette[i % len(self.color_palette)],
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    f'<b>{asset}</b><br>' +
                    'Return: %{y:.1%}<br>' +
                    'Risk: %{x:.1%}<br>' +
                    f'Sharpe: {sharpe_ratios[asset]:.2f}' +
                    '<extra></extra>'
                )
            ))
        
        # Add efficient frontier line (placeholder)
        x_range = np.linspace(annual_vol.min() * 0.8, annual_vol.max() * 1.2, 50)
        y_efficient = annual_returns.min() + (x_range - annual_vol.min()) * \
                     (annual_returns.max() - annual_returns.min()) / \
                     (annual_vol.max() - annual_vol.min())
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_efficient,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='gray', dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Risk (Annual Volatility)',
            yaxis_title='Return (Annualized)',
            template=pio.templates.default,
            xaxis=dict(tickformat='.0%'),
            yaxis=dict(tickformat='.0%')
        )
        
        return fig
    
    def create_drawdown_chart(self, returns: pd.Series,
                            title: str = "Drawdown Analysis") -> go.Figure:
        """
        Create drawdown chart.
        
        Args:
            returns: Series with returns
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns and drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add cumulative returns
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color=self.color_palette[0], width=2)
        ), secondary_y=False)
        
        # Add drawdown
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=self.color_palette[1], width=1),
            fillcolor=f'rgba({int(self.color_palette[1][1:3], 16)}, '
                     f'{int(self.color_palette[1][3:5], 16)}, '
                     f'{int(self.color_palette[1][5:7], 16)}, 0.3)'
        ), secondary_y=True)
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Cumulative Return", secondary_y=False)
        fig.update_yaxes(title_text="Drawdown", secondary_y=True, tickformat='.0%')
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=pio.templates.default,
            hovermode='x unified'
        )
        
        return fig
    
    def create_sector_allocation_chart(self, allocations: Dict[str, float],
                                     chart_type: str = 'pie',
                                     title: str = "Sector Allocation") -> go.Figure:
        """
        Create sector allocation chart.
        
        Args:
            allocations: Dictionary of sector allocations
            chart_type: 'pie', 'donut', or 'treemap'
            title: Chart title
            
        Returns:
            Plotly figure
        """
        sectors = list(allocations.keys())
        values = list(allocations.values())
        
        if chart_type == 'pie':
            fig = go.Figure(data=[go.Pie(
                labels=sectors,
                values=values,
                hole=0,
                marker_colors=self.color_palette * (len(sectors) // len(self.color_palette) + 1)
            )])
        elif chart_type == 'donut':
            fig = go.Figure(data=[go.Pie(
                labels=sectors,
                values=values,
                hole=0.4,
                marker_colors=self.color_palette * (len(sectors) // len(self.color_palette) + 1)
            )])
        elif chart_type == 'treemap':
            fig = go.Figure(go.Treemap(
                labels=sectors,
                values=values,
                parents=[""] * len(sectors),
                marker_colorscale='Blues',
                textinfo="label+percent parent"
            ))
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=pio.templates.default
        )
        
        return fig
    
    def create_performance_table(self, metrics: Dict[str, Dict[str, float]],
                               title: str = "Performance Metrics") -> go.Figure:
        """
        Create a performance metrics table.
        
        Args:
            metrics: Dictionary of metrics by asset
            title: Table title
            
        Returns:
            Plotly figure
        """
        # Prepare data for table
        assets = list(metrics.keys())
        metric_names = list(metrics[assets[0]].keys())
        
        # Create header
        header_values = ['Asset'] + metric_names
        
        # Create cells
        cell_values = [assets]
        for metric in metric_names:
            values = []
            for asset in assets:
                value = metrics[asset].get(metric, 0)
                if isinstance(value, float):
                    if 'return' in metric.lower() or 'alpha' in metric.lower():
                        values.append(f"{value:.2%}")
                    elif 'ratio' in metric.lower():
                        values.append(f"{value:.2f}")
                    else:
                        values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            cell_values.append(values)
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color=self.color_palette[0],
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=cell_values,
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=pio.templates.default,
            height=400
        )
        
        return fig
    
    def create_monte_carlo_chart(self, simulations: np.ndarray,
                               title: str = "Monte Carlo Simulation",
                               percentiles: List[float] = [5, 25, 50, 75, 95]) -> go.Figure:
        """
        Create Monte Carlo simulation chart.
        
        Args:
            simulations: Array of simulation paths (shape: n_simulations x n_periods)
            title: Chart title
            percentiles: Percentiles to display
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Calculate percentiles
        percentile_paths = {}
        for p in percentiles:
            percentile_paths[p] = np.percentile(simulations, p, axis=0)
        
        # Plot percentile bands
        time_periods = np.arange(simulations.shape[1])
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=time_periods,
            y=percentile_paths[50],
            mode='lines',
            name='Median',
            line=dict(color=self.color_palette[0], width=3)
        ))
        
        # Add percentile bands
        for i in range(len(percentiles) // 2):
            lower_p = percentiles[i]
            upper_p = percentiles[-(i+1)]
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_periods, time_periods[::-1]]),
                y=np.concatenate([percentile_paths[lower_p], percentile_paths[upper_p][::-1]]),
                fill='toself',
                fillcolor=f'rgba({int(self.color_palette[i+1][1:3], 16)}, '
                         f'{int(self.color_palette[i+1][3:5], 16)}, '
                         f'{int(self.color_palette[i+1][5:7], 16)}, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{lower_p}%-{upper_p}%',
                showlegend=True
            ))
        
        # Add sample paths
        n_sample_paths = min(100, simulations.shape[0])
        sample_indices = np.random.choice(simulations.shape[0], n_sample_paths, replace=False)
        
        for idx in sample_indices:
            fig.add_trace(go.Scatter(
                x=time_periods,
                y=simulations[idx],
                mode='lines',
                line=dict(color='gray', width=0.5),
                opacity=0.1,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title='Time Period',
            yaxis_title='Value',
            template=pio.templates.default,
            hovermode='x unified'
        )
        
        return fig
    
    def save_chart(self, fig: Union[go.Figure, plt.Figure],
                  filename: str, format: str = 'png',
                  width: int = 1200, height: int = 600):
        """
        Save chart to file.
        
        Args:
            fig: Chart figure
            filename: Output filename
            format: Output format ('png', 'jpg', 'svg', 'pdf', 'html')
            width: Image width
            height: Image height
        """
        if isinstance(fig, go.Figure):
            if format == 'html':
                fig.write_html(filename)
            else:
                fig.write_image(filename, format=format, width=width, height=height)
        elif isinstance(fig, plt.Figure):
            fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
        else:
            raise TypeError("Figure must be Plotly or Matplotlib figure")
        
        logger.info(f"Chart saved to {filename}")


class ChartFactory:
    """Factory for creating charts based on data type."""
    
    @staticmethod
    def create_chart(data: pd.DataFrame, chart_type: ChartType,
                    **kwargs) -> Union[go.Figure, plt.Figure]:
        """
        Create appropriate chart based on data and type.
        
        Args:
            data: Input data
            chart_type: Type of chart to create
            **kwargs: Additional chart parameters
            
        Returns:
            Chart figure
        """
        builder = FinancialChartBuilder(
            theme=kwargs.get('theme', ChartTheme.PROFESSIONAL)
        )
        
        chart_methods = {
            ChartType.LINE: builder.create_line_chart,
            ChartType.CANDLESTICK: builder.create_candlestick_chart,
            ChartType.HEATMAP: builder.create_heatmap,
            ChartType.HISTOGRAM: lambda d, **kw: builder.create_distribution_plot(
                d[d.columns[0]] if isinstance(d, pd.DataFrame) else d, **kw
            )
        }
        
        method = chart_methods.get(chart_type)
        if not method:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        return method(data, **kwargs)


def main():
    """Example usage of the charts module."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Price data
    prices = pd.DataFrame({
        'AAPL': 150 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates)))),
        'GOOGL': 100 * np.exp(np.cumsum(np.random.normal(0.0007, 0.025, len(dates)))),
        'MSFT': 300 * np.exp(np.cumsum(np.random.normal(0.0006, 0.018, len(dates)))),
    }, index=dates)
    
    # Returns
    returns = prices.pct_change().dropna()
    
    # Create chart builder
    builder = FinancialChartBuilder(theme=ChartTheme.PROFESSIONAL)
    
    # Create various charts
    print("Creating line chart...")
    line_chart = builder.create_line_chart(
        prices, 
        title="Stock Prices",
        interactive=True
    )
    
    print("Creating portfolio performance chart...")
    perf_chart = builder.create_portfolio_performance_chart(
        returns,
        title="Portfolio Performance Analysis"
    )
    
    print("Creating risk-return scatter...")
    scatter_chart = builder.create_risk_return_scatter(
        returns,
        title="Risk-Return Profile"
    )
    
    print("Creating correlation heatmap...")
    heatmap = builder.create_heatmap(
        returns,
        title="Asset Correlations"
    )
    
    print("Creating distribution plot...")
    dist_chart = builder.create_distribution_plot(
        returns['AAPL'],
        title="AAPL Returns Distribution"
    )
    
    # Save charts
    builder.save_chart(line_chart, "stock_prices.html", format="html")
    print("\nCharts created successfully!")


if __name__ == "__main__":
    main()
