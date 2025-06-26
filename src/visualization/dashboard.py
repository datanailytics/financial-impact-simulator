"""
Dashboard Module - Interactive Financial Analytics Dashboard
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides interactive dashboard capabilities using Dash and Streamlit
for real-time financial data visualization and analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    title: str = "Financial Analytics Dashboard"
    theme: str = "dark"
    update_interval: int = 60000  # milliseconds
    port: int = 8050
    debug: bool = False
    cache_timeout: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'theme': self.theme,
            'update_interval': self.update_interval,
            'port': self.port,
            'debug': self.debug,
            'cache_timeout': self.cache_timeout
        }


class DashDashboard:
    """Interactive dashboard using Plotly Dash."""
    
    def __init__(self, config: DashboardConfig = None):
        """
        Initialize Dash dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY if self.config.theme == "dark" else dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.data_cache = {}
        
    def create_layout(self) -> dbc.Container:
        """Create dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.config.title, className="text-center mb-4"),
                    html.P("Real-time Financial Analytics by Naiara Rodríguez Solano", 
                          className="text-center text-muted")
                ])
            ]),
            
            # Navigation tabs
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id="main-tabs", value="portfolio", children=[
                        dcc.Tab(label="Portfolio Overview", value="portfolio"),
                        dcc.Tab(label="Risk Analysis", value="risk"),
                        dcc.Tab(label="Market Analysis", value="market"),
                        dcc.Tab(label="Performance", value="performance"),
                        dcc.Tab(label="Reports", value="reports")
                    ])
                ])
            ], className="mb-4"),
            
            # Tab content
            html.Div(id="tab-content"),
            
            # Interval component for updates
            dcc.Interval(
                id='interval-component',
                interval=self.config.update_interval,
                n_intervals=0
            ),
            
            # Store components for data
            dcc.Store(id='portfolio-data'),
            dcc.Store(id='market-data'),
            dcc.Store(id='risk-data')
            
        ], fluid=True)
    
    def create_portfolio_tab(self) -> html.Div:
        """Create portfolio overview tab."""
        return html.Div([
            dbc.Row([
                # Portfolio summary cards
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Value", className="card-title"),
                            html.H2("$1,234,567", id="total-value", className="text-primary"),
                            html.P("↑ 12.5% YTD", className="text-success")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Daily P&L", className="card-title"),
                            html.H2("$12,345", id="daily-pnl", className="text-success"),
                            html.P("↑ 0.98%", className="text-success")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sharpe Ratio", className="card-title"),
                            html.H2("1.85", id="sharpe-ratio", className="text-info"),
                            html.P("Risk-adjusted return", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Max Drawdown", className="card-title"),
                            html.H2("-8.2%", id="max-drawdown", className="text-warning"),
                            html.P("Peak to trough", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                # Portfolio allocation chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Allocation"),
                        dbc.CardBody([
                            dcc.Graph(id="allocation-chart")
                        ])
                    ])
                ], width=6),
                
                # Performance chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cumulative Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                # Holdings table
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Holdings"),
                        dbc.CardBody([
                            html.Div(id="holdings-table")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_risk_tab(self) -> html.Div:
        """Create risk analysis tab."""
        return html.Div([
            dbc.Row([
                # Risk metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("VaR (95%)", className="card-title"),
                            html.H2("$45,678", id="var-95", className="text-danger"),
                            html.P("1-day Value at Risk", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("CVaR (95%)", className="card-title"),
                            html.H2("$67,890", id="cvar-95", className="text-danger"),
                            html.P("Expected Shortfall", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Beta", className="card-title"),
                            html.H2("1.12", id="portfolio-beta", className="text-info"),
                            html.P("Market sensitivity", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Volatility", className="card-title"),
                            html.H2("18.5%", id="portfolio-vol", className="text-warning"),
                            html.P("Annualized", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                # Risk decomposition
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Decomposition"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-decomposition-chart")
                        ])
                    ])
                ], width=6),
                
                # Correlation matrix
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Correlation Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-heatmap")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                # Stress test results
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stress Test Scenarios"),
                        dbc.CardBody([
                            dcc.Graph(id="stress-test-chart")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_market_tab(self) -> html.Div:
        """Create market analysis tab."""
        return html.Div([
            dbc.Row([
                # Market indicators
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Indices"),
                        dbc.CardBody([
                            html.Div(id="market-indices-table")
                        ])
                    ])
                ], width=6),
                
                # Sector performance
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sector Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="sector-performance-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                # Market heatmap
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Heatmap"),
                        dbc.CardBody([
                            dcc.Graph(id="market-heatmap")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                # Economic indicators
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Economic Indicators"),
                        dbc.CardBody([
                            dcc.Graph(id="economic-indicators-chart")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_performance_tab(self) -> html.Div:
        """Create performance analysis tab."""
        return html.Div([
            dbc.Row([
                # Time period selector
                dbc.Col([
                    html.Label("Time Period:"),
                    dcc.Dropdown(
                        id="time-period-dropdown",
                        options=[
                            {"label": "1 Day", "value": "1D"},
                            {"label": "1 Week", "value": "1W"},
                            {"label": "1 Month", "value": "1M"},
                            {"label": "3 Months", "value": "3M"},
                            {"label": "YTD", "value": "YTD"},
                            {"label": "1 Year", "value": "1Y"},
                            {"label": "All", "value": "ALL"}
                        ],
                        value="YTD"
                    )
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                # Performance metrics table
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            html.Div(id="performance-metrics-table")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                # Returns distribution
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Returns Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="returns-distribution-chart")
                        ])
                    ])
                ], width=6),
                
                # Rolling metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rolling Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="rolling-metrics-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                # Attribution analysis
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Attribution"),
                        dbc.CardBody([
                            dcc.Graph(id="attribution-chart")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_reports_tab(self) -> html.Div:
        """Create reports tab."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H3("Generate Reports"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Report Type:"),
                    dcc.Dropdown(
                        id="report-type-dropdown",
                        options=[
                            {"label": "Portfolio Summary", "value": "portfolio"},
                            {"label": "Risk Report", "value": "risk"},
                            {"label": "Performance Report", "value": "performance"},
                            {"label": "Compliance Report", "value": "compliance"},
                            {"label": "Custom Report", "value": "custom"}
                        ],
                        value="portfolio"
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Format:"),
                    dcc.Dropdown(
                        id="report-format-dropdown",
                        options=[
                            {"label": "PDF", "value": "pdf"},
                            {"label": "Excel", "value": "excel"},
                            {"label": "HTML", "value": "html"},
                            {"label": "PowerPoint", "value": "pptx"}
                        ],
                        value="pdf"
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Frequency:"),
                    dcc.Dropdown(
                        id="report-frequency-dropdown",
                        options=[
                            {"label": "One-time", "value": "once"},
                            {"label": "Daily", "value": "daily"},
                            {"label": "Weekly", "value": "weekly"},
                            {"label": "Monthly", "value": "monthly"}
                        ],
                        value="once"
                    )
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Generate Report", id="generate-report-btn", 
                             color="primary", size="lg", className="mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id="report-status"),
                    html.Div(id="report-preview")
                ])
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'value')
        )
        def render_tab_content(active_tab):
            """Render content based on selected tab."""
            if active_tab == 'portfolio':
                return self.create_portfolio_tab()
            elif active_tab == 'risk':
                return self.create_risk_tab()
            elif active_tab == 'market':
                return self.create_market_tab()
            elif active_tab == 'performance':
                return self.create_performance_tab()
            elif active_tab == 'reports':
                return self.create_reports_tab()
            else:
                return html.Div("Tab not found")
        
        @self.app.callback(
            Output('allocation-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_allocation_chart(n):
            """Update allocation chart."""
            # Sample data - replace with real data
            allocations = {
                'Stocks': 0.60,
                'Bonds': 0.25,
                'Real Estate': 0.10,
                'Commodities': 0.05
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(allocations.keys()),
                values=list(allocations.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_performance_chart(n):
            """Update performance chart."""
            # Sample data - replace with real data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            portfolio_value = 1000000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, len(dates))))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_value,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00d9ff', width=2)
            ))
            
            fig.update_layout(
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('holdings-table', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_holdings_table(n):
            """Update holdings table."""
            # Sample data - replace with real data
            holdings_data = pd.DataFrame({
                'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'BRK.B'],
                'Quantity': [1000, 500, 800, 300, 200],
                'Price': [150.25, 125.50, 380.75, 145.60, 350.25],
                'Value': [150250, 62750, 304600, 43680, 70050],
                'Weight': [0.238, 0.099, 0.482, 0.069, 0.111],
                'Daily P&L': [2500, -1200, 4800, 600, -300],
                'Daily %': [1.69, -1.88, 1.60, 1.40, -0.43]
            })
            
            return dash_table.DataTable(
                data=holdings_data.to_dict('records'),
                columns=[
                    {'name': 'Symbol', 'id': 'Symbol'},
                    {'name': 'Quantity', 'id': 'Quantity', 'type': 'numeric'},
                    {'name': 'Price', 'id': 'Price', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                    {'name': 'Value', 'id': 'Value', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                    {'name': 'Weight', 'id': 'Weight', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                    {'name': 'Daily P&L', 'id': 'Daily P&L', 'type': 'numeric', 'format': {'specifier': '+,.0f'}},
                    {'name': 'Daily %', 'id': 'Daily %', 'type': 'numeric', 'format': {'specifier': '+.2%'}}
                ],
                style_cell={'textAlign': 'right', 'font-family': 'monospace'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Daily P&L', 'filter_query': '{Daily P&L} > 0'},
                        'color': '#00d9ff'
                    },
                    {
                        'if': {'column_id': 'Daily P&L', 'filter_query': '{Daily P&L} < 0'},
                        'color': '#ff4444'
                    },
                    {
                        'if': {'column_id': 'Daily %', 'filter_query': '{Daily %} > 0'},
                        'color': '#00d9ff'
                    },
                    {
                        'if': {'column_id': 'Daily %', 'filter_query': '{Daily %} < 0'},
                        'color': '#ff4444'
                    }
                ],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)' if self.config.theme == 'dark' else 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        
        # Additional callbacks for other components...
    
    def run(self):
        """Run the dashboard."""
        self.app.layout = self.create_layout()
        self.setup_callbacks()
        
        logger.info(f"Starting dashboard on port {self.config.port}")
        self.app.run_server(
            debug=self.config.debug,
            port=self.config.port,
            host='0.0.0.0'
        )


class StreamlitDashboard:
    """Interactive dashboard using Streamlit."""
    
    def __init__(self, config: DashboardConfig = None):
        """
        Initialize Streamlit dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        st.set_page_config(
            page_title=self.config.title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply theme
        if self.config.theme == "dark":
            st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar navigation."""
        with st.sidebar:
            st.title("Navigation")
            
            page = st.radio(
                "Select Page",
                ["Portfolio Overview", "Risk Analysis", "Market Analysis", 
                 "Performance", "Reports", "Settings"]
            )
            
            st.markdown("---")
            
            # Date range selector
            st.subheader("Date Range")
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", datetime.now())
            
            st.markdown("---")
            
            # Refresh settings
            st.subheader("Auto Refresh")
            auto_refresh = st.checkbox("Enable Auto Refresh")
            if auto_refresh:
                refresh_rate = st.slider("Refresh Rate (seconds)", 5, 300, 60)
            
            st.markdown("---")
            
            # About section
            st.markdown("""
            **Financial Analytics Dashboard**
            
            Created by: Naiara Rodríguez Solano
            
            [GitHub](https://github.com/datanailytics) | 
            [Portfolio](https://datanailytics.github.io)
            """)
            
        return page
    
    def render_portfolio_overview(self):
        """Render portfolio overview page."""
        st.title("Portfolio Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Value",
                value="$1,234,567",
                delta="12.5%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="Daily P&L",
                value="$12,345",
                delta="0.98%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio",
                value="1.85",
                delta="0.15",
                delta_color="normal"
            )
        
        with col4:
            st.metric(
                label="Max Drawdown",
                value="-8.2%",
                delta="-0.5%",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Allocation")
            
            # Sample allocation data
            allocations = pd.DataFrame({
                'Asset Class': ['Stocks', 'Bonds', 'Real Estate', 'Commodities'],
                'Allocation': [0.60, 0.25, 0.10, 0.05]
            })
            
            fig = px.pie(
                allocations, 
                values='Allocation', 
                names='Asset Class',
                title='Current Allocation'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Performance Chart")
            
            # Sample performance data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            performance = pd.DataFrame({
                'Date': dates,
                'Portfolio': np.cumsum(np.random.normal(0.0003, 0.01, len(dates))),
                'Benchmark': np.cumsum(np.random.normal(0.0002, 0.008, len(dates)))
            })
            
            fig = px.line(
                performance, 
                x='Date', 
                y=['Portfolio', 'Benchmark'],
                title='Cumulative Performance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Holdings table
        st.subheader("Current Holdings")
        
        holdings = pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'BRK.B'],
            'Quantity': [1000, 500, 800, 300, 200],
            'Price': [150.25, 125.50, 380.75, 145.60, 350.25],
            'Value': [150250, 62750, 304600, 43680, 70050],
            'Weight (%)': [23.8, 9.9, 48.2, 6.9, 11.1],
            'Daily P&L': [2500, -1200, 4800, 600, -300],
            'Daily (%)': [1.69, -1.88, 1.60, 1.40, -0.43]
        })
        
        # Format the dataframe for display
        st.dataframe(
            holdings.style.format({
                'Price': '${:.2f}',
                'Value': '${:,.0f}',
                'Weight (%)': '{:.1f}%',
                'Daily P&L': '${:+,.0f}',
                'Daily (%)': '{:+.2f}%'
            }).applymap(
                lambda x: 'color: #00d9ff' if isinstance(x, (int, float)) and x > 0 else 'color: #ff4444',
                subset=['Daily P&L', 'Daily (%)']
            ),
            use_container_width=True,
            height=300
        )
    
    def render_risk_analysis(self):
        """Render risk analysis page."""
        st.title("Risk Analysis")
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", "$45,678", "-$2,345")
        with col2:
            st.metric("CVaR (95%)", "$67,890", "-$3,456")
        with col3:
            st.metric("Beta", "1.12", "0.05")
        with col4:
            st.metric("Volatility", "18.5%", "1.2%")
        
        st.markdown("---")
        
        # Risk charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Decomposition")
            
            # Sample risk data
            risk_data = pd.DataFrame({
                'Asset': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'BRK.B'],
                'Risk Contribution': [0.25, 0.20, 0.30, 0.15, 0.10]
            })
            
            fig = px.bar(
                risk_data,
                x='Asset',
                y='Risk Contribution',
                title='Risk Contribution by Asset'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Correlation Matrix")
            
            # Sample correlation matrix
            assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'BRK.B']
            corr_matrix = np.random.rand(5, 5)
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            np.fill_diagonal(corr_matrix, 1)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=assets,
                y=assets,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2),
                texttemplate='%{text}'
            ))
            fig.update_layout(title='Asset Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Stress test scenarios
        st.subheader("Stress Test Scenarios")
        
        scenarios = pd.DataFrame({
            'Scenario': ['Market Crash', 'Interest Rate Shock', 'Currency Crisis', 
                        'Liquidity Crisis', 'Tech Bubble'],
            'Portfolio Impact (%)': [-15.2, -8.5, -10.3, -12.7, -18.9],
            'VaR Impact ($)': [-125000, -75000, -95000, -110000, -145000],
            'Probability': [0.05, 0.10, 0.08, 0.06, 0.03]
        })
        
        fig = px.bar(
            scenarios,
            x='Scenario',
            y='Portfolio Impact (%)',
            color='Portfolio Impact (%)',
            color_continuous_scale='RdYlGn',
            title='Stress Test Results'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_market_analysis(self):
        """Render market analysis page."""
        st.title("Market Analysis")
        
        # Market overview
        st.subheader("Market Indices")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("S&P 500", "4,567.89", "1.23%")
        with col2:
            st.metric("NASDAQ", "14,234.56", "1.87%")
        with col3:
            st.metric("DOW", "35,678.90", "0.95%")
        with col4:
            st.metric("VIX", "15.67", "-5.43%", delta_color="inverse")
        
        st.markdown("---")
        
        # Sector performance
        st.subheader("Sector Performance")
        
        sectors = pd.DataFrame({
            'Sector': ['Technology', 'Healthcare', 'Finance', 'Energy', 
                      'Consumer', 'Industrial', 'Real Estate', 'Utilities'],
            'Performance': [3.5, 1.2, -0.5, 2.8, 0.9, -1.2, 0.3, -0.8]
        })
        
        fig = px.bar(
            sectors.sort_values('Performance'),
            x='Performance',
            y='Sector',
            orientation='h',
            color='Performance',
            color_continuous_scale='RdYlGn',
            title='Sector Performance (% Daily)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Market heatmap
        st.subheader("Market Heatmap")
        
        # Generate sample market data
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'BRK.B', 'JPM', 
                  'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'NVDA']
        market_data = pd.DataFrame({
            'Ticker': tickers,
            'Change (%)': np.random.uniform(-5, 5, len(tickers)),
            'Volume': np.random.uniform(10, 100, len(tickers)),
            'Market Cap': np.random.uniform(100, 1000, len(tickers))
        })
        
        # Create treemap
        fig = px.treemap(
            market_data,
            path=['Ticker'],
            values='Market Cap',
            color='Change (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Market Heatmap'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance(self):
        """Render performance analysis page."""
        st.title("Performance Analysis")
        
        # Time period selector
        period = st.selectbox(
            "Select Time Period",
            ["1 Day", "1 Week", "1 Month", "3 Months", "YTD", "1 Year", "All Time"]
        )
        
        # Performance metrics table
        st.subheader("Performance Metrics")
        
        metrics = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 
                      'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown',
                      'Calmar Ratio', 'Win Rate', 'Profit Factor'],
            'Portfolio': [15.3, 12.8, 18.5, 1.85, 2.15, -8.2, 1.56, 0.58, 1.45],
            'Benchmark': [10.2, 9.5, 15.2, 1.45, 1.78, -10.5, 0.95, 0.52, 1.25],
            'Alpha': [5.1, 3.3, 3.3, 0.40, 0.37, 2.3, 0.61, 0.06, 0.20]
        })
        
        st.dataframe(
            metrics.style.format({
                'Portfolio': '{:.1f}',
                'Benchmark': '{:.1f}',
                'Alpha': '{:+.1f}'
            }).applymap(
                lambda x: 'background-color: #1a3a1a' if isinstance(x, (int, float)) and x > 0 else 'background-color: #3a1a1a',
                subset=['Alpha']
            ),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Returns Distribution")
            
            # Generate sample returns
            returns = np.random.normal(0.001, 0.02, 1000)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                histnorm='probability density'
            ))
            
            # Add normal distribution overlay
            x_range = np.linspace(returns.min(), returns.max(), 100)
            from scipy import stats
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, returns.mean(), returns.std()),
                mode='lines',
                name='Normal Distribution'
            ))
            
            fig.update_layout(title='Daily Returns Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rolling Metrics")
            
            # Generate sample rolling metrics
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            rolling_sharpe = pd.DataFrame({
                'Date': dates,
                'Sharpe Ratio': 1.5 + 0.5 * np.sin(np.arange(len(dates)) / 30) + np.random.normal(0, 0.1, len(dates))
            })
            
            fig = px.line(
                rolling_sharpe,
                x='Date',
                y='Sharpe Ratio',
                title='30-Day Rolling Sharpe Ratio'
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Performance attribution
        st.subheader("Performance Attribution")
        
        attribution = pd.DataFrame({
            'Factor': ['Asset Allocation', 'Security Selection', 'Timing', 
                      'Currency', 'Other'],
            'Contribution (%)': [3.2, 2.5, 1.8, 0.5, 0.3]
        })
        
        fig = px.waterfall(
            x=attribution['Factor'],
            y=attribution['Contribution (%)'],
            title='Performance Attribution Analysis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_reports(self):
        """Render reports page."""
        st.title("Report Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Report Configuration")
            
            report_type = st.selectbox(
                "Report Type",
                ["Portfolio Summary", "Risk Report", "Performance Report", 
                 "Compliance Report", "Custom Report"]
            )
            
            report_format = st.radio(
                "Output Format",
                ["PDF", "Excel", "HTML", "PowerPoint"]
            )
            
            frequency = st.selectbox(
                "Frequency",
                ["One-time", "Daily", "Weekly", "Monthly", "Quarterly"]
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
            include_tables = st.checkbox("Include Detailed Tables", value=True)
            include_commentary = st.checkbox("Include Commentary", value=False)
            
        with col2:
            st.subheader("Report Schedule")
            
            if frequency != "One-time":
                schedule_time = st.time_input("Schedule Time", datetime.now().time())
                
                if frequency in ["Weekly", "Monthly", "Quarterly"]:
                    if frequency == "Weekly":
                        day_of_week = st.selectbox("Day of Week", 
                                                  ["Monday", "Tuesday", "Wednesday", 
                                                   "Thursday", "Friday"])
                    elif frequency == "Monthly":
                        day_of_month = st.number_input("Day of Month", 1, 31, 1)
                
                recipients = st.text_area("Email Recipients (one per line)")
            
            st.markdown("---")
            
            if st.button("Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Simulate report generation
                    import time
                    time.sleep(2)
                    
                st.success(f"Report generated successfully!")
                st.download_button(
                    label="Download Report",
                    data=b"Sample report content",
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        st.markdown("---")
        
        # Recent reports
        st.subheader("Recent Reports")
        
        recent_reports = pd.DataFrame({
            'Report Name': ['Portfolio Summary - Dec 2023', 'Risk Report - Q4 2023',
                          'Performance Report - 2023', 'Compliance Report - Nov 2023'],
            'Generated': ['2023-12-15 09:00', '2023-12-10 14:30',
                        '2023-12-01 10:00', '2023-11-30 16:45'],
            'Format': ['PDF', 'Excel', 'PDF', 'HTML'],
            'Size': ['2.3 MB', '1.5 MB', '3.7 MB', '0.8 MB']
        })
        
        st.dataframe(recent_reports, use_container_width=True)
    
    def render_settings(self):
        """Render settings page."""
        st.title("Dashboard Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Display Settings")
            
            theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
            currency = st.selectbox("Display Currency", ["USD", "EUR", "GBP", "JPY"])
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
            number_format = st.selectbox("Number Format", ["1,234.56", "1.234,56", "1 234.56"])
            
            st.subheader("Data Settings")
            
            data_source = st.selectbox("Primary Data Source", 
                                     ["Yahoo Finance", "Bloomberg", "Reuters", "Custom API"])
            update_frequency = st.slider("Update Frequency (minutes)", 1, 60, 5)
            cache_duration = st.slider("Cache Duration (minutes)", 5, 1440, 60)
            
        with col2:
            st.subheader("Alert Settings")
            
            enable_alerts = st.checkbox("Enable Email Alerts")
            
            if enable_alerts:
                alert_email = st.text_input("Alert Email Address")
                
                st.write("Alert Triggers:")
                drawdown_alert = st.number_input("Max Drawdown Alert (%)", 0, 100, 10)
                volatility_alert = st.number_input("Volatility Alert (%)", 0, 100, 25)
                var_alert = st.number_input("VaR Breach Alert ($)", 0, 1000000, 50000)
            
            st.subheader("Export Settings")
            
            default_export_format = st.selectbox("Default Export Format", 
                                               ["Excel", "CSV", "JSON", "Parquet"])
            include_metadata = st.checkbox("Include Metadata in Exports", value=True)
            compress_exports = st.checkbox("Compress Large Exports", value=True)
        
        st.markdown("---")
        
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
            
        if st.button("Reset to Defaults"):
            st.warning("Are you sure you want to reset all settings to defaults?")
    
    def run(self):
        """Run the Streamlit dashboard."""
        # Header
        st.markdown(f"# {self.config.title}")
        st.markdown("---")
        
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Render selected page
        if page == "Portfolio Overview":
            self.render_portfolio_overview()
        elif page == "Risk Analysis":
            self.render_risk_analysis()
        elif page == "Market Analysis":
            self.render_market_analysis()
        elif page == "Performance":
            self.render_performance()
        elif page == "Reports":
            self.render_reports()
        elif page == "Settings":
            self.render_settings()


def main():
    """Example usage of dashboard module."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'streamlit':
        # Run Streamlit dashboard
        dashboard = StreamlitDashboard()
        dashboard.run()
    else:
        # Run Dash dashboard
        config = DashboardConfig(
            title="Financial Analytics Dashboard - Naiara Rodríguez Solano",
            theme="dark",
            debug=True
        )
        
        dashboard = DashDashboard(config)
        dashboard.run()


if __name__ == "__main__":
    main()
