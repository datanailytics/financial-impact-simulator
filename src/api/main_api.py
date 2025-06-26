"""
Main API Module - Financial Analytics Platform REST API
Author: Naiara Rodríguez Solano
Email: datanailytics@outlook.com
GitHub: https://github.com/datanailytics
Portfolio: https://datanailytics.github.io

This module provides the main API interface for the financial analytics
platform using FastAPI framework.
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, date, timedelta
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from src.core.financial_model import (
    FinancialAnalyzer, ModelParameters, ModelType,
    DiscountedCashFlowModel, PortfolioOptimizationModel, 
    MonteCarloSimulation, RiskAnalysisModel
)
from src.core.scenario_generator import ScenarioGenerator, ScenarioType
from src.core.risk_calculator import RiskCalculator, RiskMethod
from src.data.data_loader import DataManager, DataConfig, DataRequest, DataSource, DataFrequency
from src.data.data_validator import DataValidator, DataCleaner, ValidationRule
from src.visualization.charts import FinancialChartBuilder, ChartType
from src.visualization.reports import ReportFactory, ReportConfig, ReportFormat, ReportType
from src.utils.helpers import (
    timer, validate_input, calculate_returns, calculate_sharpe_ratio,
    format_percentage, format_currency
)
from src.utils.constants import (
    APP_NAME, APP_VERSION, APP_AUTHOR, APP_EMAIL,
    DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=f"Professional Financial Analytics API by {APP_AUTHOR}",
    contact={
        "name": APP_AUTHOR,
        "email": APP_EMAIL,
        "url": "https://datanailytics.github.io"
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

# Global instances
data_manager = DataManager()
financial_analyzer = FinancialAnalyzer()
scenario_generator = ScenarioGenerator()
risk_calculator = RiskCalculator()
chart_builder = FinancialChartBuilder()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: datetime
    author: str
    contact: str


class PortfolioRequest(BaseModel):
    """Portfolio analysis request model."""
    symbols: List[str] = Field(..., min_items=1, max_items=100)
    start_date: date
    end_date: date
    initial_capital: float = Field(default=1000000, gt=0)
    rebalance_frequency: str = Field(default="monthly")
    benchmark: Optional[str] = "SPY"
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class RiskAnalysisRequest(BaseModel):
    """Risk analysis request model."""
    portfolio_id: Optional[str] = None
    symbols: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    confidence_levels: List[float] = Field(default=[0.95, 0.99])
    method: str = Field(default="historical")
    time_horizon: int = Field(default=1, ge=1, le=252)
    
    @validator('weights')
    def validate_weights(cls, v, values):
        if v and 'symbols' in values and values['symbols']:
            if len(v) != len(values['symbols']):
                raise ValueError('weights must match number of symbols')
            if abs(sum(v) - 1.0) > 0.001:
                raise ValueError('weights must sum to 1.0')
        return v


class OptimizationRequest(BaseModel):
    """Portfolio optimization request model."""
    symbols: List[str] = Field(..., min_items=2, max_items=50)
    objective: str = Field(default="max_sharpe")
    constraints: Optional[Dict[str, Any]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    @validator('objective')
    def validate_objective(cls, v):
        valid_objectives = ['max_sharpe', 'min_variance', 'max_return', 'risk_parity']
        if v not in valid_objectives:
            raise ValueError(f'objective must be one of {valid_objectives}')
        return v


class ScenarioRequest(BaseModel):
    """Scenario analysis request model."""
    portfolio_id: Optional[str] = None
    scenario_type: str = Field(default="monte_carlo")
    n_scenarios: int = Field(default=1000, ge=100, le=10000)
    time_horizon: int = Field(default=252, ge=1, le=1260)
    parameters: Optional[Dict[str, Any]] = None


class ReportRequest(BaseModel):
    """Report generation request model."""
    report_type: str = Field(default="portfolio_summary")
    format: str = Field(default="pdf")
    portfolio_id: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    include_charts: bool = Field(default=True)
    include_tables: bool = Field(default=True)
    custom_sections: Optional[List[str]] = None


class BacktestRequest(BaseModel):
    """Backtesting request model."""
    strategy_id: str
    symbols: List[str]
    start_date: date
    end_date: date
    initial_capital: float = Field(default=1000000, gt=0)
    commission: float = Field(default=0.001, ge=0, le=0.1)
    slippage: float = Field(default=0.001, ge=0, le=0.1)
    parameters: Optional[Dict[str, Any]] = None


# ============================================================================
# AUTHENTICATION
# ============================================================================

def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Simple authentication for demo purposes."""
    # In production, implement proper authentication
    correct_username = "demo"
    correct_password = "datanailytics2024"
    
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        timestamp=datetime.now(),
        author=APP_AUTHOR,
        contact=APP_EMAIL
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        timestamp=datetime.now(),
        author=APP_AUTHOR,
        contact=APP_EMAIL
    )


@app.post("/api/v1/portfolio/analyze")
async def analyze_portfolio(
    request: PortfolioRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(authenticate_user)
):
    """
    Analyze portfolio performance and risk metrics.
    
    Args:
        request: Portfolio analysis request
        background_tasks: Background task manager
        current_user: Authenticated user
        
    Returns:
        Portfolio analysis results
    """
    try:
        # Log request
        logger.info(f"Portfolio analysis request from {current_user}: {request.symbols}")
        
        # Create data request
        data_request = DataRequest(
            symbols=request.symbols,
            start_date=datetime.combine(request.start_date, datetime.min.time()),
            end_date=datetime.combine(request.end_date, datetime.min.time()),
            frequency=DataFrequency.DAILY
        )
        
        # Load data (using Yahoo Finance for demo)
        yf_config = DataConfig(
            source=DataSource.YAHOO_FINANCE,
            connection_params={}
        )
        data_manager.register_source('yahoo', yf_config)
        price_data = data_manager.load_data('yahoo', data_request)
        
        if price_data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for specified symbols and date range"
            )
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Perform analysis
        analysis_results = {
            "portfolio_value": float(request.initial_capital),
            "total_return": float((price_data.iloc[-1] / price_data.iloc[0] - 1).mean()),
            "annualized_return": float(calculate_returns(price_data.mean(axis=1)).mean() * TRADING_DAYS_PER_YEAR),
            "volatility": float(returns.std().mean() * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "sharpe_ratio": float(calculate_sharpe_ratio(returns.mean(axis=1))),
            "max_drawdown": float(((price_data / price_data.expanding().max()) - 1).min().min()),
            "symbols_analyzed": request.symbols,
            "period": {
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat(),
                "trading_days": len(price_data)
            }
        }
        
        # Schedule background report generation
        background_tasks.add_task(
            generate_portfolio_report,
            request.symbols,
            analysis_results
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "data": analysis_results,
                "message": "Portfolio analysis completed successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Portfolio analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/risk/calculate")
async def calculate_risk(
    request: RiskAnalysisRequest,
    current_user: str = Depends(authenticate_user)
):
    """
    Calculate portfolio risk metrics.
    
    Args:
        request: Risk analysis request
        current_user: Authenticated user
        
    Returns:
        Risk analysis results
    """
    try:
        logger.info(f"Risk calculation request from {current_user}")
        
        # Validate request
        if not request.portfolio_id and not request.symbols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either portfolio_id or symbols must be provided"
            )
        
        # Mock data for demonstration
        # In production, load actual portfolio data
        if request.symbols:
            returns_data = pd.DataFrame(
                np.random.normal(0.001, 0.02, (252, len(request.symbols))),
                columns=request.symbols
            )
        else:
            # Load portfolio data
            returns_data = pd.DataFrame(
                np.random.normal(0.001, 0.02, (252, 5)),
                columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            )
        
        # Calculate risk metrics
        risk_metrics = risk_calculator.calculate_risk_metrics(
            returns_data,
            method=RiskMethod[request.method.upper()]
        )
        
        # Format results
        results = {}
        for asset, metrics in risk_metrics.items():
            results[asset] = {
                "var_95": float(metrics.var_95),
                "var_99": float(metrics.var_99),
                "cvar_95": float(metrics.cvar_95),
                "cvar_99": float(metrics.cvar_99),
                "volatility": float(metrics.annual_volatility),
                "max_drawdown": float(metrics.max_drawdown),
                "sharpe_ratio": float(metrics.sharpe_ratio),
                "sortino_ratio": float(metrics.sortino_ratio)
            }
        
        # Portfolio-level metrics
        if request.weights:
            weights = np.array(request.weights)
            portfolio_var = risk_calculator.calculate_portfolio_var(
                returns_data, weights, request.confidence_levels[0]
            )
            
            results["portfolio"] = {
                "var": float(portfolio_var),
                "confidence_level": request.confidence_levels[0]
            }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "data": results,
                "message": "Risk calculation completed"
            }
        )
        
    except Exception as e:
        logger.error(f"Risk calculation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk calculation failed: {str(e)}"
        )


@app.post("/api/v1/optimize")
async def optimize_portfolio(
    request: OptimizationRequest,
    current_user: str = Depends(authenticate_user)
):
    """
    Optimize portfolio allocation.
    
    Args:
        request: Optimization request
        current_user: Authenticated user
        
    Returns:
        Optimization results
    """
    try:
        logger.info(f"Optimization request from {current_user}: {request.objective}")
        
        # Create data request
        end_date = request.end_date or date.today()
        start_date = request.start_date or (end_date - timedelta(days=365))
        
        data_request = DataRequest(
            symbols=request.symbols,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            frequency=DataFrequency.DAILY
        )
        
        # Load data
        price_data = data_manager.load_data('yahoo', data_request)
        
        if price_data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for optimization"
            )
        
        # Add optimization model
        financial_analyzer.add_model(
            "optimizer",
            ModelType.PORTFOLIO,
            {"risk_free_rate": DEFAULT_RISK_FREE_RATE}
        )
        
        # Run optimization
        optimization_results = financial_analyzer.run_analysis(price_data, ["optimizer"])
        
        if optimization_results["optimizer"]["success"]:
            results = optimization_results["optimizer"]["results"]
            
            # Extract key results
            if request.objective == "max_sharpe":
                optimal_weights = results["max_sharpe_portfolio"]["weights"]
                optimal_metrics = results["max_sharpe_portfolio"]["metrics"]
            else:
                optimal_weights = results["min_variance_portfolio"]["weights"]
                optimal_metrics = results["min_variance_portfolio"]["metrics"]
            
            response_data = {
                "optimal_weights": dict(zip(request.symbols, optimal_weights)),
                "expected_return": optimal_metrics["expected_return"],
                "volatility": optimal_metrics["volatility"],
                "sharpe_ratio": optimal_metrics["sharpe_ratio"],
                "efficient_frontier": results.get("efficient_frontier", {})
            }
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "success",
                    "data": response_data,
                    "message": f"Portfolio optimized for {request.objective}"
                }
            )
        else:
            raise ValueError("Optimization failed")
            
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.post("/api/v1/scenarios/generate")
async def generate_scenarios(
    request: ScenarioRequest,
    current_user: str = Depends(authenticate_user)
):
    """
    Generate scenario analysis.
    
    Args:
        request: Scenario request
        current_user: Authenticated user
        
    Returns:
        Scenario analysis results
    """
    try:
        logger.info(f"Scenario generation request from {current_user}: {request.scenario_type}")
        
        # Mock historical data for demonstration
        dates = pd.date_range(end=date.today(), periods=504, freq='D')
        mock_data = pd.DataFrame({
            'AAPL': 150 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates)))),
            'GOOGL': 100 * np.exp(np.cumsum(np.random.normal(0.0007, 0.025, len(dates)))),
            'MSFT': 300 * np.exp(np.cumsum(np.random.normal(0.0006, 0.018, len(dates)))),
        }, index=dates)
        
        # Generate scenarios
        scenario_type_map = {
            "historical": ScenarioType.HISTORICAL,
            "monte_carlo": ScenarioType.MONTE_CARLO,
            "stress_test": ScenarioType.STRESS_TEST,
            "bootstrap": ScenarioType.BOOTSTRAP
        }
        
        scenario_set = scenario_generator.generate_scenarios(
            mock_data,
            scenario_type_map.get(request.scenario_type, ScenarioType.MONTE_CARLO),
            n_scenarios=request.n_scenarios,
            time_horizon=request.time_horizon
        )
        
        # Analyze scenarios
        portfolio_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        impact_analysis = scenario_generator.analyze_scenario_impacts(
            scenario_set, portfolio_weights
        )
        
        # Format results
        results = {
            "scenario_type": request.scenario_type,
            "n_scenarios": len(scenario_set.scenarios),
            "time_horizon": request.time_horizon,
            "summary": impact_analysis["summary"],
            "scenarios": [
                {
                    "id": s.scenario_id,
                    "description": s.description,
                    "probability": s.probability,
                    "impact": next(
                        (i["portfolio_return"] for i in impact_analysis["scenario_impacts"] 
                         if i["scenario_id"] == s.scenario_id), 
                        0
                    )
                }
                for s in scenario_set.scenarios[:10]  # Return top 10 scenarios
            ]
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "data": results,
                "message": "Scenario analysis completed"
            }
        )
        
    except Exception as e:
        logger.error(f"Scenario generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenario generation failed: {str(e)}"
        )


@app.post("/api/v1/reports/generate")
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(authenticate_user)
):
    """
    Generate financial report.
    
    Args:
        request: Report request
        background_tasks: Background task manager
        current_user: Authenticated user
        
    Returns:
        Report generation status
    """
    try:
        logger.info(f"Report generation request from {current_user}: {request.report_type}")
        
        # Create report configuration
        report_config = ReportConfig(
            title=f"Financial Analysis Report - {datetime.now().strftime('%B %Y')}",
            subtitle=f"Generated for {current_user}",
            author=APP_AUTHOR,
            report_type=ReportType[request.report_type.upper()],
            format=ReportFormat[request.format.upper()],
            include_charts=request.include_charts,
            include_tables=request.include_tables
        )
        
        # Generate report ID
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user}"
        
        # Schedule background report generation
        background_tasks.add_task(
            generate_report_background,
            report_id,
            report_config,
            request
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "accepted",
                "data": {
                    "report_id": report_id,
                    "status_url": f"/api/v1/reports/status/{report_id}",
                    "download_url": f"/api/v1/reports/download/{report_id}"
                },
                "message": "Report generation started"
            }
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )


@app.get("/api/v1/reports/status/{report_id}")
async def get_report_status(
    report_id: str,
    current_user: str = Depends(authenticate_user)
):
    """Get report generation status."""
    # In production, check actual report status from database/cache
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "success",
            "data": {
                "report_id": report_id,
                "status": "completed",
                "progress": 100,
                "message": "Report generated successfully"
            }
        }
    )


@app.get("/api/v1/reports/download/{report_id}")
async def download_report(
    report_id: str,
    current_user: str = Depends(authenticate_user)
):
    """Download generated report."""
    # In production, retrieve actual report file
    report_path = Path(f"./reports/{report_id}.pdf")
    
    if report_path.exists():
        return FileResponse(
            path=str(report_path),
            media_type="application/pdf",
            filename=f"{report_id}.pdf"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )


@app.post("/api/v1/backtest")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(authenticate_user)
):
    """
    Run strategy backtest.
    
    Args:
        request: Backtest request
        background_tasks: Background task manager
        current_user: Authenticated user
        
    Returns:
        Backtest initiation status
    """
    try:
        logger.info(f"Backtest request from {current_user}: {request.strategy_id}")
        
        # Generate backtest ID
        backtest_id = f"backtest_{request.strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Schedule background backtest
        background_tasks.add_task(
            run_backtest_background,
            backtest_id,
            request
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "accepted",
                "data": {
                    "backtest_id": backtest_id,
                    "status_url": f"/api/v1/backtest/status/{backtest_id}",
                    "results_url": f"/api/v1/backtest/results/{backtest_id}"
                },
                "message": "Backtest started"
            }
        )
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}"
        )


@app.get("/api/v1/market/summary")
async def get_market_summary(current_user: str = Depends(authenticate_user)):
    """Get current market summary."""
    try:
        # Mock market data for demonstration
        market_data = {
            "indices": {
                "SP500": {"value": 4567.89, "change": 0.0123, "volume": 3.2e9},
                "NASDAQ": {"value": 14234.56, "change": 0.0187, "volume": 2.8e9},
                "DOW": {"value": 35678.90, "change": 0.0095, "volume": 2.5e9}
            },
            "sectors": {
                "Technology": {"change": 0.0235, "volume": 1.2e9},
                "Healthcare": {"change": 0.0112, "volume": 0.9e9},
                "Finance": {"change": -0.0045, "volume": 1.1e9},
                "Energy": {"change": 0.0189, "volume": 0.8e9}
            },
            "volatility": {
                "VIX": {"value": 15.67, "change": -0.0543}
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "data": market_data
            }
        )
        
    except Exception as e:
        logger.error(f"Market summary error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve market summary: {str(e)}"
        )


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def generate_portfolio_report(symbols: List[str], analysis_results: Dict[str, Any]):
    """Background task to generate portfolio report."""
    try:
        logger.info(f"Generating portfolio report for {symbols}")
        # Report generation logic here
        # This would save the report to storage
    except Exception as e:
        logger.error(f"Portfolio report generation failed: {str(e)}")


def generate_report_background(report_id: str, config: ReportConfig, request: ReportRequest):
    """Background task to generate report."""
    try:
        logger.info(f"Generating report {report_id}")
        # Report generation logic here
        # This would create and save the report
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")


def run_backtest_background(backtest_id: str, request: BacktestRequest):
    """Background task to run backtest."""
    try:
        logger.info(f"Running backtest {backtest_id}")
        # Backtest logic here
        # This would perform the backtest and save results
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "message": str(exc)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred"
        }
    )


# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Author: {APP_AUTHOR} ({APP_EMAIL})")
    
    # Initialize data sources
    try:
        # Register Yahoo Finance as default source
        yf_config = DataConfig(
            source=DataSource.YAHOO_FINANCE,
            connection_params={},
            cache_enabled=True
        )
        data_manager.register_source('yahoo', yf_config)
        logger.info("Data sources initialized")
    except Exception as e:
        logger.error(f"Failed to initialize data sources: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down application")
    # Cleanup code here


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
