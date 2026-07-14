from pydantic import BaseModel, Field
from typing import List, Literal,Optional,Dict,Any,TypedDict



class NewsInput(BaseModel):
    company: str = Field(description="Company name or stock ticker.")


class NewsArticle(BaseModel):
    title: str = Field(description="News title")
    summary: str = Field(description="Brief news summary")
    source: str = Field(description="News source")
    url: str = Field(description="Article URL")
    sentiment: Literal[
        "Bullish",
        "Bearish",
        "Neutral"
    ] = Field(description="News sentiment")


class NewsResponse(BaseModel):
    company: str = Field(description="Company name")
    category: str = Field(description="News category")
    articles: List[NewsArticle] = Field(description="List of news articles")

class StockPriceInput(BaseModel):
    """
    Input schema for retrieving real-time stock information.
    """
    ticker_symbol: str = Field(
        ...,
        description="The stock ticker symbol to look up (e.g., AAPL, MSFT, TSLA)."
    )


class StockPriceOutput(BaseModel):
    """
    Output schema for real-time stock price and metadata.
    """
    symbol: Optional[str] = Field(None, description="The ticker symbol of the company.")
    company_name: Optional[str] = Field(None, description="The official name of the company.")
    current_price: Optional[float] = Field(None, description="The current trading price of the stock.")
    day_high: Optional[float] = Field(None, description="The highest price during the current trading day.")
    day_low: Optional[float] = Field(None, description="The lowest price during the current trading day.")
    market_cap: Optional[int] = Field(None, description="The market capitalization of the company.")
    currency: Optional[str] = Field(None, description="The trading currency of the stock.")
    open_price: Optional[float] = Field(None, description="The opening price of the stock.")
    previous_close: Optional[float] = Field(None, description="The previous closing price of the stock.")
    volume: Optional[int] = Field(None, description="The trading volume of the stock.")
    error: Optional[str] = Field(None, description="Error message if the ticker is invalid or unavailable.")


class TechnicalIndicatorsInput(BaseModel):
    """
    Input schema for calculating technical indicators (RSI, SMA, EMA).
    """
    ticker_symbol: str = Field(
        ...,
        description="The stock ticker symbol to calculate technical indicators for."
    )


class TechnicalIndicatorsOutput(BaseModel):
    """
    Output schema for technical indicators.
    """
    sma_20: Optional[float] = Field(None, description="20-period Simple Moving Average.")
    sma_50: Optional[float] = Field(None, description="50-period Simple Moving Average.")
    ema_20: Optional[float] = Field(None, description="20-period Exponential Moving Average.")
    rsi_14: Optional[float] = Field(None, description="14-period Relative Strength Index.")
    error: Optional[str] = Field(None, description="Error message if calculation fails.")


class PositionSizingInput(BaseModel):
    """
    Input schema for position sizing and risk calculations.
    """
    ticker_symbol: str = Field(
        ...,
        description="The stock ticker symbol to size the position for."
    )
    portfolio_size: float = Field(
        ...,
        gt=0,
        description="Total portfolio size or allocated capital (e.g., 1000000 INR)."
    )
    risk_percentage: float = Field(
        default=1.0,
        gt=0,
        le=100,
        description="Maximum percentage of portfolio capital willing to risk on this trade (default: 1.0%)."
    )
    stop_loss_pct: float = Field(
        default=5.0,
        gt=0,
        le=100,
        description="Stop-loss distance as a percentage of current stock price (default: 5.0%)."
    )


class PositionSizingOutput(BaseModel):
    """
    Output schema for position sizing recommendations and risk audit.
    """
    ticker_symbol: str = Field(..., description="Ticker symbol.")
    current_price: float = Field(..., description="Current price used for calculations.")
    portfolio_size: float = Field(..., description="Total portfolio capital.")
    risk_percentage: float = Field(..., description="Percentage of portfolio at risk.")
    capital_at_risk: float = Field(..., description="Absolute capital amount at risk (portfolio_size * risk_percentage / 100).")
    stop_loss_pct: float = Field(..., description="Stop loss percentage.")
    stop_loss_price: float = Field(..., description="Calculated stop loss price level.")
    max_shares_to_buy: float = Field(..., description="Maximum recommended number of shares/units to acquire.")
    total_position_value: float = Field(..., description="Total capital required to acquire max_shares_to_buy at current_price.")
    position_pct_of_portfolio: float = Field(..., description="Total position value as a percentage of overall portfolio size.")
    risk_status: str = Field(..., description="Assessment status: 'Approved' if total position value <= portfolio_size else 'Adjusted / Exceeds Capital'.")
    error: Optional[str] = Field(None, description="Error message if calculation fails.")

class RetrievalRequest(BaseModel):
    query: str = Field(...)

class RetrievalResponse(BaseModel):
    context: str = Field(...)