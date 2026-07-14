from langchain_core .tools import tool
from ddgs import DDGS
import yfinance as yf
from schema3 import NewsInput
from typing import Dict, Any, Union
from schema3 import (
    StockPriceInput,
    StockPriceOutput,
    TechnicalIndicatorsInput,
    TechnicalIndicatorsOutput,
    PositionSizingInput,
    PositionSizingOutput,
)


POSITIVE_WORDS = [
    "profit",
    "profits",
    "growth",
    "record",
    "approval",
    "surge",
    "contract",
    "launch",
    "launched",
    "beat",
    "beats",
    "investment",
    "invest",
    "expansion",
    "partnership",
    "acquisition",
    "upgrade",
    "award",
    "wins",
    "strong",
]

NEGATIVE_WORDS = [
    "loss",
    "losses",
    "decline",
    "declined",
    "fall",
    "drop",
    "fraud",
    "lawsuit",
    "recall",
    "delay",
    "bankruptcy",
    "downgrade",
    "penalty",
    "investigation",
    "warning",
    "cuts",
]


def get_sentiment(text: str):

    text = text.lower()

    if any(word in text for word in POSITIVE_WORDS):
        return "Bullish"

    if any(word in text for word in NEGATIVE_WORDS):
        return "Bearish"

    return "Neutral"


def fetch_news(company: str, query: str, max_results: int = 5):

    try:
        with DDGS() as ddgs:

            results = list(
                ddgs.text(
                    query,
                    max_results=max_results
                )
            )

        if not results:
            return f"No news found for {company}."

        formatted_news = []

        for i, article in enumerate(results, start=1):

            title = article.get("title", "No Title")

            summary = article.get("body", "No Summary Available")

            source = article.get("source", "DuckDuckGo")

            url = article.get("href", "No URL")

            sentiment = get_sentiment(
                title + " " + summary
            )

            formatted_news.append(
                f"""
==============================

News {i}

Title:
{title}

Summary:
{summary}

Source:
{source}

URL:
{url}

Sentiment:
{sentiment}

==============================
"""
            )

        return "\n".join(formatted_news)

    except Exception as e:
        return f"Error fetching news: {str(e)}"


@tool(args_schema=NewsInput)
def company_news(company: str) -> str:
    """
    Fetch latest company news.
    """
    return fetch_news(
        company,
        f"{company} latest company news"
    )


@tool(args_schema=NewsInput)
def breaking_news(company: str) -> str:
    """
    Fetch breaking news.
    """
    return fetch_news(
        company,
        f"{company} breaking news today"
    )


@tool(args_schema=NewsInput)
def earnings_news(company: str) -> str:
    """
    Fetch earnings related news.
    """
    return fetch_news(
        company,
        f"{company} quarterly earnings OR financial results"
    )


@tool(args_schema=NewsInput)
def ceo_news(company: str) -> str:
    """
    Fetch CEO related news.
    """
    return fetch_news(
        company,
        f"{company} CEO interview OR CEO announcement"
    )


@tool(args_schema=NewsInput)
def government_orders(company: str) -> str:
    """
    Fetch government contract/order news.
    """
    return fetch_news(
        company,
        f"{company} government contract OR government order"
    )


@tool(args_schema=NewsInput)
def product_launches(company: str) -> str:
    """
    Fetch product launch news.
    """
    return fetch_news(
        company,
        f"{company} new product launch"
    )


NEWS_TOOLS = [
    company_news,
    breaking_news,
    earnings_news,
    ceo_news,
    government_orders,
    product_launches,
]

@tool("get_stock_price", args_schema=StockPriceInput)
def get_stock_price(ticker_symbol: str) -> Dict[str, Any]:
    """
    Retrieve real-time stock price and metadata for a given ticker symbol using yfinance.

    Input:
    - ticker_symbol (str): Stock ticker (e.g., AAPL, MSFT, RELIANCE.NS)

    Returns:
    - Dictionary with current price, day high/low, market cap, volume, and currency.
    """
    clean_ticker = ticker_symbol.strip().upper()
    if not clean_ticker:
        return StockPriceOutput(
            error="Ticker symbol cannot be empty."
        ).model_dump()

    try:
        ticker = yf.Ticker(clean_ticker)
        info = ticker.info

        # yfinance might return empty dict or None for invalid tickers
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price is None:
            # Fallback check via history if info dictionary misses currentPrice
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
            else:
                return StockPriceOutput(
                    error=f"Ticker '{clean_ticker}' not found or no market data available."
                ).model_dump()

        details = {
            "symbol": info.get("symbol", clean_ticker),
            "company_name": info.get("longName") or info.get("shortName", clean_ticker),
            "current_price": round(float(current_price), 2),
            "day_high": round(float(info["dayHigh"]), 2) if info.get("dayHigh") else None,
            "day_low": round(float(info["dayLow"]), 2) if info.get("dayLow") else None,
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "open_price": round(float(info["open"]), 2) if info.get("open") else None,
            "previous_close": round(float(info["previousClose"]), 2) if info.get("previousClose") else None,
            "volume": info.get("volume"),
        }
        return StockPriceOutput(**details).model_dump()

    except Exception as e:
        return StockPriceOutput(
            error=f"Failed to fetch stock data for '{clean_ticker}': {str(e)}"
        ).model_dump()


@tool("get_technical_indicators", args_schema=TechnicalIndicatorsInput)
def get_technical_indicators(ticker_symbol: str) -> Dict[str, Any]:
    """
    Calculate core technical indicators (SMA-20, SMA-50, EMA-20, and RSI-14) for a stock ticker.

    Input:
    - ticker_symbol (str): Stock ticker (e.g., AAPL, MSFT)

    Returns:
    - Dictionary containing calculated SMA, EMA, and RSI metrics.
    """
    clean_ticker = ticker_symbol.strip().upper()
    if not clean_ticker:
        return TechnicalIndicatorsOutput(
            error="Ticker symbol cannot be empty."
        ).model_dump()

    try:
        ticker = yf.Ticker(clean_ticker)
        hist = ticker.history(period="6mo")

        if hist.empty or len(hist) < 20:
            return TechnicalIndicatorsOutput(
                error=f"Insufficient historical data found for '{clean_ticker}' (needed at least 20 trading days)."
            ).model_dump()

        close = hist["Close"]

        # Simple Moving Averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()

        # Exponential Moving Average
        ema_20 = close.ewm(span=20, adjust=False).mean()

        # Relative Strength Index (RSI-14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
        rsi = 100.0 - (100.0 / (1.0 + rs))

        output = {
            "sma_20": round(float(sma_20.iloc[-1]), 2),
            "sma_50": round(float(sma_50.iloc[-1]), 2) if len(close) >= 50 else None,
            "ema_20": round(float(ema_20.iloc[-1]), 2),
            "rsi_14": round(float(rsi.iloc[-1]), 2) if len(close) >= 14 else None,
        }
        return TechnicalIndicatorsOutput(**output).model_dump()

    except Exception as e:
        return TechnicalIndicatorsOutput(
            error=f"Failed to calculate indicators for '{clean_ticker}': {str(e)}"
        ).model_dump()


@tool("calculate_position_size", args_schema=PositionSizingInput)
def calculate_position_size(
    ticker_symbol: str,
    portfolio_size: float,
    risk_percentage: float = 1.0,
    stop_loss_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    Calculate optimal position sizing and risk exposure based on portfolio capital and stop-loss criteria.

    Inputs:
    - ticker_symbol (str): Stock ticker
    - portfolio_size (float): Total portfolio capital in currency units (e.g., 1,000,000 INR)
    - risk_percentage (float): Maximum percentage of portfolio to risk (default: 1.0%)
    - stop_loss_pct (float): Stop-loss distance as a percentage of entry price (default: 5.0%)

    Returns:
    - Dictionary with recommended share quantity, total investment allocation, capital at risk, and audit status.
    """
    clean_ticker = ticker_symbol.strip().upper()
    try:
        # Fetch current stock price first
        price_data = get_stock_price.invoke({"ticker_symbol": clean_ticker})
        if price_data.get("error") or not price_data.get("current_price"):
            return PositionSizingOutput(
                ticker_symbol=clean_ticker,
                current_price=0.0,
                portfolio_size=portfolio_size,
                risk_percentage=risk_percentage,
                capital_at_risk=0.0,
                stop_loss_pct=stop_loss_pct,
                stop_loss_price=0.0,
                max_shares_to_buy=0.0,
                total_position_value=0.0,
                position_pct_of_portfolio=0.0,
                risk_status="Error: Could not retrieve current stock price.",
                error=price_data.get("error", "Stock price unavailable.")
            ).model_dump()

        current_price = float(price_data["current_price"])
        if current_price <= 0:
            return PositionSizingOutput(
                ticker_symbol=clean_ticker,
                current_price=current_price,
                portfolio_size=portfolio_size,
                risk_percentage=risk_percentage,
                capital_at_risk=0.0,
                stop_loss_pct=stop_loss_pct,
                stop_loss_price=0.0,
                max_shares_to_buy=0.0,
                total_position_value=0.0,
                position_pct_of_portfolio=0.0,
                risk_status="Error: Invalid stock price (<= 0).",
                error="Invalid stock price."
            ).model_dump()

        # Risk calculations
        capital_at_risk = portfolio_size * (risk_percentage / 100.0)
        stop_loss_price = round(current_price * (1.0 - stop_loss_pct / 100.0), 2)
        risk_per_share = current_price - stop_loss_price

        if risk_per_share <= 0:
            risk_per_share = current_price * 0.05  # fallback 5%

        # Recommended maximum shares to acquire based on acceptable capital loss
        max_shares_to_buy = round(capital_at_risk / risk_per_share, 2)
        total_position_value = round(max_shares_to_buy * current_price, 2)

        # Cap position at 100% of portfolio size if risk formula suggests oversized leverage
        if total_position_value > portfolio_size:
            max_shares_to_buy = round(portfolio_size / current_price, 2)
            total_position_value = round(max_shares_to_buy * current_price, 2)
            risk_status = "Adjusted - Position capped at 100% of portfolio capital."
        else:
            risk_status = "Approved - Within acceptable risk and portfolio sizing parameters."

        position_pct_of_portfolio = round((total_position_value / portfolio_size) * 100.0, 2)

        result = {
            "ticker_symbol": clean_ticker,
            "current_price": current_price,
            "portfolio_size": portfolio_size,
            "risk_percentage": risk_percentage,
            "capital_at_risk": round(capital_at_risk, 2),
            "stop_loss_pct": stop_loss_pct,
            "stop_loss_price": stop_loss_price,
            "max_shares_to_buy": max_shares_to_buy,
            "total_position_value": total_position_value,
            "position_pct_of_portfolio": position_pct_of_portfolio,
            "risk_status": risk_status,
        }
        return PositionSizingOutput(**result).model_dump()

    except Exception as e:
        return PositionSizingOutput(
            ticker_symbol=clean_ticker,
            current_price=0.0,
            portfolio_size=portfolio_size,
            risk_percentage=risk_percentage,
            capital_at_risk=0.0,
            stop_loss_pct=stop_loss_pct,
            stop_loss_price=0.0,
            max_shares_to_buy=0.0,
            total_position_value=0.0,
            position_pct_of_portfolio=0.0,
            risk_status="Error encountered during position sizing calculation.",
            error=str(e)
        ).model_dump()
    

def interpret_rsi(rsi: float) -> str:
    """
    Convert RSI value into a query
    suitable for the RAG retriever.
    """
    
    if rsi >= 70:
        return "RSI above 70"

    elif rsi <= 30:
        return "RSI below 30"

    return "RSI between 30 and 70"