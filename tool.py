import yfinance as yf
from langchain_core.tools import tool
from schema import StockPriceInput, StockPriceOutput, NewsInput
from ddgs import DDGS

@tool("get_stock_price", args_schema=StockPriceInput)
def get_stock_price(ticker_symbol: str) -> dict:
    """
    Retrieve real-time stock information for a given stock ticker symbol.

   Input:
   -ticker_symbol (str): Stock ticker (e.g., AAPL, MSFT, TSLA)

   Returns:
        Dictionary containing company name, stock price,
        day high, day low, market capitalization,
        and currency information.
    """
    if not ticker_symbol.strip():
          return StockPriceOutput(
              error="Ticker symbol cannot be empty."
          ).model_dump()
    try:
        
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if info.get('currentPrice') is None:
            return StockPriceOutput(
               error=f"Ticker '{ticker_symbol}' not found."
            ).model_dump()
        
        details = {
            "symbol": info.get('symbol'),
            "company_name": info.get('longName'),
            "current_price": info.get('currentPrice'),
            "day_high": info.get('dayHigh'),
            "day_low": info.get('dayLow'),
            "market_cap": info.get('marketCap'),
            "financial_currency": info.get('financialCurrency'),
            "currency": info.get('currency')
        }
        # Validate output schema
        validated_details = StockPriceOutput(**details)
        return validated_details.model_dump()
    except Exception as e:
        return StockPriceOutput(
           error=f"Failed to fetch stock data: {str(e)}"
        ).model_dump()



@tool("get_company_news",args_schema=NewsInput)
def get_company_news(company: str) -> str:
    """
    Fetch the latest news articles for a given company using DuckDuckGo Search.
    """
    if not company.strip():
        return "Company name cannot be empty."
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    f"{company} latest stock news",
                    max_results=3
                )
            )

        if not results:
            return f"No recent news found for {company}."

        formatted_news = []

        for i, article in enumerate(results, start=1):

            title = article.get("title", "No Title")
            summary = article.get("body", "No Summary Available")
            link = article.get("href", "No Link")

            formatted_news.append(
                f"News {i}\n"
                f"Title: {title}\n"
                f"Summary: {summary}\n"
                f"Link: {link}\n"
            )

        return "\n".join(formatted_news)

    except Exception as e:
        return f"Error fetching news: {str(e)}"
