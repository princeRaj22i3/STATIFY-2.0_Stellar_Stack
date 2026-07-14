from pydantic import BaseModel, Field
from typing import Optional, List

class StockPriceInput(BaseModel):
    ticker_symbol: str = Field(
        ..., 
        description="The stock ticker symbol to look up (e.g., AAPL, MSFT, TSLA)."
    )

class StockPriceOutput(BaseModel):
    symbol: Optional[str] = Field(None, description="The ticker symbol of the company")
    company_name: Optional[str] = Field(None, description="The official name of the company")
    current_price: Optional[float] = Field(None, description="The current trading price of the stock")
    day_high: Optional[float] = Field(None, description="The highest price of the stock during the current trading day")
    day_low: Optional[float] = Field(None, description="The lowest price of the stock during the current trading day")
    market_cap: Optional[int] = Field(None, description="The market capitalization of the company")
    currency: Optional[str] = Field(None, description="The trading currency of the stock")
    open_price: Optional[float] = Field(None, description="The opening price of the stock for the current trading day") 
    previous_close: Optional[float] = Field(None, description="The previous closing price of the stock")
    volume: Optional[int] = Field(None, description="The trading volume of the stock")
    error: Optional[str] = Field(None, description="Error message if the ticker is not found or invalid")

class NewsInput(BaseModel):
    company: str = Field(description="Company name to search news for")


class NewsArticle(BaseModel):
    title: str = Field(description="Title of the news article")
    summary: str = Field(description="Brief summary of the article")
    link: str = Field(description="URL of the article")


class NewsResponse(BaseModel):
    company: str = Field(description="Company name")
    articles: List[NewsArticle] = Field(description="List of latest news articles related to company")

