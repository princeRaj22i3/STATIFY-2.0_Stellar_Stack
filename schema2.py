from pydantic import BaseModel, Field
from typing import List, Literal,Optional,Dict,Any,TypedDict


class NewsInput(BaseModel):
    company: str = Field(description="Company name to search news for")


class NewsArticle(BaseModel):
    title: str = Field(description="Title of the news article")
    summary: str = Field(description="Brief summary of the article")
    source: str = Field(description="Source of the news article")
    url: str = Field(description="URL of the article")
    sentiment: Literal["Bullish", "Bearish", "Neutral"] = Field(
        description="Sentiment of the article"
    )


class NewsResponse(BaseModel):
    company: str = Field(description="Company name")
    category: str = Field(description="News category")
    articles: List[NewsArticle] = Field(
        description="List of latest news articles"
    )



class RetrievalRequest(BaseModel):
    query: str = Field(
        ...,
        description="Query to retrieve relevant technical analysis context"
    )


class RetrievalResponse(BaseModel):
    context: str = Field(
        ...,
        description="Retrieved context from the Chroma vector database"
    )




class DataFetcherOutput(BaseModel):
    stock_data: Optional[Dict[str, Any]] = Field(None, description="Real-time stock price data from yfinance")
    technical_data: Optional[Dict[str, Any]] = Field(None, description="Calculated technical indicators (RSI, SMA, EMA)")
    news: Optional[str] = Field(None, description="Formatted company news articles from search")

class AnalystOutput(BaseModel):
    recommendation: Optional[str] = Field(None, description="Buy/Sell/Hold recommendation with technical explanation")
    rag_context: Optional[str] = Field(None, description="RAG retrieved context used for analysis validation")

class RiskAuditorOutput(BaseModel):
    needs_more_data: bool = Field(default=False, description="Routing flag indicating if more data is required to resolve contradictions")
    contradiction_reason: Optional[str] = Field(None, description="Explanation of contradictions found during audit")
    final_report: Optional[str] = Field(None, description="Final approved research/audit report")
    retry_count: int = Field(default=0, description="Counter tracking the retry attempts to avoid infinite loops")

# Agent State
class AgentState(TypedDict):

    # Input
    ticker: str
    user_query: str
    chat_history: Optional[list]
    # Node 1
    stock_data: Optional[Dict[str, Any]]
    technical_data: Optional[Dict[str, Any]]
    news: Optional[str]

    # Node 2
    rag_context: Optional[str]
    recommendation: Optional[str]

    # Node 3
    needs_more_data: bool
    contradiction_reason: Optional[str]
    final_report: Optional[str]

    # Loop control
    retry_count: int