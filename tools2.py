# import commands 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from ddgs import DDGS
from schema2 import NewsInput, RetrievalRequest, RetrievalResponse
import yfinance as yf
from schema import StockPriceInput, StockPriceOutput

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#--------------------------------------------------
#  Constants
#    • POSITIVE_WORDS
#    • NEGATIVE_WORDS
POSITIVE_WORDS = [
    "profit",
    "growth",
    "record",
    "approval",
    "surge",
    "contract",
    "launch",
    "beat",
    "investment",
    "expansion",
    "partnership",
    "acquisition",
    "upgrade",
]

NEGATIVE_WORDS = [
    "loss",
    "decline",
    "fall",
    "fraud",
    "lawsuit",
    "recall",
    "delay",
    "bankruptcy",
    "downgrade",
    "penalty",
    "investigation",
]

#--------------------------------------------------
#  News Utility Functions
#    get_sentiment()
#    fetch_news()
def get_sentiment(text: str)->str:

    text = text.lower()

    if any(word in text for word in POSITIVE_WORDS):
        return "Bullish"

    if any(word in text for word in NEGATIVE_WORDS):
        return "Bearish"

    return "Neutral"


def fetch_news(company: str, query: str, max_results: int = 5)->str:

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

            sentiment = get_sentiment(title + " " + summary)

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


#--------------------------------------------------
# News tools(Node-1)
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
    Fetch government contracts/orders.
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

#--------------------------------------------------
#Stock Data tools (Node-1)
@tool("get_stock_price", args_schema=StockPriceInput)
def get_stock_price(ticker_symbol: str) -> dict[str,object]:
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

            "currency": info.get('currency'),
            "open_price" : info.get('open'),
            "previous_close" : info.get('previousClose'),   
            "volume" : info.get('volume')
        }
        validated_details = StockPriceOutput(**details)
        return validated_details.model_dump()
    except Exception as e:
        return StockPriceOutput(
           error=f"Failed to fetch stock data: {str(e)}"
        ).model_dump()

#Technical Indicator
@tool(args_schema=StockPriceInput)
def get_technical_indicators(ticker_symbol: str) -> dict[str, float] | dict[str, str]:
    """
    Calculate technical indicators (RSI, SMA, EMA) for a given ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            return {"error": "No historical data found"}
        
        close = hist['Close']
        
       
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        
     
        ema_20 = close.ewm(span=20, adjust=False).mean()
        
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain/avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return {
            "sma_20": round(float(sma_20.iloc[-1]), 2),
            "sma_50": round(float(sma_50.iloc[-1]), 2),
            "ema_20": round(float(ema_20.iloc[-1]), 2),
            "rsi_14": round(float(rsi.iloc[-1]), 2),
        }
    except Exception as e:
        return {"error": f"Failed to calculate indicators: {str(e)}"}
    
#--------------------------------------------------
#RAG Utilities
retriever = None
vector_db = None
embedding_model = None

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

def load_pdf(pdf_path: str | Path)->list:
    """Load the given pdf"""
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()

def split_documents(documents: list)->list:
    """Split the loaded pdf so, that it can be meaningfully embedded into respective vectors"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)

def create_embedding_model()->HuggingFaceEmbeddings:
    """
    Create and return the HuggingFace embedding model
    used for vector generation.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_database(chunks, embedding_model)->Chroma:

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
      vector_db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model
      )
    else:
      vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR)
      )
    return vector_db

def initialize_rag(pdf_path:str|Path, k:int=3)->None:
    """
    Load the PDF, create embeddings,
    initialize the Chroma vector database,
    and create the retriever.
    """
    global retriever
    global vector_db
    global embedding_model

    if retriever is not None:
        return
    
    try:
        documents = load_pdf(pdf_path)
    except Exception as e:
       raise RuntimeError(
        f"Failed to initialize RAG: {e}"
    )

    chunks = split_documents(documents)

    embedding_model = create_embedding_model()

    vector_db = create_vector_database(
        chunks,
        embedding_model
    )

    retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": k}
    )

    
#--------------------------------------------------
#RAG Tools(Node-2)
@tool("retrieve_context", args_schema=RetrievalRequest)
def retrieve_context(query: str) -> str:
    """
    Retrieve relevant technical analysis context from the Chroma vector database.
    """

    if retriever is None:
        raise RuntimeError(
            "RAG has not been initialized."
        )

    docs = retriever.invoke(query)

    context= "\n\n".join(
        doc.page_content
        for doc in docs
    )
    return RetrievalResponse(
       context=context
    ).model_dump()["context"]



#--------------------------------------------------
#Technical Interpretation Helper
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