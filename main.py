import os
import re
import json
from typing import TypedDict, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools3 import (
    company_news,
    breaking_news,
    earnings_news,
    government_orders,
    get_stock_price,
    get_technical_indicators,
    calculate_position_size,
    interpret_rsi,
)

from memory import (
    initialize_database,
    save_chat,
    load_chat_history,
)
from dotenv import load_dotenv

load_dotenv()
# ==========================================================
# FASTAPI INITIALIZATION
# ==========================================================

app = FastAPI(

    title="STATIFY 2.0 API",

    version="3.0",

    description="Production Financial Analysis API"

)

initialize_database()

# ==========================================================
# MCP CLIENT
# ==========================================================

MAX_RETRIES = 2

MCP_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://localhost:8001/mcp"
)

mcp_client = MultiServerMCPClient(
    {
        "vector-db": {
            "transport": "streamable_http",
            "url": MCP_URL,
        }
    },
    handle_tool_errors=False,
)

# ==========================================================
# LLM CONFIGURATION
# ==========================================================
def get_llm() -> ChatHuggingFace:

    endpoint = HuggingFaceEndpoint(

        repo_id=os.getenv("HF_MODEL_REPO_ID", "Qwen/Qwen2.5-7B-Instruct"),

        task="text-generation",

        max_new_tokens=1024,

        temperature=0.2,

        huggingfacehub_api_token=os.getenv("HF_TOKEN"),

    )

    return ChatHuggingFace(llm=endpoint)


llm = get_llm()


class RecommendationDecision(BaseModel):

    action: str = Field(..., description="BUY, SELL, or HOLD")

    confidence: str = Field(..., description="High, Medium, or Low")

    reasoning: str = Field(..., description="Short justification for the call")

    needs_more_data: bool = Field(default=False)

    contradiction_reason: Optional[str] = None


def parse_llm_json(raw_text: str) -> dict:
    """
    Manually extract and parse a JSON object from an LLM's raw text
    output. Strips markdown code fences and pulls out the first
    {...} block - same workaround pattern used earlier in the
    project for ChatHuggingFace's unreliable structured output.
    """

    cleaned = raw_text.strip()

    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in LLM response.")

    return json.loads(match.group(0))


# ==========================================================
# REQUEST / RESPONSE SCHEMA
# ==========================================================

class AnalyzeRequest(BaseModel):

    ticker: str = Field(

        ...,

        example="RELIANCE.NS"

    )

    position_size: float = Field(

        ...,

        example=1000000

    )


class AnalyzeResponse(BaseModel):

    ticker: str

    status: str

    message: str

    result: Dict[str, Any]


# ==========================================================
# LANGGRAPH STATE
# ==========================================================

class AgentState(TypedDict):

    ticker: str

    position_size: float

    user_query: str

    chat_history: list

    stock_data: Optional[dict]

    technical_data: Optional[dict]

    position_data: Optional[dict]

    news: Optional[str]

    rag_context: Optional[str]

    recommendation: Optional[str]

    needs_more_data: bool

    contradiction_reason: Optional[str]

    retry_count: int

    final_report: Optional[str]


# ==========================================================
# NODE - 1
# DATA FETCHER
# ==========================================================

def data_fetcher_node(state: AgentState):

    ticker = state["ticker"]

    try:
     stock_data = get_stock_price.invoke(
        {"ticker_symbol": ticker}
     )

     technical_data = get_technical_indicators.invoke(
        {"ticker_symbol": ticker}
     )

     position_data = calculate_position_size.invoke(
        {
            "ticker_symbol": ticker,
            "portfolio_size": state["position_size"]
        }
     )

    except Exception as e:
     raise RuntimeError(
        f"Data Fetcher failed: {e}"
     )

    if state["retry_count"] == 0:

        news = company_news.invoke(

            {

                "company": ticker

            }

        )

    else:

        news = "\n".join(

            [

                company_news.invoke(

                    {

                        "company": ticker

                    }

                ),

                breaking_news.invoke(

                    {

                        "company": ticker

                    }

                ),

                earnings_news.invoke(

                    {

                        "company": ticker

                    }

                ),

                government_orders.invoke(

                    {

                        "company": ticker

                    }

                )

            ]

        )

    return {

        "stock_data": stock_data,

        "technical_data": technical_data,

        "position_data": position_data,

        "news": news

    }


# ==========================================================
# NODE - 2
# RAG RETRIEVAL (MCP CLIENT)
# ==========================================================

async def rag_retrieval_node(state: AgentState):
    """
    Dynamically loads tools from the standalone vector-db MCP server
    via MultiServerMCPClient and calls `retrieve_context` for the
    trading theory relevant to the current RSI regime.

    MultiServerMCPClient is stateless by default - get_tools() opens
    a fresh session, runs the call, and tears it down. Any failure
    (transport failure, or an MCP execution error since we set
    handle_tool_errors=False above) is caught here and surfaced to
    the agent as an error ToolMessage instead of crashing the graph.
    """

    technical_data = state.get("technical_data") or {}

    rsi = technical_data.get("rsi_14")

    query = interpret_rsi(rsi) if rsi is not None else "SMA and EMA trend-following theory"

    chat_history = list(state.get("chat_history") or [])

    try:

        tools = await mcp_client.get_tools()

        retrieve_tool = next(

            (t for t in tools if t.name == "retrieve_context"),

            None

        )

        if retrieve_tool is None:
            raise RuntimeError("'retrieve_context' tool was not exposed by the vector-db MCP server.")

        rag_context = await retrieve_tool.ainvoke({"query": query})

        chat_history.append(

            ToolMessage(

                content=str(rag_context),

                tool_call_id="retrieve_context",

                status="success",

            )

        )

    except Exception as e:

        rag_context = ""

        chat_history.append(

            ToolMessage(

                content=f"MCP retrieval error: {str(e)}",

                tool_call_id="retrieve_context",

                status="error",

            )

        )

    return {

        "rag_context": rag_context,

        "chat_history": chat_history,

    }


# ==========================================================
# NODE - 3
# RECOMMENDATION / CONTRADICTION CHECK
# ==========================================================

async def recommendation_node(state: AgentState):

    ticker = state["ticker"]

    retry_count = state.get("retry_count", 0)

    prompt = f"""You are a disciplined quantitative equity analyst for the Statify desk.

Ticker: {ticker}
Stock data: {state.get("stock_data")}
Technical indicators: {state.get("technical_data")}
Position sizing / risk audit: {state.get("position_data")}
Recent news: {state.get("news")}
Trading theory context retrieved from the knowledge base: {state.get("rag_context")}

This is attempt {min(retry_count + 1, MAX_RETRIES)} of {MAX_RETRIES + 1}.

Your decision must consider:

1. Current stock price

2. RSI

3. SMA

4. EMA

5. Position sizing

6. News sentiment

7. Retrieved RAG context

If the technical indicators strongly contradict the news sentiment,
set needs_more_data=true.
 Respond with ONLY a raw JSON object (no markdown fences, no extra text) with exactly these keys:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "2-4 sentences tying together price action, technicals, news sentiment and the trading theory context",
  "needs_more_data": true | false,
  "contradiction_reason": "short string describing the contradiction, or null"
}}

Set "needs_more_data" to true only if the technicals, price action and news sentiment meaningfully disagree with each other and this is not already the final attempt."""

    response = await llm.ainvoke(prompt)

    raw_text = getattr(response, "content", str(response))

    try:

        decision = RecommendationDecision(**parse_llm_json(raw_text))

    except (ValueError, ValidationError, json.JSONDecodeError) as e:

        decision = RecommendationDecision(

            action="HOLD",

            confidence="Low",

            reasoning=f"Model output could not be parsed reliably ({str(e)}); defaulting to a conservative HOLD.",

            needs_more_data=False,

            contradiction_reason=None,

        )

    should_retry = decision.needs_more_data and retry_count < MAX_RETRIES

    recommendation_text = (

        f"Action: {decision.action} | Confidence: {decision.confidence}\n"

        f"Reasoning: {decision.reasoning}"

    )

    return {

        "recommendation": recommendation_text,

        "needs_more_data": should_retry,

        "contradiction_reason": decision.contradiction_reason,

        "retry_count": min(retry_count + 1, MAX_RETRIES) if should_retry else retry_count,

    }


def retry_router(state: AgentState) -> str:

    return "retry" if state.get("needs_more_data") else "finalize"


# ==========================================================
# NODE - 4
# FINAL REPORT
# ==========================================================

async def final_report_node(state: AgentState):

    ticker = state["ticker"]

    prompt = f"""Write the final trading thesis report for {ticker} based on the analysis below.
Use short headed sections: Summary, Technical View, Risk & Position Sizing, Final Recommendation.
Plain text only, no markdown fences.

Recommendation: {state.get("recommendation")}
Stock data: {state.get("stock_data")}
Technical indicators: {state.get("technical_data")}
Position sizing: {state.get("position_data")}
Trading theory context: {state.get("rag_context")}
Contradiction notes (if any): {state.get("contradiction_reason")}"""

    response = await llm.ainvoke(prompt)

    final_report = getattr(response, "content", str(response))

    try:

        save_chat(state, final_report)

    except Exception as e:

        print(f"Warning: could not persist chat history for {ticker}: {e}")

    return {"final_report": final_report}


# ==========================================================
# GRAPH ASSEMBLY
# ==========================================================

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("data_fetcher", data_fetcher_node)

    graph.add_node("rag_retrieval", rag_retrieval_node)

    graph.add_node("recommend", recommendation_node)

    graph.add_node("final_report", final_report_node)

    graph.set_entry_point("data_fetcher")

    graph.add_edge("data_fetcher", "rag_retrieval")

    graph.add_edge("rag_retrieval", "recommend")

    graph.add_conditional_edges(

        "recommend",

        retry_router,

        {

            "retry": "data_fetcher",

            "finalize": "final_report",

        },

    )

    graph.add_edge("final_report", END)

    return graph.compile()


agent_graph = build_graph()


# ==========================================================
# API ROUTE
# ==========================================================

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):

    ticker = request.ticker.strip().upper()

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    if request.position_size <= 0:
        raise HTTPException(status_code=400, detail="position_size must be greater than 0.")

    past_history = load_chat_history(ticker)

    initial_state: AgentState = {

        "ticker": ticker,

        "position_size": request.position_size,

        "user_query": f"Provide a trading thesis for {ticker}.",

        "chat_history": [],

        "stock_data": None,

        "technical_data": None,

        "position_data": None,

        "news": None,

        "rag_context": None,

        "recommendation": None,

        "needs_more_data": False,

        "contradiction_reason": None,

        "retry_count": 0,

        "final_report": None,

    }

    try:

        final_state = await agent_graph.ainvoke(initial_state)

    except Exception as e:

        raise HTTPException(
          status_code=500,
          detail={
              "message": "Agent execution failed.",
              "error": str(e)
          }
        )

    return AnalyzeResponse(

        ticker=ticker,

        status="success",

        message="Trading thesis generated successfully.",

        result={

            "stock_data": final_state.get("stock_data"),

            "technical_data": final_state.get("technical_data"),

            "position_data": final_state.get("position_data"),

            "recommendation": final_state.get("recommendation"),

            "contradiction_reason": final_state.get("contradiction_reason"),

            "retry_count": final_state.get("retry_count"),

            "final_report": final_state.get("final_report"),

            "past_analyses_on_file": len(past_history),

        },

    )


