# ==========================================================
# IMPORTS
# ==========================================================
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from schema2 import (
    AgentState,
    DataFetcherOutput,
    AnalystOutput,
    RiskAuditorOutput,
)
from tools2 import (
    initialize_rag,
    BASE_DIR,
    get_stock_price,
    get_technical_indicators,
    company_news,
    breaking_news,
    earnings_news,
    government_orders,
    retrieve_context,
    interpret_rsi,
)
from memory import (
    initialize_database,
    save_chat,
    load_chat_history,
)

# ==========================================================
# INITIALIZATION
# ==========================================================
PDF_PATH = (
    BASE_DIR
    / "data"
    / "Module 2_Technical Analysis.pdf"
)
initialize_database()

try:
    initialize_rag(PDF_PATH)
except Exception as e:
    print(e)
    exit()

# ==========================================================
# NODE-1 : DATA FETCHER AGENT
# ==========================================================
def data_fetcher_node(state: AgentState):
   
    """
    Fetch

    • Stock Information

    • Technical Indicators

    • Latest Company News
    """
    ticker = state["ticker"]

    stock_data = get_stock_price.invoke(
        {
            "ticker_symbol": ticker
        }
    )

    technical_data = get_technical_indicators.invoke(
        {
        "ticker_symbol": ticker
        }
    )
    if state.get("retry_count",0)==0:

        news = company_news.invoke({"company": ticker})

    else:

        news = "\n".join([

            company_news.invoke({"company": ticker}),

            breaking_news.invoke({"company": ticker}),

            earnings_news.invoke({"company": ticker}),

            government_orders.invoke({"company": ticker})

        ])
  

    output = DataFetcherOutput(

        stock_data=stock_data,

        technical_data=technical_data,

        news=news

    )

    print("\n========== DATA FETCHER ==========")
    print("Stock Data Retrieved")
    print("Technical Indicators Calculated")
    print("Latest News Retrieved")
    print("==================================")

    return output.model_dump()

# ==========================================================
# NODE-2 : TECHNICAL ANALYST AGENT
# ==========================================================
def technical_analyst_node(state: AgentState):
    
    """
    Node-2:
    Uses RAG to interpret RSI and generate
    a preliminary recommendation.
    """

    print("\n========== TECHNICAL ANALYST ==========\n")

    technical = state["technical_data"]

    if technical.get("error"):
        return AnalystOutput(
            recommendation="Unable to analyze.",
            rag_context=""
        ).model_dump()

    rsi = technical["rsi_14"]

    rag_query = interpret_rsi(rsi)

    rag_context = retrieve_context.invoke(
        {"query": rag_query}
    )

    if rsi >= 70:
        recommendation = "SELL"

    elif rsi <= 30:
        recommendation = "BUY"

    else:
        recommendation = "HOLD"

    return AnalystOutput(
        recommendation=recommendation,
        rag_context=rag_context
    ).model_dump()
# ==========================================================
# NODE-3 : RISK AUDITOR AGENT
# ==========================================================
def risk_auditor_node(state: AgentState):
   
    """
    Node-3

    Compare technical recommendation
    with breaking news.
    """
    print("\n========== RISK AUDITOR ==========\n")

    recommendation = state["recommendation"]

    news = (state.get("news") or "").lower()

    retry = state.get("retry_count", 0)

    contradiction = False

    reason = None

    bullish_words = [
        "contract",
        "approval",
        "acquisition",
        "record",
        "investment",
        "profit",
        "partnership",
        "surge"
    ]

    bearish_words = [
        "fraud",
        "bankruptcy",
        "loss",
        "penalty",
        "lawsuit",
        "recall",
        "downgrade"
    ]

    if recommendation == "SELL":

        if any(word in news for word in bullish_words):

            contradiction = True

            reason = (
                "Technical analysis suggests SELL "
                "but breaking news is strongly bullish."
            )

    elif recommendation == "BUY":

        if any(word in news for word in bearish_words):

            contradiction = True

            reason = (
                "Technical analysis suggests BUY "
                "but breaking news is strongly bearish."
            )

    if contradiction:

        return RiskAuditorOutput(

            needs_more_data=True,

            contradiction_reason=reason,

            retry_count=retry + 1,

            final_report=None

        ).model_dump()
    report = f"""
Ticker :
{state['ticker']}

Current Price :
{state['stock_data']['current_price']}

RSI :
{state['technical_data']['rsi_14']}

SMA 20 :
{state['technical_data']['sma_20']}

EMA 20 :
{state['technical_data']['ema_20']}

Recommendation :
{recommendation}

Technical Reason :
{state['rag_context']}

News Summary :

{state['news']}
"""

    save_chat(state, report)

    return RiskAuditorOutput(

        needs_more_data=False,

        contradiction_reason=None,

        retry_count=retry,

        final_report=report

    ).model_dump()

    
# ==========================================================
# CONDITIONAL ROUTING
# ==========================================================
def should_loop(state):

    if (
       state["needs_more_data"]
       and state["retry_count"] < 2
    ):

        print("\nContradiction Found")

        print(state["contradiction_reason"])

        print("\nFetching additional data...\n")

        return "loop"

    return "done"
# ==========================================================
# BUILD LANGGRAPH WORKFLOW
# ==========================================================
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("data_fetcher", data_fetcher_node)
graph.add_node("technical_analyst", technical_analyst_node)
graph.add_node("risk_auditor", risk_auditor_node)

# Add normal edges
graph.set_entry_point("data_fetcher")
graph.add_edge("data_fetcher", "technical_analyst")
graph.add_edge("technical_analyst", "risk_auditor")

# Add CONDITIONAL edge (the loop)
graph.add_conditional_edges(
    "risk_auditor",          
    should_loop,             
    {
        "loop": "data_fetcher",   
        "done": END               
    }
)

app = graph.compile()

# ==========================================================
# USER INTERFACE (CLI)
# ==========================================================
if __name__ == "__main__":

    print("=" * 60)
    print("      FinSight AI - Multi Agent Financial Analyst")
    print("=" * 60)

    while True:

        query = input(
            "Enter query (Example: Analyze RELIANCE.NS): "
        ).strip()

        if query.lower() == "exit":
            print("\n Thank you for using FinSight AI")
            break

        COMPANY_TO_TICKER = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "tesla": "TSLA",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "netflix": "NFLX",
            "intel": "INTC",
            "amd": "AMD",
            "oracle": "ORCL",
            "ibm": "IBM",
            "tcs": "TCS.NS",
            "infosys": "INFY.NS",
            "reliance": "RELIANCE.NS",
            "hdfc": "HDFCBANK.NS",
            "sbi": "SBIN.NS",
            "wipro": "WIPRO.NS",
        }

        words = [w.strip("?,.:;!\"'()") for w in query.split()]

        ticker = None
        # 1. Check company name mapping
        for word in words:
            if word.lower() in COMPANY_TO_TICKER:
                ticker = COMPANY_TO_TICKER[word.lower()]
                break

        # 2. Check for ticker with a dot (e.g., RELIANCE.NS) or uppercase ticker (e.g., AAPL)
        if not ticker:
            for word in words:
                if "." in word:
                    ticker = word
                    break
                elif word.isupper() and word.isalpha() and 1 <= len(word) <= 5:
                    ticker = word
                    break

        # 3. Fallback: if query is a single word, try using it directly
        if not ticker and len(words) == 1:
            ticker = words[0]

        if not ticker:
            print("Invalid query. Could not identify a ticker symbol or company name.")
            continue

        if not query:
            print("Query cannot be empty.")
            continue

        print("\nLoading previous chat history...\n")

        history = load_chat_history(ticker)

        if history:

            print("Previous Conversations")
            print("-" * 60)

            for row in history:

                print(f"Date : {row['created_at']}")
                print(f"Query : {row['user_query']}")
                print(f"Recommendation : {row['recommendation']}")
                print("-" * 60)

        else:
            print("No previous history found.")

        initial_state = {

            "ticker": ticker,

            "user_query": query,

            "chat_history": history,

            "stock_data": None,

            "technical_data": None,

            "news": None,

            "rag_context": None,

            "recommendation": None,

            "needs_more_data": False,

            "contradiction_reason": None,

            "final_report": None,

            "retry_count": 0

        }

        print("\nRunning Multi-Agent Workflow...\n")

        result = app.invoke(initial_state)

        print("\n" + "=" * 70)
        print("FINAL ANALYSIS REPORT")
        print("=" * 70)

        print(result["final_report"])

        print("=" * 70)