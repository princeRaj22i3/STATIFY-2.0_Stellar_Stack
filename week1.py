from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage,ToolMessage
from tool import get_stock_price,get_company_news
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)
model_with_tools = model.bind_tools([
    get_stock_price,
    get_company_news
])
SYSTEM_PROMPT="""You are FinSight AI, an intelligent Financial Analyst chatbot.

Your primary responsibility is to help users with financial and stock market-related queries by providing accurate, concise, and informative responses.

You have access to two external tools:

1. A stock price tool that retrieves the latest stock market information.
2. A news search tool that retrieves the latest news and market updates about a company.

Instructions:

* Use the stock price tool whenever the user asks about stock prices, market values, trading information, or company performance.
* Use the news search tool whenever the user requests recent news, market sentiment, or current events about a company.
* If both stock information and recent news are relevant, use both tools before generating your final response.
* Present information in a clear and professional manner.
* Summarize the latest news in bullet points whenever appropriate.
* If the requested information cannot be found, politely inform the user instead of making assumptions.
* Do not generate fabricated financial data or news.
* Keep responses factual, objective, and easy to understand."""
chat_history=[
    SystemMessage(content=SYSTEM_PROMPT)
]
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

    # Indian Stocks
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "wipro": "WIPRO.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "lt": "LT.NS",
    "l&t": "LT.NS"
}
print("=" * 60)
print("📈 Welcome to FinSight AI")
print("Type 'exit' to quit.")
print("=" * 60)
def get_response(chat_history):
    return model_with_tools.invoke(chat_history)
while True:
    user_input = input("You: ").strip()
    if not user_input:
      continue
    if user_input.lower() in ["exit", "quit", "bye"]:
       print("AI: Goodbye!")
       break
    chat_history.append(HumanMessage(content=user_input))
    try:
        result=get_response(chat_history)
        if result.tool_calls:
            for tool_call in result.tool_calls:

                if tool_call["name"] == "get_stock_price":
                    
                    args = tool_call["args"]

                    ticker = COMPANY_TO_TICKER.get(
                        args.get("ticker_symbol", "").lower(),
                        args.get("ticker_symbol", "")
                    )

                    tool_result = get_stock_price.invoke(
                        {"ticker_symbol": ticker}
                    )

                elif tool_call["name"] == "get_company_news":
                    tool_result = get_company_news.invoke(tool_call["args"])

                chat_history.append(result)
                chat_history.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

                result = get_response(chat_history)
                
        print("\n📈 FinSight AI:")
        print(result.content)
        print("-"*60)

        chat_history.append(result)

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
