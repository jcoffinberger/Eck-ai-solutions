from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"[Simulated web search for: {query}] – Connect Tavily in production for real results."

@tool
def calculator(expression: str) -> str:
    """Perform safe math calculations."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except:
        return "Calculation error."

def get_agent_executor():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [web_search, calculator]
    agent = create_react_agent(llm, tools)
    return agent
