from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOllama(model="llama3.2", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 10 bmw 3 series on DoneDeal before the year 2005 and give me details and links to each ad")]})
    print(result)


if __name__ == "__main__":
    main()
