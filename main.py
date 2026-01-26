from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOllama(model="llama3.2", temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)
chain = agent_executor

def main():
    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for an ai engineer in dublin, ireland on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
