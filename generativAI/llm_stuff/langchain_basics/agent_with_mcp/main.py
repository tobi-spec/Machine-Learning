import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main():

    client =  MultiServerMCPClient({
        "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["C:/Users/tobias.weiland/projects/Machine-Learning/generativAI/llm_stuff/langchain_basics/agent_with_mcp/mcp_server.py"],
        }
    })

    llm = ChatOllama(model="gemma4:e4b")

    tools = await client.get_tools()
    agent = create_agent(
        model=llm,
        tools=tools,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 3+3 multiplied with 2?"}]},
    )

    for message in result["messages"]:
        print(message.content)


if __name__ == "__main__":
    asyncio.run(main())
