import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main():
    math_server_path = Path(__file__).with_name("mcp_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(math_server_path)],
            },
            "workflow": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
            },
        }
    )

    llm = ChatOllama(model="gemma4:e4b")

    tools = await client.get_tools(server_name="math")
    agent = create_agent(
        model=llm,
        tools=tools,
    )

    workflow = await client.get_prompt("workflow", "workflow")
    result = await agent.ainvoke({"messages": workflow})

    for message in result["messages"]:
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
