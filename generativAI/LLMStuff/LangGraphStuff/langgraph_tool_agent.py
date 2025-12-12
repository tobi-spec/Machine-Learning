from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import chat_agent_executor

llm = OllamaLLM(model="mistral")

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y


agent = chat_agent_executor.create_tool_calling_executor(
    model=llm,
    tools=[multiply]
)

graph_builder = StateGraph()
graph_builder.add_node("agent", agent)

graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()

response = graph.invoke(
    {"message": [HumanMessage(content="What is 12 multiplied by 8?")]},
    config=RunnableConfig()
)

print(response)
