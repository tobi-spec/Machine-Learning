from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    message: HumanMessage

llm = ChatOllama(model="mistral")

def process(state: AgentState) -> AgentState:
    """Process the message using LLM"""
    response = llm.invoke([state["message"]])
    print(f"LLM Response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")

while user_input != "exit":
    agent.invoke({"message": HumanMessage(content="Hi")})
    user_input = input("Enter: ")

print("Exiting...")