from typing import TypedDict, List, Union

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="mistral")

def process(state: AgentState) -> AgentState:
    """Process the message using LLM"""
    response = llm.invoke(state["message"])
    state["message"].append(AIMessage(response.content))
    print(f"LLM Response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    response_state = agent.invoke({"message": conversation_history})
    conversation_history = response_state["message"]
    user_input = input("Enter: ")

print("Exiting...")