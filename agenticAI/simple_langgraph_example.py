from typing import TypedDict, Annotated, Literal

from langchain_ollama import OllamaLLM
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field

llm = OllamaLLM(model="mistral")

class State(TypedDict):
    message: Annotated[list, add_messages]
    message_type: str | None


def chatbot(state: State) -> State:
    return {"message": [llm.invoke(state["message"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("Enter a message: ")

state = graph.invoke({"message": [{"role": "user", "content": user_input}]})

print(state)