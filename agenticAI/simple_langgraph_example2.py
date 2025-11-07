from typing import TypedDict, Annotated, Literal

from langchain_ollama import OllamaLLM
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel, Field

llm = OllamaLLM(model="mistral")


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires a emotional (therapist) or logical response")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None


def classify_message(state: State) -> State:
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": "Classify the following message as either 'emotional' or 'logical'."
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State) -> State:
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}


def therapist_agent(state: State) -> State:
    last_message = state["messages"][-1]

    messages = [
        {"role": "system", "content": "You are a compassionate therapist. Provide emotional support."},
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State) -> State:
    last_message = state["messages"][-1]

    messages = [
        {"role": "system", "content": "You are a logical assistant. Provide clear and rational responses."},
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router",
                                    lambda state: state.get("next"),
                                    {"therapist": "therapist", "logical": "logical"})
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None, "next": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [{
            "role": "user",
            "content": user_input
        }]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            print("Bot:", state["messages"][-1]["content"])

if __name__ == "__main__":
    run_chatbot()

