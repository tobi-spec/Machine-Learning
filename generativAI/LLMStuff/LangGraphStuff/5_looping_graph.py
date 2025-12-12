from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import random

class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int

def greeting_node(state: AgentState) -> AgentState:
    """This node greets the user"""
    state["name"] = f"Hello {state['name']}!"
    state["counter"] = 0
    return state

def random_node(state: AgentState) -> AgentState:
    """This node adds a random number to the list"""
    state["number"].append(random.randint(0, 10))
    state["counter"] += 1
    return state

def should_continue(state: AgentState) -> str:
    """Decides whether to continue looping or finish"""
    if state["counter"] < 5:
        print("Enter loop ", state["counter"])
        return "loop"
    else:
        return "exit"


graph = StateGraph(AgentState)
graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)

graph.add_edge("greeting", "random")
graph.add_conditional_edges(
    "random",
    should_continue,
    {
        "loop": "random",
        "exit": END
    }
)

graph.set_entry_point("greeting")
app = graph.compile()

print(app.invoke({"name": "Alice", "number": [], "counter": -2}))