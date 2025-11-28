from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    values: List[int]
    name: str
    results: str


def process_values(state: AgentState) -> AgentState:
    """This function handles multiple different inputs"""
    state["results"] = f"Hi there {state["name"]}!, the sum is {sum(state["values"])}"
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process_values)
graph.set_entry_point("process")
graph.set_finish_point("process")

app = graph.compile()
answer = app.invoke({"values": [10, 20, 30], "name": "Alice"})
print(answer)
print(answer["results"])