from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    final_number: int


def adder(state: AgentState) -> AgentState:
    """Node that adds two numbers"""
    state["final_number"] = state["number1"] + state["number2"]
    return state


def subtractor(state: AgentState) -> AgentState:
    """Node that subtracts two numbers"""
    state["final_number"] = state["number1"] - state["number2"]
    return state


def decide_next_node(state: AgentState) -> AgentState:
    """This node will decide next node of the graph"""
    if state["operation"] == "+":
        return "addition_opration"
    elif state["operation"] == "-":
        return "subtraction_operation"


graph = StateGraph(AgentState)
graph.add_node("add_node", adder)
graph.add_node("subtract_node", subtractor)
graph.add_node("router", lambda state: state)  #passthrough function

graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_next_node, {
    "addition_opration": "add_node",
    "subtraction_operation": "subtract_node"})
graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)

app = graph.compile()

inital_state_1 = AgentState(number1=10, operation="-", number2=5)
print(app.invoke(inital_state_1))



