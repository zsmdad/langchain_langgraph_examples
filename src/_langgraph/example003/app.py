import random
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    number: int

def random_number(state):
    number = random.randint(1, 20)
    print(f"gen random number: {number}")
    return {"number": number}

def condition_judgment(state) -> Literal["random_number", END]:
    """条件判断"""
    print(f"condition_judgment: {state}")
    number = state['number']
    if number < 10:
        return "random_number"
    else:
        return END


workflow = StateGraph(State)
workflow.add_node("random_number", random_number)

workflow.add_edge(START, "random_number")
workflow.add_conditional_edges("random_number", condition_judgment)
workflow.add_edge("random_number", END)

graph = workflow.compile()

if __name__ == "__main__":
    # view the graph image
    # png = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png)
    # test the graph
    start = State()
    end = graph.invoke(start)
    print(f"end: {end}")