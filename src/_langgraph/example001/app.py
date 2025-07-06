from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str

def hello(state):
    """和某某某打招呼"""
    print(f"node hello: {state}")
    return {
        "message": "Hello " + state['message'] + "! ",
    }

def repeat(state):
    """重复三次"""
    print(f"node repeat: {state}")
    return {
        "message": state['message'] * 3,
    }


workflow = StateGraph(State)

workflow.add_node("hello", hello)
workflow.add_node("repeat", repeat)

workflow.add_edge(START, "hello")
workflow.add_edge("hello", "repeat")
workflow.add_edge("repeat", END)

graph = workflow.compile()

if __name__ == "__main__":
    # view the graph image
    # png = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png)
    # test the graph
    start = State({"message": "world"})
    end = graph.invoke(start)
    print(f"end: {end}")