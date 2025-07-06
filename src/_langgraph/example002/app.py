from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str

def condition_judgment(state) -> Literal["is_digit", "is_not_digit"]:
    """条件判断"""
    print(f"condition_judgment: {state}")
    message = state['message'].strip()
    if message.isdigit():
        return "is_digit"
    else:
        return "is_not_digit"

def is_digit(state):
    """纯数字节点"""
    print(f"node is_digit: {state}")
    return {"message": "is digit"}

def is_not_digit(state):
    """非纯数字节点"""
    print(f"node is_not_digit: {state}")
    return {"message": "is not digit"}

workflow = StateGraph(State)
workflow.add_node("is_digit", is_digit)
workflow.add_node("is_not_digit", is_not_digit)

workflow.add_conditional_edges(START, condition_judgment)
workflow.add_edge("is_digit", END)
workflow.add_edge("is_not_digit", END)

graph = workflow.compile()

if __name__ == "__main__":
    # view the graph image
    # png = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png)
    # test the graph
    start = State({"message": "666"})
    end = graph.invoke(start)
    print(f"end: {end}")