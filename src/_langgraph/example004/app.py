import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL"),
)

def chatbot(state: State):
    """聊天节点"""
    return {"messages": [llm.invoke(state['messages'])]}


workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)

workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# checkpointer = InMemorySaver()
# store = InMemoryStore()
graph = workflow.compile()#checkpointer=checkpointer)

if __name__ == "__main__":
    # view the graph image
    # png = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png)

    # test the graph
    while True:
        user_input = input("User: ")
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
