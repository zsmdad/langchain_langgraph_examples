# 聊天机器人， 历史消息总结成摘要并保存到缓存
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory

class Input(BaseModel):
    input: str = Field(description="用户对话输入")

# 创建一个聊天提示词模版
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's questions less than 20 words. "),
    ("placeholder", "{history}"),
    ("user", "{input}")
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("placeholder", "{history}"),
    ("user", "将上述聊天消息浓缩成一条摘要消息，尽量包含关键信息。")
])

# 创建一个OpenAI聊天模型
def get_llm(temperature=0.7):
    return init_chat_model(
            model_provider="openai",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            temperature=temperature,
        )

store = {}

def get_summary_history(session_id: str) -> BaseChatMessageHistory:
    """获取摘要历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
    # return RedisChatMessageHistory(
    #     session_id=session_id, 
    #     url="redis://127.0.0.1:16379/0",
    #     ttl=None,
    #     key_prefix="summary_history:"
    # )

def get_summarize_message(session_id):
    """获取摘要"""
    summary_history = get_summary_history(session_id)
    if len(summary_history.messages) == 0:
        stored_messages = ChatMessageHistory().messages
    else:
        stored_messages = summary_history.messages

    summary_chain = summary_prompt | get_llm(temperature=0.0)
    summary_message = summary_chain.invoke({"history": stored_messages})
    summary_history.clear()
    summary_history.add_ai_message(summary_message)
    return summary_history

parser = StrOutputParser()

# 创建一个可运行的链
chat_chain = (
    {
        "history":  (lambda x: x["history"]),
        "input": lambda x: x["input"]
    }
    | chat_prompt 
    | get_llm(temperature=0.7)
    | parser
)

chat_with_summarize_message = RunnableWithMessageHistory(
    chat_chain,
    get_summarize_message,
    input_messages_key="input",
    history_messages_key="history",
)

chain = chat_with_summarize_message.with_types(input_type=Input, output_type=str)

if __name__ == "__main__":
    import uuid
    session_id = str(uuid.uuid4())
    print("session_id: ", session_id)

    while True:
        user_input = input("user: ")
        response = chain.invoke(
            input={"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )
        print('ai: ', response.content)