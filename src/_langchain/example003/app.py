# 聊天机器人，历史消息保存到缓存
import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_message_histories import RedisChatMessageHistory

class Input(BaseModel):
    input: str = Field(description="用户对话输入")

# 创建一个聊天提示词模版
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's questions less than 20 words. "),
    ("placeholder", "{history}"),
    ("user", "{input}")
])

# 定义一个消息截断器
trimmer = trim_messages(
    # Keep the last <= n_count tokens of the messages.
    strategy="last",
    token_counter=len,
    max_tokens=10,
    start_on="human",
    # end_on=("human", "tool"),
    include_system=True,
)

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7,
    )

parser = StrOutputParser()

# 创建一个可运行的链
chat_chain = (
    {
        "history": (lambda x: x["history"]) | trimmer,
        "input": lambda x: x["input"]
    }
    | chat_prompt 
    | llm
    | parser
)

store = {}

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """获取消息历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
    # return RedisChatMessageHistory(
    #     session_id=session_id, 
    #     url="redis://127.0.0.1:16379/0",
    #     ttl=None,
    #     key_prefix="chat_history:"
    # )


chain_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain = chain_with_message_history.with_types(input_type=Input, output_type=str)


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
        print('ai: ', response)
