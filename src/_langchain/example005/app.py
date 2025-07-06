# 提示词缓存，提高返回速度
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_redis import RedisCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_core.caches import InMemoryCache, BaseCache, RETURN_VAL_TYPE
# from langchain_community.chat_message_histories import RedisChatMessageHistory

class Input(BaseModel):
    input: str = Field(description="用户对话输入")

# 创建一个提示词模版
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's questions less than 100 words. "),
    ("placeholder", "{history}"),
    ("user", "{input}")
])


class CustomInMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def lookup(self, prompt: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt."""
        print(123123, prompt)
        return self._cache.get(prompt, None)

    def update(self, prompt: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt."""
        print(456456, prompt)
        self._cache[prompt] = return_val

# Initialize Cache
# cache = RedisCache(
#     redis_url="redis://127.0.0.1:16379/0",
#     ttl=3600,
#     prefix='langchain',
# )
cache = InMemoryCache()



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

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7,
        cache=cache,
    )

parser = StrOutputParser()

chain = (chat_prompt | llm | parser).with_types(input_type=str, output_type=str)

chain = chain.with_types(input_type=Input, output_type=str)

if __name__ == "__main__":
    # import time
    # # 单独测试，需要关闭langsmith
    # start_time = time.time()
    # result = chain.invoke("拜拜")
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time} seconds")
    # print(f'result: {result}')

    import time

    while True:
        user_input = input("User: ")
        start_time = time.time()
        result = chain.invoke(
            input={"input": user_input},
        )
        end_time = time.time()
        print(f"\nExecution time: {end_time - start_time} seconds")
        print(f'ai: {result}')

