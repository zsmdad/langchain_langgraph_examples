# 提示词语义缓存，提高返回速度
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_redis import RedisSemanticCache
from langchain_core.caches import InMemoryCache


# 创建一个提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's questions less than 100 words. "),
    ("user", "{input}")
])

# Initialize RedisCache
embeddings = OllamaEmbeddings(
    base_url=os.getenv("EMBEDDINGS_BASE_URL"),
    model=os.getenv("EMBEDDINGS_MODEL")
)

try:
    semantic_cache = RedisSemanticCache(
        redis_url="redis://127.0.0.1:16379/0",
        embeddings=embeddings, 
        distance_threshold=0.2,
    )
except Exception as e:
    semantic_cache = InMemoryCache()
    print(f"RedisCache error: {e}")

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7,
        cache=semantic_cache,
    )

parser = StrOutputParser()

chain = (prompt | llm | parser).with_types(input_type=str, output_type=str)

if __name__ == "__main__":
    import time
    # 单独测试，需要关闭langsmith
    start_time = time.time()
    result = chain.invoke("西虹市怎么样")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    print(f'result: {result}')