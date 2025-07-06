# 查询测试
import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

embeddings = OllamaEmbeddings(
    base_url=os.getenv("EMBEDDINGS_BASE_URL"),
    model=os.getenv("EMBEDDINGS_MODEL")
)

collection_name = "kefu"
client = QdrantClient(
    url="http://127.0.0.1:6333"
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

start_time = time.time()
query = "如何注册新账号"
print(f'query: {query}\n')
result = vector_store.similarity_search_with_score(query, k=3)
for doc, score in result:
    print(f'score: {score}, page_content: {doc.page_content}\n')
end_time = time.time()
print(f'time: {end_time - start_time}s')
