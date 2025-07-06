# csv数据导入嵌入模型
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

loader = CSVLoader(file_path='/Users/zouxinyin/data/github/langchain_langgraph_examples/src/_langchain/example014/kefu.csv',
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
    }
)
docs = loader.load()
# for i, doc in enumerate(docs):
#     print(f'{i}/{len(docs)}')
#     print(doc)
#     if i ==10:
#         break

embeddings = OllamaEmbeddings(
    base_url=os.getenv("EMBEDDINGS_BASE_URL"),
    model=os.getenv("EMBEDDINGS_MODEL")
)

collection_name = "kefu"
client = QdrantClient(
    url="http://127.0.0.1:6333"
)
try:
    client.create_collection(
        collection_name=collection_name, 
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
except Exception as e:
    pass

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# 打印进度
for i, doc in enumerate(docs):
    print(f'{i}/{len(docs)}')
    print(doc)
    vector_store.add_documents([doc])