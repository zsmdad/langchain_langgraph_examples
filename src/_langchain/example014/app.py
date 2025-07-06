import os
import json
from typing import List
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

class Input(BaseModel):
    input: str = Field(description="用户对话输入")

class Output(BaseModel):
    output: str = Field(description="AI对话返回")

# class PydanticArgs(BaseModel):
#     args: str

# 创建向量数据库
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
# 创建一个向量检索器
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


def _format_docs(docs: List[Document]) -> str:
    """格式化文档为字符串"""
    format_docs =  "\n\n".join([doc.page_content for i, doc in enumerate(docs)])
    return format_docs

store = {}

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """获取消息历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建一个系统提示
# SYSTEM_TEMPLATE = """你是一个专业的客服助手。请基于提供的上下文信息和历史聊天录回答用户的问题。

# 回答规则：
# 1. 如果上下文中有相关信息，请基于这些信息给出准确、友好的回答
# 2. 如果上下文信息不完整，请结合常识给出有帮助的建议
# 3. 如果完全没有相关信息，请礼貌地说"抱歉，我暂时没有找到相关信息，建议您联系人工客服获得更详细的帮助"
# 4. 保持回答简洁明了，语气友好专业
# 5. 如果用户询问敏感信息，请提醒用户注意隐私保护

# 检索到的文档:
# {context}
# """
SYSTEM_TEMPLATE = """你是一个专业的客服助手。请基于提供的上下文信息和历史聊天录回答用户的问题。
如果完全没有相关信息，请礼貌地说"抱歉，我暂时没有找到相关信息。"

检索到的文档:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("placeholder", "{history}"),
        ("user", "{input}"),
    ]
)

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.0 # 模型不要随机性
)
# 创建一个输出解析器
parser = StrOutputParser()

qa_chain = (
            {
                "context": lambda x: _format_docs(retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "history": lambda x: x["history"],
            }
            | prompt
            | llm
            | parser
        )

chain_with_message_history = RunnableWithMessageHistory(
    qa_chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain = chain_with_message_history.with_types(input_type=Input, output_type=Output)

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

