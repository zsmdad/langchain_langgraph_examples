# 联网搜索
import json
import os
from dotenv import load_dotenv
from typing import List
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever


class Source(BaseModel):
    content: str = Field(description="网页的内容")
    url: str = Field(description="网页的URL")
    title: str = Field(description="网页的标题")

class Output(BaseModel):
    source: List[Source] = Field(description="相关的网页")
    answer: str = Field(description="AI的答案")


retriever = TavilySearchAPIRetriever(k=3)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

FORMAT_INSTRUCTIONS: {format_instructions}

Context: {context}

Question: {question}

"""
)


llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7,
    )

def format_docs(docs):
    result = []
    for doc in docs:
        result.append({
            "content": doc.page_content, 
            "source": doc.metadata["source"], 
            "title": doc.metadata["title"],
            "score": doc.metadata["score"]
        })
    return json.dumps(result, ensure_ascii=False, indent=4)


parser = JsonOutputParser(pydantic_object=Output)
format_instructions = parser.get_format_instructions()

chain = (
    {
        # "context": (lambda x: x["question"]) | retriever | format_docs, 
        # "question": lambda x: x["question"],
        "context": RunnablePassthrough() | retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt.partial(format_instructions=format_instructions)
    | llm
    | parser
).with_types(input_type=str, output_type=Output)


if __name__ == "__main__":
    input = {"question": "2031年亚马尔多少岁？"}
    input = "2031年亚马尔多少岁？"
    output = chain.invoke(input)
    print(f'input: {input}\noutput: {json.dumps(output, ensure_ascii=False, indent=4)}')

