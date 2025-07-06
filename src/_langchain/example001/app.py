# 根据用户输入的主题生成文本
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文本生成助手，请协助用户完成写作任务。"),
    # ("placeholder", "{history}"),
    ("user", "写一个故事，不超过50字。主题是：{topic}"),
])

llm = init_chat_model(
        model_provider="openai",
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7
    )

parser = StrOutputParser()

chain = (
    {"topic": RunnablePassthrough()} 
    | prompt
    | llm 
    | parser
).with_types(input_type=str, output_type=str)


if __name__ == "__main__":
    input = "司马光砸缸"
    output = chain.invoke(input)
    print(f"input: {input}\noutput: {output}")
