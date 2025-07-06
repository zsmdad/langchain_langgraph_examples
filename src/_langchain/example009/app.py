# mcp工具调用
import os
from typing import List
import asyncio

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient


class Input(BaseModel):
    input: str = Field(description="用户输入")

class Output(BaseModel):
    output: str = Field(description="模型输出")



async def make_chain():

    client = MultiServerMCPClient(
        {
            "amap-amap-sse": {	
                "transport": "sse",
                "url": "https://mcp.amap.com/sse?key=65507452abf40841411d9614f180afe6"
            }
        }
    )
    tools = await client.get_tools()

    # 创建一个OpenAI聊天模型
    llm = init_chat_model(
            model_provider="openai",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            temperature=0.7
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个得力助手，请尽可能使用工具来回答用户的问题。"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    chain = agent_executor.with_types(input_type=Input, output_type=Output)
    return chain

chain = asyncio.run(make_chain())

if __name__ == "__main__":
    response = chain.invoke({"input": "苏州的天气？苏州到南昌的距离？"})
    print(f'output: {response['output']}')