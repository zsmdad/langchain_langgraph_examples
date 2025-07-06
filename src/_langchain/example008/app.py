# 工具调用
import os
import base64
from typing import List

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_tool_calling_agent, AgentExecutor

class Input(BaseModel):
    input: str = Field(description="用户输入")

class Output(BaseModel):
    output: str = Field(description="模型输出")

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7
    )
# 自定义工具
@tool("get_current_time")
def get_time():
    """
    获取当前时间
    """
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool("get_weather")
def get_weather():
    """
    获取天气信息
    """
    import httpx
    url = 'https://mu5n8keghh.re.qweatherapi.com/v7/weather/now'
    location = 101010100 
    QWEATHER_API_KEY = '11e5950bdcf7492198a3949a4bdf3e57'
    headers = {
        'X-QW-Api-Key': QWEATHER_API_KEY,
    }
    params = {
        'location': location,
    }
    response = httpx.get(url, headers=headers, params=params)
    return response.json()

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# 内置工具
db = SQLDatabase.from_uri("sqlite:///example.db")
sql_tookit = SQLDatabaseToolkit(db=db, llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个得力助手，请尽可能使用工具来回答用户的问题。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [get_time, get_weather, add, multiply] + sql_tookit.get_tools()

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor.with_types(input_type=Input, output_type=Output)

if __name__ == "__main__":
    response = chain.invoke({"input": "现在什么时间，天气如何？"})
    print(f'output: {response['output']}')