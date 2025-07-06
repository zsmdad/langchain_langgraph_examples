# openai tool agent
import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import tool, create_openai_tools_agent, create_openai_functions_agent, AgentExecutor
from langchain import hub

class Input(BaseModel):
    input: str = Field(description="用户对话输入")

class Output(BaseModel):
    output: str = Field(description="AI对话返回")

@tool('add')
def add(a: int, b: int) -> int:
    """Add two integers together."""
    if isinstance(a, str):
        a = int(a)
    if isinstance(b, str):
        b = int(b)
    return a + b

@tool('minus')
def minus(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b

@tool('multiply')
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

@tool('divide')
def divide(a: int, b: int) -> float:
    """Divide two integers."""
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero."

@tool('random')
def _random() -> float:
    """generate random number"""
    import random
    return random.randint(1, 100)

# 定义一个工具列表
tools = [add, minus, multiply, divide, _random]

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.0 # 模型不要随机性
)
# openai-functions-agent
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/openai_functions_agent/base.py
# prompt = hub.pull("hwchase17/openai-functions-agent")
# print(f'prompt: \n{prompt.messages}\n\n')
# agent = create_openai_functions_agent(llm, tools, prompt)

# openai-tools-agent
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/openai_tools/base.py
prompt = hub.pull("hwchase17/openai-tools-agent")
print(f'prompt: \n{prompt.messages}\n\n')
agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor.with_types(input_type=Input, output_type=Output)

if __name__ == "__main__":
    # 测试：调用多次工具
    input_text = "随机生成两个数，计算他们的加减乘除，然后计算他们的和"
    print(f"user: {input_text}")
    response = agent_executor.invoke({'input': input_text})
    output_text = response['output']
    print(f"ai: {output_text}")