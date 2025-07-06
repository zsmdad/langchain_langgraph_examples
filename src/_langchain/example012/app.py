# openai tool agent
import os
import json

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.agents import tool, AgentExecutor
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import PlanAndExecute
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

class Input(BaseModel):
    input: str = Field(description="用户对话输入")

class Output(BaseModel):
    output: str = Field(description="AI对话返回")

class PydanticArgs(BaseModel):
    args: str

# 创建工具
@tool('add', args_schema=PydanticArgs)
def add(args: str):
    """两个数字相加
    param:
        args: json str
        example: {"a": 10, "b": 2}
    return:
        Union[int, float]: 两个数字相加的结果
    """
    if isinstance(args, str):
        args = json.loads(args)
    return args['a'] + args['b']

@tool('minus', args_schema=PydanticArgs)
def minus(args: str):
    """两个数字相减
    param:
        args: json str
        example: {"a": 10, "b": 2}
    return:
        Union[int, float]: 两个数字相减的结果
    """
    if isinstance(args, str):
        args = json.loads(args)
    return args['a'] - args['b']

@tool('multiply', args_schema=PydanticArgs)
def multiply(args: str):
    """两个数字相乘
     param:
        args: dict
        example: {"a": 10, "b": 2}
    return:
        Union[int, float]: 两个数字相乘的结果
    """
    if isinstance(args, str):
        args = json.loads(args)
    return args['a'] * args['b']

@tool('divide', args_schema=PydanticArgs)
def divide(args: str):
    """两个数字相除
    param:
        args: json str
        example: {"a": 10, "b": 2}
    return:
        Union[int, float]: 两个数字相除的结果
    """
    try:
        if isinstance(args, str):
            args = json.loads(args)
        return args['a'] / args['b']
    except ZeroDivisionError:
        return "Cannot divide by zero."

@tool('random', args_schema=None)
def _random():
    """随机生成一个数字
    param:
        None
    return:
        int: 随机生成的数字
    """
    import random
    return random.randint(1, 100)

@tool('time', args_schema=None)
def _time():
    """获取当前时间
    param:
        None
    return:
        str: 当前时间
    """
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool('search', args_schema=None)
def search(query: str):
    """DuckDuckGo搜索
    param:
        query: str
    return:
        str: 搜索结果
    """
    # 创建自定义的API包装器
    wrapper = DuckDuckGoSearchAPIWrapper(
        # region="de-de",  # 设置搜索区域为德国
        # time="w",        # 只搜索最近一天的结果
        max_results=2    # 限制结果数量为2
    )

    # 使用自定义包装器创建搜索对象
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
    return search.invoke(query)

# 定义一个工具列表
tools = [add, minus, multiply, divide, _random, _time, search]

# 创建一个提示词模版
# https://smith.langchain.com/hub/hwchase17/react-chat
prompt = hub.pull("hwchase17/react-chat")

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7 # 模型不要随机性
)

memory = ConversationBufferMemory(memory_key="chat_history")


agent = PlanAndExecute(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, 
    handle_parsing_errors=True, max_iterations=10,
    memory=memory
)

chain = agent_executor.with_types(input_type=Input, output_type=Output)

if __name__ == "__main__":
    import uuid
    # session_id = str(uuid.uuid4())
    # print("session_id: ", session_id)

    while True:
        user_input = input("user: ")
        response = chain.invoke(
            input={"input": user_input},
            # config={
            #     "configurable": {
            #         "session_id": session_id
            #     }
            # }
        )
        print('ai: ', response['output'])