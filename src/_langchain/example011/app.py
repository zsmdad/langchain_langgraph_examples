# openai tool agent
import os
import json

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

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
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# https://smith.langchain.com/hub/captain/react
# prompt = hub.pull("captain/react")
# print(f'prompt: \n{prompt.messages}\n\n')
# prompt = PromptTemplate.from_template('''
# 尽你所能回答以下问题。你可以使用以下工具：

# {tools}

# 请使用以下格式，其中Action字段后必须跟着Action Input字段，并且不要将Action Input替换成Input或者tool等字段，不能出现格式以外的字段名，每个字段在每个轮次只出现一次：

# Question: 你需要回答的输入问题 
# Thought: 你应该总是思考该做什么 
# Action: 要采取的动作，动作只能是[{tool_names}]中的一个 ，一定不要加入其它内容
# Action Input: 行动的输入，必须出现在Action后。
# Observation: 行动的结果 
# ...（这个Thought/Action/Action Input/Observation可以重复N次） 
# Thought: 我现在知道最终答案 
# Final Answer: 对原始输入问题的最终答案



# 再次重申，不要修改以上模板的字段名称，开始吧！

# Question: {input} 
# Thought:{agent_scratchpad}
# '''
# )
# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.7 # 模型不要随机性
)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, 
    handle_parsing_errors=True, max_iterations=10
)

chain = agent_executor.with_types(input_type=Input, output_type=Output)

if __name__ == "__main__":
    # 测试：调用多次工具
    input_text = "边长为3、4、5的三角形， 周长、面积、体积分别为多少？"
    input_text = "2025田径亚锦赛110栏冠军父母分别来自哪里"
    
    print(f"user: {input_text}")
    response = agent_executor.invoke({'input': input_text})
    output_text = response['output']
    print(f"ai: {output_text}")