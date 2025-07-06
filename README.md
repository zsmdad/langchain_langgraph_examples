# langchain_langgraph_examples

## 创建虚拟环境
```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

## langchain 例子
example001: 文本生成，根据用户输入的主题生成文本

example002: 联网搜索，根据互联网内容生成问题结果。

example003: 基于redis的消息管理与聊天历史存储。

example004: 总结聊天历史消息为摘要，成为新的历史消息。

example005: 提示词缓存，命中则直接返回

example006: 基于提示词语义缓存 (有bug)

example007: 多模态模型调用

example008: 工具调用（函数或者类）

example009: mcp客户端调用mcp服务端（接口或者脚本）

example010: openai_tools、openai-tools-agent智能体: 效果较差， 对模型能力要求比较高

example011: ReAct框架智能体: 思考行动观察

example012: ReAct框架智对话能体: 思考行动观察， 支持多轮对话

example013: PlanAndExecute架构智能体

example014: 客服智能问答

https://blog.csdn.net/jining11/article/details/134806188

https://python.langchain.com/docs/how_to/tools_human/

## langgraph 例子
## 简单难度例子

example001: 简单的顺序工作流， hello world

example002: 简单的顺序工作流，加上if-else判断

example003: 简单的顺序工作流，加上条件判断，不满足条件循环运行节点

example004: 聊天机器人


## 中等例子

## 复杂例子