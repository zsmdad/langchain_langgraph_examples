# 多模态模型-图片识别
import os
import base64
from typing import List

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
import httpx
from langserve import CustomUserType

class Input(CustomUserType):
    file: str = Field(..., extra={"widget": {"type": "base64file"}})


# 创建一个提示词模版
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个有帮助的助手。请用少于100字的回答用户的问题。 "),
    MessagesPlaceholder(variable_name="history"),
])

# 创建一个OpenAI聊天模型
llm = init_chat_model(
        model_provider="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("VL_MODEL", "gpt-4o"),
        temperature=0.7
    )

parser = StrOutputParser()

def process_image(request: Input) -> str:
    """
    处理图像数据，返回文本描述
    """
    image_data = request.file.encode("utf-8").decode("utf-8")
    history = {
        "history": [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "请描述图片中的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ])
        ]
    }
    response = (prompt | llm | parser).invoke(history)
    return response

chain = RunnableLambda(process_image).with_types(input_type=Input, output_type=str)

if __name__ == "__main__":
    image_url = "https://wttr.in/suzhou.png"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    response = chain.invoke(Input(file=image_data))
    print(response)
