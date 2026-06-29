"""DashScope Qwen LLM client."""

import dashscope
from dashscope import Generation

from config import DASHSCOPE_API_KEY


SYSTEM_PROMPT = "你是一个智能学习与科研助手，回答要简洁清晰，适合研究生论文学习场景。"


def ask_llm(prompt):
    """Call DashScope Qwen and return the model response."""
    dashscope.api_key = DASHSCOPE_API_KEY

    if DASHSCOPE_API_KEY == "your_dashscope_api_key_here":
        return "请先在 config.py 中填写真实的 DASHSCOPE_API_KEY。"

    response = Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        result_format="message",
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content

    return f"调用失败：{response.message}"
