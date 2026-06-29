"""DashScope Qwen LLM client."""

import dashscope
from dashscope import Generation

from config import DASHSCOPE_API_KEY


SYSTEM_PROMPT = "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u5b66\u4e60\u4e0e\u79d1\u7814\u52a9\u624b\uff0c\u56de\u7b54\u8981\u7b80\u6d01\u6e05\u6670\uff0c\u9002\u5408\u7814\u7a76\u751f\u8bba\u6587\u5b66\u4e60\u573a\u666f\u3002"


def ask_llm(prompt):
    """Call DashScope Qwen and return the model response."""
    dashscope.api_key = DASHSCOPE_API_KEY

    if DASHSCOPE_API_KEY == "your_dashscope_api_key_here":
        return "\u8bf7\u5148\u5728 config.py \u6216 Streamlit Secrets \u4e2d\u586b\u5199\u771f\u5b9e\u7684 DASHSCOPE_API_KEY\u3002"

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

    return f"\u8c03\u7528\u5931\u8d25\uff1a{response.message}"
