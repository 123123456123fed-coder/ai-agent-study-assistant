"""Configuration for API keys and project settings.

For Streamlit Cloud, set DASHSCOPE_API_KEY in App secrets.
For local development, set it as an environment variable or replace the
placeholder below temporarily.
"""

import os


def _read_streamlit_secret(name):
    """Read a Streamlit secret when running inside Streamlit."""
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


DASHSCOPE_API_KEY = (
    _read_streamlit_secret("DASHSCOPE_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or "your_dashscope_api_key_here"
)

# Kept for compatibility with older code paths.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
