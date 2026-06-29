"""Compatibility wrapper for the final orchestration layer."""

from agent.core_agent import run_agent
from rag import retriever


def run_orchestrator(query):
    """Run the core Agent with the latest loaded PDF data."""
    return run_agent(query, retriever.get_pdf_data())
