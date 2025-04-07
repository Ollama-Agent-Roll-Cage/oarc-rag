"""
AI integration modules for oarc_rag.

This package provides functionality for interacting with AI models
to generate RAG content and analyze resources.
"""

from oarc_rag.ai.client import OllamaClient
from oarc_rag.ai.prompts import PromptTemplate

__all__ = ["OllamaClient", "PromptTemplate"]
