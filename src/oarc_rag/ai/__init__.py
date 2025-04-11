"""
AI integration components for the OARC RAG system.

This package provides AI agent implementations, prompt management, and 
client interfaces for working with language models in the RAG framework.
"""

from oarc_rag.ai.client import OllamaClient
from oarc_rag.ai.agent import Agent
from oarc_rag.ai.agents import RAGAgent, RAGEnhancedAgent

__all__ = [
    "Agent", 
    "OllamaClient",
    "RAGAgent",
    "RAGEnhancedAgent"
]
