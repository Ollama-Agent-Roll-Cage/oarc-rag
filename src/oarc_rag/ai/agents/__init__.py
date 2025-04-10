"""
Agent implementations for the RAG framework.

This package provides domain-agnostic agent implementations that leverage 
the capabilities of the RAG system for different specialized tasks.
"""

from oarc_rag.ai.agents.rag_agent import RAGAgent

# Import future agent implementations as they are developed
# from oarc_rag.ai.agents.expansion_agent import ExpansionAgent
# from oarc_rag.ai.agents.merge_agent import MergeAgent
# from oarc_rag.ai.agents.split_agent import SplitAgent
# from oarc_rag.ai.agents.prune_agent import PruneAgent

__all__ = ["RAGAgent"]

# Add other agent classes to __all__ as they are implemented
# __all__ = ["RAGAgent", "ExpansionAgent", "MergeAgent", "SplitAgent", "PruneAgent"]
