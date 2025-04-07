"""
RAG-enhanced agent for retrieval-augmented content generation.

This module provides an agent that leverages the RAG system to enhance
content generation with relevant retrieved context.
"""
import time
from typing import Any, Dict, List, Optional, Union

from oarc_rag.utils.log import log
from oarc_rag.ai.agents.agent import Agent
from oarc_rag.core.engine import RAGEngine
from oarc_rag.core.context import ContextAssembler
from oarc_rag.core.query import QueryFormulator
from oarc_rag.ai.client import OllamaClient


class RAGAgent(Agent):
    """
    Agent with RAG enhancement capabilities for content generation.
    
    This class extends the basic Agent to incorporate retrieval-augmented
    generation features for enhanced content generation.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.1:latest",
        rag_engine: Optional[RAGEngine] = None,
        context_assembler: Optional[ContextAssembler] = None,
        query_formulator: Optional[QueryFormulator] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize a RAG-enhanced agent.
        
        Args:
            name: Agent name/identifier
            model: LLM model to use for generation
            rag_engine: RAG engine for retrieving context
            context_assembler: Assembler for formatting context
            query_formulator: Formulator for query generation
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
        """
        super().__init__(name, model, temperature, max_tokens)
        
        # RAG components
        self.rag_engine = rag_engine
        self.context_assembler = context_assembler or ContextAssembler()
        self.query_formulator = query_formulator or QueryFormulator()
        
        # Create Ollama client
        self.client = OllamaClient(default_model=model)
        
        # Performance tracking
        self.retrieval_stats = {
            "calls": 0,
            "total_chunks": 0,
            "average_chunks": 0,
            "total_retrieval_time": 0.0,
            "average_retrieval_time": 0.0
        }
        
    def set_rag_engine(self, rag_engine: RAGEngine) -> None:
        """
        Set the RAG engine for this agent.
        
        Args:
            rag_engine: RAG engine to use
        """
        self.rag_engine = rag_engine
        self.log_activity("RAG engine set")
        
    def retrieve_context(
        self,
        topic: str,
        query_type: str = "general",
        additional_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        deduplicate: bool = True
    ) -> str:
        """
        Retrieve and format context for generation.
        
        Args:
            topic: Main topic for retrieval
            query_type: Type of query (general, analysis, comparison, etc.)
            additional_context: Additional context for query formulation
            top_k: Maximum number of chunks to retrieve
            threshold: Minimum similarity threshold
            deduplicate: Whether to deduplicate similar chunks
            
        Returns:
            str: Formatted context for inclusion in prompts
            
        Raises:
            RuntimeError: If RAG engine is not set
        """
        if not self.rag_engine:
            raise RuntimeError("RAG engine not initialized. Call set_rag_engine() first.")
            
        # Formulate query based on topic and type
        query = self.query_formulator.formulate_query(
            topic=topic,
            query_type=query_type,
            additional_context=additional_context
        )
        
        self.log_activity(f"Retrieving context for query: {query[:50]}...")
        
        # Track retrieval performance
        start_time = time.time()
        
        # Retrieve relevant chunks
        results = self.rag_engine.retrieve(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
        
        # Calculate retrieval time
        retrieval_time = time.time() - start_time
        
        # Assemble context from chunks
        context = self.context_assembler.assemble_context(
            chunks=results,
            deduplicate=deduplicate
        )
        
        # Update retrieval statistics
        self.retrieval_stats["calls"] += 1
        self.retrieval_stats["total_chunks"] += len(results)
        self.retrieval_stats["average_chunks"] = (
            self.retrieval_stats["total_chunks"] / self.retrieval_stats["calls"]
        )
        self.retrieval_stats["total_retrieval_time"] += retrieval_time
        self.retrieval_stats["average_retrieval_time"] = (
            self.retrieval_stats["total_retrieval_time"] / self.retrieval_stats["calls"]
        )
        
        # Log retrieval stats
        self.log_activity(
            f"Retrieved {len(results)} chunks in {retrieval_time:.3f}s "
            f"(avg: {self.retrieval_stats['average_retrieval_time']:.3f}s)"
        )
        
        return context
        
    def create_enhanced_prompt(
        self,
        base_prompt: str,
        topic: str,
        query_type: str = "general",
        additional_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        context_strategy: str = "prefix"
    ) -> str:
        """
        Create a RAG-enhanced prompt by adding retrieved context.
        
        Args:
            base_prompt: Original prompt to enhance
            topic: Topic for context retrieval
            query_type: Type of query for retrieval
            additional_context: Additional context for query formulation
            top_k: Maximum number of chunks to retrieve
            context_strategy: How to incorporate context ("prefix", "suffix", "combined")
            
        Returns:
            str: Enhanced prompt with retrieved context
        """
        # Get relevant context
        context = self.retrieve_context(
            topic=topic,
            query_type=query_type,
            additional_context=additional_context,
            top_k=top_k
        )
        
        # Incorporate context based on strategy
        if context_strategy == "prefix":
            enhanced_prompt = f"Context information:\n\n{context}\n\nBased on the above context, {base_prompt}"
        elif context_strategy == "suffix":
            enhanced_prompt = f"{base_prompt}\n\nUse the following context to inform your response:\n\n{context}"
        elif context_strategy == "combined":
            enhanced_prompt = (
                f"Context information:\n\n{context}\n\n"
                f"Task: {base_prompt}\n\n"
                f"Generate a response that uses the context information to complete the task."
            )
        else:  # Default to prefix
            enhanced_prompt = f"Context information:\n\n{context}\n\nBased on the above context, {base_prompt}"
        
        return enhanced_prompt
        
    def generate(self, prompt: str) -> str:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: Prompt to generate from
            
        Returns:
            str: Generated content
        """
        self.log_activity(f"Generating content for prompt of length {len(prompt)}")
        
        start_time = time.time()
        
        # Generate response using Ollama
        response = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Update statistics
        generation_time = time.time() - start_time
        # Rough token count estimation
        tokens_used = len(prompt.split()) + len(response.split())
        self.update_stats(tokens_used, generation_time)
        
        self.log_activity(
            f"Generated {len(response)} chars in {generation_time:.3f}s"
        )
        
        return response
        
    def process(self, input_data: Dict[str, Any]) -> str:
        """
        Process input data and produce output using RAG enhancement.
        
        Args:
            input_data: Dictionary with processing parameters including:
                - topic: Main topic for retrieval
                - query_type: Type of query
                - base_prompt: Base prompt to enhance
            
        Returns:
            str: Generated content
        """
        # Extract parameters
        topic = input_data.get("topic", "")
        query_type = input_data.get("query_type", "general")
        base_prompt = input_data.get("base_prompt", "")
        additional_context = input_data.get("additional_context", {})
        
        if not topic or not base_prompt:
            raise ValueError("Input data must include 'topic' and 'base_prompt'")
            
        # Create enhanced prompt
        enhanced_prompt = self.create_enhanced_prompt(
            base_prompt=base_prompt,
            topic=topic,
            query_type=query_type,
            additional_context=additional_context
        )
        
        # Generate content
        result = self.generate(enhanced_prompt)
        
        # Store result
        self.store_result("last_generation", result)
        
        return result
        
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval performance statistics.
        
        Returns:
            Dict[str, Any]: Retrieval statistics
        """
        return self.retrieval_stats.copy()
