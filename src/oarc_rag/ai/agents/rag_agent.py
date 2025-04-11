"""
RAG-enhanced agent for retrieval-augmented content generation.

This module provides a domain-agnostic agent that leverages the RAG system
to enhance content generation with relevant retrieved context across any domain.
"""
import json
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from oarc_rag.ai.agent import OperationalMode
from oarc_rag.ai.agents.base_agent import RAGAgent
from oarc_rag.core.cache import cache_manager
from oarc_rag.core.context import ContextAssembler
from oarc_rag.core.engine import Engine
from oarc_rag.core.query import QueryFormulator
from oarc_rag.utils.log import log
from oarc_rag.utils.utils import Utils


class RAGEnhancedAgent(RAGAgent):
    """
    Domain-agnostic agent with RAG enhancement capabilities for content generation.
    
    This class extends the RAGAgent to incorporate advanced retrieval-augmented
    generation features for enhanced content generation across any domain.
    Implements concepts from the recursive self-improving RAG framework.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.1:latest",
        rag_engine: Optional[Engine] = None,
        context_assembler: Optional[ContextAssembler] = None,
        query_formulator: Optional[QueryFormulator] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        version: str = "1.0",
        operational_mode: Union[str, OperationalMode] = "awake",  # Supports "awake" or "sleep" phase
        enable_semantic_reranking: bool = True,
        use_vector_quantization: bool = False,  # From Specification.md
        pca_dimensions: Optional[int] = None,    # From Specification.md
        auto_cycle: bool = True,
        cycle_interval: int = 3600
    ):
        """
        Initialize a domain-agnostic RAG-enhanced agent.
        
        Args:
            name: Agent name/identifier
            model: LLM model to use for generation
            rag_engine: RAG engine for retrieving context
            context_assembler: Assembler for formatting context
            query_formulator: Formulator for query generation
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            version: Agent version string
            operational_mode: Agent operational mode ("awake" for real-time, "sleep" for deep processing)
            enable_semantic_reranking: Whether to use semantic reranking for results
            use_vector_quantization: Whether to use vector quantization for memory optimization
            pca_dimensions: Number of dimensions to use for PCA reduction (None for no reduction)
            auto_cycle: Whether to automatically cycle between operational modes
            cycle_interval: Seconds between operational mode transitions
        """
        # Prepare prompt templates for different purposes
        prompt_templates = {
            'retrieval': 'vector_search',
            'query_reformulation': 'query_reformulation',
            'context_presentation': 'context_presentation',
            'semantic_reranking': 'context_presentation',
            'awake_phase': 'awake_phase',
            'sleep_phase': 'sleep_phase_enrichment',
            'response_generation': 'rag_system'
        }
        
        # Initialize base agent with multiple templates
        super().__init__(
            name=name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            version=version,
            operational_mode=operational_mode,
            auto_cycle=auto_cycle,
            cycle_interval=cycle_interval,
            prompt_templates=prompt_templates,
            default_template='rag_system'
        )
        
        # RAG components
        self.rag_engine = rag_engine
        self.context_assembler = context_assembler or ContextAssembler()
        self.query_formulator = query_formulator or QueryFormulator()
        
        # Advanced vector options from Specification.md
        self.enable_semantic_reranking = enable_semantic_reranking
        self.use_vector_quantization = use_vector_quantization
        self.pca_dimensions = pca_dimensions
        
        # Get context cache
        self.context_cache = cache_manager.context_cache
        
        # Enhanced retrieval statistics
        self.retrieval_stats = {
            "calls": 0,
            "total_chunks": 0,
            "average_chunks": 0,
            "total_retrieval_time": 0.0,
            "average_retrieval_time": 0.0,
            "context_strategies_used": {},
            "query_types_used": {},
            "reranking_improvements": 0,
            "dimensionality_reductions": 0,
            "vector_quantizations": 0
        }
        
        # Knowledge refinement metrics from Big_Brain.md
        self.knowledge_metrics = {
            "recalled_chunks": [],  # Tracking chunks for importance calculations
            "chunk_frequency": Counter(),  # How often each chunk is retrieved
            "topic_clusters": {},  # Grouping related chunks by topic
            "weak_areas": set(),  # Topics with poor retrieval performance
        }
        
        # Initialize mode-specific components
        self._initialize_mode_components()
        
        log.info(f"RAGEnhancedAgent {name} initialized in {operational_mode} mode")
            
    def _initialize_mode_components(self) -> None:
        """Initialize components based on operational mode."""
        if self.operational_mode == OperationalMode.SLEEP:
            # Configure for deep processing mode
            self._configure_for_sleep_mode()
        else:
            # Configure for responsive mode
            self._configure_for_awake_mode()
            
    def _configure_for_sleep_mode(self) -> None:
        """Configure for sleep (deep processing) mode."""
        # Adjust for deep processing
        self.enable_semantic_reranking = True
        self.context_assembler.set_optimization_level("high")
        
    def _configure_for_awake_mode(self) -> None:
        """Configure for awake (responsive) mode."""
        # Adjust for responsiveness
        self.context_assembler.set_optimization_level("low")
    
    def _semantic_rerank(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply semantic reranking to retrieved results.
        
        Args:
            results: Original retrieval results
            query: Original query
            
        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        if not self.enable_semantic_reranking or len(results) <= 1:
            return results
            
        start_time = time.time()
        
        try:
            # Extract text from results for the reranking prompt
            result_texts = [{"text": r["text"][:200], "index": i} for i, r in enumerate(results)]
            
            # Generate reranking prompt using template system
            reranking_prompt = self.render_prompt_template(
                purpose="semantic_reranking",
                query=query,
                results=result_texts
            )
            
            # Get reranking from LLM
            reranking_response = self.generate(reranking_prompt)
            
            # Parse the response to extract the ranking
            # Expected format is a list like [3, 1, 5, 2, 4]
            try:
                # Use Utils function to extract JSON from text
                parsed_data = Utils.extract_json_from_text(reranking_response)
                
                if not parsed_data or "ranking" not in parsed_data:
                    # Fallback to regex parsing
                    match = re.search(r'\[([\d\s,]+)\]', reranking_response)
                    if not match:
                        log.warning("Could not parse reranking response, using original order")
                        return results
                        
                    # Parse the list of indices
                    indices_str = match.group(1)
                    indices = [int(idx) for idx in indices_str.split(',') if idx.strip().isdigit()]
                else:
                    indices = parsed_data["ranking"]
                
                # Adjust indices (model might use 1-based indexing)
                indices = [idx - 1 if idx >= 1 else idx for idx in indices]
                
                # Filter to valid indices only
                valid_indices = [idx for idx in indices if 0 <= idx < len(results)]
                
                # Rerank results based on parsed indices
                reranked_results = [results[idx] for idx in valid_indices]
                
                # Add any results not in the ranking at the end
                included_indices = set(valid_indices)
                for i, result in enumerate(results):
                    if i not in included_indices:
                        reranked_results.append(result)
                        
                # Track improvement metrics
                if reranked_results != results:
                    self.retrieval_stats["reranking_improvements"] += 1
                    
                elapsed = time.time() - start_time
                self.log_activity(f"Semantic reranking applied in {elapsed:.3f}s")
                return reranked_results
                
            except Exception as e:
                log.warning(f"Error parsing reranking response: {e}")
                return results
                
        except Exception as e:
            log.warning(f"Semantic reranking failed: {e}, using original order")
            return results
    
    def retrieve_context(
        self,
        query: str = "",
        topic: str = "",
        query_type: str = "standard",
        additional_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        deduplicate: bool = True,
        source_filters: Optional[List[str]] = None,
        use_semantic_reranking: Optional[bool] = None,
        use_cache: bool = True
    ) -> str:
        """
        Retrieve and format context for generation in a domain-agnostic way.
        
        Args:
            query: Explicit query (if provided, will bypass query formulation)
            topic: Main subject for retrieval
            query_type: Type of retrieval (standard, analytical, exploratory, etc.)
            additional_context: Additional data for query formulation
            top_k: Maximum number of chunks to retrieve
            threshold: Minimum similarity threshold
            deduplicate: Whether to deduplicate similar chunks
            source_filters: Optional list of sources to filter results
            use_semantic_reranking: Whether to apply semantic reranking to results
            use_cache: Whether to use cached context when available
            
        Returns:
            str: Formatted context for inclusion in prompts
            
        Raises:
            RuntimeError: If RAG engine is not set
        """
        if not self.rag_engine:
            raise RuntimeError("RAG engine not initialized. Call set_rag_engine() first.")
        
        # Check for auto-transition between modes
        self.check_and_switch_modes()
        
        # Check cache first if enabled
        cache_key = f"{query}:{topic}:{query_type}:{str(additional_context)}"
        if use_cache and cache_key in self.context_cache:
            self.log_activity("Using cached context")
            return self.context_cache[cache_key]
        
        # If explicit query is not provided, formulate one based on topic and type
        if not query and topic:
            # Use the query reformulation template if available
            reformulation_prompt = self.render_prompt_template(
                purpose="query_reformulation",
                original_query=topic,
                domain_context=additional_context or {}
            )
            
            reformulated_queries = self.generate(reformulation_prompt)
            
            # Try to extract JSON array of queries
            try:
                queries = Utils.extract_json_from_text(reformulated_queries)
                if isinstance(queries, list) and queries:
                    query = queries[0]  # Use first query
                else:
                    # Fallback to query formulator
                    query = self.query_formulator.formulate_query(
                        topic=topic,
                        query_type=query_type,
                        additional_context=additional_context
                    )
            except:
                # Fallback to query formulator
                query = self.query_formulator.formulate_query(
                    topic=topic,
                    query_type=query_type,
                    additional_context=additional_context
                )
        elif not query and not topic:
            raise ValueError("Either query or topic must be provided")
        
        self.log_activity(f"Retrieving context for: {query[:50]}...")
        
        # Track retrieval performance
        start_time = time.time()
        
        # Update query type statistics
        if query_type in self.retrieval_stats["query_types_used"]:
            self.retrieval_stats["query_types_used"][query_type] += 1
        else:
            self.retrieval_stats["query_types_used"][query_type] = 1
        
        # Retrieve relevant chunks with source filtering if specified
        retrieve_kwargs = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold
        }
        
        if source_filters:
            retrieve_kwargs["source_filter"] = source_filters
            
        results = self.rag_engine.retrieve(**retrieve_kwargs)
        
        # Apply semantic reranking if requested - use parameter or class setting
        should_rerank = use_semantic_reranking if use_semantic_reranking is not None else self.enable_semantic_reranking
        if should_rerank and len(results) > 1:
            results = self._semantic_rerank(results, query)
        
        # Track retrieved chunks for knowledge refinement
        for result in results:
            chunk_id = result.get("id", result.get("text")[:100])
            self.knowledge_metrics["recalled_chunks"].append(chunk_id)
            self.knowledge_metrics["chunk_frequency"][chunk_id] += 1
        
        # Calculate retrieval time
        retrieval_time = time.time() - start_time
        
        # Assemble context from chunks with appropriate strategy
        template_purpose = "sleep_phase" if self.operational_mode == OperationalMode.SLEEP else "context_presentation"
        
        context = self.context_assembler.assemble_context(
            chunks=results,
            deduplicate=deduplicate,
            include_metadata=self.operational_mode == OperationalMode.SLEEP,
            coherence_optimization=self.operational_mode == OperationalMode.SLEEP
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
        
        # Cache the result if caching is enabled
        if use_cache:
            self.context_cache[cache_key] = context
        
        # Log retrieval stats
        self.log_activity(
            f"Retrieved {len(results)} chunks in {retrieval_time:.3f}s "
            f"(avg: {self.retrieval_stats['average_retrieval_time']:.3f}s)"
        )
        
        return context
    
    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Implement knowledge consolidation for RAG agent.
        
        This analyzes retrieval patterns to improve future queries.
        
        Returns:
            Dict containing consolidation metrics
        """
        # Call parent implementation first
        parent_results = super().consolidate_knowledge()
        
        # Analyze chunk frequency to identify important information
        frequent_chunks = self.knowledge_metrics["chunk_frequency"].most_common(10)
        
        # Identify topics that need improvement
        total_retrievals = sum(self.knowledge_metrics["chunk_frequency"].values())
        weak_retrieval_topics = []
        
        # Calculate topic clusters and identify weak areas
        for query_type, count in self.retrieval_stats["query_types_used"].items():
            effectiveness = count / max(1, self.retrieval_stats["calls"])
            if effectiveness < 0.3:  # Arbitrary threshold
                weak_retrieval_topics.append(query_type)
                self.knowledge_metrics["weak_areas"].add(query_type)
                
        # Generate enhanced statistics for future improvement
        chunk_diversity = len(self.knowledge_metrics["chunk_frequency"]) / max(1, total_retrievals)
        
        # Custom RAG consolidation metrics
        rag_metrics = {
            "chunk_diversity": chunk_diversity,
            "frequent_chunks": frequent_chunks,
            "weak_retrieval_topics": weak_retrieval_topics,
            "reranking_effectiveness": self.retrieval_stats["reranking_improvements"] / max(1, self.retrieval_stats["calls"]),
            "avg_context_length": self.retrieval_stats["total_chunks"] / max(1, self.retrieval_stats["calls"])
        }
        
        # Use sleep phase template to generate consolidation insights
        consolidation_prompt = self.render_prompt_template(
            purpose="sleep_phase",
            cycle_number=self.cycle_count,
            data_summary=json.dumps(self.retrieval_stats, indent=2),
            focus_areas=[
                {"name": "Retrieval Optimization", "description": "Improve retrieval accuracy and relevance"},
                {"name": "Vector Relationships", "description": "Enhance semantic connections between chunks"},
                {"name": "Knowledge Gaps", "description": "Identify areas with incomplete information"}
            ],
            access_statistics=json.dumps(dict(self.knowledge_metrics["chunk_frequency"].most_common(20)), indent=2),
            previous_cycle_results=json.dumps(parent_results.get("cycle_results", {}), indent=2)
        )
        
        # Generate consolidation insights
        consolidation_response = self.generate(consolidation_prompt)
        
        # Try to extract structured data from response
        consolidation_data = Utils.extract_json_from_text(consolidation_response)
        
        # Merge metrics
        consolidated_results = {
            **parent_results,
            "rag_metrics": rag_metrics,
            "cycle_results": consolidation_data or {"raw_response": consolidation_response}
        }
        
        self.log_activity(f"RAG knowledge consolidation complete: identified {len(weak_retrieval_topics)} areas for improvement")
        return consolidated_results

    def set_rag_engine(self, engine: Engine) -> None:
        """
        Set the RAG engine for this agent.
        
        Args:
            engine: RAG engine to use
        """
        self.rag_engine = engine
        self.log_activity("RAG engine set")
    
    def create_enhanced_prompt(
        self,
        base_prompt: str,
        query: str = "",
        topic: str = "",
        query_type: str = "standard",
        additional_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        context_strategy: str = "prefix"
    ) -> str:
        """
        Create a RAG-enhanced prompt by adding retrieved context.
        
        Args:
            base_prompt: Original prompt to enhance
            query: Explicit query for retrieval (bypasses query formulation if provided)
            topic: Topic for context retrieval
            query_type: Type of query for retrieval
            additional_context: Additional context for query formulation
            top_k: Maximum number of chunks to retrieve
            context_strategy: How to incorporate context ("prefix", "suffix", "combined",
                              "sandwich", "framing", "reference")
            
        Returns:
            str: Enhanced prompt with retrieved context
        """
        # Update context strategy statistics
        if context_strategy in self.retrieval_stats["context_strategies_used"]:
            self.retrieval_stats["context_strategies_used"][context_strategy] += 1
        else:
            self.retrieval_stats["context_strategies_used"][context_strategy] = 1
            
        # Get relevant context
        context = self.retrieve_context(
            query=query,
            topic=topic,
            query_type=query_type,
            additional_context=additional_context,
            top_k=top_k
        )
        
        # Try to use the context presentation template first
        try:
            enhanced_prompt = self.render_prompt_template(
                purpose="context_presentation",
                query=base_prompt,
                formatted_chunks=context
            )
            return enhanced_prompt
        except Exception as e:
            # Fallback to manual context insertion based on strategy
            self.log_activity(f"Failed to use template for context presentation: {e}", level="warning")
        
        # Incorporate context based on strategy (fallback approach)
        if context_strategy == "prefix":
            # Context before prompt (traditional RAG approach)
            enhanced_prompt = f"Context information:\n\n{context}\n\nBased on the above context, {base_prompt}"
            
        elif context_strategy == "suffix":
            # Context after prompt
            enhanced_prompt = f"{base_prompt}\n\nUse the following context to inform your response:\n\n{context}"
            
        elif context_strategy == "combined":
            # Structured format with clear separation
            enhanced_prompt = (
                f"Context information:\n\n{context}\n\n"
                f"Task: {base_prompt}\n\n"
                f"Generate a response that uses the context information to complete the task."
            )
            
        elif context_strategy == "sandwich":
            # Context both before and after for emphasis
            enhanced_prompt = (
                f"Consider this context:\n\n{context}\n\n"
                f"{base_prompt}\n\n"
                f"Remember to incorporate relevant details from the context provided above."
            )
            
        elif context_strategy == "framing":
            # Context as a guiding frame
            enhanced_prompt = (
                f"You are tasked with the following:\n{base_prompt}\n\n"
                f"To help you, here is relevant information:\n\n{context}\n\n"
                f"Please provide a comprehensive response using the relevant information."
            )
            
        elif context_strategy == "reference":
            # Context as numbered reference material
            context_sections = context.split('\n\n')
            numbered_context = "\n\n".join([f"[{i+1}] {section}" for i, section in enumerate(context_sections)])
            enhanced_prompt = (
                f"Reference information:\n\n{numbered_context}\n\n"
                f"Using the numbered references above when relevant, respond to the following:\n{base_prompt}"
            )
            
        else:  # Default to prefix
            enhanced_prompt = f"Context information:\n\n{context}\n\nBased on the above context, {base_prompt}"
        
        return enhanced_prompt
    
    def process(self, input_data: Dict[str, Any]) -> str:
        """
        Process input data and produce output using RAG enhancement.
        
        This method is domain-agnostic and supports various input formats.
        
        Args:
            input_data: Dictionary with processing parameters including:
                - query: Optional explicit query string
                - topic: Main subject for retrieval (if no query provided)
                - query_type: Type of retrieval (standard, analytical, etc.)
                - base_prompt: Base prompt to enhance
                - additional_context: Optional dict with extra context
                - context_strategy: How to incorporate context
                - top_k: How many chunks to retrieve
                - source_filters: Optional list of sources to filter by
            
        Returns:
            str: Generated content
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Extract parameters with generic defaults
        query = input_data.get("query", "")
        topic = input_data.get("topic", "")
        query_type = input_data.get("query_type", "standard")
        base_prompt = input_data.get("base_prompt", "")
        additional_context = input_data.get("additional_context", {})
        context_strategy = input_data.get("context_strategy", "prefix")
        top_k = input_data.get("top_k", 5)
        source_filters = input_data.get("source_filters")
        
        # Basic validation - either need base_prompt, and either query or topic
        if not base_prompt:
            raise ValueError("Input data must include 'base_prompt'")
            
        if not query and not topic:
            raise ValueError("Input data must include either 'query' or 'topic'")
            
        # Create enhanced prompt
        enhanced_prompt = self.create_enhanced_prompt(
            base_prompt=base_prompt,
            query=query,
            topic=topic,
            query_type=query_type,
            additional_context=additional_context,
            top_k=top_k,
            context_strategy=context_strategy
        )
        
        # Generate content
        result = self.generate(enhanced_prompt)
        
        # Store result in agent memory
        self.store_result("last_generation", result)
        self.store_metadata("last_process_inputs", {
            "query_type": query_type,
            "context_strategy": context_strategy,
            "top_k": top_k
        })
        
        return result
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval performance statistics.
        
        Returns:
            Dict[str, Any]: Retrieval statistics
        """
        return self.retrieval_stats.copy()
    
    def clear_context_cache(self) -> None:
        """Clear the context cache to free memory."""
        cache_size = len(self.context_cache)
        self.context_cache.clear()
        self.log_activity(f"Cleared context cache ({cache_size} entries)")
