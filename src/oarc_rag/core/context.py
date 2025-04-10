"""
Context assembly for RAG prompts.

This module provides utilities for assembling retrieved chunks into coherent
context for LLM prompts, implementing advanced techniques for improved relevance,
organization, and synthesis based on concepts from Specification.md and Big_Brain.md.
"""
import re
import time
from typing import List, Dict, Any, Optional, Union, Callable
from collections import defaultdict

from oarc_rag.utils.log import log
from oarc_rag.core.cache import cache_manager


class ContextAssembler:
    """
    Assemble retrieved chunks into coherent context for LLM prompts.
    
    This class implements multiple strategies for organizing and formatting
    retrieved context to maximize its usefulness for LLMs, with specialized
    techniques for different operational modes from Big_Brain.md.
    """
    
    def __init__(
        self,
        max_context_length: int = 10000,
        similarity_threshold: float = 0.85,
        optimization_level: str = "balanced",
        include_chunk_metadata: bool = False,
        include_chunk_sources: bool = True,
        include_citations: bool = True,
        organizing_strategy: str = "relevance",
        use_cache: bool = True
    ):
        """
        Initialize the context assembler.
        
        Args:
            max_context_length: Maximum context length in characters
            similarity_threshold: Threshold for deduplication similarity
            optimization_level: Context assembly depth ('low', 'balanced', 'high')
            include_chunk_metadata: Whether to include metadata with chunks
            include_chunk_sources: Whether to include source information
            include_citations: Whether to include citations
            organizing_strategy: How to organize chunks ('relevance', 'chronological', 'causal')
            use_cache: Whether to use context cache
        """
        self.max_context_length = max_context_length
        self.similarity_threshold = similarity_threshold
        self.optimization_level = optimization_level
        self.include_chunk_metadata = include_chunk_metadata
        self.include_chunk_sources = include_chunk_sources
        self.include_citations = include_citations
        self.organizing_strategy = organizing_strategy
        self.use_cache = use_cache
        
        # Performance metrics
        self.performance_metrics = {
            "contexts_assembled": 0,
            "total_assembly_time": 0.0,
            "avg_assembly_time": 0.0,
            "chunks_processed": 0,
            "chunks_deduplicated": 0,
            "contexts_cached": 0,
            "cache_hits": 0
        }
        
        # Get the context cache from the cache manager
        self.context_cache = cache_manager.context_cache
    
    def assemble_context(
        self,
        chunks: List[Dict[str, Any]], 
        deduplicate: bool = True,
        include_metadata: bool = None,
        coherence_optimization: bool = None,
        citation_format: str = "numeric"
    ) -> str:
        """
        Assemble retrieved chunks into a coherent context.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            deduplicate: Whether to deduplicate similar chunks
            include_metadata: Override include_metadata setting
            coherence_optimization: Apply coherence optimization (transitional phrases)
            citation_format: Format for citations ('numeric', 'author-date', 'footnote')
            
        Returns:
            Assembled context text
        """
        if not chunks:
            return ""
        
        # Track performance
        start_time = time.time()
        
        # Use instance defaults if not overridden
        include_metadata = include_metadata if include_metadata is not None else self.include_chunk_metadata
        
        # Deduplicate if requested
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
            
        # Sort chunks by similarity score (most relevant first)
        chunks = self._sort_chunks(chunks)
        
        # Format individual chunks
        formatted_chunks = []
        sources = set()
        
        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk(
                chunk, i+1,
                include_metadata=include_metadata, 
                citation_format=citation_format
            )
            formatted_chunks.append(formatted)
            
            # Track sources for bibliography
            if self.include_chunk_sources and "source" in chunk:
                sources.add(chunk["source"])
        
        # Apply coherence optimization if requested
        if coherence_optimization or (coherence_optimization is None and self.optimization_level == "high"):
            formatted_chunks = self._add_coherence_elements(formatted_chunks)
            
        # Join chunks with spacing
        context = "\n\n".join(formatted_chunks)
        
        # Add bibliography if appropriate
        if self.include_chunk_sources and sources:
            bibliography = self._generate_bibliography(sources)
            context += f"\n\n{bibliography}"
            
        # Update metrics
        self.performance_metrics["contexts_assembled"] += 1
        self.performance_metrics["chunks_processed"] += len(chunks)
        assembly_time = time.time() - start_time
        self.performance_metrics["total_assembly_time"] += assembly_time
        self.performance_metrics["avg_assembly_time"] = (
            self.performance_metrics["total_assembly_time"] / 
            self.performance_metrics["contexts_assembled"]
        )
        
        return context
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate chunks based on text similarity.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Deduplicated chunks
        """
        if not chunks:
            return []
            
        if len(chunks) == 1:
            return chunks
            
        # Implementation of text similarity for deduplication
        unique_chunks = []
        duplicate_count = 0
        
        # Simple filtering approach first - exact duplicates
        seen_texts = set()
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text in seen_texts:
                duplicate_count += 1
                continue
                
            seen_texts.add(text)
            unique_chunks.append(chunk)
            
        # If we're in "low" optimization mode, stop here
        if self.optimization_level == "low":
            self.performance_metrics["chunks_deduplicated"] += duplicate_count
            return unique_chunks
        
        # Now check for near-duplicates using cosine similarity
        try:
            # Only import if needed (for "balanced" or "high")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            if len(unique_chunks) <= 1:
                return unique_chunks
                
            # Extract texts for vectorization
            texts = [chunk.get("text", "").strip() for chunk in unique_chunks]
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute pairwise cosine similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Identify chunks to keep
            to_keep = [True] * len(unique_chunks)
            
            # Mark duplicates for removal, prioritizing chunks with higher similarity scores
            chunks_with_scores = [(chunk, chunk.get("similarity", 0), i) 
                                for i, chunk in enumerate(unique_chunks)]
            # Sort by similarity score (highest first)
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Iterate through chunks in order of relevance
            for i in range(len(chunks_with_scores)):
                if not to_keep[chunks_with_scores[i][2]]:
                    continue  # Skip if already marked for removal
                    
                chunk_idx = chunks_with_scores[i][2]
                
                # Check similarity with other chunks
                for j in range(len(chunks_with_scores)):
                    if i == j or not to_keep[chunks_with_scores[j][2]]:
                        continue
                        
                    other_idx = chunks_with_scores[j][2]
                    
                    # If similarity exceeds threshold, mark the less relevant chunk for removal
                    if similarity_matrix[chunk_idx, other_idx] > self.similarity_threshold:
                        to_keep[other_idx] = False
                        duplicate_count += 1
            
            # Keep only non-duplicate chunks
            result = [chunk for i, chunk in enumerate(unique_chunks) if to_keep[i]]
            
            self.performance_metrics["chunks_deduplicated"] += duplicate_count
            return result
            
        except ImportError:
            log.warning("sklearn not available for advanced deduplication")
            return unique_chunks
        except Exception as e:
            log.warning(f"Error in chunk deduplication: {e}")
            return unique_chunks
    
    def _sort_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort chunks according to the organizing strategy.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Sorted chunks
        """
        if not chunks:
            return []
            
        if self.organizing_strategy == "relevance":
            # Sort by similarity score (most relevant first)
            return sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
            
        elif self.organizing_strategy == "chronological":
            # Sort by timestamp or position if available
            return sorted(chunks, key=lambda x: x.get("timestamp", x.get("position", 0)))
            
        elif self.organizing_strategy == "causal":
            # Sort by causal relationship (using a position field for now)
            # This would be more sophisticated in a real implementation
            return sorted(chunks, key=lambda x: x.get("position", 0))
            
        else:  # Default to relevance
            return sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)
    
    def _format_chunk(
        self, 
        chunk: Dict[str, Any], 
        index: int, 
        include_metadata: bool = False,
        citation_format: str = "numeric"
    ) -> str:
        """
        Format a single chunk for inclusion in context.
        
        Args:
            chunk: Chunk dictionary
            index: Chunk index (for citations)
            include_metadata: Whether to include metadata
            citation_format: Citation format
            
        Returns:
            Formatted chunk text
        """
        # Get the text content
        text = chunk.get("text", "").strip()
        if not text:
            return ""
            
        # Format citation if needed
        citation = ""
        if self.include_citations:
            if citation_format == "numeric":
                citation = f"[{index}]"
            elif citation_format == "author-date":
                source = chunk.get("source", "Unknown")
                year = chunk.get("year", "n.d.")
                citation = f"({source}, {year})"
            elif citation_format == "footnote":
                citation = f"*{index}"
                
        # Add source information
        source_info = ""
        if self.include_chunk_sources and "source" in chunk:
            source = chunk["source"]
            source_info = f"\nSource: {source}"
            
        # Add metadata if requested
        metadata_info = ""
        if include_metadata and self.include_chunk_metadata:
            metadata = {}
            # Include select metadata fields
            for key in ["page", "chapter", "section", "author", "date", "category"]:
                if key in chunk and chunk[key]:
                    metadata[key] = chunk[key]
                    
            if metadata:
                metadata_parts = [f"{k.capitalize()}: {v}" for k, v in metadata.items()]
                metadata_info = f"\nMetadata: {' | '.join(metadata_parts)}"
        
        # Combine everything
        if citation:
            formatted = f"{citation} {text}"
        else:
            formatted = text
            
        if source_info or metadata_info:
            formatted += f"{source_info}{metadata_info}"
            
        return formatted
    
    def _add_coherence_elements(self, chunks: List[str]) -> List[str]:
        """
        Add transitional phrases between chunks for coherence.
        
        Args:
            chunks: Formatted chunk texts
            
        Returns:
            Chunks with transitional elements
        """
        if not chunks or len(chunks) <= 1:
            return chunks
            
        # Only use in high optimization mode
        if self.optimization_level != "high":
            return chunks
            
        # Simple transitional phrases
        transitions = [
            "Additionally,",
            "Furthermore,",
            "Moreover,",
            "In relation to this,",
            "Building on this information,",
            "Similarly,",
            "In contrast,",
            "To elaborate further,",
            "Another relevant point is that",
            "It's also worth noting that"
        ]
        
        # Add transitions to chunks
        coherent_chunks = [chunks[0]]  # Keep first chunk unchanged
        
        for i in range(1, len(chunks)):
            # Select transition phrase
            transition = transitions[i % len(transitions)]
            
            # Get the chunk and check if it already starts with a transition
            chunk = chunks[i]
            
            # Skip adding transition if the chunk already starts with certain phrases
            skip_patterns = ["additionally", "furthermore", "moreover", "similarly", "in contrast", "however"]
            if any(chunk.lower().startswith(pattern) for pattern in skip_patterns):
                coherent_chunks.append(chunk)
            else:
                # Add transition to beginning of chunk (if it has a citation)
                match = re.match(r'^(\[\d+\]|\(.+?\)|\*\d+)\s+(.+)$', chunk)
                if match:
                    citation, text = match.groups()
                    coherent_chunks.append(f"{citation} {transition} {text}")
                else:
                    coherent_chunks.append(f"{transition} {chunk}")
                    
        return coherent_chunks
    
    def _generate_bibliography(self, sources: set) -> str:
        """
        Generate a bibliography from the sources.
        
        Args:
            sources: Set of source identifiers
            
        Returns:
            Bibliography text
        """
        if not sources:
            return ""
            
        bibliography = "Sources:\n"
        for i, source in enumerate(sorted(sources)):
            bibliography += f"{i+1}. {source}\n"
            
        return bibliography
    
    def set_optimization_level(self, level: str) -> None:
        """
        Set the optimization level for context assembly.
        
        Args:
            level: Optimization level ('low', 'balanced', 'high')
            
        Raises:
            ValueError: If level is not recognized
        """
        valid_levels = ['low', 'balanced', 'high']
        if level not in valid_levels:
            raise ValueError(f"Invalid optimization level: {level}. Valid options: {valid_levels}")
            
        self.optimization_level = level
        
        # Adjust parameters based on level
        if level == 'low':
            self.include_chunk_metadata = False
            self.include_citations = False
        elif level == 'balanced':
            self.include_chunk_metadata = True
            self.include_citations = True
        elif level == 'high':
            self.include_chunk_metadata = True
            self.include_citations = True
            # Add more sophisticated processing for high
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for context assembly.
        
        Returns:
            Dict of performance metrics
        """
        return self.performance_metrics.copy()
