"""
RAG engine core implementation.

This module provides the core RAG engine functionality, including document processing,
embedding generation, vector search, and retrieval operations.
"""
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from oarc_rag.core.chunking import TextChunker
from oarc_rag.core.context import ContextAssembler
from oarc_rag.core.database import VectorDatabase
from oarc_rag.core.embedding import EmbeddingGenerator
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log
from oarc_rag.utils.paths import Paths
from oarc_rag.utils.utils import Utils


@singleton
class RAGEngine:
    """
    Core engine for RAG (Retrieval-Augmented Generation) operations.
    
    This class manages the vector database, embedding generation,
    and retrieval operations for RAG functionality. It is designed
    as a singleton to ensure consistent state across the application.
    """
    
    def __init__(
        self,
        embedding_model: str = "llama3.1:latest",
        db_path: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        create_dirs: bool = True,
        use_cache: bool = True,
        use_pca: bool = False,
        pca_dimensions: Optional[int] = None,
        use_reranker: bool = True
    ):
        """
        Initialize the RAG engine.
        
        Args:
            embedding_model: Model to use for embeddings
            db_path: Path to vector database directory
            run_id: Unique identifier for this run
            create_dirs: Whether to create directories
            use_cache: Whether to use caching
            use_pca: Whether to use PCA for dimensionality reduction
            pca_dimensions: Number of dimensions for PCA reduction
            use_reranker: Whether to use semantic reranking
        """
        # Check if already initialized (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            log.info("RAGEngine already initialized, ignoring new parameters")
            return
            
        # Generate run ID if not provided
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        
        # Set up paths
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Paths.get_vector_db_directory() / self.run_id
            
        if create_dirs:
            self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Check for Ollama availability
        Utils.check_for_ollama(raise_error=True)
        
        # Ensure model exists
        Utils.validate_ollama_model(embedding_model, raise_error=True)
        
        # Initialize components
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        
        # Initialize vector database with PCA settings
        self.vector_db = VectorDatabase(
            use_pca=use_pca,
            pca_dimensions=pca_dimensions
        )
        
        # Initialize performance metrics
        self.metrics = {
            "ingestion": {
                "documents_added": 0,
                "chunks_added": 0,
                "total_tokens_embedded": 0,
                "embedding_time": 0.0,
            },
            "retrieval": {
                "queries": 0,
                "total_results": 0,
                "avg_results_per_query": 0.0,
                "avg_similarity": 0.0,
                "avg_query_time": 0.0,
                "reranking_applied": 0,
                "reranker_improvements": 0
            },
            "start_time": time.time()
        }
        
        # Configure reranking
        self.use_reranker = use_reranker
        
        # Set initialization flag
        self._initialized = True
        log.info(f"RAGEngine initialized with model {embedding_model} and run_id {self.run_id}")
    
    def add_document(
        self,
        text: Union[str, List[str]],
        text_chunks: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Add a document to the vector database.
        
        Args:
            text: Document text content or list of documents
            text_chunks: Optional pre-chunked text (uses text if not provided)
            metadata: Optional metadata for the document
            source: Optional source identifier
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            
        Returns:
            List of chunk IDs created
        """
        start_time = time.time()
        
        # Prepare text chunks
        chunks = text_chunks
        
        if chunks is None:
            if isinstance(text, str):
                # Split single document into chunks
                chunks = self.text_chunker.chunk_text(
                    text, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
            elif isinstance(text, list):
                # Process list of documents
                chunks = []
                for doc in text:
                    doc_chunks = self.text_chunker.chunk_text(
                        doc,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks.extend(doc_chunks)
            else:
                raise ValueError("Text must be a string or list of strings")
        
        # No chunks to add
        if not chunks:
            log.warning("No chunks to add to vector database")
            return []
            
        # Generate embeddings
        embedding_start = time.time()
        vectors = self.embedding_generator.embed_texts(chunks)
        embedding_time = time.time() - embedding_start
        
        # Estimate token count
        token_count = sum(len(chunk.split()) for chunk in chunks)
        
        # Add to vector database
        chunk_ids = self.vector_db.add_document(
            text_chunks=chunks,
            vectors=vectors,
            metadata=metadata,
            source=source
        )
        
        # Update metrics
        self.metrics["ingestion"]["documents_added"] += 1
        self.metrics["ingestion"]["chunks_added"] += len(chunks)
        self.metrics["ingestion"]["total_tokens_embedded"] += token_count
        self.metrics["ingestion"]["embedding_time"] += embedding_time
        
        total_time = time.time() - start_time
        log.info(f"Added document with {len(chunks)} chunks in {total_time:.2f}s")
        
        return chunk_ids
    
    def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> List[str]:
        """
        Add a file to the vector database.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata for the document
            source: Optional source identifier (uses filename if None)
            
        Returns:
            List of chunk IDs created
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file type is not supported
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Use filename as source if not provided
        source = source or path.name
            
        # Read file content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Add file metadata if not provided
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "file_name": path.name,
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "file_type": path.suffix.lstrip('.')
        })
            
        # Add to vector database
        return self.add_document(
            text=content,
            metadata=metadata,
            source=source
        )
    
    def retrieve(
        self,
        query: Union[str, List[float]],
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None,
        rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text or query vector
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold
            source_filter: Filter results by source
            rerank: Whether to apply semantic reranking
            
        Returns:
            List of retrieved chunks with similarity scores
        """
        start_time = time.time()
        
        # Generate query vector if query is text
        if isinstance(query, str):
            query_vector = self.embedding_generator.embed_text(query)
        else:
            query_vector = query
            
        # Apply reranking? Use instance default if not specified
        should_rerank = self.use_reranker if rerank is None else rerank
        
        # Search the vector database
        results = self.vector_db.search(
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
            source_filter=source_filter
        )
        
        # Optional semantic reranking
        if should_rerank and len(results) > 1:
            results = self._apply_semantic_reranking(results, query)
        
        # Update metrics
        query_time = time.time() - start_time
        self.metrics["retrieval"]["queries"] += 1
        self.metrics["retrieval"]["total_results"] += len(results)
        
        if len(results) > 0:
            avg_similarity = sum(r.get("similarity", 0) for r in results) / len(results)
            self.metrics["retrieval"]["avg_similarity"] = (
                (self.metrics["retrieval"]["avg_similarity"] * (self.metrics["retrieval"]["queries"] - 1)) + 
                avg_similarity
            ) / self.metrics["retrieval"]["queries"]
            
        self.metrics["retrieval"]["avg_results_per_query"] = (
            self.metrics["retrieval"]["total_results"] / self.metrics["retrieval"]["queries"]
        )
        
        self.metrics["retrieval"]["avg_query_time"] = (
            (self.metrics["retrieval"]["avg_query_time"] * (self.metrics["retrieval"]["queries"] - 1)) + 
            query_time
        ) / self.metrics["retrieval"]["queries"]
        
        log.info(f"Retrieved {len(results)} results for query in {query_time:.3f}s")
        return results
    
    def _apply_semantic_reranking(
        self, 
        results: List[Dict[str, Any]],
        query: Union[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Apply semantic reranking to retrieved results.
        
        Args:
            results: Original retrieval results
            query: Original query (text or vector)
            
        Returns:
            Reranked results
        """
        # Convert query vector to text if necessary
        if not isinstance(query, str):
            query_text = f"Search query vector with {len(query)} dimensions"
        else:
            query_text = query
        
        self.metrics["retrieval"]["reranking_applied"] += 1
        
        # Use context assembler to help with reranking
        context_assembler = ContextAssembler()
        
        # Simple reranking implementation - actual implementation would be more sophisticated
        # In practice, reranking is better done within a specialized agent module
        
        # For now, return results sorted by similarity
        sorted_results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Count as improvement if order changed
        if sorted_results != results:
            self.metrics["retrieval"]["reranker_improvements"] += 1
        
        return sorted_results
    
    def get_document_sources(self) -> List[str]:
        """
        Get a list of all document sources in the database.
        
        Returns:
            List of source identifiers
        """
        return self.vector_db.get_sources()
    
    def clear_cache(self) -> None:
        """Clear all caches in the engine."""
        self.embedding_generator.clear_cache()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the engine.
        
        Returns:
            Dictionary with metrics
        """
        uptime = time.time() - self.metrics["start_time"]
        
        # Calculate some derived metrics
        embedding_rate = 0
        if self.metrics["ingestion"]["embedding_time"] > 0:
            embedding_rate = (
                self.metrics["ingestion"]["total_tokens_embedded"] / 
                self.metrics["ingestion"]["embedding_time"]
            )
            
        retrieval_rate = 0
        if self.metrics["retrieval"]["avg_query_time"] > 0 and self.metrics["retrieval"]["queries"] > 0:
            retrieval_rate = 1.0 / self.metrics["retrieval"]["avg_query_time"]
            
        reranking_effectiveness = 0
        if self.metrics["retrieval"]["reranking_applied"] > 0:
            reranking_effectiveness = (
                self.metrics["retrieval"]["reranker_improvements"] / 
                self.metrics["retrieval"]["reranking_applied"]
            )
            
        return {
            "uptime_seconds": uptime,
            "run_id": self.run_id,
            "embedding_model": self.embedding_generator.model,
            "ingestion_metrics": {
                "documents_processed": self.metrics["ingestion"]["documents_added"],
                "chunks_processed": self.metrics["ingestion"]["chunks_added"],
                "tokens_embedded": self.metrics["ingestion"]["total_tokens_embedded"],
                "embedding_time": self.metrics["ingestion"]["embedding_time"],
                "embedding_rate_tokens_per_second": embedding_rate
            },
            "retrieval_metrics": {
                "queries_processed": self.metrics["retrieval"]["queries"],
                "avg_results_per_query": self.metrics["retrieval"]["avg_results_per_query"],
                "avg_similarity_score": self.metrics["retrieval"]["avg_similarity"],
                "avg_query_time": self.metrics["retrieval"]["avg_query_time"],
                "retrieval_rate_queries_per_second": retrieval_rate,
                "reranking": {
                    "applied": self.metrics["retrieval"]["reranking_applied"],
                    "improvements": self.metrics["retrieval"]["reranker_improvements"],
                    "effectiveness": reranking_effectiveness
                }
            },
            "database_metrics": self.vector_db.get_stats()
        }

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset singleton instance - for testing only."""
        if hasattr(cls, "_instance"):
            delattr(cls, "_instance")
