"""
Core RAG (Retrieval-Augmented Generation) system interface.

This module provides the main entry point for RAG functionality, integrating
the engine, embeddings, and generation components into a unified interface.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from oarc_rag.core.engine import RAGEngine
from oarc_rag.core.chunking import TextChunker
from oarc_rag.core.context import ContextAssembler
from oarc_rag.core.query import QueryFormulator
from oarc_rag.ai.client import OllamaClient
from oarc_rag.utils.log import log


class RAG:
    """
    High-level interface for Retrieval-Augmented Generation functionality.
    
    This class provides a simplified API for using RAG capabilities,
    integrating document processing, retrieval, and generation.
    """
    
    def __init__(
        self,
        embedding_model: str = "llama3.1:latest",
        generation_model: Optional[str] = None,
        run_id: Optional[str] = None,
        engine: Optional[RAGEngine] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Model to use for embeddings
            generation_model: Model to use for generation (uses embedding_model if None)
            run_id: Unique identifier for this run
            engine: Existing RAGEngine instance to use (creates new one if None)
            chunk_size: Default chunk size for text splitting
            chunk_overlap: Default chunk overlap for text splitting
            temperature: Temperature for generation
            max_tokens: Max tokens for generation
        """
        # Initialize RAG engine (or use the provided one)
        self.engine = engine or RAGEngine(
            embedding_model=embedding_model,
            run_id=run_id
        )
        
        # Initialize components
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.context_assembler = ContextAssembler()
        self.query_formulator = QueryFormulator()
        
        # Set up generation client
        self.generation_model = generation_model or embedding_model
        self.client = OllamaClient(self.generation_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        log.info(f"RAG system initialized with embedding model {embedding_model} and generation model {self.generation_model}")
    
    def add_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> List[str]:
        """
        Add document content to the knowledge base.
        
        Args:
            text: Document text content
            metadata: Optional metadata for the document
            source: Optional source identifier
            
        Returns:
            List of chunk IDs created
        """
        # Split text into chunks
        chunks = self.chunker.chunk_text(text)
        
        # Add chunks to engine with metadata
        return self.engine.add_document(
            text_chunks=chunks,
            metadata=metadata,
            source=source
        )
    
    def add_file(
        self, 
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Add a file to the knowledge base.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata for the document
            
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
            
        # Use engine's file processing capability
        return self.engine.add_file(
            file_path=path,
            metadata=metadata,
            source=path.name
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information based on query.
        
        Args:
            query: Query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold
            source_filter: Filter results by source
            rerank: Whether to apply semantic reranking
            
        Returns:
            List of retrieved chunks with similarity scores
        """
        # Optionally reformulate query for better retrieval
        reformulated_query = self.query_formulator.formulate_query(
            original_query=query,
            query_type="retrieval"
        )
        
        # Retrieve from engine
        return self.engine.retrieve(
            query=reformulated_query,
            top_k=top_k,
            threshold=threshold,
            source_filter=source_filter,
            rerank=rerank
        )
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response to a query using RAG.
        
        Args:
            query: User query
            context: Optional explicit context (retrieves automatically if None)
            system_prompt: Optional system prompt
            top_k: Maximum number of chunks to retrieve for context
            threshold: Minimum similarity threshold for retrieval
            source_filter: Optional source filter for retrieval
            temperature: Generation temperature (uses default if None)
            max_tokens: Maximum tokens to generate (uses default if None)
            
        Returns:
            Generated response
        """
        # Retrieve context if not provided
        if context is None:
            results = self.retrieve(
                query=query,
                top_k=top_k,
                threshold=threshold,
                source_filter=source_filter
            )
            
            # Assemble context from retrieved chunks
            context = self.context_assembler.assemble_context(
                chunks=results,
                query=query
            )
        
        # Create the complete prompt with context and query
        user_prompt = f"Context information:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Use system prompt if provided
        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            default_system = (
                "You are a helpful RAG-powered assistant. When answering questions, "
                "use only the provided context information. If the context doesn't "
                "contain relevant information, acknowledge that you don't have enough "
                "information to answer accurately."
            )
            prompt = f"{default_system}\n\n{user_prompt}"
        
        # Generate response
        return self.client.generate(
            prompt=prompt,
            model=self.generation_model,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
    
    def question_answering(
        self,
        question: str,
        source_filter: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Specialized method for question answering.
        
        Args:
            question: Question to answer
            source_filter: Filter by source
            temperature: Generation temperature (lower for factual responses)
            
        Returns:
            Dict with answer and source information
        """
        # Retrieve relevant information
        results = self.retrieve(
            query=question,
            top_k=3,
            source_filter=source_filter,
            rerank=True
        )
        
        # Check if we found relevant information
        if not results:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Format sources information
        sources = []
        for chunk in results:
            source = chunk.get("source", "Unknown")
            if source not in sources:
                sources.append(source)
        
        # Assemble context
        context = self.context_assembler.assemble_context(
            chunks=results,
            query=question
        )
        
        # Create specialized QA prompt
        prompt = (
            "You are a precise question answering assistant. Answer the question "
            "based only on the provided context information. If the context doesn't "
            "contain enough information to answer confidently, say so.\n\n"
            f"Context information:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # Generate answer
        answer = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=300
        )
        
        # Calculate crude confidence based on result similarity scores
        avg_similarity = sum(r.get("similarity", 0) for r in results) / len(results) if results else 0
        confidence = min(avg_similarity * 1.5, 1.0)  # Scale up but cap at 1.0
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunk_count": len(results)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the RAG system.
        
        Returns:
            Dictionary with metrics
        """
        return {
            "engine_metrics": self.engine.get_performance_metrics(),
            "chunker_metrics": self.chunker.get_metrics(),
            "query_metrics": self.query_formulator.get_metrics()
        }
    
    def clear_cache(self) -> None:
        """Clear all caches in the RAG system."""
        self.engine.clear_cache()
