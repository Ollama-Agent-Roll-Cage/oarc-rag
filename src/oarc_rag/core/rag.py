"""
Core RAG (Retrieval-Augmented Generation) functionality.

This module provides a high-level interface to the RAG system,
coordinating document processing, embedding generation, retrieval,
and content generation.
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from oarc_rag.utils.log import log
from oarc_rag.core.engine import RAGEngine
from oarc_rag.core.context import ContextAssembler
from oarc_rag.core.query import QueryFormulator
from oarc_rag.core.monitor import RAGMonitor
from oarc_rag.ai.client import OllamaClient


class RAG:
    """
    Main RAG system interface.
    
    This class provides a simplified interface to the RAG system,
    coordinating all the components needed for document processing,
    retrieval, and enhanced content generation.
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        embedding_model: str = "llama3.1:latest",
        generation_model: str = "llama3.1:latest",
        temperature: float = 0.7,
        monitor_enabled: bool = True
    ):
        """
        Initialize the RAG system.
        
        Args:
            run_id: Unique identifier for this run
            embedding_model: Name of the embedding model to use
            generation_model: Model to use for text generation
            temperature: Temperature setting for generation
            monitor_enabled: Whether to enable performance monitoring
        """
        # Create unique run ID if not provided
        self.run_id = run_id or str(int(time.time()))
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine(
            run_id=self.run_id,
            embedding_model=embedding_model
        )
        
        # Initialize AI components
        self.client = OllamaClient(default_model=generation_model)
        self.temperature = temperature
        
        # Initialize support components
        self.context_assembler = ContextAssembler(format_style="markdown")
        self.query_formulator = QueryFormulator()
        
        # Performance monitoring
        self.monitor = RAGMonitor() if monitor_enabled else None
        if self.monitor:
            log.info("RAG performance monitoring enabled")
            
    def add_document(
        self,
        text: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> int:
        """
        Add document(s) to the knowledge base.
        
        Args:
            text: Document text content or list of text chunks
            metadata: Additional document metadata
            source: Source identifier (file, URL, etc.)
            
        Returns:
            int: Number of chunks added to the database
        """
        start_time = time.time()
        
        # Record the operation in monitoring
        operation_id = None
        if self.monitor:
            operation_id = self.monitor.start_ingestion()
        
        try:
            # Handle single text or list of texts
            if isinstance(text, list):
                total_chunks = 0
                for i, chunk in enumerate(text):
                    chunk_source = f"{source}_{i}" if source else f"chunk_{i}"
                    chunks_added = self.rag_engine.add_document(
                        text=chunk,
                        metadata=metadata,
                        source=chunk_source
                    )
                    total_chunks += chunks_added
                result = total_chunks
            else:
                result = self.rag_engine.add_document(
                    text=text,
                    metadata=metadata,
                    source=source
                )
                
            # Record successful operation
            if self.monitor and operation_id:
                duration = time.time() - start_time
                self.monitor.record_ingestion(
                    operation_id=operation_id,
                    document_source=source,
                    chunks_added=result,
                    duration=duration
                )
                
            return result
            
        except Exception as e:
            # Record failed operation
            if self.monitor and operation_id:
                duration = time.time() - start_time
                self.monitor.record_ingestion(
                    operation_id=operation_id,
                    document_source=source,
                    chunks_added=0,
                    duration=duration,
                    error=str(e)
                )
            raise
            
    def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a file to the knowledge base using LlamaIndex document loaders.
        
        Args:
            file_path: Path to the file to add
            metadata: Additional metadata to store
            
        Returns:
            int: Number of chunks added to the database
        """
        start_time = time.time()
        
        # Record the operation in monitoring
        operation_id = None
        if self.monitor:
            operation_id = self.monitor.start_ingestion()
            
        try:
            result = self.rag_engine.add_document_with_llama(
                file_path=file_path,
                metadata=metadata
            )
            
            # Record successful operation
            if self.monitor and operation_id:
                duration = time.time() - start_time
                self.monitor.record_ingestion(
                    operation_id=operation_id,
                    document_source=str(file_path),
                    chunks_added=result,
                    duration=duration
                )
                
            return result
            
        except Exception as e:
            # Record failed operation
            if self.monitor and operation_id:
                duration = time.time() - start_time
                self.monitor.record_ingestion(
                    operation_id=operation_id,
                    document_source=str(file_path),
                    chunks_added=0,
                    duration=duration,
                    error=str(e)
                )
            raise
            
    def retrieve(
        self,
        query: str,
        query_type: str = "general",
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query or question
            query_type: Type of query for formulation
            top_k: Maximum number of chunks to retrieve
            threshold: Minimum similarity score (0-1)
            source_filter: Filter results by source
            
        Returns:
            Tuple of (retrieved chunks, formatted context)
        """
        start_time = time.time()
        
        # Record the operation in monitoring
        retrieval_id = None
        if self.monitor:
            retrieval_id = self.monitor.start_retrieval()
            
        try:
            # Formulate an effective query
            formulated_query = self.query_formulator.formulate_query(
                topic=query,
                query_type=query_type
            )
            
            # Retrieve relevant chunks
            results = self.rag_engine.retrieve(
                query=formulated_query,
                top_k=top_k,
                threshold=threshold,
                source_filter=source_filter
            )
            
            # Assemble context from chunks
            context = self.context_assembler.assemble_context(
                chunks=results,
                deduplicate=True
            )
            
            # Record successful operation
            if self.monitor and retrieval_id:
                duration = time.time() - start_time
                self.monitor.record_retrieval(
                    retrieval_id=retrieval_id,
                    query=query,
                    formulated_query=formulated_query,
                    results=results,
                    duration=duration
                )
                
            return results, context
            
        except Exception as e:
            # Record failed operation
            if self.monitor and retrieval_id:
                duration = time.time() - start_time
                self.monitor.record_retrieval(
                    retrieval_id=retrieval_id,
                    query=query,
                    formulated_query=query,  # Use original if formulation failed
                    results=[],
                    duration=duration,
                    error=str(e)
                )
            raise
            
    def generate_with_context(
        self,
        prompt: str,
        query: str,
        query_type: str = "general",
        max_tokens: int = 1000,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate content with RAG context enhancement.
        
        Args:
            prompt: Base prompt for generation
            query: Query to retrieve relevant context
            query_type: Type of query for retrieval
            max_tokens: Maximum tokens in the generated response
            top_k: Maximum number of chunks to retrieve
            threshold: Minimum similarity score (0-1)
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Record the operation in monitoring
        generation_id = None
        if self.monitor:
            generation_id = self.monitor.start_generation()
            
        try:
            # Retrieve context
            results, context = self.retrieve(
                query=query,
                query_type=query_type,
                top_k=top_k,
                threshold=threshold
            )
            
            # Create enhanced prompt
            enhanced_prompt = f"""Context information:

{context}

Based on the above context, {prompt}"""
            
            # Generate response
            response = self.client.generate(
                prompt=enhanced_prompt,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            result = {
                "generated_text": response,
                "metadata": {
                    "query": query,
                    "context_chunks": len(results),
                    "generation_time": time.time() - start_time
                }
            }
            
            # Record successful operation
            if self.monitor and generation_id:
                duration = time.time() - start_time
                self.monitor.record_generation(
                    generation_id=generation_id,
                    prompt_length=len(enhanced_prompt),
                    response_length=len(response),
                    duration=duration
                )
                
            return result
            
        except Exception as e:
            # Record failed operation
            if self.monitor and generation_id:
                duration = time.time() - start_time
                self.monitor.record_generation(
                    generation_id=generation_id,
                    prompt_length=len(prompt) if isinstance(prompt, str) else 0,
                    response_length=0,
                    duration=duration,
                    error=str(e)
                )
            raise
            
    def query(
        self,
        question: str,
        query_type: str = "general",
        max_tokens: int = 1000,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG system.
        
        This is a convenience method that combines retrieval and generation
        for question answering.
        
        Args:
            question: User question to answer
            query_type: Type of query for retrieval
            max_tokens: Maximum tokens in the generated response
            top_k: Maximum number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        prompt = f"Please answer the following question accurately and thoroughly: {question}"
        
        result = self.generate_with_context(
            prompt=prompt,
            query=question,
            query_type=query_type,
            max_tokens=max_tokens,
            top_k=top_k
        )
        
        return result
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the RAG system.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.monitor:
            return {"monitoring_enabled": False}
            
        return self.monitor.get_metrics()
        
    def purge(self) -> None:
        """
        Remove all data from the RAG system.
        
        This is a destructive operation and will delete all stored documents.
        """
        self.rag_engine.purge()
        
        # Reset monitoring
        if self.monitor:
            self.monitor.reset_metrics()
            
        log.info("Purged all data from RAG system")
