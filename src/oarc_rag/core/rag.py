"""
High-level RAG (Retrieval-Augmented Generation) interface.

This module provides a simplified interface to the RAG system,
combining vector storage, embedding generation, and retrieval
components for easy-to-use knowledge retrieval.
"""
from pathlib import Path
import time
import os
from typing import List, Dict, Any, Union, Optional, Tuple

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.core.engine import Engine
from oarc_rag.core.database import VectorDatabase
from oarc_rag.core.embedding import EmbeddingGenerator
from oarc_rag.core.chunking import TextChunker
from oarc_rag.ai.client import OllamaClient
from oarc_rag.utils.config.config import Config


class RAG:
    """
    High-level interface for Retrieval-Augmented Generation.
    
    This class provides a simplified interface to the RAG system,
    making it easy to add documents and perform RAG operations.
    """
    
    def __init__(
        self,
        vector_dir: Optional[Union[str, Path]] = None,
        embedding_model: str = "llama3.1:latest",
        generation_model: str = "llama3.1:latest",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        temperature: float = 0.7,
        verbose: bool = False,
        use_pca: bool = False,
        pca_dimensions: int = 128,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        hnsw_m: int = 16
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_dir: Directory for vector storage (defaults to config)
            embedding_model: Model for generating embeddings
            generation_model: Model for text generation
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between text chunks in tokens
            temperature: Temperature for generation
            verbose: Whether to log detailed information
            use_pca: Whether to use PCA for dimensionality reduction
            pca_dimensions: Number of dimensions for PCA
            hnsw_ef_construction: HNSW index construction quality parameter
            hnsw_ef_search: HNSW search quality parameter
            hnsw_m: HNSW graph connectivity parameter
        """
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.temperature = temperature
        self.verbose = verbose
        
        # Set up vector directory
        config = Config()
        if vector_dir is None:
            vector_dir = config.get('vector_dir', os.path.join(os.getcwd(), "vectors"))
        self.vector_dir = Path(vector_dir)
        os.makedirs(self.vector_dir, exist_ok=True)
        
        # Initialize RAG Engine with HNSW parameters
        self.engine = Engine(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_dir=self.vector_dir,
            use_pca=use_pca,
            pca_dimensions=pca_dimensions if use_pca else None,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            hnsw_m=hnsw_m
        )
        
        # Create a direct vector database with HNSW backend
        db_path = self.vector_dir / "rag_vectors.db"
        self.vector_db = VectorDatabase(
            db_path=db_path,
            auto_save=True,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            M=hnsw_m
        )
        
        # Create embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            use_pca=use_pca,
            pca_dimensions=pca_dimensions if use_pca else None
        )
        
        # Create text chunker
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        # Create Ollama client for generation
        self.client = OllamaClient(default_model=generation_model)
        
        # Performance tracking
        self.stats = {
            "documents_added": 0,
            "queries_processed": 0,
            "generations_performed": 0,
            "total_chunks": 0,
            "total_generation_time": 0.0,
            "total_retrieval_time": 0.0
        }
        
        if verbose:
            log.info(f"RAG system initialized with HNSW backend")
            log.info(f"Vector storage location: {self.vector_dir}")
    
    # ... other methods remain unchanged
