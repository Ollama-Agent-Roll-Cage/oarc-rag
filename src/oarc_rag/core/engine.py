"""
Retrieval-Augmented Generation engine.

This module provides the main RAG engine that coordinates vector storage,
retrieval, and context assembly for enhanced knowledge retrieval.
"""
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from collections import defaultdict

from llama_index.experimental.query_engine import PandasQueryEngine

from oarc_rag.utils.log import log
from oarc_rag.utils.utils import Utils
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.config.config import Config
from oarc_rag.core.database import VectorDatabase
from oarc_rag.core.embedding import EmbeddingGenerator
from oarc_rag.core.chunking import TextChunker
from oarc_rag.core.cache import cache_manager, QueryCache

# Optional import for HNSW
try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    log.warning("hnswlib not available, HNSW index will be disabled")

@singleton
class Engine:
    """
    Retrieval-Augmented Generation engine.
    
    This class integrates document processing, embedding generation,
    vector storage, and retrieval to support RAG for knowledge retrieval.
    Implements concepts from the Specification.md document.
    """
    
    @classmethod 
    def _reset_singleton(cls):
        """Reset the singleton instance for testing."""
        if hasattr(cls, '_instance'):
            delattr(cls, '_instance')
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        create_dirs: bool = True,
        use_hnsw: bool = None,
        hnsw_ef_construction: int = None,
        hnsw_ef_search: int = None,
        hnsw_m: int = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        use_pca: bool = None,
        pca_dimensions: int = None
    ):
        """
        Initialize the RAG engine with advanced options from Specification.md.
        
        Args:
            run_id: Unique identifier for this run
            base_dir: Base directory for vector storage
            embedding_model: Name of the embedding model to use (defaults to config)
            chunk_size: Size of text chunks in tokens (defaults to config)
            chunk_overlap: Overlap between consecutive chunks (defaults to config) 
            create_dirs: Whether to create directories if they don't exist
            use_hnsw: Whether to use HNSW for fast approximate search (defaults to config)
            hnsw_ef_construction: HNSW index construction parameter (defaults to config)
            hnsw_ef_search: HNSW search parameter (defaults to config)
            hnsw_m: HNSW graph connectivity parameter (defaults to config)
            use_cache: Whether to enable caching (defaults to config)
            cache_ttl: Time-to-live for cache entries in seconds (defaults to config)
            use_pca: Whether to use PCA dimensionality reduction (defaults to config)
            pca_dimensions: Target dimensions for PCA reduction (defaults to config)
            
        Raises:
            RuntimeError: If Ollama is not available
        """
        # Ensure Ollama is available before proceeding
        Utils.check_for_ollama(raise_error=True)
        
        # Get configuration values, with explicit parameters taking precedence
        config = Config()
        embedding_model = embedding_model or config.get('embedding_model', "llama3.1:latest")
        chunk_size = chunk_size or config.get('chunk_size', 512)
        chunk_overlap = chunk_overlap or config.get('chunk_overlap', 50)
        
        # Advanced options from Specification.md
        self.use_hnsw = use_hnsw if use_hnsw is not None else config.get('hnsw.enabled', True)
        self.hnsw_ef_construction = hnsw_ef_construction or config.get('hnsw.ef_construction', 200)
        self.hnsw_ef_search = hnsw_ef_search or config.get('hnsw.ef_search', 50)
        self.hnsw_m = hnsw_m or config.get('hnsw.m', 16)
        
        # Vector operation options
        self.use_pca = use_pca if use_pca is not None else config.get('vector.use_pca', False)
        self.pca_dimensions = pca_dimensions or config.get('vector.pca_dimensions', 128)
        
        # Caching options
        self.use_cache = use_cache if use_cache is not None else config.get('caching.query_cache_enabled', True)
        self.cache_ttl = cache_ttl or config.get('caching.query_cache_ttl', 3600)
        
        # Initialize configuration
        self.config = {
            "run_id": run_id or str(int(time.time())),
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_hnsw": self.use_hnsw,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "hnsw_m": self.hnsw_m,
            "use_pca": self.use_pca,
            "pca_dimensions": self.pca_dimensions,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "created_at": time.time()
        }

        # Use config values consistently
        self.run_id = self.config["run_id"]
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "output" / self.run_id
        self.vector_dir = self.base_dir / "vectors"

        # Create components using config values
        db_path = self.vector_dir / "vector.db"
        self.vector_db = VectorDatabase(db_path)
        self.embedder = EmbeddingGenerator(
            model_name=self.config["embedding_model"],
            use_pca=self.config["use_pca"],
            pca_dimensions=self.config["pca_dimensions"],
        )
        self.chunker = TextChunker(
            chunk_size=self.config["chunk_size"],
            overlap=self.config["chunk_overlap"]
        )
        
        # Initialize HNSW index if enabled
        self.hnsw_index = None
        if self.use_hnsw and HNSW_AVAILABLE:
            self._initialize_hnsw_index()
            
        # Get query cache from cache manager
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.query_cache = cache_manager.query_cache
        
        # Save config to disk if directories should be created
        if create_dirs:
            self._save_config()
        
        log.info(f"Initialized RAG engine for run {self.run_id} with model {self.config['embedding_model']}")
        
        # Initialize LlamaIndex components
        try:
            from oarc_rag.core.llama import setup_llama_index
            setup_llama_index()
            log.debug("LlamaIndex components initialized")
        except Exception as e:
            log.warning(f"LlamaIndex setup failed: {e}")

        # DataFrame storage for PandasQueryEngine
        self.dataframes = {}
        
        # Statistics tracking
        self.stats = {
            "documents_added": 0,
            "chunks_added": 0,
            "queries_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hnsw_queries": 0,
            "vector_queries": 0,
            "total_query_time": 0.0
        }
        
    def _initialize_hnsw_index(self) -> None:
        """Initialize the HNSW index for fast approximate search."""
        if not HNSW_AVAILABLE:
            log.warning("HNSW not available, using vector database search instead")
            return
            
        try:
            # Initialize index when first vectors are added
            self.hnsw_index = None
            self.hnsw_dimension = None
            self.hnsw_ids_to_db_ids = {}  # Map HNSW index IDs to vector DB IDs
            self.hnsw_db_ids_to_ids = {}  # Inverse mapping
            
            log.info(f"HNSW initialized with parameters: ef_construction={self.hnsw_ef_construction}, m={self.hnsw_m}")
        except Exception as e:
            log.error(f"HNSW initialization failed: {e}")
            self.use_hnsw = False
            
    def _save_config(self) -> None:
        """Save RAG configuration to disk."""
        self.vector_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        config_path = self.vector_dir / "metadata.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def _update_hnsw_index(self, vectors: List[List[float]], db_ids: List[str]) -> None:
        """
        Update the HNSW index with new vectors.
        
        Args:
            vectors: New vectors to add
            db_ids: Database IDs corresponding to the vectors
        """
        if not self.use_hnsw or not HNSW_AVAILABLE or not vectors:
            return
            
        try:
            # Initialize index if this is the first update
            if self.hnsw_index is None:
                dim = len(vectors[0])
                self.hnsw_dimension = dim
                self.hnsw_index = hnswlib.Index(space='cosine', dim=dim)
                self.hnsw_index.init_index(
                    max_elements=10000,  # Initial capacity
                    ef_construction=self.hnsw_ef_construction,
                    M=self.hnsw_m
                )
                self.hnsw_index.set_ef(self.hnsw_ef_search)
                
            # Add new vectors
            start_id = len(self.hnsw_ids_to_db_ids)
            hnsw_ids = list(range(start_id, start_id + len(vectors)))
            
            # Map IDs
            for hnsw_id, db_id in zip(hnsw_ids, db_ids):
                self.hnsw_ids_to_db_ids[hnsw_id] = db_id
                self.hnsw_db_ids_to_ids[db_id] = hnsw_id
                
            # Add to index
            self.hnsw_index.add_items(
                data=np.array(vectors, dtype=np.float32),
                ids=np.array(hnsw_ids)
            )
                
            log.debug(f"Added {len(vectors)} vectors to HNSW index")
            
        except Exception as e:
            log.error(f"HNSW index update failed: {e}")
            
    def _search_hnsw(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search the HNSW index for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of search results with vector DB IDs and distances
        """
        if not self.use_hnsw or not HNSW_AVAILABLE or self.hnsw_index is None:
            return []
            
        try:
            # Get similar vector IDs and distances
            hnsw_ids, distances = self.hnsw_index.knn_query(
                data=np.array([query_vector], dtype=np.float32),
                k=min(top_k, len(self.hnsw_ids_to_db_ids))
            )
            
            # Convert distances (cosine) to similarities
            similarities = 1.0 - distances[0]
            
            # Map HNSW IDs to database IDs
            results = [
                {
                    "db_id": self.hnsw_ids_to_db_ids[int(hnsw_id)],
                    "similarity": float(sim)
                }
                for hnsw_id, sim in zip(hnsw_ids[0], similarities)
            ]
            
            self.stats["hnsw_queries"] += 1
            return results
            
        except Exception as e:
            log.error(f"HNSW search failed: {e}")
            return []

    def add_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> int:
        """
        Process and add a document to the vector store.
        
        Args:
            text: Document text content
            metadata: Additional document metadata
            source: Source identifier (file, URL, etc.)
            
        Returns:
            int: Number of chunks added to the database
            
        Raises:
            RuntimeError: If vectorization or database operation fails
        """
        if not text or not text.strip():
            log.warning("Attempted to add empty document")
            return 0
            
        # Prepare metadata
        doc_metadata = metadata or {}
        if source:
            doc_metadata["source"] = source
        
        # Clean and chunk the text
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            log.warning("No chunks created from document text")
            return 0
            
        # Generate embeddings - will raise RuntimeError if Ollama is not available
        try:
            vectors = self.embedder.embed_texts(chunks)
        except Exception as e:
            log.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
        
        # Store in database
        try:
            # Add to vector database
            chunk_ids = self.vector_db.add_document(
                text_chunks=chunks,
                vectors=vectors, 
                metadata=doc_metadata,
                source=source or "unknown"
            )
            
            # Update HNSW index if enabled
            if self.use_hnsw and HNSW_AVAILABLE:
                self._update_hnsw_index(vectors, chunk_ids)
                
            # Update stats
            self.stats["documents_added"] += 1
            self.stats["chunks_added"] += len(chunks)
            
            log.info(f"Added document with {len(chunks)} chunks"
                    f"{' from ' + source if source else ''}")
            return len(chunks)
            
        except Exception as e:
            log.error(f"Failed to add document to database: {e}")
            raise RuntimeError(f"Database operation failed: {e}")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None,
        use_cache: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query with advanced options from Specification.md.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity score to include (0-1)
            source_filter: Optional filter for specific sources
            use_cache: Whether to use query cache (defaults to engine setting)
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
            
        Raises:
            RuntimeError: If embedding generation or search fails
        """
        start_time = time.time()
        use_cache = self.use_cache if use_cache is None else use_cache
        log.debug(f"Retrieving context for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Create a cache key that includes all parameters
        cache_params = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "source_filter": source_filter
        }
        
        # Check cache if enabled
        if use_cache:
            cached_results = self.query_cache.get_results(cache_params)
            if cached_results is not None:
                self.stats["cache_hits"] += 1
                log.debug(f"Using cached results for query: {query[:30]}...")
                return cached_results
            self.stats["cache_misses"] += 1
        
        # Generate query embedding - will raise RuntimeError if Ollama is not available
        try:
            query_vector = self.embedder.embed_text(query)
        except Exception as e:
            log.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
        
        # Search for similar chunks using optimal strategy
        try:
            results = []
            
            # Try HNSW first if enabled (fast approximate search)
            if self.use_hnsw and HNSW_AVAILABLE and self.hnsw_index is not None:
                hnsw_results = self._search_hnsw(query_vector, top_k)
                
                if hnsw_results:
                    # Get full results from database using IDs
                    db_ids = [r["db_id"] for r in hnsw_results]
                    similarities = {r["db_id"]: r["similarity"] for r in hnsw_results}
                    
                    chunks = self.vector_db.get_by_ids(db_ids)
                    
                    # Add similarity scores
                    for chunk in chunks:
                        chunk["similarity"] = similarities.get(chunk["id"], 0.0)
                    
                    # Apply threshold
                    results = [c for c in chunks if c["similarity"] >= threshold]
                    
                    # Apply source filter if specified
                    if source_filter:
                        sources = [source_filter] if isinstance(source_filter, str) else source_filter
                        results = [r for r in results if r.get("source") in sources]
            
            # Fall back to vector database search if needed
            if not results:
                self.stats["vector_queries"] += 1
                results = self.vector_db.search(
                    query_vector, 
                    top_k=top_k,
                    threshold=threshold,
                    source_filter=source_filter
                )
            
            # Cache results if enabled
            if use_cache:
                self.query_cache.add(cache_params, results)
                
            # Update stats
            self.stats["queries_processed"] += 1
            self.stats["total_query_time"] += (time.time() - start_time)
            
            log.debug(f"Retrieved {len(results)} chunks in {time.time() - start_time:.4f}s")
            return results
            
        except Exception as e:
            log.error(f"Failed to search vector database: {e}")
            raise RuntimeError(f"Vector search failed: {e}")

    def store_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Store a DataFrame for querying with PandasQueryEngine.
        
        Args:
            name: Name to associate with this DataFrame
            df: The DataFrame to store
        """
        self.dataframes[name] = df
        log.info(f"Stored DataFrame '{name}' with shape {df.shape}")
        
    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a stored DataFrame by name.
        
        Args:
            name: Name of the DataFrame to retrieve
            
        Returns:
            The DataFrame or None if not found
        """
        return self.dataframes.get(name)
        
    def list_dataframes(self) -> List[str]:
        """
        List all available DataFrame names.
        
        Returns:
            List of DataFrame names
        """
        return list(self.dataframes.keys())
        
    def create_pandas_query_engine(self, df_name: str = None, df: pd.DataFrame = None) -> Optional[PandasQueryEngine]:
        """
        Create a PandasQueryEngine for natural language queries on DataFrames.
        
        Args:
            df_name: Name of a stored DataFrame to use
            df: DataFrame to use directly (takes precedence over df_name)
            
        Returns:
            PandasQueryEngine or None if the DataFrame is not available
            
        Raises:
            ValueError: If neither df_name nor df is provided
        """
        if df is None and df_name is None:
            raise ValueError("Must provide either df_name or df parameter")
            
        # Use provided DataFrame or retrieve by name
        dataframe = df if df is not None else self.get_dataframe(df_name)
        
        if dataframe is None:
            log.warning(f"DataFrame '{df_name}' not found")
            return None
            
        try:
            query_engine = PandasQueryEngine(dataframe)
            return query_engine
        except Exception as e:
            log.error(f"Failed to create PandasQueryEngine: {e}")
            return None
    
    def query_dataframe(self, question: str, df_name: str = None, df: pd.DataFrame = None) -> Any:
        """
        Query a DataFrame using natural language through PandasQueryEngine.
        
        Args:
            question: Natural language question about the data
            df_name: Name of a stored DataFrame to query
            df: DataFrame to query directly (takes precedence over df_name)
            
        Returns:
            Query response from PandasQueryEngine
            
        Raises:
            ValueError: If neither df_name nor df is provided
        """
        query_engine = self.create_pandas_query_engine(df_name=df_name, df=df)
        if query_engine is None:
            raise ValueError(f"Could not create query engine for DataFrame")
            
        try:
            response = query_engine.query(question)
            return response
        except Exception as e:
            log.error(f"Failed to query DataFrame: {e}")
            raise RuntimeError(f"DataFrame query failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Dict with statistics about engine usage
        """
        # Add derived stats
        avg_query_time = 0.0
        if self.stats["queries_processed"] > 0:
            avg_query_time = self.stats["total_query_time"] / self.stats["queries_processed"]
            
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_cache_requests
            
        # Get component stats
        vector_db_stats = self.vector_db.get_stats()
        embedding_stats = self.embedder.get_performance_metrics() if hasattr(self.embedder, "get_performance_metrics") else {}
        
        # Combine all stats
        combined_stats = {
            **self.stats,
            "avg_query_time": avg_query_time,
            "cache_hit_rate": cache_hit_rate,
            "avg_chunks_per_doc": (self.stats["chunks_added"] / self.stats["documents_added"]) 
                                 if self.stats["documents_added"] > 0 else 0,
            "vector_database": vector_db_stats,
            "embeddings": embedding_stats,
            "hnsw_enabled": self.use_hnsw and HNSW_AVAILABLE,
            "hnsw_index_size": len(self.hnsw_ids_to_db_ids) if hasattr(self, "hnsw_ids_to_db_ids") else 0,
            "cache_entries": len(self.query_cache),
            "config": self.config
        }
        
        return combined_stats

    def purge(self) -> None:
        """Purge all data and reset the engine."""
        # Reset vector database
        self.vector_db.clear()
        
        # Reset HNSW index
        if self.use_hnsw and HNSW_AVAILABLE:
            self._initialize_hnsw_index()
            
        # Clear DataFrame storage
        self.dataframes = {}
        
        # Reset caches
        self.query_cache = {}
        self.query_cache_ttl = {}
        
        # Reset statistics
        self.stats = {
            "documents_added": 0,
            "chunks_added": 0,
            "queries_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hnsw_queries": 0,
            "vector_queries": 0,
            "total_query_time": 0.0
        }
        
        log.info("Engine data purged")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'Engine':
        """
        Create Engine from a configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Engine instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                
            return cls(**config)
        except Exception as e:
            raise ValueError(f"Invalid config file: {e}")

    def add_document_with_llama(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a document using LlamaIndex document loaders.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            int: Number of chunks added to the database
        """
        try:
            from oarc_rag.core.llama import LlamaDocumentLoader
            
            loader = LlamaDocumentLoader()
            documents = loader.load_document(file_path)
            
            total_chunks = 0
            for doc in documents:
                # Process each document with base method
                meta = {**(metadata or {}), **doc.metadata}
                chunk_count = self.add_document(doc.text, meta, source=str(file_path))
                total_chunks += chunk_count
                
            return total_chunks
            
        except Exception as e:
            log.error(f"Failed to load document with LlamaIndex: {e}")
            raise RuntimeError(f"Document loading failed: {e}")
