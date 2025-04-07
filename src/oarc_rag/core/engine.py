"""
Retrieval-Augmented Generation engine.

This module provides the main RAG engine that coordinates vector storage,
retrieval, and context assembly for enhanced knowledge retrieval.
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

from oarc_rag.utils.log import log
from oarc_rag.utils.utils import check_for_ollama
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.core.database import VectorDatabase
from oarc_rag.core.embedding import EmbeddingGenerator
from oarc_rag.core.chunking import TextChunker
from oarc_rag.core.llama import LlamaDocumentLoader, setup_llama_index

@singleton
class Engine:
    """
    Retrieval-Augmented Generation engine.
    
    This class integrates document processing, embedding generation,
    vector storage, and retrieval to support RAG for knowledge retrieval.
    """
    
    @classmethod 
    def _reset_singleton(cls):
        """Reset the singleton instance (primarily for testing)."""
        if cls in cls._instances:
            del cls._instances[cls]
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
        embedding_model: str = "llama3.1:latest",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        create_dirs: bool = True
    ):
        """
        Initialize the RAG engine.
        
        Args:
            run_id: Unique identifier for this run
            base_dir: Base directory for vector storage
            embedding_model: Name of the embedding model to use
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between consecutive chunks in tokens
            create_dirs: Whether to create directories if they don't exist
            
        Raises:
            RuntimeError: If Ollama is not available
        """
        # Ensure Ollama is available before proceeding
        check_for_ollama(raise_error=True)
        
        # Initialize configuration
        self.config = {
            "run_id": run_id or str(int(time.time())),
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": time.time()
        }

        # Use config values consistently
        self.run_id = self.config["run_id"]
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "output" / self.run_id
        self.vector_dir = self.base_dir / "vectors"

        # Create components using config values
        db_path = self.vector_dir / "vector.db"
        self.vector_db = VectorDatabase(db_path)
        self.embedder = EmbeddingGenerator(model_name=self.config["embedding_model"])
        self.chunker = TextChunker(
            chunk_size=self.config["chunk_size"],
            overlap=self.config["chunk_overlap"]
        )
        
        # Save config to disk if directories should be created
        if create_dirs:
            self._save_config()
        
        log.info(f"Initialized RAG engine for run {self.run_id} with model {self.config['embedding_model']}")
        
        # Initialize LlamaIndex components
        try:
            setup_llama_index()
            self.llama_loader = LlamaDocumentLoader()
            log.info("Successfully initialized LlamaIndex integration")
        except Exception as e:
            log.warning(f"Failed to initialize LlamaIndex integration: {e}")
            self.llama_loader = None

        # DataFrame storage for PandasQueryEngine
        self.dataframes = {}
        
    def _save_config(self) -> None:
        """Save RAG configuration to disk."""
        import json
        
        self.vector_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        config_path = self.vector_dir / "metadata.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

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
            log.warning("Empty document received, skipping")
            return 0
            
        # Prepare metadata
        doc_metadata = metadata or {}
        if source:
            doc_metadata["source"] = source
        
        # Clean and chunk the text
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            log.warning("No chunks produced from document")
            return 0
            
        # Generate embeddings - will raise RuntimeError if Ollama is not available
        try:
            embeddings = self.embedder.embed_texts(chunks)
            log.debug(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            log.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
        
        # Store in database
        try:
            chunk_ids = self.vector_db.add_document(
                text_chunks=chunks,
                vectors=embeddings,
                metadata=doc_metadata,
                source=source
            )
            log.info(f"Added document with {len(chunk_ids)} chunks to vector database")
            return len(chunk_ids)
        except Exception as e:
            log.error(f"Failed to store document in vector database: {e}")
            raise RuntimeError(f"Vector database operation failed: {e}")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity score to include (0-1)
            source_filter: Optional filter for specific sources
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
            
        Raises:
            RuntimeError: If embedding generation or search fails
        """
        log.debug(f"Retrieving context for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Generate query embedding - will raise RuntimeError if Ollama is not available
        try:
            query_embedding = self.embedder.embed_text(query)
            log.debug("Generated query embedding")
        except Exception as e:
            log.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")
        
        # Search for similar chunks
        try:
            results = self.vector_db.search(
                query_vector=query_embedding,
                top_k=top_k,
                threshold=threshold,
                source_filter=source_filter
            )
            log.info(f"Retrieved {len(results)} chunks for query")
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
            raise ValueError("Must provide either df_name or df")
            
        # Use provided DataFrame or retrieve by name
        dataframe = df if df is not None else self.get_dataframe(df_name)
        
        if dataframe is None:
            log.warning(f"DataFrame '{df_name}' not found")
            return None
            
        try:
            query_engine = PandasQueryEngine(dataframe)
            log.info(f"Created PandasQueryEngine for DataFrame with shape {dataframe.shape}")
            return query_engine
        except Exception as e:
            log.error(f"Failed to create PandasQueryEngine: {e}")
            raise RuntimeError(f"PandasQueryEngine creation failed: {e}")
    
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
            raise ValueError("Could not create query engine: DataFrame not available")
            
        try:
            log.info(f"Querying DataFrame with question: {question}")
            response = query_engine.query(question)
            return response
        except Exception as e:
            log.error(f"Error querying DataFrame: {e}")
            raise RuntimeError(f"DataFrame query failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine and database.
        
        Returns:
            Dict[str, Any]: Statistics about chunks, documents, etc.
        """
        try:
            db_stats = self.vector_db.get_stats()
            embedder_stats = self.embedder.get_cache_stats() if hasattr(self.embedder, "get_cache_stats") else {}
            
            stats = {
                "run_id": self.run_id,
                "created_at": self.config["created_at"],
                "database": db_stats,
                "embedding": embedder_stats,
                "dataframes": {
                    "count": len(self.dataframes),
                    "names": list(self.dataframes.keys())
                },
                "config": {k: v for k, v in self.config.items() if k != "created_at"}
            }
            
            return stats
        except Exception as e:
            log.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def purge(self) -> None:
        """
        Remove all data from the vector database.
        
        This is a destructive operation and will delete all vectors and chunks.
        """
        try:
            self.vector_db.purge()
            self.dataframes.clear()  # Also clear stored DataFrames
            log.info("Purged all data from RAG engine")
        except Exception as e:
            log.error(f"Failed to purge database: {e}")
            raise RuntimeError(f"Database purge failed: {e}")

    @classmethod
    def load(cls, run_id: str, base_dir: Optional[Union[str, Path]] = None) -> 'Engine':
        """
        Load an existing RAG engine by run ID.
        
        Args:
            run_id: Run ID of the engine to load
            base_dir: Base directory (defaults to current working directory)
            
        Returns:
            Engine: Loaded RAG engine instance
            
        Raises:
            FileNotFoundError: If the engine data cannot be found
            RuntimeError: If loading fails
        """
        import json
        
        # Determine base directory
        base = Path(base_dir) if base_dir else Path.cwd() / "output"
        vector_dir = base / run_id / "vectors"
        config_path = vector_dir / "metadata.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"RAG engine configuration not found for run ID: {run_id}")
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Create new engine with loaded configuration
            engine = cls(
                run_id=run_id,
                base_dir=base,
                embedding_model=config.get("embedding_model", "llama3.1:latest"),
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50)
            )
            
            log.info(f"Loaded RAG engine for run {run_id}")
            return engine
            
        except Exception as e:
            log.error(f"Failed to load RAG engine: {e}")
            raise RuntimeError(f"Failed to load RAG engine: {e}")

    def add_document_with_llama(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a document using LlamaIndex's document loader.
        
        This method supports various file formats including PDF, DOCX, etc.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata to store with the document
            
        Returns:
            int: Number of chunks added to the database
            
        Raises:
            RuntimeError: If document loading or processing fails
            ValueError: If LlamaIndex integration is not available
        """
        if not self.llama_loader:
            raise ValueError("LlamaIndex integration not available")
            
        try:
            # Load document using LlamaIndex
            documents = self.llama_loader.load(file_path)
            
            if not documents:
                log.warning(f"No content loaded from {file_path}")
                return 0
                
            # Extract source name from file path
            source = Path(file_path).name
            
            # Prepare metadata - merge with file metadata
            combined_metadata = metadata or {}
            combined_metadata["file_path"] = str(file_path)
            combined_metadata["file_type"] = Path(file_path).suffix.lower()
            
            # Process each document section
            total_chunks = 0
            for doc in documents:
                # Get text from document
                text = doc.text if hasattr(doc, "text") else str(doc)
                
                # Add document metadata if available
                doc_metadata = combined_metadata.copy()
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    doc_metadata.update(doc.metadata)
                    
                # Add to vector database
                chunks_added = self.add_document(
                    text=text,
                    metadata=doc_metadata,
                    source=source
                )
                total_chunks += chunks_added
                
            log.info(f"Added {total_chunks} chunks from {file_path}")
            return total_chunks
            
        except Exception as e:
            log.error(f"Failed to process document {file_path}: {e}")
            raise RuntimeError(f"Document processing failed: {e}")
