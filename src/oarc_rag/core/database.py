"""
Vector database implementation for RAG.

This module provides a comprehensive implementation of vector storage and
retrieval for the RAG pipeline, supporting HNSW backend and
advanced retrieval techniques from Specification.md and Big_Brain.md.
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import uuid
import pickle

# Optional import for HNSW backend
try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

from oarc_rag.utils.log import log
from oarc_rag.utils.utils import Utils


class VectorDatabase:
    """
    Vector database for storing and retrieving document chunks.
    
    This class implements vector storage with HNSW backend with
    configurable parameters for optimizing retrieval performance.
    """
    
    def __init__(
        self, 
        db_path: Union[str, Path], 
        vector_dim: int = 4096,
        distance_metric: str = "cosine", 
        auto_save: bool = True,
        in_memory_mode: bool = False,
        ef_construction: int = 200,  # HNSW index construction parameter
        ef_search: int = 50,         # HNSW search parameter
        M: int = 16                  # HNSW graph connectivity
    ):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the database file
            vector_dim: Dimension of the embedding vectors
            distance_metric: Distance metric for similarity ('cosine', 'l2', 'ip')
            auto_save: Whether to automatically save changes
            in_memory_mode: Whether to keep entire index in memory
            ef_construction: HNSW index construction quality parameter
            ef_search: HNSW search quality parameter
            M: HNSW graph connectivity parameter
            
        Raises:
            ImportError: If HNSW is not available
        """ 
        self.db_path = Path(db_path)
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric
        self.auto_save = auto_save
        self.in_memory_mode = in_memory_mode
        
        # HNSW specific parameters
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        
        # Create parent directory if it doesn't exist
        os.makedirs(self.db_path.parent, exist_ok=True)
        
        # Metadata storage
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        
        # ID mapping
        self.id_map = {}  # Maps backend IDs to database IDs
        self.reverse_id_map = {}  # Maps database IDs to backend IDs
        
        # Initialize the HNSW backend
        self._initialize_hnsw()
        
        # Stats tracking
        self.stats = {
            "chunks_stored": 0,
            "chunks_deleted": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_search_time": 0.0,
            "search_count": 0,
            "avg_search_time": 0.0,
            "last_save_time": 0.0,
            "backend": "hnsw"
        }
    
    def _initialize_hnsw(self) -> None:
        """Initialize HNSW backend."""
        # Create new index - will be initialized when first vectors are added
        self.index = None
        self.id_map = {}
        self.reverse_id_map = {}
        
        # Load existing index if available
        meta_path = self.db_path.with_suffix(".meta")
        if self.db_path.exists() and meta_path.exists() and not self.in_memory_mode:
            try:
                # Load metadata first to get dimensions
                with open(meta_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                
                # Get ID mapping from metadata
                self.id_map = self.metadata_store.get("__id_map__", {})
                self.reverse_id_map = {v: k for k, v in self.id_map.items()}
                
                # Create index with proper dimensions
                space = "cosine" if self.distance_metric == "cosine" else "l2"
                self.index = hnswlib.Index(space=space, dim=self.vector_dim)
                
                # Load the index data
                self.index.load_index(str(self.db_path), max_elements=len(self.id_map))
                self.index.set_ef(self.ef_search)
                
                self.stats["chunks_stored"] = len(self.id_map)
                log.info(f"Loaded existing HNSW index with {self.stats['chunks_stored']} items")
            except Exception as e:
                log.error(f"Error loading HNSW index: {e}")
                self.index = None
                self.id_map = {}
                self.reverse_id_map = {}
    
    def add_document(
        self, 
        text_chunks: List[str], 
        vectors: List[List[float]], 
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> List[str]:
        """
        Add document chunks with vectors to the database.
        
        Args:
            text_chunks: List of text chunks
            vectors: List of embedding vectors for each chunk
            metadata: Metadata for the document (applied to all chunks)
            source: Source identifier for the document
            
        Returns:
            List of chunk IDs
            
        Raises:
            ValueError: If lengths of chunks and vectors don't match
        """
        if len(text_chunks) != len(vectors):
            raise ValueError("Number of text chunks must match number of vectors")
            
        if not text_chunks:
            return []
            
        # Generate chunk IDs
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
        
        # Add chunks to HNSW
        self._add_to_hnsw(chunk_ids, text_chunks, vectors, metadata, source)
            
        # Update stats
        self.stats["chunks_stored"] += len(chunk_ids)
        
        # Auto-save if enabled
        if self.auto_save:
            self.save()
            
        return chunk_ids
    
    def _add_to_hnsw(
        self, 
        chunk_ids: List[str], 
        text_chunks: List[str], 
        vectors: List[List[float]], 
        metadata: Optional[Dict[str, Any]], 
        source: Optional[str]
    ) -> None:
        """Add chunks to HNSW backend."""
        # Initialize index if this is the first add
        if self.index is None:
            space = "cosine" if self.distance_metric == "cosine" else "l2"
            self.index = hnswlib.Index(space=space, dim=self.vector_dim)
            
            # Initialize with capacity (can be increased later)
            max_elements = max(1000, len(vectors) * 2)  # Start with reasonable capacity
            
            self.index.init_index(
                max_elements=max_elements, 
                ef_construction=self.ef_construction, 
                M=self.M
            )
            
            # Set query time parameters
            self.index.set_ef(self.ef_search)
        
        # Prepare vectors as numpy array
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Normalize if using cosine similarity
        if self.distance_metric == "cosine":
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Prevent division by zero
            vectors_array = vectors_array / norms
        
        # Get current index size for new IDs
        start_idx = len(self.id_map)
        backend_ids = list(range(start_idx, start_idx + len(vectors)))
        
        # Add to index
        self.index.add_items(vectors_array, backend_ids)
        
        # Store metadata and ID mappings
        for i, (chunk_id, text, vector, backend_id) in enumerate(zip(chunk_ids, text_chunks, vectors, backend_ids)):
            # Map IDs
            self.id_map[backend_id] = chunk_id
            self.reverse_id_map[chunk_id] = backend_id
            
            # Store metadata
            chunk_metadata = {
                "text": text,
                "id": chunk_id,
                "source": source,
                "timestamp": int(time.time()),
                "backend_idx": backend_id
            }
            
            # Add document metadata if provided
            if metadata:
                chunk_metadata.update(metadata)
                
            # Store in metadata store
            self.metadata_store[chunk_id] = chunk_metadata
        
        # Update id map in metadata store for persistence
        self.metadata_store["__id_map__"] = self.id_map
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_vector: Query embedding vector
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            source_filter: Optional filter for specific sources
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
        """
        start_time = time.time()
        
        try:
            # Convert source filter to list if it's a string
            if source_filter and isinstance(source_filter, str):
                source_filter = [source_filter]
                
            # Search using HNSW backend
            results = self._search_hnsw(query_vector, top_k, threshold)
                
            # Apply source filtering if specified
            if source_filter:
                results = [r for r in results if r.get("source") in source_filter]
                
            # Update stats
            search_time = time.time() - start_time
            self.stats["successful_searches"] += 1
            self.stats["total_search_time"] += search_time
            self.stats["search_count"] += 1
            self.stats["avg_search_time"] = (
                self.stats["total_search_time"] / self.stats["search_count"]
            )
            
            return results
            
        except Exception as e:
            self.stats["failed_searches"] += 1
            log.error(f"Error searching vectors: {e}")
            return []
    
    def _search_hnsw(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search using HNSW backend."""
        if self.index is None or len(self.id_map) == 0:
            return []
            
        # Prepare query vector
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # Normalize if using cosine similarity
        if self.distance_metric == "cosine":
            norm = np.linalg.norm(query_np)
            if norm > 0:
                query_np = query_np / norm
        
        # Limit k to the number of items in the index
        k = min(top_k, len(self.id_map))
        
        # Query the index
        try:
            labels, distances = self.index.knn_query(query_np, k=k)
            
            # Process results
            results = []
            
            # Labels is a 2D array, get the first row
            labels = labels[0]
            distances = distances[0]
            
            for i, (idx, distance) in enumerate(zip(labels, distances)):
                # Convert distance to similarity
                if self.distance_metric == "cosine":
                    # HNSW cosine distance is 1-cos(angle)
                    similarity = 1.0 - distance
                elif self.distance_metric == "l2":
                    # Euclidean distance
                    similarity = 1.0 / (1.0 + distance)
                else:
                    # Inner product (already a similarity)
                    similarity = -distance  # HNSW returns negative IP for max similarity
                    
                # Skip if below threshold
                if similarity < threshold:
                    continue
                    
                # Get chunk ID from mapping
                chunk_id = self.id_map.get(int(idx))
                if not chunk_id or chunk_id not in self.metadata_store:
                    continue
                    
                # Get metadata
                metadata = self.metadata_store[chunk_id]
                
                result = {
                    "id": chunk_id,
                    "text": metadata.get("text", ""),
                    "similarity": float(similarity),
                    "source": metadata.get("source")
                }
                
                # Add other metadata fields
                for k, v in metadata.items():
                    if k not in result and k not in ["text", "backend_idx"]:
                        result[k] = v
                        
                results.append(result)
                
            return results
            
        except Exception as e:
            log.error(f"HNSW search error: {e}")
            return []
    
    def get_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of chunk dictionaries
        """
        results = []
        
        # Get from metadata store
        for chunk_id in chunk_ids:
            if chunk_id in self.metadata_store:
                metadata = self.metadata_store[chunk_id]
                result = {
                    "id": chunk_id,
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source")
                }
                
                # Add other metadata fields
                for k, v in metadata.items():
                    if k not in result and k not in ["backend_idx"]:
                        result[k] = v
                        
                results.append(result)
                
        return results
    
    def save(self) -> None:
        """Save database changes to disk."""
        # Save HNSW index
        if hasattr(self, 'index') and self.index is not None:
            # Save the index
            self.index.save_index(str(self.db_path))
            
            # Save metadata
            meta_path = self.db_path.with_suffix(".meta")
            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
                
            self.stats["last_save_time"] = time.time()
            log.info(f"Saved HNSW index with {self.stats['chunks_stored']} vectors")
    
    def close(self) -> None:
        """Close database connections and release resources."""
        # Save any changes before closing
        if self.auto_save:
            self.save()
    
    def clear(self) -> None:
        """Clear all data from the database."""
        # Re-initialize HNSW
        self.index = None
        self._initialize_hnsw()
            
        # Reset stats
        self.stats["chunks_stored"] = 0
        self.stats["chunks_deleted"] = 0
        
        # Clear metadata
        self.metadata_store = {}
        self.id_map = {}
        self.reverse_id_map = {}
        
        log.info("Cleared all data from HNSW database")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict of database statistics
        """
        return self.stats.copy()
    
    def resize_index(self, new_size: int) -> None:
        """
        Resize the index capacity.
        
        Args:
            new_size: New maximum number of elements
            
        Raises:
            ValueError: If new size is not greater than current size
        """
        if not hasattr(self, 'index') or self.index is None:
            return
            
        current_size = self.index.get_max_elements()
        if new_size <= current_size:
            log.warning(f"New size {new_size} is not greater than current size {current_size}")
            return
            
        # Resize the index
        self.index.resize_index(new_size)
        log.info(f"Resized HNSW index from {current_size} to {new_size} elements")
