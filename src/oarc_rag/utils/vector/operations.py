"""
Vector operations utilities for the OARC-RAG system.

This module provides optimized vector operations for embeddings and similarity search,
with a focus on HNSW for approximate nearest neighbor search.
"""
import numpy as np
import time
from typing import List, Tuple, Any, Optional
import hnswlib

from oarc_rag.utils.log import log

# Set HNSW availability flag
HNSW_AVAILABLE = True

# Utility functions

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))

def inner_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute inner product (dot product) between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Inner product
    """
    return float(np.dot(vec1, vec2))

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a query vector and multiple vectors.
    
    Args:
        query: Query vector (1D array)
        vectors: Matrix of vectors (2D array)
        
    Returns:
        Array of similarity scores
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm > 0:
        query = query / query_norm
    
    # Normalize vectors
    vec_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vec_norms[vec_norms == 0] = 1
    normalized_vectors = vectors / vec_norms
    
    # Calculate similarities
    return np.dot(normalized_vectors, query)

def vector_quantize(vector: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Quantize a vector to reduce memory usage.
    
    Args:
        vector: Input vector
        bits: Bits per value (8 or 4)
        
    Returns:
        Quantized vector
    """
    if bits == 8:
        return vector.astype(np.float16).astype(np.float32)
    elif bits == 4:
        # Implement 4-bit quantization using scaling and rounding
        scale = max(abs(vector.max()), abs(vector.min()))
        if scale == 0:
            return vector
            
        # Scale to [-8, 7] for 4-bit signed integer range
        scaled = np.round((vector / scale) * 7).astype(np.int8)
        # Clip to 4-bit range
        scaled = np.clip(scaled, -8, 7)
        # Convert back to float
        return (scaled / 7) * scale
    else:
        return vector  # No quantization

def vector_dequantize(vector: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    """
    Restore a quantized vector to its original dtype.
    
    Args:
        vector: Quantized vector
        original_dtype: Original data type
        
    Returns:
        Dequantized vector
    """
    return vector.astype(original_dtype)

# HNSW Operations

def create_hnsw_index(
    vectors: List[List[float]], 
    space: str = "cosine", 
    ef_construction: int = 200,
    M: int = 16
) -> Any:
    """
    Create an HNSW index for efficient similarity search.
    
    Args:
        vectors: Vectors to index
        space: Distance space ('cosine', 'l2', 'ip')
        ef_construction: Index quality parameter (higher = better quality but slower construction)
        M: Index size parameter (higher = better recall but more memory)
        
    Returns:
        HNSW index object
        
    Raises:
        ImportError: If HNSW is not available
        ValueError: If vectors format is invalid
    """

    if not vectors:
        raise ValueError("No vectors provided to index")
        
    # Convert to numpy array
    np_vectors = np.array(vectors, dtype=np.float32)
    
    # Get dimensionality
    d = np_vectors.shape[1]
    
    # Create index
    index = hnswlib.Index(space=space, dim=d)
    
    # Initialize index
    max_elements = len(vectors)
    index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
    
    # Add vectors
    index.add_items(np_vectors)
    
    # Set search parameters
    index.set_ef(50)  # Higher values give more accurate but slower search
    
    return index

def hnsw_search(
    index: Any, 
    query_vector: List[float], 
    k: int = 5
) -> Tuple[List[float], List[int]]:
    """
    Search an HNSW index for similar vectors.
    
    Args:
        index: HNSW index created with create_hnsw_index
        query_vector: Query vector
        k: Number of results to return
        
    Returns:
        Tuple of (distances, indices)
    """
    if not HNSW_AVAILABLE:
        raise ImportError("hnswlib is not available")
        
    # Convert query to numpy array
    q_vec = np.array(query_vector, dtype=np.float32)
    
    # Search the index
    labels, distances = index.knn_query(q_vec, k=k)
    
    # Convert to Python lists
    return distances[0].tolist(), labels[0].tolist()

# PCA dimensionality reduction
def apply_pca(vectors: np.ndarray, n_components: int) -> Tuple[np.ndarray, Any]:
    """
    Apply PCA dimensionality reduction to vectors.
    
    Args:
        vectors: Input vectors (2D array)
        n_components: Number of components to keep
        
    Returns:
        Tuple of (reduced_vectors, pca_model)
        
    Raises:
        ImportError: If sklearn is not available
    """
    try:
        from sklearn.decomposition import PCA
        
        if vectors.shape[1] <= n_components:
            return vectors, None
            
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(vectors)
        
        return reduced, pca
        
    except ImportError:
        raise ImportError("sklearn is not available. Install with 'pip install scikit-learn'")

def transform_with_pca(vectors: np.ndarray, pca_model: Any) -> np.ndarray:
    """
    Transform vectors using a pre-fitted PCA model.
    
    Args:
        vectors: Input vectors (2D array)
        pca_model: Fitted PCA model from apply_pca
        
    Returns:
        Reduced vectors
    """
    if pca_model is None:
        return vectors
        
    return pca_model.transform(vectors)

# Vector cache utilities
class VectorCache:
    """Simple in-memory vector cache with LRU eviction policy."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize vector cache.
        
        Args:
            max_size: Maximum number of items in cache
        """
        self.max_size = max_size
        self.cache = {}  # key -> (vector, timestamp)
        self.access_order = []  # LRU order (least recently used at front)
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get vector from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Vector or None if not in cache
        """
        if key in self.cache:
            # Move to end (most recently used)
            self._update_access(key)
            return self.cache[key][0]
        return None
        
    def put(self, key: str, vector: np.ndarray) -> None:
        """
        Put vector into cache.
        
        Args:
            key: Cache key
            vector: Vector to cache
        """
        if key in self.cache:
            self.cache[key] = (vector, time.time())
            self._update_access(key)
        else:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict()
                
            # Add new item
            self.cache[key] = (vector, time.time())
            self.access_order.append(key)
            
    def _update_access(self, key: str) -> None:
        """Update LRU access order."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
    def _evict(self) -> None:
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]
                
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        
    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self.cache)

# Global vector cache instance
vector_cache = VectorCache()
