"""
Embedding generation for vector representations of text.
"""
import time
import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from oarc_rag.ai.client import OllamaClient
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.utils import Utils
from oarc_rag.utils.log import log
from oarc_rag.core.cache import cache_manager


@singleton
class EmbeddingGenerator:
    """
    Generate embeddings for text using Ollama's embedding API.
    
    This class manages embedding generation for converting text chunks into
    vector representations suitable for similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:latest",
        cache_embeddings: bool = True,
        max_cache_size: int = 1000,
        use_pca: bool = False,  # From Specification.md
        pca_dimensions: int = 128,  # From Specification.md
        normalization_enabled: bool = True,  # From Specification.md
        quantization_enabled: bool = False,  # From Specification.md
        batch_size: int = 32  # From Specification.md
    ):
        """
        Initialize the embedding generator with a model.
        
        Args:
            model_name: Name of the model to use for embeddings
            cache_embeddings: Whether to cache embeddings to avoid duplicate processing
            max_cache_size: Maximum number of embeddings to cache
            use_pca: Whether to use PCA dimensionality reduction
            pca_dimensions: Number of dimensions for PCA reduction
            normalization_enabled: Whether to normalize vectors
            quantization_enabled: Whether to quantize vectors to save memory
            batch_size: Maximum batch size for embedding generation
            
        Raises:
            RuntimeError: If Ollama embedding API is not available
        """
        # Ensure Ollama is available - will raise RuntimeError if not
        Utils.check_for_ollama()
        
        self.model_name = model_name
        self.client = OllamaClient()
        self.cache_embeddings = cache_embeddings
        
        # Get embedding cache from cache manager
        self.embedding_cache = cache_manager.embedding_cache
        
        # Advanced vector operations from Specification.md
        self.use_pca = use_pca
        self.pca_dimensions = pca_dimensions
        self.normalization_enabled = normalization_enabled
        self.quantization_enabled = quantization_enabled
        self.batch_size = batch_size
        self.pca_model = None
        
        # Performance tracking
        self._performance_metrics = {
            "total_embeddings": 0,
            "total_time": 0.0,
            "avg_time_per_embedding": 0.0,
            "pca_reductions": 0,
            "quantizations": 0,
            "normalizations": 0
        }
        
        # Verify model exists and can generate embeddings
        try:
            # Create an event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Test embedding generation
            loop.run_until_complete(self._test_embedding())
            
            # Initialize PCA model if needed
            if self.use_pca:
                self._initialize_pca()
                
            log.info(
                f"Successfully connected to Ollama embedding API with model {self.model_name} "
                f"(dimensions: {self._embedding_dim}, PCA: {self.use_pca})"
            )
        except Exception as e:
            log.error(f"Failed to generate embeddings with model {self.model_name}: {e}")
            raise RuntimeError(f"Model {self.model_name} is not available or cannot generate embeddings")
    
    async def _test_embedding(self) -> None:
        """Test embedding generation to verify model capability."""
        embedding = await self.client.embed("test", model=self.model_name)
        self._embedding_dim = len(embedding)
        log.debug(f"Embedding dimension: {self._embedding_dim}")
        
    def _initialize_pca(self) -> None:
        """Initialize PCA model for dimensionality reduction."""
        try:
            from sklearn.decomposition import PCA
            # Note: We'll fit the PCA model on actual data later
            self.pca_model = PCA(n_components=min(self.pca_dimensions, self._embedding_dim))
            log.info(f"Initialized PCA model to reduce dimensions from {self._embedding_dim} to {self.pca_dimensions}")
        except ImportError:
            log.warning("sklearn not available, disabling PCA reduction")
            self.use_pca = False
            
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        if not self.normalization_enabled:
            return vector
            
        # Convert to numpy array for operations
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        
        # Avoid division by zero
        if norm > 1e-10:
            v = v / norm
        
        self._performance_metrics["normalizations"] += 1
        return v.tolist()
    
    def _normalize_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Normalize multiple vectors to unit length.
        
        Args:
            vectors: List of input vectors
            
        Returns:
            List of normalized vectors
        """
        if not self.normalization_enabled or not vectors:
            return vectors
            
        # Convert to numpy array for vectorized operations
        v = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        
        # Avoid division by zero
        mask = norms > 1e-10
        v[mask] = v[mask] / norms[mask]
        
        self._performance_metrics["normalizations"] += len(vectors)
        return v.tolist()
    
    def _quantize_vector(self, vector: List[float]) -> List[float]:
        """
        Quantize a vector to reduce memory usage.
        
        Args:
            vector: Input vector
            
        Returns:
            Quantized vector
        """
        if not self.quantization_enabled:
            return vector
            
        # Convert to numpy array, quantize to float16, then back to float32
        v = np.array(vector, dtype=np.float32)
        v = v.astype(np.float16).astype(np.float32)
        
        self._performance_metrics["quantizations"] += 1
        return v.tolist()
    
    def _quantize_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Quantize multiple vectors to reduce memory usage.
        
        Args:
            vectors: List of input vectors
            
        Returns:
            List of quantized vectors
        """
        if not self.quantization_enabled or not vectors:
            return vectors
            
        # Convert to numpy array for vectorized operations
        v = np.array(vectors, dtype=np.float32)
        v = v.astype(np.float16).astype(np.float32)
        
        self._performance_metrics["quantizations"] += len(vectors)
        return v.tolist()
    
    def _reduce_dimensions(self, vector: List[float]) -> List[float]:
        """
        Reduce dimensions of a vector using PCA.
        
        Args:
            vector: Input vector
            
        Returns:
            Reduced vector
        """
        if not self.use_pca or not self.pca_model:
            return vector
            
        # Convert to numpy array for PCA
        v = np.array([vector], dtype=np.float32)
        
        # Check if PCA model is fitted
        if not hasattr(self.pca_model, "components_"):
            log.warning("PCA model not yet fitted, skipping reduction")
            return vector
            
        # Apply PCA transformation
        reduced = self.pca_model.transform(v)
        self._performance_metrics["pca_reductions"] += 1
        
        return reduced[0].tolist()
    
    def _reduce_dimensions_batch(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Reduce dimensions of multiple vectors using PCA.
        
        Args:
            vectors: List of input vectors
            
        Returns:
            List of reduced vectors
        """
        if not self.use_pca or not self.pca_model or not vectors:
            return vectors
            
        # Convert to numpy array for PCA
        v = np.array(vectors, dtype=np.float32)
        
        # Fit PCA model if not already fitted
        if not hasattr(self.pca_model, "components_"):
            log.info(f"Fitting PCA model on {len(vectors)} vectors")
            self.pca_model.fit(v)
            
        # Apply PCA transformation
        reduced = self.pca_model.transform(v)
        self._performance_metrics["pca_reductions"] += len(vectors)
        
        return reduced.tolist()
    
    def _process_vector(self, vector: List[float], apply_pca: bool = True) -> Tuple[List[float], Optional[List[float]]]:
        """
        Apply all configured vector operations.
        
        Args:
            vector: Input vector
            apply_pca: Whether to apply PCA reduction
            
        Returns:
            Tuple of (processed_vector, reduced_vector)
        """
        normalized = self._normalize_vector(vector)
        quantized = self._quantize_vector(normalized)
        
        if apply_pca and self.use_pca:
            reduced = self._reduce_dimensions(quantized)
            return quantized, reduced
        else:
            return quantized, None
    
    def _process_vectors(self, vectors: List[List[float]], apply_pca: bool = True) -> Tuple[List[List[float]], Optional[List[List[float]]]]:
        """
        Apply all configured vector operations to multiple vectors.
        
        Args:
            vectors: List of input vectors
            apply_pca: Whether to apply PCA reduction
            
        Returns:
            Tuple of (processed_vectors, reduced_vectors)
        """
        normalized = self._normalize_vectors(vectors)
        quantized = self._quantize_vectors(normalized)
        
        if apply_pca and self.use_pca:
            reduced = self._reduce_dimensions_batch(quantized)
            return quantized, reduced
        else:
            return quantized, None
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), retry=retry_if_exception_type(Exception))
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a piece of text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not text:
            return []
        
        start_time = time.time()
        
        # Check cache first if enabled
        if self.cache_embeddings:
            cached_embedding = self.embedding_cache.get_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding if not cached
        try:
            # Create an event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Generate embedding
            embedding = loop.run_until_complete(
                self.client.embed(text, model=self.model_name)
            )
            
            # Apply vector operations
            processed_vec, reduced_vec = self._process_vector(embedding)
            
            # Update performance metrics
            duration = time.time() - start_time
            self._performance_metrics["total_embeddings"] += 1
            self._performance_metrics["total_time"] += duration
            self._performance_metrics["avg_time_per_embedding"] = (
                self._performance_metrics["total_time"] / 
                self._performance_metrics["total_embeddings"]
            )
            
            # Cache the processed embedding if enabled
            if self.cache_embeddings:
                self.embedding_cache.add_embedding(text, processed_vec, reduced_vec)
            
            return processed_vec
            
        except Exception as e:
            log.error(f"Error generating embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), retry=retry_if_exception_type(Exception))
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return []
        
        start_time = time.time()
        result = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first if enabled
        if self.cache_embeddings:
            for i, text in enumerate(texts):
                # Try to get from cache
                cached_embedding = self.embedding_cache.get_embedding(text)
                if cached_embedding is not None:
                    result.append(cached_embedding)
                else:
                    result.append([])  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                # Process in batches to prevent memory issues
                all_embeddings = []
                
                for i in range(0, len(uncached_texts), self.batch_size):
                    batch = uncached_texts[i:i + self.batch_size]
                    
                    # Create an event loop if one doesn't exist
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                    # Generate embeddings for batch
                    batch_embeddings = loop.run_until_complete(
                        self.client.embed_batch(batch, model=self.model_name)
                    )
                    
                    all_embeddings.extend(batch_embeddings)
                
                # Apply vector operations
                processed_embeddings, reduced_embeddings = self._process_vectors(all_embeddings)
                
                # Update results and cache
                for i, idx in enumerate(uncached_indices):
                    embedding = processed_embeddings[i] if i < len(processed_embeddings) else []
                    result[idx] = embedding
                    
                    # Add to cache if enabled
                    if self.cache_embeddings and i < len(uncached_texts):
                        text = uncached_texts[i]
                        reduced_embedding = reduced_embeddings[i] if reduced_embeddings and i < len(reduced_embeddings) else None
                        self.embedding_cache.add_embedding(text, embedding, reduced_embedding)
                            
            except Exception as e:
                log.error(f"Error generating embeddings with Ollama: {e}")
                raise RuntimeError(f"Failed to generate embeddings: {e}")
        
        # Update performance metrics
        duration = time.time() - start_time
        self._performance_metrics["total_embeddings"] += len(uncached_texts)
        self._performance_metrics["total_time"] += duration
        if self._performance_metrics["total_embeddings"] > 0:
            self._performance_metrics["avg_time_per_embedding"] = (
                self._performance_metrics["total_time"] / self._performance_metrics["total_embeddings"]
            )
        
        # Log performance stats
        log.debug(
            f"Generated {len(uncached_texts)} embeddings in {duration:.2f}s " +
            f"(cache: {len(texts) - len(uncached_texts)} hits, {len(uncached_texts)} misses)"
        )
        
        return result
    
    def get_reduced_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get dimensionally-reduced embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of reduced embedding vectors
            
        Raises:
            RuntimeError: If embedding generation or reduction fails
        """
        if not self.use_pca:
            log.warning("PCA reduction not enabled, returning full embeddings")
            return self.embed_texts(texts)
            
        # Check if reduced embeddings are cached
        result = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first if enabled
        if self.cache_embeddings:
            for i, text in enumerate(texts):
                # Try to get reduced embedding from cache
                reduced_embedding = self.embedding_cache.get_reduced_embedding(text)
                if reduced_embedding is not None:
                    result.append(reduced_embedding)
                else:
                    result.append([])  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            
        # Generate embeddings for uncached texts
        if uncached_texts:
            # Generate full embeddings first
            full_embeddings = self.embed_texts(uncached_texts)
            
            # Apply PCA reduction
            _, reduced_embeddings = self._process_vectors(full_embeddings, apply_pca=True)
            
            if reduced_embeddings:
                # Update results
                for i, idx in enumerate(uncached_indices):
                    reduced_embedding = reduced_embeddings[i] if i < len(reduced_embeddings) else []
                    result[idx] = reduced_embedding
                    
                    # Add to cache if enabled
                    if self.cache_embeddings and i < len(uncached_texts):
                        text = uncached_texts[i]
                        full_embedding = full_embeddings[i] if i < len(full_embeddings) else []
                        self.embedding_cache.add_embedding(text, full_embedding, reduced_embedding)
            else:
                # Fallback to full embeddings if reduction failed
                for i, idx in enumerate(uncached_indices):
                    result[idx] = full_embeddings[i] if i < len(full_embeddings) else []
                
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get embedding generator performance metrics.
        
        Returns:
            Dict with performance statistics
        """
        return {
            **self._performance_metrics,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "cache_size": len(self._embedding_cache),
            "reduced_cache_size": len(self._reduced_embedding_cache) if self.use_pca else 0,
            "embedding_dim": self._embedding_dim,
            "reduced_dim": self.pca_dimensions if self.use_pca else None,
            "quantization_enabled": self.quantization_enabled,
            "normalization_enabled": self.normalization_enabled
        }
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), retry=retry_if_exception_type(Exception))
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        # Check cache first if enabled
        if self.cache_embeddings:
            cache_key = text[:1000]
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[cache_key]
            self._cache_misses += 1
            
        try:
            # Create an event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Generate embedding
            embedding = loop.run_until_complete(
                self.client.embed(text, model=self.model_name)
            )
            
            # Add to cache if enabled
            if self.cache_embeddings:
                cache_key = text[:1000]
                self._embedding_cache[cache_key] = embedding
                
                # Prune cache if it exceeds max size
                if len(self._embedding_cache) > self.max_cache_size:
                    self._prune_cache()
                    
            return embedding
            
        except Exception as e:
            log.error(f"Error generating embedding with Ollama: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
        
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self._embedding_dim
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dict with cache statistics
        """
        if not self.cache_embeddings:
            return {"enabled": False}
            
        return {
            "enabled": True,
            "size": len(self._embedding_cache),
            "max_size": self.max_cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    def _prune_cache(self) -> None:
        """Prune the embedding cache to stay within size limits."""
        # Simple LRU-like pruning - just remove oldest entries
        overflow = len(self._embedding_cache) - self.max_cache_size
        if overflow > 0:
            keys_to_remove = list(self._embedding_cache.keys())[:overflow]
            for key in keys_to_remove:
                del self._embedding_cache[key]
            log.debug(f"Pruned {overflow} entries from embedding cache")
