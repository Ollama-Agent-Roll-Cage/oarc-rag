"""
Caching functionality for RAG operations.

This module provides a comprehensive caching system for all RAG components,
including query results, embeddings, templates, contexts, and responses.
Implements concepts from Specification.md and Big_Brain.md for multi-level caching.
"""
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Callable, Tuple
from collections import OrderedDict
import re
from datetime import datetime, timedelta
import threading
from pathlib import Path
import numpy as np

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton

# Type variable for generic cache implementations
T = TypeVar('T')


class EvictionPolicy:
    """Enumeration of cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"  # Random Eviction
    TTL = "ttl"  # Time To Live


class CacheStats:
    """Statistics tracker for cache performance."""
    
    def __init__(self):
        """Initialize cache statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0
        self.expirations = 0
        self.start_time = time.time()
    
    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1
    
    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1
    
    def record_eviction(self):
        """Record a cache eviction."""
        self.evictions += 1
    
    def record_insertion(self):
        """Record a cache insertion."""
        self.insertions += 1
        
    def record_expiration(self):
        """Record a cache expiration."""
        self.expirations += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache performance.
        
        Returns:
            Dict containing cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        uptime = time.time() - self.start_time
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "insertions": self.insertions,
            "expirations": self.expirations,
            "operations_per_second": total / uptime if uptime > 0 else 0,
            "uptime_seconds": uptime
        }
    
    def __str__(self) -> str:
        """String representation of cache statistics."""
        stats = self.get_stats()
        return (f"CacheStats(hits={stats['hits']}, misses={stats['misses']}, "
                f"hit_rate={stats['hit_rate']:.2%}, evictions={stats['evictions']})")


class Cache(Generic[T]):
    """
    Base cache implementation with configurable eviction policies.
    
    This class implements a generic cache with support for multiple eviction
    policies and statistics tracking.
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        ttl: int = 3600,
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds for cached entries
            eviction_policy: Cache eviction policy ("lru", "lfu", "fifo", "random")
        """
        self.max_size = max(max_size, 10)  # Minimum size of 10
        self.ttl = ttl
        self.eviction_policy = eviction_policy
        
        # Storage for different policies
        if eviction_policy == EvictionPolicy.LRU:
            self.cache = OrderedDict()  # OrderedDict for LRU
        elif eviction_policy == EvictionPolicy.LFU:
            self.cache = {}  # Dict for items
            self.access_count = {}  # Dict for access counts
        else:
            self.cache = OrderedDict()  # Default to OrderedDict
        
        self.timestamps = {}  # Insertion timestamps for TTL
        self.stats = CacheStats()
        self.lock = threading.RLock()  # For thread safety
    
    def _make_key(self, key: Any) -> str:
        """
        Convert any key type to a string key.
        
        Args:
            key: Any hashable key
            
        Returns:
            String representation of the key
        """
        if isinstance(key, str):
            return key
        
        try:
            # Try direct string conversion
            return str(key)
        except:
            # Fall back to JSON for complex objects
            try:
                if isinstance(key, dict):
                    # Sort dictionary keys for consistency
                    return json.dumps(key, sort_keys=True)
                else:
                    return json.dumps(key)
            except:
                # Last resort: use hash of string representation
                return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[T]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached item or None if not found/expired
        """
        string_key = self._make_key(key)
        
        with self.lock:
            # Check if entry exists
            if string_key not in self.cache:
                self.stats.record_miss()
                return None
            
            # Check if entry expired
            if self.ttl > 0:
                timestamp = self.timestamps.get(string_key, 0)
                if time.time() - timestamp > self.ttl:
                    self._remove(string_key)
                    self.stats.record_expiration()
                    self.stats.record_miss()
                    return None
            
            # Update based on policy
            if self.eviction_policy == EvictionPolicy.LRU:
                value = self.cache.pop(string_key)
                self.cache[string_key] = value
            elif self.eviction_policy == EvictionPolicy.LFU:
                self.access_count[string_key] += 1
                value = self.cache[string_key]
            else:
                value = self.cache[string_key]
                
            self.stats.record_hit()
            return value
    
    def set(self, key: Any, value: T) -> None:
        """
        Add an item to the cache.
        
        Args:
            key: Cache key
            value: Item to cache
        """
        string_key = self._make_key(key)
        
        with self.lock:
            current_time = time.time()
            
            # Update cache based on policy
            if self.eviction_policy == EvictionPolicy.LRU:
                if string_key in self.cache:
                    self.cache.pop(string_key)
                elif len(self.cache) >= self.max_size:
                    self._evict_item()
                self.cache[string_key] = value
                
            elif self.eviction_policy == EvictionPolicy.LFU:
                if string_key not in self.cache and len(self.cache) >= self.max_size:
                    self._evict_item()
                self.cache[string_key] = value
                self.access_count[string_key] = 1
                
            elif self.eviction_policy == EvictionPolicy.FIFO:
                if string_key not in self.cache and len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    self._remove(oldest_key)
                self.cache[string_key] = value
                
            elif self.eviction_policy == EvictionPolicy.RANDOM:
                if string_key not in self.cache and len(self.cache) >= self.max_size:
                    import random
                    keys = list(self.cache.keys())
                    random_key = random.choice(keys)
                    self._remove(random_key)
                self.cache[string_key] = value
                
            else:  # Default to LRU
                if string_key in self.cache:
                    self.cache.pop(string_key)
                elif len(self.cache) >= self.max_size:
                    self._evict_item()
                self.cache[string_key] = value
            
            # Update timestamp
            self.timestamps[string_key] = current_time
            self.stats.record_insertion()
    
    def _evict_item(self) -> None:
        """Evict an item based on the current eviction policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove first item (oldest used)
            oldest_key, _ = self.cache.popitem(last=False)
            if oldest_key in self.timestamps:
                del self.timestamps[oldest_key]
                
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Find key with lowest access count
            if self.access_count:
                min_key = min(self.access_count, key=self.access_count.get)
                self._remove(min_key)
                
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove first inserted item
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
            
        elif self.eviction_policy == EvictionPolicy.RANDOM:
            # Remove random item
            import random
            keys = list(self.cache.keys())
            random_key = random.choice(keys)
            self._remove(random_key)
            
        else:  # Default to LRU
            # Remove first item (oldest used)
            oldest_key, _ = self.cache.popitem(last=False)
            if oldest_key in self.timestamps:
                del self.timestamps[oldest_key]
        
        self.stats.record_eviction()
    
    def _remove(self, key: str) -> None:
        """Remove an item from all tracking dictionaries."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if hasattr(self, 'access_count') and key in self.access_count:
            del self.access_count[key]
    
    def contains(self, key: Any) -> bool:
        """
        Check if key exists in cache and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists and is not expired
        """
        string_key = self._make_key(key)
        
        with self.lock:
            if string_key not in self.cache:
                return False
            
            # Check TTL if enabled
            if self.ttl > 0:
                timestamp = self.timestamps.get(string_key, 0)
                if time.time() - timestamp > self.ttl:
                    self._remove(string_key)
                    self.stats.record_expiration()
                    return False
            
            return True
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache and is not expired."""
        return self.contains(key)
    
    def remove(self, key: Any) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key to remove
        """
        string_key = self._make_key(key)
        
        with self.lock:
            self._remove(string_key)
            
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            if hasattr(self, 'access_count'):
                self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            stats = self.stats.get_stats()
            stats.update({
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "policy": self.eviction_policy
            })
            return stats
    
    def cleanup_expired(self) -> int:
        """
        Remove expired items from cache.
        
        Returns:
            Number of items removed
        """
        if self.ttl <= 0:
            return 0
        
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                self._remove(key)
                self.stats.record_expiration()
            
            return len(expired_keys)


class QueryCache(Cache[List[Dict[str, Any]]]):
    """
    Cache for query results with query normalization.
    
    This cache is specialized for storing search query results
    with normalization to improve hit rates.
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        ttl: int = 3600,
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the query cache.
        
        Args:
            max_size: Maximum number of queries to cache
            ttl: Time-to-live in seconds for cached entries
            eviction_policy: Cache eviction policy
        """
        super().__init__(max_size, ttl, eviction_policy)
        self.query_formats = {}  # Tracks original query format
    
    def add(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Add query results to the cache.
        
        Args:
            query: The query string
            results: List of result items
        """
        key = self._normalize_query(query)
        self.query_formats[key] = query  # Store original query format
        self.set(key, results)
    
    def get_results(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get results for a query from the cache.
        
        Args:
            query: The query string
            
        Returns:
            List of result items or None if not found or expired
        """
        key = self._normalize_query(query)
        return self.get(key)
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query string for consistent cache keys.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        # Remove extra whitespace, convert to lowercase, remove punctuation
        query = re.sub(r'\s+', ' ', query.strip().lower())
        query = re.sub(r'[^\w\s]', '', query)
        return query
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            super().clear()
            self.query_formats.clear()


class EmbeddingCache(Cache[List[float]]):
    """
    Cache for vector embeddings with support for dimension reduction.
    
    This cache stores both full and dimensionally-reduced embeddings
    for efficient retrieval and searching.
    """
    
    def __init__(
        self, 
        max_size: int = 5000,
        ttl: int = 86400,  # Default 24 hour TTL
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
            ttl: Time-to-live in seconds for cached entries
            eviction_policy: Cache eviction policy
        """
        super().__init__(max_size, ttl, eviction_policy)
        self.reduced_embeddings = {}  # Storage for reduced dimension embeddings
    
    def add_embedding(self, text: str, embedding: List[float], reduced_embedding: Optional[List[float]] = None) -> None:
        """
        Add an embedding to the cache.
        
        Args:
            text: Text that was embedded
            embedding: Full embedding vector
            reduced_embedding: Optional reduced dimension embedding
        """
        # Use only the first part of long texts as keys to save memory
        key = self._make_text_key(text)
        self.set(key, embedding)
        
        # Store reduced embedding if provided
        if reduced_embedding is not None:
            with self.lock:
                self.reduced_embeddings[key] = reduced_embedding
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get an embedding from the cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector or None if not found
        """
        key = self._make_text_key(text)
        return self.get(key)
    
    def get_reduced_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get reduced dimension embedding from the cache.
        
        Args:
            text: Text to get reduced embedding for
            
        Returns:
            Reduced embedding vector or None if not found
        """
        key = self._make_text_key(text)
        
        with self.lock:
            if key not in self.reduced_embeddings:
                return None
            
            # Check if expired
            if self.ttl > 0:
                timestamp = self.timestamps.get(key, 0)
                if time.time() - timestamp > self.ttl:
                    self._remove_with_reduced(key)
                    return None
            
            return self.reduced_embeddings[key]
    
    def _make_text_key(self, text: str) -> str:
        """Create a cache key from text, truncating long texts."""
        # Use at most 1000 chars of text as key to prevent huge keys
        truncated = text[:1000]
        return truncated
    
    def _remove_with_reduced(self, key: str) -> None:
        """Remove an item from cache including reduced embeddings."""
        self._remove(key)
        
        with self.lock:
            if key in self.reduced_embeddings:
                del self.reduced_embeddings[key]
    
    def remove(self, text: str) -> None:
        """Remove an embedding from the cache."""
        key = self._make_text_key(text)
        self._remove_with_reduced(key)
    
    def clear(self) -> None:
        """Clear all embeddings from the cache."""
        with self.lock:
            super().clear()
            self.reduced_embeddings.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = super().get_stats()
        
        with self.lock:
            stats.update({
                "reduced_embeddings_count": len(self.reduced_embeddings)
            })
            
        return stats


class ContextCache(Cache[str]):
    """
    Cache for assembled context strings used in prompts.
    
    This cache stores assembled context strings to avoid
    redundant context retrieval and assembly operations.
    """
    
    def __init__(
        self, 
        max_size: int = 200,
        ttl: int = 3600,  # 1 hour default
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the context cache.
        
        Args:
            max_size: Maximum number of contexts to cache
            ttl: Time-to-live in seconds for cached entries
            eviction_policy: Cache eviction policy
        """
        super().__init__(max_size, ttl, eviction_policy)
        
    def add_context(
        self, 
        query: str, 
        topic: str, 
        query_type: str,
        additional_context: Optional[Dict[str, Any]],
        context: str
    ) -> None:
        """
        Add an assembled context to the cache.
        
        Args:
            query: Query string
            topic: Topic string
            query_type: Type of query
            additional_context: Additional context dictionary
            context: Assembled context string
        """
        # Create composite key from parameters
        key = self._make_context_key(query, topic, query_type, additional_context)
        self.set(key, context)
    
    def get_context(
        self, 
        query: str, 
        topic: str, 
        query_type: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Get an assembled context from the cache.
        
        Args:
            query: Query string
            topic: Topic string
            query_type: Type of query
            additional_context: Additional context dictionary
            
        Returns:
            Assembled context or None if not found
        """
        key = self._make_context_key(query, topic, query_type, additional_context)
        return self.get(key)
    
    def _make_context_key(
        self, 
        query: str, 
        topic: str, 
        query_type: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """Create a composite key from context parameters."""
        # Remove whitespace and convert to lowercase for better matches
        query = re.sub(r'\s+', ' ', query.strip().lower()) if query else ""
        topic = re.sub(r'\s+', ' ', topic.strip().lower()) if topic else ""
        
        # Convert additional context to a sorted, stable string representation
        context_str = ""
        if additional_context:
            try:
                # Sort keys for consistent hashing
                context_str = json.dumps(additional_context, sort_keys=True)
            except:
                context_str = str(additional_context)
        
        return f"{query}||{topic}||{query_type}||{context_str}"


class TemplateCache(Cache[Dict[str, Any]]):
    """
    Cache for compiled templates.
    
    This cache stores compiled template objects to avoid
    repeated parsing of the same templates.
    """
    
    def __init__(
        self, 
        max_size: int = 100,  # Templates are typically few but reused often
        ttl: int = 0,  # No expiration by default for templates
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the template cache.
        
        Args:
            max_size: Maximum number of templates to cache
            ttl: Time-to-live in seconds (0 for no expiration)
            eviction_policy: Cache eviction policy
        """
        super().__init__(max_size, ttl, eviction_policy)
        self.template_versions = {}  # Track template versions
    
    def add_template(self, name: str, template_object: Any, template_text: str, version: str = "1.0") -> None:
        """
        Add a compiled template to the cache.
        
        Args:
            name: Template name
            template_object: Compiled template object
            template_text: Raw template text
            version: Template version string
        """
        # Extract variables from template text
        variables = self._extract_variables(template_text)
        
        # Create template info dictionary
        template_info = {
            'template': template_object,
            'text': template_text,
            'version': version,
            'variables': variables,
            'usage_count': 0,
            'successful_uses': 0
        }
        
        self.set(name, template_info)
        self.template_versions[name] = version
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a template from the cache.
        
        Args:
            name: Template name
            
        Returns:
            Template info dictionary or None if not found
        """
        template_info = self.get(name)
        
        if template_info:
            # Update usage count
            template_info['usage_count'] += 1
        
        return template_info
    
    def record_successful_use(self, name: str) -> None:
        """
        Record successful template use.
        
        Args:
            name: Template name
        """
        template_info = self.get(name)
        if template_info:
            template_info['successful_uses'] += 1
    
    def _extract_variables(self, template_text: str) -> set:
        """
        Extract variables from a template string.
        
        Args:
            template_text: Template text to analyze
            
        Returns:
            Set of variable names
        """
        if not template_text:
            return set()
            
        # Simple regex to find variables in different template formats
        # This is a basic implementation - more complex templates may need custom parsers
        variables = set()
        
        # Find Jinja2 style {{ variable }}
        jinja_vars = re.findall(r'\{\{\s*(\w+)(?:\s*\|\s*\w+(?:\(.*?\))?)?\s*\}\}', template_text)
        variables.update(jinja_vars)
        
        # Find Jinja2 if statements {% if variable %}
        if_vars = re.findall(r'\{%\s*if\s+(\w+)', template_text)
        variables.update(if_vars)
        
        # Find Jinja2 for loops {% for item in collection %}
        for_vars = re.findall(r'\{%\s*for\s+\w+\s+in\s+(\w+)', template_text)
        variables.update(for_vars)
        
        return variables


class ResponseCache(Cache[Dict[str, Any]]):
    """
    Cache for API responses.
    
    This cache stores API responses to avoid redundant API calls
    for identical requests.
    """
    
    def __init__(
        self, 
        max_size: int = 500,
        ttl: int = 3600,  # 1 hour default
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of responses to cache
            ttl: Time-to-live in seconds for cached entries
            eviction_policy: Cache eviction policy
        """
        super().__init__(max_size, ttl, eviction_policy)
    
    def add_response(self, request_params: Dict[str, Any], response: Any) -> None:
        """
        Add an API response to the cache.
        
        Args:
            request_params: API request parameters
            response: API response
        """
        # Remove non-cacheable parameters
        params_copy = request_params.copy()
        for param in ['stream', 'callback']:
            if param in params_copy:
                del params_copy[param]
        
        # Cache the response
        self.set(params_copy, {"content": response, "timestamp": time.time()})
    
    def get_response(self, request_params: Dict[str, Any]) -> Optional[Any]:
        """
        Get a cached API response.
        
        Args:
            request_params: API request parameters
            
        Returns:
            Cached response or None if not found/expired
        """
        # Remove non-cacheable parameters
        params_copy = request_params.copy()
        for param in ['stream', 'callback']:
            if param in params_copy:
                del params_copy[param]
        
        # Get from cache
        cached = self.get(params_copy)
        
        if cached:
            return cached["content"]
        return None


class DocumentCache(Cache[Dict[str, Any]]):
    """
    Cache for document chunks and embeddings.
    
    This cache stores processed document chunks and their embeddings
    to avoid redundant processing of the same documents.
    """
    
    def __init__(
        self, 
        max_size: int = 100,  # Documents can be large, so smaller default
        ttl: int = 86400,  # 24 hour default
        eviction_policy: str = EvictionPolicy.LRU
    ):
        """
        Initialize the document cache.
        
        Args:
            max_size: Maximum number of documents to cache
            ttl: Time-to-live in seconds for cached entries
            eviction_policy: Cache eviction policy
        """
        super().__init__(max_size, ttl, eviction_policy)
        
    def add_document(
        self, 
        source: str, 
        chunks: List[str], 
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add document chunks and embeddings to the cache.
        
        Args:
            source: Document source/identifier
            chunks: List of text chunks
            embeddings: List of embeddings corresponding to chunks
            metadata: Optional document metadata
        """
        doc_data = {
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": metadata or {},
            "chunk_count": len(chunks)
        }
        self.set(source, doc_data)
    
    def get_document(self, source: str) -> Optional[Dict[str, Any]]:
        """
        Get cached document data.
        
        Args:
            source: Document source/identifier
            
        Returns:
            Document data dictionary or None if not found
        """
        return self.get(source)
    
    def get_chunks(self, source: str) -> Optional[List[str]]:
        """
        Get document chunks from the cache.
        
        Args:
            source: Document source/identifier
            
        Returns:
            List of chunks or None if not found
        """
        doc_data = self.get(source)
        return doc_data.get("chunks") if doc_data else None
    
    def get_embeddings(self, source: str) -> Optional[List[List[float]]]:
        """
        Get document embeddings from the cache.
        
        Args:
            source: Document source/identifier
            
        Returns:
            List of embeddings or None if not found
        """
        doc_data = self.get(source)
        return doc_data.get("embeddings") if doc_data else None
    
    def get_metadata(self, source: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata from the cache.
        
        Args:
            source: Document source/identifier
            
        Returns:
            Document metadata or None if not found
        """
        doc_data = self.get(source)
        return doc_data.get("metadata") if doc_data else None


@singleton
class CacheManager:
    """
    Centralized manager for all cache instances.
    
    This singleton class provides a unified interface for accessing
    and managing different types of caches throughout the application.
    """
    
    def __init__(self):
        """Initialize the cache manager."""
        # Only initialize once (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Default cache TTL values (from Specification.md)
        self.default_ttls = {
            "query": 3600,       # 1 hour
            "embedding": 86400,  # 24 hours
            "document": 86400,   # 24 hours  
            "response": 3600,    # 1 hour
            "template": 0,       # No expiration
            "context": 3600      # 1 hour
        }
        
        # Default cache size limits
        self.default_sizes = {
            "query": 1000,
            "embedding": 5000,
            "document": 100,
            "response": 500,
            "template": 100,
            "context": 200
        }
        
        # Create cache registry
        self.caches = {}
        
        # Create default caches
        self.query_cache = self._create_cache('query', QueryCache)
        self.embedding_cache = self._create_cache('embedding', EmbeddingCache)
        self.document_cache = self._create_cache('document', DocumentCache)
        self.response_cache = self._create_cache('response', ResponseCache)
        self.template_cache = self._create_cache('template', TemplateCache)
        self.context_cache = self._create_cache('context', ContextCache)
        
        # Track total memory usage (aproximately)
        self._approx_memory_usage = 0
        
        # Set up periodic cleanup
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        self._initialized = True
        log.info("Cache manager initialized with default caches")
    
    def _create_cache(self, name: str, cache_class, **kwargs):
        """Create a new cache instance and register it."""
        # Use default values for this cache type if not provided
        ttl = kwargs.pop('ttl', self.default_ttls.get(name, 3600))
        max_size = kwargs.pop('max_size', self.default_sizes.get(name, 1000))
        eviction_policy = kwargs.pop('eviction_policy', EvictionPolicy.LRU)
        
        # Create cache instance
        cache = cache_class(
            max_size=max_size,
            ttl=ttl,
            eviction_policy=eviction_policy,
            **kwargs
        )
        
        # Register in cache dictionary
        self.caches[name] = cache
        
        return cache
    
    def get_cache(self, name: str) -> Optional[Cache]:
        """
        Get a cache by name.
        
        Args:
            name: Cache name
            
        Returns:
            Cache instance or None if not found
        """
        return self.caches.get(name)
    
    def register_cache(self, name: str, cache: Cache) -> None:
        """
        Register a custom cache instance.
        
        Args:
            name: Cache name
            cache: Cache instance
        """
        self.caches[name] = cache
        log.debug(f"Registered custom cache: {name}")
    
    def create_custom_cache(
        self, 
        name: str, 
        cache_type: str = "generic",
        max_size: Optional[int] = None,
        ttl: Optional[int] = None,
        eviction_policy: str = EvictionPolicy.LRU
    ) -> Cache:
        """
        Create a custom cache with specific parameters.
        
        Args:
            name: Cache name
            cache_type: Type of cache ('generic', 'query', 'embedding', etc.)
            max_size: Maximum cache size
            ttl: Cache TTL in seconds
            eviction_policy: Cache eviction policy
            
        Returns:
            Created cache instance
        """
        # Map cache types to classes
        cache_classes = {
            "generic": Cache,
            "query": QueryCache,
            "embedding": EmbeddingCache,
            "document": DocumentCache,
            "response": ResponseCache,
            "template": TemplateCache,
            "context": ContextCache
        }
        
        # Get cache class
        cache_class = cache_classes.get(cache_type, Cache)
        
        # Use default values if not provided
        if max_size is None:
            max_size = self.default_sizes.get(cache_type, 1000)
        
        if ttl is None:
            ttl = self.default_ttls.get(cache_type, 3600)
        
        # Create and register cache
        cache = self._create_cache(
            name, 
            cache_class, 
            max_size=max_size,
            ttl=ttl,
            eviction_policy=eviction_policy
        )
        
        return cache
    
    def periodic_cleanup(self, force: bool = False) -> None:
        """
        Perform periodic cleanup of all caches.
        
        Args:
            force: Force cleanup regardless of time elapsed
        """
        current_time = time.time()
        
        if force or (current_time - self._last_cleanup) > self._cleanup_interval:
            log.debug("Performing periodic cache cleanup")
            
            total_expired = 0
            for name, cache in self.caches.items():
                expired = cache.cleanup_expired()
                total_expired += expired
                
            self._last_cleanup = current_time
            
            if total_expired > 0:
                log.info(f"Cache cleanup removed {total_expired} expired items")
    
    def clear_all_caches(self) -> None:
        """Clear all managed caches."""
        for name, cache in self.caches.items():
            cache.clear()
            
        log.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all managed caches.
        
        Returns:
            Dictionary of cache statistics by cache name
        """
        # Perform cleanup first to get accurate stats
        self.periodic_cleanup()
        
        stats = {
            name: cache.get_stats() 
            for name, cache in self.caches.items()
        }
        
        # Add summary statistics
        total_size = sum(data.get('size', 0) for data in stats.values())
        total_max_size = sum(data.get('max_size', 0) for data in stats.values())
        total_hits = sum(data.get('hits', 0) for data in stats.values())
        total_misses = sum(data.get('misses', 0) for data in stats.values())
        
        stats['summary'] = {
            'total_size': total_size,
            'total_max_size': total_max_size,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
            'last_cleanup': self._last_cleanup,
        }
        
        return stats


# Legacy name for backward compatibility 
ResultCache = QueryCache


# Initialize cache manager singleton for global access
cache_manager = CacheManager()
