"""
AI client for interfacing with Ollama models using the official Ollama package.
"""
import json
import time
import os
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
import asyncio
from datetime import datetime, timedelta

from ollama import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.config import AI_CONFIG
from oarc_rag.core.cache import cache_manager, ResponseCache

# Type variable for generic return type annotations
T = TypeVar('T')

@singleton
class OllamaClient:
    """
    Async client for interacting with Ollama API.
    
    This class provides methods for generating text completions,
    chat responses, and embeddings using Ollama's local API.
    """
    
    @classmethod
    async def validate_ollama(cls) -> None:
        """Validate Ollama availability asynchronously."""
        from oarc_rag.utils.utils import Utils
        Utils.check_for_ollama(raise_error=True)
        
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        output_dir: Optional[str] = None,
        cache_responses: bool = True,
        cache_ttl: int = 3600  # 1 hour cache TTL
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            default_model: Default model to use for requests
            default_temperature: Default temperature for generation
            default_max_tokens: Default maximum tokens for generation
            output_dir: Custom output directory path
            cache_responses: Whether to cache responses
            cache_ttl: Time-to-live for cached responses in seconds
            
        Raises:
            RuntimeError: If Ollama server is not available
        """
        # Use configuration values or defaults
        self.base_url = base_url or AI_CONFIG.get('ollama_api_url', 'http://localhost:11434')
        self.api_url = f"{self.base_url}/api"  # Add this line to create api_url attribute
        self.default_model = default_model or AI_CONFIG.get('default_model', 'llama3.1:latest')
        self.default_temperature = default_temperature if default_temperature is not None else AI_CONFIG.get('temperature', 0.7)
        self.default_max_tokens = 4000  # Always use 4000 as default max_tokens
        
        # Create run directory structure
        self.run_id = str(int(time.time()))
        base_output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.output_dir = os.path.join(base_output_dir, self.run_id)
        
        # Create subdirectories for different data types
        self.collected_dir = os.path.join(self.output_dir, "collected")
        self.extracted_dir = os.path.join(self.output_dir, "extracted")
        
        # Create the directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.collected_dir, exist_ok=True)
        os.makedirs(self.extracted_dir, exist_ok=True)
        
        log.info(f"Created output directory structure in: {self.output_dir}")
        
        # Get response cache from cache manager
        self.cache_responses = cache_responses
        self.response_cache = cache_manager.response_cache
        
        # Initialize Ollama client
        self.client = AsyncClient(host=self.base_url)
        self.last_request = {}  # Add this to store last request data for debugging
        self.last_response = {}  # Add this to store last response data for debugging
        
        # Add API call monitoring
        self.api_calls = {
            "generate": 0,
            "chat": 0,
            "embed": 0,
            "embed_batch": 0,
            "list_models": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_tokens": 0
        }
        
        # Adaptive response handling
        self.operational_mode = "awake"  # Default to real-time mode
        self.response_timeouts = {
            "awake": 30,  # 30 seconds in awake mode
            "sleep": 180  # 3 minutes in sleep mode
        }
        
    def set_operational_mode(self, mode: str) -> None:
        """
        Set the client's operational mode to adjust response handling.
        
        Args:
            mode: "awake" or "sleep"
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ("awake", "sleep"):
            raise ValueError('Operational mode must be either "awake" or "sleep"')
        
        self.operational_mode = mode
        log.info(f"OllamaClient operational mode set to: {mode}")

    def _get_timeout(self) -> int:
        """Get appropriate timeout based on operational mode."""
        return self.response_timeouts.get(self.operational_mode, 30)
    
    def _cache_key(self, **kwargs) -> str:
        """
        Generate a cache key from request parameters.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Cache key string
        """
        # Remove parameters that shouldn't affect caching
        if "stream" in kwargs:
            del kwargs["stream"]
        if "callback" in kwargs:
            del kwargs["callback"]
            
        # Sort dictionary to ensure consistent ordering
        sorted_items = sorted(kwargs.items())
        
        # Convert to JSON string for hashing
        return json.dumps(sorted_items, sort_keys=True)
    
    def _clean_expired_cache(self) -> None:
        """Remove expired items from cache."""
        now = datetime.now()
        expired_keys = [k for k, v in self._cache_expiry.items() if now > v]
        
        for key in expired_keys:
            if key in self._response_cache:
                del self._response_cache[key]
            if key in self._cache_expiry:
                del self._cache_expiry[key]
                
        if expired_keys:
            log.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def invalidate_cache(self) -> None:
        """Clear all cached responses."""
        if hasattr(self, 'response_cache'):
            self.response_cache.clear()
            log.info(f"Invalidated response cache")
    
    async def initialize(self):
        """Initialize the client and validate server connection."""
        # Validate Ollama availability
        await OllamaClient.validate_ollama()
        
        # Test connection to server
        await self._validate_server()
        
    async def _validate_server(self) -> None:
        """
        Validate connection to Ollama server.
        
        Raises:
            RuntimeError: If server is not available
        """
        try:
            # Try to list models as a connection test
            await self.client.list()
            log.info(f"Successfully connected to Ollama server at {self.base_url}")
        except Exception as e:
            error_msg = f"Failed to connect to Ollama server at {self.base_url}: {e}"
            log.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def async_generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        stream: bool = False,  # Parameter kept for compatibility, but always False
        callback: Optional[Callable[[str], None]] = None  # Parameter kept for compatibility, but not used
    ) -> str:
        """
        Generate a completion from a prompt using chat API (async version).
        
        Args:
            prompt: The prompt to generate from
            model: Model to use (defaults to client's default_model)
            system: Optional system prompt for context
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stream: Not used - always set to False
            callback: Not used
            
        Returns:
            The generated text
            
        Raises:
            ValueError: If the prompt is empty
            RuntimeError: On API errors
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Prepare request params for cache lookup
        request_params = {
            "prompt": prompt,
            "model": model or self.default_model,
            "system": system,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        
        # Check cache if enabled
        if self.cache_responses:
            cached_response = self.response_cache.get_response(request_params)
            if cached_response:
                self.api_calls["cache_hits"] += 1
                return cached_response
            self.api_calls["cache_misses"] += 1
        
        # Convert parameters to chat format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Get model parameters
        model_name = model or self.default_model
        temp = temperature if temperature is not None else self.default_temperature
        num_predict = max_tokens or self.default_max_tokens
        
        # Track API call
        self.api_calls["generate"] += 1
        
        try:
            # Get appropriate timeout based on operational mode
            timeout = self._get_timeout()
            
            # Use the ollama library's chat method with timeout
            response = await asyncio.wait_for(
                self.client.chat(
                    model=model_name,
                    messages=messages,
                    options={
                        "temperature": temp,
                        "num_predict": num_predict
                    }
                ),
                timeout=timeout
            )
            
            # Extract content from chat response
            if response and "message" in response and "content" in response["message"]:
                content = response["message"]["content"].strip()
                
                # Track token usage if available
                if "eval_count" in response:
                    self.api_calls["total_tokens"] += response["eval_count"]
                
                # Cache response if enabled
                if self.cache_responses:
                    self.response_cache.add_response(request_params, content)
                
                return content
            return ""
                
        except asyncio.TimeoutError:
            self.api_calls["errors"] += 1
            log.error(f"Request timed out after {timeout} seconds")
            raise RuntimeError(f"Request timed out after {timeout} seconds")
        except Exception as e:
            self.api_calls["errors"] += 1
            log.error(f"Failed to generate text: {e}")
            return f"Error generating text: {e}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        stream: bool = False,  # Parameter kept for compatibility, but always False
        callback: Optional[Callable[[Dict[str, Any]], None]] = None  # Parameter kept for compatibility, but not used
    ) -> Dict[str, Any]:
        """
        Generate a chat completion from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to client's default_model)
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stream: Not used - always set to False
            callback: Not used
            
        Returns:
            Dict with response
            
        Raises:
            ValueError: If messages are empty or invalid
            RuntimeError: On API errors
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys")
        
        model_name = model or self.default_model
        temp = temperature if temperature is not None else self.default_temperature
        num_predict = 4000  # Always use 4000 tokens
        
        log.debug(f"Sending chat request to model {model_name}")
        
        try:
            # Use the ollama library's chat method
            response = await self.client.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": num_predict
                }
            )
            return response
                
        except Exception as e:
            log.error(f"Chat API request failed: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.
        
        Returns:
            List of dictionaries containing model information
            
        Raises:
            RuntimeError: On API errors
        """
        try:
            models = await self.client.list()
            return models["models"]
        except Exception as e:
            log.error(f"Failed to list models: {e}")
            raise RuntimeError(f"Failed to list models: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for text using Ollama's embedding API.
        
        Args:
            text: The text to embed
            model: Model to use for embedding (defaults to client's default_model)
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            ValueError: If the text is empty
            RuntimeError: On API errors
        """
        if not text:
            raise ValueError("Text cannot be empty")
            
        model_name = model or self.default_model
        
        log.debug(f"Sending embedding request using model {model_name}")
        
        try:
            # Use the ollama library's embed method
            response = await self.client.embeddings(model=model_name, prompt=text)
            
            # Extract the embedding from the response
            embedding = response.get("embedding", [])
            if not embedding:
                log.warning("Received empty embedding from Ollama API")
                
            return embedding
                
        except Exception as e:
            log.error(f"Embedding API request failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
            
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Ollama's embedding API.
        
        Args:
            texts: List of texts to embed
            model: Model to use for embedding (defaults to client's default_model)
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: On API errors
        """
        if not texts:
            return []
            
        model_name = model or self.default_model
        
        try:
            # Use the ollama library's batch embedding method
            response = await self.client.embeddings(model=model_name, prompt=texts)
            
            # Extract embeddings from response
            embeddings = response.get("embeddings", [])
            if not embeddings:
                log.warning("Received empty embeddings from Ollama API")
                return [[0.0] * 1536] * len(texts)  # Return empty embeddings
                
            return embeddings
            
        except Exception as e:
            log.error(f"Failed to process batch embeddings: {e}")
            raise RuntimeError(f"Failed to generate batch embeddings: {e}")
    
    def get_output_path(self, filename: str, subdir: Optional[str] = None) -> str:
        """
        Get a path in the run's output directory.
        
        Args:
            filename: Name of the file to create
            subdir: Optional subdirectory (collected, extracted, or None for base run dir)
            
        Returns:
            Full path to the file in the output directory
        """
        if subdir == "collected":
            return os.path.join(self.collected_dir, filename)
        elif subdir == "extracted":
            return os.path.join(self.extracted_dir, filename)
        else:
            return os.path.join(self.output_dir, filename)
    
    def save_to_output(self, filename: str, content: Union[str, Dict, List], subdir: Optional[str] = None) -> str:
        """
        Save content to a file in the output directory.
        
        Args:
            filename: Name of the file to save
            content: Content to save (string or JSON-serializable object)
            subdir: Optional subdirectory (collected, extracted, or None for base run dir)
            
        Returns:
            Path to the saved file
        """
        filepath = self.get_output_path(filename, subdir)
        
        # Create parent directories if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save content based on its type
        if isinstance(content, (dict, list)):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(content))
                
        log.debug(f"Saved content to {filepath}")
        return filepath

    async def pull_model(
        self,
        model: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bool:
        """
        Pull a model from Ollama registry with progress tracking.
        
        Args:
            model: Name of the model to pull
            progress_callback: Optional callback to report progress
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RuntimeError: On API errors
        """
        log.info(f"Pulling model {model} from Ollama registry")
        
        try:
            # Use the ollama library's pull method with progress callback
            # Note: model is passed as a positional argument, stream as a keyword argument
            async for progress in await self.client.pull(model, stream=True):
                if progress_callback:
                    progress_callback(progress)
                    
                # If progress has a 'completed' field, we can track it
                if "digest" in progress and "completed" in progress and "total" in progress:
                    digest = progress.get("digest", "")
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 0)
                    
                    # Log progress
                    if total > 0:
                        percent = (completed / total) * 100
                        log.debug(f"Pull progress: {percent:.1f}% ({completed}/{total})")
                
                # Log status updates
                if "status" in progress:
                    log.info(f"Pull status: {progress['status']}")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to pull model {model}: {e}")
            raise RuntimeError(f"Failed to pull model: {e}")

    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> str:
        """
        Generate text using the Ollama API (synchronous version).
        
        This is a synchronous wrapper around the async generate method.
        
        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate
            system: Optional system prompt
            
        Returns:
            Generated text
        """
        # Call the async_generate method
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop for this operation
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(
                        self.async_generate(prompt, model=model, system=system, 
                                   temperature=temperature, max_tokens=max_tokens)
                    )
                finally:
                    new_loop.close()
            else:
                # Use the existing loop
                return loop.run_until_complete(
                    self.async_generate(prompt, model=model, system=system, 
                               temperature=temperature, max_tokens=max_tokens)
                )
        except RuntimeError:
            # No event loop in thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.async_generate(prompt, model=model, system=system, 
                               temperature=temperature, max_tokens=max_tokens)
                )
            finally:
                loop.close()
        except Exception as e:
            log.error(f"Error generating text: {e}")
            return f"Error generating text: {e}"
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dict with usage statistics
        """
        # Calculate derived metrics
        total_calls = sum(self.api_calls[k] for k in ["generate", "chat", "embed", "embed_batch", "list_models"])
        cache_total = self.api_calls["cache_hits"] + self.api_calls["cache_misses"]
        cache_hit_rate = self.api_calls["cache_hits"] / max(1, cache_total)
        
        return {
            "calls": self.api_calls.copy(),
            "total_calls": total_calls,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": self.api_calls["errors"] / max(1, total_calls),
            "avg_tokens_per_call": self.api_calls["total_tokens"] / max(1, total_calls),
            "cache_entries": len(self._response_cache)
        }
