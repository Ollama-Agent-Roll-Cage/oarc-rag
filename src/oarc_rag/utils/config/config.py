"""
Configuration settings for the OARC-RAG system.

This module provides a comprehensive configuration system for all aspects
of the RAG framework, including vector operations, operational modes,
caching strategies, and agent behaviors in line with the self-improving
recursive RAG architecture described in the specification.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log
from oarc_rag.utils.const import (
    DEFAULT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_OPERATIONAL_MODE,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    SUPPORTED_FILE_EXTENSIONS
)


@singleton
class Config:
    """
    Configuration class for the OARC-RAG system.
    
    This class manages all configuration settings with a singleton pattern
    to ensure consistent access across the application. Settings are organized
    by category for clarity while maintaining a flat dictionary for performance.
    """
    
    def __init__(self):
        """Initialize configuration settings from environment variables and config files."""
        # Only initialize once (singleton pattern)
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        # Base paths
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        
        # Try to load from config file first
        self.config_file = os.getenv('OARC_RAG_CONFIG_FILE')
        self.settings = {}
        
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.settings = json.load(f)
                log.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                log.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Default settings organized by category
        default_settings = {
            # Base paths
            'base_dir': base_dir,
            'output_dir': Path(os.getenv('OARC_RAG_OUTPUT_DIR', base_dir / 'output')),
            'templates_dir': Path(os.getenv('OARC_RAG_TEMPLATES_DIR', base_dir / 'templates')),
            
            # AI settings
            'model': os.getenv('OARC_RAG_AI_MODEL', DEFAULT_MODEL),
            'temperature': float(os.getenv('OARC_RAG_AI_TEMP', str(DEFAULT_TEMPERATURE))),
            'max_tokens': int(os.getenv('OARC_RAG_AI_MAX_TOKENS', str(DEFAULT_MAX_TOKENS))),
            'ollama_api_url': os.getenv('OARC_RAG_OLLAMA_URL', DEFAULT_OLLAMA_URL),
            'default_model': os.getenv('OARC_RAG_DEFAULT_MODEL', DEFAULT_MODEL),
            'default_system_prompt': os.getenv('OARC_RAG_SYSTEM_PROMPT', 'rag_system'),
            'default_api_timeout': int(os.getenv('OARC_RAG_API_TIMEOUT', '60')),
            
            # RAG settings
            'embedding_model': os.getenv('OARC_RAG_EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL),
            'embedding_model_type': os.getenv('OARC_RAG_EMBEDDING_TYPE', 'ollama'),
            'embedding_dimensions': int(os.getenv('OARC_RAG_EMBEDDING_DIMENSIONS', '4096')),
            'chunk_size': int(os.getenv('OARC_RAG_CHUNK_SIZE', str(DEFAULT_CHUNK_SIZE))),
            'chunk_overlap': int(os.getenv('OARC_RAG_CHUNK_OVERLAP', str(DEFAULT_CHUNK_OVERLAP))),
            'retrieval_top_k': int(os.getenv('OARC_RAG_RETRIEVAL_TOP_K', '5')),
            'similarity_threshold': float(os.getenv('OARC_RAG_SIMILARITY_THRESHOLD', '0.7')),
            'semantic_reranking': os.getenv('OARC_RAG_SEMANTIC_RERANKING', 'false').lower() == 'true',
            
            # Vector operations (from Specification.md)
            'vector': {
                'use_pca': os.getenv('OARC_RAG_USE_PCA', 'true').lower() == 'true',
                'pca_dimensions': int(os.getenv('OARC_RAG_PCA_DIMENSIONS', '128')),
                'normalization_enabled': os.getenv('OARC_RAG_NORMALIZE_VECTORS', 'true').lower() == 'true',
                'distance_metric': os.getenv('OARC_RAG_DISTANCE_METRIC', 'cosine'),
                'batch_size': int(os.getenv('OARC_RAG_VECTOR_BATCH_SIZE', '32')),
                'enable_quantization': os.getenv('OARC_RAG_QUANTIZE_VECTORS', 'false').lower() == 'true',
                'target_precision': os.getenv('OARC_RAG_VECTOR_PRECISION', 'float32'),
            },
            
            # HNSW settings (from Specification.md)
            'hnsw': {
                'enabled': os.getenv('OARC_RAG_USE_HNSW', 'true').lower() == 'true',
                'ef_construction': int(os.getenv('OARC_RAG_HNSW_EF_CONSTRUCTION', str(DEFAULT_HNSW_EF_CONSTRUCTION))),
                'ef_search': int(os.getenv('OARC_RAG_HNSW_EF_SEARCH', str(DEFAULT_HNSW_EF_SEARCH))),
                'm': int(os.getenv('OARC_RAG_HNSW_M', str(DEFAULT_HNSW_M))),
                'index_path': os.getenv('OARC_RAG_HNSW_INDEX_PATH', str(base_dir / 'data' / 'hnsw_index')),
                'automatic_pruning': os.getenv('OARC_RAG_HNSW_AUTO_PRUNE', 'true').lower() == 'true'
            },
            
            # Vector DB settings
            'vector_db': {
                'type': os.getenv('OARC_RAG_VECTOR_DB_TYPE', 'faiss'),
                'dir': os.getenv('OARC_RAG_VECTOR_DB_DIR', str(base_dir / 'vector_db')),
                'collection_name': os.getenv('OARC_RAG_VECTOR_DB_COLLECTION', 'default_collection'),
                'persistence_format': os.getenv('OARC_RAG_DB_PERSISTENCE', 'parquet'),  # Options: parquet, arrow, both
                'in_memory_mode': os.getenv('OARC_RAG_IN_MEMORY_DB', 'true').lower() == 'true',
                'metadata_embedding': os.getenv('OARC_RAG_METADATA_EMBEDDING', 'false').lower() == 'true'
            },
            
            # GPU settings
            'hardware': {
                'use_gpu': os.getenv('OARC_RAG_USE_GPU', 'auto').lower() in ('true', 'yes', '1', 'auto'),
                'gpu_memory_threshold': float(os.getenv('OARC_RAG_GPU_MEMORY_THRESHOLD', '0.75')),
                'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES', None),
                'mixed_precision': os.getenv('OARC_RAG_MIXED_PRECISION', 'true').lower() == 'true',
                'offload_to_cpu': os.getenv('OARC_RAG_CPU_OFFLOAD', 'true').lower() == 'true'
            },
            
            # Operational modes (from Big_Brain.md)
            'operational_modes': {
                'default_mode': os.getenv('OARC_RAG_DEFAULT_MODE', DEFAULT_OPERATIONAL_MODE),
                'auto_switch': os.getenv('OARC_RAG_AUTO_MODE_SWITCH', 'true').lower() == 'true',
                'awake_timeout': int(os.getenv('OARC_RAG_AWAKE_TIMEOUT', '3600')),  # 1 hour default
                'sleep_cycle_duration': int(os.getenv('OARC_RAG_SLEEP_DURATION', '1800')),  # 30 mins default
                'knowledge_consolidation_interval': int(os.getenv('OARC_RAG_CONSOLIDATE_INTERVAL', '86400')),  # 24h
                'deep_learning_enabled': os.getenv('OARC_RAG_DEEP_LEARNING', 'true').lower() == 'true'
            },
            
            # Cache settings (from Specification.md)
            'caching': {
                'query_cache_enabled': os.getenv('OARC_RAG_QUERY_CACHE', 'true').lower() == 'true',
                'query_cache_size': int(os.getenv('OARC_RAG_QUERY_CACHE_SIZE', '1000')),
                'query_cache_ttl': int(os.getenv('OARC_RAG_QUERY_CACHE_TTL', '3600')),
                'embedding_cache_enabled': os.getenv('OARC_RAG_EMBEDDING_CACHE', 'true').lower() == 'true',
                'embedding_cache_size': int(os.getenv('OARC_RAG_EMBEDDING_CACHE_SIZE', '5000')),
                'embedding_cache_ttl': int(os.getenv('OARC_RAG_EMBEDDING_CACHE_TTL', '86400')),
                'document_cache_enabled': os.getenv('OARC_RAG_DOC_CACHE', 'true').lower() == 'true',
                'document_cache_size': int(os.getenv('OARC_RAG_DOC_CACHE_SIZE', '100')),
                'eviction_policy': os.getenv('OARC_RAG_CACHE_EVICTION', 'lru')  # Options: lru, lfu, random
            },
            
            # Agent settings (from Big_Brain.md and the agents directory)
            'agents': {
                'rag_agent': {
                    'enabled': True,
                    'context_strategies': ['prefix', 'suffix', 'combined', 'sandwich', 'framing', 'reference'],
                    'default_strategy': 'prefix',
                    'max_context_chunks': int(os.getenv('OARC_RAG_MAX_CONTEXT_CHUNKS', '10'))
                },
                'expansion_agent': {
                    'enabled': os.getenv('OARC_RAG_EXPANSION_AGENT', 'false').lower() == 'true',
                    'expansion_factor': float(os.getenv('OARC_RAG_EXPANSION_FACTOR', '1.5'))
                },
                'merge_agent': {
                    'enabled': os.getenv('OARC_RAG_MERGE_AGENT', 'false').lower() == 'true',
                    'similarity_threshold': float(os.getenv('OARC_RAG_MERGE_THRESHOLD', '0.85'))
                },
                'split_agent': {
                    'enabled': os.getenv('OARC_RAG_SPLIT_AGENT', 'false').lower() == 'true',
                    'max_chunk_length': int(os.getenv('OARC_RAG_MAX_CHUNK', '1024'))
                },
                'prune_agent': {
                    'enabled': os.getenv('OARC_RAG_PRUNE_AGENT', 'false').lower() == 'true',
                    'relevance_threshold': float(os.getenv('OARC_RAG_PRUNE_THRESHOLD', '0.6'))
                }
            },
            
            # Visualization (from Specification.md)
            'visualization': {
                'enabled': os.getenv('OARC_RAG_VISUALIZATION', 'false').lower() == 'true',
                'websocket_port': int(os.getenv('OARC_RAG_WEBSOCKET_PORT', '8765')),
                'pygame_width': int(os.getenv('OARC_RAG_PYGAME_WIDTH', '1280')),
                'pygame_height': int(os.getenv('OARC_RAG_PYGAME_HEIGHT', '800')),
                'background_color': os.getenv('OARC_RAG_BG_COLOR', '#000000'),
                'node_colors': {
                    'document': os.getenv('OARC_RAG_DOC_COLOR', '#4488ff'),
                    'chunk': os.getenv('OARC_RAG_CHUNK_COLOR', '#44ff88'), 
                    'query': os.getenv('OARC_RAG_QUERY_COLOR', '#ff4488')
                }
            },
            
            # API Settings (from Specification.md)
            'api': {
                'enabled': os.getenv('OARC_RAG_API_ENABLED', 'false').lower() == 'true',
                'host': os.getenv('OARC_RAG_API_HOST', '0.0.0.0'),
                'port': int(os.getenv('OARC_RAG_API_PORT', '8000')),
                'max_concurrent_requests': int(os.getenv('OARC_RAG_MAX_CONCURRENT', '5')),
                'request_timeout': int(os.getenv('OARC_RAG_REQUEST_TIMEOUT', '60')),
                'enable_cors': os.getenv('OARC_RAG_ENABLE_CORS', 'true').lower() == 'true',
                'cors_origins': os.getenv('OARC_RAG_CORS_ORIGINS', '*').split(','),
                'api_key_required': os.getenv('OARC_RAG_API_AUTH', 'false').lower() == 'true'
            },
            
            # Monitoring and logging (from Specification.md)
            'monitoring': {
                'performance_tracking': os.getenv('OARC_RAG_PERF_TRACKING', 'true').lower() == 'true',
                'log_level': os.getenv('OARC_RAG_LOG_LEVEL', 'info'),
                'metrics_collection': os.getenv('OARC_RAG_METRICS', 'true').lower() == 'true',
                'dashboard_enabled': os.getenv('OARC_RAG_DASHBOARD', 'false').lower() == 'true',
                'dashboard_port': int(os.getenv('OARC_RAG_DASHBOARD_PORT', '8050')),
                'profiling_enabled': os.getenv('OARC_RAG_PROFILING', 'false').lower() == 'true'
            },
            
            # Supported file types for sources
            'supported_file_extensions': SUPPORTED_FILE_EXTENSIONS,
            
            # Security settings
            'security': {
                'sandbox_mode': os.getenv('OARC_RAG_SANDBOX', 'true').lower() == 'true',
                'file_size_limit_mb': float(os.getenv('OARC_RAG_FILE_SIZE_LIMIT', '10.0')),
                'enable_content_filtering': os.getenv('OARC_RAG_CONTENT_FILTER', 'true').lower() == 'true',
                'api_rate_limit': int(os.getenv('OARC_RAG_API_RATE_LIMIT', '100'))
            }
        }
        
        # Override defaults with settings from config file
        self._merge_settings(default_settings)
        
        # Create required directories
        self._create_required_dirs()
        
        self._initialized = True
        
    def _merge_settings(self, defaults: Dict[str, Any]) -> None:
        """
        Merge default settings with loaded settings.
        
        Args:
            defaults: Default settings dictionary
        """
        # Deep merge settings with defaults
        for key, value in defaults.items():
            if key not in self.settings:
                self.settings[key] = value
            elif isinstance(value, dict) and isinstance(self.settings[key], dict):
                # Recursively merge nested dictionaries
                for inner_key, inner_value in value.items():
                    if inner_key not in self.settings[key]:
                        self.settings[key][inner_key] = inner_value
                        
    def _create_required_dirs(self) -> None:
        """Create required directories."""
        dirs_to_create = [
            self.settings['output_dir'],
            self.settings['templates_dir'],
            Path(self.settings['vector_db']['dir']),
            Path(self.settings['hnsw']['index_path']).parent
        ]
        
        for directory in dirs_to_create:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                log.warning(f"Failed to create directory {directory}: {e}")
                
    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        Save current configuration to a file.
        
        Args:
            file_path: Path to save the config file (uses current if None)
        """
        save_path = file_path or self.config_file or 'oarc_rag_config.json'
        
        try:
            # Convert Path objects to strings for JSON serialization
            settings_copy = self._prepare_settings_for_save(self.settings)
            
            with open(save_path, 'w') as f:
                json.dump(settings_copy, f, indent=2)
                
            log.info(f"Configuration saved to {save_path}")
        except Exception as e:
            log.error(f"Failed to save configuration to {save_path}: {e}")
            
    def _prepare_settings_for_save(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Path objects to strings for JSON serialization.
        
        Args:
            settings: Settings dictionary to prepare
            
        Returns:
            Dict ready for JSON serialization
        """
        prepared = {}
        
        for key, value in settings.items():
            if isinstance(value, Path):
                prepared[key] = str(value)
            elif isinstance(value, dict):
                prepared[key] = self._prepare_settings_for_save(value)
            else:
                prepared[key] = value
                
        return prepared
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._initialized = False
        self.__init__()
        log.info("Configuration reset to defaults")
        
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (use dot notation for nested keys: 'vector.use_pca')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        config = Config()
        if '.' not in key:
            return config.settings.get(key, default)
            
        # Handle nested keys
        parts = key.split('.')
        current = config.settings
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
        
    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (use dot notation for nested keys)
            value: Value to set
        """
        config = Config()
        if '.' not in key:
            config.settings[key] = value
            return
            
        # Handle nested keys
        parts = key.split('.')
        current = config.settings
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
            
        current[parts[-1]] = value
        
    @staticmethod
    def update(updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            Config.set(key, value)
            
    @staticmethod
    def get_all() -> Dict[str, Any]:
        """
        Get all configuration settings.
        
        Returns:
            Dict with all settings
        """
        return Config().settings.copy()
        
    @staticmethod
    def get_base_dir() -> Path:
        """Get the base directory path."""
        return Config().settings['base_dir']
    
    @staticmethod
    def get_output_dir() -> Path:
        """Get the output directory path."""
        return Config().settings['output_dir']
    
    @staticmethod
    def get_templates_dir() -> Path:
        """Get the templates directory path."""
        return Config().settings['templates_dir']
    
    @staticmethod
    def get_ollama_url() -> str:
        """Get the Ollama API URL."""
        return Config().settings['ollama_api_url']
    
    @staticmethod
    def get_embedding_model() -> str:
        """Get the model used for embeddings."""
        return Config().settings['embedding_model']
    
    @staticmethod
    def get_vector_db_path() -> Path:
        """Get the vector database path as a Path object."""
        return Path(Config().settings['vector_db']['dir'])
    
    @staticmethod
    def add_supported_file_extension(extension: str) -> None:
        """
        Add a new supported file extension.
        
        Args:
            extension: File extension to add (with or without dot)
        """
        ext = extension if extension.startswith('.') else f'.{extension}'
        if ext not in Config().settings['supported_file_extensions']:
            Config().settings['supported_file_extensions'].append(ext)
            
    @staticmethod
    def get_supported_file_extensions() -> List[str]:
        """Get the list of supported file extensions."""
        return Config().settings['supported_file_extensions']
        
    @staticmethod
    def get_agent_config(agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.
        
        Args:
            agent_type: Type of agent (rag_agent, expansion_agent, etc.)
            
        Returns:
            Dict with agent configuration
        """
        return Config().settings['agents'].get(agent_type, {})
        
    @staticmethod
    def get_operation_mode() -> str:
        """
        Get the current operational mode.
        
        Returns:
            str: 'awake' or 'sleep'
        """
        return Config().settings['operational_modes']['default_mode']
    
    @staticmethod
    def set_operation_mode(mode: str) -> None:
        """
        Set the operational mode.
        
        Args:
            mode: 'awake' or 'sleep'
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ('awake', 'sleep'):
            raise ValueError(f"Invalid operational mode: {mode}. Must be 'awake' or 'sleep'.")
            
        Config().settings['operational_modes']['default_mode'] = mode
    
    @staticmethod
    def get_cache_config() -> Dict[str, Any]:
        """
        Get caching configuration.
        
        Returns:
            Dict[str, Any]: Cache configuration
        """
        return Config().settings['caching']
    
    @staticmethod
    def get_vector_ops_config() -> Dict[str, Any]:
        """
        Get vector operations configuration.
        
        Returns:
            Dict[str, Any]: Vector operations configuration
        """
        return Config().settings['vector']
        
    @staticmethod
    def get_hnsw_config() -> Dict[str, Any]:
        """
        Get HNSW configuration.
        
        Returns:
            Dict[str, Any]: HNSW configuration
        """
        return Config().settings['hnsw']
        
    @staticmethod
    def is_visualization_enabled() -> bool:
        """
        Check if visualization is enabled.
        
        Returns:
            bool: True if enabled
        """
        return Config().settings['visualization']['enabled']
        
    @staticmethod
    def is_gpu_enabled() -> bool:
        """
        Check if GPU usage is enabled.
        
        Returns:
            bool: True if enabled
        """
        return Config().settings['hardware']['use_gpu']
        
    @staticmethod
    def is_api_enabled() -> bool:
        """
        Check if API is enabled.
        
        Returns:
            bool: True if enabled
        """
        return Config().settings['api']['enabled']
        
    @staticmethod
    def should_use_semantic_reranking() -> bool:
        """
        Check if semantic reranking is enabled.
        
        Returns:
            bool: True if enabled
        """
        return Config().settings['semantic_reranking']


# Create a default instance for backward compatibility
config = Config()

# Export common settings as global variables for backward compatibility
BASE_DIR = config.settings['base_dir']
OUTPUT_DIR = config.settings['output_dir']
