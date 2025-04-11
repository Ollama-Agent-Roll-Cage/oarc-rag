"""
Configuration management for oarc_rag.

This module provides a centralized configuration system that supports loading from
files, environment variables, and defaults, with hierarchical key access.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton


@singleton
class Config:
    """
    Configuration management for the oarc_rag system.
    
    This class handles loading, accessing, and managing configuration
    settings from various sources including files and environment variables.
    Implements the singleton pattern to ensure consistent configuration throughout the application.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration system.
        
        Args:
            config_path: Optional path to a configuration file
        """
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {
            "vector_dir": "vectors",
            "output_dir": "output",
            "data_dir": "data",
            "temp_dir": "temp",
            "cache_dir": "cache",
            "templates_dir": "templates",
            "embedding": {
                "model": "llama3.1:latest",
                "dimensions": 1024,
                "use_pca": False,
                "pca_dimensions": 128
            },
            "generation": {
                "model": "llama3.1:latest",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "retrieval": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 5,
                "similarity_threshold": 0.5,
                "enable_reranking": True
            },
            "ai": {
                "ollama_api_url": "http://localhost:11434",
                "default_model": "llama3.1:latest",
                "temperature": 0.7,
                "max_tokens": 4000
            },
            "operational_mode": {
                "default": "awake",
                "awake_timeout": 3600,
                "sleep_cycle_duration": 1800,
                "knowledge_consolidation_interval": 86400
            },
            "vector_db": {
                "use_quantization": False,
                "use_pca": False,
                "hnsw_ef_construction": 200,
                "hnsw_ef_search": 50,
                "hnsw_m": 16
            },
            "caching": {
                "enabled": True,
                "response_ttl": 3600,
                "context_ttl": 7200,
                "template_ttl": 86400
            }
        }
        
        # Load default configuration
        self._config = self._defaults.copy()
        
        # Load from config file if provided
        if config_path:
            self.load_config(config_path)
        else:
            # Try to find config in standard locations
            self._load_from_default_locations()
            
        # Load from environment variables
        self._load_from_env()
        
    def _load_from_default_locations(self) -> None:
        """Try to load configuration from default locations."""
        # Import here to avoid circular imports
        from oarc_rag.utils.paths import get_app_dirs, get_project_root
        
        # Check project root location
        root_config = get_project_root() / "config.json"
        if root_config.exists():
            self.load_config(root_config)
            return
            
        root_config = get_project_root() / "config.yaml"
        if root_config.exists():
            self.load_config(root_config)
            return
        
        # Check app config directory
        app_dirs = get_app_dirs()
        config_dir = app_dirs.get('config')
        if config_dir:
            config_file = config_dir / "config.json"
            if config_file.exists():
                self.load_config(config_file)
                return
                
            config_file = config_dir / "config.yaml"
            if config_file.exists():
                self.load_config(config_file)
                return
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Check for key environment variables
        env_prefix = "OARC_RAG_"
        
        # Example mappings
        mappings = {
            "VECTOR_DIR": "vector_dir",
            "OUTPUT_DIR": "output_dir",
            "OLLAMA_API_URL": "ai.ollama_api_url",
            "DEFAULT_MODEL": "ai.default_model",
            "TEMPERATURE": "ai.temperature"
        }
        
        for env_key, config_key in mappings.items():
            full_env_key = env_prefix + env_key
            if full_env_key in os.environ:
                self.set(config_key, os.environ[full_env_key])
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (json or yaml)
        """
        path = Path(config_path)
        
        if not path.exists():
            log.warning(f"Config file not found: {path}")
            return
            
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    file_config = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    file_config = yaml.safe_load(f)
            else:
                log.warning(f"Unsupported config file format: {path.suffix}")
                return
                
            # Merge with current config
            self._merge_configs(file_config)
            log.info(f"Loaded configuration from {path}")
            
        except Exception as e:
            log.error(f"Failed to load config from {path}: {e}")
    
    def _merge_configs(self, new_config: Dict[str, Any], target: Optional[Dict[str, Any]] = None, path: str = "") -> None:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            new_config: New configuration to merge
            target: Target dictionary to merge into (if None, use self._config)
            path: Current path for nested keys
        """
        if target is None:
            target = self._config
            
        for key, value in new_config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively merge nested dictionaries
                self._merge_configs(value, target[key], current_path)
            else:
                # Set or override value
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        # Handle nested keys
        if "." in key:
            parts = key.split(".")
            config = self._config
            
            for part in parts[:-1]:
                config = config.get(part, {})
                if not isinstance(config, dict):
                    return default
                    
            return config.get(parts[-1], default)
        
        # Simple top-level key
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (can be nested with dots)
            value: Value to set
        """
        # Handle nested keys
        if "." in key:
            parts = key.split(".")
            config = self._config
            
            for part in parts[:-1]:
                if part not in config or not isinstance(config[part], dict):
                    config[part] = {}
                config = config[part]
                    
            config[parts[-1]] = value
        else:
            # Simple top-level key
            self._config[key] = value
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.
        
        Returns:
            Dict with all configuration values
        """
        return self._config.copy()
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to a file.
        
        Args:
            config_path: Path to save configuration
        """
        path = Path(config_path)
        
        try:
            os.makedirs(path.parent, exist_ok=True)
            
            if path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(self._config, f)
            else:
                log.warning(f"Unsupported config file format: {path.suffix}")
                return
                
            log.info(f"Saved configuration to {path}")
            
        except Exception as e:
            log.error(f"Failed to save config to {path}: {e}")
    
    def get_base_dir(self) -> Path:
        """
        Get the base directory for application data.
        
        Returns:
            Path to base directory
        """
        # Import here to avoid circular imports
        from oarc_rag.utils.paths import get_app_dirs
        
        app_dirs = get_app_dirs()
        return app_dirs.get('data')
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._defaults.copy()
        
    @classmethod
    def get_instance(cls) -> 'Config':
        """
        Get the singleton instance of Config.
        
        Returns:
            Config singleton instance
        """
        return cls()
