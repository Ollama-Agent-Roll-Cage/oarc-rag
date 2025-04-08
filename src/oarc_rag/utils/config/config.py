"""Configuration settings for the OARC-RAG system."""
import os
from pathlib import Path
from typing import Dict, Any

from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log

@singleton
class Config:
    """
    Configuration class for the OARC-RAG system.
    
    This class manages all configuration settings with a single flat dictionary
    for simplicity and performance. All methods are static to provide a clean
    configuration API.
    """
    
    def __init__(self):
        """Initialize configuration settings from environment variables."""
        # Only initialize once (singleton pattern)
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        # Base paths
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        
        # Consolidated settings in a single flat dictionary for simplicity
        self.settings = {
            # Base paths
            'base_dir': base_dir,
            'output_dir': Path(os.getenv('OARC_RAG_OUTPUT_DIR', base_dir / 'output')),
            
            # AI settings
            'model': os.getenv('OARC_RAG_AI_MODEL', 'llama3.1:latest'),
            'temperature': float(os.getenv('OARC_RAG_AI_TEMP', '0.7')),
            'max_tokens': int(os.getenv('OARC_RAG_AI_MAX_TOKENS', '4000')),
            'ollama_api_url': os.getenv('OARC_RAG_OLLAMA_URL', 'http://localhost:11434'),
            'default_model': os.getenv('OARC_RAG_DEFAULT_MODEL', 'llama3.1:latest'),
            
            # RAG settings
            'embedding_model': os.getenv('OARC_RAG_EMBEDDING_MODEL', 'llama3.1:latest'),
            'embedding_model_type': 'ollama',  # Always use Ollama for embeddings
            'chunk_size': int(os.getenv('OARC_RAG_CHUNK_SIZE', '512')),
            'chunk_overlap': int(os.getenv('OARC_RAG_CHUNK_OVERLAP', '50')),
            'retrieval_top_k': int(os.getenv('OARC_RAG_RETRIEVAL_TOP_K', '5')),
            'similarity_threshold': float(os.getenv('OARC_RAG_SIMILARITY_THRESHOLD', '0.7')),
            
            # Vector DB settings
            'vector_db_type': os.getenv('OARC_RAG_VECTOR_DB_TYPE', 'faiss'),
            'vector_db_dir': os.getenv(
                'OARC_RAG_VECTOR_DB_DIR', 
                str(base_dir / 'vector_db')
            ),
            'collection_name': os.getenv('OARC_RAG_VECTOR_DB_COLLECTION', 'default_collection'),
            'use_gpu': os.getenv('OARC_RAG_USE_GPU', 'auto').lower() in ('true', 'yes', '1', 'auto'),
            
            # Embedding settings
            'embedding_dimensions': int(os.getenv('OARC_RAG_EMBEDDING_DIMENSIONS', '4096')),
            
            # GPU settings
            'gpu_memory_threshold': float(os.getenv('OARC_RAG_GPU_MEMORY_THRESHOLD', '0.75')),
            'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES', None),
            
            # Supported file types for sources
            'supported_file_extensions': [
                '.txt', '.md', '.tex', '.rst', '.html',
                '.py', '.js', '.java', '.cpp', '.c',
                '.json', '.yaml', '.yml', '.csv',
                '.pdf'
            ]
        }
        
        self._initialized = True
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return Config().settings.get(key, default)
    
    @staticmethod
    def update(updates: Dict[str, Any]) -> None:
        """
        Update configuration settings.
        
        Args:
            updates: Dictionary of settings to update
        """
        Config().settings.update(updates)
    
    @staticmethod
    def get_all() -> Dict[str, Any]:
        """
        Get all configuration settings.
        
        Returns:
            Dict[str, Any]: All configuration settings
        """
        return Config().settings
    
    @staticmethod
    def get_base_dir() -> Path:
        """Get the base directory path."""
        return Config().settings['base_dir']
    
    @staticmethod
    def get_output_dir() -> Path:
        """Get the output directory path."""
        return Config().settings['output_dir']
    
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
        return Path(Config().settings['vector_db_dir'])
    
    @staticmethod
    def add_supported_file_extension(extension: str) -> None:
        """
        Add a file extension to the supported list.
        
        Args:
            extension: File extension to add (with dot, e.g., '.docx')
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
            
        extensions = Config().settings['supported_file_extensions']
        if extension not in extensions:
            extensions.append(extension)
            Config().settings['supported_file_extensions'] = extensions


# Create a default instance for backward compatibility
config = Config()

# Export common settings as global variables for backward compatibility
BASE_DIR = config.settings['base_dir']
OUTPUT_DIR = config.settings['output_dir']
