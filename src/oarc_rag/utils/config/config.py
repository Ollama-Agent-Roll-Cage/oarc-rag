"""Configuration settings for the OARC-RAG system."""
import os
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log

@singleton
class Config:
    """
    Configuration class for the OARC-RAG system.
    
    This class manages all configuration settings including paths,
    AI settings, RAG parameters, vector database settings, and
    embedding configurations. All methods are static to provide
    a clean configuration API.
    """
    
    def __init__(self):
        """Initialize configuration settings from environment variables."""
        # Only initialize once (singleton pattern)
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        # Base paths
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.output_dir = Path(os.getenv('OARC_RAG_OUTPUT_DIR', self.base_dir / 'output'))
        
        # AI Configuration
        self.ai_config = {
            'model': os.getenv('OARC_RAG_AI_MODEL', 'llama3.1:latest'),
            'temperature': float(os.getenv('OARC_RAG_AI_TEMP', '0.7')),
            'max_tokens': int(os.getenv('OARC_RAG_AI_MAX_TOKENS', '4000')),
            # Ollama specific settings
            'ollama_api_url': os.getenv('OARC_RAG_OLLAMA_URL', 'http://localhost:11434'),
            'default_model': os.getenv('OARC_RAG_DEFAULT_MODEL', 'llama3.1:latest'),
        }
        
        # RAG Configuration
        self.rag_config = {
            'embedding_model': os.getenv('OARC_RAG_EMBEDDING_MODEL', 'llama3.1:latest'),
            'embedding_model_type': 'ollama',  # Always use Ollama for embeddings
            'chunk_size': int(os.getenv('OARC_RAG_CHUNK_SIZE', '512')),
            'chunk_overlap': int(os.getenv('OARC_RAG_CHUNK_OVERLAP', '50')),
            'retrieval_top_k': int(os.getenv('OARC_RAG_RETRIEVAL_TOP_K', '5')),
            'similarity_threshold': float(os.getenv('OARC_RAG_SIMILARITY_THRESHOLD', '0.7')),
        }
        
        # Vector DB Configuration
        self.vector_db_config = {
            'type': os.getenv('OARC_RAG_VECTOR_DB_TYPE', 'faiss'),
            'persist_directory': os.getenv(
                'OARC_RAG_VECTOR_DB_DIR', 
                str(self.base_dir / 'vector_db')
            ),
            'collection_name': os.getenv('OARC_RAG_VECTOR_DB_COLLECTION', 'default_collection'),
        }
        
        # Ollama Embedding Configuration
        self.ollama_embedding_config = {
            'api_url': self.ai_config['ollama_api_url'],
            'model': self.rag_config['embedding_model'],
            'dimensions': int(os.getenv('OARC_RAG_EMBEDDING_DIMENSIONS', '4096')),
        }
        
        # Supported file types for sources
        self.supported_file_extensions = [
            '.txt', '.md', '.tex', '.rst', '.html',
            '.py', '.js', '.java', '.cpp', '.c',
            '.json', '.yaml', '.yml', '.csv',
            '.pdf'
        ]
        
        self._initialized = True
        
    @staticmethod
    def get_ollama_url() -> str:
        """Get the Ollama API URL."""
        return Config().ai_config['ollama_api_url']
    
    @staticmethod
    def get_default_model() -> str:
        """Get the default model for AI operations."""
        return Config().ai_config['default_model']
    
    @staticmethod
    def get_embedding_model() -> str:
        """Get the model used for embeddings."""
        return Config().rag_config['embedding_model']
    
    @staticmethod
    def get_vector_db_path() -> Path:
        """Get the vector database path as a Path object."""
        return Path(Config().vector_db_config['persist_directory'])
    
    @staticmethod
    def get_base_dir() -> Path:
        """Get the base directory path."""
        return Config().base_dir
    
    @staticmethod
    def get_output_dir() -> Path:
        """Get the output directory path."""
        return Config().output_dir
    
    @staticmethod
    def get_ai_config() -> Dict[str, Any]:
        """Get the AI configuration dictionary."""
        return Config().ai_config
    
    @staticmethod
    def get_rag_config() -> Dict[str, Any]:
        """Get the RAG configuration dictionary."""
        return Config().rag_config
    
    @staticmethod
    def get_vector_db_config() -> Dict[str, Any]:
        """Get the vector database configuration dictionary."""
        return Config().vector_db_config
    
    @staticmethod
    def get_ollama_embedding_config() -> Dict[str, Any]:
        """Get the Ollama embedding configuration dictionary."""
        return Config().ollama_embedding_config
    
    @staticmethod
    def get_supported_file_extensions() -> List[str]:
        """Get the list of supported file extensions."""
        return Config().supported_file_extensions
    
    @staticmethod
    def update_ai_config(updates: Dict[str, Any]) -> None:
        """
        Update AI configuration settings.
        
        Args:
            updates: Dictionary of settings to update
        """
        instance = Config()
        instance.ai_config.update(updates)
        # If we update the ollama_api_url, also update it in the embedding config
        if 'ollama_api_url' in updates:
            instance.ollama_embedding_config['api_url'] = updates['ollama_api_url']
    
    @staticmethod
    def update_rag_config(updates: Dict[str, Any]) -> None:
        """
        Update RAG configuration settings.
        
        Args:
            updates: Dictionary of settings to update
        """
        instance = Config()
        instance.rag_config.update(updates)
        # If we update the embedding_model, also update it in the embedding config
        if 'embedding_model' in updates:
            instance.ollama_embedding_config['model'] = updates['embedding_model']
    
    @staticmethod
    def update_vector_db_config(updates: Dict[str, Any]) -> None:
        """
        Update vector database configuration settings.
        
        Args:
            updates: Dictionary of settings to update
        """
        Config().vector_db_config.update(updates)
    
    @staticmethod
    def set_output_dir(path: Union[str, Path]) -> None:
        """
        Set the output directory.
        
        Args:
            path: New output directory path
        """
        Config().output_dir = Path(path)
    
    @staticmethod
    def add_supported_file_extension(extension: str) -> None:
        """
        Add a file extension to the supported list.
        
        Args:
            extension: File extension to add (with dot, e.g., '.docx')
        """
        instance = Config()
        if not extension.startswith('.'):
            extension = f'.{extension}'
        if extension not in instance.supported_file_extensions:
            instance.supported_file_extensions.append(extension)
    
    @staticmethod
    def reset() -> None:
        """
        Reset configuration to defaults (primarily for testing).
        """
        # The singleton decorator doesn't provide a clean way to reset,
        # so we'll reinitialize by accessing properties to trigger __init__
        instance = Config()
        instance._initialized = False
        instance.__init__()


# Create a default instance for backward compatibility
config = Config()

# For backward compatibility with code that uses the global variables
BASE_DIR = config.base_dir
OUTPUT_DIR = config.output_dir
AI_CONFIG = config.ai_config
RAG_CONFIG = config.rag_config
VECTOR_DB_CONFIG = config.vector_db_config
OLLAMA_EMBEDDING_CONFIG = config.ollama_embedding_config
SUPPORTED_FILE_EXTENSIONS = config.supported_file_extensions
