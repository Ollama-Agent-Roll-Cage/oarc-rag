"""
General utility functions for the oarc_rag project.
"""
import os
import sys
import platform
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Union, TypeVar, Tuple, Optional

from oarc_rag.utils.log import log

# Type variable for generic functions
T = TypeVar('T')


class Utils:
    """Static utility class providing helper functions for the OARC-RAG system."""
    
    @staticmethod
    def safe_to_int(value: Any, default: int = 0) -> int:
        """
        Safely convert a value to integer with fallback.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            int: Converted integer or default value
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def safe_to_float(value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float with fallback.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            float: Converted float or default value
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def find_files_by_extensions(
        root_dir: Union[str, Path], 
        extensions: List[str],
        skip_hidden: bool = True
    ) -> List[Path]:
        """
        Find all files with specific extensions under a directory.
        
        Args:
            root_dir: Root directory to search in
            extensions: List of file extensions to include (with dot)
            skip_hidden: Whether to skip hidden files and directories
            
        Returns:
            List[Path]: List of found file paths
        """
        root_path = Path(root_dir)
        result = []
        
        if not root_path.exists():
            return result
        
        try:
            for path in root_path.rglob('*'):
                # Skip hidden items if requested
                if skip_hidden and any(p.startswith('.') for p in path.parts):
                    continue
                    
                if path.is_file() and path.suffix.lower() in extensions:
                    result.append(path)
        except PermissionError:
            # Handle permission errors gracefully
            log.warning(f"Permission denied when accessing {root_path}")
            pass
                    
        return result

    @staticmethod
    def get_app_dirs() -> Dict[str, Path]:
        """
        Get application directories for configs, cache, and data.
        
        Returns:
            Dict[str, Path]: Dictionary of directory paths
        """
        # Platform-specific config locations
        app_name = 'oarc_rag'
        
        if sys.platform == 'win32':
            app_data = Path(os.environ.get('APPDATA', ''))
            config_dir = app_data / app_name
            cache_dir = Path(os.environ.get('LOCALAPPDATA', '')) / app_name / 'cache'
            data_dir = app_data / app_name / 'data'
        elif sys.platform == 'darwin':
            home = Path.home()
            config_dir = home / 'Library' / 'Application Support' / app_name
            cache_dir = home / 'Library' / 'Caches' / app_name
            data_dir = home / 'Library' / 'Application Support' / app_name / 'data'
        else:
            # Linux and other Unix-like systems
            home = Path.home()
            config_dir = home / '.config' / app_name
            cache_dir = home / '.cache' / app_name
            data_dir = home / '.local' / 'share' / app_name
        
        # Create directories if they don't exist
        for directory in [config_dir, cache_dir, data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        return {
            'config': config_dir,
            'cache': cache_dir,
            'data': data_dir,
            'temp': Path(tempfile.gettempdir()) / app_name
        }

    @staticmethod
    def get_system_info() -> Dict[str, str]:
        """
        Get system information for diagnostics.
        
        Returns:
            Dict[str, str]: System information
        """
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'node': platform.node()
        }

    @staticmethod
    def get_timestamp(format_str: str = '%Y%m%d_%H%M%S') -> str:
        """
        Get a formatted timestamp string.
        
        Args:
            format_str: Format string for datetime.strftime
            
        Returns:
            str: Formatted timestamp
        """
        return datetime.now().strftime(format_str)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a string to be used as a filename.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Trim spaces and limit length
        filename = filename.strip()
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            name = name[:max_length - len(ext)]
            filename = name + ext
            
        return filename

    @staticmethod
    def check_for_ollama(raise_error: bool = True) -> bool:
        """
        Check if Ollama server is available and responding.
        
        Args:
            raise_error: Whether to raise an error if Ollama is not available
        
        Returns:
            bool: True if Ollama is available
            
        Raises:
            RuntimeError: If Ollama is not available and raise_error is True
        """
        import requests
        from urllib.parse import urljoin
        
        base_url = "http://localhost:11434"
        
        try:
            # Try to connect to Ollama API
            response = requests.get(base_url, timeout=5)
            if response.status_code == 200:
                return True
            
            error_msg = (
                "Ollama server is not available or returned an unexpected response. "
                "Please ensure Ollama is installed, running, and responding correctly. "
                "Visit https://ollama.ai/download for installation instructions."
            )
            if raise_error:
                raise RuntimeError(error_msg)
            return False
        except requests.RequestException as e:
            error_msg = (
                f"Failed to connect to Ollama server at {base_url}: {e}. "
                "Please ensure Ollama is installed and running. "
                "Visit https://ollama.ai/download for installation instructions."
            )
            if raise_error:
                raise RuntimeError(error_msg)
            return False

    @staticmethod
    def validate_ollama_model(model_name: str, ollama_url: str = "http://localhost:11434", raise_error: bool = True) -> bool:
        """
        Validate if a model exists in Ollama.
        
        Args:
            model_name: Name of the model to validate
            ollama_url: URL of the Ollama API
            raise_error: Whether to raise an error if the model is not found
            
        Returns:
            bool: True if model exists, False otherwise
            
        Raises:
            RuntimeError: If raise_error is True and model does not exist
        """
        import requests
        
        try:
            # First check if Ollama is available
            if not Utils.check_for_ollama(raise_error=False):
                if raise_error:
                    raise RuntimeError(f"Cannot validate model '{model_name}': Ollama server is not available")
                return False
                
            # Then check if the model exists
            response = requests.post(
                f"{ollama_url}/api/show", 
                json={"name": model_name},
                timeout=5
            )
            
            if response.status_code == 200:
                return True
            else:
                error_msg = f"Model '{model_name}' not found in Ollama"
                if raise_error:
                    raise RuntimeError(error_msg)
                log.warning(error_msg)
                return False
        except Exception as e:
            if raise_error:
                raise RuntimeError(f"Error validating model '{model_name}': {e}")
            log.warning(f"Error validating model '{model_name}': {e}")
            return False

    @staticmethod
    def get_available_ollama_models(ollama_url: str = "http://localhost:11434") -> List[str]:
        """
        Get a list of available models from Ollama.
        
        Args:
            ollama_url: URL of the Ollama API
            
        Returns:
            List[str]: List of available model names
        """
        import requests
        
        try:
            if not Utils.check_for_ollama(raise_error=False):
                return []
                
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                return []
        except Exception as e:
            log.warning(f"Error getting available models: {e}")
            return []

    @staticmethod
    def detect_gpu() -> Tuple[bool, Optional[str]]:
        """
        Detect if GPU is available for FAISS acceleration.
        
        Returns:
            Tuple[bool, Optional[str]]: (GPU available, reason if not available)
        """
        try:
            # Only import if needed to avoid dependency issues
            import torch
            if torch.cuda.is_available():
                return True, None
            else:
                return False, "CUDA not available in torch"
        except ImportError:
            return False, "PyTorch not installed or CUDA not available"
        except Exception as e:
            return False, f"Error checking GPU: {e}"
