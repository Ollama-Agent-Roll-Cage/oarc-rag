"""
General utility functions for the oarc_rag project.
"""
import os
import sys
import re
import platform
import tempfile
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Union, TypeVar, Tuple, Optional

from oarc_rag.utils.log import log
from oarc_rag.utils.const import DEFAULT_OLLAMA_URL

# Type variable for generic functions
T = TypeVar('T')


class Utils:
    """Static utility class providing helper functions for the OARC-RAG system."""
    
    @staticmethod
    def safe_to_int(value: Any, default: int = 0) -> int:
        """
        Safely convert a value to int.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            int: Converted value or default
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_to_float(value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
            
        Returns:
            float: Converted value or default
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
        Find files with specific extensions in a directory.
        
        Args:
            root_dir: Directory to search
            extensions: List of file extensions to find (with or without dot)
            skip_hidden: Whether to skip hidden files and directories
            
        Returns:
            List[Path]: List of file paths
        """
        result = []
        root_path = Path(root_dir)
        
        # Normalize extensions to always have a dot prefix
        norm_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        try:
            for ext in norm_extensions:
                for path in root_path.glob(f'**/*{ext}'):
                    # Skip hidden files/directories if requested
                    if skip_hidden and any(part.startswith('.') for part in path.parts):
                        continue
                    
                    if path.is_file():
                        result.append(path)
                        
        except PermissionError:
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
        Get system information.
        
        Returns:
            Dict[str, str]: System information
        """
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        # Add CUDA info if available
        cuda_available, cuda_version = Utils.detect_gpu()
        info["cuda_available"] = str(cuda_available)
        if cuda_version:
            info["cuda_version"] = cuda_version
            
        return info
    
    @staticmethod
    def get_timestamp(format_str: str = '%Y%m%d_%H%M%S') -> str:
        """
        Get current timestamp as formatted string.
        
        Args:
            format_str: Timestamp format
            
        Returns:
            str: Formatted timestamp
        """
        return datetime.now().strftime(format_str)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a string to be used as a filename.
        
        Args:
            filename: Input string
            
        Returns:
            str: Sanitized filename
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        
        # Collapse multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Trim leading/trailing underscores and whitespace
        sanitized = sanitized.strip('_ ')
        
        # Ensure we have at least one character
        if not sanitized:
            sanitized = 'unnamed'
            
        # Truncate if too long (max 255 chars for most filesystems)
        if len(sanitized) > 200:
            sanitized = sanitized[:197] + '...'
            
        return sanitized
    
    @staticmethod
    def check_for_ollama(raise_error: bool = False) -> bool:
        """
        Check if Ollama server is available.
        
        Args:
            raise_error: Whether to raise an error if Ollama is not available
            
        Returns:
            bool: True if Ollama is available, False otherwise
            
        Raises:
            RuntimeError: If raise_error is True and Ollama is not available
        """
        base_url = DEFAULT_OLLAMA_URL
        
        try:
            response = requests.get(f"{base_url}/api/version", timeout=5)
            if response.status_code == 200:
                log.debug(f"Ollama server is available: {response.json()}")
                return True
                
            error_msg = (
                f"Ollama server returned status code {response.status_code}. "
                "Please ensure Ollama is installed and running. "
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
        Validate that an Ollama model exists and is available.
        
        Args:
            model_name: Model name to validate
            ollama_url: Ollama API URL
            raise_error: Whether to raise an error if model is not available
            
        Returns:
            bool: True if model is available, False otherwise
            
        Raises:
            RuntimeError: If raise_error is True and model is not available
        """
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                error_msg = f"Failed to list Ollama models. Status code: {response.status_code}"
                if raise_error:
                    raise RuntimeError(error_msg)
                return False
                
            models_data = response.json()
            models = [m.get('name') for m in models_data.get('models', [])]
            
            # Check if model_name exists exactly or as a prefix (e.g., llama3:latest)
            for m in models:
                if m == model_name or m.startswith(f"{model_name}:"):
                    return True
                    
            # Model not found
            error_msg = f"Model '{model_name}' not found in Ollama. Available models: {', '.join(models)}"
            if raise_error:
                raise RuntimeError(error_msg)
            return False
            
        except requests.RequestException as e:
            error_msg = f"Error connecting to Ollama server: {e}"
            if raise_error:
                raise RuntimeError(error_msg)
            return False
    
    @staticmethod
    def get_available_ollama_models(ollama_url: str = "http://localhost:11434") -> List[str]:
        """
        Get list of available Ollama models.
        
        Args:
            ollama_url: Ollama API URL
            
        Returns:
            List[str]: List of available model names
            
        Raises:
            RuntimeError: If failed to get models
        """
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to list Ollama models. Status code: {response.status_code}")
                
            models_data = response.json()
            return [m.get('name') for m in models_data.get('models', [])]
                
        except requests.RequestException as e:
            raise RuntimeError(f"Error connecting to Ollama server: {e}")
    
    @staticmethod
    def detect_gpu() -> Tuple[bool, Optional[str]]:
        """
        Detect if GPU (CUDA) is available and get version.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_available, version_string)
        """
        # Try to detect CUDA with nvidia-smi
        try:
            import subprocess
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                            stderr=subprocess.DEVNULL)
            version = output.decode('utf-8').strip()
            return True, version
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        # Try to detect with PyTorch if available
        try:
            import torch
            if torch.cuda.is_available():
                return True, f"CUDA {torch.version.cuda}"
        except ImportError:
            pass
            
        return False, None
    
    @staticmethod
    def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load data from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List[Dict[str, Any]]: List of records from JSONL
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data.append(json.loads(line))
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL format in {file_path}: {e}")
    
    @staticmethod
    def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
        """
        Save data to a JSONL file.
        
        Args:
            data: List of records to save
            file_path: Output file path
            
        Raises:
            ValueError: If data format is invalid
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record) + '\n')
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid data format: {e}")
