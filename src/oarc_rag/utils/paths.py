"""
Path management utilities for oarc_rag.
"""
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Union

from oarc_rag.utils.log import log
from oarc_rag.utils.const import (
    DEFAULT_OUTPUT_DIRNAME, 
    DEFAULT_DATA_DIRNAME, 
    DEFAULT_CONFIG_DIRNAME,
    SUPPORTED_FILE_EXTENSIONS
)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path: Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to project root
    """
    # Calculate the project root by going up from this file
    return Path(__file__).resolve().parent.parent.parent.parent


def get_app_dirs() -> Dict[str, Path]:
    """
    Get application directories for configs, cache, and data.
    
    Returns:
        Dict[str, Path]: Dictionary of directory paths
    """
    # Platform-specific config locations
    app_name = 'oarc_rag'
    
    if os.name == 'nt':  # Windows
        app_data = Path(os.environ.get('APPDATA', ''))
        config_dir = app_data / app_name
        cache_dir = Path(os.environ.get('LOCALAPPDATA', '')) / app_name / 'cache'
        data_dir = app_data / app_name / 'data'
    elif os.name == 'darwin':  # macOS
        home = Path.home()
        config_dir = home / 'Library' / 'Application Support' / app_name
        cache_dir = home / 'Library' / 'Caches' / app_name
        data_dir = home / 'Library' / 'Application Support' / app_name / 'data'
    else:  # Linux/Unix
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


def get_temp_directory() -> Path:
    """
    Get a temporary directory for oarc_rag.
    
    Returns:
        Path: Path to temporary directory
    """
    temp_dir = Path(tempfile.gettempdir()) / "oarc_rag"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_output_directory() -> Path:
    """
    Get the output directory, creating it if necessary.
    
    Returns:
        Path: Path to output directory
    """
    # First check if it's defined in config
    try:
        from oarc_rag.utils.config import config
        output_dir = config.get_output_dir()
    except (ImportError, AttributeError):
        # Fall back to default if config isn't available
        output_dir = get_project_root() / DEFAULT_OUTPUT_DIRNAME
    
    ensure_directory(output_dir)
    return output_dir


def get_vector_db_directory() -> Path:
    """
    Get the vector database directory, creating it if necessary.
    
    Returns:
        Path: Path to vector DB directory
    """
    try:
        from oarc_rag.utils.config import config
        db_dir = config.get_vector_db_path()
    except (ImportError, AttributeError):
        # Fall back to default
        db_dir = get_project_root() / 'vector_db'
    
    ensure_directory(db_dir)
    return db_dir


def get_templates_directory() -> Path:
    """
    Get the templates directory for prompt templates.
    
    Returns:
        Path: Path to templates directory
    """
    try:
        from oarc_rag.utils.config import config
        templates_dir = config.get_templates_dir()
    except (ImportError, AttributeError):
        # Fall back to default
        templates_dir = get_project_root() / 'templates'
    
    ensure_directory(templates_dir)
    return templates_dir


def create_unique_file_path(base_dir: Union[str, Path], name: str, extension: str = "md") -> Path:
    """
    Create a unique file path that doesn't overwrite existing files.
    
    Args:
        base_dir: Base directory for the file
        name: Base name for the file
        extension: File extension (without dot)
        
    Returns:
        Path: Unique file path
    """
    # Sanitize the name and ensure extension doesn't have leading dot
    from oarc_rag.utils.utils import sanitize_filename
    safe_name = sanitize_filename(name)
    ext = extension.lstrip('.')
    
    dir_path = ensure_directory(base_dir)
    counter = 0
    
    # Start with the base name
    file_path = dir_path / f"{safe_name}.{ext}"
    
    # If file exists, append a counter until we find a unique name
    while file_path.exists():
        counter += 1
        file_path = dir_path / f"{safe_name}_{counter}.{ext}"
    
    return file_path


def is_valid_source_file(path: Union[str, Path]) -> bool:
    """
    Check if a file is a valid source file based on extension.
    
    Args:
        path: Path to file
        
    Returns:
        bool: True if valid, False otherwise
    """
    file_path = Path(path)
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS


def find_source_files(
    source_paths: List[Union[str, Path]], 
    recursive: bool = True,
    skip_hidden: bool = True
) -> List[Path]:
    """
    Find all source files from a list of directories and files.
    
    Args:
        source_paths: List of source paths (files or directories)
        recursive: Whether to search directories recursively
        skip_hidden: Whether to skip hidden files and directories
        
    Returns:
        List[Path]: List of source file paths
    """
    result = []
    
    for source in source_paths:
        path = Path(source)
        
        # Skip hidden files/dirs if requested
        if skip_hidden and path.name.startswith('.'):
            continue
            
        # If path is a file and has supported extension, add it
        if path.is_file() and is_valid_source_file(path):
            result.append(path)
            
        # If path is a directory, search for files
        elif path.is_dir():
            if recursive:
                # Use recursive glob to find all files
                for ext in SUPPORTED_FILE_EXTENSIONS:
                    for p in path.glob(f"**/*{ext}"):
                        if p.is_file() and (not skip_hidden or not any(part.startswith('.') for part in p.parts)):
                            result.append(p)
            else:
                # Only search immediate directory
                for ext in SUPPORTED_FILE_EXTENSIONS:
                    for p in path.glob(f"*{ext}"):
                        if p.is_file() and (not skip_hidden or not p.name.startswith('.')):
                            result.append(p)
                    
    return result
