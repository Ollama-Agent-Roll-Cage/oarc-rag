"""
Path management utilities for oarc_rag.
"""
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from datetime import datetime

from jinja2 import Environment

from oarc_rag.utils.log import log
from oarc_rag.utils.const import (
    DEFAULT_OUTPUT_DIRNAME, 
    DEFAULT_DATA_DIRNAME, 
    DEFAULT_CONFIG_DIRNAME,
    SUPPORTED_FILE_EXTENSIONS
)


class Paths:
    """
    Static class for path management in oarc_rag.
    
    This class provides static methods for managing paths, directories,
    and file operations throughout the application.
    """
    
    @staticmethod
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

    @staticmethod
    def get_project_root() -> Path:
        """
        Get the project root directory.
        
        Returns:
            Path: Path to project root
        """
        # Calculate the project root by going up from this file
        return Path(__file__).resolve().parent.parent.parent.parent

    @staticmethod
    def get_app_dirs() -> Dict[str, Path]:
        """
        Get application directories for configs, cache, and data.
        
        Returns:
            Dict[str, Path]: Dictionary of directory paths
        """
        # Platform-specific config locations
        app_name = 'oarc_rag'
        
        if os.name == 'nt':  # Windows
            # Use %APPDATA% on Windows
            base_dir = Path(os.environ.get('APPDATA', '~')).expanduser()
            config_dir = base_dir / app_name / DEFAULT_CONFIG_DIRNAME
            cache_dir = base_dir / app_name / 'cache'
            data_dir = base_dir / app_name / DEFAULT_DATA_DIRNAME
        elif os.name == 'darwin':  # macOS
            # Use ~/Library/Application Support/ on macOS
            base_dir = Path.home() / 'Library' / 'Application Support'
            config_dir = base_dir / app_name / DEFAULT_CONFIG_DIRNAME
            cache_dir = Path.home() / 'Library' / 'Caches' / app_name
            data_dir = base_dir / app_name / DEFAULT_DATA_DIRNAME
        else:  # Linux/Unix
            # Use XDG Base Directory Specification
            config_dir = Path(os.environ.get('XDG_CONFIG_HOME', '~/.config')).expanduser() / app_name
            cache_dir = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')).expanduser() / app_name
            data_dir = Path(os.environ.get('XDG_DATA_HOME', '~/.local/share')).expanduser() / app_name
        
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
    def get_temp_directory() -> Path:
        """
        Get a temporary directory for oarc_rag.
        
        Returns:
            Path: Path to temporary directory
        """
        temp_dir = Path(tempfile.gettempdir()) / "oarc_rag"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    @staticmethod
    def get_output_directory() -> Path:
        """
        Get the output directory, creating it if necessary.
        
        Returns:
            Path: Path to output directory
        """
        # First check if it's defined in config
        from oarc_rag.utils.config.config import Config
        
        config = Config()
        output_path = config.get('output_dir')
        
        if output_path:
            # Check if it's absolute or relative
            if os.path.isabs(output_path):
                output_dir = Path(output_path)
            else:
                output_dir = Paths.get_project_root() / output_path
        else:
            # Use default location
            output_dir = Paths.get_project_root() / DEFAULT_OUTPUT_DIRNAME
            
        # Ensure directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def get_vector_db_directory() -> Path:
        """
        Get the vector database directory.
        
        Returns:
            Path: Path to vector database directory
        """
        # Check if it's defined in config
        from oarc_rag.utils.config.config import Config
        
        config = Config()
        vector_dir = config.get('vector_dir')
        
        if vector_dir:
            # Check if it's absolute or relative
            if os.path.isabs(vector_dir):
                vector_db_dir = Path(vector_dir)
            else:
                vector_db_dir = Paths.get_project_root() / vector_dir
        else:
            # Use default location in data directory
            vector_db_dir = Paths.get_app_dirs()['data'] / 'vectors'
            
        # Ensure directory exists
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        return vector_db_dir

    @staticmethod
    def get_templates_directory() -> Path:
        """
        Get the templates directory.
        
        Returns:
            Path: Path to templates directory
        """
        # Check if it's defined in config
        from oarc_rag.utils.config.config import Config
        
        config = Config()
        templates_dir = config.get('templates_dir')
        
        if templates_dir:
            # Check if it's absolute or relative
            if os.path.isabs(templates_dir):
                templates_path = Path(templates_dir)
            else:
                templates_path = Paths.get_project_root() / templates_dir
        else:
            # Use default location
            templates_path = Paths.get_project_root() / 'templates'
            
        # Ensure directory exists
        templates_path.mkdir(parents=True, exist_ok=True)
        return templates_path

    @staticmethod
    def write_template_file(template_name: str, template_content: str) -> Path:
        """
        Write a template file to the templates directory.
        
        Args:
            template_name: Name of the template (without .j2 extension)
            template_content: Content to write to the template file
            
        Returns:
            Path to the written template file
        """
        templates_dir = Paths.get_templates_directory()
        template_path = templates_dir / f"{template_name}.j2"
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
            
        return template_path

    @staticmethod
    def template_file_exists(template_name: str) -> bool:
        """
        Check if a template file exists in the templates directory.
        
        Args:
            template_name: Name of the template (without .j2 extension)
            
        Returns:
            True if template file exists, False otherwise
        """
        templates_dir = Paths.get_templates_directory()
        template_path = templates_dir / f"{template_name}.j2"
        
        return template_path.exists()

    @staticmethod
    def get_template_files() -> List[Dict[str, Any]]:
        """
        List all template files in the templates directory.
        
        Returns:
            List of dictionaries with template file information
        """
        templates_dir = Paths.get_templates_directory()
        template_files = list(templates_dir.glob("*.j2"))
        
        result = []
        for template_path in template_files:
            result.append({
                'name': template_path.stem,
                'path': template_path,
                'size': template_path.stat().st_size,
                'modified': datetime.fromtimestamp(template_path.stat().st_mtime).isoformat()
            })
            
        return result

    @staticmethod
    def create_unique_file_path(base_dir: Union[str, Path], name: str, extension: str = "md") -> Path:
        """
        Create a unique file path by adding a number suffix if the file exists.
        
        Args:
            base_dir: The directory path
            name: The file name (without extension)
            extension: The file extension (without dot)
            
        Returns:
            Path: A unique file path
        """
        base_path = Path(base_dir)
        
        # Ensure directory exists
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Clean up extension (remove dot if present)
        extension = extension.lstrip('.')
        
        # Try the base filename first
        file_path = base_path / f"{name}.{extension}"
        if not file_path.exists():
            return file_path
        
        # If file exists, add a number suffix
        counter = 1
        while True:
            file_path = base_path / f"{name}_{counter}.{extension}"
            if not file_path.exists():
                return file_path
            counter += 1

    @staticmethod
    def is_valid_source_file(path: Union[str, Path]) -> bool:
        """
        Check if a file is a valid source file based on extension.
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if file has a supported extension, False otherwise
        """
        file_path = Path(path)
        
        # Check if it's a file
        if not file_path.is_file():
            return False
            
        # Check extension
        extension = file_path.suffix.lower().lstrip('.')
        return extension in SUPPORTED_FILE_EXTENSIONS

    @staticmethod
    def find_source_files(
        source_paths: List[Union[str, Path]], 
        recursive: bool = True,
        skip_hidden: bool = True
    ) -> List[Path]:
        """
        Find all valid source files in the given paths.
        
        Args:
            source_paths: List of paths to search
            recursive: Whether to search recursively in directories
            skip_hidden: Whether to skip hidden files and directories
            
        Returns:
            List[Path]: List of valid source file paths
        """
        result = []
        
        for source_path in source_paths:
            path = Path(source_path)
            
            # Skip if path doesn't exist
            if not path.exists():
                log.warning(f"Path does not exist: {path}")
                continue
                
            # Skip hidden files/dirs if requested
            if skip_hidden and path.name.startswith('.'):
                continue
                
            # If it's a file, check if it's valid
            if path.is_file():
                if Paths.is_valid_source_file(path):
                    result.append(path)
            # If it's a directory, search for files
            elif path.is_dir():
                if recursive:
                    # Recursively search
                    for item in path.rglob('*'):
                        # Skip hidden files/dirs if requested
                        if skip_hidden and (item.name.startswith('.') or any(p.name.startswith('.') for p in item.parents)):
                            continue
                            
                        if item.is_file() and Paths.is_valid_source_file(item):
                            result.append(item)
                else:
                    # Search only this directory
                    for item in path.glob('*'):
                        # Skip hidden files if requested
                        if skip_hidden and item.name.startswith('.'):
                            continue
                            
                        if item.is_file() and Paths.is_valid_source_file(item):
                            result.append(item)
        
        return result

    @staticmethod
    def create_template_environment(templates_dir: Optional[Path] = None) -> Environment:
        """
        Create a standard Jinja2 Environment for templates.
        
        Args:
            templates_dir: Optional custom templates directory (uses default if None)
            
        Returns:
            Configured Jinja2 Environment
        """
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        
        # Use provided directory or get default templates directory
        directory = templates_dir or Paths.get_templates_directory()
        
        # Create and return Jinja2 environment with standard settings
        return Environment(
            loader=FileSystemLoader(directory),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
