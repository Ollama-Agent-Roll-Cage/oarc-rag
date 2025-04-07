"""
Help documentation for the oarc_rag command line interface.

This module provides comprehensive help text and documentation
for the oarc_rag CLI commands related to RAG functionality.
"""
from typing import Optional, Dict, List, Any
import textwrap
import sys

from oarc_rag.utils.config.config import Config

# General help text for the CLI application
GENERAL_HELP = """
OARC-RAG: A powerful Retrieval-Augmented Generation system

Usage:
  oarc-rag <command> [options]

Available Commands:
  init       Initialize a new RAG engine
  add        Add documents to the RAG system
  query      Query the RAG system for information
  vector     Manage vector database operations
  embed      Generate embeddings for documents
  stats      Display statistics about the RAG system
  clean      Clean temporary files and caches
  help       Show help for any command

Use 'oarc-rag help <command>' for more information about a specific command.
"""

# Command-specific help text
COMMAND_HELP: Dict[str, str] = {
    "init": """
    Initialize a new RAG engine instance

    Usage:
      oarc-rag init [options]

    Options:
      --run-id ID                   Custom identifier for this run (default: timestamp)
      --base-dir PATH               Base directory for storage (default: ./output)
      --embedding-model MODEL       Model to use for embeddings (default: llama3.1:latest)
      --chunk-size SIZE             Size of text chunks in tokens (default: 512)
      --chunk-overlap OVERLAP       Overlap between consecutive chunks (default: 50)
      --vector-db {faiss,chroma}    Vector database backend (default: faiss)

    Examples:
      oarc-rag init --run-id="rag_20251231.001"
      oarc-rag init --embedding-model="llama3:latest" --chunk-size=512
    """,
    
    "add": """
    Add documents to the RAG system knowledge base

    Usage:
      oarc-rag add <source> [options]

    Arguments:
      source                        File, directory, or URL to process

    Options:
      --run-id ID                   Run ID to use (required if multiple runs exist)
      --recursive                   Process directories recursively (default: true)
      --metadata KEY=VALUE          Add metadata to the documents (can be used multiple times)
      --source-id ID                Custom source identifier
      --format {auto,text,pdf,md}   Force specific document format (default: auto)

    Examples:
      oarc-rag add ./documents/
      oarc-rag add https://example.com/article.html
      oarc-rag add ./textbook.pdf --metadata subject=physics
    """,
    
    "query": """
    Query the RAG system for information

    Usage:
      oarc-rag query <question> [options]

    Arguments:
      question                      The query to search for

    Options:
      --run-id ID                   Run ID to use (required if multiple runs exist)
      --top-k COUNT                 Number of results to return (default: 5)
      --threshold FLOAT             Minimum similarity score (0-1, default: 0.7)
      --source-filter SOURCE        Filter by source (can be used multiple times)
      --model MODEL                 Model for response generation (default: llama3.1:latest)
      --raw                         Return raw chunks without AI-generated response
      --format {text,json,markdown} Output format (default: text)

    Examples:
      oarc-rag query "What is retrieval-augmented generation?"
      oarc-rag query "How do transformers work?" --top-k=10 --source-filter=textbooks
    """,
    
    "vector": """
    Manage vector database operations

    Usage:
      oarc-rag vector <subcommand> [options]

    Subcommands:
      create     Create a new vector database
      info       Display information about a vector database
      purge      Delete all vectors from the database
      export     Export database to a file
      import     Import database from a file

    Options:
      --run-id ID                   Run ID to use (required if multiple runs exist)
      --path PATH                   Path to vector database (overrides default location)

    Examples:
      oarc-rag vector create --run-id="new_project"
      oarc-rag vector info
      oarc-rag vector export --path="./backup.vec"
    """,
    
    "embed": """
    Generate embeddings for documents without storing them

    Usage:
      oarc-rag embed <source> [options]

    Arguments:
      source                        File to generate embeddings for

    Options:
      --model MODEL                 Model to use for embeddings (default: llama3.1:latest)
      --output PATH                 Output file for embeddings (default: [source].json)
      --chunk-size SIZE             Size of text chunks in tokens (default: 512)
      --chunk-overlap OVERLAP       Overlap between consecutive chunks (default: 50)
      --batch-size SIZE             Number of chunks to process at once (default: 10)

    Examples:
      oarc-rag embed ./documents/report.pdf
      oarc-rag embed ./article.txt --model="llama3:latest" --output=vectors.json
    """,
    
    "stats": """
    Display statistics about the RAG system

    Usage:
      oarc-rag stats [options]

    Options:
      --run-id ID                   Run ID to show stats for (default: most recent)
      --detailed                    Show detailed statistics
      --format {text,json}          Output format (default: text)

    Examples:
      oarc-rag stats
      oarc-rag stats --run-id="rag_1234567.001" --detailed --format=json
    """,
    
    "clean": """
    Clean temporary files and caches

    Usage:
      oarc-rag clean [options]

    Options:
      --all                         Remove all generated files
      --cache                       Only clear cache files
      --vector-db                   Only clear vector databases
      --run-id ID                   Only clean specific run ID
      --older-than DAYS             Only clean files older than specified days

    Examples:
      oarc-rag clean --cache
      oarc-rag clean --vector-db --older-than=30
    """,
    
    "help": """
    Show help for any command

    Usage:
      oarc-rag help [command]

    Examples:
      oarc-rag help
      oarc-rag help query
    """
}

def display_help(command: Optional[str] = None, exit_after: bool = True) -> None:
    """
    Display help text for a command or general help if no command specified.
    
    Args:
        command: Specific command to show help for
        exit_after: Whether to exit the program after displaying help
    """
    help_text = ""
    
    if command is None or command not in COMMAND_HELP:
        # Show general help
        help_text = GENERAL_HELP
        if command is not None:
            help_text = f"Unknown command: '{command}'\n\n" + help_text
    else:
        # Show command-specific help
        help_text = COMMAND_HELP[command]
    
    # Remove leading indentation and print
    help_text = textwrap.dedent(help_text).strip()
    print(help_text)
    
    if exit_after:
        sys.exit(0)

def get_command_info(command: str) -> Dict[str, Any]:
    """
    Get structured information about a command for programmatic use.
    
    Args:
        command: The command to get information about
        
    Returns:
        Dictionary containing command information
        
    Raises:
        ValueError: If command doesn't exist
    """
    if command not in COMMAND_HELP:
        raise ValueError(f"Unknown command: '{command}'")
        
    # This is a simple implementation that could be expanded
    # to parse the help text into structured data
    return {
        "name": command,
        "help_text": COMMAND_HELP[command],
        "has_subcommands": command == "vector",
        "requires_args": command in ["add", "query", "embed"]
    }

def list_commands() -> List[str]:
    """
    Get a list of available commands.
    
    Returns:
        List of command names
    """
    return list(COMMAND_HELP.keys())

def print_version_info() -> None:
    """Print version information about oarc-rag."""
    print(f"OARC-RAG v{get_version()}")
    print(f"Using embedding model: {Config.get_embedding_model()}")
    print(f"Vector database: {Config.get_vector_db_config()['type']}")
    print(f"Default output directory: {Config.get_output_dir()}")

def get_version() -> str:
    """Get the current version of oarc-rag."""
    # This could be imported from a central version file
    return "0.1.0"
