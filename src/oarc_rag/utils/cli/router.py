"""
Command router for CLI execution.

This module handles the routing of CLI commands to appropriate handlers.
"""
import sys
from typing import List, Optional, Dict, Any

from oarc_rag.utils.log import log
from oarc_rag.utils.cli.parser import parse_args
from oarc_rag.utils.cli.cmd_types import CommandType
from oarc_rag.utils.cli.commands.help_cmd import HelpCommand
from oarc_rag.utils.cli.commands.clean_cmd import CleanCommand
from oarc_rag.utils.cli.commands.create_cmd import CreateCommand
from oarc_rag.utils.cli.commands.rag_query_cmd import RAGQueryCommand
from oarc_rag.utils.cli.commands.ingest_cmd import IngestCommand


def route_command(args: Optional[List[str]] = None) -> int:
    """
    Route the command to the appropriate handler.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code
    """
    # Parse arguments
    parsed_args = parse_args(args)
    
    # Dictionary of command types to handler classes
    command_handlers: Dict[CommandType, Any] = {
        CommandType.HELP: HelpCommand,
        CommandType.CLEAN: CleanCommand,
        CommandType.CREATE: CreateCommand,
        CommandType.RAG_QUERY: RAGQueryCommand,
        CommandType.INGEST: IngestCommand,
    }
    
    # Get the command handler
    handler_class = command_handlers.get(parsed_args.command_type)
    
    if not handler_class:
        log.error(f"No handler found for command type: {parsed_args.command_type}")
        return 1
        
    # Create and execute the command
    try:
        command = handler_class(parsed_args)
        return command.execute()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130  # Unix standard for SIGINT
    except Exception as e:
        log.error(f"Command execution failed: {e}")
        print(f"Error: {e}")
        return 1


# Legacy name for backwards compatibility
def handle(args: Optional[List[str]] = None) -> int:
    """
    Legacy function name for route_command.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    return route_command(args)
