"""
Generation command for creating RAG's.
"""
import argparse

from oarc_rag.utils.cli.cmd import Command
from oarc_rag.utils.cli.parser import setup_create_arguments
from oarc_rag.core.rag import RAG


class CreateCommand(Command):
    """Command for generating and exporting RAG."""
    
    @classmethod
    def register(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments to the parser."""
        setup_create_arguments(parser)
    
    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the command and return the exit code.
        
        Args:
            args: Command-line arguments
            
        Returns:
            int: Exit code
        """
        # Use the factory method from RAG to handle creation and export
        result = RAG.create(args=args)
        
        # For CLI usage, we expect a tuple of (exit_code, path)
        if isinstance(result, tuple) and len(result) == 2:
            exit_code, _ = result
            return exit_code
        
        # If we got a RAG object (API usage), return success
        return 0
