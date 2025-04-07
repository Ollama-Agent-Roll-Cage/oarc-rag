"""
Command-line interface utilities for oarc_rag.
"""

from oarc_rag.utils.cli.cmd import Command
from oarc_rag.utils.cli.cmd_desc import COMMAND_DESC
from oarc_rag.utils.cli.cmd_types import CommandType
from oarc_rag.utils.cli.commands.create_cmd import CreateCommand
from oarc_rag.utils.cli.help import (
    COMMAND_HELP,
    GENERAL_HELP,
    display_help,
    get_command_info,
    get_version,
    list_commands,
    print_version_info,
)
from oarc_rag.utils.cli.router import handle, route_command

__all__ = [
    "handle",
    "route_command",
    "CommandType",
    "Command",
    "COMMAND_DESC",
    "CreateCommand",
    "display_help",
    "get_command_info",
    "print_version_info",
    "get_version",
    "list_commands",
    "GENERAL_HELP",
    "COMMAND_HELP",
]
