"""
Utility functions and helpers.
"""
from oarc_rag.utils.log import log, is_debug_mode
from oarc_rag.utils.utils import Utils
from oarc_rag.utils.deps import DependencyManager

# Export DependencyManager methods as module-level functions
check_cuda_capability = DependencyManager.check_cuda_capability
install_cuda_toolkit = DependencyManager.install_cuda_toolkit
upgrade_faiss = DependencyManager.upgrade_faiss
check_deps = DependencyManager.check_deps

__all__ = [
    "log",
    "is_debug_mode",
    "Utils",
    "check_cuda_capability",
    "install_cuda_toolkit",
    "upgrade_faiss",
    "check_deps",
    "DependencyManager",
]
