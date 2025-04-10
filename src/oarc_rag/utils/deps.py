"""
Dependency management utilities for oarc_rag.
"""
import sys
import subprocess
from typing import Tuple, Dict, Optional

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton

@singleton
class DependencyManager:
    """Manages project dependencies and optimizations."""
    
    @staticmethod
    def check_cuda_capability() -> Tuple[bool, Optional[str]]:
        """Check if system has CUDA capability."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                return True, f"CUDA {cuda_version} capable device found: {device_name}"
        except ImportError:
            pass

        # Fallback to nvidia-smi check
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, f"NVIDIA GPU detected: {result.stdout.strip()}"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
        return False, None

    @staticmethod
    def install_cuda_toolkit() -> Tuple[bool, str]:
        """Install CUDA toolkit if system is capable."""
        cuda_capable, info = DependencyManager.check_cuda_capability()
        if not cuda_capable:
            return False, "No CUDA capable device detected"

        try:
            # Try to install CUDA toolkit via pip
            log.info("Installing CUDA toolkit dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "--upgrade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Verify installation
            import torch
            if torch.cuda.is_available():
                return True, f"CUDA toolkit installed successfully. CUDA version: {torch.version.cuda}"
            else:
                return False, "CUDA installation failed verification"
        except Exception as e:
            return False, f"Failed to install CUDA toolkit: {e}"

    @staticmethod
    def check_deps() -> Dict[str, Dict[str, any]]:
        """Check and optimize all dependencies."""
        results = {
            "cuda": {"status": False, "message": "", "action_taken": False}
        }

        # Check CUDA
        cuda_capable, cuda_info = DependencyManager.check_cuda_capability()
        if cuda_capable:
            if not DependencyManager._is_cuda_installed():
                success, message = DependencyManager.install_cuda_toolkit()
                results["cuda"] = {
                    "status": success,
                    "message": message,
                    "action_taken": True
                }
            else:
                results["cuda"] = {
                    "status": True,
                    "message": cuda_info,
                    "action_taken": False
                }

        return results

    @staticmethod
    def _is_cuda_installed() -> bool:
        """Check if CUDA toolkit is installed."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
