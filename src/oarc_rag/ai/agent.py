"""
Base Agent class for all AI agents.

This module provides the foundational Agent class that defines common
functionality and interfaces for all agent types in the system.
"""
import enum
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable

from oarc_rag.utils.log import log


class OperationalMode(enum.Enum):
    """Enum for agent operational modes."""
    AWAKE = "awake"   # Real-time processing mode
    SLEEP = "sleep"   # Batch/background processing mode


class CognitivePhase(enum.Enum):
    """Enum for agent cognitive phases."""
    REACTIVE = "reactive"         # Responding to immediate stimuli
    REFLECTIVE = "reflective"     # Considering implications and patterns
    CONSOLIDATION = "consolidation"  # Organizing and synthesizing knowledge
    CREATIVE = "creative"         # Generating novel insights


class Agent:
    """
    Base class for all agent implementations.
    
    This class provides core functionality for state management,
    operational modes, statistics tracking, and activity logging.
    """
    
    def __init__(
        self, 
        name: str,
        model: str = "llama3.1:latest",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        version: str = "1.0",
        operational_mode: Union[str, OperationalMode] = OperationalMode.AWAKE,
        auto_cycle: bool = True,
        cycle_interval: int = 3600
    ):
        """
        Initialize an agent with core parameters.
        
        Args:
            name: Agent name/identifier
            model: Default LLM model to use
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            version: Agent version string
            operational_mode: Operational mode (awake or sleep)
            auto_cycle: Whether to automatically cycle between modes
            cycle_interval: Seconds between operational mode transitions
        """
        # Core identity properties
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.version = version
        
        # Initialize operational state
        if isinstance(operational_mode, str):
            self.operational_mode = OperationalMode(operational_mode.lower())
        else:
            self.operational_mode = operational_mode
            
        self.auto_cycle = auto_cycle
        self.cycle_interval = cycle_interval
        self.last_mode_switch = datetime.now()
        self.cycle_count = 0
        
        # Initialize cognitive phase
        self.cognitive_phase = CognitivePhase.REACTIVE
        
        # Activity tracking
        self.activity_log = []
        self.max_log_entries = 100
        
        # Memory store
        self.memory = {}
        
        # Performance stats
        self.stats = {
            "requests_processed": 0,
            "tokens_generated": 0,
            "processing_time": 0.0,
            "start_time": time.time()
        }
        
        log.info(f"Agent {name} initialized with model {model}")
    
    def generate(self, prompt: str) -> str:
        """
        Abstract method for text generation.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def update_stats(self, tokens_used: int, duration: float) -> None:
        """
        Update agent performance statistics.
        
        Args:
            tokens_used: Number of tokens used
            duration: Operation duration in seconds
        """
        self.stats["requests_processed"] += 1
        self.stats["tokens_generated"] += tokens_used
        self.stats["processing_time"] += duration
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        uptime = time.time() - self.stats["start_time"]
        requests = max(1, self.stats["requests_processed"])
        
        return {
            "uptime_seconds": uptime,
            "requests_processed": self.stats["requests_processed"],
            "tokens_generated": self.stats["tokens_generated"],
            "avg_tokens_per_request": self.stats["tokens_generated"] / requests,
            "avg_processing_time": self.stats["processing_time"] / requests,
            "operational_mode": self.operational_mode.value,
            "cognitive_phase": self.cognitive_phase.value,
            "cycle_count": self.cycle_count
        }
    
    def switch_operational_mode(self, mode: Union[str, OperationalMode] = None) -> None:
        """
        Switch the agent's operational mode.
        
        Args:
            mode: New operational mode, if None toggles between modes
        """
        # If no mode specified, toggle
        if mode is None:
            if self.operational_mode == OperationalMode.AWAKE:
                new_mode = OperationalMode.SLEEP
            else:
                new_mode = OperationalMode.AWAKE
        else:
            # Convert string to enum if needed
            if isinstance(mode, str):
                new_mode = OperationalMode(mode.lower())
            else:
                new_mode = mode
        
        # If actually changing modes
        if new_mode != self.operational_mode:
            old_mode = self.operational_mode
            self.operational_mode = new_mode
            self.last_mode_switch = datetime.now()
            
            # Increment cycle count if coming back to AWAKE
            if new_mode == OperationalMode.AWAKE and old_mode == OperationalMode.SLEEP:
                self.cycle_count += 1
            
            self.log_activity(
                f"Switched operational mode from {old_mode.value} to {new_mode.value}"
            )
    
    def set_cognitive_phase(self, phase: Union[str, CognitivePhase]) -> None:
        """
        Set the agent's cognitive phase.
        
        Args:
            phase: New cognitive phase
        """
        if isinstance(phase, str):
            self.cognitive_phase = CognitivePhase(phase.lower())
        else:
            self.cognitive_phase = phase
            
        self.log_activity(f"Cognitive phase set to {self.cognitive_phase.value}")
    
    def should_switch_modes(self) -> bool:
        """
        Determine if the agent should switch operational modes based on interval.
        
        Returns:
            bool: True if mode should be switched
        """
        if not self.auto_cycle:
            return False
            
        # Check if we've been in this mode long enough
        time_in_mode = self._time_in_current_mode().total_seconds()
        return time_in_mode >= self.cycle_interval
    
    def check_cycle_transition(self) -> bool:
        """
        Check if a cycle transition is needed and perform it if so.
        
        Returns:
            bool: True if a transition occurred
        """
        if self.should_switch_modes():
            self.switch_operational_mode()
            return True
        return False
    
    def _time_in_current_mode(self) -> timedelta:
        """
        Calculate time spent in current operational mode.
        
        Returns:
            Time as timedelta
        """
        return datetime.now() - self.last_mode_switch
    
    def log_activity(self, message: str, level: str = "info") -> None:
        """
        Log agent activity with timestamp and add to activity log.
        
        Args:
            message: Activity message
            level: Log level (info, warning, error)
        """
        # Create log entry
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        
        # Add to activity log
        self.activity_log.append(entry)
        
        # Trim log if needed
        if len(self.activity_log) > self.max_log_entries:
            self.activity_log = self.activity_log[-self.max_log_entries:]
        
        # Also log to system logger
        log_method = getattr(log, level, log.info)
        log_method(f"[{self.name}] {message}")
    
    def get_activity_log(self) -> List[Dict[str, str]]:
        """
        Get the agent's activity log.
        
        Returns:
            List of activity log entries
        """
        return self.activity_log
    
    def store_result(self, key: str, result: Any) -> None:
        """
        Store a result in the agent's memory.
        
        Args:
            key: Key for storing the result
            result: Result to store
        """
        self.memory[key] = {
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_result(self, key: str) -> Optional[Any]:
        """
        Retrieve a result from agent memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Stored result or None if not found
        """
        if key in self.memory:
            return self.memory[key]["data"]
        return None
    
    def store_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Store metadata in agent memory.
        
        Args:
            key: Key for metadata
            metadata: Metadata dictionary
        """
        if key not in self.memory:
            self.memory[key] = {"data": {}, "timestamp": datetime.now().isoformat()}
            
        self.memory[key]["data"].update(metadata)
    
    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate agent knowledge during sleep phase.
        
        This method should be called during sleep phase to process
        accumulated knowledge and improve future responses.
        
        Returns:
            Dict containing consolidation metrics
        """
        self.log_activity("Consolidating knowledge", level="info")
        
        # Basic implementation, subclasses should override for specific behavior
        return {
            "cycle": self.cycle_count,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "memory_size": len(self.memory),
                "activity_count": len(self.activity_log)
            }
        }
