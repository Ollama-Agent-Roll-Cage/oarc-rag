"""
Base agent class for all AI agents in the RAG framework.

This module provides the abstract base class that all specialized agents
should extend, implementing concepts from the recursive self-improving
RAG framework including operational modes and performance tracking.
"""
import abc
import uuid
import time

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from oarc_rag.utils.log import (
    log, INFO, with_context, log_at_level
)


class OperationalMode(Enum):
    """Operational modes for agents based on recursive improvement cycles."""
    AWAKE = "awake"  # Real-time interaction mode with fast responses
    SLEEP = "sleep"  # Deep learning/processing mode for knowledge consolidation


class CognitivePhase(Enum):
    """Cognitive phases within operational modes."""
    REACTIVE = "reactive"       # Quick responses to stimuli
    REFLECTIVE = "reflective"   # Deeper consideration of inputs
    CONSOLIDATION = "consolidation"  # Knowledge organization
    PRUNING = "pruning"         # Removing less useful information
    EXPANSION = "expansion"     # Expanding on useful information


class Agent(abc.ABC):
    """
    Abstract base class for all AI agents in the RAG framework.
    
    This class defines the common interface and functionality that
    all agent implementations should provide, based on concepts from
    the recursive self-improving RAG framework.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.1:latest",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        version: str = "1.0",
        operational_mode: Union[str, OperationalMode] = OperationalMode.AWAKE,
        auto_cycle: bool = True,
        cycle_interval: int = 3600  # Default to 1 hour cycles
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name/identifier
            model: LLM model to use for generation
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            version: Agent version string
            operational_mode: Operational mode (awake or sleep)
            auto_cycle: Whether to automatically cycle between modes
            cycle_interval: Seconds between operational mode transitions when auto_cycle=True
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.version = version
        
        # Set operational mode
        if isinstance(operational_mode, str):
            try:
                self.operational_mode = OperationalMode(operational_mode.lower())
            except ValueError:
                log.warning(f"Invalid operational mode: {operational_mode}. Defaulting to AWAKE.")
                self.operational_mode = OperationalMode.AWAKE
        else:
            self.operational_mode = operational_mode
        
        # Session identifier
        self.run_id = str(uuid.uuid4())[:8]  # Shorter ID for logging
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # Storage for results and metadata
        self.results = {}
        self.metadata = {}
        self.history = []
        
        # Performance tracking
        self.execution_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_generation_time": 0.0,
            "avg_response_time": 0.0,
            "last_response_time": 0.0,
            "response_times": []  # Keep last N response times for analysis
        }
        
        # Agent state tracking 
        self.state = {
            "status": "initialized",
            "last_active": datetime.now().isoformat(),
            "current_task": None,
            "operational_mode": self.operational_mode.value
        }
        
        # Enhanced cognitive cycle tracking (from Big_Brain.md concepts)
        self.cognitive_cycle = {
            "cycle_count": 0,
            "last_sleep_phase": None, 
            "last_awake_phase": datetime.now().isoformat(),
            "knowledge_consolidation_status": "initial",
            "current_phase": CognitivePhase.REACTIVE.value,
            "cycle_duration": cycle_interval,
            "auto_cycle": auto_cycle,
            "next_transition": (datetime.now() + timedelta(seconds=cycle_interval)).isoformat(),
            "metrics": {
                "consolidation_count": 0,
                "pruning_operations": 0,
                "expansion_operations": 0,
                "knowledge_growth_rate": 0.0
            },
            "stability_index": 1.0  # 0-1 scale of knowledge stability
        }
        
        # Knowledge preservation parameters (from Big_Brain.md EWC concept)
        self.knowledge_params = {
            "importance_matrix": {},  # Maps parameter IDs to importance values
            "preserved_checkpoints": [],  # List of checkpoint IDs
            "checkpoint_intervals": 5,  # Save checkpoint every N cycles
            "parameter_stability": 0.85  # Stability factor for EWC
        }
        
        # Initialize system integration components based on operational mode
        self._initialize_mode_components()
        
        log.info(f"Agent {name} initialized with mode: {self.operational_mode.value}")
        
    def _initialize_mode_components(self) -> None:
        """Initialize components based on operational mode."""
        # Different initialization logic based on operational mode
        if self.operational_mode == OperationalMode.SLEEP:
            # Sleep mode prioritizes deep processing over latency
            self._configure_for_sleep_mode()
        else:
            # Awake mode prioritizes responsiveness
            self._configure_for_awake_mode()
    
    def _configure_for_sleep_mode(self) -> None:
        """Configure agent for sleep (deep processing) mode."""
        # Extended processing settings
        self.cognitive_cycle["last_sleep_phase"] = datetime.now().isoformat()
        self.log_activity("Entered sleep mode for deep processing")
    
    def _configure_for_awake_mode(self) -> None:
        """Configure agent for awake (responsive) mode."""
        # Real-time settings
        self.cognitive_cycle["last_awake_phase"] = datetime.now().isoformat()
        self.log_activity("Entered awake mode for responsive interaction")
        
    def set_operational_mode(self, mode: Union[str, OperationalMode]) -> None:
        """
        Switch the agent's operational mode and reconfigure components.
        
        Args:
            mode: New operational mode (awake or sleep)
            
        Raises:
            ValueError: If mode string is invalid
        """
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = OperationalMode(mode.lower())
            except ValueError:
                raise ValueError(f"Invalid operational mode: {mode}. Use 'awake' or 'sleep'.")
        
        # Skip if already in this mode
        if self.operational_mode == mode:
            self.log_activity(f"Already in {mode.value} mode")
            return
            
        # Update mode
        previous_mode = self.operational_mode
        self.operational_mode = mode
        self.state["operational_mode"] = mode.value
        
        # Reconfigure components
        self._initialize_mode_components()
        
        # Enhanced cognitive cycle tracking
        now = datetime.now()
        self.cognitive_cycle["cycle_count"] += 1
        
        # Calculate next transition time if auto-cycle is enabled
        if self.cognitive_cycle["auto_cycle"]:
            next_transition = now + timedelta(seconds=self.cognitive_cycle["cycle_duration"])
            self.cognitive_cycle["next_transition"] = next_transition.isoformat()
        
        # Implement Elastic Weight Consolidation concepts from Big_Brain.md
        if mode == OperationalMode.SLEEP and self.cognitive_cycle["cycle_count"] % self.knowledge_params["checkpoint_intervals"] == 0:
            self._create_knowledge_checkpoint()
            
        self.log_activity(
            f"Changed operational mode: {previous_mode.value} → {mode.value} "
            f"(cycle #{self.cognitive_cycle['cycle_count']})"
        )
    
    def set_cognitive_phase(self, phase: Union[str, CognitivePhase]) -> None:
        """
        Set the agent's cognitive phase within its operational mode.
        
        Args:
            phase: Cognitive phase (reactive, reflective, etc.)
            
        Raises:
            ValueError: If phase is invalid
        """
        # Convert string phase to enum if needed
        if isinstance(phase, str):
            try:
                phase = CognitivePhase(phase.lower())
            except ValueError:
                valid_phases = [p.value for p in CognitivePhase]
                raise ValueError(f"Invalid cognitive phase: {phase}. Valid phases: {valid_phases}")
        
        # Update cognitive phase
        old_phase = self.cognitive_cycle["current_phase"]
        self.cognitive_cycle["current_phase"] = phase.value
        
        # Log phase transition
        self.log_activity(f"Cognitive phase transition: {old_phase} → {phase.value}")
        
        # Execute phase-specific initialization
        if phase == CognitivePhase.CONSOLIDATION:
            self.consolidate_knowledge()
        elif phase == CognitivePhase.PRUNING:
            self._prune_knowledge()
        elif phase == CognitivePhase.EXPANSION:
            self._expand_knowledge()
        
    def set_model(self, model: str) -> None:
        """
        Set the model for this agent.
        
        Args:
            model: Model name to use
        """
        self.model = model
        self.log_activity(f"Model changed to: {model}")
        
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature for generation.
        
        Args:
            temperature: Temperature value (0.0-1.0)
        """
        self.temperature = max(0.0, min(1.0, temperature))
        self.log_activity(f"Temperature set to: {self.temperature}")
        
    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Set maximum tokens for generation.
        
        Args:
            max_tokens: Maximum number of tokens
        """
        self.max_tokens = max_tokens
        self.log_activity(f"Max tokens set to: {max_tokens}")
        
    def store_result(self, key: str, value: Any) -> None:
        """
        Store a result from the agent's operations.
        
        Args:
            key: Result key/identifier
            value: Result value
        """
        self.results[key] = value
        self._add_to_history("result_stored", {"key": key, "value_type": type(value).__name__})
        
    def get_result(self, key: str, default: Any = None) -> Any:
        """
        Get a stored result.
        
        Args:
            key: Result key/identifier
            default: Default value if key doesn't exist
            
        Returns:
            The stored result or default value
        """
        result = self.results.get(key, default)
        self._add_to_history("result_retrieved", {"key": key, "found": key in self.results})
        return result
        
    def store_metadata(self, key: str, value: Any) -> None:
        """
        Store metadata about agent operations.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get stored metadata.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            The metadata value or default
        """
        return self.metadata.get(key, default)
    
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: Prompt to generate from
            
        Returns:
            Generated content
        """
        pass
    
    @abc.abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and produce output.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processed output
        """
        pass
        
    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate knowledge during sleep phase for self-improvement.
        
        This method implements the knowledge consolidation concept from Big_Brain.md,
        analyzing agent interactions and improving internal representations.
        
        Returns:
            Dict containing consolidation metrics and results
        """
        if self.operational_mode != OperationalMode.SLEEP:
            self.log_activity(
                "Knowledge consolidation optimally runs in SLEEP mode. "
                "Consider switching modes for better results.", 
                level="warning"
            )
            
        start_time = time.time()
        self.cognitive_cycle["knowledge_consolidation_status"] = "in_progress"
        self.log_activity("Beginning knowledge consolidation process")
        
        # Analyze historical interactions to identify patterns
        history = self.get_history(limit=50)
        
        # Extract key insights (implementation would analyze actual interaction patterns)
        recurring_topics = self._identify_recurring_topics(history)
        knowledge_gaps = self._identify_knowledge_gaps(history)
        
        # Update knowledge metrics
        self.cognitive_cycle["metrics"]["consolidation_count"] += 1
        self.cognitive_cycle["metrics"]["knowledge_growth_rate"] = self._calculate_knowledge_growth()
        self.cognitive_cycle["stability_index"] = min(1.0, self.cognitive_cycle["stability_index"] + 0.05)
        
        # Update status
        elapsed_time = time.time() - start_time
        results = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "processing_time": elapsed_time,
            "improvements": {
                "recurring_topics": recurring_topics,
                "knowledge_gaps": knowledge_gaps,
                "stability_index": self.cognitive_cycle["stability_index"]
            },
            "cycle_number": self.cognitive_cycle["cycle_count"]
        }
        
        self.cognitive_cycle["knowledge_consolidation_status"] = "completed"
        self._add_to_history("knowledge_consolidated", results)
        self.log_activity(f"Knowledge consolidation completed in {elapsed_time:.2f}s")
        
        return results
    
    def _identify_recurring_topics(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify recurring topics from interaction history."""
        # This would be implemented with actual NLP analysis
        return ["topic1", "topic2"]
    
    def _identify_knowledge_gaps(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify knowledge gaps from interaction history."""
        # This would be implemented with actual NLP analysis
        return ["gap1", "gap2"]
    
    def _calculate_knowledge_growth(self) -> float:
        """Calculate knowledge growth rate based on interactions."""
        # This would track actual growth metrics
        return 0.05
    
    def _create_knowledge_checkpoint(self) -> str:
        """
        Create a checkpoint of current knowledge state for EWC.
        
        Returns:
            Checkpoint ID
        """
        # Create unique checkpoint ID
        checkpoint_id = f"{self.run_id}-{self.cognitive_cycle['cycle_count']}"
        
        # In a real implementation, this would store model parameters
        # and their importance for elastic weight consolidation
        
        # Store checkpoint reference
        self.knowledge_params["preserved_checkpoints"].append(checkpoint_id)
        
        # Keep only the most recent checkpoints
        max_checkpoints = 5
        if len(self.knowledge_params["preserved_checkpoints"]) > max_checkpoints:
            self.knowledge_params["preserved_checkpoints"] = self.knowledge_params["preserved_checkpoints"][-max_checkpoints:]
            
        self.log_activity(f"Created knowledge checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def _prune_knowledge(self) -> Dict[str, Any]:
        """
        Prune less relevant knowledge to optimize representations.
        
        Returns:
            Metrics about pruning operation
        """
        # In a real implementation, this would identify and remove
        # redundant or low-value information
        self.cognitive_cycle["metrics"]["pruning_operations"] += 1
        
        metrics = {
            "items_considered": 100,
            "items_pruned": 15,
            "estimated_efficiency_gain": 0.08
        }
        
        self.log_activity(f"Pruned {metrics['items_pruned']} knowledge items")
        return metrics
    
    def _expand_knowledge(self) -> Dict[str, Any]:
        """
        Expand knowledge on high-value topics.
        
        Returns:
            Metrics about expansion operation
        """
        # In a real implementation, this would generate new insights
        # and connections between existing knowledge
        self.cognitive_cycle["metrics"]["expansion_operations"] += 1
        
        metrics = {
            "topics_expanded": 3,
            "new_connections": 12,
            "estimated_knowledge_gain": 0.1
        }
        
        self.log_activity(f"Expanded knowledge on {metrics['topics_expanded']} topics")
        return metrics
    
    def check_cycle_transition(self) -> bool:
        """
        Check if it's time to transition between operational modes.
        
        Returns:
            True if mode should transition, False otherwise
        """
        if not self.cognitive_cycle["auto_cycle"]:
            return False
            
        next_transition = datetime.fromisoformat(self.cognitive_cycle["next_transition"])
        if datetime.now() >= next_transition:
            # Toggle between modes
            new_mode = OperationalMode.SLEEP if self.operational_mode == OperationalMode.AWAKE else OperationalMode.AWAKE
            self.set_operational_mode(new_mode)
            return True
            
        return False

    def get_cognitive_status(self) -> Dict[str, Any]:
        """
        Get detailed status of the agent's cognitive processes.
        
        Returns:
            Dict with cognitive status information
        """
        return {
            "operational_mode": self.operational_mode.value,
            "cognitive_phase": self.cognitive_cycle["current_phase"],
            "cycle_count": self.cognitive_cycle["cycle_count"],
            "next_transition": self.cognitive_cycle["next_transition"],
            "auto_cycle": self.cognitive_cycle["auto_cycle"],
            "stability_index": self.cognitive_cycle["stability_index"],
            "metrics": self.cognitive_cycle["metrics"],
            "checkpoints": len(self.knowledge_params["preserved_checkpoints"])
        }
        
    def log_activity(self, activity: str, level: str = "info") -> None:
        """
        Log agent activity with appropriate context.
        
        Args:
            activity: Activity description
            level: Log level (debug, info, warning, error)
        """
        with with_context(agent=self.name, session=self.session_id[:8], run=self.run_id):
            log_at_level(getattr(log, level.upper(), INFO), activity)
        
        # Update activity timestamp and history
        self.state["last_active"] = datetime.now().isoformat()
        self._add_to_history("activity_logged", {"message": activity, "level": level})
    
    def _add_to_history(self, action_type: str, details: Dict[str, Any]) -> None:
        """
        Add an entry to the agent's history.
        
        Args:
            action_type: Type of action
            details: Action details
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "details": details,
            "operational_mode": self.operational_mode.value
        }
        self.history.append(entry)
        
        # Limit history size to prevent unlimited growth
        max_history = 1000
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        
    def update_stats(self, tokens_used: int, generation_time: float) -> None:
        """
        Update agent performance statistics.
        
        Args:
            tokens_used: Number of tokens used
            generation_time: Time taken for generation in seconds
        """
        self.execution_stats["total_calls"] += 1
        self.execution_stats["total_tokens"] += tokens_used
        self.execution_stats["total_generation_time"] += generation_time
        self.execution_stats["avg_response_time"] = (
            self.execution_stats["total_generation_time"] / 
            self.execution_stats["total_calls"]
        )
        self.execution_stats["last_response_time"] = generation_time
        
        # Store recent response times (keep last 100)
        self.execution_stats["response_times"].append(generation_time)
        if len(self.execution_stats["response_times"]) > 100:
            self.execution_stats["response_times"].pop(0)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for this agent.
        
        Returns:
            Dict containing various performance metrics
        """
        metrics = self.execution_stats.copy()
        
        # Add calculated metrics if we have enough data
        response_times = self.execution_stats["response_times"]
        if response_times:
            metrics.update({
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": sorted(response_times)[len(response_times)//2],
                "recent_avg_response_time": sum(response_times[-10:]) / min(10, len(response_times))
            })
            
        # Add token efficiency metrics
        if self.execution_stats["total_tokens"] > 0:
            metrics["tokens_per_second"] = self.execution_stats["total_tokens"] / max(0.001, self.execution_stats["total_generation_time"])
            
        return metrics
        
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive status report for this agent.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "agent": {
                "name": self.name,
                "model": self.model,
                "version": self.version,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat()
            },
            "state": self.state,
            "cognitive_cycle": self.cognitive_cycle,
            "performance": self.get_performance_metrics(),
            "storage": {
                "results_count": len(self.results),
                "metadata_count": len(self.metadata),
                "history_entries": len(self.history)
            }
        }
    
    def set_state(self, status: str, task: Optional[str] = None) -> None:
        """
        Update the agent's state.
        
        Args:
            status: New status string
            task: Current task description (optional)
        """
        previous_status = self.state["status"]
        self.state["status"] = status
        self.state["last_active"] = datetime.now().isoformat()
        
        if task is not None:
            self.state["current_task"] = task
            
        self._add_to_history("state_changed", {
            "previous": previous_status, 
            "current": status,
            "task": task
        })
        
    def get_history(self, limit: int = 10, action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get agent history entries.
        
        Args:
            limit: Maximum number of entries to return
            action_type: Optional filter by action type
            
        Returns:
            List[Dict[str, Any]]: History entries
        """
        if action_type:
            filtered = [entry for entry in self.history if entry["action"] == action_type]
            return filtered[-limit:] if limit > 0 else filtered
        else:
            return self.history[-limit:] if limit > 0 else self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the agent's history to free memory."""
        history_size = len(self.history)
        self.history = []
        self.log_activity(f"Cleared history ({history_size} entries)")
        
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        old_stats = self.execution_stats.copy()
        self.execution_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_generation_time": 0.0,
            "avg_response_time": 0.0,
            "last_response_time": 0.0,
            "response_times": []
        }
        self._add_to_history("stats_reset", {"previous_stats": old_stats})
        self.log_activity("Performance statistics reset")
