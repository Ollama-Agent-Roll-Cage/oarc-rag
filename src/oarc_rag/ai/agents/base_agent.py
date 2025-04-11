"""
Base agent class for specialized RAG agents.

This module provides a specialized base agent class for RAG operations,
extending the core Agent class with RAG-specific functionality.
"""
import time
from typing import Any, Dict, Optional, Union

from oarc_rag.ai.agent import Agent, OperationalMode
from oarc_rag.ai.client import OllamaClient
from oarc_rag.ai.prompts import get_prompt_template_manager, render_template
from oarc_rag.utils.log import log


class RAGAgent(Agent):
    """
    Base agent class for specialized RAG operations.
    
    This class extends the core Agent class with functionality for
    retrieving and processing information using the RAG framework.
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
        cycle_interval: int = 3600,
        prompt_templates: Optional[Dict[str, str]] = None,
        default_template: Optional[str] = None
    ):
        """
        Initialize the RAG agent with additional RAG-specific parameters.
        
        Args:
            name: Agent name/identifier
            model: LLM model to use for generation
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            version: Agent version string
            operational_mode: Operational mode (awake or sleep)
            auto_cycle: Whether to automatically cycle between modes
            cycle_interval: Seconds between operational mode transitions
            prompt_templates: Dictionary of template names to use (key: purpose, value: template name)
            default_template: Name of the default template to use when no specific template is specified
        """
        super().__init__(
            name=name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            version=version,
            operational_mode=operational_mode,
            auto_cycle=auto_cycle,
            cycle_interval=cycle_interval
        )
        
        # Initialize RAG-specific components
        self.client = OllamaClient(default_model=model)
        self.prompt_template_manager = get_prompt_template_manager()
        
        # Multiple template support
        self.prompt_templates = prompt_templates or {}
        self.default_template = default_template
        
        # Track performance metrics specific to RAG operations
        self.metrics = {
            "processed_items": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_operation_time": 0.0,
            "total_operation_time": 0.0,
            "avg_tokens_per_operation": 0.0,
            "total_tokens": 0
        }
        
        log.info(f"RAG Agent {name} initialized with {len(self.prompt_templates)} templates")
    
    def generate(self, prompt: str) -> str:
        """
        Generate content using the configured client and model.
        
        Args:
            prompt: Prompt to generate from
            
        Returns:
            Generated content
        """
        start_time = time.time()
        
        try:
            # Use the Ollama client to generate
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Track metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Estimate tokens (this is approximate)
            tokens_used = len(prompt.split()) + len(response.split())
            
            # Update stats
            self.update_stats(tokens_used, duration)
            
            return response
            
        except Exception as e:
            self.log_activity(f"Generation failed: {e}", level="error")
            raise
    
    def add_prompt_template(self, purpose: str, template_name: str) -> None:
        """
        Add or update a prompt template for a specific purpose.
        
        Args:
            purpose: Purpose identifier (e.g., 'query', 'context', 'response')
            template_name: Name of the template to use for this purpose
        """
        self.prompt_templates[purpose] = template_name
        if not self.default_template:
            self.default_template = template_name
        self.log_activity(f"Added template '{template_name}' for purpose '{purpose}'")
    
    def get_prompt_template(self, purpose: Optional[str] = None) -> Optional[str]:
        """
        Get the template name for the specified purpose.
        
        Args:
            purpose: Purpose identifier (if None, returns default template)
            
        Returns:
            Template name or None if not found
        """
        if purpose:
            return self.prompt_templates.get(purpose)
        return self.default_template
    
    def render_prompt_template(
        self, 
        template_name: Optional[str] = None, 
        purpose: Optional[str] = None, 
        **kwargs
    ) -> str:
        """
        Render a prompt template with the given parameters.
        
        Args:
            template_name: Name of the template to render (direct specification)
            purpose: Purpose identifier to select template (used if template_name is None)
            **kwargs: Parameters to pass to the template
            
        Returns:
            Rendered prompt
            
        Raises:
            ValueError: If no template can be determined
        """
        # Determine which template to use
        template_to_use = None
        
        if template_name:
            # Direct template specification takes precedence
            template_to_use = template_name
        elif purpose and purpose in self.prompt_templates:
            # If purpose is specified and exists in our templates
            template_to_use = self.prompt_templates[purpose]
        elif self.default_template:
            # Fall back to default template
            template_to_use = self.default_template
        
        if not template_to_use:
            raise ValueError("No template specified for rendering and no default template available")
        
        try:
            return render_template(template_to_use, **kwargs)
        except Exception as e:
            self.log_activity(f"Failed to render template '{template_to_use}': {e}", level="error")
            raise
    
    def process(self, input_data: Any) -> Any:
        """
        Abstract method to process input data.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processed output
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def update_performance_metrics(self, processed_items: int = 1, success: bool = True, 
                                operation_time: float = 0.0, tokens: int = 0) -> None:
        """
        Update RAG-specific performance metrics.
        
        Args:
            processed_items: Number of items processed
            success: Whether the operation was successful
            operation_time: Time taken for the operation
            tokens: Number of tokens used
        """
        self.metrics["processed_items"] += processed_items
        
        if success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1
            
        self.metrics["total_operation_time"] += operation_time
        self.metrics["total_tokens"] += tokens
        
        # Update averages
        total_ops = self.metrics["successful_operations"] + self.metrics["failed_operations"]
        if total_ops > 0:
            self.metrics["avg_operation_time"] = self.metrics["total_operation_time"] / total_ops
            self.metrics["avg_tokens_per_operation"] = self.metrics["total_tokens"] / total_ops
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()
