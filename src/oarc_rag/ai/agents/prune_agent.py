"""
Agent for pruning content while preserving critical information.

This agent intelligently prunes content to reduce size while
maintaining critical knowledge using EWC-based importance assessment.
"""
import time
import json
import re
from typing import Dict, Any

from oarc_rag.ai.agents.base_agent import RAGAgent


class PruneAgent(RAGAgent):
    """
    Agent for pruning content while preserving critical information.
    
    This agent uses LLM-based intelligence with EWC concepts to prune
    content while preserving important information.
    """
    
    def __init__(self, name: str = "prune_agent", **kwargs):
        """
        Initialize the prune agent.
        
        Args:
            name: Agent name/identifier
            **kwargs: Additional parameters for the base RAGAgent
        """
        # Set default prompt template
        kwargs.setdefault("prompt_template", "prune_agent")
        
        # Initialize with base class
        super().__init__(name=name, **kwargs)
        
        # Specific metrics for pruning operations
        self.prune_metrics = {
            "avg_reduction_percentage": 0.0,
            "total_prunes": 0,
            "avg_information_retention": 0.0,
            "total_text_removed": 0,
            "avg_items_pruned": 0
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prune content while preserving critical information.
        
        Args:
            input_data: Dictionary containing:
                - content: Text content to prune
                - usage_analytics: Usage analytics data (optional)
                - importance_assessment: Knowledge importance assessment (optional)
                - target_reduction_percentage: Target reduction percentage (optional)
                - preservation_threshold: Critical knowledge preservation threshold (optional)
                - min_coherence: Minimum semantic coherence (optional)
                - max_density: Maximum information density (optional)
                
        Returns:
            Dictionary with pruning results
        """
        start_time = time.time()
        
        # Extract parameters
        content = input_data.get("content", "")
        if not content:
            self.log_activity("No content provided for pruning", level="warning")
            return {"error": "No content provided", "success": False}
        
        usage_analytics = input_data.get("usage_analytics", "")
        importance_assessment = input_data.get("importance_assessment", "")
        target_reduction_percentage = input_data.get("target_reduction_percentage", 30)
        preservation_threshold = input_data.get("preservation_threshold", 0.9)
        min_coherence = input_data.get("min_coherence", 0.8)
        max_density = input_data.get("max_density", 0.9)
        
        # Prepare prompt parameters
        prompt_params = {
            "content": content,
            "usage_analytics": usage_analytics,
            "importance_assessment": importance_assessment,
            "target_reduction_percentage": target_reduction_percentage,
            "preservation_threshold": preservation_threshold,
            "min_coherence": min_coherence,
            "max_density": max_density
        }
        
        # Render prompt from template
        prompt = self.render_prompt_template(**prompt_params)
        
        try:
            # Generate pruned content
            response = self.generate(prompt)
            
            # Extract JSON response
            result = self._extract_json_response(response)
            
            # Fallback to original content if pruning fails
            if not result or "pruned_content" not in result:
                self.log_activity("LLM pruning failed, returning original content", level="warning")
                result = {
                    "pruned_content": content,
                    "pruning_decisions": [],
                    "preservation_metrics": {
                        "critical_information_retention": 1.0,
                        "meaning_preservation": 1.0,
                        "readability_impact": "0%",
                        "token_reduction": "0%"
                    },
                    "vector_impact": {
                        "expected_similarity_shift": 0.0,
                        "query_coverage_change": "0%",
                        "recommended_embedding_update": False
                    }
                }
            
            # Update metrics
            duration = time.time() - start_time
            
            self.prune_metrics["total_prunes"] += 1
            
            # Calculate reduction percentage
            pruned_content = result.get("pruned_content", "")
            original_length = len(content.split())
            pruned_length = len(pruned_content.split())
            reduction_percentage = ((original_length - pruned_length) / original_length) * 100
            
            self.prune_metrics["avg_reduction_percentage"] = ((self.prune_metrics["avg_reduction_percentage"] * 
                                                         (self.prune_metrics["total_prunes"] - 1)) + 
                                                        reduction_percentage) / self.prune_metrics["total_prunes"]
            
            # Update information retention metrics
            if "preservation_metrics" in result:
                retention = result["preservation_metrics"].get("critical_information_retention", 1.0)
                self.prune_metrics["avg_information_retention"] = ((self.prune_metrics["avg_information_retention"] * 
                                                             (self.prune_metrics["total_prunes"] - 1)) + 
                                                            retention) / self.prune_metrics["total_prunes"]
            
            # Track text removed
            text_removed = original_length - pruned_length
            self.prune_metrics["total_text_removed"] += text_removed
            
            # Track pruning decisions
            pruning_decisions = result.get("pruning_decisions", [])
            self.prune_metrics["avg_items_pruned"] = ((self.prune_metrics["avg_items_pruned"] * 
                                                  (self.prune_metrics["total_prunes"] - 1)) + 
                                                 len(pruning_decisions)) / self.prune_metrics["total_prunes"]
            
            # Update general metrics
            self.update_performance_metrics(
                processed_items=1,
                success=True,
                operation_time=duration,
                tokens=original_length + pruned_length  # Input + output tokens
            )
            
            # Add metadata to result
            result["metadata"] = {
                "processing_time": duration,
                "agent": self.name,
                "model": self.model,
                "timestamp": time.time(),
                "original_length": original_length,
                "pruned_length": pruned_length,
                "actual_reduction_percentage": reduction_percentage,
                "target_reduction_percentage": target_reduction_percentage
            }
            
            # Set token_reduction correctly if not already set
            if "preservation_metrics" in result:
                if not result["preservation_metrics"].get("token_reduction"):
                    result["preservation_metrics"]["token_reduction"] = f"{reduction_percentage:.1f}%"
            
            self.log_activity(f"Pruned content by {reduction_percentage:.2f}% in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.log_activity(f"Content pruning failed: {e}", level="error")
            duration = time.time() - start_time
            
            self.update_performance_metrics(
                processed_items=1,
                success=False,
                operation_time=duration
            )
            
            return {
                "error": str(e),
                "success": False,
                "metadata": {
                    "processing_time": duration,
                    "agent": self.name,
                    "model": self.model
                }
            }
    
    def _extract_json_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON response from LLM output.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed JSON object or empty dict if parsing fails
        """
        try:
            # Try to find JSON block in the response
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no code block, try to parse the whole response
            return json.loads(response)
            
        except Exception as e:
            self.log_activity(f"Failed to parse JSON from response: {e}", level="warning")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get combined metrics specific to the prune agent.
        
        Returns:
            Dictionary with metrics
        """
        return {
            **super().get_performance_metrics(),
            **self.prune_metrics
        }
