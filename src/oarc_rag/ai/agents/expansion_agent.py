"""
Agent for expanding content with additional context and detail.

This agent intelligently expands content while maintaining semantic coherence
and optimizing vector space representation.
"""
import json
import re
import time
from typing import Any, Dict

from oarc_rag.ai.agents.base_agent import RAGAgent


class ExpansionAgent(RAGAgent):
    """
    Agent for expanding content with additional detail and context.
    
    This agent uses LLM-based intelligence to expand content while
    maintaining semantic coherence and vector space relationships.
    """
    
    def __init__(self, name: str = "expansion_agent", **kwargs):
        """
        Initialize the expansion agent.
        
        Args:
            name: Agent name/identifier
            **kwargs: Additional parameters for the base RAGAgent
        """
        # Set default prompt template
        kwargs.setdefault("prompt_template", "expansion_agent")
        
        # Initialize with base class
        super().__init__(name=name, **kwargs)
        
        # Specific metrics for expansion operations
        self.expansion_metrics = {
            "avg_expansion_factor": 0.0,
            "total_expansions": 0,
            "avg_confidence": 0.0,
            "concepts_added": 0
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand content with additional detail and context.
        
        Args:
            input_data: Dictionary containing:
                - content: Text content to expand
                - expansion_objectives: List of expansion objectives
                - vector_neighborhood: Vector neighborhood context (optional)
                - gap_analysis: Information gap analysis (optional)
                - max_expansion_factor: Maximum expansion factor (optional)
                
        Returns:
            Dictionary with expansion results
        """
        start_time = time.time()
        
        # Extract parameters
        content = input_data.get("content", "")
        if not content:
            self.log_activity("No content provided for expansion", level="warning")
            return {"error": "No content provided", "success": False}
        
        expansion_objectives = input_data.get("expansion_objectives", [])
        vector_neighborhood = input_data.get("vector_neighborhood", "")
        gap_analysis = input_data.get("gap_analysis", "")
        max_expansion_factor = input_data.get("max_expansion_factor", 2.0)
        
        # Prepare prompt parameters
        prompt_params = {
            "content": content,
            "expansion_objectives": expansion_objectives,
            "vector_neighborhood": vector_neighborhood,
            "gap_analysis": gap_analysis,
            "max_expansion_factor": max_expansion_factor
        }
        
        # Render prompt from template
        prompt = self.render_prompt_template(**prompt_params)
        
        try:
            # Generate expanded content
            response = self.generate(prompt)
            
            # Extract JSON response
            result = self._extract_json_response(response)
            
            # Fallback to original content if expansion fails
            if not result or "expanded_content" not in result:
                self.log_activity("LLM expansion failed, returning original content", level="warning")
                result = {
                    "expanded_content": content,
                    "rationale": "Expansion failed, returning original content",
                    "confidence": 0.0,
                    "vector_impact_assessment": {
                        "expected_similarity_to_original": 1.0,
                        "neighborhood_preservation": 1.0,
                        "query_match_improvement": []
                    },
                    "metadata": {
                        "expansion_factor": 1.0,
                        "added_concepts": [],
                        "information_sources": []
                    }
                }
            
            # Update metrics
            duration = time.time() - start_time
            
            self.expansion_metrics["total_expansions"] += 1
            
            # Calculate expansion factor
            expanded_content = result.get("expanded_content", "")
            expansion_factor = len(expanded_content.split()) / max(1, len(content.split()))
            
            # Ensure metadata exists
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"]["expansion_factor"] = expansion_factor
            
            self.expansion_metrics["avg_expansion_factor"] = ((self.expansion_metrics["avg_expansion_factor"] * 
                                                         (self.expansion_metrics["total_expansions"] - 1)) + 
                                                        expansion_factor) / self.expansion_metrics["total_expansions"]
            
            # Update confidence metrics
            confidence = result.get("confidence", 0.0)
            self.expansion_metrics["avg_confidence"] = ((self.expansion_metrics["avg_confidence"] * 
                                                     (self.expansion_metrics["total_expansions"] - 1)) + 
                                                    confidence) / self.expansion_metrics["total_expansions"]
            
            # Track added concepts
            added_concepts = len(result.get("metadata", {}).get("added_concepts", []))
            self.expansion_metrics["concepts_added"] += added_concepts
            
            # Update general metrics
            self.update_performance_metrics(
                processed_items=1,
                success=True,
                operation_time=duration,
                tokens=len(content.split()) + len(expanded_content.split())
            )
            
            # Add additional metadata to result
            result["metadata"].update({
                "processing_time": duration,
                "agent": self.name,
                "model": self.model,
                "timestamp": time.time(),
                "original_length": len(content.split()),
                "expanded_length": len(expanded_content.split())
            })
            
            self.log_activity(
                f"Expanded content by factor of {expansion_factor:.2f}x in {duration:.2f}s "
                f"(added {added_concepts} concepts)"
            )
            return result
            
        except Exception as e:
            self.log_activity(f"Content expansion failed: {e}", level="error")
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
        Get combined metrics specific to the expansion agent.
        
        Returns:
            Dictionary with metrics
        """
        return {
            **super().get_performance_metrics(),
            **self.expansion_metrics
        }
