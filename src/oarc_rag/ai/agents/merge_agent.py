"""
Agent for merging content fragments for RAG.

This agent intelligently merges content fragments while preserving semantic
relationships and resolving potential contradictions or duplications.
"""
import json
import re
import time
from typing import Any, Dict

from oarc_rag.ai.agents.base_agent import RAGAgent


class MergeAgent(RAGAgent):
    """
    Agent for merging content fragments for RAG processing.
    
    This agent uses LLM-based intelligence to merge content fragments while
    preserving semantic coherence, resolving conflicts, and maintaining
    vector space relationships.
    """
    
    def __init__(self, name: str = "merge_agent", **kwargs):
        """
        Initialize the merge agent.
        
        Args:
            name: Agent name/identifier
            **kwargs: Additional parameters for the base RAGAgent
        """
        # Set default prompt template
        kwargs.setdefault("prompt_template", "merge_agent")
        
        # Initialize with base class
        super().__init__(name=name, **kwargs)
        
        # Specific metrics for merging operations
        self.merge_metrics = {
            "avg_fragments_per_merge": 0.0,
            "total_merges": 0,
            "avg_content_preservation": 0.0,
            "contradiction_resolutions": 0,
            "enhanced_connections": 0
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge content fragments into unified content.
        
        Args:
            input_data: Dictionary containing:
                - fragments: List of fragment dictionaries with text and metadata
                - vector_constraints: Vector space constraints (optional)
                - merge_objectives: Merge objectives (optional)
                
        Returns:
            Dictionary with merging results
        """
        start_time = time.time()
        
        # Extract parameters
        fragments = input_data.get("fragments", [])
        if not fragments:
            self.log_activity("No fragments provided for merging", level="warning")
            return {"error": "No fragments provided", "success": False}
        
        vector_constraints = input_data.get("vector_constraints", "")
        merge_objectives = input_data.get("merge_objectives", "")
        
        # Ensure fragments have the expected structure
        processed_fragments = []
        for i, fragment in enumerate(fragments):
            if isinstance(fragment, str):
                # Convert string fragments to dict structure
                processed_fragments.append({
                    "text": fragment,
                    "centroid_distance": 0.0,
                    "key_concepts": []
                })
            elif isinstance(fragment, dict) and "text" in fragment:
                # Use dict as is, ensuring it has all required fields
                if "centroid_distance" not in fragment:
                    fragment["centroid_distance"] = 0.0
                if "key_concepts" not in fragment:
                    fragment["key_concepts"] = []
                processed_fragments.append(fragment)
            else:
                self.log_activity(f"Skipping invalid fragment #{i}", level="warning")
        
        if not processed_fragments:
            self.log_activity("No valid fragments to process", level="error")
            return {"error": "No valid fragments", "success": False}
        
        # Prepare prompt parameters
        prompt_params = {
            "fragments": processed_fragments,
            "vector_constraints": vector_constraints,
            "merge_objectives": merge_objectives
        }
        
        # Render prompt from template
        prompt = self.render_prompt_template(**prompt_params)
        
        try:
            # Generate merged content
            response = self.generate(prompt)
            
            # Extract JSON response
            result = self._extract_json_response(response)
            
            # Fallback to basic merging if LLM merging fails
            if not result or "merged_content" not in result:
                self.log_activity("LLM merging failed, using basic concatenation", level="warning")
                merged_text = "\n\n".join(f.get("text", "") for f in processed_fragments)
                result = {
                    "merged_content": merged_text,
                    "merge_decisions": [{"fragments": list(range(len(processed_fragments))), 
                                      "strategy": "simple_concatenation", 
                                      "rationale": "Automatic fallback merge"}],
                    "content_preservation": {
                        "score": 1.0,
                        "unrepresented_content": "",
                        "enhanced_connections": 0
                    }
                }
            
            # Update metrics
            duration = time.time() - start_time
            
            self.merge_metrics["total_merges"] += 1
            self.merge_metrics["avg_fragments_per_merge"] = ((self.merge_metrics["avg_fragments_per_merge"] * 
                                                         (self.merge_metrics["total_merges"] - 1)) + 
                                                        len(processed_fragments)) / self.merge_metrics["total_merges"]
            
            # Update content preservation metrics
            if "content_preservation" in result:
                preservation = result["content_preservation"]
                score = preservation.get("score", 0.0)
                self.merge_metrics["avg_content_preservation"] = ((self.merge_metrics["avg_content_preservation"] * 
                                                             (self.merge_metrics["total_merges"] - 1)) + 
                                                            score) / self.merge_metrics["total_merges"]
                
                # Track enhanced connections
                self.merge_metrics["enhanced_connections"] += preservation.get("enhanced_connections", 0)
            
            # Track contradiction resolutions
            if "merge_decisions" in result:
                for decision in result["merge_decisions"]:
                    if decision.get("strategy", "").lower() == "contradiction_resolution":
                        self.merge_metrics["contradiction_resolutions"] += 1
            
            # Update general metrics
            total_input_tokens = sum(len(f.get("text", "").split()) for f in processed_fragments)
            total_output_tokens = len(result.get("merged_content", "").split())
            
            self.update_performance_metrics(
                processed_items=len(processed_fragments),
                success=True,
                operation_time=duration,
                tokens=total_input_tokens + total_output_tokens
            )
            
            # Add metadata to result
            result["metadata"] = {
                "processing_time": duration,
                "agent": self.name,
                "model": self.model,
                "timestamp": time.time(),
                "fragment_count": len(processed_fragments)
            }
            
            self.log_activity(f"Merged {len(processed_fragments)} fragments in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.log_activity(f"Content merging failed: {e}", level="error")
            duration = time.time() - start_time
            
            self.update_performance_metrics(
                processed_items=len(processed_fragments),
                success=False,
                operation_time=duration
            )
            
            return {
                "error": str(e),
                "success": False,
                "metadata": {
                    "processing_time": duration,
                    "agent": self.name,
                    "model": self.model,
                    "fragment_count": len(processed_fragments)
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
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                self.log_activity("No JSON found in response", level="warning")
                return {}
        except json.JSONDecodeError as e:
            self.log_activity(f"JSON decoding failed: {e}", level="error")
            return {}
        except Exception as e:
            self.log_activity(f"Unexpected error during JSON extraction: {e}", level="error")
            return {}