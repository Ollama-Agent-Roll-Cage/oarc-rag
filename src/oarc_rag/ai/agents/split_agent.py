"""
Agent for optimally splitting content for RAG.

This agent intelligently splits content into optimized chunks for RAG,
preserving semantic coherence and information completeness.
"""
import time
import json
import re
from typing import Dict, Any

from oarc_rag.ai.agents.base_agent import RAGAgent
from oarc_rag.core.chunking import TextChunker


class SplitAgent(RAGAgent):
    """
    Agent for optimally splitting content into chunks for RAG processing.
    
    This agent uses LLM-based intelligence to split content into chunks
    that optimize for semantic coherence, information completeness, and
    retrieval effectiveness.
    """
    
    def __init__(self, name: str = "split_agent", **kwargs):
        """
        Initialize the split agent.
        
        Args:
            name: Agent name/identifier
            **kwargs: Additional parameters for the base RAGAgent
        """
        # Set default prompt template
        kwargs.setdefault("prompt_template", "split_agent")
        
        # Initialize with base class
        super().__init__(name=name, **kwargs)
        
        # Initialize chunking tools
        self.text_chunker = TextChunker()
        
        # Specific metrics for splitting operations
        self.split_metrics = {
            "avg_chunks_per_split": 0.0,
            "total_chunks_created": 0,
            "avg_chunk_size": 0,
            "semantic_coherence_scores": [],
            "information_completeness_scores": []
        }
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split content into optimal chunks.
        
        Args:
            input_data: Dictionary containing:
                - content: Text content to split
                - chunk_count: Target number of chunks (optional)
                - chunk_size: Target size of each chunk (optional)
                - min_coherence: Minimum semantic coherence (optional)
                - max_fragmentation: Maximum information fragmentation (optional)
                - retrieval_metrics: Current retrieval performance metrics (optional)
                - content_analysis: Analysis of content (optional)
                
        Returns:
            Dictionary with splitting results
        """
        start_time = time.time()
        
        # Extract parameters
        content = input_data.get("content", "")
        if not content:
            self.log_activity("No content provided for splitting", level="warning")
            return {"error": "No content provided", "success": False}
        
        chunk_count = input_data.get("chunk_count", 0)
        chunk_size = input_data.get("chunk_size", 512)
        min_coherence = input_data.get("min_coherence", 0.7)
        max_fragmentation = input_data.get("max_fragmentation", 0.3)
        retrieval_metrics = input_data.get("retrieval_metrics", {})
        content_analysis = input_data.get("content_analysis", "")
        
        # Prepare prompt parameters
        prompt_params = {
            "content": content,
            "chunk_count": chunk_count,
            "chunk_size": chunk_size,
            "min_coherence": min_coherence,
            "max_fragmentation": max_fragmentation,
            "retrieval_metrics": retrieval_metrics,
            "content_analysis": content_analysis
        }
        
        # Render prompt from template
        prompt = self.render_prompt_template(**prompt_params)
        
        try:
            # Generate split plan
            response = self.generate(prompt)
            
            # Extract JSON response
            result = self._extract_json_response(response)
            
            # Fallback to basic chunking if LLM splitting fails
            if not result or "chunks" not in result:
                self.log_activity("LLM splitting failed, using basic chunker", level="warning")
                chunks = self.text_chunker.chunk_text(content)
                result = {
                    "chunks": [
                        {"content": chunk, "estimated_tokens": len(chunk.split()), 
                         "main_topic": "auto-chunk", "semantic_coherence": 0.5} 
                        for chunk in chunks
                    ],
                    "splitting_rationale": "Automatic fallback chunking",
                    "expected_improvements": {
                        "retrieval_accuracy": "unknown",
                        "query_coverage": "unknown",
                        "vector_quality": "unknown"
                    }
                }
            
            # Update metrics
            chunks = result.get("chunks", [])
            duration = time.time() - start_time
            chunk_count = len(chunks)
            
            self.split_metrics["total_chunks_created"] += chunk_count
            total_chunks = self.split_metrics["total_chunks_created"]
            
            # Calculate average chunk size
            if chunk_count > 0:
                avg_tokens = sum(chunk.get("estimated_tokens", 0) for chunk in chunks) / chunk_count
                self.split_metrics["avg_chunk_size"] = avg_tokens
            
            # Calculate average chunks per split
            operations = self.metrics["successful_operations"] + 1  # Including this one
            self.split_metrics["avg_chunks_per_split"] = total_chunks / operations
            
            # Process semantic coherence scores
            coherence_scores = [chunk.get("semantic_coherence", 0) for chunk in chunks if "semantic_coherence" in chunk]
            if coherence_scores:
                self.split_metrics["semantic_coherence_scores"].extend(coherence_scores)
            
            # Update general metrics
            self.update_performance_metrics(
                processed_items=1,
                success=True,
                operation_time=duration,
                tokens=len(content.split()) + sum(len(c.get("content", "").split()) for c in chunks)
            )
            
            # Add metadata to result
            result["metadata"] = {
                "processing_time": duration,
                "agent": self.name,
                "model": self.model,
                "timestamp": time.time()
            }
            
            self.log_activity(f"Split content into {chunk_count} chunks in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.log_activity(f"Content splitting failed: {e}", level="error")
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
        Get combined metrics specific to the split agent.
        
        Returns:
            Dictionary with metrics
        """
        return {
            **super().get_performance_metrics(),
            **self.split_metrics
        }
