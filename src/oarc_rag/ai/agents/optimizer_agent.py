"""
Agent for optimizing vector representations for RAG.

This agent specializes in optimizing vector representations for improved
retrieval performance, implementing concepts from Big_Brain.md.
"""
import json
import re
import time
from typing import Any, Dict

from oarc_rag.ai.agents.base_agent import RAGAgent


class OptimizerAgent(RAGAgent):
    """
    Agent for optimizing vector representations for RAG.
    
    This agent uses LLM-based intelligence to optimize vector storage
    parameters, dimensionality, and HNSW graph structure.
    """
    
    def __init__(self, name: str = "optimizer_agent", **kwargs):
        """
        Initialize the optimizer agent.
        
        Args:
            name: Agent name/identifier
            **kwargs: Additional parameters for the base RAGAgent
        """
        # Choose appropriate default template based on optimization target
        optimization_target = kwargs.pop("optimization_target", "pca")
        if optimization_target == "hnsw":
            kwargs.setdefault("prompt_template", "hnsw_optimization")
        else:
            kwargs.setdefault("prompt_template", "pca_optimization")
        
        # Initialize with base class
        super().__init__(name=name, **kwargs)
        
        # Store optimization target
        self.optimization_target = optimization_target
        
        # Specific metrics for optimization operations
        self.optimization_metrics = {
            "total_optimizations": 0,
            "avg_improvement": {
                "storage": 0.0,
                "speed": 0.0,
                "accuracy": 0.0
            },
            "dimension_reductions": 0,
            "hnsw_parameter_tunings": 0
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize vector representations for improved retrieval.
        
        Args:
            input_data: Dictionary containing parameters specific to the optimization target.
            
            For PCA optimization:
                - vector_count: Number of vectors
                - current_dimensions: Current dimensionality
                - storage_size: Current storage size
                - avg_search_time: Average search time
                - usage_patterns: Usage pattern information
                - performance_requirements: Performance requirements
                
            For HNSW optimization:
                - M_value: Current M value
                - ef_construction: Current ef_construction
                - ef_search: Current ef_search
                - vector_dimensions: Vector dimensions
                - usage_analytics: Usage analytics
                - performance_metrics: Performance metrics
                
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Determine which optimization to perform
        if self.optimization_target == "hnsw":
            return self._process_hnsw_optimization(input_data, start_time)
        else:
            return self._process_pca_optimization(input_data, start_time)
    
    def _process_pca_optimization(self, input_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Process PCA-based dimensionality optimization."""
        # Extract parameters
        vector_count = input_data.get("vector_count", 0)
        current_dimensions = input_data.get("current_dimensions", 0)
        
        if vector_count <= 0 or current_dimensions <= 0:
            self.log_activity("Missing required parameters for PCA optimization", level="warning")
            return {"error": "Missing required parameters", "success": False}
        
        storage_size = input_data.get("storage_size", "Unknown")
        avg_search_time = input_data.get("avg_search_time", 0)
        usage_patterns = input_data.get("usage_patterns", "")
        performance_requirements = input_data.get("performance_requirements", "")
        
        # Prepare prompt parameters
        prompt_params = {
            "vector_count": vector_count,
            "current_dimensions": current_dimensions,
            "storage_size": storage_size,
            "avg_search_time": avg_search_time,
            "usage_patterns": usage_patterns,
            "performance_requirements": performance_requirements
        }
        
        # Render prompt from template
        prompt = self.render_prompt_template(**prompt_params)
        
        try:
            # Generate optimization recommendations
            response = self.generate(prompt)
            
            # Extract JSON response
            result = self._extract_json_response(response)
            
            # Fallback if optimization fails
            if not result or "recommendation" not in result:
                self.log_activity("LLM optimization failed, returning default recommendations", level="warning")
                result = {
                    "recommendation": {
                        "current_dimensions": current_dimensions,
                        "recommended_dimensions": min(128, current_dimensions),
                        "reduction_method": "pca",
                        "variance_preservation": 0.95
                    },
                    "expected_benefits": {
                        "storage_reduction": f"{(1 - (min(128, current_dimensions) / current_dimensions)) * 100:.0f}%",
                        "query_speedup": "Estimated 30-50%",
                        "indexing_speedup": "Estimated 20-40%"
                    },
                    "expected_costs": {
                        "one_time_processing_time": "Depends on vector count",
                        "accuracy_impact": "Minimal if variance preservation is high",
                        "reindexing_required": True
                    },
                    "implementation_strategy": {
                        "suggested_approach": "phased_implementation",
                        "verification_method": "random_query_sampling",
                        "rollback_plan": "store_original_vectors_for_30_days"
                    }
                }
            
            # Update metrics
            duration = time.time() - start_time
            
            self.optimization_metrics["total_optimizations"] += 1
            self.optimization_metrics["dimension_reductions"] += 1
            
            # Update expected improvement metrics
            if "expected_benefits" in result:
                benefits = result["expected_benefits"]
                
                # Extract numeric values
                storage_reduction = self._extract_percentage(benefits.get("storage_reduction", "0%"))
                query_speedup = self._extract_percentage(benefits.get("query_speedup", "0%"))
                
                # Average with previous metrics
                prev_count = self.optimization_metrics["total_optimizations"] - 1
                
                self.optimization_metrics["avg_improvement"]["storage"] = (
                    (self.optimization_metrics["avg_improvement"]["storage"] * prev_count + storage_reduction) / 
                    self.optimization_metrics["total_optimizations"]
                )
                
                self.optimization_metrics["avg_improvement"]["speed"] = (
                    (self.optimization_metrics["avg_improvement"]["speed"] * prev_count + query_speedup) / 
                    self.optimization_metrics["total_optimizations"]
                )
            
            # Update general metrics
            self.update_performance_metrics(
                processed_items=1,
                success=True,
                operation_time=duration,
                tokens=len(prompt.split()) + len(json.dumps(result).split())
            )
            
            # Add metadata to result
            result["metadata"] = {
                "processing_time": duration,
                "agent": self.name,
                "model": self.model,
                "timestamp": time.time(),
                "optimization_target": "pca",
                "input_dimensions": current_dimensions
            }
            
            self.log_activity(f"Generated PCA optimization plan in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.log_activity(f"PCA optimization failed: {e}", level="error")
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
                    "model": self.model,
                    "optimization_target": "pca"
                }
            }
    
    def _process_hnsw_optimization(self, input_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Process HNSW graph structure optimization."""
        # Extract parameters
        M_value = input_data.get("M_value", 0)
        ef_construction = input_data.get("ef_construction", 0)
        ef_search = input_data.get("ef_search", 0)
        vector_dimensions = input_data.get("vector_dimensions", 0)
        
        if not all([M_value, ef_construction, ef_search, vector_dimensions]):
            self.log_activity("Missing required parameters for HNSW optimization", level="warning")
            return {"error": "Missing required parameters", "success": False}
        
        usage_analytics = input_data.get("usage_analytics", "")
        performance_metrics = input_data.get("performance_metrics", "")
        
        # Prepare prompt parameters
        prompt_params = {
            "M_value": M_value,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "vector_dimensions": vector_dimensions,
            "usage_analytics": usage_analytics,
            "performance_metrics": performance_metrics,
            # Parameters needed for template formatting
            "recommended_M": M_value * 2,
            "recommended_ef_construction": ef_construction * 1.5,
            "recommended_ef_search": ef_search * 1.5,
            "recommended_layers": "auto"
        }
        
        # Render prompt from template
        prompt = self.render_prompt_template(**prompt_params)
        
        try:
            # Generate optimization recommendations
            response = self.generate(prompt)
            
            # Extract JSON response
            result = self._extract_json_response(response)
            
            # Fallback if optimization fails
            if not result or "parameter_recommendations" not in result:
                self.log_activity("LLM optimization failed, returning default recommendations", level="warning")
                result = {
                    "parameter_recommendations": {
                        "M": min(32, M_value * 2),
                        "ef_construction": min(300, ef_construction * 1.5),
                        "ef_search": min(250, ef_search * 2),
                        "num_layers": "auto"
                    },
                    "expected_improvements": {
                        "average_query_time": "-20%",
                        "p99_latency": "-15%",
                        "recall_at_10": "+5%"
                    },
                    "implementation_plan": {
                        "approach": "incremental",
                        "estimated_time": "Depends on vector count",
                        "resource_requirements": {
                            "cpu_cores": 4,
                            "memory_gb": 8,
                            "temporary_storage_gb": 2
                        }
                    },
                    "specialized_optimizations": []
                }
            
            # Update metrics
            duration = time.time() - start_time
            
            self.optimization_metrics["total_optimizations"] += 1
            self.optimization_metrics["hnsw_parameter_tunings"] += 1
            
            # Update expected improvement metrics
            if "expected_improvements" in result:
                improvements = result["expected_improvements"]
                
                # Extract numeric values
                query_time_improvement = self._extract_percentage(improvements.get("average_query_time", "0%"))
                recall_improvement = self._extract_percentage(improvements.get("recall_at_10", "0%"))
                
                # Average with previous metrics
                prev_count = self.optimization_metrics["total_optimizations"] - 1
                
                self.optimization_metrics["avg_improvement"]["speed"] = (
                    (self.optimization_metrics["avg_improvement"]["speed"] * prev_count + query_time_improvement) / 
                    self.optimization_metrics["total_optimizations"]
                )
                
                self.optimization_metrics["avg_improvement"]["accuracy"] = (
                    (self.optimization_metrics["avg_improvement"]["accuracy"] * prev_count + recall_improvement) / 
                    self.optimization_metrics["total_optimizations"]
                )
            
            # Update general metrics
            self.update_performance_metrics(
                processed_items=1,
                success=True,
                operation_time=duration,
                tokens=len(prompt.split()) + len(json.dumps(result).split())
            )
            
            # Add metadata to result
            result["metadata"] = {
                "processing_time": duration,
                "agent": self.name,
                "model": self.model,
                "timestamp": time.time(),
                "optimization_target": "hnsw",
                "input_parameters": {
                    "M": M_value,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search
                }
            }
            
            self.log_activity(f"Generated HNSW optimization plan in {duration:.2f}s")
            return result
            
        except Exception as e:
            self.log_activity(f"HNSW optimization failed: {e}", level="error")
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
                    "model": self.model,
                    "optimization_target": "hnsw"
                }
            }
    
    def _extract_percentage(self, percentage_str: str) -> float:
        """
        Extract numeric percentage value from string.
        
        Args:
            percentage_str: String containing a percentage
            
        Returns:
            Numeric percentage value (positive or negative)
        """
        try:
            match = re.search(r'([\+\-]?\d+(?:\.\d+)?)', percentage_str)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
            
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
        Get combined metrics specific to the optimizer agent.
        
        Returns:
            Dictionary with metrics
        """
        return {
            **super().get_performance_metrics(),
            **self.optimization_metrics,
            "optimization_target": self.optimization_target
        }
