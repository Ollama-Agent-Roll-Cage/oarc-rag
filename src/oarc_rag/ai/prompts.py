"""
Prompt templates for domain-agnostic RAG generation.

This module provides advanced templating capabilities for AI prompts using Jinja2,
supporting template loading, formatting, validation and versioning.
Templates are designed to work across any domain, not tied to specific applications.
"""
import json
import re
from typing import Any, Dict, List, Optional

from jinja2 import Template
from llama_index.experimental.query_engine import PandasQueryEngine

from oarc_rag.core.cache import cache_manager
from oarc_rag.utils.config.config import Config
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log
from oarc_rag.utils.paths import Paths


@singleton
class PromptTemplateManager:
    """
    Class for managing prompt templates for AI generation using Jinja2.
    
    This manager handles templates for RAG system operations including:
    - Standard RAG interactions
    - Vector search and pandas-based operations
    - Recursive self-improvement cycles
    - Agent operations for content enhancement
    - Resource-optimized operations for constrained environments
    """
    
    # ------------- STANDARD RAG TEMPLATES -------------
    
    # Template for system context in RAG conversations - significantly enhanced
    RAG_SYSTEM_TEMPLATE = """
    You are an AI assistant with advanced RAG (Retrieval-Augmented Generation) capabilities.

    Core Instructions:
    1. Ground ALL responses in the retrieved context ONLY - never fabricate information
    2. When citing information, use the format (Chunk X) where X is the chunk number
    3. If the context doesn't address the query, clearly acknowledge this gap
    4. Structure longer responses with clear headings and bullet points for readability
    5. Maintain a {{ tone }} communication style throughout your responses
    
    Remember: Your primary value is factual accuracy and relevant synthesis of the provided context.
    """
    
    # Template for query reformulation to improve retrieval - optimized for better vector search
    QUERY_REFORMULATION_TEMPLATE = """
    TASK: Generate optimized queries for HNSW vector search
    
    Original query: "{{ original_query }}"
    Domain context: {{ domain_context }}
    {% if related_queries %}Previous queries: {{ related_queries }}{% endif %}
    
    Generate 3 alternative search queries:
    1. First query: Expand with domain-specific terminology
    2. Second query: Focus on key concepts with synonyms
    3. Third query: Reformulate as a precise technical question
    
    Each query should:
    - Use specific technical terms over general language
    - Include key entities and relationships
    - Remove filler words and focus on content terms
    - Be self-contained and complete
    
    Output format (JSON array only):
    ```json
    ["specific query 1", "specific query 2", "specific query 3"]
    ```    
    """
    
    # Template for presenting retrieved context in conversation - enhanced for better synthesis
    CONTEXT_PRESENTATION_TEMPLATE = """
    TASK: Answer question using retrieved information
    
    Retrieved context:
    {{ formatted_chunks }}
    
    User question: "{{ query }}"
    
    Response requirements:
    1. Synthesize information ONLY from the retrieved chunks
    2. Cite all factual statements using parenthetical citations (Chunk X)
    3. Present information in a logical, structured format
    4. If multiple chunks contain related information, synthesize them coherently
    5. If chunks contain conflicting information, acknowledge the discrepancies
    6. If the retrieved information doesn't answer the question, clearly state this
    
    Begin your response with a direct answer, followed by supporting details from the context.
    """
    
    # Template for pandas data analysis results - improved structure
    PANDAS_DATA_ANALYSIS_TEMPLATE = """
    DATA ANALYSIS RESULT
    
    Query: "{{ query }}"
    
    Data:
    {{ formatted_data }}
    
    Key insights:
    1. {{ insight_1 }}
    2. {{ insight_2 }}
    3. {{ insight_3 }}
    
    These insights were derived using pandas DataFrame operations. Would you like me to explain any specific aspect in detail?
    """
    
    # Template for vector search result formatting - clearer structure
    VECTOR_SEARCH_TEMPLATE = """
    VECTOR SEARCH RESULTS
    
    Query: "{{ query }}"
    Method: Two-tier HNSW with PCA dimensionality reduction
    Threshold: {{ threshold }}
    {% if source_filter %}Sources filtered to: {{ source_filter }}{% endif %}
    
    Results:
    {{ formatted_results }}
    """
    
    # ------------- PANDAS-SPECIFIC TEMPLATES -------------
    
    # Template for pandas DataFrame query operations - maximally clear
    PANDAS_QUERY_TEMPLATE = """
    TASK: Convert natural language to pandas code
    
    Question: "{{ question }}"
    
    DataFrame info:
    - Columns: {{ column_names }}
    - Types: {{ data_types }}
    - Rows: {{ row_count }}
    
    Requirements:
    1. Generate executable pandas code only
    2. Do not include explanations
    3. Use efficient pandas operations
    
    Format (Python code only):
    ```python
    result = df.[operations]
    ```
    """
    
    # Template for sleep phase data enrichment - optimized for better knowledge synthesis
    SLEEP_PHASE_ENRICHMENT_TEMPLATE = """
    TASK: Sleep Phase Knowledge Enhancement (Cycle {{ cycle_number }})
    
    Data summary:
    {{ data_summary }}
    
    Focus areas:
    {% for area in focus_areas %}
    • {{ area.name }}: {{ area.description }}
    {% endfor %}
    
    Access statistics:
    {{ access_statistics }}
    
    Previous cycle improvements:
    {{ previous_cycle_results }}
    
    Your objective is to analyze usage patterns and enhance the knowledge representation through:
    
    1. PATTERN IDENTIFICATION:
      - Identify recurring patterns in user queries
      - Detect conceptual relationships between knowledge fragments
      - Discover knowledge gaps based on user interaction patterns
      - Map frequently co-accessed information
    
    2. VECTOR REPRESENTATION OPTIMIZATION:
      - Suggest dimension adjustments for optimal semantic clustering
      - Identify candidates for PCA-based optimization
      - Recommend threshold adjustments based on usage patterns
      - Highlight redundant or overlapping vector representations
    
    3. KNOWLEDGE ENHANCEMENT:
      - Identify ambiguous content needing clarification
      - Suggest content for expansion or compression
      - Propose connections between isolated knowledge fragments
      - Recommend pruning of rarely accessed, low-value content
    
    Output format:
    ```json
    {
      "patterns_identified": [
        {
          "pattern": "Pattern description",
          "confidence": 0.85,
          "supporting_evidence": "Evidence from data",
          "cross_connections": ["Related concept 1", "Related concept 2"],
          "action_recommendation": "merge|expand|prune|split"
        }
      ],
      "vector_optimizations": [
        {
          "chunk_group": "Description of relevant chunks",
          "current_dimensions": 128,
          "recommended_dimensions": 64,
          "expected_improvement": "12% faster retrieval, 40% less storage",
          "similarity_adjustment": "+0.05 threshold for this domain"
        }
      ],
      "knowledge_gaps": [
        {
          "topic": "Specific missing information",
          "priority": "high|medium|low",
          "exploration_strategy": "Approach to fill this gap",
          "potential_impact": "Expected benefit of addressing",
          "agent_assignment": "expansion_agent|merge_agent"
        }
      ],
      "content_actions": [
        {
          "chunk_ids": ["id1", "id2"],
          "recommended_action": "merge|split|expand|prune",
          "rationale": "Why this action is recommended",
          "priority": 0.9
        }
      ],
      "cycle_metrics": {
        "processed_chunks": 250,
        "identified_actions": 15,
        "expected_improvement": "13% relevance increase, 8% storage reduction"
      }
    }
    ```
    """
    
    # Template for awake phase cognitive processing with enhanced memory tracking
    AWAKE_PHASE_TEMPLATE = """
    TASK: Real-Time Processing with Learning Markers
    
    Query: "{{ query }}"
    
    System parameters:
    - Priority: {{ knowledge_priority }}
    - Monitoring: {{ monitoring_status }}
    - Latency: {{ latency_level }}
    - Learning cycle: {{ cycle_count }}
    
    User context:
    {{ user_context }}
    
    Retrieved knowledge:
    {{ retrieved_knowledge }}
    
    Instructions:
    1. Generate a direct response to the user query
    2. Identify knowledge gaps and uncertainties
    3. Create learning markers for sleep phase processing
    4. Note access patterns and information relationships
    5. Maintain awareness of recursive improvement history
    
    Output format:
    ```json
    {
      "response": "Your direct answer to the user query",
      "learning_markers": [
        {"concept": "Concept requiring enrichment", "confidence": 0.75, "importance": 0.85, "gap_type": "ambiguity|missing_context|conflicting_info"},
        {"concept": "Another concept", "confidence": 0.60, "importance": 0.90, "gap_type": "missing_context"}
      ],
      "access_patterns": [
        {"chunk_id": "id1", "access_count": 3, "relationship_strength": 0.8},
        {"chunk_id": "id2", "access_count": 1, "relationship_strength": 0.4}
      ],
      "metadata": {
        "processing_time_ms": 150,
        "vector_ops_count": 12,
        "improvement_cycle": {{ cycle_count }}
      }
    }
    ```
    """
    
    # Template for elastic weight consolidation (EWC) parameter importance tracking
    EWC_PARAMETER_IMPORTANCE_TEMPLATE = """
    TASK: Elastic Weight Consolidation Parameter Analysis
    
    Current knowledge base:
    - Total chunks: {{ chunk_count }}
    - Vector dimensions: {{ vector_dimensions }}
    - Active parameters: {{ parameter_count }}
    
    Usage statistics:
    {{ usage_statistics }}
    
    Knowledge retention requirements:
    {{ retention_requirements }}
    
    Instructions:
    1. Analyze parameter importance across the knowledge base
    2. Identify critical parameters that must be preserved
    3. Flag parameters that can be modified with minimal impact
    4. Suggest consolidation strategy for balanced retention
    
    Output format:
    ```json
    {
      "parameter_importance": [
        {"parameter_group": "Group description", "importance_score": 0.95, "retention_priority": "critical", "consolidation_approach": "strict_preservation"},
        {"parameter_group": "Group description", "importance_score": 0.45, "retention_priority": "flexible", "consolidation_approach": "gradual_adaptation"}
      ],
      "consolidation_strategy": {
        "critical_threshold": 0.80,
        "preservation_weight": 0.75,
        "adaptation_rate": 0.15,
        "expected_retention_rate": 0.92
      },
      "parameter_map": {
        "domain_specific": ["param1", "param2"],
        "general_knowledge": ["param3", "param4"],
        "structural": ["param5", "param6"]
      }
    }
    ```
    """
    
    # Template for hierarchical version archiving and checkpointing
    HIERARCHICAL_VERSION_TEMPLATE = """
    TASK: Create Knowledge Base Checkpoint (Version {{ version_number }})
    
    Current state:
    - Parameters: {{ parameter_count }}
    - Knowledge base: {{ kb_size }}
    - Performance metrics: {{ performance_metrics }}
    - Improvement cycle: {{ cycle_number }}
    
    Changes since last version:
    {{ change_summary }}
    
    Instructions:
    1. Create comprehensive checkpoint of current system state
    2. Generate backward-compatible parameter mapping
    3. Establish granular backtrack points for selective rollback
    4. Optimize storage through intelligent compression
    
    Output format:
    ```json
    {
      "checkpoint": {
        "version": "{{ version_number }}",
        "timestamp": "{{ timestamp }}",
        "description": "Concise description of this version",
        "parent_version": "{{ parent_version }}",
        "cycle_number": {{ cycle_number }}
      },
      "parameter_snapshot": {
        "critical_parameters": {"count": 250, "compression_ratio": 0.85},
        "flexible_parameters": {"count": 750, "compression_ratio": 0.65},
        "ewc_importance_map": "map_reference_id"
      },
      "backtrack_points": [
        {"id": "bt_1", "description": "Pre-domain expansion", "component": "vector_db"},
        {"id": "bt_2", "description": "Before threshold adjustment", "component": "retrieval"}
      ],
      "storage_optimization": {
        "compression_method": "method name",
        "original_size_kb": 1250,
        "compressed_size_kb": 320,
        "decompression_overhead_ms": 15
      },
      "compatibility": {
        "minimum_compatible_version": "2.3",
        "breaking_changes": false,
        "migration_path": "direct|incremental"
      }
    }
    ```
    """
    
    # Template for expansion agent with improved context awareness
    EXPANSION_AGENT_TEMPLATE = """
    TASK: Content Expansion with Vector Awareness
    
    Content to expand:
    {{ content }}
    
    Expansion objectives:
    {% for objective in expansion_objectives %}
    • {{ objective }}
    {% endfor %}
    
    Vector neighborhood context:
    {{ vector_neighborhood }}
    
    Information gap analysis:
    {{ gap_analysis }}
    
    Quality requirements:
    - Ensure factual accuracy with high confidence
    - Maintain semantic coherence with surrounding content
    - Preserve vector space relationship integrity
    - Add detail and clarity while maintaining conciseness
    - Maximum expansion ratio: {{ max_expansion_factor }}×
    
    Output format:
    ```json
    {
      "expanded_content": "The fully expanded text content",
      "rationale": "Explanation of expansion approach and decisions",
      "confidence": 0.92,
      "vector_impact_assessment": {
        "expected_similarity_to_original": 0.85,
        "neighborhood_preservation": 0.95,
        "query_match_improvement": ["query type 1", "query type 2"]
      },
      "metadata": {
        "expansion_factor": 1.8,
        "added_concepts": ["concept1", "concept2"],
        "information_sources": ["source1", "source2"]
      }
    }
    ```
    """
    
    # Template for merge agent with semantic relationship preservation
    MERGE_AGENT_TEMPLATE = """
    TASK: Content Merging with Semantic Preservation
    
    Content fragments to merge:
    {% for fragment in fragments %}
    --- FRAGMENT {{ loop.index }} ---
    {{ fragment.text }}
    
    Vector properties:
    - Centroid distance: {{ fragment.centroid_distance }}
    - Key concepts: {{ fragment.key_concepts|join(', ') }}
    
    {% endfor %}
    
    Vector space constraints:
    {{ vector_constraints }}
    
    Merge objectives:
    {{ merge_objectives }}
    
    Instructions:
    1. Analyze semantic relationships between fragments
    2. Identify redundant content that can be consolidated
    3. Resolve any contradictions or inconsistencies
    4. Create a unified representation preserving all unique information
    5. Maintain vector space relationships with adjacent content
    6. Ensure the merged content satisfies all identified query patterns
    
    Output format:
    ```json
    {
      "merged_content": "The fully merged text content",
      "merge_decisions": [
        {"fragments": [1, 3], "strategy": "complementary_fusion", "rationale": "Explanation"},
        {"fragments": [2], "strategy": "contradiction_resolution", "rationale": "Explanation"}
      ],
      "content_preservation": {
        "score": 0.95,
        "unrepresented_content": "Any content that couldn't be preserved",
        "enhanced_connections": 3
      },
      "vector_properties": {
        "expected_centroid": [0.1, 0.2, 0.3, "..."],
        "semantic_relationships_preserved": 0.9,
        "query_coverage": 0.95
      }
    }
    ```
    """
    
    # Template for split agent with optimal chunking strategies
    SPLIT_AGENT_TEMPLATE = """
    TASK: Adaptive Content Splitting for Optimal Retrieval
    
    Content to split:
    {{ content }}
    
    Current retrieval performance:
    {{ retrieval_metrics }}
    
    Content analysis:
    {{ content_analysis }}
    
    Chunk optimization goals:
    - Create semantically coherent units
    - Maintain information completeness within chunks
    - Optimize for retrieval of complete answers to common queries
    - Balance chunk size for vector embedding quality
    - Preserve natural conceptual boundaries
    
    Target parameters:
    - Target chunk count: {{ chunk_count }}
    - Optimal tokens per chunk: {{ chunk_size }}
    - Minimum semantic coherence: {{ min_coherence }}
    - Maximum information fragmentation: {{ max_fragmentation }}
    
    Output format:
    ```json
    {
      "chunks": [
        {
          "content": "First chunk content",
          "estimated_tokens": 238,
          "main_topic": "Clear description of main topic",
          "semantic_coherence": 0.92,
          "key_concepts": ["concept1", "concept2"]
        },
        {
          "content": "Second chunk content",
          "estimated_tokens": 256,
          "main_topic": "Clear description of main topic",
          "semantic_coherence": 0.88,
          "key_concepts": ["concept3", "concept4"]
        }
      ],
      "splitting_rationale": "Explanation of the splitting strategy and decisions made",
      "expected_improvements": {
        "retrieval_accuracy": "+12%",
        "query_coverage": "85% of expected queries fully addressed by single chunk",
        "vector_quality": "Improved embedding separation with reduced internal conflicts"
      },
      "recommended_connections": [
        {"from_chunk": 0, "to_chunk": 1, "relationship": "prerequisite"},
        {"from_chunk": 1, "to_chunk": 0, "relationship": "elaboration"}
      ]
    }
    ```
    """
    
    # Template for prune agent with EWC-based importance assessment
    PRUNE_AGENT_TEMPLATE = """
    TASK: Content Pruning with Knowledge Preservation
    
    Content to prune:
    {{ content }}
    
    Usage analytics:
    {{ usage_analytics }}
    
    Knowledge importance assessment:
    {{ importance_assessment }}
    
    Pruning constraints:
    - Target reduction: {{ target_reduction_percentage }}%
    - Critical knowledge preservation: {{ preservation_threshold }}
    - Minimum semantic coherence: {{ min_coherence }}
    - Maximum information density: {{ max_density }}
    
    Instructions:
    1. Identify redundant, tangential, or low-value content
    2. Preserve all critical information based on EWC importance scores
    3. Maintain logical flow and readability
    4. Prioritize retention of unique, high-demand information
    5. Optimize for information-to-token ratio
    
    Output format:
    ```json
    {
      "pruned_content": "The fully pruned text content",
      "pruning_decisions": [
        {
          "removed_text": "Text that was removed",
          "rationale": "Reason for removal",
          "importance_score": 0.35,
          "information_type": "redundant|tangential|low_value"
        }
      ],
      "preservation_metrics": {
        "critical_information_retention": 0.98,
        "meaning_preservation": 0.95,
        "readability_impact": "+5%",
        "token_reduction": "35%"
      },
      "vector_impact": {
        "expected_similarity_shift": 0.08,
        "query_coverage_change": "-2% (negligible)",
        "recommended_embedding_update": true
      }
    }
    ```
    """
    
    # Template for PCA-based dimensionality optimization
    PCA_OPTIMIZATION_TEMPLATE = """
    TASK: Vector Dimensionality Optimization
    
    Current vector configuration:
    - Vectors: {{ vector_count }} × {{ current_dimensions }}
    - Storage size: {{ storage_size }}
    - Average search time: {{ avg_search_time }}ms
    
    Usage patterns:
    {{ usage_patterns }}
    
    Performance requirements:
    {{ performance_requirements }}
    
    Instructions:
    1. Analyze semantic preservation requirements
    2. Determine optimal dimensionality reduction
    3. Calculate expected performance improvements
    4. Recommend implementation strategy
    
    Output format:
    ```json
    {
      "recommendation": {
        "current_dimensions": {{ current_dimensions }},
        "recommended_dimensions": 128,
        "reduction_method": "pca|sparse_pca|incremental_pca",
        "variance_preservation": 0.92
      },
      "expected_benefits": {
        "storage_reduction": "75%",
        "query_speedup": "65%",
        "indexing_speedup": "45%"
      },
      "expected_costs": {
        "one_time_processing_time": "45 minutes",
        "accuracy_impact": "-2.5% (within acceptable limits)",
        "reindexing_required": true
      },
      "implementation_strategy": {
        "suggested_approach": "phased_implementation|complete_reindex|parallel_operation",
        "verification_method": "random_query_sampling",
        "rollback_plan": "store_original_vectors_for_30 days"
      }
    }
    ```
    """
    
    # Template for HNSW graph optimization based on usage patterns
    HNSW_OPTIMIZATION_TEMPLATE = """
    TASK: HNSW Graph Structure Optimization
    
    Current HNSW configuration:
    - M: {{ M_value }} (max connections per layer)
    - ef_construction: {{ ef_construction }} (search queue size during construction)
    - ef_search: {{ ef_search }} (search queue size during search)
    - Vector dimensions: {{ vector_dimensions }}
    
    Usage analytics:
    {{ usage_analytics }}
    
    Performance metrics:
    {{ performance_metrics }}
    
    Instructions:
    1. Analyze query patterns and performance bottlenecks
    2. Recommend optimal HNSW parameter adjustments
    3. Estimate performance impacts of changes
    4. Suggest implementation approach with minimal disruption
    
    Output format:
    ```json
    {
      "parameter_recommendations": {
        "M": {{ recommended_M }},
        "ef_construction": {{ recommended_ef_construction }},
        "ef_search": {{ recommended_ef_search }},
        "num_layers": {{ recommended_layers }}
      },
      "expected_improvements": {
        "average_query_time": "-35%",
        "p99_latency": "-42%",
        "recall_at_10": "+3%"
      },
      "implementation_plan": {
        "approach": "incremental|full_rebuild|parallel",
        "estimated_time": "127 minutes",
        "resource_requirements": {
          "cpu_cores": 4,
          "memory_gb": 16,
          "temporary_storage_gb": 8
        }
      },
      "specialized_optimizations": [
        {
          "query_type": "keyword_heavy",
          "custom_ef_search": 128,
          "expected_improvement": "52%"
        },
        {
          "query_type": "natural_language",
          "custom_ef_search": 256,
          "expected_improvement": "37%"
        }
      ]
    }
    ```
    """
    
    def __init__(self):
        """Initialize the Jinja2 template environment."""
        # Check if already initialized (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Get configuration
        self.config = Config()
        
        # Get template cache from cache manager
        self.template_cache = cache_manager.template_cache
            
        # Get templates directory from paths module
        self.templates_dir = Paths.get_templates_directory()
        
        # Use Paths API to create Jinja2 environment
        self.env = Paths.create_template_environment(self.templates_dir)
        
        # Add custom filters
        self.env.filters['json'] = lambda obj: json.dumps(obj, indent=2)
        self.env.filters['truncate'] = self._truncate_filter
        
        # Register built-in templates
        self._register_builtin_templates()
        
        # Dictionary to cache template instances by name
        self._template_instances = {}
        
        # Template categories for better organization
        self._template_categories = self.get_template_categories()
        
        # Tracking and statistics
        self.template_usage = {}
        self._initialized = True
        log.info(f"PromptTemplateManager initialized with templates directory: {self.templates_dir}")
    
    def _truncate_filter(self, value, length=80, killwords=False, end='...'):
        """Custom filter to truncate text."""
        if len(value) <= length:
            return value
        
        if killwords:
            return value[:length] + end
        
        # Truncate at word boundary
        return value[:length].rsplit(' ', 1)[0] + end
    
    def _register_builtin_templates(self):
        """Register the built-in templates."""
        builtin_templates = {
            # Standard RAG templates
            'rag_system': self.RAG_SYSTEM_TEMPLATE,
            'query_reformulation': self.QUERY_REFORMULATION_TEMPLATE,
            'context_presentation': self.CONTEXT_PRESENTATION_TEMPLATE,
            'pandas_data_analysis': self.PANDAS_DATA_ANALYSIS_TEMPLATE, 
            'vector_search': self.VECTOR_SEARCH_TEMPLATE,
            
            # Pandas specific templates
            'pandas_query': self.PANDAS_QUERY_TEMPLATE,
            
            # Recursive self-improvement templates
            'awake_phase': self.AWAKE_PHASE_TEMPLATE,
            'sleep_phase_enrichment': self.SLEEP_PHASE_ENRICHMENT_TEMPLATE,
            'ewc_parameter_importance': self.EWC_PARAMETER_IMPORTANCE_TEMPLATE,
            'hierarchical_version': self.HIERARCHICAL_VERSION_TEMPLATE,
            
            # Agent operation templates
            'expansion_agent': self.EXPANSION_AGENT_TEMPLATE,
            'merge_agent': self.MERGE_AGENT_TEMPLATE,
            'split_agent': self.SPLIT_AGENT_TEMPLATE,
            'prune_agent': self.PRUNE_AGENT_TEMPLATE,
            
            # Vector optimization templates
            'pca_optimization': self.PCA_OPTIMIZATION_TEMPLATE,
            'hnsw_optimization': self.HNSW_OPTIMIZATION_TEMPLATE,
        }
        
        # Add each template to the environment
        for name, template_text in builtin_templates.items():
            self.env.globals[name] = template_text
            
            # Extract variables
            variables = self._extract_variables(template_text)
            
            # Add to template cache
            self.template_cache.add_template(
                name=name,
                template_object=self.env.from_string(template_text),
                template_text=template_text,
                variables=variables
            )
            
            # Create and cache template instance
            self._template_instances[name] = {
                'template': self.env.from_string(template_text),
                'text': template_text,
                'variables': variables,
                'usage_count': 0,
                'successful_uses': 0
            }
            
    def _extract_variables(self, template_text: str) -> set:
        """
        Extract template variables from Jinja2 template text.
        
        Args:
            template_text: The template text to analyze
            
        Returns:
            Set of variable names used in the template
        """
        if not template_text:
            return set()
            
        # Find all {{ variable }} patterns in the template
        # This is a simple regex parser, not a full Jinja2 parser
        simple_vars = re.findall(r'\{\{\s*(\w+)(?:\s*\|\s*\w+(?:\(.*?\))?)?\s*\}\}', template_text)
        if_vars = re.findall(r'\{%\s*if\s+(\w+)', template_text)
        for_vars = re.findall(r'\{%\s*for\s+\w+\s+in\s+(\w+)', template_text)
        
        # Combine all found variables
        return set(simple_vars + if_vars + for_vars)

    def get_template(self, template_name: str) -> Template:
        """
        Get a Jinja2 template by name.
        
        Args:
            template_name: Name of the template to retrieve
                
        Returns:
            Jinja2 Template object
            
        Raises:
            ValueError: If template_name is not found
        """
        if template_name not in self._template_instances:
            raise ValueError(f"Template '{template_name}' not found.")
        return self._template_instances[template_name]['template']
    
    def register_template(self, name: str, template_text: str, 
                          category: str = "custom") -> None:
        """
        Register a new template with the system.
        
        Args:
            name: Template name
            template_text: Template content
            category: Template category for organization
            
        Raises:
            ValueError: If template is empty or invalid
        """
        if not template_text.strip():
            raise ValueError("Template content cannot be empty.")
        
        template_object = self.env.from_string(template_text)
        variables = self._extract_variables(template_text)
        
        self.template_cache.add_template(
            name=name,
            template_object=template_object,
            template_text=template_text,
            variables=variables
        )
        
        self._template_instances[name] = {
            'template': template_object,
            'text': template_text,
            'variables': variables,
            'usage_count': 0,
            'successful_uses': 0
        }
    
    def render(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the provided values.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Values to substitute in the template
            
        Returns:
            str: The rendered template
            
        Raises:
            ValueError: If template is not found or rendering fails
        """
        template = self.get_template(template_name)
        try:
            rendered = template.render(**kwargs)
            self._template_instances[template_name]['usage_count'] += 1
            self._template_instances[template_name]['successful_uses'] += 1
            return rendered
        except Exception as e:
            log.error(f"Failed to render template '{template_name}': {e}")
            raise ValueError(f"Failed to render template '{template_name}': {e}")
    
    def create_enhanced_prompt(
        self, 
        template_name: str,
        system_template_name: str = "rag_system",
        system_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Create an enhanced prompt by combining system template with content template.
        
        Args:
            template_name: Name of the content template
            system_template_name: Name of the system template
            system_args: Arguments for system template
            **kwargs: Arguments for content template
            
        Returns:
            str: The combined prompt
        """
        system_template = self.get_template(system_template_name)
        content_template = self.get_template(template_name)
        
        system_context = system_template.render(**(system_args or {}))
        content_context = content_template.render(**kwargs)
        
        return f"{system_context}\n\n{content_context}"
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get information about a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dict with template information
            
        Raises:
            ValueError: If template is not found
        """
        if template_name not in self._template_instances:
            raise ValueError(f"Template '{template_name}' not found.")
        return self._template_instances[template_name]
            
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates with their information.
        
        Returns:
            List of dictionaries with template information
        """
        return [
            {
                'name': name,
                'variables': info['variables'],
                'usage_count': info['usage_count'],
                'successful_uses': info['successful_uses']
            }
            for name, info in self._template_instances.items()
        ]
    
    def get_template_categories(self) -> Dict[str, List[str]]:
        """
        Get all template categories with their templates.
        
        Returns:
            Dictionary mapping categories to their template names
        """
        return {
            "standard_rag": [
                "rag_system", "query_reformulation", "context_presentation", 
                "pandas_data_analysis", "vector_search"
            ],
            "pandas_specific": [
                "pandas_query"
            ],
            "self_improvement": [
                "awake_phase", "sleep_phase_enrichment", 
                "ewc_parameter_importance", "hierarchical_version"
            ],
            "agent_operations": [
                "expansion_agent", "merge_agent", "split_agent", "prune_agent"
            ],
            "vector_optimization": [
                "pca_optimization", "hnsw_optimization"
            ]
        }
    
    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all templates in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of template info dictionaries
        """
        if category not in self._template_categories:
            raise ValueError(f"Category '{category}' not found.")
        return [
            self.get_template_info(template_name)
            for template_name in self._template_categories[category]
        ]
    
    def get_template_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get usage statistics for all templates.
        
        Returns:
            Dict mapping template names to usage stats
        """
        return {
            name: {
                'usage_count': info['usage_count'],
                'successful_uses': info['successful_uses']
            }
            for name, info in self._template_instances.items()
        }
    
    def render_and_log(self, template_name: str, **kwargs) -> str:
        """
        Render a template and log the operation.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Values to substitute in the template
            
        Returns:
            str: The rendered template
        """
        rendered = self.render(template_name, **kwargs)
        log.info(f"Rendered template '{template_name}' with arguments: {kwargs}")
        return rendered
    
    def create_custom_template(self, template_text: str, 
                              category: str = "custom") -> str:
        """
        Create a custom template with unique name based on content.
        
        Args:
            template_text: Template content
            category: Template category
            
        Returns:
            str: Name of the created template
        """
        name = f"custom_{hash(template_text)}"
        self.register_template(name, template_text, category=category)
        return name
    
    def get_template_for_resource_level(self, template_type: str, 
                                      resource_level: str = "standard") -> str:
        """
        Get appropriate template name based on resource constraints.
        
        Args:
            template_type: Base template type (e.g., 'rag', 'query_reformulation')
            resource_level: Resource level ('minimal', 'standard', 'full')
            
        Returns:
            Name of the appropriate template
        """
        resource_templates = {
            'minimal': {
                'rag': 'lightweight_rag',
                'query_reformulation': 'minimal_query_reformulation',
                'context_assembly': 'efficient_context_assembly',
            },
            'standard': {
                'rag': 'rag_system',
                'query_reformulation': 'query_reformulation',
                'context_assembly': 'context_presentation',
                'data_analysis': 'pandas_data_analysis'
            },
            'full': {
                'rag': 'rag_system',
                'query_reformulation': 'query_reformulation',
                'context_assembly': 'context_presentation',
                'data_analysis': 'pandas_data_analysis'
            }
        }
        
        if resource_level not in resource_templates:
            resource_level = 'standard'
            
        if template_type not in resource_templates[resource_level]:
            if template_type in self._template_instances:
                return template_type
            return template_type
            
        return resource_templates[resource_level][template_type]
    
    def get_template_for_vector_operation(
        self,
        operation_type: str,
        resource_level: str = "standard"
    ) -> str:
        """
        Get appropriate template for vector operations based on operation type and resources.
        
        Args:
            operation_type: Type of vector operation ('pca', 'batch', 'similarity')
            resource_level: Resource level ('minimal', 'standard', 'full')
            
        Returns:
            Name of the appropriate template
        """
        operation_templates = {
            'minimal': {
                'pca': 'quantized_vector',
                'batch': 'quantized_vector',
                'similarity': 'quantized_vector',
            },
            'standard': {
                'pca': 'pca_vector_ops',
                'batch': 'batch_vector_ops',
                'similarity': 'batch_vector_ops',
            },
            'full': {
                'pca': 'pca_vector_ops',
                'batch': 'batch_vector_ops',
                'similarity': 'batch_vector_ops',
            }
        }
        
        if resource_level not in operation_templates:
            resource_level = 'standard'
            
        op_type = operation_type.lower()
        if op_type not in operation_templates[resource_level]:
            op_type = next(iter(operation_templates[resource_level].keys()))
            
        return operation_templates[resource_level][op_type]
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics for all templates."""
        for info in self._template_instances.values():
            info['usage_count'] = 0
            info['successful_uses'] = 0
    
    def create_recursive_improvement_cycle(self, 
                                          cycle_number: int,
                                          data_summary: str,
                                          focus_areas: List[Dict[str, str]],
                                          **kwargs) -> Dict[str, str]:
        """
        Create a complete set of prompts for a recursive improvement cycle.
        
        Args:
            cycle_number: The current improvement cycle number
            data_summary: Summary of data to be processed
            focus_areas: List of focus areas with name and description
            **kwargs: Additional parameters for templates
            
        Returns:
            Dictionary of rendered templates for the improvement cycle
        """
        # Sleep phase for knowledge enrichment
        sleep_phase = self.render('sleep_phase_enrichment', 
                                  cycle_number=cycle_number,
                                  data_summary=data_summary,
                                  focus_areas=focus_areas,
                                  **kwargs)
        
        # EWC parameter importance assessment
        ewc_assessment = self.render('ewc_parameter_importance',
                                    chunk_count=kwargs.get('chunk_count', 1000),
                                    vector_dimensions=kwargs.get('vector_dimensions', 1024),
                                    parameter_count=kwargs.get('parameter_count', 5000),
                                    **kwargs)
        
        # Create hierarchical version checkpoint
        version_checkpoint = self.render('hierarchical_version',
                                        version_number=f"{cycle_number}.0",
                                        parameter_count=kwargs.get('parameter_count', 5000),
                                        kb_size=kwargs.get('kb_size', "1.2GB"),
                                        cycle_number=cycle_number,
                                        **kwargs)
        
        # Optimization templates
        pca_optimization = self.render('pca_optimization',
                                     vector_count=kwargs.get('vector_count', 10000),
                                     current_dimensions=kwargs.get('current_dimensions', 1024),
                                     **kwargs)
        
        hnsw_optimization = self.render('hnsw_optimization',
                                      M_value=kwargs.get('M_value', 16),
                                      ef_construction=kwargs.get('ef_construction', 200),
                                      ef_search=kwargs.get('ef_search', 128),
                                      vector_dimensions=kwargs.get('vector_dimensions', 1024),
                                      **kwargs)
        
        return {
            'sleep_phase': sleep_phase,
            'ewc_assessment': ewc_assessment,
            'version_checkpoint': version_checkpoint,
            'pca_optimization': pca_optimization,
            'hnsw_optimization': hnsw_optimization
        }
    
    def get_agent_for_content_action(self, action_type: str, **kwargs) -> str:
        """
        Get the appropriate agent prompt for a content action.
        
        Args:
            action_type: Type of action ('expand', 'merge', 'split', 'prune')
            **kwargs: Parameters for the agent template
            
        Returns:
            Rendered agent prompt
        """
        agent_map = {
            'expand': 'expansion_agent',
            'merge': 'merge_agent',
            'split': 'split_agent',
            'prune': 'prune_agent'
        }
        
        template_name = agent_map.get(action_type.lower(), 'expansion_agent')
        return self.render(template_name, **kwargs)


# Helper functions for specific operations

def get_prompt_template_manager() -> PromptTemplateManager:
    """Get the singleton instance of PromptTemplateManager."""
    return PromptTemplateManager()

def render_template(template_name: str, **kwargs) -> str:
    """
    Utility function to render a template with the given arguments.
    
    Args:
        template_name: Name of the template to render
        **kwargs: Values to substitute in the template
        
    Returns:
        The rendered template as a string
    """
    manager = get_prompt_template_manager()
    return manager.render(template_name, **kwargs)

def get_resource_appropriate_template(template_type: str, 
                                    resource_level: str = "standard") -> str:
    """
    Get the appropriate template for the given resource level.
    
    Args:
        template_type: The type of template needed
        resource_level: Resource constraint level ('minimal', 'standard', 'full')
        
    Returns:
        Name of the appropriate template
    """
    manager = get_prompt_template_manager()
    return manager.get_template_for_resource_level(template_type, resource_level)

def create_template_for_phase(phase: str) -> str:
    """
    Create appropriate template for the current operational phase.
    
    Args:
        phase: Operational phase ('awake' or 'sleep')
        
    Returns:
        Template name appropriate for the phase
    """
    if phase.lower() == 'awake':
        return 'awake_phase'
    elif phase.lower() == 'sleep':
        return 'sleep_phase_enrichment'
    else:
        return 'rag_system'

def query_pandas(df, question: str):
    """
    Query a Pandas DataFrame using LlamaIndex's PandasQueryEngine
    and return a DataFrame-like result.
    
    Args:
        df: Pandas DataFrame to query
        question: Natural language question to ask about the data
        
    Returns:
        Response containing DataFrame result and explanation
    """
    pqe = PandasQueryEngine(df)
    response = pqe.query(question)
    return response

def format_dataframe_for_prompt(df, max_rows: int = 10, max_cols: int = 5) -> str:
    """
    Format a pandas DataFrame for inclusion in a prompt with appropriate size limits.
    
    Args:
        df: DataFrame to format
        max_rows: Maximum number of rows to include
        max_cols: Maximum number of columns to include
        
    Returns:
        Formatted DataFrame as string
    """
    if df is None or df.empty:
        return "No data available"
    
    display_df = df.iloc[:max_rows, :max_cols]
    
    rows_truncated = df.shape[0] > max_rows
    cols_truncated = df.shape[1] > max_cols
    
    result = display_df.to_string()
    
    if rows_truncated or cols_truncated:
        notes = []
        if rows_truncated:
            notes.append(f"Showing {max_rows} of {df.shape[0]} rows")
        if cols_truncated:
            notes.append(f"Showing {max_cols} of {df.shape[1]} columns")
        
        result += f"\n\nNote: {'; '.join(notes)}"
    
    return result

def format_chunks_for_prompt(chunks: List[Dict], max_chars_per_chunk: int = 500) -> str:
    """
    Format retrieved chunks for inclusion in a prompt.
    
    Args:
        chunks: List of chunk dictionaries from the RAG engine
        max_chars_per_chunk: Maximum characters to include per chunk
        
    Returns:
        Formatted chunks as string
    """
    if not chunks:
        return "No relevant information found."
    
    formatted_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', 'No text available')
        source = chunk.get('source', 'Unknown source')
        similarity = chunk.get('similarity', 0.0)
        
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        
        formatted_chunk = f"CHUNK {i} [source: {source}, relevance: {similarity:.2f}]:\n{text}\n"
        formatted_chunks.append(formatted_chunk)
    
    return "\n".join(formatted_chunks)

def create_agent_prompt(agent_type: str, **kwargs) -> str:
    """
    Create a prompt for a specific agent type.
    
    Args:
        agent_type: Type of agent ('expansion', 'merge', 'split', 'prune')
        **kwargs: Agent-specific parameters
        
    Returns:
        Rendered agent prompt
    """
    manager = get_prompt_template_manager()
    
    template_map = {
        'expansion': 'expansion_agent',
        'merge': 'merge_agent',
        'split': 'split_agent',
        'prune': 'prune_agent'
    }
    
    template_name = template_map.get(agent_type.lower(), 'expansion_agent')
    return manager.render(template_name, **kwargs)

def create_vector_operation_prompt(operation_type: str, resource_level: str = "standard", **kwargs) -> str:
    """
    Create a prompt for vector operations with appropriate optimization level.
    
    Args:
        operation_type: Type of vector operation ('pca', 'batch', 'similarity')
        resource_level: Resource level ('minimal', 'standard', 'full')
        **kwargs: Operation-specific parameters
        
    Returns:
        Rendered vector operation prompt
    """
    manager = get_prompt_template_manager()
    template_name = manager.get_template_for_vector_operation(operation_type, resource_level)
    return manager.render(template_name, operation_type=operation_type, **kwargs)

def create_recursive_improvement_cycle(cycle_number: int, **kwargs) -> Dict[str, str]:
    """
    Create templates for a complete recursive improvement cycle.
    
    Args:
        cycle_number: The current improvement cycle number
        **kwargs: Parameters for the templates
        
    Returns:
        Dictionary of rendered templates
    """
    manager = get_prompt_template_manager()
    return manager.create_recursive_improvement_cycle(cycle_number, **kwargs)

def get_agent_for_action(action_type: str, **kwargs) -> str:
    """
    Get the appropriate agent prompt based on action type.
    
    Args:
        action_type: Type of action ('expand', 'merge', 'split', 'prune')
        **kwargs: Agent-specific parameters
        
    Returns:
        Rendered agent prompt
    """
    manager = get_prompt_template_manager()
    return manager.get_agent_for_content_action(action_type, **kwargs)

def create_optimization_templates(vector_count: int, dimensions: int, **kwargs) -> Dict[str, str]:
    """
    Create optimization templates for vector operations.
    
    Args:
        vector_count: Number of vectors in the database
        dimensions: Current vector dimensions
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with rendered optimization templates
    """
    manager = get_prompt_template_manager()
    
    return {
        'pca': manager.render('pca_optimization', 
                            vector_count=vector_count, 
                            current_dimensions=dimensions,
                            **kwargs),
        'hnsw': manager.render('hnsw_optimization',
                             vector_dimensions=dimensions,
                             **kwargs)
    }
