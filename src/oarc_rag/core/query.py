"""
Query formulation for effective RAG retrieval.

This module provides utilities to generate and optimize queries for
the RAG system to improve retrieval quality and support data analysis.
"""
import re
import json
import time
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple

import pandas as pd
from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate
from llama_index.experimental.query_engine import PandasQueryEngine

from oarc_rag.utils.log import log
from oarc_rag.ai.client import OllamaClient
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.core.cache import cache_manager, QueryCache
from oarc_rag.ai.agent import OperationalMode


class QueryFormulator:
    """
    Formulates effective queries for RAG retrieval.
    
    This class provides methods to generate queries that will
    yield the most relevant results from the vector database and
    supports DataFrame-specific query formulation. It implements
    concepts from both Specification.md and Big_Brain.md, including
    domain-specific templates and operational modes.
    """
    
    def __init__(
        self, 
        default_modifiers: Optional[Dict[str, str]] = None,
        operational_mode: Union[str, OperationalMode] = OperationalMode.AWAKE,
        query_cache_enabled: bool = True
    ):
        """
        Initialize the query formulator.
        
        Args:
            default_modifiers: Optional dict of query type to query modifier templates
            operational_mode: Operational mode (awake or sleep)
            query_cache_enabled: Whether to cache previously formulated queries
        """
        self.default_modifiers = default_modifiers or {
            "general": "detailed information about {topic}",
            "definition": "define {topic} and explain key concepts",
            "explanation": "detailed explanation of {topic} with examples",
            "comparison": "compare different aspects of {topic}",
            "analysis": "analytical breakdown of {topic} with key insights",
            # Data-focused query types
            "data_analysis": "analyze {topic} data and extract insights",
            "data_summary": "summarize key statistics and patterns in {topic} data",
            "data_comparison": "compare different aspects of {topic} data",
            "trend_analysis": "identify trends and patterns in {topic} data over time",
            "recommendations": "recommend actions based on {topic} data analysis",
            # Advanced cognitive modes from Big_Brain.md
            "reflective": "deep analytical understanding of {topic} with connections to related concepts",
            "consolidation": "comprehensive synthesis of knowledge about {topic} organized systematically",
            "expansion": "expanded exploration of {topic} incorporating adjacent concepts and applications",
            "pruning": "focused essential information about {topic} without tangential details"
        }
        
        # Category-specific template overrides
        self.category_templates = {
            "technical": {
                "general": "technical details and specifications about {topic}",
                "explanation": "technical explanation of {topic} with implementation details"
            },
            "business": {
                "general": "business implications and applications of {topic}",
                "analysis": "business analysis of {topic} with market considerations"
            },
            "scientific": {
                "general": "scientific principles and research related to {topic}",
                "analysis": "scientific analysis of {topic} with methodological considerations"
            },
            "medical": {
                "general": "medical information and clinical relevance of {topic}",
                "analysis": "clinical analysis of {topic} with diagnostic considerations"
            },
            "legal": {
                "general": "legal frameworks and regulations concerning {topic}",
                "analysis": "legal analysis of {topic} with precedent considerations"
            }
        }
        
        # Common domain-specific terms by field to enhance queries
        self.domain_terms = {
            "technical": ["implementation", "architecture", "system", "framework", "protocol"],
            "scientific": ["research", "experiment", "methodology", "analysis", "hypothesis"],
            "business": ["strategy", "market", "revenue", "customer", "solution"],
            "medical": ["treatment", "diagnosis", "condition", "patient", "clinical"],
            "legal": ["regulation", "compliance", "statute", "ruling", "jurisdiction"]
        }
        
        # Initialize LlamaIndex custom prompt templates
        self.llama_prompt_templates = {}
        self._setup_llama_templates()

        # Set operational mode (from Big_Brain.md concepts)
        if isinstance(operational_mode, str):
            try:
                self.operational_mode = OperationalMode(operational_mode.lower())
            except ValueError:
                log.warning(f"Invalid operational mode: {operational_mode}. Defaulting to AWAKE.")
                self.operational_mode = OperationalMode.AWAKE
        else:
            self.operational_mode = operational_mode
            
        # Query formulation stats for self-improvement
        self.formulation_stats = {
            "queries_formulated": 0,
            "query_types_used": {},
            "domain_distribution": {},
            "total_formulation_time": 0.0,
            "avg_formulation_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        # Memory for recurring patterns (for Big_Brain.md's sleep phase optimization)
        self.recurring_patterns = {
            "topics": set(),
            "effective_terms": {},
            "quality_scores": {},  # Track how well queries perform
        }
        
        # Query cache integration
        self.query_cache_enabled = query_cache_enabled
        if query_cache_enabled:
            self.query_cache = cache_manager.query_cache
            log.debug("Query cache enabled for QueryFormulator")
        
        # Initialize Ollama client for advanced query operations (lazy-loaded)
        self.client = None
        
        log.info(f"QueryFormulator initialized with mode: {self.operational_mode.value}")
    
    def _setup_llama_templates(self):
        """Set up LlamaIndex custom prompt templates for advanced querying."""
        try:
            # Create custom templates for different query types
            general_template = """
            Given data about {topic}, 
            analyze and provide a comprehensive response.
            
            Data schema: {schema}
            
            Query: {query_str}
            """
            
            analysis_template = """
            Given the following DataFrame:
            {df_str}
            
            Schema information:
            {schema}
            
            Please analyze the data to answer: {query_str}
            
            Provide a detailed analysis with insights and, when appropriate, include:
            1. Key statistics
            2. Notable patterns or trends
            3. Significant correlations or relationships
            4. Actionable recommendations
            """
            
            # Template for sleep mode deep query formulation (from Big_Brain.md)
            sleep_mode_template = """
            I need to create an optimal query formulation for deep knowledge retrieval
            on the topic of: {topic}
            
            Query type: {query_type}
            
            Additional context:
            {additional_context}
            
            Historical pattern information:
            {pattern_info}
            
            Please formulate a comprehensive query that:
            1. Captures essential concepts related to the topic
            2. Includes relevant domain-specific terminology
            3. Incorporates appropriate synonyms and related terms
            4. Uses precise language to maximize retrieval quality
            5. Addresses potential ambiguities or polysemy
            
            The query should be structured to retrieve in-depth, authoritative information
            that consolidates knowledge on this topic.
            """
            
            self.llama_prompt_templates = {
                "general": LlamaPromptTemplate(general_template),
                "data_analysis": LlamaPromptTemplate(analysis_template),
                "sleep_mode": LlamaPromptTemplate(sleep_mode_template)
            }
            
            log.info("Initialized LlamaIndex prompt templates")
        except Exception as e:
            log.warning(f"Failed to initialize LlamaIndex templates: {e}")
    
    def get_llama_prompt_template(self, query_type: str):
        """
        Get LlamaIndex prompt template for a specific query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            LlamaPromptTemplate or None if not available
        """
        # Map query types to template types
        template_map = {
            "general": "general", 
            "definition": "general",
            "explanation": "general",
            "comparison": "general",
            "analysis": "general",
            "data_analysis": "data_analysis",
            "data_summary": "data_analysis",
            "data_comparison": "data_analysis",
            "trend_analysis": "data_analysis",
            "recommendations": "data_analysis",
            "reflective": "sleep_mode",
            "consolidation": "sleep_mode",
            "expansion": "sleep_mode",
            "pruning": "general"
        }
        
        template_type = template_map.get(query_type, "general")
        return self.llama_prompt_templates.get(template_type)
    
    def formulate_pandas_query(self, 
                              question: str, 
                              df: pd.DataFrame,
                              context: Optional[Dict[str, Any]] = None) -> str:
        """
        Formulate a query optimized for PandasQueryEngine.
        
        Args:
            question: Natural language question about the data
            df: DataFrame to query
            context: Additional context to enhance the query
            
        Returns:
            str: Formulated query for PandasQueryEngine
        """
        # Extract DataFrame structure information
        column_info = "\n".join([
            f"- {col} ({df[col].dtype}): {df[col].nunique()} unique values" 
            for col in df.columns
        ])
        
        schema_info = (
            f"DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.\n"
            f"Column information:\n{column_info}"
        )
        
        # Enhance the question with DataFrame metadata
        enhanced_query = (
            f"{question}\n\n"
            f"Consider this DataFrame structure:\n{schema_info}"
        )
        
        # Add any additional context
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            enhanced_query += f"\n\nAdditional context:\n{context_str}"
        
        log.debug(f"Formulated pandas query: {enhanced_query[:100]}...")
        return enhanced_query
    
    def execute_pandas_query(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a natural language query against a pandas DataFrame using PandasQueryEngine.
        
        Args:
            question: Natural language question
            df: DataFrame to query
            
        Returns:
            Dict with query results and metadata
        """
        try:
            # Create the query engine
            query_engine = PandasQueryEngine(df)
            
            # Execute the query
            response = query_engine.query(question)
            
            # Process the response
            result = {
                "query": question,
                "response": response,
                "success": True,
                "error": None
            }
            
            # Try to extract DataFrame result if available
            try:
                if hasattr(response, "metadata") and "pandas_df_result" in response.metadata:
                    result["df_result"] = response.metadata["pandas_df_result"]
            except:
                pass
                
            return result
            
        except Exception as e:
            log.error(f"Error executing pandas query: {str(e)}")
            return {
                "query": question,
                "response": None,
                "success": False,
                "error": str(e)
            }
    
    def formulate_query(
        self,
        topic: str,
        query_type: str = "general",
        additional_context: Optional[Dict[str, Any]] = None,
        use_llm_enhancement: bool = False
    ) -> str:
        """
        Formulate a query for effective RAG retrieval.
        
        Args:
            topic: The main topic of interest
            query_type: Type of query (general, definition, explanation, etc.)
            additional_context: Optional additional context to include
            use_llm_enhancement: Whether to use LLM to enhance the query in sleep mode
            
        Returns:
            str: Formulated query
        """
        start_time = time.time()
        
        # Check cache first if enabled
        cache_key = f"{topic}_{query_type}_{json.dumps(additional_context or {})}"
        if self.query_cache_enabled:
            cached_query = self.query_cache.get(cache_key)
            if cached_query:
                log.debug(f"Using cached query for: {topic}, type: {query_type}")
                return cached_query
        
        # Determine domain area for topic-specific templates
        domain = self._detect_domain(topic)
        
        # Get domain-specific template if available, otherwise use default
        if (domain and domain in self.category_templates and 
            query_type in self.category_templates[domain]):
            template = self.category_templates[domain][query_type]
        else:
            # Get base template
            template = self.default_modifiers.get(
                query_type.lower(), 
                "information about {topic}"
            )
        
        # Format with basic information
        query = template.format(topic=topic)
        
        # Add domain-specific terms to enhance query relevance
        if domain and domain in self.domain_terms:
            domain_keywords = " ".join(f"{term}" for term in self.domain_terms[domain][:3])
            query += f" including {domain_keywords}"
        
        # Add additional context if provided
        if additional_context:
            context_parts = []
            for key, value in additional_context.items():
                if isinstance(value, str) and value.strip():
                    context_parts.append(f"{key}: {value}")
            
            if context_parts:
                query += " " + " ".join(context_parts)
        
        # In sleep mode, perform more extensive query enhancement if requested
        if use_llm_enhancement and self.operational_mode == OperationalMode.SLEEP:
            enhanced_query = self._generate_enhanced_query(topic, query_type, additional_context)
            if enhanced_query:
                query = enhanced_query
        
        # Apply operational mode-specific optimizations (from Big_Brain.md)
        if self.operational_mode == OperationalMode.SLEEP:
            # In sleep mode, expand query with broader terms for comprehensive retrieval
            query = self._apply_sleep_mode_enhancements(query, topic, domain)
        else:
            # In awake mode, focus query on most relevant aspects for speed
            query = self._apply_awake_mode_optimization(query)
        
        # Update statistics
        self.formulation_stats["queries_formulated"] += 1
        if query_type in self.formulation_stats["query_types_used"]:
            self.formulation_stats["query_types_used"][query_type] += 1
        else:
            self.formulation_stats["query_types_used"][query_type] = 1
            
        if domain:
            if domain in self.formulation_stats["domain_distribution"]:
                self.formulation_stats["domain_distribution"][domain] += 1
            else:
                self.formulation_stats["domain_distribution"][domain] = 1
        
        # Track patterns for future optimization
        self.recurring_patterns["topics"].add(topic.lower())
        
        # Update timing statistics
        elapsed = time.time() - start_time
        self.formulation_stats["total_formulation_time"] += elapsed
        self.formulation_stats["avg_formulation_time"] = (
            self.formulation_stats["total_formulation_time"] / 
            self.formulation_stats["queries_formulated"]
        )
        
        # Cache the result if caching is enabled
        if self.query_cache_enabled:
            self.query_cache.set(cache_key, query)
        
        log.debug(f"Formulated query: {query}")
        return query
    
    def _apply_sleep_mode_enhancements(self, query: str, topic: str, domain: Optional[str]) -> str:
        """
        Apply sleep mode enhancements to query for comprehensive retrieval.
        
        Args:
            query: Original formulated query
            topic: Query topic
            domain: Detected domain
            
        Returns:
            Enhanced query for sleep mode
        """
        # Add synonyms for key terms
        synonyms = self._get_topic_synonyms(topic)
        if synonyms:
            synonym_str = " OR ".join([f'"{syn}"' for syn in synonyms[:3]])
            query = f"{query} ({synonym_str})"
        
        # Add related concepts for broader context
        if domain and domain in self.recurring_patterns.get("effective_terms", {}):
            effective_terms = self.recurring_patterns["effective_terms"][domain]
            if effective_terms:
                terms_str = " ".join(effective_terms[:3])
                query = f"{query} in relation to {terms_str}"
        
        return query
    
    def _apply_awake_mode_optimization(self, query: str) -> str:
        """
        Apply awake mode optimizations to query for faster retrieval.
        
        Args:
            query: Original formulated query
            
        Returns:
            Optimized query for awake mode
        """
        # Focus on keywords and remove filler words for faster retrieval
        keywords = self._extract_keywords(query)
        keyword_query = " ".join(keywords)
        
        # Keep original query but optimize with extracted keywords
        return f"{query} {keyword_query}"
    
    def _generate_enhanced_query(
        self, 
        topic: str, 
        query_type: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generate an enhanced query using LLM in sleep mode for better results.
        
        Args:
            topic: Query topic
            query_type: Type of query
            additional_context: Additional context
            
        Returns:
            Enhanced query or None if generation failed
        """
        if not self.client:
            try:
                self.client = OllamaClient()
            except Exception as e:
                log.warning(f"Failed to initialize OllamaClient: {e}")
                return None
        
        # Create pattern info from recurring patterns
        pattern_info = {}
        if topic.lower() in self.recurring_patterns["topics"]:
            pattern_info["topic_frequency"] = "recurring"
        else:
            pattern_info["topic_frequency"] = "new"
            
        domain = self._detect_domain(topic)
        if domain and domain in self.formulation_stats["domain_distribution"]:
            pattern_info["domain_frequency"] = self.formulation_stats["domain_distribution"][domain]
            
        pattern_info_str = "\n".join([f"{k}: {v}" for k, v in pattern_info.items()])
        
        # Format prompt for LLM 
        prompt = f"""
        I need to create an optimal query formulation for deep knowledge retrieval
        on the topic of: {topic}
        
        Query type: {query_type}
        
        Additional context:
        {json.dumps(additional_context or {})}
        
        Historical pattern information:
        {pattern_info_str}
        
        Please formulate a comprehensive query that:
        1. Captures essential concepts related to the topic
        2. Includes relevant domain-specific terminology
        3. Incorporates appropriate synonyms and related terms
        4. Uses precise language to maximize retrieval quality
        5. Addresses potential ambiguities or polysemy
        
        Return only the optimized query text with no additional explanation or formatting.
        """
        
        try:
            response = self.client.generate(prompt=prompt)
            
            # Extract query from response (remove common prefixes LLMs might add)
            query = response.strip()
            for prefix in ["Query:", "Enhanced query:", "Optimized query:", "Formulated query:"]:
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
            
            log.debug(f"Generated enhanced query for {topic}: {query}")
            return query
        except Exception as e:
            log.warning(f"Failed to generate enhanced query: {e}")
            return None
            
    def _get_topic_synonyms(self, topic: str) -> List[str]:
        """
        Get synonyms for a topic to enhance query.
        
        Args:
            topic: Topic to find synonyms for
            
        Returns:
            List of synonyms
        """
        # This would ideally use a thesaurus API but for now returns simple alternatives
        # In a real implementation, this would use WordNet or similar resource
        
        # Simple predefined synonyms for common terms
        synonym_map = {
            "programming": ["coding", "development", "software engineering"],
            "machine learning": ["ML", "AI learning", "statistical learning"],
            "artificial intelligence": ["AI", "machine intelligence", "cognitive computing"],
            "database": ["DB", "data store", "data repository"],
            "algorithm": ["method", "procedure", "computation process"],
            "framework": ["library", "platform", "toolkit"]
        }
        
        lower_topic = topic.lower()
        
        # Check direct matches
        if lower_topic in synonym_map:
            return synonym_map[lower_topic]
        
        # Check partial matches
        for key, synonyms in synonym_map.items():
            if key in lower_topic:
                return synonyms
                
        return []
    
    def formulate_multi_queries(
        self,
        topic: str,
        query_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Formulate multiple queries for different aspects of a topic.
        
        Args:
            topic: The main topic of interest
            query_types: List of query types to generate (defaults to all)
            
        Returns:
            Dict[str, str]: Dictionary of query_type to formulated query
        """
        query_types = query_types or list(self.default_modifiers.keys())
        
        result = {}
        for query_type in query_types:
            query = self.formulate_query(topic, query_type)
            result[query_type] = query
            
        return result
    
    def expand_query(self, query: str, subtopics: List[str]) -> List[str]:
        """
        Expand a query into multiple related queries.
        
        Args:
            query: Base query
            subtopics: List of subtopics to include
            
        Returns:
            List[str]: List of expanded queries
        """
        expanded = [query]  # Start with the base query
        
        # Add a query for each subtopic
        for subtopic in subtopics:
            expanded.append(f"{query} focused on {subtopic}")
            
        return expanded
    
    def translate_to_dataframe_query(self, 
                                    natural_query: str, 
                                    df_description: str = "") -> Dict[str, Any]:
        """
        Translate a natural language query into a structure suitable for DataFrame analysis.
        
        Args:
            natural_query: Natural language query
            df_description: Description of the DataFrame
            
        Returns:
            Dict with query components for DataFrame operations
        """
        # Extract operations often needed in DataFrame analysis
        operations = {
            "filter": self._extract_filter_conditions(natural_query),
            "group": self._extract_grouping(natural_query),
            "sort": self._extract_sorting(natural_query),
            "aggregate": self._extract_aggregation(natural_query),
            "limit": self._extract_limit(natural_query),
        }
        
        # Determine the query intent
        intent = self._determine_query_intent(natural_query)
        
        return {
            "original_query": natural_query,
            "operations": operations,
            "intent": intent,
            "df_description": df_description
        }
    
    def _detect_domain(self, topic: str) -> Optional[str]:
        """
        Detect the domain area of a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Optional[str]: Detected domain or None
        """
        topic_lower = topic.lower()
        
        # Simple keyword-based domain detection
        domain_keywords = {
            "technical": ["software", "programming", "technology", "system", "computer", "engineering", "hardware", "network"],
            "scientific": ["science", "research", "biology", "chemistry", "physics", "experiment", "laboratory"],
            "business": ["business", "marketing", "finance", "economics", "management", "company", "market"],
            "medical": ["medical", "health", "disease", "treatment", "doctor", "patient", "clinical"],
            "legal": ["legal", "law", "regulation", "compliance", "court", "judge", "attorney"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
                
        return None
    
    def _extract_filter_conditions(self, query: str) -> List[str]:
        """Extract potential filter conditions from natural query."""
        filter_patterns = [
            r"where\s+(\w+\s+(?:>|<|>=|<=|==|!=)\s+[\w\.]+)",
            r"with\s+(\w+\s+(?:>|<|>=|<=|==|!=)\s+[\w\.]+)",
            r"(?:equals?|equal\s+to|same\s+as)\s+([\w\.]+)",
            r"(?:greater|more|higher)\s+than\s+([\w\.]+)",
            r"(?:less|lower|smaller)\s+than\s+([\w\.]+)"
        ]
        
        conditions = []
        for pattern in filter_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            conditions.extend(matches)
            
        return conditions
    
    def _extract_grouping(self, query: str) -> List[str]:
        """Extract grouping columns from natural query."""
        group_patterns = [
            r"group\s+by\s+([\w\s,]+)",
            r"grouped\s+by\s+([\w\s,]+)",
            r"for\s+each\s+([\w\s,]+)"
        ]
        
        groups = []
        for pattern in group_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                groups.extend([col.strip() for col in match.split(",")])
                
        return groups
    
    def _extract_sorting(self, query: str) -> List[Dict[str, str]]:
        """Extract sorting information from natural query."""
        sort_patterns = [
            r"(?:sort|order)\s+by\s+([\w\s,]+)\s+(asc|ascending|desc|descending)",
            r"(?:sort|order)\s+by\s+([\w\s,]+)"
        ]
        
        sorting = []
        for pattern in sort_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:  # Column and direction
                    cols, direction = match
                    direction = "asc" if "asc" in direction.lower() else "desc"
                else:  # Just column, default ascending
                    cols = match[0]
                    direction = "asc"
                    
                for col in cols.split(","):
                    sorting.append({"column": col.strip(), "direction": direction})
                    
        return sorting
    
    def _extract_aggregation(self, query: str) -> List[Dict[str, str]]:
        """Extract aggregation operations from natural query."""
        agg_patterns = [
            r"(sum|average|avg|mean|max|maximum|min|minimum|count)\s+of\s+([\w\s]+)",
            r"(sum|average|avg|mean|max|maximum|min|minimum|count)\s+([\w\s]+)"
        ]
        
        aggregations = []
        for pattern in agg_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                op, col = match
                
                # Normalize operation names
                if op.lower() in ["avg", "average", "mean"]:
                    op = "mean"
                elif op.lower() in ["max", "maximum"]:
                    op = "max"
                elif op.lower() in ["min", "minimum"]:
                    op = "min"
                
                aggregations.append({"operation": op.lower(), "column": col.strip()})
                
        return aggregations
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract result limit from natural query."""
        limit_patterns = [
            r"(?:show|return|get|limit)\s+(?:the\s+)?(?:top|first)\s+(\d+)",
            r"limit\s+(\d+)"
        ]
        
        for pattern in limit_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches and matches[0].isdigit():
                return int(matches[0])
                
        return None
    
    def _determine_query_intent(self, query: str) -> str:
        """
        Determine the primary intent of the query.
        
        Args:
            query: Natural language query
            
        Returns:
            str: Query intent category
        """
        query = query.lower()
        
        if any(term in query for term in ["trend", "change over time", "pattern", "evolution"]):
            return "trend_analysis"
        elif any(term in query for term in ["compare", "difference between", "versus", "vs"]):
            return "comparison"
        elif any(term in query for term in ["average", "maximum", "minimum", "sum", "count"]):
            return "aggregation"
        elif any(term in query for term in ["predict", "forecast", "estimate"]):
            return "prediction"
        elif any(term in query for term in ["group", "categorize", "segment"]):
            return "grouping"
        elif any(term in query for term in ["filter", "where", "only", "just"]):
            return "filtering"
        elif any(term in query for term in ["summarize", "summary", "overview"]):
            return "summary"
        elif any(term in query for term in ["recommend", "suggest", "best"]):
            return "recommendation"
        else:
            return "information"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for query optimization.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Extracted keywords
        """
        stopwords = {'the', 'and', 'is', 'in', 'at', 'of', 'for', 'with', 'by', 'to', 'a', 'an'}
        words = re.findall(r'\b\w+\b', text.lower())
        content_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = [word for word in content_words if not (word in seen or seen.add(word))]
        
        return unique_words
    
    def optimize_query_for_retrieval(
        self, 
        query: str, 
        query_purpose: str = "semantic", 
        max_length: int = 100
    ) -> str:
        """
        Optimize a query for specific retrieval purposes.
        
        Args:
            query: Original query
            query_purpose: Purpose of optimization ("semantic", "keyword", "hybrid")
            max_length: Maximum length of optimized query
            
        Returns:
            str: Optimized query
        """
        if query_purpose == "keyword":
            # Extract and focus on keywords for keyword-based retrieval
            keywords = self._extract_keywords(query)
            return " ".join(keywords)[:max_length]
            
        elif query_purpose == "hybrid":
            # Combine original query with extracted keywords
            keywords = self._extract_keywords(query)
            keyword_str = " ".join(keywords)
            
            # Build hybrid query but respect max length
            hybrid = f"{query} {keyword_str}"
            if len(hybrid) > max_length:
                return hybrid[:max_length]
            return hybrid
            
        else:  # semantic - default
            # Just trim if too long
            if len(query) > max_length:
                return query[:max_length]
            return query
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get formulation statistics.
        
        Returns:
            Dict[str, Any]: Statistics about query formulation
        """
        return {
            "stats": self.formulation_stats,
            "patterns": {
                "topics_tracked": len(self.recurring_patterns["topics"]),
                "effective_terms_count": len(self.recurring_patterns.get("effective_terms", {})),
                "quality_scores_count": len(self.recurring_patterns.get("quality_scores", {}))
            },
            "operational_mode": self.operational_mode.value
        }
            
    def set_operational_mode(self, mode: Union[str, OperationalMode]) -> None:
        """
        Set the operational mode for query formulation.
        
        Args:
            mode: New operational mode (awake or sleep)
        """
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = OperationalMode(mode.lower())
            except ValueError:
                log.warning(f"Invalid operational mode: {mode}. Defaulting to AWAKE.")
                mode = OperationalMode.AWAKE
                
        # Skip if already in this mode
        if self.operational_mode == mode:
            return
            
        # Update mode
        prev_mode = self.operational_mode
        self.operational_mode = mode
        
        log.info(f"QueryFormulator mode changed: {prev_mode.value} → {mode.value}")
            
    def record_query_effectiveness(
        self, 
        query: str, 
        effectiveness_score: float, 
        domain: Optional[str] = None
    ) -> None:
        """
        Record effectiveness of a query for self-improvement.
        
        Args:
            query: The query used
            effectiveness_score: Score between 0.0 and 1.0 indicating effectiveness
            domain: Optional domain the query pertains to
        """
        # Extract keywords from effective queries
        if effectiveness_score > 0.7:  # Consider 0.7+ as effective
            self.formulation_stats["successful_queries"] += 1
            
            # Extract and store effective terms
            keywords = self._extract_keywords(query)
            if domain and keywords:
                if domain not in self.recurring_patterns.get("effective_terms", {}):
                    self.recurring_patterns.setdefault("effective_terms", {})[domain] = []
                
                # Add effective terms without duplicates
                current_terms = set(self.recurring_patterns["effective_terms"].get(domain, []))
                for keyword in keywords:
                    if keyword not in current_terms and len(keyword) > 2:
                        self.recurring_patterns["effective_terms"].setdefault(domain, []).append(keyword)
                        current_terms.add(keyword)
        else:
            self.formulation_stats["failed_queries"] += 1
            
        # Store quality score
        query_hash = hash(query)
        self.recurring_patterns.setdefault("quality_scores", {})[query_hash] = effectiveness_score
        
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if hasattr(self, 'query_cache'):
            self.query_cache.clear()
            log.info("QueryFormulator cache cleared")
            
    def reset_stats(self) -> None:
        """Reset formulation statistics."""
        self.formulation_stats = {
            "queries_formulated": 0,
            "query_types_used": {},
            "domain_distribution": {},
            "total_formulation_time": 0.0,
            "avg_formulation_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        log.info("QueryFormulator statistics reset")
