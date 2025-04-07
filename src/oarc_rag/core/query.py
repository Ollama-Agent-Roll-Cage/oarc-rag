"""
Query formulation for effective RAG retrieval.

This module provides utilities to generate and optimize queries for
the RAG system to improve retrieval quality and support data analysis.
"""
import re
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate
from llama_index.experimental.query_engine import PandasQueryEngine

from oarc_rag.utils.log import log


class QueryFormulator:
    """
    Formulates effective queries for RAG retrieval.
    
    This class provides methods to generate queries that will
    yield the most relevant results from the vector database and
    supports DataFrame-specific query formulation.
    """
    
    def __init__(self, default_modifiers: Optional[Dict[str, str]] = None):
        """
        Initialize the query formulator.
        
        Args:
            default_modifiers: Optional dict of query type to query modifier templates
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
            
            self.llama_prompt_templates = {
                "general": LlamaPromptTemplate(general_template),
                "data_analysis": LlamaPromptTemplate(analysis_template)
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
            "recommendations": "data_analysis"
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
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Formulate a query for effective RAG retrieval.
        
        Args:
            topic: The main topic of interest
            query_type: Type of query (general, definition, explanation, etc.)
            additional_context: Optional additional context to include
            
        Returns:
            str: Formulated query
        """
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
        
        log.debug(f"Formulated query: {query}")
        return query
    
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
        
        match query:
            case _ if any(term in query for term in ["trend", "change over time", "pattern", "evolution"]):
                return "trend_analysis"
            case _ if any(term in query for term in ["compare", "difference between", "versus", "vs"]):
                return "comparison"
            case _ if any(term in query for term in ["average", "maximum", "minimum", "sum", "count"]):
                return "aggregation"
            case _ if any(term in query for term in ["predict", "forecast", "estimate"]):
                return "prediction"
            case _ if any(term in query for term in ["group", "categorize", "segment"]):
                return "grouping"
            case _ if any(term in query for term in ["filter", "where", "only", "just"]):
                return "filtering"
            case _ if any(term in query for term in ["summarize", "summary", "overview"]):
                return "summary"
            case _ if any(term in query for term in ["recommend", "suggest", "best"]):
                return "recommendation"
            case _:
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
