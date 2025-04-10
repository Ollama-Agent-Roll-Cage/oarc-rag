"""
Prompt templates for domain-agnostic RAG generation.

This module provides advanced templating capabilities for AI prompts using Jinja2,
supporting template loading, formatting, validation and versioning.
Templates are designed to work across any domain, not tied to specific applications.
"""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape, BaseLoader, Template
from llama_index.experimental.query_engine import PandasQueryEngine

from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log
from oarc_rag.utils.config.config import Config
from oarc_rag.core.cache import cache_manager, TemplateCache


@singleton
class PromptTemplateManager:
    """
    Class for managing prompt templates for AI generation using Jinja2.
    """
    
    # Template for system context in RAG conversations
    RAG_SYSTEM_TEMPLATE = """
    You are an AI assistant equipped with a retrieval-augmented generation (RAG) system.
    You can access relevant documents and data to answer questions about any topic.
    
    When responding:
    - Use the retrieved context to inform your answers when available
    - Cite sources when information comes from specific documents
    - Acknowledge when you're unsure or when the retrieval system doesn't provide relevant information
    - Present information in a clear, structured way
    - Focus on giving accurate, helpful information based on the provided context
    - Maintain a {{ tone }} tone in responses
    
    Your goal is to provide accurate, helpful responses enhanced by retrieved information.
    """
    
    # Template for query reformulation to improve retrieval
    QUERY_REFORMULATION_TEMPLATE = """
    I need to find relevant information about the following query:
    "{{ original_query }}"
    
    To improve retrieval results, please convert this into a more effective search query by:
    1. Identifying key concepts and terminology
    2. Adding synonyms for important terms
    3. Focusing on specific aspects rather than general questions
    4. Including any relevant technical terms mentioned
    5. Removing unnecessary words and fillers
    
    Domain context: {{ domain_context }}
    {% if related_queries %}Previous related queries: {{ related_queries }}{% endif %}
    """
    
    # Template for presenting retrieved context in conversation
    CONTEXT_PRESENTATION_TEMPLATE = """
    Based on retrieved information:
    
    {{ formatted_chunks }}
    
    Use these relevant passages to address the following question:
    {{ query }}
    
    Provide a comprehensive answer that synthesizes information from the retrieved content.
    Cite specific sources when drawing from particular chunks.
    """
    
    # Template for data analysis results
    DATA_ANALYSIS_TEMPLATE = """
    Here's the data analysis for your query "{{ query }}":
    
    {{ formatted_data }}
    
    Key insights:
    - {{ insight_1 }}
    - {{ insight_2 }}
    - {{ insight_3 }}
    
    Would you like me to explain any specific aspect of these results in more detail?
    """
    
    # Template for summarizing lengthy chunks
    CHUNK_SUMMARIZATION_TEMPLATE = """
    Summarize the following information in a concise way while preserving key details:
    
    "{{ chunk_content }}"
    
    Create a summary that:
    1. Captures essential facts and concepts
    2. Maintains technical accuracy
    3. Is approximately {{ target_length }} words long
    4. Focuses on aspects relevant to {{ focus_area }}
    """
    
    # Template for agent handoff with context
    AGENT_HANDOFF_TEMPLATE = """
    [AGENT HANDOFF: {{ source_agent }} → {{ target_agent }}]
    
    QUERY: {{ original_query }}
    
    RETRIEVED CONTEXT:
    {{ context_summary }}
    
    ACTIONS TAKEN:
    {{ previous_actions }}
    
    NEXT STEPS:
    {{ recommended_actions }}
    
    ADDITIONAL NOTES:
    {{ notes }}
    """
    
    # Template for vector search result formatting
    SEARCH_RESULTS_TEMPLATE = """
    Based on your query "{{ query }}", I found the following relevant information:
    
    {{ formatted_results }}
    
    These results were selected based on semantic similarity to your question.
    """
    
    # Template for operational mode transition (from Big_Brain.md concepts)
    MODE_TRANSITION_TEMPLATE = """
    [System Notification: Transitioning from {{ previous_mode }} to {{ new_mode }} mode]
    
    During this transition, the system will:
    1. {{ transition_action_1 }}
    2. {{ transition_action_2 }} 
    3. {{ transition_action_3 }}
    
    This process enables continuous self-improvement through iterative refinement as
    described in the recursive self-improving RAG framework.
    
    Estimated completion time: {{ estimated_time }}
    """

    # Template for semantic reranking of search results 
    SEMANTIC_RERANKING_TEMPLATE = """
    Please analyze the following search results for relevance to the query: "{{ query }}"
    
    {% for result in results %}
    [{{ loop.index }}] {{ result.text | truncate(200) }}
    {% endfor %}
    
    Provide a ranking of the result indices from most to least relevant, considering:
    - Direct relevance to the query
    - Information quality and completeness
    - Factual accuracy
    
    Format your response exactly like this: [1, 5, 2, 4, 3]
    """
    
    # Template for knowledge consolidation during sleep phase
    KNOWLEDGE_CONSOLIDATION_TEMPLATE = """
    [SLEEP PHASE KNOWLEDGE CONSOLIDATION]
    
    Previous interactions to analyze:
    {% for interaction in interactions %}
    INTERACTION {{ loop.index }}:
    Query: {{ interaction.query }}
    Response: {{ interaction.response | truncate(200) }}
    {% endfor %}
    
    CONSOLIDATION OBJECTIVES:
    1. Identify recurring themes or topics
    2. Detect knowledge gaps in previous responses
    3. Propose improvements for future interactions
    4. Recommend knowledge areas to enhance
    
    Provide a structured analysis optimized for improving the RAG system's performance.
    """
    
    # Version tracking for templates
    TEMPLATE_VERSIONS = {
        "rag_system": "1.1",
        "query_reformulation": "1.1", 
        "context_presentation": "1.1",
        "data_analysis": "1.1",
        "chunk_summarization": "1.1",
        "agent_handoff": "1.1",
        "search_results": "1.1",
        "mode_transition": "1.1",
        "semantic_reranking": "1.0",
        "knowledge_consolidation": "1.0"
    }
    
    def __init__(self):
        """Initialize the Jinja2 template environment."""
        # Check if already initialized (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Setup the Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self._get_templates_directory()),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['json'] = lambda obj: json.dumps(obj, indent=2)
        self.env.filters['truncate'] = self._truncate_filter
        
        # Register built-in templates
        self._register_builtin_templates()
        
        # Dictionary to cache template instances by name
        self._template_instances = {}
        
        # Tracking and statistics
        self.template_usage = {}
        self._initialized = True
        
    def _get_templates_directory(self) -> Path:
        """Get or create the directory for storing template files."""
        template_dir = Config.get_base_dir() / "templates"
        os.makedirs(template_dir, exist_ok=True)
        return template_dir
    
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
            'rag_system': self.RAG_SYSTEM_TEMPLATE,
            'query_reformulation': self.QUERY_REFORMULATION_TEMPLATE,
            'context_presentation': self.CONTEXT_PRESENTATION_TEMPLATE,
            'data_analysis': self.DATA_ANALYSIS_TEMPLATE, 
            'chunk_summarization': self.CHUNK_SUMMARIZATION_TEMPLATE,
            'agent_handoff': self.AGENT_HANDOFF_TEMPLATE,
            'search_results': self.SEARCH_RESULTS_TEMPLATE,
            'mode_transition': self.MODE_TRANSITION_TEMPLATE,
            'semantic_reranking': self.SEMANTIC_RERANKING_TEMPLATE,
            'knowledge_consolidation': self.KNOWLEDGE_CONSOLIDATION_TEMPLATE
        }
        
        # Add each template to the environment
        for name, template_text in builtin_templates.items():
            self.env.globals[name] = template_text
            
            # Create and cache template instance
            template = self.env.from_string(template_text)
            self._template_instances[name] = {
                'template': template,
                'text': template_text,
                'version': self.TEMPLATE_VERSIONS.get(name, '1.0'),
                'variables': self._extract_variables(template_text),
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
        # Check if we have a cached built-in template
        if template_name in self._template_instances:
            self._template_instances[template_name]['usage_count'] += 1
            return self._template_instances[template_name]['template']
            
        # Try to load from filesystem
        try:
            return self.env.get_template(f"{template_name}.j2")
        except jinja2.exceptions.TemplateNotFound:
            valid_templates = ", ".join(self._template_instances.keys())
            raise ValueError(f"Template '{template_name}' not found. Available templates: {valid_templates}")
    
    def register_template(self, name: str, template_text: str, version: str = "custom-1.0") -> None:
        """
        Register a new template with the system.
        
        Args:
            name: Template name
            template_text: Template content
            version: Template version
            
        Raises:
            ValueError: If template is empty or invalid
        """
        if not template_text or not template_text.strip():
            raise ValueError("Cannot register empty template")
            
        try:
            # Create Jinja2 template
            template = self.env.from_string(template_text)
            
            # Extract variables
            variables = self._extract_variables(template_text)
            
            # Store in cache
            self._template_instances[name] = {
                'template': template,
                'text': template_text,
                'version': version,
                'variables': variables,
                'usage_count': 0,
                'successful_uses': 0
            }
            
            # Save to file system if it's a custom template
            if name.startswith('custom_'):
                template_path = self._get_templates_directory() / f"{name}.j2"
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_text)
            
            log.info(f"Registered template '{name}' (v{version}) with {len(variables)} variables")
            
        except jinja2.exceptions.TemplateSyntaxError as e:
            raise ValueError(f"Invalid template syntax: {str(e)}")
    
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
        try:
            template = self.get_template(template_name)
            
            # Check for missing variables
            template_info = self.template_cache.get_template(template_name)
            
            if template_info:
                # Check for missing variables
                missing_vars = template_info['variables'] - set(kwargs.keys())
                if missing_vars:
                    # Use default values for optional variables (if provided)
                    defaults = kwargs.get('defaults', {})
                    for var in list(missing_vars):
                        if var in defaults:
                            kwargs[var] = defaults[var]
                            missing_vars.remove(var)
                
                # Raise error if required variables are still missing
                if missing_vars and not kwargs.get('ignore_missing', False):
                    missing_list = ", ".join(missing_vars)
                    raise ValueError(f"Missing required template variables: {missing_list}")
            
            # Render template
            result = template.render(**kwargs)
            
            # Update statistics
            if template_info:
                self.template_cache.record_successful_use(template_name)
                
            return result
            
        except jinja2.exceptions.UndefinedError as e:
            log.error(f"Template rendering error: {str(e)}")
            raise ValueError(f"Missing template variable: {str(e)}")
        except jinja2.exceptions.TemplateSyntaxError as e:
            log.error(f"Template syntax error: {str(e)}")
            raise ValueError(f"Template syntax error: {str(e)}")
        except Exception as e:
            log.error(f"Template rendering failed: {str(e)}")
            raise ValueError(f"Failed to render template '{template_name}': {str(e)}")
    
    def from_file(self, file_path: Union[str, Path]) -> str:
        """
        Load a template from a file and register it with the system.
        
        Args:
            file_path: Path to the template file (j2, jinja, txt, or json)
            
        Returns:
            str: Template name for later reference
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
            
        try:
            # Determine file type and load accordingly
            if path.suffix.lower() in ['.json']:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                template_text = data.get('template')
                template_name = data.get('name', path.stem)
                version = data.get('version', "file-1.0")
                
                if not template_text:
                    raise ValueError(f"JSON template file must contain a 'template' field")
                    
            else:  # Default to raw text file (.j2, .jinja, .txt, etc)
                with open(path, 'r', encoding='utf-8') as f:
                    template_text = f.read()
                    
                template_name = path.stem
                version = "file-1.0"
            
            # Register the template with cache system
            template_object = self.env.from_string(template_text)
            self.template_cache.add_template(
                name=template_name,
                template_object=template_object,
                template_text=template_text,
                version=version
            )
            
            log.info(f"Loaded template '{template_name}' from {path.name}")
            return template_name
            
        except Exception as e:
            log.error(f"Failed to load template from {path}: {e}")
            raise ValueError(f"Failed to load template from {file_path}: {e}")
    
    def save_to_file(self, template_name: str, file_path: Union[str, Path], format: str = 'json') -> None:
        """
        Save a template to a file.
        
        Args:
            template_name: Name of the template to save
            file_path: Path where to save the template
            format: Format to save as ('json', 'j2', or 'txt')
            
        Raises:
            ValueError: If the template is not found or format is invalid
        """
        # Get template info from cache
        template_info = self.template_cache.get_template(template_name)
        if not template_info:
            raise ValueError(f"Template '{template_name}' not found")
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'json':
                data = {
                    'name': template_name,
                    'version': template_info['version'],
                    'template': template_info['text'],
                    'variables': list(template_info['variables']),
                    'stats': {
                        'usage_count': template_info['usage_count'],
                        'successful_uses': template_info['successful_uses']
                    }
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
            else:  # Default to raw text (j2 or txt)
                with open(path, 'w', encoding='utf-8') as f:
                    if format.lower() == 'j2':
                        # For j2 format, just write the raw template
                        f.write(template_info['text'])
                    else:
                        # For txt format, include some metadata as comments
                        f.write(f"{{% # Template: {template_name} (v{template_info['version']}) %}}\n\n")
                        f.write(template_info['text'])
                
            log.info(f"Saved template '{template_name}' to {path}")
            
        except Exception as e:
            log.error(f"Failed to save template to {path}: {e}")
            raise ValueError(f"Failed to save template: {e}")
    
    def add_examples(self, template_name: str, examples: List[Dict[str, Any]]) -> str:
        """
        Add few-shot learning examples to a template.
        
        Args:
            template_name: Template to enhance
            examples: List of example dictionaries with keys matching template variables
            
        Returns:
            str: Name of the new template with examples
            
        Raises:
            ValueError: If template is not found or examples are invalid
        """
        # Get template from cache
        template_info = self.template_cache.get_template(template_name)
        if not template_info:
            raise ValueError(f"Template '{template_name}' not found")
            
        template_text = template_info['text']
            
        if not examples:
            return template_name
            
        # Format example section using Jinja2 syntax
        example_section = """

{# EXAMPLES SECTION #}
{% if examples %}
EXAMPLES:

{% for example in examples %}
Example {{ loop.index }}:

Input:
```
{{ example.input }}
```
Output:
```
{{ example.output }}
```
{% endfor %}
{% endif %}
"""
        
        # Create new template with examples appended
        new_template_text = template_text + example_section
        new_template_name = f"{template_name}_with_examples"
        
        # Register the new template
        self.register_template(new_template_name, new_template_text, f"{template_info['version']}+examples")
        
        return new_template_name
    
    def validate(self, template_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the template structure and variables.
        
        Args:
            template_name: Name of the template to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if template_name not in self._template_instances:
            return False, f"Template '{template_name}' not found"
            
        template_info = self._template_instances[template_name]
        template_text = template_info['text']
        
        if not template_text:
            return False, "Template is empty"
            
        # Check for balanced braces in variable placeholders
        open_count = template_text.count('{{')
        close_count = template_text.count('}}')
        
        if open_count != close_count:
            return False, f"Unbalanced variable placeholders: {{ {open_count} opening vs {close_count} closing }}"
            
        # Check template size
        if len(template_text) > 10000:
            return False, f"Template too large: {len(template_text)} chars (max 10000)"
            
        return True, None

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
    
    # Limit number of rows and columns
    display_df = df.iloc[:max_rows, :max_cols]
    
    # Check if DataFrame was truncated and add indicators
    rows_truncated = df.shape[0] > max_rows
    cols_truncated = df.shape[1] > max_cols
    
    result = display_df.to_string()
    
    # Add truncation indicators
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
        # Extract necessary chunk information
        text = chunk.get('text', 'No text available')
        source = chunk.get('source', 'Unknown source')
        similarity = chunk.get('similarity', 0.0)
        
        # Truncate chunk if necessary
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        
        # Format chunk with metadata
        formatted_chunk = f"CHUNK {i} [source: {source}, relevance: {similarity:.2f}]:\n{text}\n"
        formatted_chunks.append(formatted_chunk)
    
    return "\n".join(formatted_chunks)
