"""
Prompt templates for RAG generation.

This module provides advanced templating capabilities for AI prompts,
supporting template loading, formatting, validation, and versioning.
"""
import json
import re
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple, Union

from llama_index.experimental.query_engine import PandasQueryEngine
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.log import log


@singleton
class PromptTemplate:
    """
    Class for managing prompt templates for AI generation.
    """
    
    # Template for system context in RAG conversations
    RAG_SYSTEM_TEMPLATE = """
    You are an AI assistant equipped with a retrieval-augmented generation (RAG) system.
    You can access relevant documents and data for questions on ${domain}.
    
    When responding:
    - Use retrieved context to inform your answers when available
    - Cite sources when information comes from specific documents
    - Acknowledge when you're unsure or when the retrieval system doesn't provide relevant information
    - Present structured data in a clear, formatted way
    - Focus on giving accurate, helpful information based on the provided context
    - Maintain a ${tone} tone in responses
    
    Your goal is to provide accurate, helpful responses enhanced by retrieved information.
    """
    
    # Template for query reformulation to improve retrieval
    QUERY_REFORMULATION_TEMPLATE = """
    I need to find relevant information about the following query:
    "${original_query}"
    
    To improve retrieval results, please convert this into a more effective search query by:
    1. Identifying key concepts and terminology
    2. Adding synonyms for important terms
    3. Focusing on specific aspects rather than general questions
    4. Including any relevant technical terms mentioned
    5. Removing unnecessary words and fillers
    
    Context about the domain: ${domain_context}
    Previous related queries: ${related_queries}
    """
    
    # Template for presenting retrieved context in conversation
    CONTEXT_PRESENTATION_TEMPLATE = """
    Based on retrieved information:
    
    ${formatted_chunks}
    
    Use these relevant passages to address the following question:
    ${query}
    
    Provide a comprehensive answer that synthesizes information from the retrieved content.
    Cite specific sources when drawing from particular chunks.
    """
    
    # Template for presenting dataframe results
    DATAFRAME_RESULTS_TEMPLATE = """
    Here's the data analysis for your query "${query}":
    
    ${formatted_data}
    
    Key insights:
    - ${insight_1}
    - ${insight_2}
    - ${insight_3}
    
    Would you like me to explain any specific aspect of these results in more detail?
    """
    
    # Template for summarizing lengthy chunks
    CHUNK_SUMMARIZATION_TEMPLATE = """
    Summarize the following information in a concise way while preserving key details:
    
    "${chunk_content}"
    
    Create a summary that:
    1. Captures essential facts and concepts
    2. Maintains technical accuracy
    3. Is approximately ${target_length} words long
    4. Focuses on aspects relevant to ${focus_area}
    """
    
    # Template for agent handoff with context
    AGENT_HANDOFF_TEMPLATE = """
    [AGENT HANDOFF: ${source_agent} → ${target_agent}]
    
    QUERY: ${original_query}
    
    RETRIEVED CONTEXT:
    ${context_summary}
    
    ACTIONS TAKEN:
    ${previous_actions}
    
    NEXT STEPS:
    ${recommended_actions}
    
    ADDITIONAL NOTES:
    ${notes}
    """
    
    # Version tracking for templates
    TEMPLATE_VERSIONS = {
        "rag_system": "1.0",
        "query_reformulation": "1.0",
        "context_presentation": "1.0",
        "dataframe_results": "1.0",
        "chunk_summarization": "1.0",
        "agent_handoff": "1.0"
    }
    
    # Dictionary to cache template instances by name
    _template_instances = {}
    
    def __init__(self, template_text: Optional[str] = None, template_name: Optional[str] = None):
        """
        Initialize a prompt template.
        
        Args:
            template_text: Custom template text, or None to create an empty template
            template_name: Optional name for the template for tracking purposes
        """
        self.template = Template(template_text) if template_text else None
        self.template_name = template_name
        self.template_text = template_text
        self.variables = self._extract_variables(template_text) if template_text else set()
        self.version = "custom-1.0"
        self.usage_count = 0
        self.successful_uses = 0
        
    def _extract_variables(self, template_text: str) -> set:
        """
        Extract template variables from the template text.
        
        Args:
            template_text: The template text to analyze
            
        Returns:
            Set of variable names used in the template
        """
        if not template_text:
            return set()
            
        # Find all ${variable} patterns in the template
        matches = re.findall(r'\${([^}]+)}', template_text)
        return set(matches)
            
    @classmethod
    def from_preset(cls, template_name: str) -> 'PromptTemplate':
        """
        Create a template from a predefined preset.
        
        Args:
            template_name: Name of the preset template to use
                (rag_system, query_reformulation, context_presentation,
                 dataframe_results, chunk_summarization, agent_handoff)
                
        Returns:
            PromptTemplate: Initialized with the selected preset
            
        Raises:
            ValueError: If template_name is not a valid preset
        """
        # Check if we already have a cached instance
        if template_name in cls._template_instances:
            return cls._template_instances[template_name]
            
        template_map = {
            'rag_system': cls.RAG_SYSTEM_TEMPLATE,
            'query_reformulation': cls.QUERY_REFORMULATION_TEMPLATE,
            'context_presentation': cls.CONTEXT_PRESENTATION_TEMPLATE,
            'dataframe_results': cls.DATAFRAME_RESULTS_TEMPLATE,
            'chunk_summarization': cls.CHUNK_SUMMARIZATION_TEMPLATE,
            'agent_handoff': cls.AGENT_HANDOFF_TEMPLATE
        }
        
        if template_name not in template_map:
            valid_options = ", ".join(template_map.keys())
            raise ValueError(f"Unknown template preset: '{template_name}'. "
                            f"Available presets: {valid_options}")
            
        template = cls(template_map[template_name], template_name)
        template.version = cls.TEMPLATE_VERSIONS.get(template_name, "1.0")
        
        # Cache the template instance
        cls._template_instances[template_name] = template
        
        log.debug(f"Created template from preset '{template_name}' (v{template.version})")
        return template
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'PromptTemplate':
        """
        Load a template from a file.
        
        Args:
            file_path: Path to the template file (txt or json)
            
        Returns:
            PromptTemplate: Initialized with the loaded template
            
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
                    
                template = cls(template_text, template_name)
                template.version = version
                
            else:  # Default to raw text file
                with open(path, 'r', encoding='utf-8') as f:
                    template_text = f.read()
                    
                template = cls(template_text, path.stem)
                template.version = "file-1.0"
                
            log.info(f"Loaded template '{template.template_name}' from {path.name}")
            return template
            
        except Exception as e:
            log.error(f"Failed to load template from {path}: {e}")
            raise ValueError(f"Failed to load template from {file_path}: {e}")
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'json') -> None:
        """
        Save the template to a file.
        
        Args:
            file_path: Path where to save the template
            format: Format to save as ('json' or 'txt')
            
        Raises:
            ValueError: If the template is not initialized or format is invalid
        """
        if not self.template_text:
            raise ValueError("Cannot save empty template")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'json':
                data = {
                    'name': self.template_name or 'unnamed_template',
                    'version': self.version,
                    'template': self.template_text,
                    'variables': list(self.variables),
                    'stats': {
                        'usage_count': self.usage_count,
                        'successful_uses': self.successful_uses
                    }
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
            else:  # Default to raw text
                with open(path, 'w', encoding='utf-8') as f:
                    if self.template_name:
                        f.write(f"# Template: {self.template_name} (v{self.version})\n\n")
                    f.write(self.template_text)
                
            log.info(f"Saved template to {path}")
            
        except Exception as e:
            log.error(f"Failed to save template to {path}: {e}")
            raise ValueError(f"Failed to save template: {e}")
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided values.
        
        Args:
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            str: The formatted prompt string
            
        Raises:
            ValueError: If template is not initialized
            KeyError: If required template variables are missing
        """
        if not self.template:
            raise ValueError("Template not initialized")
        
        # Track usage stats
        self.usage_count += 1
            
        try:
            # Check for missing variables
            missing_vars = self.variables - set(kwargs.keys())
            if missing_vars:
                missing_list = ", ".join(missing_vars)
                raise KeyError(f"Missing required template variables: {missing_list}")
                
            # Format template
            result = self.template.substitute(**kwargs)
            
            # Track successful formatting
            self.successful_uses += 1
            return result
            
        except KeyError as e:
            log.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing required template variable: {e}")
        
    def add_examples(self, examples: List[Dict[str, str]]) -> 'PromptTemplate':
        """
        Add few-shot learning examples to a template.
        
        Args:
            examples: List of example dictionaries with keys matching template variables
            
        Returns:
            PromptTemplate: Updated template with examples
            
        Raises:
            ValueError: If template is not initialized or examples are invalid
        """
        if not self.template_text:
            raise ValueError("Cannot add examples to empty template")
            
        if not examples:
            return self
            
        # Format example section
        example_section = "\n\nEXAMPLES:\n"
        
        for i, example in enumerate(examples, 1):
            example_section += f"\nExample {i}:\n"
            
            # Format each example using the available variables
            try:
                # Extract input and output from example
                example_inputs = {k: v for k, v in example.items() if k != 'output'}
                example_output = example.get('output', '')
                
                # Format example with available inputs
                example_prompt = self.template.safe_substitute(**example_inputs)
                
                # Add formatted example
                example_section += f"Input:\n```\n{example_prompt}\n```\n"
                if example_output:
                    example_section += f"Output:\n```\n{example_output}\n```\n"
                    
            except Exception as e:
                log.warning(f"Skipped invalid example {i}: {e}")
        
        # Create new template with examples appended
        new_template_text = self.template_text + example_section
        new_template = PromptTemplate(new_template_text, f"{self.template_name}_with_examples")
        new_template.version = f"{self.version}+examples"
        
        return new_template
    
    def create_chat_messages(
        self, 
        system_template: Optional['PromptTemplate'] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Create a list of chat messages from the template.
        
        Args:
            system_template: Optional system template to use
            **kwargs: Key-value pairs to substitute in the templates
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
            
        Raises:
            ValueError: If user template is not initialized
        """
        messages = []
        
        # Add system message if provided
        if system_template:
            try:
                system_content = system_template.format(**kwargs)
                messages.append({
                    "role": "system",
                    "content": system_content
                })
            except Exception as e:
                log.warning(f"Failed to format system template: {e}")
                # Continue with just the user message
        
        # Add user message from this template
        user_content = self.format(**kwargs)
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the template structure and variables.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.template_text:
            return False, "Template is empty"
            
        # Check for balanced braces in variable placeholders
        open_count = self.template_text.count('${')
        close_count = self.template_text.count('}')
        
        if open_count != close_count:
            return False, f"Unbalanced variable placeholders: ${open_count} opening vs {close_count} closing"
            
        # Check template size
        if len(self.template_text) > 10000:
            return False, f"Template too large: {len(self.template_text)} chars (max 10000)"
            
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
