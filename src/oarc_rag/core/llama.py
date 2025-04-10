"""
LlamaIndex integration for enhancing RAG capabilities.

This module provides integration with LlamaIndex for document loading,
processing, and RAG operations, implementing concepts from both
Specification.md and Big_Brain.md for advanced retrieval capabilities.
"""
import os
import importlib
import json
import time
from typing import List, Dict, Any, Union, Callable, Optional, Tuple, Set
from pathlib import Path
from enum import Enum

from oarc_rag.utils.log import log
from oarc_rag.utils.decorators.singleton import singleton
from oarc_rag.utils.config.config import Config

# Import LlamaIndex components
from llama_index.core import Settings
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.readers.file.docs import PDFReader, DocxReader
from llama_index.readers.file import HTMLTagReader
from llama_index.readers.json import JSONReader
from llama_index.readers.file.markdown import MarkdownReader
from llama_index.readers.file import CSVReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.llms import MockLLM
from llama_index.core.ingestion import IngestionPipeline
from llama_index.experimental.query_engine import PandasQueryEngine

# Sleep/Awake modes from Big_Brain.md
class OperationalMode(Enum):
    """
    Operational modes for LlamaIndex integration based on Big_Brain.md.
    """
    AWAKE = "awake"  # Real-time processing with lower latency
    SLEEP = "sleep"  # Deep processing with higher quality

# Check if we're in test mode for mocking
_IN_TEST_MODE = False


class LlamaDocumentLoader:
    """
    Document loader using LlamaIndex readers for various file formats.
    
    This class provides a unified interface for loading documents from
    different sources with appropriate readers based on file format.
    Implements modular document processing as described in Specification.md.
    """
    
    def __init__(self, operational_mode: OperationalMode = OperationalMode.AWAKE):
        """
        Initialize the document loader with operational mode.
        
        Args:
            operational_mode: Processing mode (awake=fast, sleep=thorough)
        """
        global _IN_TEST_MODE
            
        # Set operational mode
        self.operational_mode = operational_mode
            
        # Initialize parsers and readers
        self.node_parser = SimpleNodeParser.from_defaults()
        
        # Initialize file-specific readers
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        self.html_reader = HTMLTagReader()
        self.json_reader = JSONReader()
        self.markdown_reader = MarkdownReader()
        self.csv_reader = CSVReader()
        
        # Document processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "avg_chunks_per_doc": 0.0,
            "processing_time": 0.0,
            "avg_processing_time": 0.0,
            "formats_processed": {}
        }
        
    def load(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load documents from file using appropriate LlamaIndex reader.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of documents as dictionaries
            
        Raises:
            FileNotFoundError: If file not found
            ValueError: If file format not supported
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        # Track processing start time
        start_time = time.time()
            
        # Use appropriate reader based on file extension
        doc_format = path.suffix.lower()
        
        try:
            if doc_format == '.pdf':
                # Check which API method is available (parse_file or load_data)
                documents = self._call_reader_method(self.pdf_reader, str(path))
                
            elif doc_format in ['.docx', '.doc']:
                documents = self._call_reader_method(self.docx_reader, str(path))
                
            elif doc_format in ['.html', '.htm']:
                documents = self._call_reader_method(self.html_reader, str(path))
                
            elif doc_format == '.json':
                documents = self._call_reader_method(self.json_reader, str(path))
                
            elif doc_format in ['.md', '.markdown']:
                documents = self._call_reader_method(self.markdown_reader, str(path))
                
            elif doc_format == '.csv':
                documents = self._call_reader_method(self.csv_reader, str(path))
                
            else:
                raise ValueError(f"Unsupported file format: {doc_format}")
                
            # Parse into nodes and convert to dict
            result = self._convert_to_oarc_rag_format(documents)
            
            # Update statistics
            elapsed = time.time() - start_time
            self.stats["documents_processed"] += 1
            self.stats["total_chunks"] += len(result)
            self.stats["avg_chunks_per_doc"] = self.stats["total_chunks"] / self.stats["documents_processed"]
            self.stats["processing_time"] += elapsed
            self.stats["avg_processing_time"] = self.stats["processing_time"] / self.stats["documents_processed"]
            
            # Track file format statistics
            if doc_format in self.stats["formats_processed"]:
                self.stats["formats_processed"][doc_format] += 1
            else:
                self.stats["formats_processed"][doc_format] = 1
            
            log.info(f"Loaded {len(result)} chunks from {path.name} in {elapsed:.3f}s")
            return result
            
        except Exception as e:
            log.error(f"Error loading document {path}: {e}")
            raise
    
    def load_from_web(self, url: str, depth: int = 1, max_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Load documents from a web URL using the SpiderReader.
        
        Args:
            url: Web URL to crawl
            depth: Crawling depth (1 = just the URL, 2 = URL + linked pages)
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of documents as dictionaries
            
        Raises:
            ValueError: If URL is invalid or unreachable
        """
        # Track processing start time
        start_time = time.time()
        
        try:
            # Configure and use the spider reader
            spider_config = {
                "urls": [url],
                "depth": depth,
                "max_pages": max_pages,
                "use_async": True,
                "verify_ssl": False
            }
            
            # Based on operational mode, adjust spider settings
            if self.operational_mode == OperationalMode.SLEEP:
                # More comprehensive crawl during sleep mode
                spider_config["extract_images"] = True
                spider_config["timeout"] = 30
            else:
                # Faster but less thorough crawl during awake mode
                spider_config["extract_images"] = False
                spider_config["timeout"] = 10
            
            # Load content from web
            log.info(f"Crawling web content from {url} with depth {depth}...")
            documents = self.spider_reader.load_data(**spider_config)
            
            # Parse into nodes and convert to dict
            result = self._convert_to_oarc_rag_format(documents)
            
            # Update statistics
            elapsed = time.time() - start_time
            self.stats["documents_processed"] += 1
            self.stats["total_chunks"] += len(result)
            self.stats["avg_chunks_per_doc"] = self.stats["total_chunks"] / self.stats["documents_processed"]
            self.stats["processing_time"] += elapsed
            self.stats["avg_processing_time"] = self.stats["processing_time"] / self.stats["documents_processed"]
            
            # Track web content statistics
            web_format = "web"
            if web_format in self.stats["formats_processed"]:
                self.stats["formats_processed"][web_format] += 1
            else:
                self.stats["formats_processed"][web_format] = 1
            
            log.info(f"Loaded {len(result)} chunks from {url} in {elapsed:.3f}s")
            return result
            
        except Exception as e:
            log.error(f"Error loading web content from {url}: {e}")
            raise ValueError(f"Failed to load from URL {url}: {str(e)}")
    
    def _call_reader_method(self, reader: Any, file_path: str) -> List[LlamaDocument]:
        """
        Try different API methods that might exist on the reader.
        
        Args:
            reader: LlamaIndex reader
            file_path: Path to the file to read
            
        Returns:
            List of LlamaIndex documents
        """
        # Try the methods in order of likelihood
        for method_name in ['parse_file', 'load_data', 'load']:
            if hasattr(reader, method_name) and callable(getattr(reader, method_name)):
                return getattr(reader, method_name)(file_path)
        
        # If we get here, none of the methods worked
        raise AttributeError(f"Reader {reader.__class__.__name__} has no valid loading method")
    
    def _convert_to_oarc_rag_format(self, llama_docs: List[LlamaDocument]) -> List[Dict[str, Any]]:
        """
        Convert LlamaIndex documents to oarc_rag format.
        
        Args:
            llama_docs: List of LlamaIndex documents
            
        Returns:
            List of documents in oarc_rag format
        """
        results = []
        
        for doc in llama_docs:
            # Extract metadata
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            
            # Create document dict in oarc_rag format
            results.append({
                "content": doc.text,
                "metadata": metadata,
                "source": metadata.get("source", "unknown"),
                "content_type": metadata.get("content_type", "text"),
                "embedding": None,  # Will be populated later in the pipeline
                "timestamp": metadata.get("timestamp", int(time.time()))
            })
            
        return results
    
    def create_nodes_from_documents(self, documents: List[Dict[str, Any]]) -> List[TextNode]:
        """
        Create LlamaIndex nodes from document dictionaries.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of LlamaIndex TextNode objects
        """
        nodes = []
        for doc in documents:
            node = TextNode(
                text=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            nodes.append(node)
        return nodes
        
    def set_operational_mode(self, mode: OperationalMode) -> None:
        """
        Set operational mode for document processing.
        
        Args:
            mode: New operational mode
        """
        self.operational_mode = mode
        
        # Adjust processing settings based on mode
        if mode == OperationalMode.SLEEP:
            # In sleep mode, use more comprehensive processing
            if self.node_parser and hasattr(self.node_parser, "set_chunk_overlap"):
                self.node_parser.set_chunk_overlap(100)  # Increased overlap in sleep mode
        else:
            # In awake mode, optimize for speed
            if self.node_parser and hasattr(self.node_parser, "set_chunk_overlap"):
                self.node_parser.set_chunk_overlap(50)  # Default overlap in awake mode
                
        log.info(f"Document loader operational mode set to: {mode.value}")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get document processing statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()


class TextProcessor:
    """
    Text processing utilities using LlamaIndex components.
    
    This class provides text chunking and processing capabilities
    using LlamaIndex functionality, implementing concepts from
    Specification.md for optimal chunking strategies.
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        operational_mode: OperationalMode = OperationalMode.AWAKE
    ):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between consecutive chunks
            operational_mode: Processing mode (awake=fast, sleep=thorough)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.operational_mode = operational_mode
        
        # Initialize parser
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Processing statistics
        self.stats = {
            "texts_processed": 0,
            "total_chunks": 0,
            "avg_chunks_per_text": 0.0,
            "processing_time": 0.0
        }
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using LlamaIndex.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        start_time = time.time()
        
        # Process based on operational mode
        if self.operational_mode == OperationalMode.SLEEP:
            # In sleep mode, use more sophisticated chunking
            # This would ideally use more advanced LlamaIndex features
            nodes = self.node_parser.get_nodes_from_documents(
                [LlamaDocument(text=text)]
            )
        else:
            # In awake mode, use standard chunking
            nodes = self.node_parser.get_nodes_from_documents(
                [LlamaDocument(text=text)]
            )
            
        # Extract text from nodes
        chunks = [node.get_content() for node in nodes]
        
        # Update statistics
        elapsed = time.time() - start_time
        self.stats["texts_processed"] += 1
        self.stats["total_chunks"] += len(chunks)
        self.stats["avg_chunks_per_text"] = self.stats["total_chunks"] / self.stats["texts_processed"] 
        self.stats["processing_time"] += elapsed
        
        return chunks
    
    def create_ingestion_pipeline(self, transformations: Optional[List[Callable]] = None) -> Any:
        """
        Create an ingestion pipeline for document processing.
        
        Args:
            transformations: Optional list of transformation functions
            
        Returns:
            LlamaIndex IngestionPipeline
        """
        # Create pipeline with specified transformations
        pipeline = IngestionPipeline()
        
        if transformations:
            for transform in transformations:
                pipeline.add_transformation(transform)
                
        return pipeline
        
    def set_operational_mode(self, mode: OperationalMode) -> None:
        """
        Set operational mode for text processing.
        
        Args:
            mode: New operational mode
        """
        prev_mode = self.operational_mode
        self.operational_mode = mode
        
        # Adjust processing parameters based on mode
        if mode == OperationalMode.SLEEP:
            # In sleep mode, use more thorough processing
            self.chunk_overlap = self.chunk_size // 5  # 20% overlap
            
            if hasattr(self.node_parser, "chunk_overlap"):
                self.node_parser.chunk_overlap = self.chunk_overlap
                
        else:
            # In awake mode, optimize for speed
            self.chunk_overlap = self.chunk_size // 10  # 10% overlap
            
            if hasattr(self.node_parser, "chunk_overlap"):
                self.node_parser.chunk_overlap = self.chunk_overlap
        
        log.info(f"Text processor mode changed: {prev_mode.value} → {mode.value}")


def setup_llama_index() -> bool:
    """
    Configure LlamaIndex global settings.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Set global properties for LlamaIndex
        # The mock pattern needs to explicitly set the properties on the mock object
        settings = Settings
        settings.chunk_size = 512
        settings.chunk_overlap = 50
        settings.embed_model = None  # Use OARC-RAG's embedding model instead
        
        # In test mode, use a mock LLM
        global _IN_TEST_MODE
        if _IN_TEST_MODE:
            settings.llm = MockLLM()
        
        return True
    except Exception as e:
        log.error(f"Error setting up LlamaIndex: {e}")
        return False


def set_test_mode(enabled: bool = True) -> None:
    """
    Set test mode to allow mocking LlamaIndex components.
    Only for internal test usage.
    
    Args:
        enabled: Whether to enable test mode
    """
    global _IN_TEST_MODE
    _IN_TEST_MODE = enabled
    
    # Set environment variable for LlamaIndex itself to use mock LLM
    if enabled:
        os.environ["IS_TESTING"] = "True"
    else:
        os.environ.pop("IS_TESTING", None)


@singleton
class LlamaIndexManager:
    """
    Manager for LlamaIndex integration following the singleton pattern.
    
    This class provides centralized management of LlamaIndex components
    and implements the document consolidation concepts from Big_Brain.md.
    """
    
    def __init__(self):
        """Initialize the LlamaIndex manager."""
        # Initialize components
        self.document_loader = LlamaDocumentLoader()
        self.text_processor = TextProcessor()
        
        # Set operational mode from config or default to awake
        config = Config()
        mode_str = config.get("llama_index.operational_mode", "awake")
        self.operational_mode = OperationalMode(mode_str)
        
        # Set mode on components
        self.document_loader.set_operational_mode(self.operational_mode)
        self.text_processor.set_operational_mode(self.operational_mode)
        
        # Knowledge consolidation metrics from Big_Brain.md
        self.knowledge_metrics = {
            "docs_processed": 0,
            "last_consolidation": None,
            "consolidation_count": 0,
            "chunk_improvement_rate": 0.0,
        }
        
        log.info(f"LlamaIndex Manager initialized with mode: {self.operational_mode.value}")
        
    def set_operational_mode(self, mode: Union[str, OperationalMode]) -> None:
        """
        Set operational mode for all LlamaIndex components.
        
        Args:
            mode: New operational mode
            
        Raises:
            ValueError: If mode string is invalid
        """
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = OperationalMode(mode.lower())
            except ValueError:
                raise ValueError(f"Invalid operational mode: {mode}. Use 'awake' or 'sleep'.")
        
        # Update mode on manager and components
        prev_mode = self.operational_mode
        self.operational_mode = mode
        
        self.document_loader.set_operational_mode(mode)
        self.text_processor.set_operational_mode(mode)
        
        log.info(f"LlamaIndex operational mode changed: {prev_mode.value} → {mode.value}")
        
    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate document knowledge based on Big_Brain.md concepts.
        
        Returns:
            Dict with consolidation metrics
        """
        # This would implement the knowledge consolidation concept from Big_Brain.md
        self.knowledge_metrics["last_consolidation"] = int(time.time())
        self.knowledge_metrics["consolidation_count"] += 1
        
        # In a real implementation, this would analyze document statistics,
        # identify patterns, and optimize future document processing
        
        return {
            "status": "completed",
            "consolidation_count": self.knowledge_metrics["consolidation_count"],
            "processed_docs": self.knowledge_metrics["docs_processed"]
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive LlamaIndex integration statistics.
        
        Returns:
            Dict with combined statistics
        """
        return {
            "document_loader": self.document_loader.get_stats(),
            "text_processor": self.text_processor.stats,
            "knowledge_metrics": self.knowledge_metrics,
            "operational_mode": self.operational_mode.value
        }


# Initialize the manager for easy access
llama_index_manager = LlamaIndexManager()