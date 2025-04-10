"""
Text chunking functionality for RAG content processing.

This module provides algorithms for splitting documents into optimal chunks
for embedding and retrieval, implementing concepts from Specification.md for
improved semantic coherence and retrieval effectiveness.
"""
import re
import nltk
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from oarc_rag.utils.log import log
from oarc_rag.utils.config.config import Config


class TextChunker:
    """
    Split text content into semantically meaningful chunks for embedding.
    
    This class implements multiple chunking strategies with configurable
    parameters for optimizing retrieval performance and semantic coherence.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        chunk_strategy: str = "paragraph", 
        preserve_headers: bool = True,
        combine_short_chunks: bool = True,
        min_chunk_length: int = 100,
        max_chunk_length: int = 1000,
        split_long_sentences: bool = False
    ):
        """
        Initialize the text chunker with configurable parameters.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Target overlap between consecutive chunks in tokens
            chunk_strategy: Chunking strategy ('fixed', 'sentence', 'paragraph', 'semantic')
            preserve_headers: Whether to keep headers with their content
            combine_short_chunks: Whether to combine short chunks
            min_chunk_length: Minimum length of a chunk in characters
            max_chunk_length: Maximum length of a chunk in characters
            split_long_sentences: Whether to split very long sentences
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_strategy = chunk_strategy
        self.preserve_headers = preserve_headers
        self.combine_short_chunks = combine_short_chunks
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.split_long_sentences = split_long_sentences
        
        # Initialize semantic splitting if needed
        self._initialize_semantic_splitting()
        
        # Track chunking performance
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "avg_chunk_size": 0,
            "empty_chunks_filtered": 0,
            "short_chunks_combined": 0
        }
    
    def _initialize_semantic_splitting(self) -> None:
        """Initialize semantic splitting components if that strategy is selected."""
        if self.chunk_strategy == "semantic":
            try:
                # Optional import for semantic splitting
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    log.warning("Downloading spaCy model for semantic chunking...")
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                log.info("Semantic chunking initialized with spaCy")
            except ImportError:
                log.warning("spaCy not available, falling back to paragraph chunking")
                self.chunk_strategy = "paragraph"
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into optimal chunks based on configured strategy.
        
        Args:
            text: Text content to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Choose chunking strategy
        if self.chunk_strategy == "fixed":
            chunks = self._chunk_fixed_size(text)
        elif self.chunk_strategy == "sentence":
            chunks = self._chunk_by_sentence(text)
        elif self.chunk_strategy == "semantic":
            chunks = self._chunk_semantic(text)
        else:  # Default to paragraph
            chunks = self._chunk_by_paragraph(text)
            
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        # Update statistics
        self.stats["documents_processed"] += 1
        self.stats["chunks_created"] += len(chunks)
        if chunks:
            avg_size = sum(len(chunk) for chunk in chunks) / max(1, len(chunks))
            self.stats["avg_chunk_size"] = (
                (self.stats["avg_chunk_size"] * (self.stats["documents_processed"] - 1) + avg_size) / 
                self.stats["documents_processed"]
            )
            
        return chunks
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # Use simple token-based chunking (approximating token count from character count)
        tokens = text.split()
        chunks = []
        
        # Average English token is ~5 characters, so approximate
        chunk_size_tokens = self.chunk_size
        overlap_tokens = self.overlap
        
        if len(tokens) <= chunk_size_tokens:
            return [text]
            
        for i in range(0, len(tokens), chunk_size_tokens - overlap_tokens):
            chunk_tokens = tokens[i:i + chunk_size_tokens]
            chunk = " ".join(chunk_tokens)
            chunks.append(chunk)
            
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """
        Split text into chunks preserving sentence boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Handle empty or trivial case
        if not sentences:
            return []
        if len(sentences) == 1 or len(text) <= self.chunk_size * 5:  # Approximating tokens
            return [text]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Approximate token count
            sentence_length = len(sentence.split())
            
            # Split long sentences if needed
            if sentence_length > self.chunk_size and self.split_long_sentences:
                sentence_chunks = self._split_long_sentence(sentence)
                for s_chunk in sentence_chunks:
                    s_chunk_length = len(s_chunk.split())
                    
                    if current_length + s_chunk_length > self.chunk_size:
                        # Start new chunk if this one is full
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            # Keep some sentences for overlap
                            overlap_sentences = current_chunk[-self.overlap//10:]
                            current_chunk = overlap_sentences
                            current_length = sum(len(s.split()) for s in overlap_sentences)
                    
                    current_chunk.append(s_chunk)
                    current_length += s_chunk_length
            else:
                # Regular sentence processing
                if current_length + sentence_length > self.chunk_size:
                    # Start new chunk if this one is full
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Keep some sentences for overlap
                        overlap_sentences = current_chunk[-self.overlap//10:]
                        current_chunk = overlap_sentences
                        current_length = sum(len(s.split()) for s in overlap_sentences)
                
                current_chunk.append(sentence)
                current_length += sentence_length
                
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """
        Split text into chunks preserving paragraph boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # Split text into paragraphs - look for double newlines or markdown paragraph breaks
        paragraph_pattern = r"\n\s*\n|\n#{1,6}\s"
        paragraphs = re.split(paragraph_pattern, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Handle empty or trivial case
        if not paragraphs:
            return []
        if len(paragraphs) == 1:
            # If just one paragraph and it's small, return it directly
            if len(paragraphs[0].split()) <= self.chunk_size:
                return [paragraphs[0]]
            # Otherwise break it down by sentences
            return self._chunk_by_sentence(paragraphs[0])
        
        # Process paragraphs, grouping them into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Find headers
        header_indices = []
        if self.preserve_headers:
            for i, para in enumerate(paragraphs):
                if para.startswith('#'):
                    header_indices.append(i)
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            para_length = len(para.split())
            
            # Check if this is a header
            is_header = i in header_indices
            
            # If adding this paragraph would exceed chunk size and it's not a header
            if current_length + para_length > self.chunk_size and not is_header and current_chunk:
                # Save current chunk and start a new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_length += para_length
            
            # If this is a header, include the next paragraph in this chunk
            if is_header and i + 1 < len(paragraphs):
                i += 1
                next_para = paragraphs[i]
                current_chunk.append(next_para)
                current_length += len(next_para.split())
            
            i += 1
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # This relies on spaCy being available
        if not hasattr(self, 'nlp'):
            log.warning("Semantic chunking unavailable, falling back to paragraph chunking")
            return self._chunk_by_paragraph(text)
            
        # Process the document with spaCy
        doc = self.nlp(text)
        
        # Group sentences into semantically coherent chunks
        chunks = []
        current_chunk = []
        current_entities = set()
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Get sentence entities
            sent_entities = {ent.text for ent in sent.ents}
            sent_length = len(sent_text.split())
            
            # Check if this sentence shares entities with current chunk
            has_overlap = bool(sent_entities & current_entities)
            
            # Start a new chunk if:
            # 1. Current chunk is getting too large, or
            # 2. This sentence has no entity overlap with current chunk
            if (current_length + sent_length > self.chunk_size) or (current_length > 0 and not has_overlap):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_entities = sent_entities
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_entities.update(sent_entities)
                current_length += sent_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split very long sentences at punctuation or phrases.
        
        Args:
            sentence: Long sentence to split
            
        Returns:
            List of smaller sentences
        """
        # Look for natural breaking points: semicolons, commas, dashes, etc.
        split_points = [
            r'(?<=;)', r'(?<=:)', r'(?<=—)', r'(?<=--)',
            r'(?<=\))', r'(?<=\])', r'\.{3}',
            r', (?=and |but |or |nor |yet |so )'
        ]
        
        pattern = '|'.join(split_points)
        chunks = re.split(pattern, sentence)
        
        # Further split if chunks are still too large
        result = []
        for chunk in chunks:
            if len(chunk.split()) > self.chunk_size:
                # If still too large, split by commas
                comma_chunks = re.split(r'(?<=, )', chunk)
                for c in comma_chunks:
                    if c.strip():
                        result.append(c.strip())
            else:
                if chunk.strip():
                    result.append(chunk.strip())
        
        return result
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Apply post-processing to chunks.
        
        Args:
            chunks: Initial chunks
            
        Returns:
            Processed chunks
        """
        # Filter empty chunks
        filtered_chunks = [chunk for chunk in chunks if chunk.strip()]
        self.stats["empty_chunks_filtered"] += (len(chunks) - len(filtered_chunks))
        
        # Handle short chunks
        if self.combine_short_chunks:
            combined = []
            current_chunk = ""
            
            for chunk in filtered_chunks:
                # If chunk is very short, combine with previous
                if len(chunk) < self.min_chunk_length:
                    if current_chunk:
                        # Combine with previous chunk
                        current_chunk += " " + chunk
                        self.stats["short_chunks_combined"] += 1
                    else:
                        # No previous chunk to combine with
                        current_chunk = chunk
                else:
                    # Regular chunk processing
                    if current_chunk:
                        # Add previous accumulated chunk
                        combined.append(current_chunk)
                        current_chunk = ""
                    
                    # Check if this chunk exceeds max length
                    if len(chunk) > self.max_chunk_length:
                        # Split into smaller chunks
                        sub_chunks = self._chunk_by_sentence(chunk)
                        combined.extend(sub_chunks)
                    else:
                        combined.append(chunk)
            
            # Add any remaining chunk
            if current_chunk:
                combined.append(current_chunk)
                
            return combined
                
        return filtered_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get chunking statistics.
        
        Returns:
            Dict with chunking statistics
        """
        return self.stats.copy()

    def set_chunk_strategy(self, strategy: str) -> None:
        """
        Change the chunking strategy.
        
        Args:
            strategy: New strategy ('fixed', 'sentence', 'paragraph', 'semantic')
            
        Raises:
            ValueError: If strategy is not recognized
        """
        valid_strategies = ['fixed', 'sentence', 'paragraph', 'semantic']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid chunking strategy: {strategy}. Valid options: {valid_strategies}")
            
        self.chunk_strategy = strategy
        # Re-initialize if needed
        if strategy == "semantic":
            self._initialize_semantic_splitting()
