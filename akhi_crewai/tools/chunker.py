#!/usr/bin/env python3
"""
Text Chunking Tool for CrewAI

This module implements a CrewAI-compatible tool for chunking transcribed text
into optimized segments for embedding and retrieval. Designed specifically
for Islamic content with semantic boundary preservation.

Author: Assistant
Date: 2024
"""

import os
import re
import json
import yaml
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator
from crewai.tools import BaseTool


class ChunkingInput(BaseModel):
    """Input schema for text chunking operations."""
    
    text: str = Field(
        ...,
        description="Text content to be chunked (can be plain text or JSON transcript)",
        min_length=1
    )
    
    chunk_size: Optional[int] = Field(
        default=None,
        description="Maximum number of characters per chunk (overrides config)",
        ge=100,
        le=5000
    )
    
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="Number of characters to overlap between chunks (overrides config)",
        ge=0,
        le=1000
    )
    
    separator: Optional[str] = Field(
        default=None,
        description="Text separator for chunking (overrides config)"
    )
    
    preserve_sentences: Optional[bool] = Field(
        default=None,
        description="Whether to preserve sentence boundaries (overrides config)"
    )
    
    strategy: str = Field(
        default="sliding_window",
        description="Chunking strategy to use"
    )
    
    output_format: str = Field(
        default="list",
        description="Output format: 'list' or 'json'"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Whether to include chunk metadata"
    )
    
    @validator('strategy')
    def validate_strategy(cls, v):
        allowed_strategies = ['sliding_window', 'semantic', 'sentence', 'paragraph']
        if v not in allowed_strategies:
            raise ValueError(f"Strategy must be one of: {allowed_strategies}")
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        allowed_formats = ['list', 'json']
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v


class TextChunk(BaseModel):
    """Represents a single text chunk with metadata."""
    
    id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    start_char: int = Field(..., description="Starting character position in original text")
    end_char: int = Field(..., description="Ending character position in original text")
    chunk_index: int = Field(..., description="Sequential chunk number")
    word_count: int = Field(..., description="Number of words in chunk")
    char_count: int = Field(..., description="Number of characters in chunk")
    
    # Optional metadata for transcript chunks
    start_time: Optional[float] = Field(None, description="Start timestamp (for transcript chunks)")
    end_time: Optional[float] = Field(None, description="End timestamp (for transcript chunks)")
    speaker: Optional[str] = Field(None, description="Speaker name (for transcript chunks)")
    
    # Islamic content specific metadata
    contains_arabic: bool = Field(default=False, description="Whether chunk contains Arabic text")
    contains_quran: bool = Field(default=False, description="Whether chunk contains Quranic verses")
    contains_hadith: bool = Field(default=False, description="Whether chunk contains Hadith references")
    topic_keywords: List[str] = Field(default_factory=list, description="Extracted topic keywords")


class TextChunkerTool(BaseTool):
    """CrewAI tool for chunking text content into optimized segments."""
    
    name: str = "Text Chunker Tool"
    description: str = (
        "Chunks text content into optimized segments for embedding and retrieval. "
        "Supports multiple chunking strategies with semantic boundary preservation "
        "and special handling for Islamic content including Arabic text, Quranic verses, and Hadith."
    )
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Text Chunker Tool.
        
        Args:
            config_path: Path to YAML configuration file
        """
        super().__init__()
        self._config_path = config_path or self._find_config_path()
        self._config = self._load_config()
        
        # Islamic content patterns
        self._arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        self._quran_patterns = [
            re.compile(r'\b(?:Quran|Qur\'an|Koran)\b', re.IGNORECASE),
            re.compile(r'\b(?:Surah|Sura)\s+\w+', re.IGNORECASE),
            re.compile(r'\b(?:Ayah|Verse)\s+\d+', re.IGNORECASE),
            re.compile(r'\b\d+:\d+\b')  # Verse references like 2:255
        ]
        self._hadith_patterns = [
            re.compile(r'\b(?:Hadith|Hadeeth)\b', re.IGNORECASE),
            re.compile(r'\b(?:Bukhari|Muslim|Tirmidhi|Abu Dawud|Nasa\'i|Ibn Majah)\b', re.IGNORECASE),
            re.compile(r'\b(?:Sahih|Sunan)\s+\w+', re.IGNORECASE)
        ]
        
        # Islamic topic keywords
        self._islamic_keywords = [
            'Allah', 'Prophet', 'Muhammad', 'Islam', 'Muslim', 'Quran', 'Hadith',
            'Salah', 'Prayer', 'Zakat', 'Hajj', 'Ramadan', 'Fasting', 'Sunnah',
            'Dua', 'Dhikr', 'Tawhid', 'Iman', 'Taqwa', 'Jihad', 'Ummah',
            'Masjid', 'Mosque', 'Imam', 'Scholar', 'Fatwa', 'Halal', 'Haram'
        ]
    
    @property
    def config(self):
        """Get the configuration."""
        return self._config
    
    def _find_config_path(self) -> str:
        """Find the configuration file path."""
        possible_paths = [
            "config/crew_config.yaml",
            "../config/crew_config.yaml",
            "akhi_crewai/config/crew_config.yaml",
            "/data/work/dev/akhi_data_builder/akhi_crewai/config/crew_config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find crew_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('text_processing', {}).get('chunking', {})
        except Exception as e:
            print(f"Warning: Could not load config from {self._config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default chunking configuration."""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'separator': '\n\n',
            'preserve_sentences': True
        }
    
    def _detect_islamic_content(self, text: str) -> Dict[str, Any]:
        """Detect Islamic content patterns in text."""
        contains_arabic = bool(self._arabic_pattern.search(text))
        contains_quran = any(pattern.search(text) for pattern in self._quran_patterns)
        contains_hadith = any(pattern.search(text) for pattern in self._hadith_patterns)
        
        # Extract topic keywords
        topic_keywords = []
        text_lower = text.lower()
        for keyword in self._islamic_keywords:
            if keyword.lower() in text_lower:
                topic_keywords.append(keyword)
        
        return {
            'contains_arabic': contains_arabic,
            'contains_quran': contains_quran,
            'contains_hadith': contains_hadith,
            'topic_keywords': topic_keywords
        }
    
    def _parse_transcript_input(self, text: str) -> Dict[str, Any]:
        """Parse transcript input (JSON or plain text)."""
        try:
            # Try to parse as JSON transcript
            transcript_data = json.loads(text)
            if isinstance(transcript_data, dict) and 'transcript_text' in transcript_data:
                return {
                    'text': transcript_data['transcript_text'],
                    'segments': transcript_data.get('segments', []),
                    'metadata': transcript_data.get('metadata', {}),
                    'is_transcript': True
                }
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Treat as plain text
        return {
            'text': text,
            'segments': [],
            'metadata': {},
            'is_transcript': False
        }
    
    def _sliding_window_chunking(self, text: str, chunk_size: int, chunk_overlap: int, 
                                preserve_sentences: bool = True) -> List[str]:
        """Implement sliding window chunking strategy."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size, text_length)
            
            # Extract chunk
            chunk = text[start:end]
            
            # Preserve sentence boundaries if requested
            if preserve_sentences and end < text_length:
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start, end - int(chunk_size * 0.2))
                sentence_endings = []
                
                for i in range(search_start, end):
                    if text[i] in '.!?\n':
                        # Check if it's a real sentence ending
                        if i + 1 < text_length and text[i + 1] in ' \n\t':
                            sentence_endings.append(i + 1)
                
                if sentence_endings:
                    end = sentence_endings[-1]
                    chunk = text[start:end]
            
            # Clean up chunk
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            if end >= text_length:
                break
            
            start = max(start + 1, end - chunk_overlap)
        
        return chunks
    
    def _semantic_chunking(self, text: str, chunk_size: int) -> List[str]:
        """Implement semantic chunking based on paragraphs and topics."""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Paragraph itself is too long, split it
                    chunks.extend(self._sliding_window_chunking(paragraph, chunk_size, 0, True))
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sentence_chunking(self, text: str, chunk_size: int) -> List[str]:
        """Implement sentence-based chunking."""
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Sentence itself is too long
                    chunks.append(sentence)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _paragraph_chunking(self, text: str, chunk_size: int) -> List[str]:
        """Implement paragraph-based chunking."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                chunks.append(paragraph)
            else:
                # Split long paragraphs
                chunks.extend(self._sliding_window_chunking(paragraph, chunk_size, 0, True))
        
        return chunks
    
    def _create_chunks_with_metadata(self, text_chunks: List[str], original_text: str,
                                   parsed_input: Dict[str, Any]) -> List[TextChunk]:
        """Create TextChunk objects with metadata."""
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(text_chunks):
            # Find chunk position in original text
            start_char = original_text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            
            end_char = start_char + len(chunk_text)
            current_pos = end_char
            
            # Detect Islamic content
            islamic_metadata = self._detect_islamic_content(chunk_text)
            
            # Create chunk object
            chunk = TextChunk(
                id=f"chunk_{i:04d}",
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=i,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                **islamic_metadata
            )
            
            # Add transcript-specific metadata if available
            if parsed_input['is_transcript'] and parsed_input['segments']:
                # Find overlapping segments
                overlapping_segments = []
                for segment in parsed_input['segments']:
                    segment_text = segment.get('text', '')
                    if segment_text in chunk_text or chunk_text in segment_text:
                        overlapping_segments.append(segment)
                
                if overlapping_segments:
                    chunk.start_time = min(s.get('start', 0) for s in overlapping_segments)
                    chunk.end_time = max(s.get('end', 0) for s in overlapping_segments)
                    
                    # Extract speaker if consistent
                    speakers = set(s.get('speaker') for s in overlapping_segments if s.get('speaker'))
                    if len(speakers) == 1:
                        chunk.speaker = speakers.pop()
            
            chunks.append(chunk)
        
        return chunks
    
    def _run(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None,
             separator: Optional[str] = None, preserve_sentences: Optional[bool] = None,
             strategy: str = "sliding_window", output_format: str = "list",
             include_metadata: bool = True, **kwargs) -> Union[List[Dict[str, Any]], str]:
        """Execute the text chunking operation.
        
        Args:
            text: Text content to chunk
            chunk_size: Maximum characters per chunk (overrides config)
            chunk_overlap: Characters to overlap between chunks (overrides config)
            separator: Text separator for chunking (overrides config)
            preserve_sentences: Whether to preserve sentence boundaries (overrides config)
            strategy: Chunking strategy ('sliding_window', 'semantic', 'sentence', 'paragraph')
            output_format: Output format ('list' or 'json')
            include_metadata: Whether to include chunk metadata
            **kwargs: Additional arguments
        
        Returns:
            List of chunk dictionaries or JSON string
        """
        try:
            # Parse input
            parsed_input = self._parse_transcript_input(text)
            text_to_chunk = parsed_input['text']
            
            if not text_to_chunk.strip():
                return [] if output_format == "list" else "[]"
            
            # Get chunking parameters
            chunk_size = chunk_size or self._config.get('chunk_size', 1000)
            chunk_overlap = chunk_overlap or self._config.get('chunk_overlap', 200)
            separator = separator or self._config.get('separator', '\n\n')
            preserve_sentences = preserve_sentences if preserve_sentences is not None else self._config.get('preserve_sentences', True)
            
            # Apply chunking strategy
            if strategy == "sliding_window":
                text_chunks = self._sliding_window_chunking(text_to_chunk, chunk_size, chunk_overlap, preserve_sentences)
            elif strategy == "semantic":
                text_chunks = self._semantic_chunking(text_to_chunk, chunk_size)
            elif strategy == "sentence":
                text_chunks = self._sentence_chunking(text_to_chunk, chunk_size)
            elif strategy == "paragraph":
                text_chunks = self._paragraph_chunking(text_to_chunk, chunk_size)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
            
            # Create chunks with metadata
            if include_metadata:
                chunks = self._create_chunks_with_metadata(text_chunks, text_to_chunk, parsed_input)
                result = [chunk.dict() for chunk in chunks]
            else:
                result = [{'text': chunk, 'index': i} for i, chunk in enumerate(text_chunks)]
            
            # Return in requested format
            if output_format == "json":
                return json.dumps(result, indent=2, ensure_ascii=False)
            else:
                return result
        
        except Exception as e:
            error_msg = f"Error in text chunking: {str(e)}"
            print(error_msg)
            if output_format == "json":
                return json.dumps({"error": error_msg})
            else:
                return [{"error": error_msg}]


# Example usage and testing
if __name__ == "__main__":
    # Initialize the tool
    chunker = TextChunkerTool()
    
    # Test with sample text
    sample_text = """
    In the name of Allah, the Most Gracious, the Most Merciful.
    
    The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him) over a period of 23 years. It contains 114 chapters called Surahs, each containing verses called Ayahs.
    
    One of the most famous verses is Ayat al-Kursi (Quran 2:255), which speaks about the greatness of Allah. The verse states: "Allah - there is no deity except Him, the Ever-Living, the Sustainer of existence."
    
    The five daily prayers (Salah) are one of the Five Pillars of Islam. Muslims are required to pray five times a day: Fajr (dawn), Dhuhr (midday), Asr (afternoon), Maghrib (sunset), and Isha (night).
    
    The Prophet Muhammad (peace be upon him) said in a Hadith recorded by Bukhari: "The best of people are those who benefit others." This teaches us the importance of helping our fellow human beings.
    """
    
    print("=== Text Chunking Tool Demo ===")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Word count: {len(sample_text.split())} words")
    print()
    
    # Test different strategies
    strategies = ['sliding_window', 'semantic', 'sentence', 'paragraph']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} CHUNKING ---")
        chunks = chunker._run(
            text=sample_text,
            chunk_size=300,
            chunk_overlap=50,
            strategy=strategy,
            include_metadata=True
        )
        
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Word count: {chunk['word_count']}")
            print(f"  Contains Arabic: {chunk['contains_arabic']}")
            print(f"  Contains Quran: {chunk['contains_quran']}")
            print(f"  Contains Hadith: {chunk['contains_hadith']}")
            print(f"  Topic keywords: {chunk['topic_keywords'][:3]}")
    
    print("\n=== Demo Complete ===")