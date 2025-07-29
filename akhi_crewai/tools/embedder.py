#!/usr/bin/env python3
"""
Embedding Tool for CrewAI Agentic System

This tool generates vector embeddings from text chunks using sentence-transformers.
Optimized for Islamic educational content with support for Arabic text.

Features:
- Multiple embedding model support
- Batch processing for efficiency
- GPU/CPU device selection
- Dimension consistency checking
- Islamic content optimization
- CrewAI BaseTool integration
- YAML configuration support

Author: Assistant
Date: 2024
"""

import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# CrewAI imports
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, validator

# Embedding imports
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class EmbeddingInput(BaseModel):
    """Input schema for embedding generation."""
    
    text_chunks: Union[List[str], List[Dict[str, Any]]] = Field(
        description="List of text chunks to embed (strings or chunk dictionaries)"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Embedding model name (overrides config)"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for computation (cpu/cuda/auto)"
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for processing"
    )
    normalize_embeddings: Optional[bool] = Field(
        default=None,
        description="Whether to normalize embeddings"
    )
    output_format: Optional[str] = Field(
        default="list",
        description="Output format: 'list', 'numpy', 'dict'"
    )
    include_metadata: Optional[bool] = Field(
        default=True,
        description="Include chunk metadata in output"
    )
    
    @validator('text_chunks')
    def validate_text_chunks(cls, v):
        if not v:
            raise ValueError("text_chunks cannot be empty")
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        if v not in ['list', 'numpy', 'dict']:
            raise ValueError("output_format must be 'list', 'numpy', or 'dict'")
        return v


class EmbeddingOutput(BaseModel):
    """Output schema for generated embeddings."""
    
    embedding_id: str
    text: str
    embedding: List[float]
    dimension: int
    model_name: str
    chunk_index: int
    
    # Optional metadata from chunks
    chunk_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    
    # Timestamp information
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    
    # Islamic content flags
    contains_arabic: Optional[bool] = None
    contains_quran: Optional[bool] = None
    contains_hadith: Optional[bool] = None
    topic_keywords: Optional[List[str]] = None
    
    # Processing metadata
    created_at: str
    processing_time: float


class EmbedderTool(BaseTool):
    """CrewAI tool for generating text embeddings."""
    
    name: str = "Text Embedder"
    description: str = (
        "Generates vector embeddings from text chunks using sentence-transformers. "
        "Supports batch processing, multiple models, and Islamic content optimization."
    )
    args_schema: type = EmbeddingInput
    
    def __init__(self, config_path: str = None):
        """Initialize the embedding tool."""
        super().__init__()
        
        # Set configuration path
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "crew_config.yaml"
        
        self._config_path = str(config_path)
        self._config = self._load_config()
        
        # Initialize model (lazy loading)
        self._model = None
        self._tokenizer = None
        self._current_model_name = None
        
        # Device detection
        self._device = self._detect_device()
        
        print(f"EmbedderTool initialized with config: {self._config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('text_processing', {}).get('embedding', {})
        except Exception as e:
            print(f"Warning: Could not load config from {self._config_path}: {e}")
            return {
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'auto',
                'batch_size': 32,
                'normalize_embeddings': True
            }
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        device_config = self._config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            return device_config
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    
    def _load_model(self, model_name: str, device: str) -> SentenceTransformer:
        """Load or reload the embedding model."""
        if self._model is None or self._current_model_name != model_name:
            print(f"Loading embedding model: {model_name} on {device}")
            
            try:
                self._model = SentenceTransformer(model_name, device=device)
                self._current_model_name = model_name
                
                # Load tokenizer for additional text analysis
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                except:
                    self._tokenizer = None
                    
                print(f"Model loaded successfully. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
                
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                print("Falling back to default model: all-MiniLM-L6-v2")
                self._model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                self._current_model_name = 'all-MiniLM-L6-v2'
        
        return self._model
    
    def _extract_text_from_chunks(self, chunks: Union[List[str], List[Dict[str, Any]]]) -> List[str]:
        """Extract text content from chunks (handles both string and dict formats)."""
        texts = []
        
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                # Extract text from chunk dictionary
                text = chunk.get('text', '')
                if not text:
                    # Try alternative keys
                    text = chunk.get('content', chunk.get('chunk_text', str(chunk)))
                texts.append(text)
            else:
                texts.append(str(chunk))
        
        return texts
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality."""
        # Basic text cleaning
        text = text.strip()
        
        # Handle Arabic text - preserve RTL markers and diacritics
        # Remove excessive whitespace but preserve Arabic formatting
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure text is not empty
        if not text:
            text = "[Empty text]"
        
        return text
    
    def _generate_embeddings(
        self, 
        texts: List[str], 
        model: SentenceTransformer, 
        batch_size: int, 
        normalize: bool
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Generate embeddings in batches
        embeddings = model.encode(
            processed_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(processed_texts) > 10,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def _create_embedding_output(
        self,
        text: str,
        embedding: np.ndarray,
        chunk_index: int,
        model_name: str,
        chunk_metadata: Dict[str, Any],
        processing_time: float
    ) -> EmbeddingOutput:
        """Create structured embedding output."""
        
        # Generate unique embedding ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        embedding_id = f"emb_{timestamp}_{chunk_index:04d}"
        
        # Handle empty text for display
        display_text = text if text.strip() else "[Empty text]"
        
        # Extract metadata from chunk if available
        chunk_id = chunk_metadata.get('id')
        start_char = chunk_metadata.get('start_char')
        end_char = chunk_metadata.get('end_char')
        word_count = chunk_metadata.get('word_count')
        char_count = chunk_metadata.get('char_count')
        
        # Timestamp information
        start_time = chunk_metadata.get('start_time')
        end_time = chunk_metadata.get('end_time')
        speaker = chunk_metadata.get('speaker')
        
        # Islamic content flags
        contains_arabic = chunk_metadata.get('contains_arabic')
        contains_quran = chunk_metadata.get('contains_quran')
        contains_hadith = chunk_metadata.get('contains_hadith')
        topic_keywords = chunk_metadata.get('topic_keywords')
        
        return EmbeddingOutput(
            embedding_id=embedding_id,
            text=display_text,
            embedding=embedding.tolist(),
            dimension=len(embedding),
            model_name=model_name,
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            word_count=word_count,
            char_count=char_count,
            start_time=start_time,
            end_time=end_time,
            speaker=speaker,
            contains_arabic=contains_arabic,
            contains_quran=contains_quran,
            contains_hadith=contains_hadith,
            topic_keywords=topic_keywords,
            created_at=datetime.now().isoformat(),
            processing_time=processing_time
        )
    
    def _format_output(
        self, 
        embedding_outputs: List[EmbeddingOutput], 
        output_format: str
    ) -> Union[List[Dict], np.ndarray, Dict[str, Any]]:
        """Format the output according to the specified format."""
        
        if output_format == "list":
            return [output.dict() for output in embedding_outputs]
        
        elif output_format == "numpy":
            embeddings = np.array([output.embedding for output in embedding_outputs])
            return embeddings
        
        elif output_format == "dict":
            result = {
                'embeddings': [output.embedding for output in embedding_outputs],
                'texts': [output.text for output in embedding_outputs],
                'metadata': [
                    {
                        'embedding_id': output.embedding_id,
                        'chunk_index': output.chunk_index,
                        'dimension': output.dimension,
                        'model_name': output.model_name,
                        'chunk_id': output.chunk_id,
                        'word_count': output.word_count,
                        'char_count': output.char_count,
                        'start_time': output.start_time,
                        'end_time': output.end_time,
                        'speaker': output.speaker,
                        'contains_arabic': output.contains_arabic,
                        'contains_quran': output.contains_quran,
                        'contains_hadith': output.contains_hadith,
                        'topic_keywords': output.topic_keywords,
                        'created_at': output.created_at,
                        'processing_time': output.processing_time
                    }
                    for output in embedding_outputs
                ],
                'summary': {
                    'total_chunks': len(embedding_outputs),
                    'model_name': embedding_outputs[0].model_name if embedding_outputs else None,
                    'dimension': embedding_outputs[0].dimension if embedding_outputs else None,
                    'total_processing_time': sum(output.processing_time for output in embedding_outputs)
                }
            }
            return result
        
        else:
            return [output.dict() for output in embedding_outputs]
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()
    
    def _run(
        self,
        text_chunks: Union[List[str], List[Dict[str, Any]]],
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
        output_format: Optional[str] = "list",
        include_metadata: Optional[bool] = True
    ) -> Union[List[Dict], np.ndarray, Dict[str, Any]]:
        """Generate embeddings for text chunks."""
        
        start_time = datetime.now()
        
        # Use config defaults if parameters not provided
        model_name = model_name or self._config.get('model_name', 'all-MiniLM-L6-v2')
        device = device or self._device
        batch_size = batch_size or self._config.get('batch_size', 32)
        normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else self._config.get('normalize_embeddings', True)
        
        print(f"Generating embeddings for {len(text_chunks)} chunks using {model_name}")
        
        try:
            # Load model
            model = self._load_model(model_name, device)
            
            # Extract texts and metadata
            texts = self._extract_text_from_chunks(text_chunks)
            
            # Prepare metadata for each chunk
            chunk_metadata_list = []
            for i, chunk in enumerate(text_chunks):
                if isinstance(chunk, dict):
                    chunk_metadata_list.append(chunk)
                else:
                    chunk_metadata_list.append({'text': chunk, 'index': i})
            
            # Generate embeddings
            embeddings = self._generate_embeddings(
                texts, model, batch_size, normalize_embeddings
            )
            
            # Create structured outputs
            embedding_outputs = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, chunk_metadata_list)):
                chunk_processing_time = (datetime.now() - start_time).total_seconds() / len(texts)
                
                output = self._create_embedding_output(
                    text=text,
                    embedding=embedding,
                    chunk_index=i,
                    model_name=model_name,
                    chunk_metadata=metadata if include_metadata else {},
                    processing_time=chunk_processing_time
                )
                embedding_outputs.append(output)
            
            # Format output
            result = self._format_output(embedding_outputs, output_format)
            
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"Embedding generation completed in {total_time:.2f} seconds")
            print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)


if __name__ == "__main__":
    # Demo usage
    print("Text Embedder Tool - Demo")
    print("=" * 40)
    
    # Initialize tool
    embedder = EmbedderTool()
    
    # Sample Islamic text chunks
    sample_chunks = [
        "بسم الله الرحمن الرحيم - In the name of Allah, the Most Gracious, the Most Merciful.",
        "The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him).",
        "The five daily prayers (Salah) are fundamental to Islamic practice.",
        "Zakat is the third pillar of Islam, involving charity to those in need."
    ]
    
    # Generate embeddings
    try:
        embeddings = embedder._run(
            text_chunks=sample_chunks,
            output_format="dict",
            include_metadata=True
        )
        
        print(f"\nGenerated embeddings for {embeddings['summary']['total_chunks']} chunks")
        print(f"Model: {embeddings['summary']['model_name']}")
        print(f"Dimension: {embeddings['summary']['dimension']}")
        print(f"Processing time: {embeddings['summary']['total_processing_time']:.3f}s")
        
        # Show first embedding info
        if embeddings['metadata']:
            first_meta = embeddings['metadata'][0]
            print(f"\nFirst chunk: '{embeddings['texts'][0][:50]}...'")
            print(f"Embedding ID: {first_meta['embedding_id']}")
            print(f"Embedding preview: {embeddings['embeddings'][0][:5]}...")
        
    except Exception as e:
        print(f"Demo failed: {e}")