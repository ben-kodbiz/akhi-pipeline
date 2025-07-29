#!/usr/bin/env python3
"""
Summarization Tool for CrewAI Agentic System

This tool provides text summarization capabilities using local LLM models.
Optimized for Islamic educational content with context-aware summarization.

Features:
- Local LLM integration (llama-cpp-python)
- Context-aware summarization
- Configurable summary length
- Multiple summarization strategies
- Islamic content optimization
- CrewAI BaseTool integration
- YAML configuration support

Author: Assistant
Date: 2024
"""

import os
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# CrewAI imports
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, validator

# LLM imports
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Install with: pip install llama-cpp-python")


class SummarizationInput(BaseModel):
    """Input schema for summarization requests."""
    
    text: str = Field(
        description="Text content to summarize",
        min_length=10
    )
    
    summary_length: str = Field(
        default="medium",
        description="Desired summary length: 'short', 'medium', 'long', or 'custom'"
    )
    
    max_words: Optional[int] = Field(
        default=None,
        description="Maximum words for custom length (only used when summary_length='custom')",
        ge=10,
        le=1000
    )
    
    strategy: str = Field(
        default="extractive",
        description="Summarization strategy: 'extractive', 'abstractive', 'hybrid'"
    )
    
    focus_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords to focus on during summarization"
    )
    
    preserve_islamic_terms: bool = Field(
        default=True,
        description="Whether to preserve Islamic terminology and concepts"
    )
    
    include_key_points: bool = Field(
        default=True,
        description="Whether to include key points in the summary"
    )
    
    context: Optional[str] = Field(
        default=None,
        description="Additional context for better summarization"
    )
    
    @validator('summary_length')
    def validate_summary_length(cls, v):
        allowed = ['short', 'medium', 'long', 'custom']
        if v not in allowed:
            raise ValueError(f"summary_length must be one of {allowed}")
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        allowed = ['extractive', 'abstractive', 'hybrid']
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v


class SummarizationOutput(BaseModel):
    """Output schema for summarization results."""
    
    success: bool = Field(description="Whether summarization was successful")
    summary: str = Field(description="Generated summary text")
    key_points: List[str] = Field(description="Extracted key points")
    word_count: int = Field(description="Word count of the summary")
    compression_ratio: float = Field(description="Compression ratio (original/summary)")
    strategy_used: str = Field(description="Summarization strategy that was used")
    islamic_terms_preserved: List[str] = Field(description="Islamic terms found and preserved")
    processing_time: float = Field(description="Processing time in seconds")
    confidence_score: float = Field(description="Confidence score of the summary quality")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    message: str = Field(description="Status message or error details")


class SummarizerTool(BaseTool):
    """Tool for generating summaries from text content."""
    
    name: str = "summarizer"
    description: str = (
        "Generates summaries from text content using local LLM models. "
        "Supports multiple summarization strategies and is optimized for Islamic content."
    )
    args_schema: type = SummarizationInput
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the summarization tool."""
        super().__init__()
        
        # Load configuration
        self._config = self._load_config(config_path)
        
        # Initialize model variables
        self._llm_model = None
        self._current_model_path = None
        
        # Islamic terms to preserve
        self._islamic_terms = {
            'Allah', 'Quran', 'Qur\'an', 'Hadith', 'Sunnah', 'Prophet', 'Muhammad',
            'Islam', 'Muslim', 'Salah', 'Zakat', 'Hajj', 'Ramadan', 'Ummah',
            'Jihad', 'Sharia', 'Imam', 'Masjid', 'Dua', 'Dhikr', 'Tawhid',
            'Shirk', 'Halal', 'Haram', 'Makruh', 'Mustahabb', 'Fiqh', 'Tafsir',
            'Sahaba', 'Tabi\'in', 'Ulama', 'Sheikh', 'Mufti', 'Fatwa'
        }
        
        print(f"SummarizerTool initialized with config: {config_path or 'default'}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            'model': {
                'path': None,  # Path to local LLM model
                'context_length': 2048,
                'temperature': 0.3,
                'max_tokens': 512,
                'top_p': 0.9,
                'top_k': 40
            },
            'summarization': {
                'short_words': 50,
                'medium_words': 150,
                'long_words': 300,
                'min_compression_ratio': 0.1,
                'max_compression_ratio': 0.8
            },
            'islamic_content': {
                'preserve_terms': True,
                'context_aware': True,
                'respect_religious_context': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in file_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    
    def _load_llm_model(self, model_path: Optional[str] = None) -> Optional[object]:
        """Load or reload the LLM model."""
        if not LLAMA_CPP_AVAILABLE:
            return None
        
        target_path = model_path or self.config['model']['path']
        
        if not target_path or not Path(target_path).exists():
            print("Warning: No valid LLM model path provided or model file not found")
            return None
        
        if self._llm_model is None or self._current_model_path != target_path:
            try:
                print(f"Loading LLM model: {target_path}")
                self._llm_model = Llama(
                    model_path=target_path,
                    n_ctx=self._config['model']['context_length'],
                    verbose=False
                )
                self._current_model_path = target_path
                print("LLM model loaded successfully")
            except Exception as e:
                print(f"Error loading LLM model: {e}")
                return None
        
        return self._llm_model
    
    def _extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """Extract key points from text using simple heuristics."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Score sentences based on length and keyword presence
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence.split())  # Base score on word count
            
            # Boost score for Islamic terms
            for term in self._islamic_terms:
                if term.lower() in sentence.lower():
                    score += 10
            
            # Boost score for question words and important indicators
            important_words = ['important', 'key', 'main', 'primary', 'essential', 'crucial']
            for word in important_words:
                if word in sentence.lower():
                    score += 5
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and return top points
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sentence for _, sentence in scored_sentences[:max_points]]
    
    def _find_islamic_terms(self, text: str) -> List[str]:
        """Find Islamic terms in the text."""
        found_terms = []
        text_lower = text.lower()
        
        for term in self._islamic_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        
        return list(set(found_terms))
    
    def _extractive_summarization(self, text: str, target_words: int) -> str:
        """Perform extractive summarization by selecting important sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return text[:target_words * 5]  # Fallback
        
        # Score sentences
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.split()
            
            # Position score (earlier sentences get higher score)
            score += max(0, 10 - i)
            
            # Length score (prefer medium-length sentences)
            if 10 <= len(words) <= 30:
                score += 5
            
            # Islamic terms score
            for term in self._islamic_terms:
                if term.lower() in sentence.lower():
                    score += 8
            
            # Important words score
            important_words = ['important', 'key', 'main', 'therefore', 'because', 'however']
            for word in important_words:
                if word in sentence.lower():
                    score += 3
            
            scored_sentences.append((score, sentence, len(words)))
        
        # Sort by score and select sentences until target word count
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        selected_sentences = []
        current_words = 0
        
        for score, sentence, word_count in scored_sentences:
            if current_words + word_count <= target_words:
                selected_sentences.append(sentence)
                current_words += word_count
            elif current_words < target_words * 0.8:  # Allow some flexibility
                selected_sentences.append(sentence)
                break
        
        return '. '.join(selected_sentences) + '.'
    
    def _abstractive_summarization(self, text: str, target_words: int, context: Optional[str] = None) -> str:
        """Perform abstractive summarization using LLM."""
        llm = self._load_llm_model()
        
        if not llm:
            # Fallback to extractive if LLM not available
            return self._extractive_summarization(text, target_words)
        
        # Prepare prompt
        context_part = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""You are an expert in Islamic education and content summarization. Please provide a concise summary of the following text.

{context_part}Text to summarize:
{text}

Requirements:
- Maximum {target_words} words
- Preserve Islamic terminology and concepts
- Focus on key teachings and main points
- Maintain respectful tone for religious content

Summary:"""
        
        try:
            response = llm(
                prompt,
                max_tokens=self._config['model']['max_tokens'],
                temperature=self._config['model']['temperature'],
                top_p=self._config['model']['top_p'],
                top_k=self._config['model']['top_k'],
                stop=["\n\n", "Text to summarize:", "Requirements:"]
            )
            
            summary = response['choices'][0]['text'].strip()
            
            # Clean up the summary
            summary = re.sub(r'^Summary:?\s*', '', summary, flags=re.IGNORECASE)
            summary = summary.strip()
            
            return summary if summary else self._extractive_summarization(text, target_words)
            
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            return self._extractive_summarization(text, target_words)
    
    def _hybrid_summarization(self, text: str, target_words: int, context: Optional[str] = None) -> str:
        """Perform hybrid summarization combining extractive and abstractive approaches."""
        # First, use extractive to get key sentences (more words than target)
        extractive_words = min(target_words * 2, len(text.split()))
        extractive_summary = self._extractive_summarization(text, extractive_words)
        
        # Then use abstractive to refine and compress
        if len(extractive_summary.split()) > target_words:
            return self._abstractive_summarization(extractive_summary, target_words, context)
        else:
            return extractive_summary
    
    def _calculate_confidence_score(self, original_text: str, summary: str, strategy: str) -> float:
        """Calculate confidence score for the summary quality."""
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        if original_words == 0 or summary_words == 0:
            return 0.0
        
        # Base score on compression ratio
        compression_ratio = summary_words / original_words
        if 0.1 <= compression_ratio <= 0.5:
            base_score = 0.8
        elif 0.05 <= compression_ratio <= 0.8:
            base_score = 0.6
        else:
            base_score = 0.4
        
        # Boost for Islamic terms preservation
        original_islamic_terms = self._find_islamic_terms(original_text)
        summary_islamic_terms = self._find_islamic_terms(summary)
        
        if original_islamic_terms:
            preservation_ratio = len(summary_islamic_terms) / len(original_islamic_terms)
            base_score += preservation_ratio * 0.2
        
        # Strategy-specific adjustments
        if strategy == 'abstractive':
            base_score += 0.1  # Abstractive generally produces better summaries
        elif strategy == 'hybrid':
            base_score += 0.05
        
        return min(1.0, base_score)
    
    def _run(self, **kwargs) -> str:
        """Execute the summarization tool."""
        start_time = self._get_current_time()
        
        try:
            # Validate input using Pydantic
            input_data = SummarizationInput(**kwargs)
            
            # Determine target word count
            if input_data.summary_length == 'custom' and input_data.max_words:
                target_words = input_data.max_words
            else:
                length_mapping = {
                    'short': self._config['summarization']['short_words'],
                    'medium': self._config['summarization']['medium_words'],
                    'long': self._config['summarization']['long_words']
                }
                target_words = length_mapping[input_data.summary_length]
            
            # Generate summary based on strategy
            if input_data.strategy == 'extractive':
                summary = self._extractive_summarization(input_data.text, target_words)
            elif input_data.strategy == 'abstractive':
                summary = self._abstractive_summarization(input_data.text, target_words, input_data.context)
            else:  # hybrid
                summary = self._hybrid_summarization(input_data.text, target_words, input_data.context)
            
            # Extract key points if requested
            key_points = []
            if input_data.include_key_points:
                key_points = self._extract_key_points(input_data.text)
            
            # Find preserved Islamic terms
            islamic_terms_preserved = []
            if input_data.preserve_islamic_terms:
                islamic_terms_preserved = self._find_islamic_terms(summary)
            
            # Calculate metrics
            original_words = len(input_data.text.split())
            summary_words = len(summary.split())
            compression_ratio = summary_words / original_words if original_words > 0 else 0
            confidence_score = self._calculate_confidence_score(input_data.text, summary, input_data.strategy)
            processing_time = self._get_current_time() - start_time
            
            # Create output
            result = SummarizationOutput(
                success=True,
                summary=summary,
                key_points=key_points,
                word_count=summary_words,
                compression_ratio=compression_ratio,
                strategy_used=input_data.strategy,
                islamic_terms_preserved=islamic_terms_preserved,
                processing_time=processing_time,
                confidence_score=confidence_score,
                metadata={
                    'original_word_count': original_words,
                    'target_words': target_words,
                    'summary_length': input_data.summary_length,
                    'llm_available': LLAMA_CPP_AVAILABLE and self._llm_model is not None,
                    'timestamp': datetime.now().isoformat()
                },
                message="Summary generated successfully"
            )
            
            return json.dumps(result.dict(), indent=2, ensure_ascii=False)
            
        except Exception as e:
            error_result = SummarizationOutput(
                success=False,
                summary="",
                key_points=[],
                word_count=0,
                compression_ratio=0.0,
                strategy_used="none",
                islamic_terms_preserved=[],
                processing_time=self._get_current_time() - start_time,
                confidence_score=0.0,
                metadata={},
                message=f"Error in summarization: {str(e)}"
            )
            
            return json.dumps(error_result.dict(), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    tool = SummarizerTool()
    
    sample_text = """
    In the name of Allah, the Most Gracious, the Most Merciful. The Quran is the holy book of Islam, 
    revealed to Prophet Muhammad (peace be upon him) through the angel Gabriel. It contains guidance 
    for all aspects of life and is considered the final revelation from Allah to humanity. The Quran 
    emphasizes the importance of prayer (Salah), charity (Zakat), fasting during Ramadan, and the 
    pilgrimage to Mecca (Hajj). These are known as the Five Pillars of Islam. The book also teaches 
    about the Day of Judgment, the importance of good deeds, and the concept of Tawhid (the oneness 
    of Allah). Muslims believe that the Quran is the direct word of Allah and has remained unchanged 
    since its revelation over 1400 years ago.
    """
    
    result = tool._run(
        text=sample_text,
        summary_length="medium",
        strategy="extractive",
        preserve_islamic_terms=True,
        include_key_points=True
    )
    
    print("Summarization Tool Demo:")
    print("=" * 50)
    print(result)