#!/usr/bin/env python3
"""
Answer Generation Tool for CrewAI Agentic System

This tool provides RAG-based question answering capabilities using retrieved context
and local LLM models. Optimized for Islamic educational content.

Features:
- RAG-based question answering
- Context injection from retrieved chunks
- Citation generation with source tracking
- Confidence scoring
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


class AnswerGenerationInput(BaseModel):
    """Input schema for answer generation requests."""
    
    question: str = Field(
        description="Question to answer",
        min_length=5
    )
    
    context_chunks: List[Dict[str, Any]] = Field(
        description="Retrieved context chunks with metadata",
        min_items=1
    )
    
    max_answer_length: int = Field(
        default=300,
        description="Maximum length of the answer in words",
        ge=50,
        le=1000
    )
    
    include_citations: bool = Field(
        default=True,
        description="Whether to include source citations in the answer"
    )
    
    citation_style: str = Field(
        default="numbered",
        description="Citation style: 'numbered', 'inline', 'footnote'"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence threshold for providing an answer",
        ge=0.0,
        le=1.0
    )
    
    islamic_context_aware: bool = Field(
        default=True,
        description="Whether to apply Islamic context awareness"
    )
    
    language: str = Field(
        default="en",
        description="Response language: 'en', 'ar', 'auto'"
    )
    
    answer_style: str = Field(
        default="comprehensive",
        description="Answer style: 'brief', 'comprehensive', 'detailed'"
    )
    
    @validator('citation_style')
    def validate_citation_style(cls, v):
        allowed = ['numbered', 'inline', 'footnote']
        if v not in allowed:
            raise ValueError(f"citation_style must be one of {allowed}")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        allowed = ['en', 'ar', 'auto']
        if v not in allowed:
            raise ValueError(f"language must be one of {allowed}")
        return v
    
    @validator('answer_style')
    def validate_answer_style(cls, v):
        allowed = ['brief', 'comprehensive', 'detailed']
        if v not in allowed:
            raise ValueError(f"answer_style must be one of {allowed}")
        return v


class Citation(BaseModel):
    """Citation information for sources."""
    
    id: str = Field(description="Citation identifier")
    source: str = Field(description="Source name or file")
    chunk_text: str = Field(description="Relevant chunk text")
    confidence: float = Field(description="Relevance confidence score")
    metadata: Dict[str, Any] = Field(description="Additional metadata")


class AnswerGenerationOutput(BaseModel):
    """Output schema for answer generation results."""
    
    success: bool = Field(description="Whether answer generation was successful")
    answer: str = Field(description="Generated answer text")
    confidence_score: float = Field(description="Overall confidence in the answer")
    citations: List[Citation] = Field(description="Source citations used")
    sources_used: int = Field(description="Number of sources used")
    answer_length: int = Field(description="Length of answer in words")
    islamic_terms_used: List[str] = Field(description="Islamic terms referenced in answer")
    language_detected: str = Field(description="Detected language of the question")
    processing_time: float = Field(description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    message: str = Field(description="Status message or error details")


class AnswerGeneratorTool(BaseTool):
    """Tool for generating answers from questions using retrieved context."""
    
    name: str = "answer_generator"
    description: str = (
        "Generates answers to questions using retrieved context chunks and local LLM models. "
        "Provides citations and confidence scoring, optimized for Islamic content."
    )
    args_schema: type = AnswerGenerationInput
    config: Dict[str, Any] = {}
    islamic_terms: set = set()
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the answer generation tool."""
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize model variables
        self._llm_model = None
        self._current_model_path = None
        
        # Islamic terms for context awareness
        self.islamic_terms = {
            'Allah', 'Quran', 'Qur\'an', 'Hadith', 'Sunnah', 'Prophet', 'Muhammad',
            'Islam', 'Muslim', 'Salah', 'Zakat', 'Hajj', 'Ramadan', 'Ummah',
            'Jihad', 'Sharia', 'Imam', 'Masjid', 'Dua', 'Dhikr', 'Tawhid',
            'Shirk', 'Halal', 'Haram', 'Makruh', 'Mustahabb', 'Fiqh', 'Tafsir',
            'Sahaba', 'Tabi\'in', 'Ulama', 'Sheikh', 'Mufti', 'Fatwa'
        }
        
        print(f"AnswerGeneratorTool initialized with config: {config_path or 'default'}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            'model': {
                'path': None,  # Path to local LLM model
                'context_length': 4096,
                'temperature': 0.2,
                'max_tokens': 1024,
                'top_p': 0.9,
                'top_k': 40
            },
            'answer_generation': {
                'max_context_chunks': 5,
                'min_chunk_relevance': 0.3,
                'default_confidence_threshold': 0.5,
                'max_citation_text_length': 200
            },
            'islamic_content': {
                'context_aware': True,
                'respectful_language': True,
                'preserve_arabic_terms': True
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
                    n_ctx=self.config['model']['context_length'],
                    verbose=False
                )
                self._current_model_path = target_path
                print("LLM model loaded successfully")
            except Exception as e:
                print(f"Error loading LLM model: {e}")
                return None
        
        return self._llm_model
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        # Simple heuristic-based language detection
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.findall(r'[\w]', text))
        
        if total_chars == 0:
            return 'en'
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.3:
            return 'ar'
        else:
            return 'en'
    
    def _calculate_chunk_relevance(self, question: str, chunk: Dict[str, Any]) -> float:
        """Calculate relevance score between question and context chunk."""
        chunk_text = chunk.get('text', '')
        
        if not chunk_text:
            return 0.0
        
        question_words = set(question.lower().split())
        chunk_words = set(chunk_text.lower().split())
        
        # Calculate word overlap
        common_words = question_words.intersection(chunk_words)
        if not question_words:
            return 0.0
        
        base_score = len(common_words) / len(question_words)
        
        # Boost for Islamic terms
        islamic_boost = 0.0
        for term in self.islamic_terms:
            if term.lower() in question.lower() and term.lower() in chunk_text.lower():
                islamic_boost += 0.1
        
        # Boost for metadata relevance
        metadata_boost = 0.0
        metadata = chunk.get('metadata', {})
        
        if metadata.get('is_quran', False) and any(term in question.lower() for term in ['quran', 'qur\'an', 'verse']):
            metadata_boost += 0.2
        
        if metadata.get('is_hadith', False) and 'hadith' in question.lower():
            metadata_boost += 0.2
        
        return min(1.0, base_score + islamic_boost + metadata_boost)
    
    def _select_relevant_chunks(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the most relevant chunks for answering the question."""
        # Calculate relevance scores
        scored_chunks = []
        for chunk in chunks:
            relevance = self._calculate_chunk_relevance(question, chunk)
            if relevance >= self.config['answer_generation']['min_chunk_relevance']:
                scored_chunks.append((relevance, chunk))
        
        # Sort by relevance and take top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        max_chunks = self.config['answer_generation']['max_context_chunks']
        
        return [chunk for _, chunk in scored_chunks[:max_chunks]]
    
    def _create_citations(self, chunks: List[Dict[str, Any]], style: str) -> List[Citation]:
        """Create citations from context chunks."""
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Truncate chunk text for citation
            max_length = self.config['answer_generation']['max_citation_text_length']
            if len(chunk_text) > max_length:
                chunk_text = chunk_text[:max_length] + "..."
            
            citation = Citation(
                id=str(i),  # Always use numbered format for consistency
                source=metadata.get('source', f"Source {i}"),
                chunk_text=chunk_text,
                confidence=0.8,  # Default confidence for citations
                metadata=metadata
            )
            
            citations.append(citation)
        
        return citations
    
    def _format_citations_in_text(self, text: str, citations: List[Citation], style: str) -> str:
        """Format citations within the answer text."""
        if style == 'numbered':
            # Add numbered citations at the end
            citation_text = "\n\nSources:\n"
            for citation in citations:
                citation_text += f"[{citation.id}] {citation.source}\n"
            return text + citation_text
        
        elif style == 'inline':
            # Citations are already embedded in the text during generation
            return text
        
        elif style == 'footnote':
            # Add footnote-style citations
            citation_text = "\n\nReferences:\n"
            for citation in citations:
                citation_text += f"{citation.id}. {citation.source}: {citation.chunk_text[:100]}...\n"
            return text + citation_text
        
        return text
    
    def _generate_answer_with_llm(self, question: str, context_chunks: List[Dict[str, Any]], 
                                  max_length: int, style: str, language: str) -> str:
        """Generate answer using LLM with context."""
        llm = self._load_llm_model()
        
        if not llm:
            return self._generate_fallback_answer(question, context_chunks, max_length)
        
        # Prepare context
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk.get('text', '')
            source = chunk.get('metadata', {}).get('source', f'Source {i}')
            context_text += f"[{i}] From {source}: {chunk_text}\n\n"
        
        # Prepare prompt based on language and style
        if language == 'ar':
            system_prompt = "أنت مساعد خبير في التعليم الإسلامي. أجب على الأسئلة بدقة واحترام."
            instruction = "أجب على السؤال التالي بناءً على المصادر المقدمة:"
        else:
            system_prompt = "You are an expert Islamic education assistant. Answer questions accurately and respectfully."
            instruction = "Answer the following question based on the provided sources:"
        
        style_instructions = {
            'brief': "Provide a concise, direct answer.",
            'comprehensive': "Provide a thorough answer covering key points.",
            'detailed': "Provide a detailed, comprehensive answer with explanations."
        }
        
        prompt = f"""{system_prompt}

{instruction}

Context Sources:
{context_text}

Question: {question}

Instructions:
- {style_instructions.get(style, style_instructions['comprehensive'])}
- Maximum {max_length} words
- Include source references using [1], [2], etc.
- Preserve Islamic terminology
- Maintain respectful tone

Answer:"""
        
        try:
            response = llm(
                prompt,
                max_tokens=self.config['model']['max_tokens'],
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                top_k=self.config['model']['top_k'],
                stop=["\n\nQuestion:", "Context Sources:", "Instructions:"]
            )
            
            answer = response['choices'][0]['text'].strip()
            
            # Clean up the answer
            answer = re.sub(r'^Answer:?\s*', '', answer, flags=re.IGNORECASE)
            answer = answer.strip()
            
            return answer if answer else self._generate_fallback_answer(question, context_chunks, max_length)
            
        except Exception as e:
            print(f"Error in LLM answer generation: {e}")
            return self._generate_fallback_answer(question, context_chunks, max_length)
    
    def _generate_fallback_answer(self, question: str, context_chunks: List[Dict[str, Any]], max_length: int) -> str:
        """Generate a fallback answer without LLM."""
        if not context_chunks:
            return "I don't have enough information to answer this question."
        
        # Simple extractive approach
        relevant_sentences = []
        
        for chunk in context_chunks:
            chunk_text = chunk.get('text', '')
            sentences = re.split(r'[.!?]+', chunk_text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Minimum sentence length
                    # Check if sentence is relevant to question
                    question_words = set(question.lower().split())
                    sentence_words = set(sentence.lower().split())
                    overlap = len(question_words.intersection(sentence_words))
                    
                    if overlap > 0:
                        relevant_sentences.append((overlap, sentence))
        
        # Sort by relevance and combine
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        
        answer_parts = []
        current_length = 0
        
        for _, sentence in relevant_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                answer_parts.append(sentence)
                current_length += sentence_length
            else:
                break
        
        if answer_parts:
            return '. '.join(answer_parts) + '.'
        else:
            return "Based on the available sources, I cannot provide a specific answer to this question."
    
    def _calculate_answer_confidence(self, question: str, answer: str, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the generated answer."""
        if not answer or not chunks:
            return 0.0
        
        # Base score on answer length and completeness
        answer_words = len(answer.split())
        if answer_words < 10:
            base_score = 0.3
        elif answer_words < 50:
            base_score = 0.6
        else:
            base_score = 0.8
        
        # Boost for Islamic terms preservation
        question_islamic_terms = [term for term in self.islamic_terms if term.lower() in question.lower()]
        answer_islamic_terms = [term for term in self.islamic_terms if term.lower() in answer.lower()]
        
        if question_islamic_terms:
            term_preservation = len(answer_islamic_terms) / len(question_islamic_terms)
            base_score += term_preservation * 0.2
        
        # Boost for source citations
        citation_count = len(re.findall(r'\[\d+\]', answer))
        if citation_count > 0:
            base_score += min(0.1, citation_count * 0.05)
        
        # Penalty for generic responses
        generic_phrases = ['i don\'t know', 'cannot answer', 'not enough information']
        for phrase in generic_phrases:
            if phrase in answer.lower():
                base_score *= 0.5
                break
        
        return min(1.0, base_score)
    
    def _find_islamic_terms_in_answer(self, answer: str) -> List[str]:
        """Find Islamic terms used in the answer."""
        found_terms = []
        answer_lower = answer.lower()
        
        for term in self.islamic_terms:
            if term.lower() in answer_lower:
                found_terms.append(term)
        
        return list(set(found_terms))
    
    def _run(self, **kwargs) -> str:
        """Execute the answer generation tool."""
        start_time = self._get_current_time()
        
        try:
            # Validate input using Pydantic
            input_data = AnswerGenerationInput(**kwargs)
            
            # Detect language if set to auto
            detected_language = input_data.language
            if detected_language == 'auto':
                detected_language = self._detect_language(input_data.question)
            
            # Select relevant chunks
            relevant_chunks = self._select_relevant_chunks(input_data.question, input_data.context_chunks)
            
            if not relevant_chunks:
                # No relevant context found
                result = AnswerGenerationOutput(
                    success=False,
                    answer="I don't have relevant information to answer this question.",
                    confidence_score=0.0,
                    citations=[],
                    sources_used=0,
                    answer_length=0,
                    islamic_terms_used=[],
                    language_detected=detected_language,
                    processing_time=self._get_current_time() - start_time,
                    metadata={},
                    message="No relevant context chunks found"
                )
                return result
            
            # Generate answer
            answer = self._generate_answer_with_llm(
                input_data.question,
                relevant_chunks,
                input_data.max_answer_length,
                input_data.answer_style,
                detected_language
            )
            
            # Create citations
            citations = []
            if input_data.include_citations:
                citations = self._create_citations(relevant_chunks, input_data.citation_style)
                answer = self._format_citations_in_text(answer, citations, input_data.citation_style)
            
            # Calculate confidence
            confidence_score = self._calculate_answer_confidence(input_data.question, answer, relevant_chunks)
            
            # Check confidence threshold
            if confidence_score < input_data.confidence_threshold:
                result = AnswerGenerationOutput(
                    success=False,
                    answer=answer,
                    confidence_score=confidence_score,
                    citations=citations,
                    sources_used=len(relevant_chunks),
                    answer_length=len(answer.split()),
                    islamic_terms_used=self._find_islamic_terms_in_answer(answer),
                    language_detected=detected_language,
                    processing_time=self._get_current_time() - start_time,
                    metadata={
                        'confidence_threshold': input_data.confidence_threshold,
                        'llm_available': LLAMA_CPP_AVAILABLE and self._llm_model is not None
                    },
                    message=f"Answer confidence ({confidence_score:.2f}) below threshold ({input_data.confidence_threshold})"
                )
                return result
            
            # Successful answer generation
            result = AnswerGenerationOutput(
                success=True,
                answer=answer,
                confidence_score=confidence_score,
                citations=citations,
                sources_used=len(relevant_chunks),
                answer_length=len(answer.split()),
                islamic_terms_used=self._find_islamic_terms_in_answer(answer),
                language_detected=detected_language,
                processing_time=self._get_current_time() - start_time,
                metadata={
                    'question_length': len(input_data.question.split()),
                    'total_context_chunks': len(input_data.context_chunks),
                    'relevant_chunks_used': len(relevant_chunks),
                    'llm_available': LLAMA_CPP_AVAILABLE and self._llm_model is not None,
                    'timestamp': datetime.now().isoformat()
                },
                message="Answer generated successfully"
            )
            
            return result
            
        except Exception as e:
            error_result = AnswerGenerationOutput(
                success=False,
                answer="",
                confidence_score=0.0,
                citations=[],
                sources_used=0,
                answer_length=0,
                islamic_terms_used=[],
                language_detected="en",
                processing_time=self._get_current_time() - start_time,
                metadata={},
                message=f"Error in answer generation: {str(e)}"
            )
            
            return error_result


if __name__ == "__main__":
    # Example usage
    tool = AnswerGeneratorTool()
    
    sample_chunks = [
        {
            "text": "The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him). It contains 114 chapters called Surahs.",
            "metadata": {
                "source": "Islamic_Basics_Lecture_1.mp3",
                "timestamp": "00:05:30",
                "is_quran": True,
                "is_hadith": False
            }
        },
        {
            "text": "Prayer (Salah) is one of the Five Pillars of Islam. Muslims are required to pray five times a day facing the Kaaba in Mecca.",
            "metadata": {
                "source": "Five_Pillars_Explanation.mp3",
                "timestamp": "00:12:15",
                "is_quran": False,
                "is_hadith": False
            }
        }
    ]
    
    result = tool._run(
        question="What is the Quran and how many chapters does it have?",
        context_chunks=sample_chunks,
        max_answer_length=150,
        include_citations=True,
        citation_style="numbered",
        confidence_threshold=0.3
    )
    
    print("Answer Generation Tool Demo:")
    print("=" * 50)
    print(result)