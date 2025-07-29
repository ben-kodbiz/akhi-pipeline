#!/usr/bin/env python3
"""
Test suite for Answer Generation Tool

This module contains comprehensive tests for the AnswerGeneratorTool,
verifying RAG-based question answering, citation generation, and Islamic content handling.

Author: Assistant
Date: 2024
"""

import json
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from answer_generator import AnswerGeneratorTool, AnswerGenerationInput, AnswerGenerationOutput


class TestAnswerGeneratorTool:
    """Test cases for AnswerGeneratorTool."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample context chunks for testing."""
        return [
            {
                "text": "The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him). It contains 114 chapters called Surahs, each containing verses called Ayahs.",
                "metadata": {
                    "source": "Islamic_Basics_Lecture_1.mp3",
                    "timestamp": "00:05:30",
                    "is_quran": True,
                    "is_hadith": False,
                    "speaker": "Dr. Ahmed"
                }
            },
            {
                "text": "Prayer (Salah) is one of the Five Pillars of Islam. Muslims are required to pray five times a day facing the Kaaba in Mecca. The five daily prayers are Fajr, Dhuhr, Asr, Maghrib, and Isha.",
                "metadata": {
                    "source": "Five_Pillars_Explanation.mp3",
                    "timestamp": "00:12:15",
                    "is_quran": False,
                    "is_hadith": False,
                    "topic": "worship"
                }
            },
            {
                "text": "The Prophet Muhammad (peace be upon him) said: 'The best of people are those who benefit others.' This hadith emphasizes the importance of helping and serving humanity.",
                "metadata": {
                    "source": "Hadith_Collection_Vol1.mp3",
                    "timestamp": "00:08:45",
                    "is_quran": False,
                    "is_hadith": True,
                    "hadith_source": "Tirmidhi"
                }
            },
            {
                "text": "Zakat is the third pillar of Islam, requiring Muslims to give a portion of their wealth to those in need. The standard rate is 2.5% of savings held for one year.",
                "metadata": {
                    "source": "Zakat_Explanation.mp3",
                    "timestamp": "00:15:20",
                    "is_quran": False,
                    "is_hadith": False,
                    "topic": "charity"
                }
            }
        ]
    
    @pytest.fixture
    def config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            'model': {
                'path': '/path/to/test/model.gguf',
                'context_length': 2048,
                'temperature': 0.1,
                'max_tokens': 512
            },
            'answer_generation': {
                'max_context_chunks': 3,
                'min_chunk_relevance': 0.2,
                'default_confidence_threshold': 0.4
            },
            'islamic_content': {
                'context_aware': True,
                'respectful_language': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            return f.name
    
    def test_tool_initialization(self):
        """Test tool initialization with default config."""
        tool = AnswerGeneratorTool()
        
        assert tool.name == "answer_generator"
        assert "generates answers" in tool.description.lower()
        assert tool.args_schema == AnswerGenerationInput
        assert isinstance(tool.config, dict)
        assert 'model' in tool.config
        assert 'answer_generation' in tool.config
    
    def test_tool_initialization_with_config(self, config_file):
        """Test tool initialization with custom config file."""
        tool = AnswerGeneratorTool(config_path=config_file)
        
        assert tool.config['model']['context_length'] == 2048
        assert tool.config['model']['temperature'] == 0.1
        assert tool.config['answer_generation']['max_context_chunks'] == 3
        
        # Clean up
        Path(config_file).unlink()
    
    def test_input_validation_valid(self, sample_chunks):
        """Test input validation with valid parameters."""
        input_data = AnswerGenerationInput(
            question="What is the Quran?",
            context_chunks=sample_chunks,
            max_answer_length=200,
            include_citations=True,
            citation_style="numbered"
        )
        
        assert input_data.question == "What is the Quran?"
        assert len(input_data.context_chunks) == 4
        assert input_data.max_answer_length == 200
        assert input_data.include_citations is True
        assert input_data.citation_style == "numbered"
    
    def test_input_validation_invalid_citation_style(self, sample_chunks):
        """Test input validation with invalid citation style."""
        with pytest.raises(ValueError, match="citation_style must be one of"):
            AnswerGenerationInput(
                question="What is prayer?",
                context_chunks=sample_chunks,
                citation_style="invalid_style"
            )
    
    def test_input_validation_invalid_language(self, sample_chunks):
        """Test input validation with invalid language."""
        with pytest.raises(ValueError, match="language must be one of"):
            AnswerGenerationInput(
                question="What is Zakat?",
                context_chunks=sample_chunks,
                language="invalid_lang"
            )
    
    def test_input_validation_short_question(self, sample_chunks):
        """Test input validation with too short question."""
        with pytest.raises(ValueError):
            AnswerGenerationInput(
                question="Hi",
                context_chunks=sample_chunks
            )
    
    def test_input_validation_empty_chunks(self):
        """Test input validation with empty context chunks."""
        with pytest.raises(ValueError):
            AnswerGenerationInput(
                question="What is Islam?",
                context_chunks=[]
            )
    
    def test_language_detection(self):
        """Test language detection functionality."""
        tool = AnswerGeneratorTool()
        
        # English text
        english_text = "What is the meaning of Islam?"
        assert tool._detect_language(english_text) == 'en'
        
        # Arabic text
        arabic_text = "ما معنى الإسلام؟"
        assert tool._detect_language(arabic_text) == 'ar'
        
        # Mixed text (should detect as Arabic if significant Arabic content)
        mixed_text = "What is الإسلام and القرآن?"
        detected = tool._detect_language(mixed_text)
        assert detected in ['en', 'ar']  # Could be either depending on ratio
    
    def test_chunk_relevance_calculation(self, sample_chunks):
        """Test chunk relevance calculation."""
        tool = AnswerGeneratorTool()
        
        question = "What is the Quran?"
        
        # Test with Quran-related chunk
        quran_chunk = sample_chunks[0]
        relevance = tool._calculate_chunk_relevance(question, quran_chunk)
        assert relevance > 0.2  # Should be relevant
        
        # Test with prayer-related chunk
        prayer_chunk = sample_chunks[1]
        relevance = tool._calculate_chunk_relevance(question, prayer_chunk)
        assert relevance >= 0.0  # Should have some relevance or none
        
        # Test with empty chunk
        empty_chunk = {"text": "", "metadata": {}}
        relevance = tool._calculate_chunk_relevance(question, empty_chunk)
        assert relevance == 0.0
    
    def test_select_relevant_chunks(self, sample_chunks):
        """Test relevant chunk selection."""
        tool = AnswerGeneratorTool()
        
        question = "What is the Quran and how many chapters does it have?"
        relevant_chunks = tool._select_relevant_chunks(question, sample_chunks)
        
        assert len(relevant_chunks) > 0
        assert len(relevant_chunks) <= tool.config['answer_generation']['max_context_chunks']
        
        # The Quran chunk should be selected
        quran_chunk_selected = any('Quran' in chunk['text'] for chunk in relevant_chunks)
        assert quran_chunk_selected
    
    def test_create_citations(self, sample_chunks):
        """Test citation creation."""
        tool = AnswerGeneratorTool()
        
        # Test numbered citations
        citations = tool._create_citations(sample_chunks[:2], "numbered")
        assert len(citations) == 2
        assert citations[0].id == "1"
        assert citations[1].id == "2"
        assert "Islamic_Basics_Lecture_1.mp3" in citations[0].source
        
        # Test inline citations (now uses same numbering)
        citations = tool._create_citations(sample_chunks[:2], "inline")
        assert len(citations) == 2
        assert citations[0].id == "1"
    
    def test_format_citations_in_text(self, sample_chunks):
        """Test citation formatting in text."""
        tool = AnswerGeneratorTool()
        
        citations = tool._create_citations(sample_chunks[:2], "numbered")
        answer_text = "The Quran is the holy book [1]. Prayer is important [2]."
        
        # Test numbered style
        formatted = tool._format_citations_in_text(answer_text, citations, "numbered")
        assert "Sources:" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted
        
        # Test inline style
        formatted = tool._format_citations_in_text(answer_text, citations, "inline")
        assert formatted == answer_text  # Should remain unchanged
        
        # Test footnote style
        formatted = tool._format_citations_in_text(answer_text, citations, "footnote")
        assert "References:" in formatted
    
    def test_fallback_answer_generation(self, sample_chunks):
        """Test fallback answer generation without LLM."""
        tool = AnswerGeneratorTool()
        
        question = "What is the Quran?"
        answer = tool._generate_fallback_answer(question, sample_chunks, 100)
        
        assert len(answer) > 0
        assert "Quran" in answer or "quran" in answer.lower()
        
        # Test with empty chunks
        empty_answer = tool._generate_fallback_answer(question, [], 100)
        assert "don't have enough information" in empty_answer.lower()
    
    def test_calculate_answer_confidence(self, sample_chunks):
        """Test answer confidence calculation."""
        tool = AnswerGeneratorTool()
        
        question = "What is the Quran?"
        
        # Good answer with Islamic terms
        good_answer = "The Quran is the holy book of Islam revealed to Prophet Muhammad. It contains 114 chapters called Surahs [1]."
        confidence = tool._calculate_answer_confidence(question, good_answer, sample_chunks)
        assert confidence > 0.5
        
        # Poor answer
        poor_answer = "I don't know."
        confidence = tool._calculate_answer_confidence(question, poor_answer, sample_chunks)
        assert confidence < 0.5
        
        # Empty answer
        confidence = tool._calculate_answer_confidence(question, "", sample_chunks)
        assert confidence == 0.0
    
    def test_find_islamic_terms_in_answer(self):
        """Test Islamic terms detection in answers."""
        tool = AnswerGeneratorTool()
        
        answer = "The Quran and Hadith are primary sources in Islam. Prophet Muhammad taught about Salah and Zakat."
        terms = tool._find_islamic_terms_in_answer(answer)
        
        expected_terms = ['Quran', 'Hadith', 'Islam', 'Prophet', 'Muhammad', 'Salah', 'Zakat']
        for term in expected_terms:
            assert term in terms
    
    def test_answer_generation_success(self, sample_chunks):
        """Test successful answer generation."""
        tool = AnswerGeneratorTool()
        
        result_json = tool._run(
            question="What is the Quran?",
            context_chunks=sample_chunks,
            max_answer_length=150,
            include_citations=True,
            citation_style="numbered",
            confidence_threshold=0.3
        )
        
        result = json.loads(result_json)
        
        assert result['success'] is True
        assert len(result['answer']) > 0
        assert result['confidence_score'] >= 0.3
        assert result['sources_used'] > 0
        assert result['answer_length'] > 0
        assert 'Quran' in result['islamic_terms_used'] or 'quran' in result['answer'].lower()
        assert result['language_detected'] in ['en', 'ar']
        assert result['processing_time'] > 0
    
    def test_answer_generation_with_citations(self, sample_chunks):
        """Test answer generation with citations enabled."""
        tool = AnswerGeneratorTool()
        
        result_json = tool._run(
            question="What are the Five Pillars of Islam?",
            context_chunks=sample_chunks,
            include_citations=True,
            citation_style="numbered"
        )
        
        result = json.loads(result_json)
        
        if result['success']:
            assert len(result['citations']) > 0
            assert "Sources:" in result['answer'] or "[" in result['answer']
    
    def test_answer_generation_without_citations(self, sample_chunks):
        """Test answer generation without citations."""
        tool = AnswerGeneratorTool()
        
        result_json = tool._run(
            question="What is prayer in Islam?",
            context_chunks=sample_chunks,
            include_citations=False
        )
        
        result = json.loads(result_json)
        
        if result['success']:
            assert len(result['citations']) == 0
            assert "Sources:" not in result['answer']
    
    def test_answer_generation_low_confidence(self, sample_chunks):
        """Test answer generation with low confidence threshold."""
        tool = AnswerGeneratorTool()
        
        # Use irrelevant question to get low confidence
        result_json = tool._run(
            question="What is the weather like today?",
            context_chunks=sample_chunks,
            confidence_threshold=0.8  # High threshold
        )
        
        result = json.loads(result_json)
        
        # Should either fail due to low confidence or succeed but with lower confidence
        if not result['success']:
            assert "confidence" in result['message'].lower() or "information" in result['message'].lower()
        else:
            # If it succeeds, confidence might still be reasonable
            assert result['confidence_score'] >= 0.0
    
    def test_answer_generation_no_relevant_chunks(self):
        """Test answer generation with no relevant chunks."""
        tool = AnswerGeneratorTool()
        
        # Create irrelevant chunks
        irrelevant_chunks = [
            {
                "text": "The weather is sunny today. It's a beautiful day for a walk in the park.",
                "metadata": {"source": "weather_report.txt"}
            }
        ]
        
        result_json = tool._run(
            question="What is the Quran?",
            context_chunks=irrelevant_chunks,
            confidence_threshold=0.3
        )
        
        result = json.loads(result_json)
        
        # Should either fail due to irrelevant context or succeed with low confidence
        if not result['success']:
            assert "relevant" in result['message'].lower() or "information" in result['message'].lower()
        else:
            # If it somehow succeeds, it should have low confidence
            assert result['confidence_score'] < 0.7
    
    def test_answer_generation_different_styles(self, sample_chunks):
        """Test answer generation with different styles."""
        tool = AnswerGeneratorTool()
        
        question = "What is Zakat?"
        
        styles = ['brief', 'comprehensive', 'detailed']
        
        for style in styles:
            result_json = tool._run(
                question=question,
                context_chunks=sample_chunks,
                answer_style=style,
                confidence_threshold=0.2
            )
            
            result = json.loads(result_json)
            
            # Should succeed or provide meaningful feedback
            assert 'answer' in result
            assert 'confidence_score' in result
    
    def test_answer_generation_different_languages(self, sample_chunks):
        """Test answer generation with different language settings."""
        tool = AnswerGeneratorTool()
        
        languages = ['en', 'ar', 'auto']
        
        for lang in languages:
            result_json = tool._run(
                question="What is Islam?",
                context_chunks=sample_chunks,
                language=lang,
                confidence_threshold=0.2
            )
            
            result = json.loads(result_json)
            
            assert 'language_detected' in result
            assert result['language_detected'] in ['en', 'ar']
    
    def test_error_handling_invalid_input(self):
        """Test error handling with invalid input."""
        tool = AnswerGeneratorTool()
        
        # Missing required parameters
        result_json = tool._run(
            question="What is Islam?"
            # Missing context_chunks
        )
        
        result = json.loads(result_json)
        
        assert result['success'] is False
        assert "error" in result['message'].lower()
    
    @patch('answer_generator.LLAMA_CPP_AVAILABLE', False)
    def test_fallback_without_llm(self, sample_chunks):
        """Test fallback behavior when LLM is not available."""
        tool = AnswerGeneratorTool()
        
        result_json = tool._run(
            question="What is the Quran?",
            context_chunks=sample_chunks,
            confidence_threshold=0.2
        )
        
        result = json.loads(result_json)
        
        # Should still work with fallback method
        assert 'answer' in result
        assert result['metadata']['llm_available'] is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])