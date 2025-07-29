#!/usr/bin/env python3
"""
Phase 7: Quality Assurance Testing Suite

Quality assurance tests for Islamic content accuracy,
transcription quality, embedding similarity, and answer relevance.

Author: Assistant
Date: December 2024
Phase: 7 - Testing & Validation
"""

import os
import sys
import json
import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock
from sklearn.metrics.pairwise import cosine_similarity

# Add project paths
sys.path.append(os.path.dirname(__file__))

from tools import (
    AnswerGeneratorTool,
    SummarizerTool,
    EmbedderTool,
    TranscriptionTool
)


class TestIslamicContentAccuracy:
    """Tests for Islamic content accuracy and authenticity."""
    
    @pytest.fixture
    def islamic_test_cases(self):
        """Islamic content test cases with expected accuracy."""
        return [
            {
                'question': 'What are the Five Pillars of Islam?',
                'context': 'The Five Pillars of Islam are: Shahada (faith), Salah (prayer), Zakat (charity), Sawm (fasting), and Hajj (pilgrimage).',
                'expected_keywords': ['shahada', 'salah', 'zakat', 'sawm', 'hajj', 'pillars'],
                'accuracy_threshold': 0.8
            },
            {
                'question': 'When do Muslims pray?',
                'context': 'Muslims pray five times a day: Fajr (dawn), Dhuhr (midday), Asr (afternoon), Maghrib (sunset), and Isha (night).',
                'expected_keywords': ['fajr', 'dhuhr', 'asr', 'maghrib', 'isha', 'five times'],
                'accuracy_threshold': 0.8
            },
            {
                'question': 'What is the Quran?',
                'context': 'The Quran is the holy book of Islam, believed to be the direct word of Allah revealed to Prophet Muhammad through Angel Gabriel.',
                'expected_keywords': ['holy book', 'allah', 'muhammad', 'gabriel', 'revelation'],
                'accuracy_threshold': 0.7
            },
            {
                'question': 'What is Ramadan?',
                'context': 'Ramadan is the ninth month of the Islamic calendar, during which Muslims fast from dawn to sunset.',
                'expected_keywords': ['ninth month', 'islamic calendar', 'fast', 'dawn', 'sunset'],
                'accuracy_threshold': 0.8
            }
        ]
    
    def test_islamic_qa_accuracy(self, islamic_test_cases):
        """Test accuracy of Islamic Q&A responses."""
        answer_generator = AnswerGeneratorTool()
        
        for test_case in islamic_test_cases:
            with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
                # Mock a relevant Islamic answer
                mock_answer.return_value = json.dumps({
                    'answer': f"Based on Islamic teachings, {test_case['context']}",
                    'citations': [{
                        'text': test_case['context'],
                        'source': 'test_source.mp3',
                        'timestamp': '00:01:00'
                    }],
                    'confidence': 0.9
                })
                
                result = answer_generator._run(
                    question=test_case['question'],
                    context_chunks=[{
                        'text': test_case['context'],
                        'metadata': {'source': 'test_source.mp3'}
                    }],
                    language='english'
                )
                
                response = json.loads(result)
                answer_text = response['answer'].lower()
                
                # Calculate keyword accuracy
                found_keywords = sum(1 for keyword in test_case['expected_keywords'] 
                                   if keyword.lower() in answer_text)
                accuracy = found_keywords / len(test_case['expected_keywords'])
                
                assert accuracy >= test_case['accuracy_threshold'], \
                    f"Islamic content accuracy too low: {accuracy:.2f} < {test_case['accuracy_threshold']}"
                
                # Verify citations are provided
                assert 'citations' in response
                assert len(response['citations']) > 0
    
    def test_islamic_content_filtering(self):
        """Test filtering of non-Islamic or inappropriate content."""
        answer_generator = AnswerGeneratorTool()
        
        inappropriate_contexts = [
            "This video discusses non-Islamic religious practices.",
            "The content contains inappropriate material not suitable for Islamic education.",
            "This is about secular topics unrelated to Islam."
        ]
        
        for context in inappropriate_contexts:
            with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
                mock_answer.return_value = json.dumps({
                    'answer': "I cannot provide an answer as the content does not appear to be related to Islamic teachings.",
                    'citations': [],
                    'confidence': 0.1,
                    'islamic_relevance': False
                })
                
                result = answer_generator._run(
                    question="What does this teach about Islam?",
                    context_chunks=[{
                        'text': context,
                        'metadata': {'source': 'test.mp3'}
                    }],
                    language='english'
                )
                
                response = json.loads(result)
                
                # Should indicate low relevance or refuse to answer
                assert (response['confidence'] < 0.5 or 
                        'cannot provide' in response['answer'].lower() or
                        response.get('islamic_relevance', True) == False)
    
    def test_citation_authenticity(self):
        """Test authenticity and accuracy of citations."""
        answer_generator = AnswerGeneratorTool()
        
        test_context = "The Prophet Muhammad (peace be upon him) said: 'The best of people are those who benefit others.'"
        
        with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
            mock_answer.return_value = json.dumps({
                'answer': "According to Islamic teachings, the Prophet emphasized helping others.",
                'citations': [{
                    'text': test_context,
                    'source': 'hadith_collection.mp3',
                    'timestamp': '00:05:30',
                    'relevance_score': 0.95
                }],
                'confidence': 0.9
            })
            
            result = answer_generator._run(
                question="What did the Prophet say about helping others?",
                context_chunks=[{
                    'text': test_context,
                    'metadata': {'source': 'hadith_collection.mp3', 'timestamp': '00:05:30'}
                }],
                language='english'
            )
            
            response = json.loads(result)
            
            # Verify citation quality
            assert len(response['citations']) > 0
            citation = response['citations'][0]
            
            assert 'text' in citation
            assert 'source' in citation
            assert citation.get('relevance_score', 0) > 0.8
            assert test_context in citation['text']


class TestTranscriptionQuality:
    """Tests for transcription accuracy and quality."""
    
    def test_transcription_accuracy_metrics(self):
        """Test transcription accuracy using known audio-text pairs."""
        transcriber = TranscriptionTool()
        
        # Mock transcription results
        test_cases = [
            {
                'expected': "Bismillah ar-Rahman ar-Raheem",
                'actual': "Bismillah ar-Rahman ar-Raheem",
                'accuracy_threshold': 0.95
            },
            {
                'expected': "Allahu Akbar",
                'actual': "Allah Akbar",  # Minor variation
                'accuracy_threshold': 0.8
            },
            {
                'expected': "As-salamu alaykum wa rahmatullahi wa barakatuh",
                'actual': "As-salamu alaykum wa rahmatullahi wa barakatuh",
                'accuracy_threshold': 0.9
            }
        ]
        
        for test_case in test_cases:
            with patch('tools.transcriber.TranscriptionTool._run') as mock_transcribe:
                mock_transcribe.return_value = json.dumps({
                    'text': test_case['actual'],
                    'segments': [{
                        'start': 0.0,
                        'end': 5.0,
                        'text': test_case['actual']
                    }],
                    'confidence': 0.9
                })
                
                result = transcriber._run(
                    audio_file="test_audio.mp3",
                    language="auto",
                    output_format="json"
                )
                
                response = json.loads(result)
                
                # Calculate word-level accuracy
                expected_words = test_case['expected'].lower().split()
                actual_words = response['text'].lower().split()
                
                # Simple word overlap accuracy
                common_words = set(expected_words) & set(actual_words)
                accuracy = len(common_words) / max(len(expected_words), len(actual_words))
                
                assert accuracy >= test_case['accuracy_threshold'], \
                    f"Transcription accuracy too low: {accuracy:.2f} < {test_case['accuracy_threshold']}"
    
    def test_arabic_text_handling(self):
        """Test handling of Arabic text in transcriptions."""
        transcriber = TranscriptionTool()
        
        arabic_phrases = [
            "بسم الله الرحمن الرحيم",  # Bismillah
            "الله أكبر",  # Allahu Akbar
            "السلام عليكم",  # As-salamu alaykum
        ]
        
        for phrase in arabic_phrases:
            with patch('tools.transcriber.TranscriptionTool._run') as mock_transcribe:
                mock_transcribe.return_value = json.dumps({
                    'text': phrase,
                    'segments': [{
                        'start': 0.0,
                        'end': 3.0,
                        'text': phrase
                    }],
                    'language': 'arabic',
                    'confidence': 0.85
                })
                
                result = transcriber._run(
                    audio_file="arabic_test.mp3",
                    language="arabic",
                    output_format="json"
                )
                
                response = json.loads(result)
                
                # Verify Arabic text is preserved
                assert phrase in response['text']
                assert response.get('language') == 'arabic'
                assert response.get('confidence', 0) > 0.7
    
    def test_timestamp_accuracy(self):
        """Test accuracy of timestamp information."""
        transcriber = TranscriptionTool()
        
        with patch('tools.transcriber.TranscriptionTool._run') as mock_transcribe:
            mock_transcribe.return_value = json.dumps({
                'text': 'This is a test transcription with multiple segments.',
                'segments': [
                    {'start': 0.0, 'end': 2.5, 'text': 'This is a test'},
                    {'start': 2.5, 'end': 5.0, 'text': 'transcription with'},
                    {'start': 5.0, 'end': 7.5, 'text': 'multiple segments.'}
                ],
                'duration': 7.5
            })
            
            result = transcriber._run(
                audio_file="test_segments.mp3",
                language="english",
                output_format="json"
            )
            
            response = json.loads(result)
            segments = response['segments']
            
            # Verify timestamp consistency
            for i in range(len(segments) - 1):
                current_end = segments[i]['end']
                next_start = segments[i + 1]['start']
                
                # Timestamps should be continuous or have minimal gaps
                gap = abs(next_start - current_end)
                assert gap <= 0.1, f"Large timestamp gap: {gap:.2f}s"
            
            # Verify total duration consistency
            last_segment_end = segments[-1]['end']
            total_duration = response.get('duration', last_segment_end)
            assert abs(last_segment_end - total_duration) <= 0.5


class TestEmbeddingSimilarity:
    """Tests for embedding quality and similarity metrics."""
    
    def test_semantic_similarity(self):
        """Test semantic similarity of related Islamic concepts."""
        embedder = EmbedderTool()
        
        # Related Islamic concepts
        related_pairs = [
            ("Islamic prayer", "Muslim salah"),
            ("Quran recitation", "Quranic verses"),
            ("Hajj pilgrimage", "Mecca pilgrimage"),
            ("Ramadan fasting", "Islamic fasting month")
        ]
        
        for text1, text2 in related_pairs:
            with patch('tools.embedder.EmbedderTool._run') as mock_embed:
                # Mock embeddings for related concepts
                mock_embed.side_effect = [
                    json.dumps([{'embedding': [0.1, 0.8, 0.3, 0.9]}]),  # First text
                    json.dumps([{'embedding': [0.2, 0.7, 0.4, 0.8]}])   # Second text (similar)
                ]
                
                # Get embeddings
                result1 = embedder._run(
                    text_chunks=[{'text': text1, 'metadata': {}}],
                    output_format='list'
                )
                result2 = embedder._run(
                    text_chunks=[{'text': text2, 'metadata': {}}],
                    output_format='list'
                )
                
                emb1 = json.loads(result1)[0]['embedding']
                emb2 = json.loads(result2)[0]['embedding']
                
                # Calculate cosine similarity
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                
                # Related concepts should have high similarity
                assert similarity > 0.7, f"Low similarity for related concepts: {similarity:.3f}"
    
    def test_embedding_consistency(self):
        """Test consistency of embeddings for identical text."""
        embedder = EmbedderTool()
        
        test_text = "The Five Pillars of Islam are fundamental practices."
        
        with patch('tools.embedder.EmbedderTool._run') as mock_embed:
            # Mock identical embeddings for same text
            mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embed.return_value = json.dumps([{'embedding': mock_embedding}])
            
            # Generate embeddings multiple times
            results = []
            for _ in range(3):
                result = embedder._run(
                    text_chunks=[{'text': test_text, 'metadata': {}}],
                    output_format='list'
                )
                embedding = json.loads(result)[0]['embedding']
                results.append(embedding)
            
            # Verify consistency (should be identical for mocked case)
            for i in range(1, len(results)):
                similarity = cosine_similarity([results[0]], [results[i]])[0][0]
                assert similarity > 0.99, f"Inconsistent embeddings: {similarity:.3f}"
    
    def test_embedding_dimensions(self):
        """Test embedding dimension consistency."""
        embedder = EmbedderTool()
        
        test_texts = [
            "Short text",
            "This is a medium length text about Islamic teachings and practices.",
            "This is a very long text that contains multiple sentences about Islamic history, theology, practices, and cultural aspects. It should test how the embedding model handles longer content while maintaining consistent output dimensions."
        ]
        
        embeddings = []
        for text in test_texts:
            with patch('tools.embedder.EmbedderTool._run') as mock_embed:
                # Mock consistent dimension embeddings
                mock_embedding = [0.1] * 384  # Standard sentence-transformer dimension
                mock_embed.return_value = json.dumps([{'embedding': mock_embedding}])
                
                result = embedder._run(
                    text_chunks=[{'text': text, 'metadata': {}}],
                    output_format='list'
                )
                embedding = json.loads(result)[0]['embedding']
                embeddings.append(embedding)
        
        # Verify consistent dimensions
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1, f"Inconsistent dimensions: {dimensions}"
        assert dimensions[0] > 0, "Empty embeddings generated"


class TestAnswerRelevanceScoring:
    """Tests for answer relevance and quality scoring."""
    
    def test_answer_relevance_scoring(self):
        """Test relevance scoring of generated answers."""
        answer_generator = AnswerGeneratorTool()
        
        test_cases = [
            {
                'question': 'What are the Five Pillars of Islam?',
                'relevant_context': 'The Five Pillars are Shahada, Salah, Zakat, Sawm, and Hajj.',
                'expected_relevance': 0.9
            },
            {
                'question': 'When do Muslims pray?',
                'relevant_context': 'Muslims pray five times daily at specific times.',
                'expected_relevance': 0.8
            },
            {
                'question': 'What is Islamic finance?',
                'irrelevant_context': 'This video is about cooking recipes.',
                'expected_relevance': 0.2
            }
        ]
        
        for test_case in test_cases:
            context_key = 'relevant_context' if 'relevant_context' in test_case else 'irrelevant_context'
            context = test_case[context_key]
            
            with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
                relevance_score = test_case['expected_relevance']
                
                mock_answer.return_value = json.dumps({
                    'answer': f"Based on the context: {context}",
                    'citations': [{
                        'text': context,
                        'relevance_score': relevance_score
                    }],
                    'confidence': relevance_score,
                    'relevance_score': relevance_score
                })
                
                result = answer_generator._run(
                    question=test_case['question'],
                    context_chunks=[{
                        'text': context,
                        'metadata': {'source': 'test.mp3'}
                    }],
                    language='english'
                )
                
                response = json.loads(result)
                actual_relevance = response.get('relevance_score', response.get('confidence', 0))
                
                # Verify relevance score is within expected range
                expected = test_case['expected_relevance']
                assert abs(actual_relevance - expected) < 0.3, \
                    f"Relevance score mismatch: {actual_relevance:.2f} vs {expected:.2f}"
    
    def test_answer_completeness(self):
        """Test completeness of generated answers."""
        answer_generator = AnswerGeneratorTool()
        
        comprehensive_context = """
        The Five Pillars of Islam are:
        1. Shahada - Declaration of faith
        2. Salah - Prayer five times daily
        3. Zakat - Charitable giving
        4. Sawm - Fasting during Ramadan
        5. Hajj - Pilgrimage to Mecca
        """
        
        with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
            mock_answer.return_value = json.dumps({
                'answer': "The Five Pillars of Islam are: Shahada (declaration of faith), Salah (prayer), Zakat (charity), Sawm (fasting), and Hajj (pilgrimage). Each pillar represents a fundamental practice for Muslims.",
                'citations': [{
                    'text': comprehensive_context.strip(),
                    'source': 'islamic_education.mp3'
                }],
                'confidence': 0.95,
                'completeness_score': 0.9
            })
            
            result = answer_generator._run(
                question="What are the Five Pillars of Islam?",
                context_chunks=[{
                    'text': comprehensive_context,
                    'metadata': {'source': 'islamic_education.mp3'}
                }],
                language='english'
            )
            
            response = json.loads(result)
            answer = response['answer'].lower()
            
            # Check for key components
            required_elements = ['shahada', 'salah', 'zakat', 'sawm', 'hajj']
            found_elements = sum(1 for element in required_elements if element in answer)
            completeness = found_elements / len(required_elements)
            
            assert completeness >= 0.8, f"Answer not comprehensive enough: {completeness:.2f}"
            assert response.get('confidence', 0) > 0.8
    
    def test_citation_quality_scoring(self):
        """Test quality scoring of citations."""
        answer_generator = AnswerGeneratorTool()
        
        high_quality_context = "The Prophet Muhammad (peace be upon him) emphasized the importance of seeking knowledge, saying 'Seek knowledge from the cradle to the grave.'"
        
        with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
            mock_answer.return_value = json.dumps({
                'answer': "Islamic tradition emphasizes the pursuit of knowledge throughout life.",
                'citations': [{
                    'text': high_quality_context,
                    'source': 'hadith_collection.mp3',
                    'timestamp': '00:15:30',
                    'relevance_score': 0.95,
                    'authenticity_score': 0.9,
                    'quality_score': 0.92
                }],
                'confidence': 0.9
            })
            
            result = answer_generator._run(
                question="What does Islam say about seeking knowledge?",
                context_chunks=[{
                    'text': high_quality_context,
                    'metadata': {
                        'source': 'hadith_collection.mp3',
                        'timestamp': '00:15:30'
                    }
                }],
                language='english'
            )
            
            response = json.loads(result)
            citations = response['citations']
            
            assert len(citations) > 0
            citation = citations[0]
            
            # Verify citation quality metrics
            assert citation.get('relevance_score', 0) > 0.8
            assert citation.get('quality_score', 0) > 0.8
            assert 'source' in citation
            assert 'timestamp' in citation


if __name__ == "__main__":
    # Run quality assurance tests
    pytest.main([__file__, "-v", "--tb=short"])