#!/usr/bin/env python3
"""
Test Suite for Embedding Tool

Comprehensive tests for the EmbedderTool including:
- Tool initialization and configuration
- Input validation with Pydantic schemas
- Embedding generation with different models
- Batch processing capabilities
- Output format variations
- Islamic content metadata preservation
- Error handling and edge cases
- Performance and dimension consistency

Author: Assistant
Date: 2024
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add the tools directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder import EmbedderTool, EmbeddingInput, EmbeddingOutput


def create_test_config() -> str:
    """Create a temporary test configuration file."""
    config_content = """
text_processing:
  embedding:
    model_name: "all-MiniLM-L6-v2"
    device: "cpu"
    batch_size: 16
    normalize_embeddings: true
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        return f.name


def test_tool_initialization():
    """Test 1: Tool initialization and configuration loading."""
    print("Test 1: Tool initialization and configuration loading")
    
    try:
        # Test with custom config
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        assert tool.name == "Text Embedder"
        assert "embedding" in tool.description.lower()
        assert tool.args_schema == EmbeddingInput
        
        # Check configuration loading
        config = tool.config
        assert config['model_name'] == 'all-MiniLM-L6-v2'
        assert config['device'] == 'cpu'
        assert config['batch_size'] == 16
        assert config['normalize_embeddings'] is True
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Tool initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Tool initialization test failed: {e}")
        return False


def test_input_validation():
    """Test 2: Input schema validation with Pydantic."""
    print("\nTest 2: Input schema validation with Pydantic")
    
    try:
        # Valid input
        valid_input = EmbeddingInput(
            text_chunks=["Test text 1", "Test text 2"],
            model_name="all-MiniLM-L6-v2",
            batch_size=8,
            output_format="list"
        )
        assert len(valid_input.text_chunks) == 2
        assert valid_input.model_name == "all-MiniLM-L6-v2"
        assert valid_input.batch_size == 8
        assert valid_input.output_format == "list"
        
        # Test invalid output format
        try:
            invalid_input = EmbeddingInput(
                text_chunks=["Test"],
                output_format="invalid_format"
            )
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected
        
        # Test empty chunks
        try:
            empty_input = EmbeddingInput(text_chunks=[])
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected
        
        print("✅ Input validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False


def test_basic_embedding_generation():
    """Test 3: Basic embedding generation with string inputs."""
    print("\nTest 3: Basic embedding generation with string inputs")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Test with simple string chunks
        text_chunks = [
            "This is a test sentence.",
            "Another test sentence for embedding.",
            "Third sentence to test batch processing."
        ]
        
        result = tool._run(
            text_chunks=text_chunks,
            output_format="list",
            include_metadata=True
        )
        
        # Validate results
        assert isinstance(result, list)
        assert len(result) == 3
        
        for i, embedding_data in enumerate(result):
            assert 'embedding_id' in embedding_data
            assert 'text' in embedding_data
            assert 'embedding' in embedding_data
            assert 'dimension' in embedding_data
            assert 'model_name' in embedding_data
            assert 'chunk_index' in embedding_data
            
            assert embedding_data['text'] == text_chunks[i]
            assert embedding_data['chunk_index'] == i
            assert embedding_data['model_name'] == 'all-MiniLM-L6-v2'
            assert isinstance(embedding_data['embedding'], list)
            assert len(embedding_data['embedding']) == embedding_data['dimension']
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Basic embedding generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic embedding generation test failed: {e}")
        return False


def test_chunk_dict_processing():
    """Test 4: Processing chunk dictionaries with metadata."""
    print("\nTest 4: Processing chunk dictionaries with metadata")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Test with chunk dictionaries (from chunker tool)
        chunk_dicts = [
            {
                'id': 'chunk_001',
                'text': 'بسم الله الرحمن الرحيم - In the name of Allah.',
                'start_char': 0,
                'end_char': 45,
                'word_count': 9,
                'char_count': 45,
                'contains_arabic': True,
                'contains_quran': False,
                'topic_keywords': ['Allah', 'Bismillah']
            },
            {
                'id': 'chunk_002',
                'text': 'The Quran is the holy book of Islam.',
                'start_char': 46,
                'end_char': 82,
                'word_count': 8,
                'char_count': 36,
                'contains_arabic': False,
                'contains_quran': True,
                'topic_keywords': ['Quran', 'Islam']
            }
        ]
        
        result = tool._run(
            text_chunks=chunk_dicts,
            output_format="list",
            include_metadata=True
        )
        
        # Validate metadata preservation
        assert len(result) == 2
        
        first_result = result[0]
        assert first_result['chunk_id'] == 'chunk_001'
        assert first_result['start_char'] == 0
        assert first_result['end_char'] == 45
        assert first_result['word_count'] == 9
        assert first_result['char_count'] == 45
        assert first_result['contains_arabic'] is True
        assert first_result['contains_quran'] is False
        assert first_result['topic_keywords'] == ['Allah', 'Bismillah']
        
        second_result = result[1]
        assert second_result['chunk_id'] == 'chunk_002'
        assert second_result['contains_arabic'] is False
        assert second_result['contains_quran'] is True
        assert second_result['topic_keywords'] == ['Quran', 'Islam']
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Chunk dictionary processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Chunk dictionary processing test failed: {e}")
        return False


def test_output_formats():
    """Test 5: Different output formats (list, numpy, dict)."""
    print("\nTest 5: Different output formats")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        text_chunks = ["Test sentence one.", "Test sentence two."]
        
        # Test list format
        list_result = tool._run(
            text_chunks=text_chunks,
            output_format="list"
        )
        assert isinstance(list_result, list)
        assert len(list_result) == 2
        assert 'embedding' in list_result[0]
        
        # Test numpy format
        numpy_result = tool._run(
            text_chunks=text_chunks,
            output_format="numpy"
        )
        assert isinstance(numpy_result, np.ndarray)
        assert numpy_result.shape[0] == 2
        assert numpy_result.shape[1] > 0  # Has embedding dimension
        
        # Test dict format
        dict_result = tool._run(
            text_chunks=text_chunks,
            output_format="dict"
        )
        assert isinstance(dict_result, dict)
        assert 'embeddings' in dict_result
        assert 'texts' in dict_result
        assert 'metadata' in dict_result
        assert 'summary' in dict_result
        assert len(dict_result['embeddings']) == 2
        assert len(dict_result['texts']) == 2
        assert len(dict_result['metadata']) == 2
        assert dict_result['summary']['total_chunks'] == 2
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Output formats test passed")
        return True
        
    except Exception as e:
        print(f"❌ Output formats test failed: {e}")
        return False


def test_islamic_content_embedding():
    """Test 6: Embedding Islamic content with Arabic text."""
    print("\nTest 6: Embedding Islamic content with Arabic text")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Islamic content with Arabic
        islamic_texts = [
            "بسم الله الرحمن الرحيم",  # Bismillah in Arabic
            "الحمد لله رب العالمين",  # Alhamdulillah in Arabic
            "The Prophet Muhammad (peace be upon him) said in a Hadith...",
            "Quran 2:255 is known as Ayat al-Kursi.",
            "The five daily prayers are Fajr, Dhuhr, Asr, Maghrib, and Isha."
        ]
        
        result = tool._run(
            text_chunks=islamic_texts,
            output_format="list",
            include_metadata=True
        )
        
        # Validate that Arabic text is processed correctly
        assert len(result) == 5
        
        # Check that embeddings are generated for Arabic text
        arabic_embeddings = [result[0], result[1]]
        for emb_data in arabic_embeddings:
            assert 'embedding' in emb_data
            assert len(emb_data['embedding']) > 0
            assert emb_data['text'] in islamic_texts
        
        # Check that all embeddings have the same dimension
        dimensions = [emb_data['dimension'] for emb_data in result]
        assert all(dim == dimensions[0] for dim in dimensions)
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Islamic content embedding test passed")
        return True
        
    except Exception as e:
        print(f"❌ Islamic content embedding test failed: {e}")
        return False


def test_batch_processing():
    """Test 7: Batch processing with different batch sizes."""
    print("\nTest 7: Batch processing with different batch sizes")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Create a larger set of text chunks
        text_chunks = [f"This is test sentence number {i+1}." for i in range(10)]
        
        # Test with small batch size
        result_small_batch = tool._run(
            text_chunks=text_chunks,
            batch_size=2,
            output_format="numpy"
        )
        
        # Test with large batch size
        result_large_batch = tool._run(
            text_chunks=text_chunks,
            batch_size=8,
            output_format="numpy"
        )
        
        # Results should be the same regardless of batch size
        assert result_small_batch.shape == result_large_batch.shape
        assert result_small_batch.shape[0] == 10
        
        # Embeddings should be very similar (allowing for minor numerical differences)
        similarity = np.allclose(result_small_batch, result_large_batch, rtol=1e-5)
        assert similarity, "Batch processing should produce consistent results"
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Batch processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False


def test_transcript_processing():
    """Test 8: Processing transcript data with timestamps."""
    print("\nTest 8: Processing transcript data with timestamps")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Simulate transcript chunks with timestamps
        transcript_chunks = [
            {
                'text': 'Assalamu alaikum brothers and sisters.',
                'start_time': 0.0,
                'end_time': 2.5,
                'speaker': 'Imam Ahmad'
            },
            {
                'text': 'Today we will discuss the importance of prayer.',
                'start_time': 2.5,
                'end_time': 6.0,
                'speaker': 'Imam Ahmad'
            }
        ]
        
        result = tool._run(
            text_chunks=transcript_chunks,
            output_format="list",
            include_metadata=True
        )
        
        # Validate timestamp preservation
        assert len(result) == 2
        
        first_result = result[0]
        assert first_result['start_time'] == 0.0
        assert first_result['end_time'] == 2.5
        assert first_result['speaker'] == 'Imam Ahmad'
        
        second_result = result[1]
        assert second_result['start_time'] == 2.5
        assert second_result['end_time'] == 6.0
        assert second_result['speaker'] == 'Imam Ahmad'
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Transcript processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Transcript processing test failed: {e}")
        return False


def test_error_handling():
    """Test 9: Error handling for invalid inputs."""
    print("\nTest 9: Error handling for invalid inputs")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Test with empty text
        empty_chunks = [""]
        result = tool._run(
            text_chunks=empty_chunks,
            output_format="list"
        )
        # Should handle empty text gracefully
        assert len(result) == 1
        assert result[0]['text'] == "[Empty text]"
        
        # Test with very long text (should not crash)
        long_text = "This is a very long sentence. " * 1000
        long_chunks = [long_text]
        result = tool._run(
            text_chunks=long_chunks,
            output_format="list"
        )
        assert len(result) == 1
        assert len(result[0]['embedding']) > 0
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_configuration_override():
    """Test 10: Configuration parameter overrides."""
    print("\nTest 10: Configuration parameter overrides")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        text_chunks = ["Test sentence for configuration override."]
        
        # Test with default config
        result_default = tool._run(
            text_chunks=text_chunks,
            output_format="list"
        )
        
        # Test with overridden parameters
        result_override = tool._run(
            text_chunks=text_chunks,
            batch_size=1,  # Override batch size
            normalize_embeddings=False,  # Override normalization
            output_format="list"
        )
        
        # Both should work but may produce different results due to normalization
        assert len(result_default) == 1
        assert len(result_override) == 1
        assert len(result_default[0]['embedding']) == len(result_override[0]['embedding'])
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Configuration override test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration override test failed: {e}")
        return False


def test_dimension_consistency():
    """Test 11: Embedding dimension consistency."""
    print("\nTest 11: Embedding dimension consistency")
    
    try:
        config_path = create_test_config()
        tool = EmbedderTool(config_path=config_path)
        
        # Test with various text lengths
        text_chunks = [
            "Short.",
            "This is a medium length sentence for testing.",
            "This is a much longer sentence that contains more words and should test whether the embedding dimension remains consistent regardless of the input text length and complexity."
        ]
        
        result = tool._run(
            text_chunks=text_chunks,
            output_format="list"
        )
        
        # All embeddings should have the same dimension
        dimensions = [emb_data['dimension'] for emb_data in result]
        assert all(dim == dimensions[0] for dim in dimensions)
        
        # Dimension should match the model's expected dimension (384 for all-MiniLM-L6-v2)
        expected_dimension = 384
        assert all(dim == expected_dimension for dim in dimensions)
        
        # Clean up
        os.unlink(config_path)
        
        print("✅ Dimension consistency test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dimension consistency test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("EMBEDDING TOOL - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_tool_initialization,
        test_input_validation,
        test_basic_embedding_generation,
        test_chunk_dict_processing,
        test_output_formats,
        test_islamic_content_embedding,
        test_batch_processing,
        test_transcript_processing,
        test_error_handling,
        test_configuration_override,
        test_dimension_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The Embedding Tool is ready for use.")
        print("\nKey features verified:")
        print("✓ CrewAI BaseTool integration")
        print("✓ Sentence-transformers model support")
        print("✓ Batch processing capabilities")
        print("✓ Multiple output formats (list, numpy, dict)")
        print("✓ Islamic content processing (Arabic text)")
        print("✓ Metadata preservation from chunker tool")
        print("✓ Transcript timestamp handling")
        print("✓ Configuration override support")
        print("✓ Error handling and edge cases")
        print("✓ Dimension consistency across inputs")
        print("✓ YAML configuration loading")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)