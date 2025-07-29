#!/usr/bin/env python3
"""
Embedding Tool Demo Script

Demonstrates the capabilities of the EmbedderTool including:
- Basic text embedding generation
- Islamic content processing with Arabic text
- Chunk dictionary processing from chunker tool
- Multiple output formats (list, numpy, dict)
- Batch processing with different sizes
- Transcript data with timestamps
- Configuration overrides
- Error handling scenarios

Author: Assistant
Date: 2024
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add the tools directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder import EmbedderTool


def create_sample_islamic_text() -> List[str]:
    """Create sample Islamic text for demonstration."""
    return [
        "بسم الله الرحمن الرحيم",  # Bismillah in Arabic
        "الحمد لله رب العالمين",  # Alhamdulillah in Arabic
        "In the name of Allah, the Most Gracious, the Most Merciful.",
        "The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him).",
        "The five daily prayers (Salah) are fundamental pillars of Islamic practice.",
        "Zakat is the third pillar of Islam, involving charity to those in need.",
        "The Hajj pilgrimage to Mecca is a once-in-a-lifetime obligation for Muslims.",
        "Prophet Muhammad (PBUH) said: 'The best of people are those who benefit others.'",
        "Surah Al-Fatiha is the opening chapter of the Quran and is recited in every prayer.",
        "The concept of Tawhid (monotheism) is central to Islamic belief."
    ]


def create_sample_chunk_dicts() -> List[Dict[str, Any]]:
    """Create sample chunk dictionaries (as would come from chunker tool)."""
    return [
        {
            'id': 'chunk_001',
            'text': 'بسم الله الرحمن الرحيم - In the name of Allah, the Most Gracious, the Most Merciful.',
            'start_char': 0,
            'end_char': 85,
            'word_count': 15,
            'char_count': 85,
            'contains_arabic': True,
            'contains_quran': True,
            'contains_hadith': False,
            'topic_keywords': ['Allah', 'Bismillah', 'Mercy']
        },
        {
            'id': 'chunk_002',
            'text': 'The Quran was revealed to Prophet Muhammad (peace be upon him) over 23 years.',
            'start_char': 86,
            'end_char': 162,
            'word_count': 13,
            'char_count': 76,
            'contains_arabic': False,
            'contains_quran': True,
            'contains_hadith': False,
            'topic_keywords': ['Quran', 'Prophet Muhammad', 'revelation']
        },
        {
            'id': 'chunk_003',
            'text': 'The Prophet (PBUH) said: "Seek knowledge from the cradle to the grave."',
            'start_char': 163,
            'end_char': 236,
            'word_count': 12,
            'char_count': 73,
            'contains_arabic': False,
            'contains_quran': False,
            'contains_hadith': True,
            'topic_keywords': ['Prophet', 'knowledge', 'education']
        }
    ]


def create_sample_transcript() -> List[Dict[str, Any]]:
    """Create sample transcript data with timestamps."""
    return [
        {
            'text': 'Assalamu alaikum brothers and sisters, welcome to today\'s Islamic lecture.',
            'start_time': 0.0,
            'end_time': 4.5,
            'speaker': 'Imam Ahmad',
            'confidence': 0.95
        },
        {
            'text': 'Today we will discuss the importance of seeking knowledge in Islam.',
            'start_time': 4.5,
            'end_time': 8.2,
            'speaker': 'Imam Ahmad',
            'confidence': 0.92
        },
        {
            'text': 'The Prophet Muhammad, peace be upon him, emphasized education for all Muslims.',
            'start_time': 8.2,
            'end_time': 13.1,
            'speaker': 'Imam Ahmad',
            'confidence': 0.94
        },
        {
            'text': 'Knowledge is considered a light that guides us on the straight path.',
            'start_time': 13.1,
            'end_time': 17.3,
            'speaker': 'Imam Ahmad',
            'confidence': 0.91
        }
    ]


def demo_basic_embedding():
    """Demo 1: Basic text embedding generation."""
    print("\n" + "=" * 60)
    print("DEMO 1: BASIC TEXT EMBEDDING GENERATION")
    print("=" * 60)
    
    # Initialize the tool
    embedder = EmbedderTool()
    
    # Simple text samples
    texts = [
        "This is a simple English sentence.",
        "Another example of text for embedding.",
        "Testing the embedding generation process."
    ]
    
    print(f"\nGenerating embeddings for {len(texts)} text samples...")
    
    # Generate embeddings
    result = embedder._run(
        text_chunks=texts,
        output_format="list",
        include_metadata=True
    )
    
    print(f"\n✅ Successfully generated {len(result)} embeddings")
    
    # Display results
    for i, emb_data in enumerate(result):
        print(f"\nText {i+1}: '{emb_data['text'][:50]}...'")
        print(f"  Embedding ID: {emb_data['embedding_id']}")
        print(f"  Dimension: {emb_data['dimension']}")
        print(f"  Model: {emb_data['model_name']}")
        print(f"  Processing time: {emb_data['processing_time']:.3f}s")
        print(f"  Embedding preview: {emb_data['embedding'][:5]}...")


def demo_islamic_content():
    """Demo 2: Islamic content processing with Arabic text."""
    print("\n" + "=" * 60)
    print("DEMO 2: ISLAMIC CONTENT PROCESSING")
    print("=" * 60)
    
    embedder = EmbedderTool()
    islamic_texts = create_sample_islamic_text()
    
    print(f"\nProcessing {len(islamic_texts)} Islamic text samples (including Arabic)...")
    
    result = embedder._run(
        text_chunks=islamic_texts,
        output_format="dict",
        include_metadata=True
    )
    
    print(f"\n✅ Successfully processed Islamic content")
    print(f"Total chunks: {result['summary']['total_chunks']}")
    print(f"Model: {result['summary']['model_name']}")
    print(f"Dimension: {result['summary']['dimension']}")
    print(f"Total processing time: {result['summary']['total_processing_time']:.3f}s")
    
    # Show Arabic text handling
    print("\n📖 Arabic Text Processing:")
    for i, text in enumerate(result['texts'][:2]):  # First two are Arabic
        print(f"  {i+1}. {text}")
        print(f"     Embedding dimension: {len(result['embeddings'][i])}")
    
    # Show English Islamic content
    print("\n📚 English Islamic Content:")
    for i, text in enumerate(result['texts'][2:5], 2):  # Next few are English
        print(f"  {i+1}. {text[:60]}...")


def demo_chunk_processing():
    """Demo 3: Processing chunk dictionaries from chunker tool."""
    print("\n" + "=" * 60)
    print("DEMO 3: CHUNK DICTIONARY PROCESSING")
    print("=" * 60)
    
    embedder = EmbedderTool()
    chunk_dicts = create_sample_chunk_dicts()
    
    print(f"\nProcessing {len(chunk_dicts)} chunk dictionaries with metadata...")
    
    result = embedder._run(
        text_chunks=chunk_dicts,
        output_format="list",
        include_metadata=True
    )
    
    print(f"\n✅ Successfully processed chunk dictionaries")
    
    # Display metadata preservation
    for i, emb_data in enumerate(result):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {emb_data['chunk_id']}")
        print(f"  Text: '{emb_data['text'][:50]}...'")
        print(f"  Character range: {emb_data['start_char']}-{emb_data['end_char']}")
        print(f"  Word count: {emb_data['word_count']}")
        print(f"  Contains Arabic: {emb_data['contains_arabic']}")
        print(f"  Contains Quran: {emb_data['contains_quran']}")
        print(f"  Contains Hadith: {emb_data['contains_hadith']}")
        print(f"  Topic keywords: {emb_data['topic_keywords']}")
        print(f"  Embedding dimension: {emb_data['dimension']}")


def demo_output_formats():
    """Demo 4: Different output formats."""
    print("\n" + "=" * 60)
    print("DEMO 4: MULTIPLE OUTPUT FORMATS")
    print("=" * 60)
    
    embedder = EmbedderTool()
    texts = [
        "Sample text for format demonstration.",
        "Another text to show different outputs."
    ]
    
    print(f"\nGenerating embeddings in different formats...")
    
    # List format
    print("\n📋 List Format:")
    list_result = embedder._run(
        text_chunks=texts,
        output_format="list"
    )
    print(f"  Type: {type(list_result)}")
    print(f"  Length: {len(list_result)}")
    print(f"  First item keys: {list(list_result[0].keys())}")
    
    # NumPy format
    print("\n🔢 NumPy Format:")
    numpy_result = embedder._run(
        text_chunks=texts,
        output_format="numpy"
    )
    print(f"  Type: {type(numpy_result)}")
    print(f"  Shape: {numpy_result.shape}")
    print(f"  Data type: {numpy_result.dtype}")
    print(f"  First embedding preview: {numpy_result[0][:5]}...")
    
    # Dictionary format
    print("\n📚 Dictionary Format:")
    dict_result = embedder._run(
        text_chunks=texts,
        output_format="dict"
    )
    print(f"  Type: {type(dict_result)}")
    print(f"  Keys: {list(dict_result.keys())}")
    print(f"  Embeddings shape: {np.array(dict_result['embeddings']).shape}")
    print(f"  Summary: {dict_result['summary']}")


def demo_batch_processing():
    """Demo 5: Batch processing with different sizes."""
    print("\n" + "=" * 60)
    print("DEMO 5: BATCH PROCESSING")
    print("=" * 60)
    
    embedder = EmbedderTool()
    
    # Create a larger set of texts
    texts = [f"This is test sentence number {i+1} for batch processing demonstration." for i in range(12)]
    
    print(f"\nProcessing {len(texts)} texts with different batch sizes...")
    
    # Small batch size
    print("\n🔄 Small Batch Size (batch_size=3):")
    start_time = embedder._get_current_time()
    small_batch_result = embedder._run(
        text_chunks=texts,
        batch_size=3,
        output_format="numpy"
    )
    small_batch_time = embedder._get_current_time() - start_time
    print(f"  Result shape: {small_batch_result.shape}")
    print(f"  Processing time: {small_batch_time:.3f}s")
    
    # Large batch size
    print("\n🚀 Large Batch Size (batch_size=8):")
    start_time = embedder._get_current_time()
    large_batch_result = embedder._run(
        text_chunks=texts,
        batch_size=8,
        output_format="numpy"
    )
    large_batch_time = embedder._get_current_time() - start_time
    print(f"  Result shape: {large_batch_result.shape}")
    print(f"  Processing time: {large_batch_time:.3f}s")
    
    # Compare results
    similarity = np.allclose(small_batch_result, large_batch_result, rtol=1e-5)
    print(f"\n🔍 Results consistency: {'✅ Identical' if similarity else '❌ Different'}")
    print(f"📊 Performance: Large batch is {small_batch_time/large_batch_time:.1f}x faster")


def demo_transcript_processing():
    """Demo 6: Processing transcript data with timestamps."""
    print("\n" + "=" * 60)
    print("DEMO 6: TRANSCRIPT DATA PROCESSING")
    print("=" * 60)
    
    embedder = EmbedderTool()
    transcript_data = create_sample_transcript()
    
    print(f"\nProcessing {len(transcript_data)} transcript segments with timestamps...")
    
    result = embedder._run(
        text_chunks=transcript_data,
        output_format="list",
        include_metadata=True
    )
    
    print(f"\n✅ Successfully processed transcript data")
    
    # Display transcript segments with preserved metadata
    for i, emb_data in enumerate(result):
        print(f"\nSegment {i+1}:")
        print(f"  Time: {emb_data['start_time']:.1f}s - {emb_data['end_time']:.1f}s")
        print(f"  Speaker: {emb_data['speaker']}")
        print(f"  Text: '{emb_data['text']}'")
        print(f"  Embedding ID: {emb_data['embedding_id']}")
        print(f"  Dimension: {emb_data['dimension']}")


def demo_configuration_options():
    """Demo 7: Configuration overrides and options."""
    print("\n" + "=" * 60)
    print("DEMO 7: CONFIGURATION OPTIONS")
    print("=" * 60)
    
    embedder = EmbedderTool()
    text = ["Configuration demonstration text for embedding."]
    
    print("\nTesting different configuration options...")
    
    # Default configuration
    print("\n⚙️ Default Configuration:")
    default_result = embedder._run(
        text_chunks=text,
        output_format="list"
    )
    print(f"  Model: {default_result[0]['model_name']}")
    print(f"  Dimension: {default_result[0]['dimension']}")
    print(f"  Processing time: {default_result[0]['processing_time']:.3f}s")
    
    # Override batch size
    print("\n🔧 Custom Batch Size:")
    custom_batch_result = embedder._run(
        text_chunks=text,
        batch_size=1,
        output_format="list"
    )
    print(f"  Batch size: 1 (overridden)")
    print(f"  Processing time: {custom_batch_result[0]['processing_time']:.3f}s")
    
    # Override normalization
    print("\n📏 Custom Normalization:")
    custom_norm_result = embedder._run(
        text_chunks=text,
        normalize_embeddings=False,
        output_format="list"
    )
    print(f"  Normalization: False (overridden)")
    print(f"  Processing time: {custom_norm_result[0]['processing_time']:.3f}s")
    
    # Compare embedding magnitudes
    default_magnitude = np.linalg.norm(default_result[0]['embedding'])
    custom_magnitude = np.linalg.norm(custom_norm_result[0]['embedding'])
    print(f"\n📊 Embedding Magnitudes:")
    print(f"  Default (normalized): {default_magnitude:.6f}")
    print(f"  Custom (not normalized): {custom_magnitude:.6f}")


def demo_error_handling():
    """Demo 8: Error handling scenarios."""
    print("\n" + "=" * 60)
    print("DEMO 8: ERROR HANDLING")
    print("=" * 60)
    
    embedder = EmbedderTool()
    
    print("\nTesting error handling scenarios...")
    
    # Empty text
    print("\n🔍 Empty Text Handling:")
    empty_result = embedder._run(
        text_chunks=[""],
        output_format="list"
    )
    print(f"  Input: '' (empty string)")
    print(f"  Output text: '{empty_result[0]['text']}'")
    print(f"  Embedding generated: {'✅ Yes' if len(empty_result[0]['embedding']) > 0 else '❌ No'}")
    
    # Very long text
    print("\n📏 Long Text Handling:")
    long_text = "This is a very long sentence that will be repeated many times. " * 100
    long_result = embedder._run(
        text_chunks=[long_text],
        output_format="list"
    )
    print(f"  Input length: {len(long_text)} characters")
    print(f"  Processing successful: {'✅ Yes' if len(long_result) > 0 else '❌ No'}")
    print(f"  Embedding dimension: {long_result[0]['dimension']}")
    
    # Mixed content types
    print("\n🔀 Mixed Content Types:")
    mixed_content = [
        "Regular English text.",
        "بسم الله الرحمن الرحيم",  # Arabic
        "Text with numbers: 123, 456, 789.",
        "Special characters: @#$%^&*()!",
        ""  # Empty string
    ]
    mixed_result = embedder._run(
        text_chunks=mixed_content,
        output_format="list"
    )
    print(f"  Processed {len(mixed_result)} diverse content types")
    print(f"  All embeddings have same dimension: {'✅ Yes' if all(emb['dimension'] == mixed_result[0]['dimension'] for emb in mixed_result) else '❌ No'}")


def main():
    """Run all embedding tool demonstrations."""
    print("🚀 EMBEDDING TOOL - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the full capabilities of the EmbedderTool")
    print("including Islamic content processing, batch operations, and more.")
    
    try:
        # Run all demonstrations
        demo_basic_embedding()
        demo_islamic_content()
        demo_chunk_processing()
        demo_output_formats()
        demo_batch_processing()
        demo_transcript_processing()
        demo_configuration_options()
        demo_error_handling()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\n✅ All embedding tool features demonstrated successfully!")
        print("\n🔧 Key Features Showcased:")
        print("   • Basic text embedding generation")
        print("   • Islamic content processing (Arabic + English)")
        print("   • Chunk dictionary metadata preservation")
        print("   • Multiple output formats (list, numpy, dict)")
        print("   • Batch processing optimization")
        print("   • Transcript timestamp handling")
        print("   • Configuration override capabilities")
        print("   • Robust error handling")
        print("   • Dimension consistency")
        print("   • CrewAI BaseTool integration")
        
        print("\n🚀 The Embedding Tool is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()