#!/usr/bin/env python3
"""
Demo Script for Text Chunking Tool

This script demonstrates the capabilities of the TextChunkerTool including:
- Different chunking strategies
- Islamic content detection
- Transcript processing
- Configuration options
- Output formats

Author: Assistant
Date: 2024
"""

import os
import sys
import json
from pathlib import Path

# Add the tools directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import TextChunkerTool


def create_sample_islamic_text():
    """Create sample Islamic educational text for demonstration."""
    return """
بسم الله الرحمن الرحيم
In the name of Allah, the Most Gracious, the Most Merciful.

The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him) over a period of 23 years. It contains 114 chapters called Surahs, each containing verses called Ayahs.

One of the most famous verses is Ayat al-Kursi (Quran 2:255), which speaks about the greatness of Allah. The verse states: "Allah - there is no deity except Him, the Ever-Living, the Sustainer of existence. Neither drowsiness overtakes Him nor sleep. To Him belongs whatever is in the heavens and whatever is on the earth."

The five daily prayers (Salah) are one of the Five Pillars of Islam. Muslims are required to pray five times a day: Fajr (dawn), Dhuhr (midday), Asr (afternoon), Maghrib (sunset), and Isha (night). Each prayer has its specific time and consists of a series of physical movements and recitations.

The Prophet Muhammad (peace be upon him) said in a Hadith recorded by Bukhari: "The best of people are those who benefit others." This teaches us the importance of helping our fellow human beings and contributing positively to society.

Zakat, the third pillar of Islam, is the obligatory charity that Muslims must give to help the poor and needy. It is calculated as 2.5% of one's savings and wealth that has been held for a full lunar year. This system ensures wealth circulation and helps reduce inequality in society.

The month of Ramadan is a time of fasting, prayer, reflection, and community. During this holy month, Muslims fast from dawn to sunset, abstaining from food, drink, and other physical needs during daylight hours. The fast is broken each evening with a meal called Iftar.

Hajj, the pilgrimage to Mecca, is the fifth pillar of Islam and must be performed at least once in a lifetime by every Muslim who is physically and financially able. The pilgrimage occurs during the Islamic month of Dhu al-Hijjah and involves several rituals that commemorate the actions of Prophet Ibrahim (Abraham) and his family.

The concept of Taqwa, often translated as God-consciousness or piety, is central to Islamic spirituality. It involves being constantly aware of Allah's presence and striving to live according to His guidance. A person with Taqwa makes decisions based on what pleases Allah rather than personal desires.

Islamic education emphasizes both worldly knowledge and spiritual development. The first word revealed in the Quran was "Iqra" (Read), highlighting the importance of learning and knowledge in Islam. Muslims are encouraged to seek knowledge from the cradle to the grave.

The Ummah, or global Muslim community, represents the unity of believers regardless of their race, nationality, or social status. This concept promotes brotherhood, mutual support, and collective responsibility among Muslims worldwide.
"""


def create_sample_transcript():
    """Create sample transcript JSON for demonstration."""
    return {
        "transcript_text": "Assalamu alaikum brothers and sisters. Today we will discuss the importance of prayer in Islam. The Prophet Muhammad peace be upon him said that prayer is the pillar of religion. When we pray, we connect directly with Allah. The five daily prayers are Fajr, Dhuhr, Asr, Maghrib, and Isha. Each prayer has its own significance and timing.",
        "segments": [
            {
                "text": "Assalamu alaikum brothers and sisters.",
                "start": 0.0,
                "end": 2.5,
                "speaker": "Imam Ahmad"
            },
            {
                "text": "Today we will discuss the importance of prayer in Islam.",
                "start": 2.5,
                "end": 6.0,
                "speaker": "Imam Ahmad"
            },
            {
                "text": "The Prophet Muhammad peace be upon him said that prayer is the pillar of religion.",
                "start": 6.0,
                "end": 11.0,
                "speaker": "Imam Ahmad"
            },
            {
                "text": "When we pray, we connect directly with Allah.",
                "start": 11.0,
                "end": 14.0,
                "speaker": "Imam Ahmad"
            },
            {
                "text": "The five daily prayers are Fajr, Dhuhr, Asr, Maghrib, and Isha.",
                "start": 14.0,
                "end": 18.5,
                "speaker": "Imam Ahmad"
            },
            {
                "text": "Each prayer has its own significance and timing.",
                "start": 18.5,
                "end": 22.0,
                "speaker": "Imam Ahmad"
            }
        ],
        "metadata": {
            "language": "en",
            "duration": 22.0,
            "model_size": "base",
            "device": "cpu"
        }
    }


def demo_basic_chunking():
    """Demonstrate basic text chunking functionality."""
    print("\n" + "=" * 60)
    print("DEMO 1: BASIC TEXT CHUNKING")
    print("=" * 60)
    
    chunker = TextChunkerTool()
    sample_text = create_sample_islamic_text()
    
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Word count: {len(sample_text.split())} words")
    print()
    
    # Basic chunking with default settings
    chunks = chunker._run(
        text=sample_text,
        chunk_size=400,
        chunk_overlap=50,
        strategy="sliding_window",
        include_metadata=True
    )
    
    print(f"Number of chunks created: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i+1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Length: {chunk['char_count']} chars, {chunk['word_count']} words")
        print(f"  Position: {chunk['start_char']}-{chunk['end_char']}")
        print(f"  Text preview: {chunk['text'][:100]}...")
        print(f"  Contains Arabic: {chunk['contains_arabic']}")
        print(f"  Contains Quran: {chunk['contains_quran']}")
        print(f"  Contains Hadith: {chunk['contains_hadith']}")
        print(f"  Topic keywords: {chunk['topic_keywords'][:5]}")
        print()


def demo_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("\n" + "=" * 60)
    print("DEMO 2: CHUNKING STRATEGIES COMPARISON")
    print("=" * 60)
    
    chunker = TextChunkerTool()
    
    # Shorter text for strategy comparison
    text = """
    The Quran is the holy book of Islam. It was revealed to Prophet Muhammad (peace be upon him).
    
    The five daily prayers are fundamental to Islamic practice. They are Fajr, Dhuhr, Asr, Maghrib, and Isha.
    
    Zakat is the third pillar of Islam. It involves giving charity to those in need.
    """
    
    strategies = ['sliding_window', 'semantic', 'sentence', 'paragraph']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} STRATEGY ---")
        
        chunks = chunker._run(
            text=text,
            chunk_size=150,
            chunk_overlap=20,
            strategy=strategy,
            include_metadata=False
        )
        
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk['text'])} chars - '{chunk['text'][:50]}...'")


def demo_islamic_content_detection():
    """Demonstrate Islamic content detection capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 3: ISLAMIC CONTENT DETECTION")
    print("=" * 60)
    
    chunker = TextChunkerTool()
    
    test_texts = [
        "بسم الله الرحمن الرحيم - In the name of Allah, the Most Gracious, the Most Merciful.",
        "The Quran 2:255 is known as Ayat al-Kursi and speaks about Allah's greatness.",
        "This Hadith is recorded in Sahih Bukhari and teaches us about kindness.",
        "The Prophet Muhammad (peace be upon him) emphasized the importance of seeking knowledge.",
        "Regular text without Islamic content for comparison purposes."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text[:60]}...")
        
        chunks = chunker._run(
            text=text,
            chunk_size=200,
            include_metadata=True
        )
        
        if chunks:
            chunk = chunks[0]
            print(f"  Contains Arabic: {chunk['contains_arabic']}")
            print(f"  Contains Quran: {chunk['contains_quran']}")
            print(f"  Contains Hadith: {chunk['contains_hadith']}")
            print(f"  Keywords found: {chunk['topic_keywords']}")


def demo_transcript_processing():
    """Demonstrate transcript JSON processing."""
    print("\n" + "=" * 60)
    print("DEMO 4: TRANSCRIPT PROCESSING")
    print("=" * 60)
    
    chunker = TextChunkerTool()
    transcript_data = create_sample_transcript()
    transcript_json = json.dumps(transcript_data)
    
    print("Processing transcript JSON with timestamps...")
    print(f"Original transcript: {len(transcript_data['transcript_text'])} chars")
    print(f"Number of segments: {len(transcript_data['segments'])}")
    print(f"Duration: {transcript_data['metadata']['duration']} seconds")
    print()
    
    chunks = chunker._run(
        text=transcript_json,
        chunk_size=100,
        chunk_overlap=20,
        strategy="sliding_window",
        include_metadata=True
    )
    
    print(f"Chunks created: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Text: {chunk['text']}")
        print(f"  Start time: {chunk.get('start_time', 'N/A')}")
        print(f"  End time: {chunk.get('end_time', 'N/A')}")
        print(f"  Speaker: {chunk.get('speaker', 'N/A')}")
        print(f"  Islamic keywords: {chunk['topic_keywords']}")
        print()


def demo_output_formats():
    """Demonstrate different output formats."""
    print("\n" + "=" * 60)
    print("DEMO 5: OUTPUT FORMATS")
    print("=" * 60)
    
    chunker = TextChunkerTool()
    text = "Allah is the Most Merciful. The Quran teaches us about compassion and justice."
    
    print("Text to chunk:", text)
    print()
    
    # List format with metadata
    print("--- LIST FORMAT (with metadata) ---")
    list_result = chunker._run(
        text=text,
        chunk_size=50,
        output_format="list",
        include_metadata=True
    )
    print(f"Type: {type(list_result)}")
    print(f"Number of chunks: {len(list_result)}")
    print(f"First chunk keys: {list(list_result[0].keys()) if list_result else 'None'}")
    print()
    
    # JSON format
    print("--- JSON FORMAT ---")
    json_result = chunker._run(
        text=text,
        chunk_size=50,
        output_format="json",
        include_metadata=True
    )
    print(f"Type: {type(json_result)}")
    print(f"JSON preview: {json_result[:200]}...")
    print()
    
    # Simple format without metadata
    print("--- SIMPLE FORMAT (no metadata) ---")
    simple_result = chunker._run(
        text=text,
        chunk_size=50,
        include_metadata=False
    )
    print(f"Type: {type(simple_result)}")
    print(f"First chunk: {simple_result[0] if simple_result else 'None'}")


def demo_configuration_options():
    """Demonstrate configuration options and overrides."""
    print("\n" + "=" * 60)
    print("DEMO 6: CONFIGURATION OPTIONS")
    print("=" * 60)
    
    chunker = TextChunkerTool()
    
    print("Current configuration:")
    config = chunker.config
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    text = "This is a test sentence. This is another sentence. And here is a third sentence for testing."
    
    print("Text to chunk:", text)
    print(f"Text length: {len(text)} characters")
    print()
    
    # Test different configurations
    configs = [
        {"chunk_size": 30, "chunk_overlap": 5, "preserve_sentences": True},
        {"chunk_size": 30, "chunk_overlap": 5, "preserve_sentences": False},
        {"chunk_size": 50, "chunk_overlap": 15, "preserve_sentences": True}
    ]
    
    for i, config_override in enumerate(configs, 1):
        print(f"--- Configuration {i}: {config_override} ---")
        
        chunks = chunker._run(
            text=text,
            **config_override,
            include_metadata=False
        )
        
        print(f"Number of chunks: {len(chunks)}")
        for j, chunk in enumerate(chunks):
            print(f"  Chunk {j+1}: '{chunk['text']}'")
        print()


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("TEXT CHUNKING TOOL - COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases the capabilities of the TextChunkerTool")
    print("for processing Islamic educational content.")
    
    try:
        demo_basic_chunking()
        demo_chunking_strategies()
        demo_islamic_content_detection()
        demo_transcript_processing()
        demo_output_formats()
        demo_configuration_options()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE - KEY FEATURES DEMONSTRATED")
        print("=" * 60)
        print("✓ Multiple chunking strategies (sliding window, semantic, sentence, paragraph)")
        print("✓ Islamic content detection (Arabic text, Quran references, Hadith)")
        print("✓ Topic keyword extraction for Islamic content")
        print("✓ Transcript JSON processing with timestamp preservation")
        print("✓ Multiple output formats (list, JSON, simple)")
        print("✓ Comprehensive metadata generation")
        print("✓ Configuration override capabilities")
        print("✓ Sentence boundary preservation")
        print("✓ CrewAI BaseTool integration")
        print("\n🎉 The Text Chunking Tool is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)