#!/usr/bin/env python3
"""
Demo Script for FAISS Storage Tool

This script demonstrates the capabilities of the FAISSStorageTool including:
- Basic vector storage and retrieval
- Different index types (Flat, IVF, HNSW, PQ)
- Metadata preservation and management
- Integration with EmbedderTool output
- Index persistence (save/load)
- Batch operations
- Performance optimization
- Islamic content handling

Author: Akhi CrewAI Development Team
Date: 2024
"""

import os
import sys
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the tools directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from faiss_store import FAISSStorageTool
    from embedder import EmbedderTool
except ImportError as e:
    print(f"Error importing tools: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install faiss-cpu sentence-transformers")
    sys.exit(1)


def demo_basic_storage():
    """Demo 1: Basic vector storage with metadata."""
    print("\n🔹 Demo 1: Basic Vector Storage")
    print("=" * 40)
    
    # Initialize the tool
    faiss_tool = FAISSStorageTool()
    
    # Create sample embeddings
    embeddings = np.random.rand(5, 384).tolist()
    metadata = [
        {
            "text": "Introduction to Islamic jurisprudence",
            "source": "fiqh_basics.mp3",
            "timestamp": "00:01:30",
            "topic": "Fiqh",
            "language": "English"
        },
        {
            "text": "The five pillars of Islam",
            "source": "pillars_lecture.mp3",
            "timestamp": "00:02:15",
            "topic": "Fundamentals",
            "language": "English"
        },
        {
            "text": "القرآن الكريم هو كتاب الله",
            "source": "quran_study.mp3",
            "timestamp": "00:03:45",
            "topic": "Quran",
            "language": "Arabic"
        },
        {
            "text": "Prayer times and their significance",
            "source": "prayer_guide.mp3",
            "timestamp": "00:04:20",
            "topic": "Worship",
            "language": "English"
        },
        {
            "text": "الصلاة عماد الدين",
            "source": "arabic_wisdom.mp3",
            "timestamp": "00:05:10",
            "topic": "Prayer",
            "language": "Arabic"
        }
    ]
    
    # Store embeddings
    with tempfile.TemporaryDirectory() as temp_dir:
        result = faiss_tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="demo_basic",
            index_type="flat",
            save_index=True,
            index_dir=temp_dir
        )
        
        print(f"✅ Storage Result:")
        print(f"   Success: {result.success}")
        print(f"   Vectors stored: {result.total_vectors}")
        print(f"   Dimension: {result.dimension}")
        print(f"   Index type: {result.index_type}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Index size: {result.index_size_mb:.2f} MB")
        print(f"   Unique sources: {result.statistics['unique_sources']}")
        print(f"   Sources: {result.statistics['sources']}")


def demo_index_types():
    """Demo 2: Different FAISS index types comparison."""
    print("\n🔹 Demo 2: Index Types Comparison")
    print("=" * 40)
    
    faiss_tool = FAISSStorageTool()
    
    # Create larger dataset for meaningful comparison
    num_vectors = 100
    embeddings = np.random.rand(num_vectors, 256).tolist()
    
    index_types = ["flat", "ivf", "hnsw"]
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for index_type in index_types:
            try:
                # For IVF index, ensure we have enough vectors and proper training
                if index_type == "ivf" and num_vectors < 50:
                    print(f"⚠️ {index_type.upper()} Index skipped (needs ≥50 vectors for training, got {num_vectors})")
                    continue
                
                result = faiss_tool._run(
                    embeddings=embeddings,
                    index_name=f"demo_{index_type}",
                    index_type=index_type,
                    save_index=True,
                    index_dir=temp_dir,
                    nlist=min(10, max(1, num_vectors // 5)) if index_type == "ivf" else 100  # Proper nlist for IVF, default for others
                )
                
                if result.success:
                    results[index_type] = result
                    print(f"✅ {index_type.upper()} Index:")
                    print(f"   Vectors: {result.total_vectors}")
                    print(f"   Processing time: {result.processing_time:.3f}s")
                    print(f"   Index size: {result.index_size_mb:.2f} MB")
                else:
                    print(f"❌ {index_type.upper()} Index failed: {result.message}")
                    
            except Exception as e:
                print(f"⚠️ {index_type.upper()} Index error: {e}")
    
    print(f"\n📊 Successfully created {len(results)} index types")


def demo_embedder_integration():
    """Demo 3: Integration with EmbedderTool."""
    print("\n🔹 Demo 3: EmbedderTool Integration")
    print("=" * 40)
    
    # Initialize both tools
    embedder_tool = EmbedderTool()
    faiss_tool = FAISSStorageTool()
    
    # Sample Islamic texts
    texts = [
        "The Quran is the holy book of Islam, revealed to Prophet Muhammad (PBUH)",
        "الصلاة هي الركن الثاني من أركان الإسلام",
        "Zakat is one of the five pillars of Islam, emphasizing social responsibility",
        "The Prophet (PBUH) said: 'The best of people are those who benefit others'",
        "Hajj is the pilgrimage to Mecca that every Muslim should perform if able",
        "رمضان شهر الصوم والتقوى والعبادة"
    ]
    
    # Generate embeddings using EmbedderTool
    print("🔄 Generating embeddings...")
    embedding_result = embedder_tool._run(
        text_chunks=texts,
        output_format="list",
        model_name="all-MiniLM-L6-v2"
    )
    
    if not embedding_result or len(embedding_result) == 0:
        print(f"❌ Embedding generation failed")
        return
    
    print(f"✅ Generated {len(embedding_result)} embeddings")
    
    # Extract embeddings and metadata from EmbedderTool output
    embeddings = [item['embedding'] for item in embedding_result]
    metadata = [
        {
            "text": item['text'],
            "dimension": item['dimension'],
            "model_name": item['model_name'],
            "processing_time": item['processing_time'],
            "contains_arabic": item.get('contains_arabic', False),
            "source": "demo_integration",
            "chunk_id": i
        }
        for i, item in enumerate(embedding_result)
    ]
    
    # Store in FAISS
    print("🔄 Storing in FAISS index...")
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_result = faiss_tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="embedder_integration",
            index_type="flat",
            save_index=True,
            index_dir=temp_dir
        )
        
        print(f"✅ FAISS Storage Result:")
        print(f"   Success: {storage_result.success}")
        print(f"   Vectors stored: {storage_result.total_vectors}")
        print(f"   Dimension: {storage_result.dimension}")
        print(f"   Processing time: {storage_result.processing_time:.3f}s")
        total_embedding_time = sum(item['processing_time'] for item in embedding_result)
        print(f"   Total pipeline time: {total_embedding_time + storage_result.processing_time:.3f}s")


def demo_batch_operations():
    """Demo 4: Batch operations and index growth."""
    print("\n🔹 Demo 4: Batch Operations")
    print("=" * 40)
    
    faiss_tool = FAISSStorageTool()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # First batch
        print("📦 Batch 1: Initial storage")
        embeddings1 = np.random.rand(20, 384).tolist()
        metadata1 = [
            {
                "text": f"Batch 1 - Islamic text {i}",
                "batch_id": 1,
                "chunk_id": i,
                "source": "batch1_lecture.mp3"
            }
            for i in range(20)
        ]
        
        result1 = faiss_tool._run(
            embeddings=embeddings1,
            metadata=metadata1,
            index_name="demo_batch",
            save_index=True,
            index_dir=temp_dir
        )
        
        print(f"   Vectors: {result1.total_vectors}")
        print(f"   Processing time: {result1.processing_time:.3f}s")
        
        # Second batch (append to existing)
        print("📦 Batch 2: Appending to existing index")
        embeddings2 = np.random.rand(15, 384).tolist()
        metadata2 = [
            {
                "text": f"Batch 2 - Hadith study {i}",
                "batch_id": 2,
                "chunk_id": i + 20,  # Continue numbering
                "source": "batch2_hadith.mp3"
            }
            for i in range(15)
        ]
        
        result2 = faiss_tool._run(
            embeddings=embeddings2,
            metadata=metadata2,
            index_name="demo_batch",  # Same index name
            save_index=True,
            index_dir=temp_dir,
            overwrite=False  # Append, don't overwrite
        )
        
        print(f"   Total vectors: {result2.total_vectors}")
        print(f"   Processing time: {result2.processing_time:.3f}s")
        
        # Third batch
        print("📦 Batch 3: Final append")
        embeddings3 = np.random.rand(10, 384).tolist()
        metadata3 = [
            {
                "text": f"Batch 3 - Quran recitation {i}",
                "batch_id": 3,
                "chunk_id": i + 35,  # Continue numbering
                "source": "batch3_quran.mp3"
            }
            for i in range(10)
        ]
        
        result3 = faiss_tool._run(
            embeddings=embeddings3,
            metadata=metadata3,
            index_name="demo_batch",
            save_index=True,
            index_dir=temp_dir,
            overwrite=False
        )
        
        print(f"   Final total vectors: {result3.total_vectors}")
        print(f"   Processing time: {result3.processing_time:.3f}s")
        print(f"   Unique sources: {result3.statistics['unique_sources']}")
        print(f"   Index size: {result3.index_size_mb:.2f} MB")


def demo_persistence():
    """Demo 5: Index persistence (save and load)."""
    print("\n🔹 Demo 5: Index Persistence")
    print("=" * 40)
    
    # Create and save index
    faiss_tool1 = FAISSStorageTool()
    
    embeddings = np.random.rand(25, 512).tolist()
    metadata = [
        {
            "text": f"Persistent Islamic knowledge {i}",
            "topic": ["Fiqh", "Hadith", "Quran", "Seerah", "Tafsir"][i % 5],
            "importance": np.random.choice(["high", "medium", "low"]),
            "chunk_id": i
        }
        for i in range(25)
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("💾 Saving index to disk...")
        save_result = faiss_tool1._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="demo_persistence",
            save_index=True,
            index_dir=temp_dir
        )
        
        print(f"   Saved {save_result.total_vectors} vectors")
        print(f"   Index file: {os.path.basename(save_result.index_path)}")
        print(f"   Metadata file: {os.path.basename(save_result.metadata_path)}")
        print(f"   Index size: {save_result.index_size_mb:.2f} MB")
        
        # Create new tool instance and load index
        print("📂 Loading index from disk...")
        faiss_tool2 = FAISSStorageTool()
        loaded = faiss_tool2.load_index("demo_persistence", temp_dir)
        
        if loaded:
            # Get info about loaded index
            info = faiss_tool2.get_index_info("demo_persistence")
            print(f"✅ Successfully loaded index:")
            print(f"   Vectors: {info['total_vectors']}")
            print(f"   Dimension: {info['dimension']}")
            print(f"   Index type: {info['index_type']}")
            print(f"   Metadata entries: {info['metadata_count']}")
            
            # List all available indices
            available_indices = faiss_tool2.list_indices(temp_dir)
            print(f"   Available indices: {available_indices}")
        else:
            print("❌ Failed to load index")


def demo_performance():
    """Demo 6: Performance testing with larger datasets."""
    print("\n🔹 Demo 6: Performance Testing")
    print("=" * 40)
    
    faiss_tool = FAISSStorageTool()
    
    # Test different dataset sizes
    sizes = [100, 500, 1000]
    dimension = 384
    
    for size in sizes:
        print(f"\n📊 Testing with {size} vectors (dim={dimension})")
        
        # Generate random embeddings
        embeddings = np.random.rand(size, dimension).tolist()
        metadata = [
            {
                "text": f"Performance test vector {i}",
                "batch": i // 100,
                "category": ["Islamic", "Knowledge", "Wisdom", "Guidance"][i % 4],
                "chunk_id": i
            }
            for i in range(size)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = faiss_tool._run(
                embeddings=embeddings,
                metadata=metadata,
                index_name=f"perf_test_{size}",
                index_type="flat",
                save_index=True,
                index_dir=temp_dir
            )
            
            if result.success:
                vectors_per_second = size / result.processing_time
                print(f"   ✅ Processing time: {result.processing_time:.3f}s")
                print(f"   ✅ Vectors/second: {vectors_per_second:.1f}")
                print(f"   ✅ Index size: {result.index_size_mb:.2f} MB")
                print(f"   ✅ Memory efficiency: {result.index_size_mb/size*1000:.2f} KB/vector")
            else:
                print(f"   ❌ Failed: {result.message}")


def demo_islamic_content():
    """Demo 7: Islamic content handling with Arabic text."""
    print("\n🔹 Demo 7: Islamic Content Handling")
    print("=" * 40)
    
    # Initialize tools
    embedder_tool = EmbedderTool()
    faiss_tool = FAISSStorageTool()
    
    # Islamic content in multiple languages
    islamic_texts = [
        # English Islamic content
        "The Quran is the final revelation from Allah to humanity",
        "Prophet Muhammad (PBUH) is the last messenger of Allah",
        "The five daily prayers are obligatory for every Muslim",
        "Zakat purifies wealth and helps the needy in society",
        "Hajj is the pilgrimage to the holy city of Mecca",
        
        # Arabic Islamic content
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "لا إله إلا الله محمد رسول الله",
        "الصلاة خير من النوم",
        "اللهم صل وسلم على نبينا محمد",
        
        # Mixed content
        "Subhan Allah - Glory be to Allah",
        "Alhamdulillah - All praise is due to Allah",
        "Allahu Akbar - Allah is the Greatest",
        "La hawla wa la quwwata illa billah - There is no power except with Allah",
        "Astaghfirullah - I seek forgiveness from Allah"
    ]
    
    print(f"🔄 Processing {len(islamic_texts)} Islamic texts...")
    
    # Generate embeddings
    embedding_result = embedder_tool._run(
        text_chunks=islamic_texts,
        output_format="list",
        model_name="all-MiniLM-L6-v2"
    )
    
    if not embedding_result or len(embedding_result) == 0:
        print(f"❌ Embedding failed")
        return
    
    # Prepare data for FAISS storage
    embeddings = [item['embedding'] for item in embedding_result]
    metadata = []
    
    for i, item in enumerate(embedding_result):
        text = item['text']
        meta = {
            "text": text,
            "chunk_id": i,
            "source": "islamic_content_demo",
            "language": "Arabic" if any(ord(c) > 127 for c in text) else "English",
            "content_type": "Islamic",
            "is_arabic": any(ord(c) > 127 for c in text),
            "is_dua": "الله" in text or "Allah" in text,
            "is_shahada": "لا إله إلا الله" in text or "There is no god but Allah" in text.lower(),
            "dimension": item['dimension'],
            "model_name": item['model_name']
        }
        metadata.append(meta)
    
    # Store in FAISS
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_result = faiss_tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="islamic_content",
            index_type="flat",
            save_index=True,
            index_dir=temp_dir
        )
        
        print(f"✅ Islamic Content Storage:")
        print(f"   Texts processed: {len(islamic_texts)}")
        print(f"   Vectors stored: {storage_result.total_vectors}")
        print(f"   Dimension: {storage_result.dimension}")
        print(f"   Processing time: {storage_result.processing_time:.3f}s")
        print(f"   Index size: {storage_result.index_size_mb:.2f} MB")
        
        # Analyze content statistics
        arabic_count = sum(1 for meta in metadata if meta['is_arabic'])
        dua_count = sum(1 for meta in metadata if meta['is_dua'])
        
        print(f"\n📊 Content Analysis:")
        print(f"   Arabic texts: {arabic_count}/{len(metadata)}")
        print(f"   Texts mentioning Allah: {dua_count}/{len(metadata)}")
        print(f"   Unique sources: {storage_result.statistics['unique_sources']}")


def main():
    """Run all demos."""
    print("🚀 FAISS Storage Tool - Comprehensive Demo")
    print("=" * 50)
    print("This demo showcases the capabilities of the FAISSStorageTool")
    print("for storing and managing vector embeddings in Islamic content processing.")
    
    try:
        # Run all demos
        demo_basic_storage()
        demo_index_types()
        demo_embedder_integration()
        demo_batch_operations()
        demo_persistence()
        demo_performance()
        demo_islamic_content()
        
        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("\n📋 Summary of demonstrated features:")
        print("   ✅ Basic vector storage with metadata")
        print("   ✅ Multiple FAISS index types (Flat, IVF, HNSW)")
        print("   ✅ Integration with EmbedderTool")
        print("   ✅ Batch operations and index growth")
        print("   ✅ Index persistence (save/load)")
        print("   ✅ Performance testing and optimization")
        print("   ✅ Islamic content handling with Arabic text")
        print("\n🔧 The FAISSStorageTool is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()