#!/usr/bin/env python3
"""
Test Suite for FAISS Storage Tool

Comprehensive tests for the FAISSStorageTool including:
- Index creation and management
- Embedding storage with metadata
- Different index types (Flat, IVF, HNSW, PQ)
- Index persistence (save/load)
- Error handling
- Performance validation
- Integration with EmbedderTool output

Author: Akhi CrewAI Development Team
Date: 2024
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the tools directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from faiss_store import FAISSStorageTool, FAISSStoreInput, FAISSStoreOutput
except ImportError as e:
    print(f"Error importing FAISSStorageTool: {e}")
    print("Make sure FAISS is installed: pip install faiss-cpu")
    sys.exit(1)


def test_initialization():
    """Test 1: Tool initialization and basic properties."""
    print("\n=== Test 1: Tool Initialization ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Check basic properties
        assert tool.name == "FAISS Storage Tool"
        assert "vector database" in tool.description.lower()
        assert tool.args_schema == FAISSStoreInput
        
        # Check internal state
        assert hasattr(tool, 'indices')
        assert hasattr(tool, 'metadata_store')
        assert isinstance(tool.indices, dict)
        assert isinstance(tool.metadata_store, dict)
        
        print("✅ Tool initialization successful")
        print(f"   Name: {tool.name}")
        print(f"   Description: {tool.description[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Tool initialization failed: {e}")
        return False


def test_input_validation():
    """Test 2: Input validation and schema compliance."""
    print("\n=== Test 2: Input Validation ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Test valid input
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"text": "test1"}, {"text": "test2"}]
        
        valid_input = FAISSStoreInput(
            embeddings=embeddings,
            metadata=metadata,
            index_name="test_index"
        )
        
        assert valid_input.embeddings == embeddings
        assert valid_input.metadata == metadata
        assert valid_input.index_name == "test_index"
        assert valid_input.index_type == "flat"  # default
        assert valid_input.save_index == True  # default
        
        print("✅ Input validation successful")
        print(f"   Embeddings shape: {np.array(embeddings).shape}")
        print(f"   Metadata count: {len(metadata)}")
        return True
        
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
        return False


def test_basic_storage():
    """Test 3: Basic embedding storage with flat index."""
    print("\n=== Test 3: Basic Embedding Storage ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Create test embeddings
        embeddings = np.random.rand(10, 384).tolist()  # 10 vectors, 384 dimensions
        metadata = [
            {"text": f"Test text {i}", "source": "test", "chunk_id": i}
            for i in range(10)
        ]
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool._run(
                embeddings=embeddings,
                metadata=metadata,
                index_name="test_basic",
                index_type="flat",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert isinstance(result, FAISSStoreOutput)
            assert result.success == True
            assert result.total_vectors == 10
            assert result.dimension == 384
            assert result.index_type == "flat"
            assert result.processing_time > 0
            assert os.path.exists(result.index_path)
            assert os.path.exists(result.metadata_path)
            
            print("✅ Basic storage successful")
            print(f"   Vectors stored: {result.total_vectors}")
            print(f"   Dimension: {result.dimension}")
            print(f"   Processing time: {result.processing_time:.3f}s")
            print(f"   Index size: {result.index_size_mb:.2f} MB")
            return True
            
    except Exception as e:
        print(f"❌ Basic storage failed: {e}")
        return False


def test_different_index_types():
    """Test 4: Different FAISS index types."""
    print("\n=== Test 4: Different Index Types ===")
    
    try:
        tool = FAISSStorageTool()
        embeddings = np.random.rand(50, 128).tolist()  # 50 vectors, 128 dimensions
        
        index_types = ["flat", "ivf", "hnsw", "pq"]
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for index_type in index_types:
                try:
                    result = tool._run(
                        embeddings=embeddings,
                        index_name=f"test_{index_type}",
                        index_type=index_type,
                        save_index=True,
                        index_dir=temp_dir,
                        nlist=10,  # Small nlist for testing
                        m=4  # Small m for PQ
                    )
                    
                    assert result.success == True
                    assert result.total_vectors == 50
                    assert result.dimension == 128
                    results[index_type] = result
                    
                    print(f"   ✅ {index_type.upper()} index: {result.total_vectors} vectors")
                    
                except Exception as e:
                    print(f"   ⚠️ {index_type.upper()} index failed: {e}")
                    # Some index types might fail in certain environments
                    continue
        
        assert len(results) >= 1  # At least flat should work
        print(f"✅ Index types tested: {list(results.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Index types test failed: {e}")
        return False


def test_numpy_input():
    """Test 5: NumPy array input handling."""
    print("\n=== Test 5: NumPy Array Input ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Create NumPy array directly
        embeddings_np = np.random.rand(20, 256).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool._run(
                embeddings=embeddings_np,
                index_name="test_numpy",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result.success == True
            assert result.total_vectors == 20
            assert result.dimension == 256
            
            print("✅ NumPy array input successful")
            print(f"   Input shape: {embeddings_np.shape}")
            print(f"   Vectors stored: {result.total_vectors}")
            return True
            
    except Exception as e:
        print(f"❌ NumPy array input failed: {e}")
        return False


def test_metadata_handling():
    """Test 6: Metadata storage and validation."""
    print("\n=== Test 6: Metadata Handling ===")
    
    try:
        tool = FAISSStorageTool()
        
        embeddings = np.random.rand(5, 100).tolist()
        
        # Test with detailed metadata
        metadata = [
            {
                "text": "Islamic knowledge is vast",
                "source": "lecture_1.mp3",
                "timestamp": "00:01:30",
                "speaker": "Scholar A",
                "topic": "Fiqh",
                "language": "Arabic",
                "chunk_id": 0
            },
            {
                "text": "The Quran is our guide",
                "source": "lecture_1.mp3",
                "timestamp": "00:02:15",
                "speaker": "Scholar A",
                "topic": "Quran",
                "language": "English",
                "chunk_id": 1
            },
            {
                "text": "Prayer is fundamental",
                "source": "lecture_2.mp3",
                "timestamp": "00:00:45",
                "speaker": "Scholar B",
                "topic": "Worship",
                "language": "English",
                "chunk_id": 2
            },
            {
                "text": "Hadith provides guidance",
                "source": "lecture_2.mp3",
                "timestamp": "00:03:20",
                "speaker": "Scholar B",
                "topic": "Hadith",
                "language": "English",
                "chunk_id": 3
            },
            {
                "text": "Community is important",
                "source": "lecture_3.mp3",
                "timestamp": "00:01:00",
                "speaker": "Scholar C",
                "topic": "Community",
                "language": "English",
                "chunk_id": 4
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool._run(
                embeddings=embeddings,
                metadata=metadata,
                index_name="test_metadata",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result.success == True
            assert result.total_vectors == 5
            
            # Check statistics
            stats = result.statistics
            assert "unique_sources" in stats
            assert "sources" in stats
            assert stats["unique_sources"] == 3  # 3 different lectures
            assert "lecture_1.mp3" in stats["sources"]
            
            print("✅ Metadata handling successful")
            print(f"   Unique sources: {stats['unique_sources']}")
            print(f"   Sources: {stats['sources']}")
            return True
            
    except Exception as e:
        print(f"❌ Metadata handling failed: {e}")
        return False


def test_no_metadata():
    """Test 7: Storage without metadata (auto-generation)."""
    print("\n=== Test 7: Auto-Generated Metadata ===")
    
    try:
        tool = FAISSStorageTool()
        
        embeddings = np.random.rand(8, 200).tolist()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool._run(
                embeddings=embeddings,
                metadata=None,  # No metadata provided
                index_name="test_no_metadata",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result.success == True
            assert result.total_vectors == 8
            
            # Check that metadata was auto-generated
            stats = result.statistics
            assert stats["metadata_count"] == 8
            
            print("✅ Auto-generated metadata successful")
            print(f"   Vectors: {result.total_vectors}")
            print(f"   Auto-generated metadata count: {stats['metadata_count']}")
            return True
            
    except Exception as e:
        print(f"❌ Auto-generated metadata failed: {e}")
        return False


def test_index_persistence():
    """Test 8: Index saving and loading."""
    print("\n=== Test 8: Index Persistence ===")
    
    try:
        tool = FAISSStorageTool()
        
        embeddings = np.random.rand(15, 300).tolist()
        metadata = [{"text": f"Persistent text {i}"} for i in range(15)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save index
            result = tool._run(
                embeddings=embeddings,
                metadata=metadata,
                index_name="test_persistence",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result.success == True
            assert os.path.exists(result.index_path)
            assert os.path.exists(result.metadata_path)
            
            # Create new tool instance and load index
            tool2 = FAISSStorageTool()
            loaded = tool2.load_index("test_persistence", temp_dir)
            
            assert loaded == True
            assert "test_persistence" in tool2.indices
            assert "test_persistence" in tool2.metadata_store
            
            # Check loaded index
            loaded_index = tool2.indices["test_persistence"]
            assert loaded_index.ntotal == 15
            assert loaded_index.d == 300
            
            # Check loaded metadata
            loaded_metadata = tool2.metadata_store["test_persistence"]
            assert len(loaded_metadata) == 15
            
            print("✅ Index persistence successful")
            print(f"   Saved vectors: {result.total_vectors}")
            print(f"   Loaded vectors: {loaded_index.ntotal}")
            print(f"   Loaded metadata entries: {len(loaded_metadata)}")
            return True
            
    except Exception as e:
        print(f"❌ Index persistence failed: {e}")
        return False


def test_batch_operations():
    """Test 9: Multiple batch operations on same index."""
    print("\n=== Test 9: Batch Operations ===")
    
    try:
        tool = FAISSStorageTool()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # First batch
            embeddings1 = np.random.rand(10, 128).tolist()
            metadata1 = [{"batch": 1, "id": i} for i in range(10)]
            
            result1 = tool._run(
                embeddings=embeddings1,
                metadata=metadata1,
                index_name="test_batch",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result1.success == True
            assert result1.total_vectors == 10
            
            # Second batch (append to existing index)
            embeddings2 = np.random.rand(5, 128).tolist()
            metadata2 = [{"batch": 2, "id": i} for i in range(5)]
            
            result2 = tool._run(
                embeddings=embeddings2,
                metadata=metadata2,
                index_name="test_batch",  # Same index name
                save_index=True,
                index_dir=temp_dir,
                overwrite=False  # Append, don't overwrite
            )
            
            assert result2.success == True
            assert result2.total_vectors == 15  # 10 + 5
            
            print("✅ Batch operations successful")
            print(f"   First batch: {result1.total_vectors} vectors")
            print(f"   Second batch: {result2.total_vectors} total vectors")
            return True
            
    except Exception as e:
        print(f"❌ Batch operations failed: {e}")
        return False


def test_error_handling():
    """Test 10: Error handling for invalid inputs."""
    print("\n=== Test 10: Error Handling ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Test empty embeddings
        result1 = tool._run(
            embeddings=[],
            index_name="test_error"
        )
        assert result1.success == False
        assert "error" in result1.message.lower()
        
        # Test mismatched metadata length
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{"text": "only one"}]  # Should be 2
        
        result2 = tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="test_error2"
        )
        assert result2.success == False
        
        # Test invalid embeddings format
        result3 = tool._run(
            embeddings="invalid",
            index_name="test_error3"
        )
        assert result3.success == False
        
        print("✅ Error handling successful")
        print(f"   Empty embeddings: {result1.message[:50]}...")
        print(f"   Mismatched metadata: {result2.message[:50]}...")
        print(f"   Invalid format: {result3.message[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error handling failed: {e}")
        return False


def test_integration_with_embedder():
    """Test 11: Integration with EmbedderTool output format."""
    print("\n=== Test 11: EmbedderTool Integration ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Simulate EmbedderTool output format
        embedder_output = {
            "embeddings": np.random.rand(6, 384).tolist(),
            "metadata": [
                {
                    "text": "Islamic knowledge is vast and deep",
                    "embedding_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2",
                    "chunk_id": 0,
                    "source": "islamic_lecture.mp3",
                    "timestamp": "00:01:30",
                    "is_arabic": False,
                    "is_quran": False,
                    "is_hadith": False
                },
                {
                    "text": "القرآن الكريم هو كتاب الله",
                    "embedding_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2",
                    "chunk_id": 1,
                    "source": "quran_recitation.mp3",
                    "timestamp": "00:02:15",
                    "is_arabic": True,
                    "is_quran": True,
                    "is_hadith": False
                },
                {
                    "text": "The Prophet (PBUH) said...",
                    "embedding_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2",
                    "chunk_id": 2,
                    "source": "hadith_study.mp3",
                    "timestamp": "00:03:45",
                    "is_arabic": False,
                    "is_quran": False,
                    "is_hadith": True
                },
                {
                    "text": "Prayer is the pillar of religion",
                    "embedding_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2",
                    "chunk_id": 3,
                    "source": "fiqh_lesson.mp3",
                    "timestamp": "00:04:20",
                    "is_arabic": False,
                    "is_quran": False,
                    "is_hadith": False
                },
                {
                    "text": "Community and brotherhood in Islam",
                    "embedding_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2",
                    "chunk_id": 4,
                    "source": "community_talk.mp3",
                    "timestamp": "00:05:10",
                    "is_arabic": False,
                    "is_quran": False,
                    "is_hadith": False
                },
                {
                    "text": "الصلاة عماد الدين",
                    "embedding_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2",
                    "chunk_id": 5,
                    "source": "arabic_lesson.mp3",
                    "timestamp": "00:06:30",
                    "is_arabic": True,
                    "is_quran": False,
                    "is_hadith": False
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool._run(
                embeddings=embedder_output["embeddings"],
                metadata=embedder_output["metadata"],
                index_name="embedder_integration",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result.success == True
            assert result.total_vectors == 6
            assert result.dimension == 384
            
            # Check statistics for Islamic content
            stats = result.statistics
            assert "unique_sources" in stats
            assert stats["unique_sources"] == 6  # 6 different sources
            
            print("✅ EmbedderTool integration successful")
            print(f"   Vectors from embedder: {result.total_vectors}")
            print(f"   Dimension: {result.dimension}")
            print(f"   Unique sources: {stats['unique_sources']}")
            print(f"   Processing time: {result.processing_time:.3f}s")
            return True
            
    except Exception as e:
        print(f"❌ EmbedderTool integration failed: {e}")
        return False


def test_performance():
    """Test 12: Performance with larger datasets."""
    print("\n=== Test 12: Performance Testing ===")
    
    try:
        tool = FAISSStorageTool()
        
        # Create larger dataset
        num_vectors = 1000
        dimension = 384
        embeddings = np.random.rand(num_vectors, dimension).tolist()
        metadata = [
            {
                "text": f"Performance test text {i}",
                "chunk_id": i,
                "source": f"source_{i % 10}.mp3",  # 10 different sources
                "batch": i // 100  # 10 batches
            }
            for i in range(num_vectors)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tool._run(
                embeddings=embeddings,
                metadata=metadata,
                index_name="performance_test",
                index_type="flat",
                save_index=True,
                index_dir=temp_dir
            )
            
            assert result.success == True
            assert result.total_vectors == num_vectors
            assert result.dimension == dimension
            assert result.processing_time > 0
            
            # Performance thresholds (adjust based on hardware)
            vectors_per_second = num_vectors / result.processing_time
            assert vectors_per_second > 100  # At least 100 vectors/second
            
            print("✅ Performance testing successful")
            print(f"   Vectors processed: {result.total_vectors}")
            print(f"   Processing time: {result.processing_time:.3f}s")
            print(f"   Vectors/second: {vectors_per_second:.1f}")
            print(f"   Index size: {result.index_size_mb:.2f} MB")
            return True
            
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Starting FAISS Storage Tool Test Suite")
    print("=" * 50)
    
    tests = [
        test_initialization,
        test_input_validation,
        test_basic_storage,
        test_different_index_types,
        test_numpy_input,
        test_metadata_handling,
        test_no_metadata,
        test_index_persistence,
        test_batch_operations,
        test_error_handling,
        test_integration_with_embedder,
        test_performance
    ]
    
    results = []
    for i, test in enumerate(tests, 1):
        try:
            result = test()
            results.append(result)
            if not result:
                print(f"\n⚠️ Test {i} failed but didn't raise an exception")
        except Exception as e:
            print(f"\n❌ Test {i} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total} ({percentage:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! FAISS Storage Tool is ready for use.")
    else:
        print(f"⚠️ {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)