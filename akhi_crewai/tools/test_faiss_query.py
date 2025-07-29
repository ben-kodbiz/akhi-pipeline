#!/usr/bin/env python3
"""
Test Suite for FAISS Query Tool

Comprehensive tests for the FAISSQueryTool including:
- Basic similarity search
- Text and vector queries
- Metadata filtering
- Result ranking and reranking
- Integration with FAISSStorageTool
- Error handling
- Performance testing

Author: Assistant
Date: 2024
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from faiss_query import FAISSQueryTool
    from faiss_store import FAISSStorageTool
    from embedder import EmbedderTool
except ImportError as e:
    print(f"Error importing tools: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install faiss-cpu sentence-transformers")
    sys.exit(1)


class TestFAISSQueryTool(unittest.TestCase):
    """Test cases for FAISS Query Tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.query_tool = FAISSQueryTool()
        self.storage_tool = FAISSStorageTool()
        self.embedder_tool = EmbedderTool()
        
        # Create test data
        self.test_texts = [
            "Islam is a monotheistic religion founded by Prophet Muhammad.",
            "The Quran is the holy book of Islam revealed to Prophet Muhammad.",
            "Muslims perform five daily prayers called Salah.",
            "Zakat is the third pillar of Islam involving charity.",
            "Hajj is the pilgrimage to Mecca that Muslims must perform.",
            "Ramadan is the holy month of fasting for Muslims.",
            "The Shahada is the declaration of faith in Islam.",
            "Prophet Muhammad is the final messenger of Allah.",
            "The Kaaba in Mecca is the holiest site in Islam.",
            "Islamic law is called Sharia and guides Muslim life."
        ]
        
        # Create test index
        self._create_test_index()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_index(self):
        """Create a test FAISS index with embeddings."""
        # Generate embeddings for test data
        embedding_result = self.embedder_tool._run(
            text_chunks=self.test_texts,
            output_format="list",
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Extract embeddings and create metadata
        embeddings = [item['embedding'] for item in embedding_result]
        metadata = [
            {
                "text": item['text'],
                "chunk_id": i,
                "source": "test_data",
                "topic": self._classify_topic(item['text']),
                "word_count": len(item['text'].split()),
                "contains_arabic": item.get('contains_arabic', False),
                "dimension": item['dimension'],
                "model_name": item['model_name']
            }
            for i, item in enumerate(embedding_result)
        ]
        
        # Store in FAISS
        storage_result = self.storage_tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="test_index",
            index_type="flat",
            save_index=True,
            index_dir=self.temp_dir
        )
        
        self.assertTrue(storage_result.success)
        self.test_index_name = "test_index"
    
    def _classify_topic(self, text: str) -> str:
        """Simple topic classification for test metadata."""
        text_lower = text.lower()
        if "prayer" in text_lower or "salah" in text_lower:
            return "prayer"
        elif "quran" in text_lower:
            return "quran"
        elif "pilgrimage" in text_lower or "hajj" in text_lower:
            return "pilgrimage"
        elif "charity" in text_lower or "zakat" in text_lower:
            return "charity"
        elif "fasting" in text_lower or "ramadan" in text_lower:
            return "fasting"
        else:
            return "general"
    
    def test_initialization(self):
        """Test FAISSQueryTool initialization."""
        tool = FAISSQueryTool()
        self.assertEqual(tool.name, "FAISS Query Tool")
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.embedder_tool)
    
    def test_basic_text_query(self):
        """Test basic text similarity search."""
        query = "What is the holy book of Islam?"
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=3,
            index_dir=self.temp_dir
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.index_name, self.test_index_name)
        self.assertGreater(result.total_results, 0)
        self.assertLessEqual(result.total_results, 3)
        self.assertGreater(result.processing_time, 0)
        
        # Check result structure
        for res in result.results:
            self.assertIn('rank', res)
            self.assertIn('score', res)
            self.assertIn('similarity', res)
            self.assertIn('vector_id', res)
            self.assertIn('text', res)
    
    def test_vector_query(self):
        """Test query with embedding vector."""
        # Get an embedding from our test data
        embedding_result = self.embedder_tool._run(
            text_chunks=["Quran"],
            output_format="list"
        )
        query_embedding = embedding_result[0]['embedding']
        
        result = self.query_tool._run(
            query=query_embedding,
            index_name=self.test_index_name,
            k=2,
            index_dir=self.temp_dir,
            embed_query=False
        )
        
        self.assertTrue(result.success)
        self.assertGreater(result.total_results, 0)
        self.assertIn("Vector query", result.query_text)
    
    def test_metadata_filtering(self):
        """Test metadata-based filtering."""
        query = "Islamic practices"
        filter_criteria = {"topic": "prayer"}
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=10,
            index_dir=self.temp_dir,
            filter_metadata=filter_criteria
        )
        
        self.assertTrue(result.success)
        
        # Check that all results match the filter
        for res in result.results:
            if 'metadata' in res:
                self.assertEqual(res['metadata']['topic'], 'prayer')
    
    def test_threshold_search(self):
        """Test threshold-based search."""
        query = "What is Islam?"
        threshold = 0.8
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=10,
            index_dir=self.temp_dir,
            search_type="threshold",
            threshold=threshold
        )
        
        self.assertTrue(result.success)
        
        # Check that all results meet the threshold
        for res in result.results:
            self.assertLessEqual(res['score'], threshold)
    
    def test_reranking(self):
        """Test result reranking by metadata field."""
        query = "Islamic knowledge"
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=5,
            index_dir=self.temp_dir,
            rerank_by="word_count"
        )
        
        self.assertTrue(result.success)
        
        # Check that results are reranked (word_count should be in descending order for ties)
        if len(result.results) > 1:
            word_counts = []
            for res in result.results:
                if 'metadata' in res and 'word_count' in res['metadata']:
                    word_counts.append(res['metadata']['word_count'])
            
            # Should be sorted in descending order
            if len(word_counts) > 1:
                self.assertGreaterEqual(word_counts[0], word_counts[-1])
    
    def test_include_embeddings(self):
        """Test including embeddings in results."""
        query = "Islam"
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=2,
            index_dir=self.temp_dir,
            include_embeddings=True
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.query_embedding)
        self.assertIsInstance(result.query_embedding, list)
    
    def test_no_metadata_mode(self):
        """Test search without metadata."""
        query = "Islamic teachings"
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=3,
            index_dir=self.temp_dir,
            include_metadata=False
        )
        
        self.assertTrue(result.success)
        
        # Check that metadata is not included
        for res in result.results:
            self.assertNotIn('metadata', res)
            self.assertNotIn('text', res)  # Text comes from metadata
    
    def test_empty_results(self):
        """Test handling of queries with no results."""
        # Use a very restrictive filter that should return no results
        query = "Islam"
        filter_criteria = {"topic": "nonexistent_topic"}
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=5,
            index_dir=self.temp_dir,
            filter_metadata=filter_criteria
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.total_results, 0)
        self.assertEqual(len(result.results), 0)
    
    def test_invalid_index(self):
        """Test error handling for invalid index."""
        query = "Test query"
        
        result = self.query_tool._run(
            query=query,
            index_name="nonexistent_index",
            k=3,
            index_dir=self.temp_dir
        )
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.message)
    
    def test_invalid_search_type(self):
        """Test validation of search type."""
        from pydantic import ValidationError
        
        # Test that ValidationError is raised for invalid search_type
        try:
            result = self.query_tool._run(
                query="test",
                index_name=self.test_index_name,
                search_type="invalid_type",
                index_dir=self.temp_dir
            )
            # If no exception was raised, check that the result indicates failure
            self.assertFalse(result.success)
            self.assertIn("search_type must be one of", result.message)
        except ValidationError:
            # This is the expected behavior - validation should catch invalid input
            pass
    
    def test_utility_methods(self):
        """Test utility methods."""
        # Test search_similar
        results = self.query_tool.search_similar(
            query="What is the Quran?",
            index_name=self.test_index_name,
            k=2,
            index_dir=self.temp_dir
        )
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Test search_with_threshold
        threshold_results = self.query_tool.search_with_threshold(
            query="Islamic practices",
            index_name=self.test_index_name,
            threshold=1.0,
            index_dir=self.temp_dir
        )
        
        self.assertIsInstance(threshold_results, list)
        
        # Test list_available_indices
        available_indices = self.query_tool.list_available_indices(self.temp_dir)
        self.assertIn(self.test_index_name, available_indices)
    
    def test_statistics_generation(self):
        """Test query statistics generation."""
        query = "Islamic knowledge"
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=5,
            index_dir=self.temp_dir
        )
        
        self.assertTrue(result.success)
        self.assertIn('statistics', result.__dict__)
        
        stats = result.statistics
        self.assertIn('total_results', stats)
        self.assertIn('processing_time', stats)
        self.assertIn('avg_similarity', stats)
        self.assertIn('min_similarity', stats)
        self.assertIn('max_similarity', stats)
        
        if result.total_results > 0:
            self.assertGreaterEqual(stats['avg_similarity'], 0)
            self.assertLessEqual(stats['avg_similarity'], 1)
    
    def test_large_k_value(self):
        """Test handling of k larger than index size."""
        query = "Islam"
        large_k = 100  # Larger than our test index
        
        result = self.query_tool._run(
            query=query,
            index_name=self.test_index_name,
            k=large_k,
            index_dir=self.temp_dir
        )
        
        self.assertTrue(result.success)
        self.assertLessEqual(result.total_results, len(self.test_texts))
    
    def test_integration_with_storage_tool(self):
        """Test integration between query and storage tools."""
        # Create a new index with specific content
        new_texts = [
            "The Five Pillars of Islam are fundamental practices.",
            "Shahada is the declaration of faith.",
            "Salah refers to the five daily prayers."
        ]
        
        # Generate embeddings and store
        embedding_result = self.embedder_tool._run(
            text_chunks=new_texts,
            output_format="list"
        )
        
        embeddings = [item['embedding'] for item in embedding_result]
        metadata = [
            {
                "text": item['text'],
                "chunk_id": i,
                "source": "integration_test"
            }
            for i, item in enumerate(embedding_result)
        ]
        
        # Store in new index
        storage_result = self.storage_tool._run(
            embeddings=embeddings,
            metadata=metadata,
            index_name="integration_test",
            index_dir=self.temp_dir
        )
        
        self.assertTrue(storage_result.success)
        
        # Query the new index
        query_result = self.query_tool._run(
            query="What are the pillars of Islam?",
            index_name="integration_test",
            k=3,
            index_dir=self.temp_dir
        )
        
        self.assertTrue(query_result.success)
        self.assertGreater(query_result.total_results, 0)
        
        # Verify content
        found_pillars = False
        for res in query_result.results:
            if 'text' in res and 'Pillars' in res['text']:
                found_pillars = True
                break
        
        self.assertTrue(found_pillars)


if __name__ == '__main__':
    print("Running FAISS Query Tool Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFAISSQueryTool)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed! FAISS Query Tool is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)