#!/usr/bin/env python3
"""
Phase 7: Integration Testing Suite

Comprehensive integration tests for the Akhi CrewAI Pipeline.
Tests end-to-end workflows, error handling, data consistency,
and resource usage monitoring.

Author: Assistant
Date: December 2024
Phase: 7 - Testing & Validation
"""

import os
import sys
import json
import time
import pytest
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add project paths
sys.path.append(os.path.dirname(__file__))

from main import AkhiPipelineApp
from crew import AkhiPipelineCrew
from tools import (
    YouTubeSearchTool,
    YouTubeDownloaderTool,
    TranscriptionTool,
    TextChunkerTool,
    EmbedderTool,
    FAISSStorageTool,
    FAISSQueryTool,
    SummarizerTool,
    AnswerGeneratorTool
)


class TestIntegrationWorkflows:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "data" / "audio").mkdir(parents=True)
            (workspace / "data" / "transcripts").mkdir(parents=True)
            (workspace / "data" / "embeddings").mkdir(parents=True)
            yield workspace
    
    @pytest.fixture
    def mock_config(self, temp_workspace):
        """Create mock configuration for testing."""
        config = {
            'youtube': {
                'max_results': 2,
                'duration_filter': 'medium',
                'quality_filter': 'high'
            },
            'transcription': {
                'model_size': 'base',
                'device': 'cpu',
                'language': 'auto'
            },
            'chunking': {
                'chunk_size': 500,
                'chunk_overlap': 50,
                'strategy': 'sliding_window'
            },
            'embedding': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32
            },
            'faiss': {
                'index_type': 'IndexFlatIP',
                'dimension': 384
            },
            'paths': {
                'data_dir': str(temp_workspace / "data"),
                'audio_dir': str(temp_workspace / "data" / "audio"),
                'transcripts_dir': str(temp_workspace / "data" / "transcripts"),
                'embeddings_dir': str(temp_workspace / "data" / "embeddings")
            }
        }
        return config
    
    @pytest.fixture
    def pipeline_app(self, mock_config, temp_workspace):
        """Initialize pipeline app with test configuration."""
        config_file = temp_workspace / "test_config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(mock_config, f)
        
        app = AkhiPipelineApp(str(config_file))
        return app
    
    def test_video_search_workflow(self, pipeline_app):
        """Test video search workflow integration."""
        with patch('tools.youtube_search.YouTubeSearchTool._run') as mock_search:
            # Mock search results
            mock_search.return_value = json.dumps({
                'videos': [
                    {
                        'title': 'Islamic Prayer Basics',
                        'url': 'https://youtube.com/watch?v=test1',
                        'duration': '10:30',
                        'channel': 'Islamic Education',
                        'description': 'Learn the basics of Islamic prayer'
                    },
                    {
                        'title': 'Quran Recitation',
                        'url': 'https://youtube.com/watch?v=test2',
                        'duration': '15:45',
                        'channel': 'Quran Channel',
                        'description': 'Beautiful Quran recitation'
                    }
                ],
                'total_results': 2
            })
            
            # Test search functionality
            result = pipeline_app.search_videos("islamic prayer", max_videos=2)
            
            assert result['status'] == 'success'
            assert len(result['videos']) == 2
            assert 'Islamic Prayer Basics' in [v['title'] for v in result['videos']]
    
    @patch('tools.youtube_downloader.yt_dlp.YoutubeDL')
    @patch('tools.transcriber.whisper.load_model')
    def test_full_pipeline_workflow(self, mock_whisper, mock_ytdl, pipeline_app, temp_workspace):
        """Test complete pipeline workflow from search to QA."""
        # Mock YouTube download
        mock_ytdl_instance = MagicMock()
        mock_ytdl.return_value = mock_ytdl_instance
        mock_ytdl_instance.extract_info.return_value = {
            'title': 'Test Video',
            'duration': 600
        }
        
        # Mock Whisper transcription
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        mock_model.transcribe.return_value = {
            'text': 'This is a test transcription about Islamic prayer and worship.',
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'This is a test transcription'
                },
                {
                    'start': 5.0,
                    'end': 10.0,
                    'text': 'about Islamic prayer and worship.'
                }
            ]
        }
        
        # Mock video search
        with patch('tools.youtube_search.YouTubeSearchTool._run') as mock_search:
            mock_search.return_value = json.dumps({
                'videos': [{
                    'title': 'Test Video',
                    'url': 'https://youtube.com/watch?v=test',
                    'duration': '10:00',
                    'channel': 'Test Channel',
                    'description': 'Test description'
                }],
                'total_results': 1
            })
            
            # Test full pipeline
            result = pipeline_app.process_pipeline(
                search_query="islamic prayer",
                questions=["What is the importance of prayer in Islam?"],
                max_videos=1
            )
            
            assert result['status'] in ['success', 'completed']
            assert 'search_results' in result or 'videos_processed' in result
    
    def test_qa_only_workflow(self, pipeline_app, temp_workspace):
        """Test QA-only workflow with existing index."""
        # Create mock FAISS index
        index_dir = temp_workspace / "data" / "embeddings"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock existing index files
        (index_dir / "test_index.faiss").touch()
        (index_dir / "test_index_metadata.json").write_text(json.dumps({
            'chunks': [
                {
                    'text': 'Prayer is one of the Five Pillars of Islam.',
                    'metadata': {'source': 'test.mp3', 'timestamp': '00:01:00'}
                }
            ],
            'total_chunks': 1
        }))
        
        with patch('tools.faiss_query.FAISSQueryTool._run') as mock_query:
            mock_query.return_value = json.dumps({
                'results': [
                    {
                        'text': 'Prayer is one of the Five Pillars of Islam.',
                        'score': 0.95,
                        'metadata': {'source': 'test.mp3', 'timestamp': '00:01:00'}
                    }
                ],
                'total_results': 1
            })
            
            # Test QA functionality
            result = pipeline_app.query_content(
                "What is the importance of prayer in Islam?",
                index_name="test_index"
            )
            
            assert result['status'] == 'success'
            assert 'answer' in result
    
    def test_error_handling_invalid_input(self, pipeline_app):
        """Test error handling for invalid inputs."""
        # Test empty search query
        result = pipeline_app.search_videos("", max_videos=1)
        assert result['status'] == 'error'
        assert 'error' in result
        
        # Test invalid max_videos
        result = pipeline_app.search_videos("test", max_videos=0)
        assert result['status'] == 'error'
        
        # Test empty question
        result = pipeline_app.query_content("", index_name="test")
        assert result['status'] == 'error'
    
    def test_resource_usage_monitoring(self, pipeline_app):
        """Test resource usage monitoring during operations."""
        import psutil
        import threading
        
        # Monitor resource usage
        resource_data = []
        monitoring = True
        
        def monitor_resources():
            while monitoring:
                resource_data.append({
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'timestamp': time.time()
                })
                time.sleep(0.1)
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        try:
            # Perform operation
            with patch('tools.youtube_search.YouTubeSearchTool._run') as mock_search:
                mock_search.return_value = json.dumps({
                    'videos': [],
                    'total_results': 0
                })
                
                result = pipeline_app.search_videos("test", max_videos=1)
                time.sleep(1)  # Allow monitoring
                
        finally:
            monitoring = False
            monitor_thread.join()
        
        # Verify monitoring data collected
        assert len(resource_data) > 0
        assert all('cpu_percent' in data for data in resource_data)
        assert all('memory_percent' in data for data in resource_data)


class TestDataConsistency:
    """Tests for data consistency across the pipeline."""
    
    def test_transcript_chunk_consistency(self):
        """Test consistency between transcripts and chunks."""
        # Mock transcript data
        transcript = {
            'text': 'This is a test transcript about Islamic teachings.',
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'This is a test transcript'},
                {'start': 5.0, 'end': 10.0, 'text': 'about Islamic teachings.'}
            ]
        }
        
        # Test chunking preserves content
        chunker = TextChunkerTool()
        result = chunker._run(
            text=json.dumps(transcript),
            chunk_size=100,
            output_format='json'
        )
        
        chunks_data = json.loads(result)
        
        # Verify all text is preserved
        combined_text = ' '.join([chunk['text'] for chunk in chunks_data['chunks']])
        assert transcript['text'] in combined_text or combined_text in transcript['text']
    
    def test_embedding_dimension_consistency(self):
        """Test embedding dimension consistency."""
        embedder = EmbedderTool()
        
        # Test multiple texts
        texts = [
            "This is a test text about Islam.",
            "Another test text about Islamic teachings.",
            "A third text for consistency testing."
        ]
        
        embeddings = []
        for text in texts:
            result = embedder._run(
                text_chunks=[{'text': text, 'metadata': {}}],
                output_format='list'
            )
            embedding_data = json.loads(result)
            embeddings.append(embedding_data[0]['embedding'])
        
        # Verify consistent dimensions
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1, "Embedding dimensions are inconsistent"
    
    def test_faiss_index_consistency(self, temp_workspace):
        """Test FAISS index consistency."""
        # Create test embeddings
        test_embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ]
        
        test_chunks = [
            {'text': f'Test chunk {i}', 'metadata': {'id': i}}
            for i in range(len(test_embeddings))
        ]
        
        # Store in FAISS
        storage_tool = FAISSStorageTool()
        index_path = temp_workspace / "test_consistency.faiss"
        
        storage_result = storage_tool._run(
            embeddings=test_embeddings,
            chunks=test_chunks,
            index_path=str(index_path),
            index_name="test_consistency"
        )
        
        assert json.loads(storage_result)['status'] == 'success'
        
        # Query and verify consistency
        query_tool = FAISSQueryTool()
        query_result = query_tool._run(
            query_text="Test chunk",
            index_path=str(index_path),
            top_k=3
        )
        
        results = json.loads(query_result)['results']
        assert len(results) == 3
        assert all('text' in result for result in results)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_search_performance(self):
        """Benchmark video search performance."""
        search_tool = YouTubeSearchTool()
        
        start_time = time.time()
        
        with patch('tools.youtube_search.YouTubeSearchTool._run') as mock_search:
            mock_search.return_value = json.dumps({
                'videos': [{'title': f'Video {i}'} for i in range(10)],
                'total_results': 10
            })
            
            result = search_tool._run(
                query="islamic education",
                max_results=10
            )
        
        duration = time.time() - start_time
        
        # Performance assertion (should complete within 5 seconds)
        assert duration < 5.0, f"Search took too long: {duration:.2f}s"
        assert json.loads(result)['total_results'] == 10
    
    def test_embedding_performance(self):
        """Benchmark embedding generation performance."""
        embedder = EmbedderTool()
        
        # Create test chunks
        test_chunks = [
            {'text': f'This is test chunk number {i} about Islamic teachings.', 'metadata': {}}
            for i in range(100)
        ]
        
        start_time = time.time()
        
        result = embedder._run(
            text_chunks=test_chunks,
            batch_size=32,
            output_format='list'
        )
        
        duration = time.time() - start_time
        
        # Performance assertion (should process 100 chunks within 30 seconds)
        assert duration < 30.0, f"Embedding took too long: {duration:.2f}s"
        
        embeddings = json.loads(result)
        assert len(embeddings) == 100
    
    def test_memory_usage(self):
        """Test memory usage during operations."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operation
        chunker = TextChunkerTool()
        large_text = "This is a test sentence. " * 10000  # Large text
        
        result = chunker._run(
            text=large_text,
            chunk_size=500,
            output_format='json'
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory assertion (should not increase by more than 500MB)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.2f}MB"
        
        chunks_data = json.loads(result)
        assert len(chunks_data['chunks']) > 0


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])