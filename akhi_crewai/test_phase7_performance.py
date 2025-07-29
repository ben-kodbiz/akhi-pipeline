#!/usr/bin/env python3
"""
Phase 7: Performance Benchmarking Suite

Comprehensive performance tests for the Akhi CrewAI Pipeline.
Includes throughput testing, latency measurement, resource monitoring,
and scalability assessment.

Author: Assistant
Date: December 2024
Phase: 7 - Testing & Validation
"""

import os
import sys
import json
import time
import pytest
import psutil
import threading
import statistics
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project paths
sys.path.append(os.path.dirname(__file__))

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


class PerformanceMonitor:
    """Utility class for monitoring system performance."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval=0.1):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics = []
        
        def monitor():
            process = psutil.Process()
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory_info = process.memory_info()
                    memory_percent = psutil.virtual_memory().percent
                    
                    self.metrics.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_rss_mb': memory_info.rss / 1024 / 1024,
                        'memory_vms_mb': memory_info.vms / 1024 / 1024,
                        'memory_percent': memory_percent
                    })
                    time.sleep(interval)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics
    
    def get_summary(self):
        """Get performance summary statistics."""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_rss_mb'] for m in self.metrics]
        
        return {
            'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg_mb': statistics.mean(memory_values),
            'memory_max_mb': max(memory_values),
            'memory_peak_percent': max(m['memory_percent'] for m in self.metrics)
        }


class TestThroughputPerformance:
    """Tests for system throughput and processing speed."""
    
    def test_video_search_throughput(self):
        """Test video search throughput with multiple queries."""
        search_tool = YouTubeSearchTool()
        
        test_queries = [
            "islamic prayer", "quran recitation", "hajj pilgrimage",
            "ramadan fasting", "islamic history", "prophet muhammad",
            "islamic education", "muslim community", "islamic art",
            "islamic finance"
        ]
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        results = []
        
        with patch('tools.youtube_search.YouTubeSearchTool._run') as mock_search:
            mock_search.return_value = json.dumps({
                'videos': [{'title': f'Video {i}'} for i in range(5)],
                'total_results': 5
            })
            
            for query in test_queries:
                query_start = time.time()
                result = search_tool._run(query=query, max_results=5)
                query_duration = time.time() - query_start
                
                results.append({
                    'query': query,
                    'duration': query_duration,
                    'success': 'videos' in json.loads(result)
                })
        
        total_duration = time.time() - start_time
        performance_metrics = monitor.stop_monitoring()
        
        # Performance assertions
        avg_query_time = statistics.mean([r['duration'] for r in results])
        throughput = len(test_queries) / total_duration  # queries per second
        
        assert avg_query_time < 2.0, f"Average query time too high: {avg_query_time:.2f}s"
        assert throughput > 2.0, f"Throughput too low: {throughput:.2f} queries/sec"
        assert all(r['success'] for r in results), "Some queries failed"
        
        # Resource usage assertions
        summary = monitor.get_summary()
        assert summary['cpu_avg'] < 80.0, f"High CPU usage: {summary['cpu_avg']:.1f}%"
        assert summary['memory_peak_percent'] < 90.0, f"High memory usage: {summary['memory_peak_percent']:.1f}%"
    
    def test_embedding_batch_throughput(self):
        """Test embedding generation throughput with batch processing."""
        embedder = EmbedderTool()
        
        # Create test chunks of varying sizes
        test_chunks = []
        for i in range(100):
            chunk_size = 50 + (i % 200)  # Varying chunk sizes
            text = f"This is test chunk {i} about Islamic teachings. " * (chunk_size // 50)
            test_chunks.append({
                'text': text,
                'metadata': {'chunk_id': i}
            })
        
        batch_sizes = [10, 20, 50]
        results = {}
        
        for batch_size in batch_sizes:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            start_time = time.time()
            
            with patch('tools.embedder.EmbedderTool._run') as mock_embed:
                # Mock embeddings with realistic processing time
                def mock_embedding_generation(*args, **kwargs):
                    time.sleep(0.01 * batch_size)  # Simulate processing time
                    return json.dumps([
                        {'embedding': [0.1] * 384, 'metadata': chunk['metadata']}
                        for chunk in kwargs.get('text_chunks', [])
                    ])
                
                mock_embed.side_effect = mock_embedding_generation
                
                # Process in batches
                for i in range(0, len(test_chunks), batch_size):
                    batch = test_chunks[i:i + batch_size]
                    result = embedder._run(
                        text_chunks=batch,
                        batch_size=batch_size,
                        output_format='list'
                    )
                    embeddings = json.loads(result)
                    assert len(embeddings) == len(batch)
            
            duration = time.time() - start_time
            performance_metrics = monitor.stop_monitoring()
            
            throughput = len(test_chunks) / duration  # chunks per second
            results[batch_size] = {
                'duration': duration,
                'throughput': throughput,
                'performance': monitor.get_summary()
            }
        
        # Verify batch processing efficiency
        for batch_size, metrics in results.items():
            assert metrics['throughput'] > 10.0, f"Low throughput for batch {batch_size}: {metrics['throughput']:.1f} chunks/sec"
            assert metrics['performance']['cpu_avg'] < 85.0, f"High CPU for batch {batch_size}"
    
    def test_concurrent_processing_throughput(self):
        """Test throughput with concurrent processing."""
        answer_generator = AnswerGeneratorTool()
        
        test_questions = [
            "What are the Five Pillars of Islam?",
            "When do Muslims pray?",
            "What is the significance of Ramadan?",
            "What is the Hajj pilgrimage?",
            "What does the Quran teach?"
        ]
        
        test_context = "Islamic teachings emphasize prayer, charity, fasting, pilgrimage, and faith as fundamental practices."
        
        def process_question(question):
            start_time = time.time()
            
            with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
                mock_answer.return_value = json.dumps({
                    'answer': f"Based on Islamic teachings, {question.lower().replace('?', '')} is important.",
                    'citations': [{'text': test_context}],
                    'confidence': 0.9
                })
                
                result = answer_generator._run(
                    question=question,
                    context_chunks=[{'text': test_context, 'metadata': {}}],
                    language='english'
                )
                
                duration = time.time() - start_time
                return {
                    'question': question,
                    'duration': duration,
                    'success': 'answer' in json.loads(result)
                }
        
        # Test sequential processing
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        sequential_start = time.time()
        sequential_results = [process_question(q) for q in test_questions]
        sequential_duration = time.time() - sequential_start
        
        sequential_metrics = monitor.stop_monitoring()
        
        # Test concurrent processing
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        concurrent_start = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_question, q) for q in test_questions]
            concurrent_results = [future.result() for future in as_completed(futures)]
        concurrent_duration = time.time() - concurrent_start
        
        concurrent_metrics = monitor.stop_monitoring()
        
        # Performance comparisons
        sequential_throughput = len(test_questions) / sequential_duration
        concurrent_throughput = len(test_questions) / concurrent_duration
        
        assert all(r['success'] for r in sequential_results), "Sequential processing failed"
        assert all(r['success'] for r in concurrent_results), "Concurrent processing failed"
        assert concurrent_throughput >= sequential_throughput * 0.8, "Concurrent processing not efficient"


class TestLatencyPerformance:
    """Tests for system latency and response times."""
    
    def test_search_latency(self):
        """Test search operation latency."""
        search_tool = YouTubeSearchTool()
        
        latencies = []
        
        with patch('tools.youtube_search.YouTubeSearchTool._run') as mock_search:
            mock_search.return_value = json.dumps({
                'videos': [{'title': 'Test Video'}],
                'total_results': 1
            })
            
            # Measure latency over multiple runs
            for _ in range(10):
                start_time = time.time()
                result = search_tool._run(query="test query", max_results=1)
                latency = time.time() - start_time
                latencies.append(latency)
                
                assert 'videos' in json.loads(result)
        
        # Latency statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        max_latency = max(latencies)
        
        # Latency assertions
        assert avg_latency < 1.0, f"High average latency: {avg_latency:.3f}s"
        assert p95_latency < 2.0, f"High P95 latency: {p95_latency:.3f}s"
        assert max_latency < 3.0, f"High max latency: {max_latency:.3f}s"
    
    def test_qa_response_latency(self):
        """Test Q&A response latency."""
        answer_generator = AnswerGeneratorTool()
        
        test_cases = [
            {
                'question': "Short question?",
                'context': "Short context.",
                'expected_latency': 1.0
            },
            {
                'question': "What is a medium length question about Islamic practices?",
                'context': "This is a medium length context about Islamic practices and teachings that provides some detail.",
                'expected_latency': 2.0
            },
            {
                'question': "What is a comprehensive question about Islamic history, theology, practices, and cultural significance?",
                'context': "This is a comprehensive context about Islamic history, theology, practices, and cultural significance. It contains detailed information about various aspects of Islam including historical development, theological principles, practical implementations, and cultural impacts across different societies and time periods.",
                'expected_latency': 3.0
            }
        ]
        
        for test_case in test_cases:
            latencies = []
            
            with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
                mock_answer.return_value = json.dumps({
                    'answer': f"Answer to: {test_case['question']}",
                    'citations': [{'text': test_case['context']}],
                    'confidence': 0.9
                })
                
                # Measure latency over multiple runs
                for _ in range(5):
                    start_time = time.time()
                    result = answer_generator._run(
                        question=test_case['question'],
                        context_chunks=[{'text': test_case['context'], 'metadata': {}}],
                        language='english'
                    )
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    assert 'answer' in json.loads(result)
            
            avg_latency = statistics.mean(latencies)
            assert avg_latency < test_case['expected_latency'], \
                f"High latency for {len(test_case['question'])} char question: {avg_latency:.3f}s"
    
    def test_index_query_latency(self):
        """Test FAISS index query latency."""
        query_tool = FAISSQueryTool()
        
        query_sizes = [1, 5, 10, 20]  # Different top_k values
        
        for top_k in query_sizes:
            latencies = []
            
            with patch('tools.faiss_query.FAISSQueryTool._run') as mock_query:
                # Mock results based on top_k
                mock_results = [
                    {
                        'text': f'Result {i}',
                        'score': 0.9 - (i * 0.1),
                        'metadata': {'source': f'test_{i}.mp3'}
                    }
                    for i in range(top_k)
                ]
                
                mock_query.return_value = json.dumps({
                    'results': mock_results,
                    'total_results': top_k
                })
                
                # Measure query latency
                for _ in range(5):
                    start_time = time.time()
                    result = query_tool._run(
                        query_text="test query",
                        index_path="test_index.faiss",
                        top_k=top_k
                    )
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    response = json.loads(result)
                    assert len(response['results']) == top_k
            
            avg_latency = statistics.mean(latencies)
            max_expected_latency = 0.5 + (top_k * 0.05)  # Scale with result size
            
            assert avg_latency < max_expected_latency, \
                f"High query latency for top_k={top_k}: {avg_latency:.3f}s"


class TestResourceUsage:
    """Tests for resource usage and efficiency."""
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during operations."""
        chunker = TextChunkerTool()
        
        # Test with increasing text sizes
        text_sizes = [1000, 5000, 10000, 50000]  # Characters
        
        for size in text_sizes:
            test_text = "This is a test sentence about Islamic teachings. " * (size // 50)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = chunker._run(
                text=test_text,
                chunk_size=500,
                output_format='json'
            )
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            performance_metrics = monitor.stop_monitoring()
            summary = monitor.get_summary()
            
            # Memory usage assertions
            max_memory_increase = size / 1000 + 50  # MB, scale with input size
            assert memory_increase < max_memory_increase, \
                f"Excessive memory usage for {size} chars: {memory_increase:.1f}MB"
            
            # Verify processing succeeded
            chunks_data = json.loads(result)
            assert len(chunks_data['chunks']) > 0
    
    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency during processing."""
        embedder = EmbedderTool()
        
        # Create test workload
        test_chunks = [
            {'text': f'Test chunk {i} about Islamic education and learning.', 'metadata': {}}
            for i in range(50)
        ]
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        with patch('tools.embedder.EmbedderTool._run') as mock_embed:
            # Simulate CPU-intensive embedding generation
            def cpu_intensive_mock(*args, **kwargs):
                # Simulate processing time
                time.sleep(0.1)
                chunks = kwargs.get('text_chunks', [])
                return json.dumps([
                    {'embedding': [0.1] * 384, 'metadata': chunk['metadata']}
                    for chunk in chunks
                ])
            
            mock_embed.side_effect = cpu_intensive_mock
            
            result = embedder._run(
                text_chunks=test_chunks,
                batch_size=10,
                output_format='list'
            )
        
        duration = time.time() - start_time
        performance_metrics = monitor.stop_monitoring()
        summary = monitor.get_summary()
        
        # CPU efficiency assertions
        assert summary['cpu_avg'] > 10.0, "CPU usage too low (not utilizing resources)"
        assert summary['cpu_avg'] < 90.0, f"CPU usage too high: {summary['cpu_avg']:.1f}%"
        
        # Throughput assertion
        throughput = len(test_chunks) / duration
        assert throughput > 5.0, f"Low processing throughput: {throughput:.1f} chunks/sec"
        
        # Verify processing succeeded
        embeddings = json.loads(result)
        assert len(embeddings) == len(test_chunks)
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup after operations."""
        tools = [
            YouTubeSearchTool(),
            TextChunkerTool(),
            EmbedderTool(),
            AnswerGeneratorTool()
        ]
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform operations with each tool
        for tool in tools:
            if isinstance(tool, YouTubeSearchTool):
                with patch('tools.youtube_search.YouTubeSearchTool._run') as mock:
                    mock.return_value = json.dumps({'videos': [], 'total_results': 0})
                    tool._run(query="test", max_results=1)
            
            elif isinstance(tool, TextChunkerTool):
                tool._run(text="Test text", chunk_size=100, output_format='json')
            
            elif isinstance(tool, EmbedderTool):
                with patch('tools.embedder.EmbedderTool._run') as mock:
                    mock.return_value = json.dumps([{'embedding': [0.1] * 384}])
                    tool._run(text_chunks=[{'text': 'test', 'metadata': {}}], output_format='list')
            
            elif isinstance(tool, AnswerGeneratorTool):
                with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock:
                    mock.return_value = json.dumps({'answer': 'test', 'citations': []})
                    tool._run(question="test?", context_chunks=[{'text': 'test', 'metadata': {}}], language='english')
        
        # Allow time for cleanup
        time.sleep(1)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Resource cleanup assertion
        assert memory_increase < 100, f"Memory not properly cleaned up: {memory_increase:.1f}MB increase"


class TestScalabilityPerformance:
    """Tests for system scalability and load handling."""
    
    def test_concurrent_user_simulation(self):
        """Simulate multiple concurrent users."""
        answer_generator = AnswerGeneratorTool()
        
        def simulate_user_session(user_id):
            """Simulate a user session with multiple queries."""
            questions = [
                f"User {user_id}: What are the Five Pillars of Islam?",
                f"User {user_id}: When do Muslims pray?",
                f"User {user_id}: What is Ramadan?"
            ]
            
            results = []
            for question in questions:
                start_time = time.time()
                
                with patch('tools.answer_generator.AnswerGeneratorTool._run') as mock_answer:
                    mock_answer.return_value = json.dumps({
                        'answer': f"Answer for {question}",
                        'citations': [{'text': 'Islamic context'}],
                        'confidence': 0.9
                    })
                    
                    result = answer_generator._run(
                        question=question,
                        context_chunks=[{'text': 'Islamic context', 'metadata': {}}],
                        language='english'
                    )
                    
                    duration = time.time() - start_time
                    results.append({
                        'user_id': user_id,
                        'question': question,
                        'duration': duration,
                        'success': 'answer' in json.loads(result)
                    })
            
            return results
        
        # Test with increasing number of concurrent users
        user_counts = [1, 3, 5, 10]
        
        for num_users in user_counts:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(simulate_user_session, i) for i in range(num_users)]
                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())
            
            total_duration = time.time() - start_time
            performance_metrics = monitor.stop_monitoring()
            summary = monitor.get_summary()
            
            # Scalability assertions
            total_queries = num_users * 3  # 3 questions per user
            throughput = total_queries / total_duration
            
            assert all(r['success'] for r in all_results), f"Some queries failed with {num_users} users"
            assert throughput > 1.0, f"Low throughput with {num_users} users: {throughput:.2f} queries/sec"
            assert summary['cpu_avg'] < 95.0, f"High CPU with {num_users} users: {summary['cpu_avg']:.1f}%"
            assert summary['memory_peak_percent'] < 95.0, f"High memory with {num_users} users"
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        chunker = TextChunkerTool()
        
        # Create increasingly large datasets
        dataset_sizes = [100, 500, 1000, 2000]  # Number of chunks
        
        for size in dataset_sizes:
            large_text = "\n".join([
                f"This is chunk {i} about Islamic teachings and practices. "
                f"It contains information about various aspects of Islam including "
                f"theology, history, and cultural significance."
                for i in range(size)
            ])
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            start_time = time.time()
            
            result = chunker._run(
                text=large_text,
                chunk_size=200,
                output_format='json'
            )
            
            duration = time.time() - start_time
            performance_metrics = monitor.stop_monitoring()
            summary = monitor.get_summary()
            
            chunks_data = json.loads(result)
            actual_chunks = len(chunks_data['chunks'])
            
            # Scalability assertions
            processing_rate = actual_chunks / duration  # chunks per second
            max_expected_duration = size * 0.01 + 5  # Scale with input size
            
            assert duration < max_expected_duration, \
                f"Processing too slow for {size} input chunks: {duration:.2f}s"
            assert processing_rate > 10.0, \
                f"Low processing rate for {size} chunks: {processing_rate:.1f} chunks/sec"
            assert summary['memory_max_mb'] < 1000, \
                f"Excessive memory for {size} chunks: {summary['memory_max_mb']:.1f}MB"


if __name__ == "__main__":
    # Run performance benchmarking tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])