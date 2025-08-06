#!/usr/bin/env python3
"""
Performance Benchmarking Suite for CrewAI Tools - Phase 2.4

This script provides comprehensive performance benchmarking for all CrewAI tools
including latency, throughput, memory usage, and accuracy metrics.

Author: Assistant
Date: 2024
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import tempfile
import tracemalloc

# Add the tools directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

# Import all tools
from tools import (
    YouTubeSearchTool, YouTubeSearchInput,
    YouTubeDownloaderTool, YouTubeDownloadInput,
    TextChunkerTool, ChunkingInput,
    EmbedderTool, EmbeddingInput,
    FAISSStorageTool, FAISSStoreInput,
    FAISSQueryTool, FAISSQueryInput,
    SummarizerTool, SummarizationInput,
    AnswerGeneratorTool, AnswerGenerationInput
)


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""
    tool_name: str
    operation: str
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    throughput_ops_per_sec: Optional[float] = None
    accuracy_score: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "latency_ms": (end_time - self.start_time) * 1000,
            "memory_mb": end_memory - self.start_memory,
            "peak_memory_mb": peak / 1024 / 1024,
            "cpu_percent": max(end_cpu, self.start_cpu)
        }


class BenchmarkSuite:
    """Main benchmarking suite."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="akhi_benchmark_")
        self.results: List[BenchmarkResult] = []
        self.monitor = PerformanceMonitor()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "embeddings"), exist_ok=True)
        
        # Sample data for benchmarking
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data for benchmarking."""
        return {
            "texts": [
                "بسم الله الرحمن الرحيم - In the name of Allah, the Most Gracious, the Most Merciful.",
                "The Quran is the holy book of Islam, revealed to Prophet Muhammad (peace be upon him).",
                "Prayer (Salah) is one of the Five Pillars of Islam, performed five times daily.",
                "Zakat is the third pillar of Islam, involving charity to those in need.",
                "The Hajj pilgrimage to Mecca is a religious duty for Muslims who are able.",
                "Fasting during Ramadan is the fourth pillar of Islam.",
                "The Shahada is the declaration of faith in Islam.",
                "Islamic jurisprudence (Fiqh) provides guidance for Muslim life.",
                "The Sunnah represents the teachings and practices of Prophet Muhammad.",
                "Tawhid is the concept of the oneness and uniqueness of Allah."
            ] * 10,  # 100 texts total
            "queries": [
                "What is Islam?",
                "Tell me about the Five Pillars",
                "What is the Quran?",
                "How do Muslims pray?",
                "What is Hajj?",
                "Explain Islamic fasting",
                "What is Zakat?",
                "Tell me about Prophet Muhammad"
            ],
            "youtube_queries": [
                "Nouman Ali Khan Quran tafseer",
                "Islamic prayer tutorial",
                "Five Pillars of Islam explanation",
                "Hajj pilgrimage guide",
                "Ramadan fasting rules"
            ]
        }
    
    def benchmark_youtube_search(self) -> List[BenchmarkResult]:
        """Benchmark YouTube Search Tool."""
        print("Benchmarking YouTube Search Tool...")
        tool = YouTubeSearchTool()
        results = []
        
        for query in self.sample_data["youtube_queries"]:
            self.monitor.start_monitoring()
            
            try:
                result = tool._run(
                    query=query,
                    max_results=10,
                    duration="medium"
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Check if result contains expected content
                success = "Found" in result or "videos" in result.lower()
                
                benchmark_result = BenchmarkResult(
                    tool_name="YouTubeSearchTool",
                    operation=f"search_query_{len(query)}_chars",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=success,
                    additional_metrics={
                        "query_length": len(query),
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="YouTubeSearchTool",
                    operation=f"search_query_{len(query)}_chars",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
            time.sleep(1)  # Rate limiting
        
        return results
    
    def benchmark_text_chunker(self) -> List[BenchmarkResult]:
        """Benchmark Text Chunker Tool."""
        print("Benchmarking Text Chunker Tool...")
        tool = TextChunkerTool()
        results = []
        
        # Test different text sizes
        text_sizes = [100, 500, 1000, 5000, 10000]  # Number of words
        
        for size in text_sizes:
            # Create text of specified size
            text = " ".join(self.sample_data["texts"][:size % len(self.sample_data["texts"])] * (size // len(self.sample_data["texts"]) + 1))[:size * 5]  # Approximate word count
            
            self.monitor.start_monitoring()
            
            try:
                result = tool._run(
                    text=text,
                    chunk_size=300,
                    chunk_overlap=50,
                    output_format="json"
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Parse result to get chunk count
                result_data = json.loads(result)
                chunk_count = len(result_data.get("chunks", []))
                
                # Calculate throughput (chunks per second)
                throughput = chunk_count / (metrics["latency_ms"] / 1000) if metrics["latency_ms"] > 0 else 0
                
                benchmark_result = BenchmarkResult(
                    tool_name="TextChunkerTool",
                    operation=f"chunk_text_{size}_words",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=result_data.get("success", False),
                    throughput_ops_per_sec=throughput,
                    additional_metrics={
                        "text_length": len(text),
                        "chunk_count": chunk_count,
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="TextChunkerTool",
                    operation=f"chunk_text_{size}_words",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
        
        return results
    
    def benchmark_embedder(self) -> List[BenchmarkResult]:
        """Benchmark Embedder Tool."""
        print("Benchmarking Embedder Tool...")
        tool = EmbedderTool()
        results = []
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        
        for batch_size in batch_sizes:
            text_chunks = self.sample_data["texts"][:batch_size]
            
            self.monitor.start_monitoring()
            
            try:
                result = tool._run(
                    text_chunks=text_chunks,
                    batch_size=min(batch_size, 10),
                    output_format="dict"
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Calculate throughput (embeddings per second)
                throughput = batch_size / (metrics["latency_ms"] / 1000) if metrics["latency_ms"] > 0 else 0
                
                # Check embedding quality
                embeddings = result.get("embeddings", [])
                embedding_dimension = len(embeddings[0]) if embeddings else 0
                
                benchmark_result = BenchmarkResult(
                    tool_name="EmbedderTool",
                    operation=f"embed_{batch_size}_texts",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=len(embeddings) == batch_size,
                    throughput_ops_per_sec=throughput,
                    additional_metrics={
                        "batch_size": batch_size,
                        "embedding_dimension": embedding_dimension,
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="EmbedderTool",
                    operation=f"embed_{batch_size}_texts",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
        
        return results
    
    def benchmark_faiss_storage(self) -> List[BenchmarkResult]:
        """Benchmark FAISS Storage Tool."""
        print("Benchmarking FAISS Storage Tool...")
        tool = FAISSStorageTool()
        results = []
        
        # Test different vector counts
        vector_counts = [100, 500, 1000, 5000]
        
        for count in vector_counts:
            # Generate random embeddings
            embeddings = np.random.rand(count, 384).astype(np.float32)
            metadata = [
                {
                    "text": f"Sample text {i}",
                    "source": f"test_{i}.txt",
                    "islamic_content": i % 2 == 0
                }
                for i in range(count)
            ]
            
            self.monitor.start_monitoring()
            
            try:
                result = tool._run(
                    embeddings=embeddings.tolist(),
                    metadata=metadata,
                    index_name=f"benchmark_{count}",
                    index_dir=os.path.join(self.output_dir, "embeddings"),
                    save_index=True
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Calculate throughput (vectors per second)
                throughput = count / (metrics["latency_ms"] / 1000) if metrics["latency_ms"] > 0 else 0
                
                benchmark_result = BenchmarkResult(
                    tool_name="FAISSStorageTool",
                    operation=f"store_{count}_vectors",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=result.success,
                    throughput_ops_per_sec=throughput,
                    additional_metrics={
                        "vector_count": count,
                        "dimension": 384,
                        "index_size_mb": os.path.getsize(result.index_path) / 1024 / 1024 if os.path.exists(result.index_path) else 0,
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="FAISSStorageTool",
                    operation=f"store_{count}_vectors",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
        
        return results
    
    def benchmark_faiss_query(self) -> List[BenchmarkResult]:
        """Benchmark FAISS Query Tool."""
        print("Benchmarking FAISS Query Tool...")
        
        # First create a test index
        storage_tool = FAISSStorageTool()
        embeddings = np.random.rand(1000, 384).astype(np.float32)
        metadata = [
            {"text": f"Sample text {i}", "source": f"test_{i}.txt"}
            for i in range(1000)
        ]
        
        try:
            storage_result = storage_tool._run(
                embeddings=embeddings.tolist(),
                metadata=metadata,
                index_name="query_benchmark",
                index_dir=os.path.join(self.output_dir, "embeddings")
            )
            
            if not storage_result.success:
                return [BenchmarkResult(
                    tool_name="FAISSQueryTool",
                    operation="setup_failed",
                    latency_ms=0,
                    memory_mb=0,
                    cpu_percent=0,
                    success=False,
                    error_message="Failed to create test index"
                )]
        
        except Exception as e:
            return [BenchmarkResult(
                tool_name="FAISSQueryTool",
                operation="setup_failed",
                latency_ms=0,
                memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message=str(e)
            )]
        
        # Now benchmark queries
        query_tool = FAISSQueryTool()
        results = []
        
        # Test different k values
        k_values = [1, 5, 10, 20, 50]
        
        for k in k_values:
            query_embedding = np.random.rand(384).astype(np.float32)
            
            self.monitor.start_monitoring()
            
            try:
                result = query_tool._run(
                    query=query_embedding.tolist(),
                    index_name="query_benchmark",
                    index_dir=os.path.join(self.output_dir, "embeddings"),
                    k=k,
                    embed_query=False
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Calculate throughput (queries per second)
                throughput = 1 / (metrics["latency_ms"] / 1000) if metrics["latency_ms"] > 0 else 0
                
                benchmark_result = BenchmarkResult(
                    tool_name="FAISSQueryTool",
                    operation=f"query_k_{k}",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=result.success and len(result.results) <= k,
                    throughput_ops_per_sec=throughput,
                    additional_metrics={
                        "k_value": k,
                        "results_returned": len(result.results) if result.success else 0,
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="FAISSQueryTool",
                    operation=f"query_k_{k}",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
        
        return results
    
    def benchmark_summarizer(self) -> List[BenchmarkResult]:
        """Benchmark Summarizer Tool."""
        print("Benchmarking Summarizer Tool...")
        tool = SummarizerTool()
        results = []
        
        # Test different text lengths and strategies
        test_configs = [
            {"length": "short", "strategy": "extractive", "text_multiplier": 5},
            {"length": "medium", "strategy": "extractive", "text_multiplier": 10},
            {"length": "long", "strategy": "extractive", "text_multiplier": 20},
            {"length": "short", "strategy": "abstractive", "text_multiplier": 5},
        ]
        
        for config in test_configs:
            text = " ".join(self.sample_data["texts"] * config["text_multiplier"])
            
            self.monitor.start_monitoring()
            
            try:
                result = tool._run(
                    text=text,
                    summary_length=config["length"],
                    strategy=config["strategy"],
                    preserve_islamic_terms=True
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Parse result
                result_data = json.loads(result)
                
                # Calculate compression ratio
                original_words = len(text.split())
                summary_words = result_data.get("word_count", 0)
                compression_ratio = original_words / summary_words if summary_words > 0 else 0
                
                benchmark_result = BenchmarkResult(
                    tool_name="SummarizerTool",
                    operation=f"summarize_{config['length']}_{config['strategy']}",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=result_data.get("success", False),
                    additional_metrics={
                        "original_words": original_words,
                        "summary_words": summary_words,
                        "compression_ratio": compression_ratio,
                        "islamic_terms_preserved": len(result_data.get("islamic_terms_preserved", [])),
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="SummarizerTool",
                    operation=f"summarize_{config['length']}_{config['strategy']}",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
        
        return results
    
    def benchmark_answer_generator(self) -> List[BenchmarkResult]:
        """Benchmark Answer Generator Tool."""
        print("Benchmarking Answer Generator Tool...")
        tool = AnswerGeneratorTool()
        results = []
        
        # Test different context sizes
        context_sizes = [1, 3, 5, 10]
        
        for size in context_sizes:
            context_chunks = [
                {
                    "text": self.sample_data["texts"][i % len(self.sample_data["texts"])],
                    "metadata": {
                        "source": f"test_{i}.txt",
                        "confidence": 0.9 - (i * 0.1)
                    }
                }
                for i in range(size)
            ]
            
            question = self.sample_data["queries"][0]
            
            self.monitor.start_monitoring()
            
            try:
                result = tool._run(
                    question=question,
                    context_chunks=context_chunks,
                    max_answer_length=200,
                    include_citations=True
                )
                
                metrics = self.monitor.stop_monitoring()
                
                # Parse result
                result_data = json.loads(result)
                
                benchmark_result = BenchmarkResult(
                    tool_name="AnswerGeneratorTool",
                    operation=f"generate_answer_{size}_contexts",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=result_data.get("success", False),
                    additional_metrics={
                        "context_count": size,
                        "answer_length": len(result_data.get("answer", "")),
                        "confidence_score": result_data.get("confidence_score", 0),
                        "citations_count": len(result_data.get("citations", [])),
                        "peak_memory_mb": metrics["peak_memory_mb"]
                    }
                )
                
            except Exception as e:
                metrics = self.monitor.stop_monitoring()
                benchmark_result = BenchmarkResult(
                    tool_name="AnswerGeneratorTool",
                    operation=f"generate_answer_{size}_contexts",
                    latency_ms=metrics["latency_ms"],
                    memory_mb=metrics["memory_mb"],
                    cpu_percent=metrics["cpu_percent"],
                    success=False,
                    error_message=str(e)
                )
            
            results.append(benchmark_result)
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmarks and return results."""
        print("Starting comprehensive tool benchmarking...")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        all_results = {}
        
        # Run individual tool benchmarks
        benchmark_functions = [
            ("youtube_search", self.benchmark_youtube_search),
            ("text_chunker", self.benchmark_text_chunker),
            ("embedder", self.benchmark_embedder),
            ("faiss_storage", self.benchmark_faiss_storage),
            ("faiss_query", self.benchmark_faiss_query),
            ("summarizer", self.benchmark_summarizer),
            ("answer_generator", self.benchmark_answer_generator)
        ]
        
        for name, func in benchmark_functions:
            try:
                print(f"\nRunning {name} benchmarks...")
                results = func()
                all_results[name] = results
                self.results.extend(results)
                
                # Print summary
                success_count = sum(1 for r in results if r.success)
                print(f"  Completed: {success_count}/{len(results)} successful")
                
            except Exception as e:
                print(f"  Error in {name}: {e}")
                all_results[name] = []
        
        return all_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        # Convert results to DataFrame for analysis
        data = []
        for result in self.results:
            row = {
                "tool_name": result.tool_name,
                "operation": result.operation,
                "latency_ms": result.latency_ms,
                "memory_mb": result.memory_mb,
                "cpu_percent": result.cpu_percent,
                "success": result.success,
                "throughput_ops_per_sec": result.throughput_ops_per_sec or 0,
                "error_message": result.error_message or ""
            }
            
            # Add additional metrics
            if result.additional_metrics:
                row.update(result.additional_metrics)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Generate report
        report = []
        report.append("AKHI CREWAI TOOLS PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Successful Tests: {df['success'].sum()}")
        report.append(f"Failed Tests: {(~df['success']).sum()}")
        report.append("")
        
        # Summary by tool
        report.append("PERFORMANCE SUMMARY BY TOOL")
        report.append("-" * 40)
        
        for tool in df['tool_name'].unique():
            tool_data = df[df['tool_name'] == tool]
            successful_data = tool_data[tool_data['success']]
            
            if len(successful_data) > 0:
                report.append(f"\n{tool}:")
                report.append(f"  Success Rate: {len(successful_data)}/{len(tool_data)} ({len(successful_data)/len(tool_data)*100:.1f}%)")
                report.append(f"  Avg Latency: {successful_data['latency_ms'].mean():.2f} ms")
                report.append(f"  Avg Memory: {successful_data['memory_mb'].mean():.2f} MB")
                report.append(f"  Avg CPU: {successful_data['cpu_percent'].mean():.2f}%")
                
                if successful_data['throughput_ops_per_sec'].sum() > 0:
                    report.append(f"  Avg Throughput: {successful_data['throughput_ops_per_sec'].mean():.2f} ops/sec")
        
        # Performance thresholds check
        report.append("\n\nPERFORMANCE THRESHOLD ANALYSIS")
        report.append("-" * 40)
        
        thresholds = {
            "YouTubeSearchTool": 10000,  # 10 seconds
            "TextChunkerTool": 2000,     # 2 seconds
            "EmbedderTool": 5000,        # 5 seconds
            "FAISSStorageTool": 3000,    # 3 seconds
            "FAISSQueryTool": 1000,      # 1 second
            "SummarizerTool": 10000,     # 10 seconds
            "AnswerGeneratorTool": 15000 # 15 seconds
        }
        
        for tool, threshold in thresholds.items():
            tool_data = df[(df['tool_name'] == tool) & (df['success'])]
            if len(tool_data) > 0:
                avg_latency = tool_data['latency_ms'].mean()
                status = "✓ PASS" if avg_latency <= threshold else "✗ FAIL"
                report.append(f"{tool}: {avg_latency:.2f}ms (threshold: {threshold}ms) {status}")
        
        # Error analysis
        failed_tests = df[~df['success']]
        if len(failed_tests) > 0:
            report.append("\n\nERROR ANALYSIS")
            report.append("-" * 40)
            
            for _, row in failed_tests.iterrows():
                report.append(f"{row['tool_name']} - {row['operation']}: {row['error_message']}")
        
        # Recommendations
        report.append("\n\nRECOMMENDations")
        report.append("-" * 40)
        
        # Memory usage recommendations
        high_memory_tools = df[df['memory_mb'] > 100]
        if len(high_memory_tools) > 0:
            report.append("• High memory usage detected in:")
            for tool in high_memory_tools['tool_name'].unique():
                avg_memory = high_memory_tools[high_memory_tools['tool_name'] == tool]['memory_mb'].mean()
                report.append(f"  - {tool}: {avg_memory:.2f} MB average")
        
        # Latency recommendations
        slow_operations = df[df['latency_ms'] > 5000]
        if len(slow_operations) > 0:
            report.append("• Slow operations detected:")
            for _, row in slow_operations.iterrows():
                report.append(f"  - {row['tool_name']}.{row['operation']}: {row['latency_ms']:.2f} ms")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # Save detailed results as JSON
        json_file = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_file, 'w') as f:
            json.dump([
                {
                    "tool_name": r.tool_name,
                    "operation": r.operation,
                    "latency_ms": r.latency_ms,
                    "memory_mb": r.memory_mb,
                    "cpu_percent": r.cpu_percent,
                    "success": r.success,
                    "error_message": r.error_message,
                    "throughput_ops_per_sec": r.throughput_ops_per_sec,
                    "accuracy_score": r.accuracy_score,
                    "additional_metrics": r.additional_metrics
                }
                for r in self.results
            ], f, indent=2)
        
        # Save report as text
        report_file = os.path.join(self.output_dir, f"{filename}_report.txt")
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
        
        # Save CSV for analysis
        csv_file = os.path.join(self.output_dir, f"{filename}.csv")
        data = []
        for result in self.results:
            row = {
                "tool_name": result.tool_name,
                "operation": result.operation,
                "latency_ms": result.latency_ms,
                "memory_mb": result.memory_mb,
                "cpu_percent": result.cpu_percent,
                "success": result.success,
                "throughput_ops_per_sec": result.throughput_ops_per_sec or 0
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")
        print(f"  CSV: {csv_file}")


def main():
    """Main function to run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CrewAI Tools")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--tools", nargs="+", help="Specific tools to benchmark",
                       choices=["youtube_search", "text_chunker", "embedder", 
                               "faiss_storage", "faiss_query", "summarizer", "answer_generator"])
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with reduced test cases")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)
    
    try:
        # Run benchmarks
        if args.tools:
            # Run specific tools
            results = {}
            for tool in args.tools:
                if hasattr(suite, f"benchmark_{tool}"):
                    func = getattr(suite, f"benchmark_{tool}")
                    results[tool] = func()
                    suite.results.extend(results[tool])
        else:
            # Run all benchmarks
            results = suite.run_all_benchmarks()
        
        # Generate and display report
        print("\n" + "="*60)
        print(suite.generate_report())
        
        # Save results
        suite.save_results()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()