#!/usr/bin/env python3
"""
Evaluation Script for Islamic RAG Pipeline
Uses RAGAS to evaluate the performance of the RAG system.
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from datasets import Dataset

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import IslamicRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluates RAG pipeline performance using RAGAS"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the evaluator"""
        self.config_path = config_path
        self.rag_pipeline = None
        self.evaluation_results = {}
        
    def setup_pipeline(self) -> None:
        """Setup the RAG pipeline for evaluation"""
        logger.info("Setting up RAG pipeline for evaluation...")
        
        self.rag_pipeline = IslamicRAGPipeline(self.config_path)
        self.rag_pipeline.initialize_pipeline(rebuild_index=False)
        
        logger.info("RAG pipeline setup completed")
    
    def load_evaluation_dataset(self, dataset_path: str) -> Dict[str, List[str]]:
        """Load evaluation dataset from JSON file"""
        logger.info(f"Loading evaluation dataset from {dataset_path}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data['questions'])} evaluation questions")
        return data
    
    def generate_answers(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Generate answers for evaluation questions"""
        logger.info("Generating answers for evaluation questions...")
        
        results = []
        total_time = 0
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            start_time = time.time()
            
            try:
                # Get answer from RAG pipeline
                result = self.rag_pipeline.query(question)
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # Extract contexts from source documents
                contexts = []
                if result.get("source_documents"):
                    contexts = [doc["content"] for doc in result["source_documents"]]
                
                results.append({
                    "question": question,
                    "answer": result["answer"],
                    "contexts": contexts,
                    "processing_time": processing_time,
                    "source_count": len(contexts)
                })
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "contexts": [],
                    "processing_time": 0,
                    "source_count": 0
                })
        
        avg_time = total_time / len(questions) if questions else 0
        logger.info(f"Answer generation completed. Average time: {avg_time:.2f}s per question")
        
        return results
    
    def evaluate_with_ragas(self, eval_data: Dict[str, List[str]], generated_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate using RAGAS metrics"""
        logger.info("Starting RAGAS evaluation...")
        
        try:
            from ragas import evaluate
            from ragas.metrics import (
                context_relevancy,
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision
            )
            
            # Prepare dataset for RAGAS
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            
            for i, result in enumerate(generated_results):
                questions.append(result["question"])
                answers.append(result["answer"])
                contexts.append(result["contexts"])
                
                # Get corresponding ground truth
                if i < len(eval_data["ground_truth"]):
                    ground_truths.append(eval_data["ground_truth"][i])
                else:
                    ground_truths.append("")
            
            # Create RAGAS dataset
            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths
            })
            
            # Define metrics to evaluate
            metrics = [
                context_relevancy,
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision
            ]
            
            # Run evaluation
            logger.info("Running RAGAS evaluation...")
            result = evaluate(dataset, metrics=metrics)
            
            logger.info("RAGAS evaluation completed")
            return result
            
        except ImportError:
            logger.warning("RAGAS not available. Performing basic evaluation...")
            return self.basic_evaluation(eval_data, generated_results)
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            return self.basic_evaluation(eval_data, generated_results)
    
    def basic_evaluation(self, eval_data: Dict[str, List[str]], generated_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform basic evaluation without RAGAS"""
        logger.info("Performing basic evaluation...")
        
        total_questions = len(generated_results)
        successful_answers = 0
        total_processing_time = 0
        total_sources = 0
        
        keyword_matches = 0
        length_scores = []
        
        for i, result in enumerate(generated_results):
            # Check if answer was generated successfully
            if not result["answer"].startswith("Error:"):
                successful_answers += 1
            
            # Accumulate processing time
            total_processing_time += result["processing_time"]
            
            # Count sources
            total_sources += result["source_count"]
            
            # Basic keyword matching with ground truth
            if i < len(eval_data["ground_truth"]):
                ground_truth = eval_data["ground_truth"][i].lower()
                answer = result["answer"].lower()
                
                # Extract key terms from ground truth
                key_terms = []
                if "114" in ground_truth:
                    key_terms.append("114")
                if "five pillars" in ground_truth:
                    key_terms.extend(["five", "pillars", "shahada", "salah", "zakat", "sawm", "hajj"])
                if "abu bakr" in ground_truth:
                    key_terms.extend(["abu", "bakr"])
                if "shahada" in ground_truth:
                    key_terms.extend(["shahada", "declaration", "faith"])
                if "ramadan" in ground_truth:
                    key_terms.append("ramadan")
                
                # Check for keyword matches
                matches = sum(1 for term in key_terms if term in answer)
                if matches > 0:
                    keyword_matches += 1
            
            # Answer length score (prefer moderate length)
            answer_length = len(result["answer"].split())
            if 10 <= answer_length <= 100:
                length_scores.append(1.0)
            elif answer_length < 10:
                length_scores.append(0.5)
            else:
                length_scores.append(0.7)
        
        # Calculate metrics
        success_rate = successful_answers / total_questions if total_questions > 0 else 0
        avg_processing_time = total_processing_time / total_questions if total_questions > 0 else 0
        avg_sources = total_sources / total_questions if total_questions > 0 else 0
        keyword_accuracy = keyword_matches / total_questions if total_questions > 0 else 0
        avg_length_score = sum(length_scores) / len(length_scores) if length_scores else 0
        
        return {
            "success_rate": success_rate,
            "keyword_accuracy": keyword_accuracy,
            "avg_processing_time": avg_processing_time,
            "avg_sources_per_answer": avg_sources,
            "avg_length_score": avg_length_score,
            "total_questions": total_questions,
            "successful_answers": successful_answers,
            "evaluation_type": "basic"
        }
    
    def calculate_additional_metrics(self, generated_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate additional performance metrics"""
        logger.info("Calculating additional metrics...")
        
        processing_times = [r["processing_time"] for r in generated_results]
        source_counts = [r["source_count"] for r in generated_results]
        answer_lengths = [len(r["answer"].split()) for r in generated_results]
        
        metrics = {
            "performance": {
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "total_processing_time": sum(processing_times)
            },
            "retrieval": {
                "min_sources": min(source_counts) if source_counts else 0,
                "max_sources": max(source_counts) if source_counts else 0,
                "avg_sources": sum(source_counts) / len(source_counts) if source_counts else 0,
                "questions_with_sources": sum(1 for count in source_counts if count > 0)
            },
            "generation": {
                "min_answer_length": min(answer_lengths) if answer_lengths else 0,
                "max_answer_length": max(answer_lengths) if answer_lengths else 0,
                "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
                "empty_answers": sum(1 for length in answer_lengths if length == 0)
            }
        }
        
        return metrics
    
    def run_evaluation(self, dataset_path: str, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        logger.info("Starting RAG pipeline evaluation...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Setup pipeline
        self.setup_pipeline()
        
        # Load evaluation dataset
        eval_data = self.load_evaluation_dataset(dataset_path)
        
        # Generate answers
        generated_results = self.generate_answers(eval_data["questions"])
        
        # Evaluate with RAGAS
        ragas_results = self.evaluate_with_ragas(eval_data, generated_results)
        
        # Calculate additional metrics
        additional_metrics = self.calculate_additional_metrics(generated_results)
        
        # Combine all results
        final_results = {
            "evaluation_summary": {
                "total_questions": len(eval_data["questions"]),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_path": self.config_path
            },
            "ragas_metrics": ragas_results,
            "performance_metrics": additional_metrics,
            "detailed_results": generated_results
        }
        
        # Save results
        results_file = output_path / f"evaluation_results_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary report
        self.generate_summary_report(final_results, output_path)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        
        return final_results
    
    def generate_summary_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate a human-readable summary report"""
        logger.info("Generating summary report...")
        
        report_file = output_path / "evaluation_summary.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Islamic RAG Pipeline Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            summary = results["evaluation_summary"]
            f.write(f"Evaluation Date: {summary['evaluation_date']}\n")
            f.write(f"Total Questions: {summary['total_questions']}\n")
            f.write(f"Configuration: {summary['config_path']}\n\n")
            
            # RAGAS metrics
            if "ragas_metrics" in results and results["ragas_metrics"]:
                f.write("RAGAS Metrics:\n")
                f.write("-" * 20 + "\n")
                ragas = results["ragas_metrics"]
                
                if isinstance(ragas, dict):
                    for metric, value in ragas.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{metric}: {value:.4f}\n")
                        else:
                            f.write(f"{metric}: {value}\n")
                else:
                    f.write(f"Overall Score: {ragas}\n")
                f.write("\n")
            
            # Performance metrics
            if "performance_metrics" in results:
                perf = results["performance_metrics"]
                
                f.write("Performance Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Processing Time: {perf['performance']['avg_processing_time']:.2f}s\n")
                f.write(f"Average Sources per Answer: {perf['retrieval']['avg_sources']:.1f}\n")
                f.write(f"Average Answer Length: {perf['generation']['avg_answer_length']:.1f} words\n")
                f.write(f"Questions with Sources: {perf['retrieval']['questions_with_sources']}\n\n")
            
            # Sample results
            f.write("Sample Results:\n")
            f.write("-" * 20 + "\n")
            
            detailed = results.get("detailed_results", [])
            for i, result in enumerate(detailed[:3]):
                f.write(f"\nQuestion {i+1}: {result['question']}\n")
                f.write(f"Answer: {result['answer'][:200]}...\n")
                f.write(f"Sources: {result['source_count']}\n")
                f.write(f"Processing Time: {result['processing_time']:.2f}s\n")
        
        logger.info(f"Summary report saved to {report_file}")

def main():
    """Main function for running evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Islamic RAG Pipeline")
    parser.add_argument("--config", default="../config.yaml", help="Configuration file path")
    parser.add_argument("--dataset", default="../data/evaluation_dataset.json", help="Evaluation dataset path")
    parser.add_argument("--output", default="../evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = RAGEvaluator(args.config)
    results = evaluator.run_evaluation(args.dataset, args.output)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 30)
    
    if "ragas_metrics" in results and results["ragas_metrics"]:
        print(f"RAGAS Score: {results['ragas_metrics']}")
    
    if "performance_metrics" in results:
        perf = results["performance_metrics"]
        print(f"Average Processing Time: {perf['performance']['avg_processing_time']:.2f}s")
        print(f"Average Sources: {perf['retrieval']['avg_sources']:.1f}")
        print(f"Average Answer Length: {perf['generation']['avg_answer_length']:.1f} words")
    
    print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()