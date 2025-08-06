#!/usr/bin/env python3
"""
Demo Script for Islamic RAG Pipeline
Interactive demonstration of the RAG system
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

class RAGDemo:
    """Interactive demo for the RAG pipeline"""
    
    def __init__(self, config_path: str = None):
        """Initialize demo"""
        if config_path is None:
            self.config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self.pipeline = None
        self.session_id = "demo_session"
    
    def setup_pipeline(self):
        """Setup the RAG pipeline"""
        print("🚀 Initializing Islamic RAG Pipeline...")
        
        try:
            from rag_pipeline import IslamicRAGPipeline
            
            # Initialize pipeline
            self.pipeline = IslamicRAGPipeline(str(self.config_path))
            
            print("📚 Loading models and building index...")
            self.pipeline.setup()
            
            print("✅ Pipeline ready!\n")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup pipeline: {str(e)}")
            print("\n💡 Make sure to run setup first:")
            print("   python scripts/setup.py")
            return False
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions for demonstration"""
        return [
            "What are the five pillars of Islam?",
            "Tell me about the importance of prayer in Islam",
            "What is the significance of Ramadan?",
            "How should Muslims treat their parents?",
            "What does the Quran say about charity?",
            "Explain the concept of Jihad in Islam",
            "What are the qualities of a good Muslim?",
            "Tell me about the Prophet Muhammad's teachings on kindness",
            "What is the Islamic perspective on knowledge and learning?",
            "How does Islam promote social justice?"
        ]
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format the response for display"""
        output = []
        
        # Answer
        output.append(f"🤖 **Answer:**")
        output.append(f"{response['answer']}\n")
        
        # Sources
        if response.get('sources'):
            output.append(f"📖 **Sources ({len(response['sources'])}):**")
            for i, source in enumerate(response['sources'][:3], 1):  # Show top 3
                content = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                score = source.get('score', 0)
                metadata = source.get('metadata', {})
                
                source_info = f"{i}. {content}"
                if metadata:
                    source_details = []
                    if 'source' in metadata:
                        source_details.append(f"Source: {metadata['source']}")
                    if 'chapter' in metadata:
                        source_details.append(f"Chapter: {metadata['chapter']}")
                    if 'verse' in metadata:
                        source_details.append(f"Verse: {metadata['verse']}")
                    
                    if source_details:
                        source_info += f" ({', '.join(source_details)})"
                
                source_info += f" [Score: {score:.2f}]"
                output.append(f"   {source_info}")
            output.append("")
        
        # Metadata
        if response.get('response_time'):
            output.append(f"⏱️  Response time: {response['response_time']:.2f}s")
        
        if response.get('model_used'):
            output.append(f"🧠 Model: {response['model_used']}")
        
        return "\n".join(output)
    
    def interactive_mode(self):
        """Run interactive demo mode"""
        print("🎯 Interactive Mode")
        print("Type your questions about Islam, or 'quit' to exit\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🔍 Searching knowledge base...")
                start_time = time.time()
                
                response = self.pipeline.query(
                    question,
                    session_id=self.session_id,
                    max_tokens=512
                )
                
                print(f"\n{self.format_response(response)}")
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try again.\n")
    
    def sample_questions_mode(self):
        """Run demo with sample questions"""
        print("📝 Sample Questions Demo")
        
        questions = self.get_sample_questions()
        
        print(f"\nRunning {len(questions)} sample questions...\n")
        
        for i, question in enumerate(questions, 1):
            print(f"{'='*80}")
            print(f"Question {i}/{len(questions)}: {question}")
            print(f"{'='*80}")
            
            try:
                start_time = time.time()
                
                response = self.pipeline.query(
                    question,
                    session_id=f"sample_{i}",
                    max_tokens=512
                )
                
                print(f"\n{self.format_response(response)}")
                
                # Pause between questions
                if i < len(questions):
                    input("\nPress Enter for next question...")
                    print()
                
            except Exception as e:
                print(f"❌ Error processing question: {str(e)}")
                continue
        
        print("\n🎉 Sample questions demo completed!")
    
    def benchmark_mode(self):
        """Run benchmark mode"""
        print("⚡ Benchmark Mode")
        
        questions = self.get_sample_questions()[:5]  # Use first 5 for benchmark
        
        print(f"\nBenchmarking with {len(questions)} questions...\n")
        
        total_time = 0
        successful_queries = 0
        
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}...", end=" ")
            
            try:
                start_time = time.time()
                
                response = self.pipeline.query(
                    question,
                    session_id=f"bench_{i}",
                    max_tokens=256  # Shorter for benchmark
                )
                
                elapsed = time.time() - start_time
                total_time += elapsed
                successful_queries += 1
                
                print(f"✅ {elapsed:.2f}s")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        # Results
        print(f"\n📊 Benchmark Results:")
        print(f"   Successful queries: {successful_queries}/{len(questions)}")
        print(f"   Total time: {total_time:.2f}s")
        if successful_queries > 0:
            print(f"   Average time per query: {total_time/successful_queries:.2f}s")
            print(f"   Queries per minute: {60*successful_queries/total_time:.1f}")
    
    def conversation_mode(self):
        """Run conversation mode"""
        print("💬 Conversation Mode")
        print("Have a conversation about Islam. Type 'quit' to exit\n")
        
        conversation_session = f"conversation_{int(time.time())}"
        
        while True:
            try:
                message = input("💬 You: ").strip()
                
                if message.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not message:
                    continue
                
                print("\n🤖 Assistant: ", end="")
                
                response = self.pipeline.query(
                    message,
                    session_id=conversation_session,
                    max_tokens=512
                )
                
                print(response['answer'])
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try again.\n")
    
    def run_demo(self, mode: str = "interactive"):
        """Run the demo"""
        print("🕌 Islamic RAG Pipeline Demo")
        print("="*50)
        
        # Setup pipeline
        if not self.setup_pipeline():
            return False
        
        # Run selected mode
        if mode == "interactive":
            self.interactive_mode()
        elif mode == "samples":
            self.sample_questions_mode()
        elif mode == "benchmark":
            self.benchmark_mode()
        elif mode == "conversation":
            self.conversation_mode()
        else:
            print(f"Unknown mode: {mode}")
            return False
        
        return True

def print_usage():
    """Print usage information"""
    print("🕌 Islamic RAG Pipeline Demo")
    print("="*50)
    print()
    print("Usage: python scripts/demo.py [mode]")
    print()
    print("Available modes:")
    print("  interactive  - Interactive Q&A mode (default)")
    print("  samples      - Demo with predefined sample questions")
    print("  benchmark    - Performance benchmark mode")
    print("  conversation - Conversational mode with memory")
    print()
    print("Examples:")
    print("  python scripts/demo.py")
    print("  python scripts/demo.py interactive")
    print("  python scripts/demo.py samples")
    print("  python scripts/demo.py benchmark")
    print()
    print("Prerequisites:")
    print("  1. Run: python scripts/setup.py")
    print("  2. Ensure QLoRA model is available")
    print("  3. Prepare data: python scripts/data_preparation.py")
    print()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Islamic RAG Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "mode",
        nargs="?",
        default="interactive",
        choices=["interactive", "samples", "benchmark", "conversation"],
        help="Demo mode to run"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--help-usage",
        action="store_true",
        help="Show detailed usage information"
    )
    
    args = parser.parse_args()
    
    if args.help_usage:
        print_usage()
        return
    
    # Initialize and run demo
    demo = RAGDemo(args.config)
    
    try:
        success = demo.run_demo(args.mode)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()