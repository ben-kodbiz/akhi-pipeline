#!/usr/bin/env python3
"""
Akhi CrewAI Pipeline - Main Application

Main entry point for the Islamic Content Processing Pipeline.
Provides CLI interface and interactive mode for:
- Video research and processing
- Question answering on indexed content
- Content summarization
- System status monitoring

Author: Assistant
Date: December 2024
Phase: 6 - Main Application & CLI
"""

import os
import sys
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project paths
sys.path.append(os.path.dirname(__file__))

from crew import AkhiPipelineCrew
from utils.llm_config import LLMConfig
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


class AkhiPipelineApp:
    """
    Main application class for the Akhi CrewAI Pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/crew_config.yaml"
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.llm_config = LLMConfig(self.config_path)
        self.crew = None
        self.tools = self._initialize_tools()
        
        # Session management
        self.session_history = []
        self.current_session_id = None
    
    def setup_logging(self):
        """
        Setup logging configuration.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('akhi_pipeline.log')
            ]
        )
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """
        Initialize all pipeline tools.
        
        Returns:
            dict: Dictionary of initialized tools
        """
        try:
            tools = {
                'search': YouTubeSearchTool(config_path=self.config_path),
                'downloader': YouTubeDownloaderTool(config_path=self.config_path),
                'transcriber': TranscriptionTool(config_path=self.config_path),
                'chunker': TextChunkerTool(config_path=self.config_path),
                'embedder': EmbedderTool(config_path=self.config_path),
                'faiss_store': FAISSStorageTool(config_path=self.config_path),
                'faiss_query': FAISSQueryTool(config_path=self.config_path),
                'summarizer': SummarizerTool(config_path=self.config_path),
                'answer_generator': AnswerGeneratorTool(config_path=self.config_path)
            }
            self.logger.info(f"Initialized {len(tools)} tools successfully")
            return tools
        except Exception as e:
            self.logger.error(f"Failed to initialize tools: {e}")
            return {}
    
    def _initialize_crew(self) -> AkhiPipelineCrew:
        """
        Initialize the CrewAI pipeline.
        
        Returns:
            AkhiPipelineCrew: Initialized crew instance
        """
        if self.crew is None:
            try:
                self.crew = AkhiPipelineCrew(config_path=self.config_path)
                self.logger.info("CrewAI pipeline initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize crew: {e}")
                raise
        return self.crew
    
    def search_videos(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for videos on a given topic.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            dict: Search results
        """
        try:
            self.logger.info(f"Searching videos for: {query}")
            search_tool = self.tools.get('search')
            if not search_tool:
                raise ValueError("Search tool not available")
            
            result = search_tool._run(
                query=query,
                max_results=max_results
            )
            
            # YouTubeSearchTool._run() returns a formatted string, not a dict
            # Parse the result to extract video count for logging
            if "Found" in result and "videos" in result:
                # Extract number from "Found X videos" pattern
                import re
                match = re.search(r'Found (\d+) videos', result)
                video_count = int(match.group(1)) if match else 0
                self.logger.info(f"Found {video_count} videos")
            else:
                self.logger.info("Search completed")
            
            return {'result': result, 'videos': []}
            
        except Exception as e:
            self.logger.error(f"Video search failed: {e}")
            return {'error': str(e), 'videos': []}
    
    def process_pipeline(self, query: str, full_process: bool = True) -> Dict[str, Any]:
        """
        Execute the full pipeline for a given query.
        
        Args:
            query: Research query
            full_process: Whether to run full pipeline or just search
            
        Returns:
            dict: Pipeline execution results
        """
        try:
            self.logger.info(f"Starting pipeline for: {query}")
            
            # Initialize crew
            crew = self._initialize_crew()
            
            # Execute pipeline
            if full_process:
                result = crew.execute_full_pipeline(
                    search_query=query,
                    questions=["What are the main topics discussed?", "What are the key Islamic teachings?"],
                    index_name="default"
                )
            else:
                # Just search
                result = self.search_videos(query)
            
            self.logger.info("Pipeline execution completed")
            return {
                'status': 'success',
                'query': query,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def query_content(self, question: str, index_name: str = "default") -> Dict[str, Any]:
        """
        Ask questions about indexed content.
        
        Args:
            question: Question to ask
            index_name: Name of the FAISS index to query
            
        Returns:
            dict: Answer and supporting information
        """
        try:
            self.logger.info(f"Querying content: {question}")
            
            # Query FAISS index
            query_tool = self.tools.get('faiss_query')
            if not query_tool:
                raise ValueError("Query tool not available")
            
            search_result = query_tool._run(
                query=question,
                index_name=index_name,
                k=5
            )
            
            # Generate answer
            answer_tool = self.tools.get('answer_generator')
            if not answer_tool:
                raise ValueError("Answer generator not available")
            
            answer_result = answer_tool._run(
                question=question,
                context_chunks=search_result.get('results', []),
                max_context_chunks=3
            )
            
            self.logger.info("Content query completed")
            return {
                'status': 'success',
                'question': question,
                'answer': answer_result.get('answer', ''),
                'sources': search_result.get('results', []),
                'confidence': answer_result.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Content query failed: {e}")
            return {
                'status': 'error',
                'question': question,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def summarize_content(self, content_type: str = "recent", limit: int = 5) -> Dict[str, Any]:
        """
        Generate summaries of content.
        
        Args:
            content_type: Type of content to summarize
            limit: Number of items to include
            
        Returns:
            dict: Summary results
        """
        try:
            self.logger.info(f"Generating summary for: {content_type}")
            
            summarizer_tool = self.tools.get('summarizer')
            if not summarizer_tool:
                raise ValueError("Summarizer tool not available")
            
            # Get recent content (this would need to be implemented based on your data structure)
            # For now, return a placeholder
            result = {
                'status': 'success',
                'content_type': content_type,
                'summary': 'Summary functionality will be implemented based on available content.',
                'items_processed': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Summary generation completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {
                'status': 'error',
                'content_type': content_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Check system status and health.
        
        Returns:
            dict: System status information
        """
        try:
            status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'llm_config': 'unknown',
                    'tools': 'unknown',
                    'crew': 'unknown'
                },
                'statistics': {
                    'tools_loaded': len(self.tools),
                    'session_history_count': len(self.session_history)
                }
            }
            
            # Check LLM configuration
            try:
                if self.llm_config.validate_llm_connection():
                    status['components']['llm_config'] = 'healthy'
                else:
                    status['components']['llm_config'] = 'warning'
            except Exception:
                status['components']['llm_config'] = 'error'
            
            # Check tools
            if self.tools:
                status['components']['tools'] = 'healthy'
            else:
                status['components']['tools'] = 'error'
            
            # Check crew
            try:
                if self.crew or self._initialize_crew():
                    status['components']['crew'] = 'healthy'
            except Exception:
                status['components']['crew'] = 'error'
            
            # Determine overall status
            component_statuses = list(status['components'].values())
            if 'error' in component_statuses:
                status['status'] = 'degraded'
            elif 'warning' in component_statuses:
                status['status'] = 'warning'
            
            return status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def interactive_mode(self):
        """
        Start interactive chat interface.
        """
        print("\n🕌 Welcome to Akhi CrewAI Pipeline - Interactive Mode")
        print("=" * 60)
        print("Available commands:")
        print("  /search <query>     - Search for videos")
        print("  /process <query>    - Run full pipeline")
        print("  /query <question>   - Ask about indexed content")
        print("  /summarize          - Generate content summary")
        print("  /status             - Check system status")
        print("  /history            - Show session history")
        print("  /help               - Show this help")
        print("  /quit               - Exit interactive mode")
        print("=" * 60)
        
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while True:
            try:
                user_input = input("\n🤖 Akhi> ").strip()
                
                if not user_input:
                    continue
                
                # Add to session history
                self.session_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'input': user_input,
                    'session_id': self.current_session_id
                })
                
                if user_input.startswith('/quit'):
                    print("\n👋 Goodbye! May Allah bless your learning journey.")
                    break
                
                elif user_input.startswith('/help'):
                    print("\n📖 Available commands:")
                    print("  /search <query>     - Search for videos on Islamic topics")
                    print("  /process <query>    - Run full pipeline (search, download, transcribe, index)")
                    print("  /query <question>   - Ask questions about indexed content")
                    print("  /summarize          - Generate summary of recent content")
                    print("  /status             - Check system health and status")
                    print("  /history            - Show current session history")
                    print("  /quit               - Exit interactive mode")
                
                elif user_input.startswith('/search '):
                    query = user_input[8:].strip()
                    if query:
                        print(f"\n🔍 Searching for: {query}")
                        result = self.search_videos(query)
                        if result.get('error'):
                            print(f"❌ Error: {result['error']}")
                        else:
                            videos = result.get('videos', [])
                            print(f"✅ Found {len(videos)} videos:")
                            for i, video in enumerate(videos[:5], 1):
                                print(f"  {i}. {video.get('title', 'Unknown')}")
                                print(f"     Channel: {video.get('channel', 'Unknown')}")
                                print(f"     Duration: {video.get('duration', 'Unknown')}")
                    else:
                        print("❌ Please provide a search query")
                
                elif user_input.startswith('/process '):
                    query = user_input[9:].strip()
                    if query:
                        print(f"\n⚙️ Processing pipeline for: {query}")
                        print("This may take several minutes...")
                        result = self.process_pipeline(query)
                        if result.get('status') == 'success':
                            print("✅ Pipeline completed successfully")
                        else:
                            print(f"❌ Pipeline failed: {result.get('error')}")
                    else:
                        print("❌ Please provide a research query")
                
                elif user_input.startswith('/query '):
                    question = user_input[7:].strip()
                    if question:
                        print(f"\n❓ Querying: {question}")
                        result = self.query_content(question)
                        if result.get('status') == 'success':
                            print(f"\n💡 Answer: {result.get('answer')}")
                            print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
                            sources = result.get('sources', [])
                            if sources:
                                print(f"\n📚 Sources ({len(sources)}):")
                                for i, source in enumerate(sources[:3], 1):
                                    print(f"  {i}. {source.get('text', '')[:100]}...")
                        else:
                            print(f"❌ Query failed: {result.get('error')}")
                    else:
                        print("❌ Please provide a question")
                
                elif user_input.startswith('/summarize'):
                    print("\n📝 Generating content summary...")
                    result = self.summarize_content()
                    if result.get('status') == 'success':
                        print(f"✅ Summary: {result.get('summary')}")
                    else:
                        print(f"❌ Summary failed: {result.get('error')}")
                
                elif user_input.startswith('/status'):
                    print("\n🔍 Checking system status...")
                    status = self.get_system_status()
                    print(f"\n📊 System Status: {status.get('status', 'unknown').upper()}")
                    components = status.get('components', {})
                    for component, state in components.items():
                        emoji = {'healthy': '✅', 'warning': '⚠️', 'error': '❌', 'unknown': '❓'}.get(state, '❓')
                        print(f"  {emoji} {component}: {state}")
                    stats = status.get('statistics', {})
                    print(f"\n📈 Statistics:")
                    print(f"  Tools loaded: {stats.get('tools_loaded', 0)}")
                    print(f"  Session commands: {stats.get('session_history_count', 0)}")
                
                elif user_input.startswith('/history'):
                    print(f"\n📜 Session History (Session: {self.current_session_id}):")
                    session_commands = [h for h in self.session_history if h['session_id'] == self.current_session_id]
                    for i, entry in enumerate(session_commands[-10:], 1):  # Show last 10
                        timestamp = entry['timestamp'].split('T')[1][:8]  # Just time
                        print(f"  {i}. [{timestamp}] {entry['input']}")
                
                else:
                    # Treat as a direct question
                    print(f"\n❓ Interpreting as question: {user_input}")
                    result = self.query_content(user_input)
                    if result.get('status') == 'success':
                        print(f"\n💡 Answer: {result.get('answer')}")
                        print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
                    else:
                        print(f"❌ Could not answer: {result.get('error')}")
                        print("💡 Try using specific commands like /search, /process, or /query")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! May Allah bless your learning journey.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                self.logger.error(f"Interactive mode error: {e}")


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description="Akhi CrewAI Pipeline - Islamic Content Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --search "Islamic finance principles"
  %(prog)s --process "Quran recitation techniques" --max-videos 3
  %(prog)s --query "What is the importance of prayer in Islam?"
  %(prog)s --summarize
  %(prog)s --interactive
  %(prog)s --status

For interactive mode with real-time Q&A:
  %(prog)s --interactive
        """
    )
    
    # Command options
    parser.add_argument('--search', type=str, metavar='QUERY',
                       help='Research videos on specified topic')
    parser.add_argument('--process', type=str, metavar='QUERY',
                       help='Execute full pipeline for specified topic')
    parser.add_argument('--query', type=str, metavar='QUESTION',
                       help='Ask questions about indexed content')
    parser.add_argument('--summarize', action='store_true',
                       help='Generate summaries of recent content')
    parser.add_argument('--status', action='store_true',
                       help='Check system status and health')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive chat interface')
    
    # Configuration options
    parser.add_argument('--config', type=str, metavar='PATH',
                       help='Path to configuration file')
    parser.add_argument('--max-videos', type=int, default=5, metavar='N',
                       help='Maximum number of videos to process (default: 5)')
    parser.add_argument('--index-name', type=str, default='default', metavar='NAME',
                       help='FAISS index name for queries (default: default)')
    
    # Output options
    parser.add_argument('--output', type=str, metavar='FILE',
                       help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize application
        app = AkhiPipelineApp(config_path=args.config)
        
        result = None
        
        # Execute commands
        if args.search:
            print(f"🔍 Searching for videos: {args.search}")
            result = app.search_videos(args.search, max_results=args.max_videos)
            
        elif args.process:
            print(f"⚙️ Processing pipeline: {args.process}")
            print("This may take several minutes...")
            result = app.process_pipeline(args.process)
            
        elif args.query:
            print(f"❓ Querying content: {args.query}")
            result = app.query_content(args.query, index_name=args.index_name)
            
        elif args.summarize:
            print("📝 Generating content summary...")
            result = app.summarize_content()
            
        elif args.status:
            print("🔍 Checking system status...")
            result = app.get_system_status()
            
        elif args.interactive:
            app.interactive_mode()
            return
            
        else:
            # No command specified, show help
            parser.print_help()
            return
        
        # Display results
        if result:
            if args.output:
                # Save to file
                output_path = Path(args.output)
                output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                print(f"\n💾 Results saved to: {output_path}")
            else:
                # Display to console
                print("\n📋 Results:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. May Allah bless your learning journey.")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()