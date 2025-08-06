#!/usr/bin/env python3
"""
Vector Indexer Agent for CrewAI Agentic System

This agent specializes in processing transcripts into searchable vector embeddings.
It handles text chunking, embedding generation, and FAISS index management for
semantic search capabilities.

Author: Assistant
Date: 2024
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# CrewAI imports
from crewai import Agent, LLM

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from chunker import TextChunkerTool
from embedder import EmbedderTool
from faiss_store import FAISSStorageTool
from faiss_query import FAISSQueryTool

# Import unified config loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.config_loader import get_config


class VectorIndexerAgent:
    """
    Vector Indexer Agent for transcript processing and semantic indexing.
    
    This agent is responsible for:
    - Chunking transcripts into optimal segments
    - Generating vector embeddings for text chunks
    - Storing embeddings in FAISS indices
    - Managing vector database operations
    - Enabling semantic search capabilities
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Vector Indexer Agent.
        
        Args:
            config_path: Path to the crew configuration file
        """
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from unified config.yaml.
        
        Args:
            config_path: Path to configuration file (ignored, using unified config)
            
        Returns:
            Configuration dictionary
        """
        try:
            config_loader = get_config()
            return {
                'llm': config_loader.get_llm_config(),
                'agents': config_loader.get_crewai_config().get('agents', {}),
                'text_processing': config_loader.get_section('text_processing', {}),
                'vector_store': config_loader.get_vector_store_config()
            }
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            # Fallback configuration
            return {
                'llm': {
                    'provider': 'lm_studio',
                    'base_url': 'http://localhost:1234/v1',
                    'api_key': None,
                    'model_name': 'lm_studio/qwen-3-14b',
                    'temperature': 0.7,
                    'max_tokens': 2048
                },
                'agents': {
                    'vector_indexer': {
                        'role': 'Vector Database Indexer',
                        'goal': 'Process and index text content for efficient retrieval',
                        'backstory': 'Specialized in text processing and vector database management',
                        'max_iter': 3,
                        'max_execution_time': 300,
                        'verbose': True,
                        'allow_delegation': False
                    }
                },
                'text_processing': {
                    'chunking': {
                        'chunk_size': 1000,
                        'chunk_overlap': 200,
                        'separator': '\n\n',
                        'preserve_sentences': True,
                        'min_chunk_size': 100
                    },
                    'embedding': {
                        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                        'device': 'auto',
                        'batch_size': 32,
                        'normalize_embeddings': True
                    }
                },
                'vector_store': {
                    'index_type': 'IndexFlatIP',
                    'dimension': 384,
                    'nlist': 100,
                    'nprobe': 10,
                    'index_dir': 'data/embeddings',
                    'metadata_file': 'data/embeddings/metadata.jsonl',
                    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
                }
            }
    
    def _setup_llm(self):
        """
        Setup the local LLM for the agent.
        Uses local GGUF model when available, falls back to HTTP API.
        
        Returns:
            Configured LLM instance (LocalGGUFLLM or LLM)
        """
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.llm_config import LLMConfig
        
        llm_config = LLMConfig()
        return llm_config.get_local_llm()
    
    def _setup_tools(self) -> list:
        """
        Setup tools for the agent.
        
        Returns:
            List of tools for the agent
        """
        return [
            TextChunkerTool(),
            EmbedderTool(),
            FAISSStorageTool(),
            FAISSQueryTool()
        ]
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI agent instance.
        
        Returns:
            Configured Agent instance
        """
        agent_config = self.config.get('agents', {}).get('vector_indexer', {})
        
        return Agent(
            role=agent_config.get(
                'role', 
                'Embedding Engineer'
            ),
            goal=agent_config.get(
                'goal',
                'Process transcripts into searchable vector embeddings for semantic retrieval'
            ),
            backstory=agent_config.get(
                'backstory',
                'You are a vector database specialist with expertise in semantic '
                'search and information retrieval. You create efficient and '
                'accurate search indices.'
            ),
            tools=self.tools,
            llm=self.llm,
            max_iter=agent_config.get('max_iter', 3),
            max_execution_time=agent_config.get('max_execution_time', 300),
            verbose=agent_config.get('verbose', True),
            allow_delegation=agent_config.get('allow_delegation', False)
        )
    
    def get_agent(self) -> Agent:
        """
        Get the configured agent instance.
        
        Returns:
            The CrewAI Agent instance
        """
        return self.agent
    
    def process_transcript(self, transcript_text: str, 
                          source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a transcript through the complete indexing pipeline.
        
        Args:
            transcript_text: The transcript text to process
            source_metadata: Optional metadata about the source
            
        Returns:
            Processing results dictionary
        """
        try:
            # Step 1: Chunk the transcript
            chunker_tool = self.tools[0]  # TextChunkerTool
            chunking_result = chunker_tool._run(
                text=transcript_text,
                chunk_size=self.config.get('text_processing', {}).get('chunking', {}).get('chunk_size', 1000),
                chunk_overlap=self.config.get('text_processing', {}).get('chunking', {}).get('chunk_overlap', 200),
                preserve_sentences=True
            )
            
            if not chunking_result.get('success', False):
                return {
                    'success': False,
                    'error': f"Chunking failed: {chunking_result.get('error', 'Unknown error')}"
                }
            
            chunks = chunking_result.get('chunks', [])
            if not chunks:
                return {
                    'success': False,
                    'error': 'No chunks generated from transcript'
                }
            
            # Step 2: Generate embeddings
            embedder_tool = self.tools[1]  # EmbedderTool
            embedding_result = embedder_tool._run(
                texts=chunks,
                model_name=self.config.get('text_processing', {}).get('embedding', {}).get('model_name', 'all-MiniLM-L6-v2'),
                batch_size=self.config.get('text_processing', {}).get('embedding', {}).get('batch_size', 32)
            )
            
            if not embedding_result.get('success', False):
                return {
                    'success': False,
                    'error': f"Embedding generation failed: {embedding_result.get('error', 'Unknown error')}"
                }
            
            embeddings = embedding_result.get('embeddings', [])
            if not embeddings:
                return {
                    'success': False,
                    'error': 'No embeddings generated'
                }
            
            # Step 3: Store in FAISS index
            storage_tool = self.tools[2]  # FAISSStorageTool
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'chunk_id': i,
                    'text': chunk,
                    'chunk_length': len(chunk)
                }
                if source_metadata:
                    metadata.update(source_metadata)
                chunk_metadata.append(metadata)
            
            storage_result = storage_tool._run(
                vectors=embeddings,
                metadata_list=chunk_metadata,
                index_name="islamic_content"
            )
            
            if not storage_result.get('success', False):
                return {
                    'success': False,
                    'error': f"FAISS storage failed: {storage_result.get('error', 'Unknown error')}"
                }
            
            return {
                'success': True,
                'chunks_processed': len(chunks),
                'embeddings_generated': len(embeddings),
                'index_name': 'islamic_content',
                'storage_result': storage_result,
                'chunking_stats': chunking_result.get('stats', {}),
                'embedding_stats': embedding_result.get('stats', {})
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Vector indexing pipeline failed: {str(e)}"
            }
    
    def search_similar(self, query: str, k: int = 5, 
                      index_name: str = "islamic_content") -> Dict[str, Any]:
        """
        Search for similar content in the vector index.
        
        Args:
            query: Search query text
            k: Number of results to return
            index_name: Name of the FAISS index to search
            
        Returns:
            Search results dictionary
        """
        try:
            query_tool = self.tools[3]  # FAISSQueryTool
            
            search_result = query_tool._run(
                query=query,
                index_name=index_name,
                k=k
            )
            
            return search_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Vector search failed: {str(e)}"
            }
    
    def batch_process_transcripts(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple transcripts in batch.
        
        Args:
            transcripts: List of transcript dictionaries with 'text' and optional 'metadata'
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, transcript_data in enumerate(transcripts, 1):
            print(f"Processing transcript {i}/{len(transcripts)}")
            
            text = transcript_data.get('text', '')
            metadata = transcript_data.get('metadata', {})
            
            if not text:
                result = {
                    'success': False,
                    'error': 'No text provided in transcript data'
                }
            else:
                result = self.process_transcript(text, metadata)
            
            results.append(result)
            
            if not result.get('success', False):
                print(f"❌ Failed to process transcript {i}: {result.get('error', 'Unknown error')}")
            else:
                print(f"✅ Successfully processed transcript {i} ({result.get('chunks_processed', 0)} chunks)")
        
        return results


def create_vector_indexer_agent(config_path: Optional[str] = None) -> Agent:
    """
    Factory function to create a Vector Indexer Agent.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured CrewAI Agent instance
    """
    indexer = VectorIndexerAgent(config_path)
    return indexer.get_agent()


if __name__ == "__main__":
    # Demo usage
    print("🔍 Creating Vector Indexer Agent...")
    
    try:
        agent = create_vector_indexer_agent()
        print(f"✅ Agent created successfully!")
        print(f"Role: {agent.role}")
        print(f"Goal: {agent.goal}")
        print(f"Tools: {[tool.__class__.__name__ for tool in agent.tools]}")
        
        # Test indexing workflow
        indexer = VectorIndexerAgent()
        print("\n🔍 Vector Indexer Agent ready for transcript processing")
        print("Use process_transcript(text, metadata) to index content")
        print("Use search_similar(query, k) to search indexed content")
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()