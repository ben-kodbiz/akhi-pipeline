#!/usr/bin/env python3
"""
Content QA Agent for CrewAI Agentic System

This agent specializes in answering questions and generating summaries based on
processed Islamic content. It uses RAG (Retrieval-Augmented Generation) to provide
accurate, contextual responses with proper citations.

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
from summarizer import SummarizerTool
from answer_generator import AnswerGeneratorTool
from faiss_query import FAISSQueryTool

# Import unified config loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.config_loader import get_config


class ContentQAAgent:
    """
    Content QA Agent for Islamic educational content analysis and Q&A.
    
    This agent is responsible for:
    - Answering questions based on indexed Islamic content
    - Generating summaries of transcripts and content
    - Providing contextual responses with citations
    - Handling Islamic terminology and concepts appropriately
    - Supporting multiple query types and formats
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Content QA Agent.
        
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
                'retrieval': config_loader.get_section('retrieval', {}),
                'generation': config_loader.get_section('generation', {}),
                'faiss': config_loader.get_section('faiss', {})
            }
        except Exception as e:
            print(f"Warning: Could not load unified config: {e}")
            # Fallback configuration
            return {
                'llm': {
                    'provider': 'lm_studio',
                    'base_url': 'http://localhost:1234/v1',
                    'api_key': None,
                    'model_name': 'local-model',
                    'temperature': 0.7,
                    'max_tokens': 2048
                },
                'agents': {
                    'content_qa': {
                        'role': 'Islamic Content Q&A Specialist',
                        'goal': 'Answer questions and generate summaries based on Islamic educational content with accuracy and proper citations',
                        'backstory': 'You are an Islamic studies expert with deep knowledge of Quran, Hadith, and Islamic scholarship. You provide accurate, well-cited responses to questions about Islamic content.',
                        'max_iter': 5,
                        'max_execution_time': 300,
                        'verbose': True,
                        'allow_delegation': False
                    }
                },
                'retrieval': {
                    'top_k': 5,
                    'similarity_threshold': 0.7,
                    'rerank': True
                },
                'generation': {
                    'max_length': 512,
                    'temperature': 0.7,
                    'context_window': 4096
                },
                'faiss': {
                    'index_type': 'IndexFlatIP',
                    'dimension': 384,
                    'nlist': 100,
                    'nprobe': 10,
                    'index_dir': 'data/embeddings',
                    'metadata_file': 'data/embeddings/metadata.jsonl'
                },
                'vector_store': {
                    'index_dir': 'data/embeddings',
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
            SummarizerTool(),
            AnswerGeneratorTool(),
            FAISSQueryTool()
        ]
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI agent instance.
        
        Returns:
            Configured Agent instance
        """
        agent_config = self.config.get('agents', {}).get('content_qa', {})
        
        return Agent(
            role=agent_config.get(
                'role', 
                'Islamic Content Q&A Specialist'
            ),
            goal=agent_config.get(
                'goal',
                'Answer questions and generate summaries based on Islamic educational '
                'content with accuracy and proper citations'
            ),
            backstory=agent_config.get(
                'backstory',
                'You are an Islamic studies expert with deep knowledge of Quran, '
                'Hadith, and Islamic scholarship. You provide accurate, well-cited '
                'responses to questions about Islamic content.'
            ),
            tools=self.tools,
            llm=self.llm,
            max_iter=agent_config.get('max_iter', 5),
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
    
    def answer_question(self, question: str, 
                       index_name: str = "islamic_content",
                       max_chunks: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG-based approach.
        
        Args:
            question: The question to answer
            index_name: Name of the FAISS index to search
            max_chunks: Maximum number of context chunks to retrieve
            
        Returns:
            Answer results dictionary
        """
        try:
            # Step 1: Retrieve relevant context
            query_tool = self.tools[2]  # FAISSQueryTool
            
            # Get configuration values
            retrieval_config = self.config.get('retrieval', {})
            vector_config = self.config.get('vector_store', {})
            index_dir = vector_config.get('index_dir', 'data/embeddings')
            
            search_result = query_tool._run(
                query=question,
                index_name=index_name,
                k=max_chunks,
                index_dir=index_dir
            )
            
            if not search_result.success:
                return {
                    'success': False,
                    'error': f"Context retrieval failed: {search_result.message}"
                }
            
            # Extract context chunks
            results = search_result.results
            context_chunks = []
            for result in results:
                metadata = result.get('metadata', {})
                text = metadata.get('text', '')
                if text:
                    context_chunks.append({
                        'text': text,
                        'score': result.get('score', 0.0),
                        'metadata': metadata
                    })
            
            if not context_chunks:
                return {
                    'success': False,
                    'error': 'No relevant context found for the question'
                }
            
            # Step 2: Generate answer using retrieved context
            answer_tool = self.tools[1]  # AnswerGeneratorTool
            answer_result = answer_tool._run(
                question=question,
                context_chunks=context_chunks,
                max_length=500,
                include_citations=True
            )
            
            if not answer_result.success:
                return {
                    'success': False,
                    'error': f"Answer generation failed: {answer_result.message}"
                }
            
            return {
                'success': True,
                'question': question,
                'answer': answer_result.answer,
                'citations': answer_result.citations,
                'confidence': answer_result.confidence_score,
                'context_used': len(context_chunks),
                'search_results': len(results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Question answering failed: {str(e)}"
            }
    
    def summarize_content(self, text: str, 
                         summary_type: str = "abstractive",
                         max_length: int = 300) -> Dict[str, Any]:
        """
        Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            summary_type: Type of summary (extractive, abstractive, hybrid)
            max_length: Maximum length of summary
            
        Returns:
            Summary results dictionary
        """
        try:
            summarizer_tool = self.tools[0]  # SummarizerTool
            
            summary_result_str = summarizer_tool._run(
                text=text,
                strategy=summary_type,
                max_words=max_length,
                focus_keywords=["Islamic", "Quran", "Hadith", "Prophet", "Allah"]
            )
            
            # Parse JSON string to dictionary
            import json
            summary_result = json.loads(summary_result_str)
            
            return summary_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Summarization failed: {str(e)}"
            }
    
    def assess_content(self, content: str, content_type: str = "transcript") -> Dict[str, Any]:
        """
        Assess the quality and relevance of Islamic content.
        
        Args:
            content: Content text to assess
            content_type: Type of content (transcript, summary, etc.)
            
        Returns:
            Assessment results dictionary with quality metrics
        """
        try:
            # Basic content validation
            if not content or len(content.strip()) < 50:
                return {
                    'success': False,
                    'quality_score': 0.0,
                    'relevance_score': 0.0,
                    'issues': ['Content too short or empty'],
                    'recommendations': ['Provide more substantial content']
                }
            
            # Islamic content keywords for relevance scoring
            islamic_keywords = [
                'allah', 'quran', 'hadith', 'prophet', 'muhammad', 'islam', 'islamic',
                'muslim', 'prayer', 'salah', 'dua', 'sunnah', 'ummah', 'jihad',
                'ramadan', 'hajj', 'zakat', 'shahada', 'tawhid', 'iman', 'taqwa'
            ]
            
            content_lower = content.lower()
            keyword_matches = sum(1 for keyword in islamic_keywords if keyword in content_lower)
            relevance_score = min(keyword_matches / 5.0, 1.0)  # Normalize to 0-1
            
            # Quality assessment based on content characteristics
            word_count = len(content.split())
            sentence_count = len([s for s in content.split('.') if s.strip()])
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Quality scoring factors
            length_score = min(word_count / 500.0, 1.0)  # Prefer longer content
            structure_score = 1.0 if 10 <= avg_sentence_length <= 30 else 0.7
            
            quality_score = (length_score * 0.4 + structure_score * 0.3 + relevance_score * 0.3)
            
            # Identify potential issues
            issues = []
            recommendations = []
            
            if word_count < 100:
                issues.append('Content is quite short')
                recommendations.append('Consider providing more detailed content')
            
            if relevance_score < 0.3:
                issues.append('Low Islamic content relevance')
                recommendations.append('Ensure content focuses on Islamic topics')
            
            if avg_sentence_length > 40:
                issues.append('Sentences may be too long')
                recommendations.append('Consider breaking down complex sentences')
            
            return {
                'success': True,
                'quality_score': round(quality_score, 3),
                'relevance_score': round(relevance_score, 3),
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': round(avg_sentence_length, 1),
                'keyword_matches': keyword_matches,
                'issues': issues,
                'recommendations': recommendations,
                'assessment': 'high' if quality_score >= 0.7 else 'medium' if quality_score >= 0.4 else 'low'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Content assessment failed: {str(e)}",
                'quality_score': 0.0,
                'relevance_score': 0.0
            }
    
    def batch_qa(self, questions: List[str], 
                index_name: str = "islamic_content") -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            index_name: Name of the FAISS index to search
            
        Returns:
            List of answer results
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"Answering question {i}/{len(questions)}: {question[:50]}...")
            
            result = self.answer_question(
                question=question,
                index_name=index_name
            )
            
            results.append(result)
            
            if not result.get('success', False):
                print(f"❌ Failed to answer question {i}: {result.get('error', 'Unknown error')}")
            else:
                confidence = result.get('confidence', 0.0)
                print(f"✅ Answered question {i} (confidence: {confidence:.2f})")
        
        return results
    
    def interactive_qa(self, index_name: str = "islamic_content"):
        """
        Start an interactive Q&A session.
        
        Args:
            index_name: Name of the FAISS index to search
        """
        print("🤖 Islamic Content Q&A Assistant")
        print("Ask questions about Islamic content. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("🔍 Searching for relevant content...")
                result = self.answer_question(question, index_name)
                
                if result.get('success', False):
                    print(f"\n📝 Answer: {result.get('answer', '')}")
                    print(f"🎯 Confidence: {result.get('confidence', 0.0):.2f}")
                    
                    citations = result.get('citations', [])
                    if citations:
                        print("\n📚 Citations:")
                        for i, citation in enumerate(citations, 1):
                            print(f"  [{i}] {citation}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def create_content_qa_agent(config_path: Optional[str] = None) -> Agent:
    """
    Factory function to create a Content QA Agent.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured CrewAI Agent instance
    """
    qa_agent = ContentQAAgent(config_path)
    return qa_agent.get_agent()


if __name__ == "__main__":
    # Demo usage
    print("🤖 Creating Content QA Agent...")
    
    try:
        agent = create_content_qa_agent()
        print(f"✅ Agent created successfully!")
        print(f"Role: {agent.role}")
        print(f"Goal: {agent.goal}")
        print(f"Tools: {[tool.__class__.__name__ for tool in agent.tools]}")
        
        # Test QA workflow
        qa_agent = ContentQAAgent()
        print("\n🤖 Content QA Agent ready for questions and summarization")
        print("Use answer_question(question) to get answers")
        print("Use summarize_content(text) to generate summaries")
        print("Use interactive_qa() for interactive session")
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()