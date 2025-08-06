#!/usr/bin/env python3
"""
RAG Pipeline for QLoRA-Fine-Tuned Qwen3-1.7B
Integrates retrieval-augmented generation with Islamic knowledge base.
"""

import os
import yaml
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Import our enhanced document loader
from document_loader import EnhancedDocumentLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IslamicRAGPipeline:
    """RAG Pipeline for Islamic Question Answering"""
    
    def __init__(self, config_path_or_dict = "config.yaml"):
        """Initialize the RAG pipeline with configuration"""
        if isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            self.config = self._load_config(config_path_or_dict)
        
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store = None
        
        # Document cache management
        self.cache_dir = Path(self.config['vector_store']['persist_directory']).parent / "document_cache"
        self.cache_file = self.cache_dir / "processed_files.json"
        self.cache_dir.mkdir(exist_ok=True)
        self.retriever = None
        self.rag_chain = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for cache validation"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load processed files cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self, cache: Dict[str, Dict[str, Any]]) -> None:
        """Save document cache to file"""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Document cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _should_process_file(self, file_path: Path, cache: Dict[str, Dict[str, Any]]) -> bool:
        """Check if file needs processing based on cache"""
        file_key = str(file_path.absolute())
        
        if file_key not in cache:
            return True
        
        try:
            current_hash = self._get_file_hash(file_path)
            cached_hash = cache[file_key].get('hash')
            return current_hash != cached_hash
        except Exception as e:
            logger.warning(f"Error checking file hash for {file_path}: {e}")
            return True
    
    def setup_model(self) -> None:
        """Load QLoRA-fine-tuned Qwen3-1.7B model"""
        logger.info("Loading QLoRA-fine-tuned Qwen3-1.7B model...")
        
        model_path = self.config['model']['path']
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model loaded successfully")
    
    def setup_embeddings(self) -> None:
        """Initialize embedding model"""
        logger.info("Setting up embeddings...")
        
        model_name = self.config['embeddings']['model_name']
        device = self.config['embeddings']['device']
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        
        logger.info(f"Embeddings initialized with {model_name}")
    
    def load_and_process_documents(self, data_dir: str = "data") -> List[Document]:
        """Load and process Islamic knowledge documents using enhanced loader with caching"""
        logger.info("Loading and processing documents with caching...")
        
        # Load cache
        cache = self._load_cache()
        
        # Initialize enhanced document loader
        loader = EnhancedDocumentLoader(
            chunk_size=self.config['retrieval']['chunk_size'],
            chunk_overlap=self.config['retrieval']['chunk_overlap']
        )
        
        # Check which files need processing
        sources = self.config['data']['sources']
        files_to_process = []
        cached_documents = []
        
        for source in sources:
            source_path = Path(source)
            if source_path.is_file():
                if self._should_process_file(source_path, cache):
                    files_to_process.append(str(source_path))
                    logger.info(f"File needs processing: {source_path}")
                else:
                    logger.info(f"Using cached version: {source_path}")
                    # Load cached documents if available
                    file_key = str(source_path.absolute())
                    if 'documents' in cache[file_key]:
                        for doc_data in cache[file_key]['documents']:
                            doc = Document(
                                page_content=doc_data['page_content'],
                                metadata=doc_data['metadata']
                            )
                            cached_documents.append(doc)
            elif source_path.is_dir():
                # For directories, check each file
                for file_path in source_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.docx', '.html', '.json']:
                        if self._should_process_file(file_path, cache):
                            files_to_process.append(str(file_path))
                            logger.info(f"File needs processing: {file_path}")
                        else:
                            logger.info(f"Using cached version: {file_path}")
                            file_key = str(file_path.absolute())
                            if file_key in cache and 'documents' in cache[file_key]:
                                for doc_data in cache[file_key]['documents']:
                                    doc = Document(
                                        page_content=doc_data['page_content'],
                                        metadata=doc_data['metadata']
                                    )
                                    cached_documents.append(doc)
        
        # Process only new/modified files
        new_documents = []
        if files_to_process:
            logger.info(f"Processing {len(files_to_process)} new/modified files...")
            new_documents = loader.load_documents_from_paths(files_to_process)
            
            # Update cache with new documents
            for source in files_to_process:
                try:
                    source_path = Path(source)
                    file_key = str(source_path.absolute())
                    
                    # Get documents for this file
                    file_documents = [doc for doc in new_documents if doc.metadata.get('file_path') == str(source_path)]
                    
                    cache[file_key] = {
                        'hash': self._get_file_hash(source_path),
                        'processed_at': str(Path().cwd()),
                        'documents': [
                            {
                                'page_content': doc.page_content,
                                'metadata': doc.metadata
                            } for doc in file_documents
                        ]
                    }
                    
                    # Save cache incrementally after each file
                    self._save_cache(cache)
                    logger.info(f"Cached documents for {source_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to cache documents for {source}: {e}")
                    continue
        else:
            logger.info("No new files to process - using cached documents")
        
        # Combine cached and new documents
        all_documents = cached_documents + new_documents
        
        logger.info(f"Total documents: {len(all_documents)} ({len(cached_documents)} cached, {len(new_documents)} newly processed)")
        return all_documents
    
    def build_vector_store(self, documents: List[Document]) -> None:
        """Build Chroma vector store from documents"""
        logger.info("Building vector store...")
        
        persist_directory = self.config['vector_store']['persist_directory']
        collection_name = self.config['vector_store']['collection_name']
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Persist the vector store
        self.vector_store.persist()
        
        logger.info(f"Vector store created with {len(documents)} documents")
    
    def load_vector_store(self) -> None:
        """Load existing vector store"""
        logger.info("Loading existing vector store...")
        
        persist_directory = self.config['vector_store']['persist_directory']
        collection_name = self.config['vector_store']['collection_name']
        
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        logger.info("Vector store loaded successfully")
    
    def setup_retriever(self) -> None:
        """Setup document retriever with optional reranking"""
        logger.info("Setting up retriever...")
        
        top_k = self.config['retrieval']['top_k']
        use_reranker = self.config['retrieval']['use_reranker']
        
        # Base retriever
        base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        if use_reranker:
            # Add reranking for better relevance
            reranker_model = self.config['retrieval']['reranker_model']
            reranker_top_n = self.config['retrieval']['reranker_top_n']
            
            reranker = CrossEncoderReranker(
                model=reranker_model,
                top_n=reranker_top_n
            )
            
            self.retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=reranker
            )
            logger.info(f"Retriever setup with reranking ({reranker_model})")
        else:
            self.retriever = base_retriever
            logger.info("Basic retriever setup completed")
    
    def create_rag_chain(self) -> None:
        """Create the RAG chain for question answering"""
        logger.info("Creating RAG chain...")
        
        # Create text generation pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config['model']['max_new_tokens'],
            device_map="auto",
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Wrap in LangChain LLM
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        # Define ChatML-compatible prompt template
        template = """<|im_start|>system
You are a knowledgeable assistant on Islamic topics. Use the following context to answer the question accurately. If the context doesn't provide enough information, rely on your fine-tuned knowledge but avoid speculation.

Context: {context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Create RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info("RAG chain created successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        logger.info(f"Processing query: {question}")
        
        try:
            result = self.rag_chain({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }
    
    def initialize_pipeline(self, rebuild_index: bool = False) -> None:
        """Initialize the complete RAG pipeline"""
        logger.info("Initializing RAG pipeline...")
        
        # Setup components
        self.setup_embeddings()
        self.setup_model()
        
        # Handle vector store
        if rebuild_index or not Path(self.config['vector_store']['persist_directory']).exists():
            documents = self.load_and_process_documents()
            self.build_vector_store(documents)
        else:
            self.load_vector_store()
        
        # Setup retrieval and RAG chain
        self.setup_retriever()
        self.create_rag_chain()
        
        logger.info("RAG pipeline initialization complete")


def main():
    """Main function for testing the pipeline"""
    # Initialize pipeline
    rag = IslamicRAGPipeline()
    rag.initialize_pipeline(rebuild_index=True)
    
    # Test queries
    test_queries = [
        "How many chapters are in the Quran?",
        "What are the five pillars of Islam?",
        "Who was the first Caliph after Prophet Muhammad?",
        "What is the meaning of Shahada?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        result = rag.query(query)
        print(f"Answer: {result['answer']}")
        print(f"\nSources: {len(result['source_documents'])} documents")
        
        for i, doc in enumerate(result['source_documents'][:2]):
            print(f"\nSource {i+1}: {doc['metadata']['source']}")
            print(f"Content: {doc['content'][:200]}...")


if __name__ == "__main__":
    main()