#!/usr/bin/env python3
"""
Test Script for Islamic RAG Pipeline
Quick tests to verify functionality
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipelineTester:
    """Test suite for the RAG pipeline"""
    
    def __init__(self, config_path: str = None):
        """Initialize tester"""
        if config_path is None:
            self.config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self.test_results = {}
        self.pipeline = None
    
    def test_imports(self) -> bool:
        """Test if all required packages can be imported"""
        logger.info("Testing imports...")
        
        required_packages = [
            'torch',
            'transformers',
            'langchain',
            'chromadb',
            'sentence_transformers',
            'fastapi',
            'uvicorn',
            'numpy',
            'pandas',
            'yaml',
            'pydantic'
        ]
        
        failed_imports = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package}")
            except ImportError as e:
                logger.error(f"✗ {package}: {str(e)}")
                failed_imports.append(package)
        
        if failed_imports:
            logger.error(f"Failed to import: {', '.join(failed_imports)}")
            return False
        
        logger.info("All imports successful")
        return True
    
    def test_configuration(self) -> bool:
        """Test configuration loading"""
        logger.info("Testing configuration...")
        
        try:
            import yaml
            
            if not self.config_path.exists():
                logger.error(f"Config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['model', 'embeddings', 'vector_store', 'retrieval', 'data']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing config section: {section}")
                    return False
                logger.info(f"✓ {section} section found")
            
            logger.info("Configuration valid")
            return True
            
        except Exception as e:
            logger.error(f"Configuration test failed: {str(e)}")
            return False
    
    def test_embeddings(self) -> bool:
        """Test embedding model loading and inference"""
        logger.info("Testing embeddings...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load embedding model
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Test encoding
            test_texts = [
                "What are the five pillars of Islam?",
                "Tell me about prayer in Islam",
                "What is the meaning of Ramadan?"
            ]
            
            embeddings = model.encode(test_texts)
            
            logger.info(f"✓ Embeddings shape: {embeddings.shape}")
            logger.info(f"✓ Embedding dimension: {embeddings.shape[1]}")
            
            # Test similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            logger.info(f"✓ Sample similarity: {similarity:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Embeddings test failed: {str(e)}")
            return False
    
    def test_vector_store(self) -> bool:
        """Test vector store functionality"""
        logger.info("Testing vector store...")
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            # Create temporary client
            client = chromadb.Client()
            collection = client.create_collection("test_collection")
            
            # Test documents
            documents = [
                "The five pillars of Islam are Shahada, Salah, Zakat, Sawm, and Hajj.",
                "Prayer (Salah) is performed five times a day facing Mecca.",
                "Ramadan is the ninth month of the Islamic calendar."
            ]
            
            # Add documents
            collection.add(
                documents=documents,
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
            
            # Test query
            results = collection.query(
                query_texts=["What are the pillars of Islam?"],
                n_results=2
            )
            
            logger.info(f"✓ Retrieved {len(results['documents'][0])} documents")
            logger.info(f"✓ Top result: {results['documents'][0][0][:50]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Vector store test failed: {str(e)}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test model loading (if available)"""
        logger.info("Testing model loading...")
        
        try:
            import yaml
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_path = config['model']['path']
            
            # Check if model exists
            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}")
                logger.info("Skipping model loading test")
                return True
            
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("✓ Tokenizer loaded")
            
            # Try to load model (with minimal resources)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if config['model'].get('load_in_4bit', False):
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            logger.info(f"✓ Model loaded on {device}")
            logger.info(f"✓ Model parameters: {model.num_parameters():,}")
            
            # Test tokenization
            test_text = "What are the five pillars of Islam?"
            tokens = tokenizer.encode(test_text, return_tensors="pt")
            logger.info(f"✓ Tokenization test: {tokens.shape[1]} tokens")
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading test failed: {str(e)}")
            return False
    
    def test_data_availability(self) -> bool:
        """Test if data sources are available"""
        logger.info("Testing data availability...")
        
        try:
            import yaml
            
            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            data_sources = config['data']['sources']
            
            available_sources = 0
            total_files = 0
            
            for source in data_sources:
                source_path = Path(source)
                if not source_path.is_absolute():
                    source_path = self.config_path.parent / source
                
                if source_path.exists():
                    if source_path.is_dir():
                        files = list(source_path.glob("**/*.txt")) + list(source_path.glob("**/*.json"))
                        file_count = len(files)
                        total_files += file_count
                        logger.info(f"✓ {source}: {file_count} files")
                        available_sources += 1
                    else:
                        logger.info(f"✓ {source}: file exists")
                        total_files += 1
                        available_sources += 1
                else:
                    logger.warning(f"⚠ {source}: not found")
            
            logger.info(f"Available sources: {available_sources}/{len(data_sources)}")
            logger.info(f"Total files: {total_files}")
            
            return available_sources > 0
            
        except Exception as e:
            logger.error(f"Data availability test failed: {str(e)}")
            return False
    
    def test_pipeline_initialization(self) -> bool:
        """Test RAG pipeline initialization"""
        logger.info("Testing pipeline initialization...")
        
        try:
            from rag_pipeline import IslamicRAGPipeline
            
            # Initialize pipeline
            self.pipeline = IslamicRAGPipeline(str(self.config_path))
            
            # Test setup (without model loading to save resources)
            # self.pipeline.setup()
            
            logger.info("✓ Pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            return False
    
    def test_api_server(self) -> bool:
        """Test API server startup"""
        logger.info("Testing API server...")
        
        try:
            from api_server import app
            from fastapi.testclient import TestClient
            
            # Create test client
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/health")
            
            if response.status_code == 200:
                logger.info("✓ API server health check passed")
                return True
            else:
                logger.error(f"API health check failed: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"API server test failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests"""
        logger.info("Starting RAG pipeline tests...")
        
        tests = [
            ("imports", self.test_imports),
            ("configuration", self.test_configuration),
            ("embeddings", self.test_embeddings),
            ("vector_store", self.test_vector_store),
            ("data_availability", self.test_data_availability),
            ("pipeline_initialization", self.test_pipeline_initialization),
            ("api_server", self.test_api_server),
            ("model_loading", self.test_model_loading)  # Last as it's resource intensive
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                result = test_func()
                results[test_name] = result
                
                elapsed = time.time() - start_time
                status = "PASS" if result else "FAIL"
                logger.info(f"Test {test_name}: {status} ({elapsed:.2f}s)")
                
            except Exception as e:
                results[test_name] = False
                elapsed = time.time() - start_time
                logger.error(f"Test {test_name}: ERROR ({elapsed:.2f}s) - {str(e)}")
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        logger.info(f"\n{'='*50}")
        logger.info("Test Summary")
        logger.info(f"{'='*50}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name:25}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("\n🎉 All tests passed! RAG pipeline is ready.")
        elif passed >= total * 0.8:
            logger.info("\n⚠️  Most tests passed. Check failed tests.")
        else:
            logger.error("\n❌ Multiple tests failed. Please resolve issues.")
        
        # Recommendations
        if not results.get('model_loading', True):
            logger.info("\n💡 Tip: Ensure your QLoRA model is available in the checkpoints directory")
        
        if not results.get('data_availability', True):
            logger.info("\n💡 Tip: Run 'python scripts/data_preparation.py' to generate sample data")
        
        if not results.get('imports', True):
            logger.info("\n💡 Tip: Run 'pip install -r requirements.txt' to install dependencies")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Islamic RAG Pipeline")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--test", help="Run specific test only")
    parser.add_argument("--quick", action="store_true", help="Skip resource-intensive tests")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = RAGPipelineTester(args.config)
    
    if args.test:
        # Run specific test
        test_method = getattr(tester, f"test_{args.test}", None)
        if test_method:
            result = test_method()
            print(f"Test {args.test}: {'PASS' if result else 'FAIL'}")
        else:
            print(f"Test '{args.test}' not found")
            print("Available tests: imports, configuration, embeddings, vector_store, data_availability, pipeline_initialization, api_server, model_loading")
    else:
        # Run all tests
        if args.quick:
            # Skip model loading test
            tester.test_model_loading = lambda: True
            logger.info("Quick mode: skipping model loading test")
        
        results = tester.run_all_tests()
        
        # Exit with appropriate code
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        if passed == total:
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()