#!/usr/bin/env python3
"""
Setup Script for Islamic RAG Pipeline
Installs dependencies and initializes the system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSetup:
    """Setup manager for the RAG pipeline"""
    
    def __init__(self, project_root: str = None):
        """Initialize setup manager"""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.scripts_dir = self.project_root / "scripts"
        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.rag_index_dir = self.project_root / "rag_index"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available for inference"""
        logger.info("Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                logger.info(f"GPU available: {gpu_name}")
                logger.info(f"GPU memory: {memory:.1f} GB")
                logger.info(f"GPU count: {gpu_count}")
                
                if memory < 4.0:
                    logger.warning("GPU memory < 4GB. May not be sufficient for Qwen3-1.7B")
                
                return True
            else:
                logger.warning("No GPU available. Will use CPU (slower)")
                return False
                
        except ImportError:
            logger.warning("PyTorch not installed. Cannot check GPU")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        logger.info("Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Install requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
            
            logger.info("Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {str(e)}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        logger.info("Creating project directories...")
        
        directories = [
            self.data_dir,
            self.checkpoints_dir,
            self.rag_index_dir,
            self.project_root / "evaluation_results",
            self.project_root / "logs"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            return False
    
    def prepare_sample_data(self) -> bool:
        """Prepare sample Islamic knowledge data"""
        logger.info("Preparing sample data...")
        
        try:
            # Import and run data preparation
            sys.path.append(str(self.scripts_dir))
            from data_preparation import IslamicDataCollector
            
            collector = IslamicDataCollector(str(self.data_dir))
            collector.prepare_all_data()
            
            logger.info("Sample data prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing sample data: {str(e)}")
            return False
    
    def check_model_availability(self) -> bool:
        """Check if QLoRA model is available"""
        logger.info("Checking model availability...")
        
        # Check for existing model in checkpoints
        model_paths = [
            self.checkpoints_dir / "qwen3-1.7b-qlora",
            self.checkpoints_dir / "pytorch_model.bin",
            self.checkpoints_dir / "adapter_model.bin",
            self.checkpoints_dir / "adapter_model.safetensors"
        ]
        
        model_found = False
        for path in model_paths:
            if path.exists():
                logger.info(f"Found model file: {path}")
                model_found = True
                break
        
        if not model_found:
            logger.warning("No QLoRA model found in checkpoints directory")
            logger.info("Please ensure your trained model is available in:")
            logger.info(f"  {self.checkpoints_dir}")
            logger.info("Or update the model path in config.yaml")
        
        return model_found
    
    def validate_configuration(self) -> bool:
        """Validate configuration file"""
        logger.info("Validating configuration...")
        
        config_file = self.project_root / "config.yaml"
        
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_file}")
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['model', 'embeddings', 'vector_store', 'retrieval', 'data']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing configuration section: {section}")
                    return False
            
            # Validate model path
            model_path = config['model']['path']
            if not Path(model_path).exists() and not Path(self.project_root / model_path).exists():
                logger.warning(f"Model path not found: {model_path}")
            
            # Validate data sources
            for source in config['data']['sources']:
                source_path = Path(source)
                if not source_path.exists() and not Path(self.project_root / source).exists():
                    logger.warning(f"Data source not found: {source}")
            
            logger.info("Configuration validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality of the RAG pipeline"""
        logger.info("Testing basic functionality...")
        
        try:
            # Test imports
            import torch
            import transformers
            import langchain
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            logger.info("All required packages imported successfully")
            
            # Test embedding model
            logger.info("Testing embedding model...")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            test_embedding = model.encode(["Test sentence"])
            logger.info(f"Embedding shape: {test_embedding.shape}")
            
            # Test Chroma
            logger.info("Testing Chroma vector store...")
            import chromadb
            client = chromadb.Client()
            collection = client.create_collection("test_collection")
            logger.info("Chroma vector store working")
            
            logger.info("Basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {str(e)}")
            return False
    
    def run_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("Starting RAG pipeline setup...")
        
        setup_steps = [
            ("Python version check", self.check_python_version),
            ("GPU availability check", self.check_gpu_availability),
            ("Directory creation", self.create_directories),
            ("Dependency installation", self.install_dependencies),
            ("Sample data preparation", self.prepare_sample_data),
            ("Configuration validation", self.validate_configuration),
            ("Model availability check", self.check_model_availability),
            ("Basic functionality test", self.test_basic_functionality)
        ]
        
        results = {}
        
        for step_name, step_func in setup_steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = step_func()
                results[step_name] = result
                
                if result:
                    logger.info(f"✓ {step_name} - PASSED")
                else:
                    logger.warning(f"⚠ {step_name} - FAILED")
                    
            except Exception as e:
                logger.error(f"✗ {step_name} - ERROR: {str(e)}")
                results[step_name] = False
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("Setup Summary")
        logger.info(f"{'='*50}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for step_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{step_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} steps passed")
        
        if passed == total:
            logger.info("\n🎉 Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Ensure your QLoRA model is in the checkpoints directory")
            logger.info("2. Run: python scripts/rag_pipeline.py (to test the pipeline)")
            logger.info("3. Run: python scripts/api_server.py (to start the API server)")
            logger.info("4. Run: python scripts/evaluation.py (to evaluate performance)")
            return True
        else:
            logger.error("\n❌ Setup completed with errors")
            logger.error("Please resolve the failed steps before proceeding")
            return False

def main():
    """Main function for setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Islamic RAG Pipeline")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = RAGSetup(args.project_root)
    
    # Modify setup steps based on arguments
    if args.skip_deps:
        setup.install_dependencies = lambda: True
        logger.info("Skipping dependency installation")
    
    if args.skip_data:
        setup.prepare_sample_data = lambda: True
        logger.info("Skipping data preparation")
    
    # Run setup
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()