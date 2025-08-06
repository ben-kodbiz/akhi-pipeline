#!/usr/bin/env python3
"""
Automatic RAG to Axolotl Training Pipeline
Automated end-to-end pipeline from RAG documents to trained model

This script automates the complete workflow:
1. Convert RAG documents to training format
2. Prepare Axolotl dataset
3. Generate training configuration
4. Optionally start training

Usage:
    python auto_train_from_rag.py --prepare-only
    python auto_train_from_rag.py --train
    python auto_train_from_rag.py --model microsoft/DialoGPT-medium --epochs 5

Author: Akhi Data Builder Team
Date: January 2025
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoTrainingPipeline:
    """
    Automated pipeline from RAG to trained model
    """
    
    def __init__(self, base_dir: str = None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Auto-detect base directory
            current_dir = Path(__file__).parent.parent
            self.base_dir = current_dir.parent  # Should be akhi_data_builder
        
        self.rag_dir = self.base_dir / "RAG"
        self.crewai_dir = self.base_dir / "akhi_crewai"
        
        logger.info(f"Pipeline initialized with base directory: {self.base_dir}")
        
        # Verify directories exist
        if not self.rag_dir.exists():
            raise FileNotFoundError(f"RAG directory not found: {self.rag_dir}")
        if not self.crewai_dir.exists():
            raise FileNotFoundError(f"CrewAI directory not found: {self.crewai_dir}")
    
    def step1_convert_rag_to_training(self, output_dir: str = "training_data") -> Path:
        """Step 1: Convert RAG documents to training format"""
        logger.info("Step 1: Converting RAG documents to training format...")
        
        bridge_script = self.rag_dir / "scripts" / "rag_to_axolotl_bridge.py"
        config_file = self.rag_dir / "config.yaml"
        output_path = self.rag_dir / output_dir
        
        if not bridge_script.exists():
            raise FileNotFoundError(f"Bridge script not found: {bridge_script}")
        
        # Run the bridge script
        cmd = [
            sys.executable, str(bridge_script),
            "--config", str(config_file),
            "--output", output_dir,
            "--verbose"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.rag_dir),
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("RAG to training conversion completed successfully")
            logger.debug(f"Output: {result.stdout}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"RAG conversion failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def step2_prepare_axolotl_dataset(self, 
                                     training_data_dir: Path, 
                                     model: str = "microsoft/DialoGPT-medium",
                                     output_dir: str = "axolotl_ready") -> Path:
        """Step 2: Prepare Axolotl dataset"""
        logger.info("Step 2: Preparing Axolotl dataset...")
        
        prepare_script = self.crewai_dir / "prepare_axolotl_dataset.py"
        output_path = self.crewai_dir / output_dir
        
        if not prepare_script.exists():
            raise FileNotFoundError(f"Axolotl preparation script not found: {prepare_script}")
        
        # Run the preparation script
        cmd = [
            sys.executable, str(prepare_script),
            "--input", str(training_data_dir),
            "--output", output_dir,
            "--model", model,
            "--val-split", "0.1",
            "--max-length", "2048"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.crewai_dir),
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Axolotl dataset preparation completed successfully")
            logger.debug(f"Output: {result.stdout}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Axolotl preparation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def step3_start_training(self, axolotl_dir: Path) -> bool:
        """Step 3: Start Axolotl training"""
        logger.info("Step 3: Starting Axolotl training...")
        
        training_script = axolotl_dir / "train_model.sh"
        
        if not training_script.exists():
            logger.error(f"Training script not found: {training_script}")
            return False
        
        # Make script executable
        training_script.chmod(0o755)
        
        logger.info(f"Starting training with script: {training_script}")
        logger.info("Note: Training will run in the background. Check logs for progress.")
        
        try:
            # Start training in background
            process = subprocess.Popen(
                [str(training_script)],
                cwd=str(axolotl_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Training started with PID: {process.pid}")
            logger.info(f"Training directory: {axolotl_dir}")
            logger.info("Training is running in the background.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False
    
    def run_pipeline(self, 
                    model: str = "microsoft/DialoGPT-medium",
                    prepare_only: bool = False,
                    start_training: bool = False) -> dict:
        """Run the complete pipeline"""
        logger.info("Starting automated RAG to Axolotl training pipeline...")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'output_directories': {},
            'success': False
        }
        
        try:
            # Step 1: Convert RAG to training format
            training_data_dir = self.step1_convert_rag_to_training()
            results['steps_completed'].append('rag_conversion')
            results['output_directories']['training_data'] = str(training_data_dir)
            
            # Step 2: Prepare Axolotl dataset
            axolotl_dir = self.step2_prepare_axolotl_dataset(training_data_dir, model)
            results['steps_completed'].append('axolotl_preparation')
            results['output_directories']['axolotl_ready'] = str(axolotl_dir)
            
            if not prepare_only and start_training:
                # Step 3: Start training
                training_started = self.step3_start_training(axolotl_dir)
                if training_started:
                    results['steps_completed'].append('training_started')
                else:
                    logger.warning("Training could not be started")
            
            results['success'] = True
            results['end_time'] = datetime.now().isoformat()
            
            logger.info("Pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            return results
    
    def print_summary(self, results: dict):
        """Print pipeline summary"""
        print("\n" + "="*70)
        print("AUTOMATED RAG TO AXOLOTL TRAINING PIPELINE")
        print("="*70)
        
        if results['success']:
            print("✅ Pipeline Status: SUCCESS")
        else:
            print("❌ Pipeline Status: FAILED")
            if 'error' in results:
                print(f"   Error: {results['error']}")
        
        print(f"\n📊 Execution Summary:")
        print(f"   Start time: {results['start_time']}")
        print(f"   End time: {results.get('end_time', 'N/A')}")
        print(f"   Steps completed: {', '.join(results['steps_completed'])}")
        
        if 'output_directories' in results:
            print(f"\n📁 Generated Directories:")
            for name, path in results['output_directories'].items():
                print(f"   {name}: {path}")
        
        if 'training_started' in results['steps_completed']:
            print(f"\n🚀 Training Status:")
            print(f"   Training has been started in the background")
            print(f"   Monitor progress in: {results['output_directories']['axolotl_ready']}")
            print(f"   Check logs for training progress")
        elif results['success']:
            print(f"\n🚀 Next Steps:")
            print(f"   1. Review the prepared dataset in: {results['output_directories']['axolotl_ready']}")
            print(f"   2. Start training manually:")
            print(f"      cd {results['output_directories']['axolotl_ready']}")
            print(f"      ./train_model.sh")
        
        print("="*70)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Automated RAG to Axolotl training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_train_from_rag.py --prepare-only
  python auto_train_from_rag.py --train
  python auto_train_from_rag.py --model microsoft/DialoGPT-medium --epochs 5 --train
  python auto_train_from_rag.py --base-dir /path/to/akhi_data_builder --prepare-only

The script will:
1. Convert RAG documents to training format
2. Prepare Axolotl dataset with proper configuration
3. Optionally start training process
"""
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='microsoft/DialoGPT-medium',
        help='Base model for fine-tuning (default: microsoft/DialoGPT-medium)'
    )
    
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare dataset, do not start training'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Start training after preparation'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory (default: auto-detect)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Default behavior
    if not args.prepare_only and not args.train:
        args.prepare_only = True
        logger.info("No action specified, defaulting to --prepare-only")
    
    try:
        # Initialize pipeline
        pipeline = AutoTrainingPipeline(base_dir=args.base_dir)
        
        # Run pipeline
        results = pipeline.run_pipeline(
            model=args.model,
            prepare_only=args.prepare_only,
            start_training=args.train
        )
        
        # Print summary
        pipeline.print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()