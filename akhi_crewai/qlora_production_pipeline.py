#!/usr/bin/env python3
"""
QLoRA Production Pipeline Implementation
Complete workflow from YouTube video gathering to Axolotl JSON submission

Author: Akhi CrewAI Team
Date: January 2025
Version: 1.0
"""

import os
import json
import yaml
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import Akhi CrewAI tools
from tools.youtube_downloader import YouTubeDownloaderTool
from tools.transcriber import TranscriptionTool
from tools.qlora_formatter import QLoRAFormatterTool
from tools.axolotl_trainer import AxolotlTrainerTool
from tools.model_validator import ModelValidatorTool
from tools.model_deployer import ModelDeployerTool
from agents.content_qa import ContentQAAgent
from agents.vector_indexer import VectorIndexerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qlora_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for QLoRA production pipeline"""
    # Input configuration
    youtube_urls: List[str]
    output_dir: str = "./qlora_production_output"
    
    # Processing configuration
    batch_size: int = 10
    max_workers: int = 4
    quality_threshold: float = 0.8
    
    # QLoRA configuration
    model_name: str = "microsoft/DialoGPT-medium"
    max_sequence_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training configuration
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size_training: int = 4
    gradient_accumulation_steps: int = 4
    
    # Validation configuration
    validation_split: float = 0.1
    islamic_keywords: List[str] = None
    
    def __post_init__(self):
        if self.islamic_keywords is None:
            self.islamic_keywords = [
                "Allah", "Prophet", "Quran", "Hadith", "Islam", "Muslim",
                "Salah", "Zakat", "Hajj", "Ramadan", "Tawheed", "Fiqh"
            ]

class QLoRAProductionPipeline:
    """Complete QLoRA production pipeline implementation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.setup_directories()
        
        # Initialize tools
        self.youtube_downloader = YouTubeDownloaderTool()
        self.transcriber = TranscriptionTool()
        self.qlora_formatter = QLoRAFormatterTool()
        self.axolotl_trainer = AxolotlTrainerTool()
        self.model_validator = ModelValidatorTool()
        self.model_deployer = ModelDeployerTool()
        
        # Initialize agents
        self.content_qa = ContentQAAgent()
        self.vector_indexer = VectorIndexerAgent()
        
        # Pipeline state
        self.pipeline_state = {
            "phase": "initialized",
            "processed_videos": 0,
            "total_videos": len(config.youtube_urls),
            "start_time": datetime.now().isoformat(),
            "errors": [],
            "metrics": {}
        }
        
        logger.info(f"QLoRA Production Pipeline initialized with {len(config.youtube_urls)} videos")
    
    def setup_directories(self):
        """Create necessary directories for pipeline output"""
        directories = [
            self.output_dir,
            self.output_dir / "videos",
            self.output_dir / "audio",
            self.output_dir / "transcripts",
            self.output_dir / "qa_content",
            self.output_dir / "qlora_datasets",
            self.output_dir / "axolotl_configs",
            self.output_dir / "trained_models",
            self.output_dir / "logs",
            self.output_dir / "metrics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directories in {self.output_dir}")
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete QLoRA production pipeline"""
        try:
            logger.info("Starting QLoRA Production Pipeline")
            
            # Phase 1: Data Collection
            await self.phase_1_data_collection()
            
            # Phase 2: Content Processing
            await self.phase_2_content_processing()
            
            # Phase 3: QLoRA Formatting
            await self.phase_3_qlora_formatting()
            
            # Phase 4: Axolotl Configuration
            await self.phase_4_axolotl_configuration()
            
            # Phase 5: Model Training
            await self.phase_5_model_training()
            
            # Phase 6: Model Validation
            await self.phase_6_model_validation()
            
            # Phase 7: Model Deployment
            await self.phase_7_model_deployment()
            
            # Generate final report
            final_report = self.generate_final_report()
            
            logger.info("QLoRA Production Pipeline completed successfully")
            return final_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.pipeline_state["errors"].append({
                "phase": self.pipeline_state["phase"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    async def phase_1_data_collection(self):
        """Phase 1: YouTube video downloading and audio extraction"""
        logger.info("Phase 1: Starting data collection")
        self.pipeline_state["phase"] = "data_collection"
        
        downloaded_videos = []
        failed_downloads = []
        
        for i, url in enumerate(self.config.youtube_urls):
            try:
                logger.info(f"Processing video {i+1}/{len(self.config.youtube_urls)}: {url}")
                
                # Download video and extract audio
                result = await self.youtube_downloader.download_video(
                    url=url,
                    output_dir=str(self.output_dir / "videos"),
                    extract_audio=True,
                    audio_dir=str(self.output_dir / "audio")
                )
                
                if result["success"]:
                    downloaded_videos.append(result)
                    self.pipeline_state["processed_videos"] += 1
                    logger.info(f"Successfully downloaded: {result['title']}")
                else:
                    failed_downloads.append({"url": url, "error": result.get("error", "Unknown error")})
                    logger.warning(f"Failed to download: {url}")
                
            except Exception as e:
                failed_downloads.append({"url": url, "error": str(e)})
                logger.error(f"Error downloading {url}: {str(e)}")
        
        # Save download results
        download_report = {
            "successful_downloads": len(downloaded_videos),
            "failed_downloads": len(failed_downloads),
            "success_rate": len(downloaded_videos) / len(self.config.youtube_urls) * 100,
            "downloaded_videos": downloaded_videos,
            "failed_downloads": failed_downloads
        }
        
        with open(self.output_dir / "logs" / "download_report.json", "w") as f:
            json.dump(download_report, f, indent=2)
        
        self.pipeline_state["metrics"]["download_success_rate"] = download_report["success_rate"]
        logger.info(f"Phase 1 completed: {download_report['success_rate']:.1f}% success rate")
    
    async def phase_2_content_processing(self):
        """Phase 2: Transcription and content quality assessment"""
        logger.info("Phase 2: Starting content processing")
        self.pipeline_state["phase"] = "content_processing"
        
        audio_files = list((self.output_dir / "audio").glob("*.wav"))
        transcription_results = []
        
        for audio_file in audio_files:
            try:
                logger.info(f"Transcribing: {audio_file.name}")
                
                # Transcribe audio
                transcript_result = await self.transcriber.transcribe_audio(
                    audio_path=str(audio_file),
                    output_dir=str(self.output_dir / "transcripts")
                )
                
                if transcript_result["success"]:
                    # Quality assessment - read transcript content
                    try:
                        with open(transcript_result["transcript_path"], 'r', encoding='utf-8') as f:
                            transcript_content = f.read()
                        
                        qa_result = self.content_qa.assess_content(
                            content=transcript_content,
                            content_type="transcript"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read transcript {transcript_result['transcript_path']}: {e}")
                        qa_result = {'success': False, 'error': str(e), 'quality_score': 0.0}
                    
                    # Combine results
                    combined_result = {
                        **transcript_result,
                        "quality_assessment": qa_result,
                        "audio_file": str(audio_file)
                    }
                    
                    transcription_results.append(combined_result)
                    logger.info(f"Processed: {audio_file.name} (Quality: {qa_result.get('quality_score', 'N/A')})")
                
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {str(e)}")
        
        # Filter high-quality content
        high_quality_content = [
            result for result in transcription_results
            if result.get("quality_assessment", {}).get("quality_score", 0) >= self.config.quality_threshold
        ]
        
        # Save processing results
        processing_report = {
            "total_processed": len(transcription_results),
            "high_quality_content": len(high_quality_content),
            "quality_filter_rate": len(high_quality_content) / len(transcription_results) * 100 if transcription_results else 0,
            "transcription_results": transcription_results
        }
        
        with open(self.output_dir / "logs" / "processing_report.json", "w") as f:
            json.dump(processing_report, f, indent=2)
        
        self.pipeline_state["metrics"]["quality_filter_rate"] = processing_report["quality_filter_rate"]
        logger.info(f"Phase 2 completed: {len(high_quality_content)} high-quality transcripts")
    
    async def phase_3_qlora_formatting(self):
        """Phase 3: Format content for QLoRA training"""
        logger.info("Phase 3: Starting QLoRA formatting")
        self.pipeline_state["phase"] = "qlora_formatting"
        
        # Load high-quality transcripts
        transcript_files = list((self.output_dir / "transcripts").glob("*.txt"))
        qlora_datasets = []
        
        for transcript_file in transcript_files:
            try:
                logger.info(f"Formatting: {transcript_file.name}")
                
                # Format for QLoRA
                format_result_str = self.qlora_formatter._run(
                    transcript_dir=str(transcript_file.parent),
                    output_file=str(self.output_dir / "qlora_datasets" / f"{transcript_file.stem}.jsonl"),
                    min_segment_words=50,
                    max_segment_words=500,
                    include_metadata=True,
                    filter_islamic_content=True
                )
                
                # Parse the result (it returns a JSON string)
                try:
                    format_result = json.loads(format_result_str)
                except json.JSONDecodeError:
                    format_result = {'success': False, 'error': 'Failed to parse QLoRA formatter output'}
                
                if format_result["success"]:
                    qlora_datasets.append(format_result)
                    logger.info(f"Formatted: {transcript_file.name} -> {format_result['samples_generated']} samples")
                
            except Exception as e:
                logger.error(f"Error formatting {transcript_file}: {str(e)}")
        
        # Combine all datasets
        combined_dataset = await self.combine_qlora_datasets(qlora_datasets)
        
        # Save formatting results
        formatting_report = {
            "datasets_created": len(qlora_datasets),
            "total_samples": combined_dataset["total_samples"],
            "combined_dataset_path": combined_dataset["dataset_path"],
            "formatting_results": qlora_datasets
        }
        
        with open(self.output_dir / "logs" / "formatting_report.json", "w") as f:
            json.dump(formatting_report, f, indent=2)
        
        self.pipeline_state["metrics"]["total_training_samples"] = combined_dataset["total_samples"]
        logger.info(f"Phase 3 completed: {combined_dataset['total_samples']} training samples")
    
    async def phase_4_axolotl_configuration(self):
        """Phase 4: Generate Axolotl configuration files"""
        logger.info("Phase 4: Starting Axolotl configuration")
        self.pipeline_state["phase"] = "axolotl_configuration"
        
        # Generate Axolotl config
        axolotl_config = {
            "base_model": self.config.model_name,
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            
            "load_in_8bit": False,
            "load_in_4bit": True,
            "strict": False,
            
            "datasets": [
                {
                    "path": str(self.output_dir / "qlora_datasets" / "combined_dataset.json"),
                    "type": "alpaca"
                }
            ],
            
            "dataset_prepared_path": str(self.output_dir / "qlora_datasets" / "prepared"),
            "val_set_size": self.config.validation_split,
            "output_dir": str(self.output_dir / "trained_models"),
            
            "sequence_len": self.config.max_sequence_length,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            
            "adapter": "qlora",
            "lora_model_dir": str(self.output_dir / "trained_models" / "lora_adapters"),
            
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            "lora_target_linear": True,
            "lora_fan_in_fan_out": False,
            
            "wandb_project": "akhi-qlora-islamic-ai",
            "wandb_entity": "",
            "wandb_watch": "",
            "wandb_name": f"islamic-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "wandb_log_model": "",
            
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "micro_batch_size": self.config.batch_size_training,
            "num_epochs": self.config.num_epochs,
            "optimizer": "adamw_bnb_8bit",
            "lr_scheduler": "cosine",
            "learning_rate": self.config.learning_rate,
            
            "train_on_inputs": False,
            "group_by_length": False,
            "bf16": True,
            "fp16": False,
            "tf32": False,
            
            "gradient_checkpointing": True,
            "early_stopping_patience": "",
            "resume_from_checkpoint": "",
            "local_rank": "",
            
            "logging_steps": 1,
            "xformers_attention": "",
            "flash_attention": True,
            
            "warmup_steps": 10,
            "evals_per_epoch": 4,
            "eval_table_size": "",
            "eval_max_new_tokens": 128,
            "saves_per_epoch": 1,
            "debug": "",
            "deepspeed": "",
            "weight_decay": 0.0,
            "fsdp": "",
            "fsdp_config": "",
            "special_tokens": {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>"
            }
        }
        
        # Save Axolotl configuration
        config_path = self.output_dir / "axolotl_configs" / "qlora_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(axolotl_config, f, default_flow_style=False)
        
        # Generate training script
        training_script = self.generate_training_script(str(config_path))
        script_path = self.output_dir / "axolotl_configs" / "train_model.sh"
        with open(script_path, "w") as f:
            f.write(training_script)
        script_path.chmod(0o755)
        
        logger.info(f"Phase 4 completed: Axolotl config saved to {config_path}")
    
    async def phase_5_model_training(self):
        """Phase 5: Train the model using Axolotl"""
        logger.info("Phase 5: Starting model training")
        self.pipeline_state["phase"] = "model_training"
        
        config_path = self.output_dir / "axolotl_configs" / "qlora_config.yml"
        
        try:
            # Start training
            training_result = await self.axolotl_trainer.train_model(
                config_path=str(config_path),
                output_dir=str(self.output_dir / "trained_models"),
                monitor_training=True
            )
            
            if training_result["success"]:
                logger.info(f"Training completed successfully: {training_result['model_path']}")
                self.pipeline_state["metrics"]["training_success"] = True
                self.pipeline_state["metrics"]["model_path"] = training_result["model_path"]
            else:
                raise Exception(f"Training failed: {training_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.pipeline_state["metrics"]["training_success"] = False
            raise
    
    async def phase_6_model_validation(self):
        """Phase 6: Validate the trained model"""
        logger.info("Phase 6: Starting model validation")
        self.pipeline_state["phase"] = "model_validation"
        
        model_path = self.pipeline_state["metrics"].get("model_path")
        if not model_path:
            raise Exception("No trained model found for validation")
        
        try:
            # Validate model
            validation_result = await self.model_validator.validate_model(
                model_path=model_path,
                test_prompts=self.generate_test_prompts(),
                islamic_keywords=self.config.islamic_keywords,
                output_dir=str(self.output_dir / "metrics")
            )
            
            if validation_result["success"]:
                logger.info(f"Validation completed: {validation_result['overall_score']:.2f} score")
                self.pipeline_state["metrics"]["validation_score"] = validation_result["overall_score"]
                self.pipeline_state["metrics"]["validation_details"] = validation_result
            else:
                raise Exception(f"Validation failed: {validation_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
    
    async def phase_7_model_deployment(self):
        """Phase 7: Deploy the validated model"""
        logger.info("Phase 7: Starting model deployment")
        self.pipeline_state["phase"] = "model_deployment"
        
        model_path = self.pipeline_state["metrics"].get("model_path")
        validation_score = self.pipeline_state["metrics"].get("validation_score", 0)
        
        if validation_score < 0.8:
            logger.warning(f"Model validation score ({validation_score:.2f}) below threshold (0.8)")
            return
        
        try:
            # Deploy model
            deployment_result = await self.model_deployer.deploy_model(
                model_path=model_path,
                deployment_type="local",
                api_endpoint=True,
                output_dir=str(self.output_dir / "deployment")
            )
            
            if deployment_result["success"]:
                logger.info(f"Deployment completed: {deployment_result['endpoint_url']}")
                self.pipeline_state["metrics"]["deployment_success"] = True
                self.pipeline_state["metrics"]["endpoint_url"] = deployment_result["endpoint_url"]
            else:
                raise Exception(f"Deployment failed: {deployment_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            self.pipeline_state["metrics"]["deployment_success"] = False
            raise
    
    async def combine_qlora_datasets(self, datasets: List[Dict]) -> Dict[str, Any]:
        """Combine multiple QLoRA datasets into a single training dataset"""
        combined_data = []
        total_samples = 0
        
        for dataset in datasets:
            if dataset["success"] and "dataset_path" in dataset:
                with open(dataset["dataset_path"], "r") as f:
                    data = json.load(f)
                    combined_data.extend(data)
                    total_samples += len(data)
        
        # Save combined dataset
        combined_path = self.output_dir / "qlora_datasets" / "combined_dataset.json"
        with open(combined_path, "w") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        return {
            "dataset_path": str(combined_path),
            "total_samples": total_samples,
            "success": True
        }
    
    def generate_training_script(self, config_path: str) -> str:
        """Generate Axolotl training script"""
        return f"""#!/bin/bash
# Axolotl QLoRA Training Script
# Generated by Akhi CrewAI Pipeline

set -e

echo "Starting QLoRA training with Axolotl..."
echo "Config: {config_path}"
echo "Timestamp: $(date)"

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Install/update Axolotl if needed
# pip install -e git+https://github.com/OpenAccess-AI-Collective/axolotl.git

# Preprocess dataset
echo "Preprocessing dataset..."
python -m axolotl.cli.preprocess {config_path}

# Start training
echo "Starting training..."
python -m axolotl.cli.train {config_path}

# Merge LoRA adapters
echo "Merging LoRA adapters..."
python -m axolotl.cli.merge_lora {config_path} --lora_model_dir="$(dirname {config_path})/../trained_models/lora_adapters"

echo "Training completed successfully!"
echo "Model saved to: $(dirname {config_path})/../trained_models/"
"""
    
    def generate_test_prompts(self) -> List[str]:
        """Generate test prompts for Islamic AI validation"""
        return [
            "What are the five pillars of Islam?",
            "Explain the concept of Tawheed in Islam.",
            "What is the importance of Salah in a Muslim's daily life?",
            "Describe the significance of Zakat in Islamic society.",
            "What are the conditions for performing Hajj?",
            "Explain the concept of Jihad in its true Islamic context.",
            "What is the role of the Quran in Islamic guidance?",
            "Describe the importance of following the Sunnah of Prophet Muhammad (PBUH).",
            "What are the major themes in Surah Al-Fatiha?",
            "Explain the Islamic perspective on social justice."
        ]
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.pipeline_state["start_time"])
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "pipeline_summary": {
                "status": "completed" if not self.pipeline_state["errors"] else "completed_with_errors",
                "start_time": self.pipeline_state["start_time"],
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_videos": self.pipeline_state["total_videos"],
                "processed_videos": self.pipeline_state["processed_videos"]
            },
            "metrics": self.pipeline_state["metrics"],
            "errors": self.pipeline_state["errors"],
            "output_files": {
                "axolotl_config": str(self.output_dir / "axolotl_configs" / "qlora_config.yml"),
                "training_script": str(self.output_dir / "axolotl_configs" / "train_model.sh"),
                "combined_dataset": str(self.output_dir / "qlora_datasets" / "combined_dataset.json"),
                "logs_directory": str(self.output_dir / "logs"),
                "metrics_directory": str(self.output_dir / "metrics")
            },
            "next_steps": [
                "Review training logs for optimization opportunities",
                "Conduct additional validation with Islamic scholars",
                "Deploy model to production environment",
                "Set up monitoring and feedback collection",
                "Plan for continuous model improvement"
            ]
        }
        
        # Save final report
        report_path = self.output_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to: {report_path}")
        return report

# Example usage and testing functions
async def run_production_test():
    """Run a production test of the QLoRA pipeline"""
    
    # Example configuration
    test_config = PipelineConfig(
        youtube_urls=[
            "https://www.youtube.com/watch?v=example1",  # Replace with actual Islamic content URLs
            "https://www.youtube.com/watch?v=example2",
            "https://www.youtube.com/watch?v=example3"
        ],
        output_dir="./qlora_production_test",
        batch_size=5,
        quality_threshold=0.7,
        num_epochs=1,  # Reduced for testing
        learning_rate=2e-4
    )
    
    # Initialize and run pipeline
    pipeline = QLoRAProductionPipeline(test_config)
    
    try:
        result = await pipeline.run_complete_pipeline()
        print("\n" + "="*50)
        print("QLORA PRODUCTION PIPELINE COMPLETED")
        print("="*50)
        print(f"Status: {result['pipeline_summary']['status']}")
        print(f"Duration: {result['pipeline_summary']['duration_seconds']:.1f} seconds")
        print(f"Videos Processed: {result['pipeline_summary']['processed_videos']}/{result['pipeline_summary']['total_videos']}")
        
        if "download_success_rate" in result["metrics"]:
            print(f"Download Success Rate: {result['metrics']['download_success_rate']:.1f}%")
        
        if "total_training_samples" in result["metrics"]:
            print(f"Training Samples Generated: {result['metrics']['total_training_samples']}")
        
        if "validation_score" in result["metrics"]:
            print(f"Model Validation Score: {result['metrics']['validation_score']:.2f}")
        
        print(f"\nOutput Directory: {pipeline.output_dir}")
        print(f"Axolotl Config: {result['output_files']['axolotl_config']}")
        print(f"Training Dataset: {result['output_files']['combined_dataset']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the production test
    import asyncio
    asyncio.run(run_production_test())