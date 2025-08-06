#!/usr/bin/env python3
"""
End-to-End Pipeline Test System
Tests complete workflow from YouTube search to QLoRA fine-tuning

Test Scenario:
- Search YouTube for trauma content from Mufti Menk, Nouman Khan
- Filter videos under 20 minutes
- Process through complete pipeline
- Validate QLoRA integration with Qwen 1.7B
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'akhi_crewai'))

# Import pipeline components
try:
    from akhi_crewai.crew.akhi_pipeline import AkhiPipelineCrew
    from akhi_crewai.main import AkhiPipelineApp
    from akhi_crewai.tools.youtube_search import YouTubeSearchTool
    from akhi_crewai.tools.youtube_downloader import YouTubeDownloaderTool
    from akhi_crewai.tools.transcriber import TranscriptionTool
    from akhi_crewai.tools.qlora_formatter import QLoRAFormatterTool
    from akhi_crewai.tools.axolotl_trainer import AxolotlTrainerTool
    from akhi_crewai.tools.model_validator import ModelValidatorTool
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct")
    sys.exit(1)

class EndToEndPipelineTest:
    """Comprehensive end-to-end pipeline testing system"""
    
    def __init__(self, test_output_dir: str = "test_results"):
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_config = {
            "search_keywords": [
                "trauma healing islam mufti menk",
                "dealing with trauma nouman ali khan",
                "islamic counseling trauma mufti menk",
                "mental health islam nouman khan"
            ],
            "target_channels": [
                "Mufti Menk",
                "Nouman Ali Khan",
                "Bayyinah Institute"
            ],
            "max_duration_minutes": 20,
            "max_videos_per_search": 3,
            "test_model": "qwen1.5-1.8b",
            "qlora_config": {
                "num_epochs": 1,  # Reduced for testing
                "learning_rate": 2e-4,
                "batch_size": 2,
                "max_seq_length": 1024
            }
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize YouTube tools directly
        try:
            from akhi_crewai.tools.youtube_search import YouTubeSearchTool
            from akhi_crewai.tools.youtube_downloader import YouTubeDownloaderTool
            self.youtube_search_tool = YouTubeSearchTool()
            self.youtube_downloader_tool = YouTubeDownloaderTool()
            self.logger.info("YouTube tools initialized successfully")
        except Exception as e:
            self.logger.error(f"Could not initialize YouTube tools: {str(e)}")
            raise
        
        # Test results tracking
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "test_phases": {},
            "errors": [],
            "warnings": [],
            "success": False
        }
        
    def setup_logging(self):
        """Setup comprehensive logging for test execution"""
        log_file = self.test_output_dir / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("End-to-End Pipeline Test System Initialized")
        
    def log_phase_result(self, phase: str, success: bool, details: Dict[str, Any]):
        """Log results for each test phase"""
        self.test_results["test_phases"][phase] = {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        self.logger.info(f"Phase {phase}: {status}")
        
    def test_phase_1_youtube_search(self) -> bool:
        """Test Phase 1: YouTube Search and Filtering"""
        self.logger.info("=== PHASE 1: YouTube Search and Filtering ===")
        
        try:
            search_tool = YouTubeSearchTool()
            all_videos = []
            
            for keyword in self.test_config["search_keywords"]:
                self.logger.info(f"Searching for: {keyword}")
                
                # Perform search
                search_result = search_tool._run(
                    query=keyword,
                    max_results=self.test_config["max_videos_per_search"],
                    duration="medium"
                )
                
                self.logger.info(f"Search result type: {type(search_result)}")
                self.logger.info(f"Search result preview: {str(search_result)[:200]}...")
                
                # Parse search result - handle different return formats
                if search_result:
                    try:
                        if isinstance(search_result, str):
                            # Try to parse as JSON first
                            try:
                                search_data = json.loads(search_result)
                                videos = search_data.get("videos", [])
                            except json.JSONDecodeError:
                                # If not JSON, try to extract video info from string
                                videos = self._parse_search_result_string(search_result)
                        elif isinstance(search_result, list):
                            videos = search_result
                        elif isinstance(search_result, dict):
                            videos = search_result.get("videos", [])
                        else:
                            self.logger.warning(f"Unexpected search result type: {type(search_result)}")
                            videos = []
                        
                        self.logger.info(f"Parsed {len(videos)} videos from search result")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not parse search result for '{keyword}': {str(e)}")
                        self.logger.debug(f"Full search result: {search_result}")
                        continue
                    
                    # Filter by target channels and duration
                    filtered_videos = []
                    for video in videos:
                        if any(channel.lower() in video.get("channel", "").lower() 
                              for channel in self.test_config["target_channels"]):
                            # Parse duration and check if under 20 minutes
                            duration_str = video.get("duration", "0:00")
                            try:
                                parts = duration_str.split(":")
                                if len(parts) == 2:
                                    minutes = int(parts[0])
                                    if minutes <= self.test_config["max_duration_minutes"]:
                                        filtered_videos.append(video)
                            except ValueError:
                                continue
                    
                    all_videos.extend(filtered_videos)
                    self.logger.info(f"Found {len(filtered_videos)} suitable videos for '{keyword}'")
            
            # Remove duplicates
            unique_videos = []
            seen_urls = set()
            for video in all_videos:
                if video["url"] not in seen_urls:
                    unique_videos.append(video)
                    seen_urls.add(video["url"])
            
            self.logger.info(f"Total unique videos found: {len(unique_videos)}")
            
            # Save search results
            search_results_file = self.test_output_dir / "phase1_search_results.json"
            with open(search_results_file, 'w') as f:
                json.dump(unique_videos, f, indent=2)
            
            success = len(unique_videos) > 0
            self.log_phase_result("phase_1_youtube_search", success, {
                "videos_found": len(unique_videos),
                "search_keywords": self.test_config["search_keywords"],
                "results_file": str(search_results_file)
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {str(e)}")
            self.test_results["errors"].append(f"Phase 1: {str(e)}")
            self.log_phase_result("phase_1_youtube_search", False, {"error": str(e)})
            return False
    
    def _parse_search_result_string(self, result_string: str) -> List[Dict[str, Any]]:
        """Parse video information from search result string"""
        videos = []
        
        try:
            # Try to extract video URLs and titles from the string
            import re
            
            # Look for YouTube URLs
            url_pattern = r'https://www\.youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
            urls = re.findall(url_pattern, result_string)
            
            # Look for video titles (assuming they're in markdown format)
            title_pattern = r'\*\*(.+?)\*\*'
            titles = re.findall(title_pattern, result_string)
            
            # Look for channel names
            channel_pattern = r'Channel: (.+?)\n'
            channels = re.findall(channel_pattern, result_string)
            
            # Look for durations
            duration_pattern = r'Duration: (.+?)\n'
            durations = re.findall(duration_pattern, result_string)
            
            self.logger.info(f"Found {len(urls)} URLs, {len(titles)} titles, {len(channels)} channels, {len(durations)} durations")
            
            # Combine information
            for i, video_id in enumerate(urls):
                video_info = {
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': titles[i] if i < len(titles) else f'Video {i+1}',
                    'duration': durations[i] if i < len(durations) else 'unknown',
                    'channel': channels[i] if i < len(channels) else 'unknown'
                }
                videos.append(video_info)
                self.logger.debug(f"Parsed video: {video_info}")
            
        except Exception as e:
            self.logger.error(f"Error parsing search result string: {str(e)}")
            self.logger.debug(f"Result string: {result_string}")
        
        return videos
    
    def test_phase_2_download_and_transcribe(self) -> bool:
        """Test Phase 2: Download and Transcription"""
        self.logger.info("=== PHASE 2: Download and Transcription ===")
        
        try:
            # Load search results from Phase 1
            search_results_file = self.test_output_dir / "phase1_search_results.json"
            if not search_results_file.exists():
                raise FileNotFoundError("Phase 1 results not found")
            
            with open(search_results_file, 'r') as f:
                videos = json.load(f)
            
            if not videos:
                raise ValueError("No videos to process")
            
            # Take first 2 videos for testing
            test_videos = videos[:2]
            
            downloader = YouTubeDownloaderTool()
            transcriber = TranscriptionTool()
            
            processed_videos = []
            
            for video in test_videos:
                self.logger.info(f"Processing video: {video['title']}")
                
                # Download audio
                download_result = downloader._run(
                    url=video["url"],
                    output_dir=str(self.test_output_dir / "audio")
                )
                
                # Parse download result
                if download_result:
                    try:
                        if isinstance(download_result, str):
                            try:
                                download_data = json.loads(download_result)
                                audio_file = download_data.get("file_path") or download_data.get("output_file")
                            except json.JSONDecodeError:
                                # If not JSON, assume the string is the file path
                                audio_file = download_result
                        elif isinstance(download_result, dict):
                            audio_file = download_result.get("file_path") or download_result.get("output_file")
                        else:
                            self.logger.warning(f"Unexpected download result type: {type(download_result)}")
                            continue
                    except Exception as e:
                        self.logger.warning(f"Could not parse download result for '{video['title']}': {str(e)}")
                        continue
                    
                    # Transcribe audio
                    transcript_result = transcriber._run(
                        audio_file_path=audio_file,
                        output_dir=str(self.test_output_dir / "transcripts")
                    )
                    
                    # Parse transcription result
                    if transcript_result:
                        try:
                            if isinstance(transcript_result, str):
                                try:
                                    transcript_data = json.loads(transcript_result)
                                    transcript_file = transcript_data.get("transcript_file") or transcript_data.get("output_file")
                                except json.JSONDecodeError:
                                    # If not JSON, assume the string is the file path
                                    transcript_file = transcript_result
                            elif isinstance(transcript_result, dict):
                                transcript_file = transcript_result.get("transcript_file") or transcript_result.get("output_file")
                            else:
                                self.logger.warning(f"Unexpected transcript result type: {type(transcript_result)}")
                                continue
                            
                            if transcript_file:
                                processed_videos.append({
                                    "video": video,
                                    "audio_file": audio_file,
                                    "transcript_file": transcript_file
                                })
                        except Exception as e:
                            self.logger.warning(f"Could not parse transcript result for '{video['title']}': {str(e)}")
                            continue
                        
                        self.logger.info(f"Successfully processed: {video['title']}")
            
            # Save processing results
            processing_results_file = self.test_output_dir / "phase2_processing_results.json"
            with open(processing_results_file, 'w') as f:
                json.dump(processed_videos, f, indent=2)
            
            success = len(processed_videos) > 0
            self.log_phase_result("phase_2_download_transcribe", success, {
                "videos_processed": len(processed_videos),
                "results_file": str(processing_results_file)
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {str(e)}")
            self.test_results["errors"].append(f"Phase 2: {str(e)}")
            self.log_phase_result("phase_2_download_transcribe", False, {"error": str(e)})
            return False
    
    def test_phase_3_qlora_formatting(self) -> bool:
        """Test Phase 3: QLoRA Data Formatting"""
        self.logger.info("=== PHASE 3: QLoRA Data Formatting ===")
        
        try:
            # Load processing results from Phase 2
            processing_results_file = self.test_output_dir / "phase2_processing_results.json"
            if not processing_results_file.exists():
                raise FileNotFoundError("Phase 2 results not found")
            
            with open(processing_results_file, 'r') as f:
                processed_videos = json.load(f)
            
            if not processed_videos:
                raise ValueError("No processed videos to format")
            
            formatter = QLoRAFormatterTool()
            formatted_datasets = []
            
            for item in processed_videos:
                transcript_file = item["transcript_file"]
                
                # Format transcript for QLoRA training
                output_file = str(self.test_output_dir / "qlora_data" / f"formatted_{Path(transcript_file).stem}.json")
                format_result = formatter._run(
                    transcript_dir=str(Path(transcript_file).parent),
                    output_file=output_file
                )
                
                self.logger.info(f"QLoRA format result: {format_result}")
                
                # Handle different result formats
                if format_result:
                    if isinstance(format_result, dict):
                        success = format_result.get("success", False)
                        examples_count = format_result.get("examples_generated", 0)
                        files_processed = format_result.get("files_processed", 0)
                    else:
                        # If result is a string or other format, assume success if not empty
                        success = bool(format_result)
                        examples_count = 0
                        files_processed = 1
                    
                    if success:
                        # Check for the actual output file (formatter uses its own path)
                        actual_output_file = "/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json"
                        if Path(actual_output_file).exists():
                            formatted_datasets.append({
                                "original_video": item["video"]["title"],
                                "transcript_file": transcript_file,
                                "formatted_file": actual_output_file,
                                "stats": {
                                    "examples_generated": examples_count,
                                    "files_processed": files_processed
                                }
                            })
                            
                            self.logger.info(f"Formatted data for: {item['video']['title']} ({examples_count} examples)")
                        else:
                            # Check if the specified output file exists
                            if Path(output_file).exists():
                                formatted_datasets.append({
                                    "original_video": item["video"]["title"],
                                    "transcript_file": transcript_file,
                                    "formatted_file": output_file,
                                    "stats": {
                                        "examples_generated": examples_count,
                                        "files_processed": files_processed
                                    }
                                })
                                
                                self.logger.info(f"Formatted data for: {item['video']['title']} ({examples_count} examples)")
                            else:
                                self.logger.warning(f"QLoRA formatting succeeded but output file not found at expected locations")
                                self.logger.warning(f"Expected: {output_file}")
                                self.logger.warning(f"Also checked: {actual_output_file}")
                    else:
                        self.logger.warning(f"QLoRA formatting failed for: {item['video']['title']}")
                else:
                    self.logger.warning(f"No result from QLoRA formatter for: {item['video']['title']}")
            
            # Combine all formatted datasets
            if formatted_datasets:
                combined_data = []
                for dataset in formatted_datasets:
                    with open(dataset["formatted_file"], 'r') as f:
                        data = json.load(f)
                        combined_data.extend(data)
                
                # Save combined training data
                combined_file = self.test_output_dir / "qlora_data" / "combined_training_data.json"
                combined_file.parent.mkdir(exist_ok=True)
                with open(combined_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                
                self.logger.info(f"Combined training data: {len(combined_data)} examples")
            
            # Save formatting results
            formatting_results_file = self.test_output_dir / "phase3_formatting_results.json"
            with open(formatting_results_file, 'w') as f:
                json.dump(formatted_datasets, f, indent=2)
            
            success = len(formatted_datasets) > 0
            self.log_phase_result("phase_3_qlora_formatting", success, {
                "datasets_formatted": len(formatted_datasets),
                "total_examples": len(combined_data) if formatted_datasets else 0,
                "results_file": str(formatting_results_file)
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {str(e)}")
            self.test_results["errors"].append(f"Phase 3: {str(e)}")
            self.log_phase_result("phase_3_qlora_formatting", False, {"error": str(e)})
            return False
    
    def test_phase_4_qlora_training(self) -> bool:
        """Test Phase 4: QLoRA Training with Qwen 1.7B"""
        self.logger.info("=== PHASE 4: QLoRA Training ===")
        
        try:
            # Check if training data exists
            training_data_file = self.test_output_dir / "qlora_data" / "combined_training_data.json"
            if not training_data_file.exists():
                raise FileNotFoundError("Training data not found")
            
            trainer = AxolotlTrainerTool()
            
            # Prepare training configuration
            training_config = {
                "base_model": self.test_config["test_model"],
                "training_data": str(training_data_file),
                "output_dir": str(self.test_output_dir / "models" / "qlora_adapter"),
                "num_epochs": self.test_config["qlora_config"]["num_epochs"],
                "learning_rate": self.test_config["qlora_config"]["learning_rate"],
                "batch_size": self.test_config["qlora_config"]["batch_size"],
                "max_seq_length": self.test_config["qlora_config"]["max_seq_length"]
            }
            
            self.logger.info(f"Starting QLoRA training with config: {training_config}")
            
            # Start training
            training_result = trainer._run(
                training_data_path=training_config["training_data"],
                base_model=training_config["base_model"],
                output_dir=training_config["output_dir"],
                num_epochs=training_config["num_epochs"],
                learning_rate=training_config["learning_rate"],
                batch_size=training_config["batch_size"],
                max_seq_length=training_config["max_seq_length"]
            )
            
            # Handle different return types from AxolotlTrainerTool
            if isinstance(training_result, dict):
                success = training_result.get("success", False)
                result_to_save = training_result
            elif isinstance(training_result, str):
                # If it's a string, assume success if no error keywords
                success = "error" not in training_result.lower() and "failed" not in training_result.lower()
                result_to_save = {"message": training_result, "success": success}
            else:
                success = bool(training_result)
                result_to_save = {"result": str(training_result), "success": success}
            
            # Save training results
            training_results_file = self.test_output_dir / "phase4_training_results.json"
            with open(training_results_file, 'w') as f:
                json.dump(result_to_save, f, indent=2)
            
            self.log_phase_result("phase_4_qlora_training", success, {
                "training_config": training_config,
                "training_result": result_to_save,
                "results_file": str(training_results_file)
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Phase 4 failed: {str(e)}")
            self.test_results["errors"].append(f"Phase 4: {str(e)}")
            self.log_phase_result("phase_4_qlora_training", False, {"error": str(e)})
            return False
    
    def test_phase_5_model_validation(self) -> bool:
        """Test Phase 5: Model Validation and Testing"""
        self.logger.info("=== PHASE 5: Model Validation ===")
        
        try:
            # Check if trained model exists
            model_dir = self.test_output_dir / "models" / "qlora_adapter"
            if not model_dir.exists():
                raise FileNotFoundError("Trained model not found")
            
            validator = ModelValidatorTool()
            
            # Prepare validation test cases
            test_cases = [
                "How does Islam help with trauma healing?",
                "What does the Quran say about dealing with difficult times?",
                "How can prayer help with mental health?",
                "What is the Islamic perspective on seeking therapy?"
            ]
            
            validation_config = {
                "model_path": str(model_dir),
                "base_model": self.test_config["test_model"],
                "test_cases": test_cases,
                "islamic_content_validation": True
            }
            
            self.logger.info("Starting model validation...")
            
            # Run validation
            validation_result = validator._run(
                model_path=validation_config["model_path"],
                base_model_name=validation_config["base_model"],
                num_test_samples=len(validation_config["test_cases"]),
                include_islamic_accuracy=validation_config["islamic_content_validation"]
            )
            
            # Handle different return types from ModelValidatorTool
            if isinstance(validation_result, dict):
                success = validation_result.get("success", False)
                result_to_save = validation_result
            elif isinstance(validation_result, str):
                # If it's a string, assume success if no error keywords
                success = "error" not in validation_result.lower() and "failed" not in validation_result.lower()
                result_to_save = {"message": validation_result, "success": success}
            else:
                success = bool(validation_result)
                result_to_save = {"result": str(validation_result), "success": success}
            
            # Save validation results
            validation_results_file = self.test_output_dir / "phase5_validation_results.json"
            with open(validation_results_file, 'w') as f:
                json.dump(result_to_save, f, indent=2)
            
            self.log_phase_result("phase_5_model_validation", success, {
                "validation_config": validation_config,
                "validation_result": result_to_save,
                "results_file": str(validation_results_file)
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Phase 5 failed: {str(e)}")
            self.test_results["errors"].append(f"Phase 5: {str(e)}")
            self.log_phase_result("phase_5_model_validation", False, {"error": str(e)})
            return False
    
    def run_full_test_suite(self) -> bool:
        """Run the complete end-to-end test suite"""
        self.logger.info("🚀 Starting End-to-End Pipeline Test Suite")
        self.logger.info(f"Test configuration: {json.dumps(self.test_config, indent=2)}")
        
        # Test phases in sequence
        test_phases = [
            ("YouTube Search", self.test_phase_1_youtube_search),
            ("Download & Transcribe", self.test_phase_2_download_and_transcribe),
            ("QLoRA Formatting", self.test_phase_3_qlora_formatting),
            ("QLoRA Training", self.test_phase_4_qlora_training),
            ("Model Validation", self.test_phase_5_model_validation)
        ]
        
        overall_success = True
        
        for phase_name, phase_function in test_phases:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting {phase_name}")
            self.logger.info(f"{'='*60}")
            
            phase_success = phase_function()
            
            if not phase_success:
                self.logger.error(f"❌ {phase_name} FAILED - Stopping test suite")
                overall_success = False
                break
            else:
                self.logger.info(f"✅ {phase_name} PASSED")
        
        # Finalize test results
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["success"] = overall_success
        
        # Save final test report
        final_report_file = self.test_output_dir / "final_test_report.json"
        with open(final_report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        self.print_test_summary()
        
        return overall_success
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🏁 END-TO-END PIPELINE TEST SUMMARY")
        self.logger.info("="*80)
        
        overall_status = "✅ PASSED" if self.test_results["success"] else "❌ FAILED"
        self.logger.info(f"Overall Status: {overall_status}")
        
        self.logger.info(f"Start Time: {self.test_results['start_time']}")
        self.logger.info(f"End Time: {self.test_results['end_time']}")
        
        self.logger.info("\nPhase Results:")
        for phase, result in self.test_results["test_phases"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            self.logger.info(f"  {phase}: {status}")
        
        if self.test_results["errors"]:
            self.logger.info("\nErrors Encountered:")
            for error in self.test_results["errors"]:
                self.logger.error(f"  - {error}")
        
        if self.test_results["warnings"]:
            self.logger.info("\nWarnings:")
            for warning in self.test_results["warnings"]:
                self.logger.warning(f"  - {warning}")
        
        self.logger.info(f"\nDetailed results saved to: {self.test_output_dir}")
        self.logger.info("="*80)

def main():
    """Main test execution function"""
    print("🧪 Akhi CrewAI End-to-End Pipeline Test System")
    print("Testing complete workflow: YouTube Search → QLoRA Training")
    print("Target: Trauma content from Mufti Menk, Nouman Khan (<20 mins)")
    print("Model: Qwen 1.7B with QLoRA fine-tuning\n")
    
    # Create test instance
    test_system = EndToEndPipelineTest(
        test_output_dir=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    try:
        # Run complete test suite
        success = test_system.run_full_test_suite()
        
        if success:
            print("\n🎉 All tests passed! Pipeline is working correctly.")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test system error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()