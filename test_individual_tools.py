#!/usr/bin/env python3
"""
Individual Tools Test Script
Tests each CrewAI tool individually to ensure they work correctly
before running the full end-to-end pipeline test.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'akhi_crewai'))

class IndividualToolsTest:
    """Test individual CrewAI tools"""
    
    def __init__(self, test_output_dir: str = "tool_test_results"):
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Test results
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tool_tests": {},
            "errors": [],
            "success": False
        }
    
    def setup_logging(self):
        """Setup logging for tool tests"""
        log_file = self.test_output_dir / f"tool_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Individual Tools Test System Initialized")
    
    def test_youtube_search_tool(self) -> bool:
        """Test YouTube Search Tool"""
        self.logger.info("=== Testing YouTube Search Tool ===")
        
        try:
            from akhi_crewai.tools.youtube_search import YouTubeSearchTool
            
            tool = YouTubeSearchTool()
            
            # Test search
            result = tool._run(
                query="mufti menk trauma",
                max_results=3,
                duration="medium"
            )
            
            self.logger.info(f"Search result type: {type(result)}")
            self.logger.info(f"Search result: {result[:200] if isinstance(result, str) else str(result)[:200]}...")
            
            success = result is not None
            self.test_results["tool_tests"]["youtube_search"] = {
                "success": success,
                "result_type": str(type(result)),
                "has_content": len(str(result)) > 0
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"YouTube Search Tool test failed: {str(e)}")
            self.test_results["errors"].append(f"YouTube Search: {str(e)}")
            self.test_results["tool_tests"]["youtube_search"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_youtube_downloader_tool(self) -> bool:
        """Test YouTube Downloader Tool"""
        self.logger.info("=== Testing YouTube Downloader Tool ===")
        
        try:
            from akhi_crewai.tools.youtube_downloader import YouTubeDownloaderTool
            
            tool = YouTubeDownloaderTool()
            
            # Test with a short video URL (you may need to replace with a valid URL)
            test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll as test
            
            result = tool._run(
                url=test_url,
                output_dir=str(self.test_output_dir / "test_audio")
            )
            
            self.logger.info(f"Download result type: {type(result)}")
            self.logger.info(f"Download result: {result[:200] if isinstance(result, str) else str(result)[:200]}...")
            
            success = result is not None
            self.test_results["tool_tests"]["youtube_downloader"] = {
                "success": success,
                "result_type": str(type(result)),
                "has_content": len(str(result)) > 0
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"YouTube Downloader Tool test failed: {str(e)}")
            self.test_results["errors"].append(f"YouTube Downloader: {str(e)}")
            self.test_results["tool_tests"]["youtube_downloader"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_transcription_tool(self) -> bool:
        """Test Transcription Tool"""
        self.logger.info("=== Testing Transcription Tool ===")
        
        try:
            from akhi_crewai.tools.transcriber import TranscriptionTool
            
            tool = TranscriptionTool()
            
            # Create a dummy audio file for testing (or skip if no audio available)
            test_audio_dir = self.test_output_dir / "test_audio"
            if not test_audio_dir.exists() or not list(test_audio_dir.glob("*.mp3")):
                self.logger.warning("No audio files available for transcription test - skipping")
                self.test_results["tool_tests"]["transcription"] = {
                    "success": True,
                    "skipped": True,
                    "reason": "No audio files available"
                }
                return True
            
            # Use first available audio file
            audio_file = list(test_audio_dir.glob("*.mp3"))[0]
            
            result = tool._run(
                audio_file_path=str(audio_file),
                output_dir=str(self.test_output_dir / "test_transcripts"),
                model_size="tiny",
                device="cpu"
            )
            
            self.logger.info(f"Transcription result type: {type(result)}")
            self.logger.info(f"Transcription result: {result[:200] if isinstance(result, str) else str(result)[:200]}...")
            
            success = result is not None
            self.test_results["tool_tests"]["transcription"] = {
                "success": success,
                "result_type": str(type(result)),
                "has_content": len(str(result)) > 0
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Transcription Tool test failed: {str(e)}")
            self.test_results["errors"].append(f"Transcription: {str(e)}")
            self.test_results["tool_tests"]["transcription"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_qlora_formatter_tool(self) -> bool:
        """Test QLoRA Formatter Tool"""
        self.logger.info("=== Testing QLoRA Formatter Tool ===")
        
        try:
            from akhi_crewai.tools.qlora_formatter import QLoRAFormatterTool
            
            tool = QLoRAFormatterTool()
            
            # Create dummy transcript for testing
            test_transcript_dir = self.test_output_dir / "test_transcripts"
            test_transcript_dir.mkdir(exist_ok=True)
            
            dummy_transcript = test_transcript_dir / "dummy_transcript.json"
            with open(dummy_transcript, 'w') as f:
                json.dump({
                    "text": "This is a test transcript about Islamic teachings and trauma healing. Allah provides guidance for those who seek it. Prayer and patience are important virtues in Islam.",
                    "segments": [
                        {"start": 0, "end": 10, "text": "This is a test transcript about Islamic teachings and trauma healing."},
                        {"start": 10, "end": 20, "text": "Allah provides guidance for those who seek it."},
                        {"start": 20, "end": 30, "text": "Prayer and patience are important virtues in Islam."}
                    ]
                }, f, indent=2)
            
            result = tool._run(
                transcript_dir=str(test_transcript_dir),
                output_file=str(self.test_output_dir / "test_qlora_data.json")
            )
            
            self.logger.info(f"QLoRA Formatter result type: {type(result)}")
            self.logger.info(f"QLoRA Formatter result: {result[:200] if isinstance(result, str) else str(result)[:200]}...")
            
            success = result is not None
            self.test_results["tool_tests"]["qlora_formatter"] = {
                "success": success,
                "result_type": str(type(result)),
                "has_content": len(str(result)) > 0
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"QLoRA Formatter Tool test failed: {str(e)}")
            self.test_results["errors"].append(f"QLoRA Formatter: {str(e)}")
            self.test_results["tool_tests"]["qlora_formatter"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_axolotl_trainer_tool(self) -> bool:
        """Test Axolotl Trainer Tool (configuration only)"""
        self.logger.info("=== Testing Axolotl Trainer Tool (Config Only) ===")
        
        try:
            from akhi_crewai.tools.axolotl_trainer import AxolotlTrainerTool
            
            tool = AxolotlTrainerTool()
            
            # Test configuration creation only (not actual training)
            config_path = tool._create_axolotl_config(
                training_data_path=str(self.test_output_dir / "test_qlora_data.json"),
                base_model="microsoft/DialoGPT-small",  # Smaller model for testing
                output_dir=str(self.test_output_dir / "test_model"),
                config_file=str(self.test_output_dir / "test_axolotl_config.yaml")
            )
            
            self.logger.info(f"Axolotl config created: {config_path}")
            
            success = Path(config_path).exists()
            self.test_results["tool_tests"]["axolotl_trainer"] = {
                "success": success,
                "config_created": success,
                "config_path": config_path,
                "note": "Only tested configuration creation, not actual training"
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Axolotl Trainer Tool test failed: {str(e)}")
            self.test_results["errors"].append(f"Axolotl Trainer: {str(e)}")
            self.test_results["tool_tests"]["axolotl_trainer"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_model_validator_tool(self) -> bool:
        """Test Model Validator Tool (basic initialization)"""
        self.logger.info("=== Testing Model Validator Tool (Init Only) ===")
        
        try:
            from akhi_crewai.tools.model_validator import ModelValidatorTool
            
            tool = ModelValidatorTool()
            
            # Test basic initialization and configuration loading
            config = tool._load_config()
            keywords = tool._load_islamic_keywords()
            prompts = tool._load_test_prompts()
            
            self.logger.info(f"Model Validator initialized successfully")
            self.logger.info(f"Config loaded: {len(config)} items")
            self.logger.info(f"Islamic keywords: {len(keywords)} items")
            self.logger.info(f"Test prompts: {len(prompts)} items")
            
            success = True
            self.test_results["tool_tests"]["model_validator"] = {
                "success": success,
                "config_items": len(config),
                "keywords_count": len(keywords),
                "prompts_count": len(prompts),
                "note": "Only tested initialization, not actual model validation"
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Model Validator Tool test failed: {str(e)}")
            self.test_results["errors"].append(f"Model Validator: {str(e)}")
            self.test_results["tool_tests"]["model_validator"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def run_all_tool_tests(self) -> bool:
        """Run all individual tool tests"""
        self.logger.info("🧪 Starting Individual Tools Test Suite")
        
        # Define test functions
        tool_tests = [
            ("YouTube Search", self.test_youtube_search_tool),
            ("YouTube Downloader", self.test_youtube_downloader_tool),
            ("Transcription", self.test_transcription_tool),
            ("QLoRA Formatter", self.test_qlora_formatter_tool),
            ("Axolotl Trainer", self.test_axolotl_trainer_tool),
            ("Model Validator", self.test_model_validator_tool)
        ]
        
        overall_success = True
        
        for tool_name, test_function in tool_tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Testing {tool_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                test_success = test_function()
                
                if test_success:
                    self.logger.info(f"✅ {tool_name} test PASSED")
                else:
                    self.logger.error(f"❌ {tool_name} test FAILED")
                    overall_success = False
                    
            except Exception as e:
                self.logger.error(f"💥 {tool_name} test ERROR: {str(e)}")
                self.test_results["errors"].append(f"{tool_name}: {str(e)}")
                overall_success = False
        
        # Finalize results
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["success"] = overall_success
        
        # Save results
        results_file = self.test_output_dir / "tool_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        self.print_test_summary()
        
        return overall_success
    
    def print_test_summary(self):
        """Print test summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🏁 INDIVIDUAL TOOLS TEST SUMMARY")
        self.logger.info("="*60)
        
        overall_status = "✅ PASSED" if self.test_results["success"] else "❌ FAILED"
        self.logger.info(f"Overall Status: {overall_status}")
        
        self.logger.info("\nTool Test Results:")
        for tool_name, result in self.test_results["tool_tests"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            if result.get("skipped"):
                status = "⏭️  SKIPPED"
            self.logger.info(f"  {tool_name}: {status}")
            if "note" in result:
                self.logger.info(f"    Note: {result['note']}")
        
        if self.test_results["errors"]:
            self.logger.info("\nErrors:")
            for error in self.test_results["errors"]:
                self.logger.error(f"  - {error}")
        
        self.logger.info(f"\nResults saved to: {self.test_output_dir}")
        self.logger.info("="*60)

def main():
    """Main test execution"""
    print("🔧 Akhi CrewAI Individual Tools Test")
    print("Testing each tool individually before full pipeline test\n")
    
    # Create test instance
    test_system = IndividualToolsTest(
        test_output_dir=f"tool_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    try:
        # Run all tool tests
        success = test_system.run_all_tool_tests()
        
        if success:
            print("\n🎉 All tool tests passed! Ready for end-to-end testing.")
            sys.exit(0)
        else:
            print("\n⚠️  Some tool tests failed. Check logs for details.")
            print("Fix issues before running end-to-end tests.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test system error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()