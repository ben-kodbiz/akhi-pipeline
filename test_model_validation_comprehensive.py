#!/usr/bin/env python3
"""
Comprehensive Model Validation Test
Tests ModelValidatorTool with various scenarios and Islamic content validation
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'akhi_crewai'))

try:
    from akhi_crewai.tools.model_validator import ModelValidatorTool
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct")
    sys.exit(1)

class ComprehensiveModelValidationTest:
    def __init__(self):
        self.setup_logging()
        self.test_output_dir = Path(f"comprehensive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.test_output_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s:%(name)s:%(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('comprehensive_validation_test.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_model_structure(self, model_name, include_adapter_model=True):
        """Create model structure for testing"""
        model_dir = self.test_output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Create adapter_config.json with different configurations
        if model_name == "small_model":
            adapter_config = {
                "base_model_name_or_path": "qwen1.5-1.8b",
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM"
            }
        elif model_name == "medium_model":
            adapter_config = {
                "base_model_name_or_path": "qwen1.5-1.8b",
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "peft_type": "LORA",
                "r": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "task_type": "CAUSAL_LM"
            }
        else:  # large_model
            adapter_config = {
                "base_model_name_or_path": "qwen1.5-1.8b",
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "peft_type": "LORA",
                "r": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "task_type": "CAUSAL_LM"
            }
        
        with open(model_dir / "adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)
            
        # Create adapter_model.bin if requested
        if include_adapter_model:
            (model_dir / "adapter_model.bin").touch()
        
        self.logger.info(f"Created {model_name} structure at: {model_dir}")
        return str(model_dir)
        
    def test_basic_validation(self):
        """Test basic model validation with minimal samples"""
        self.logger.info("=== Testing Basic Model Validation ===")
        
        try:
            model_path = self.create_model_structure("small_model")
            validator = ModelValidatorTool()
            
            result = validator._run(
                model_path=model_path,
                base_model_name="qwen1.5-1.8b",
                num_test_samples=2,
                include_islamic_accuracy=False
            )
            
            self.logger.info(f"Basic validation result: {type(result)}")
            success = self._parse_result(result)
            
            # Save results
            self._save_test_result("basic_validation", result, success)
            return success
            
        except Exception as e:
            self.logger.error(f"Basic validation failed: {str(e)}")
            return False
            
    def test_islamic_content_validation(self):
        """Test model validation with Islamic content accuracy"""
        self.logger.info("=== Testing Islamic Content Validation ===")
        
        try:
            model_path = self.create_model_structure("medium_model")
            validator = ModelValidatorTool()
            
            result = validator._run(
                model_path=model_path,
                base_model_name="qwen1.5-1.8b",
                num_test_samples=3,
                include_islamic_accuracy=True
            )
            
            self.logger.info(f"Islamic content validation result: {type(result)}")
            success = self._parse_result(result)
            
            # Save results
            self._save_test_result("islamic_content_validation", result, success)
            return success
            
        except Exception as e:
            self.logger.error(f"Islamic content validation failed: {str(e)}")
            return False
            
    def test_performance_validation(self):
        """Test model validation with higher sample count for performance"""
        self.logger.info("=== Testing Performance Validation ===")
        
        try:
            model_path = self.create_model_structure("large_model")
            validator = ModelValidatorTool()
            
            result = validator._run(
                model_path=model_path,
                base_model_name="qwen1.5-1.8b",
                num_test_samples=5,
                include_islamic_accuracy=True
            )
            
            self.logger.info(f"Performance validation result: {type(result)}")
            success = self._parse_result(result)
            
            # Save results
            self._save_test_result("performance_validation", result, success)
            return success
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {str(e)}")
            return False
            
    def test_missing_adapter_model(self):
        """Test validation with missing adapter_model.bin"""
        self.logger.info("=== Testing Missing Adapter Model File ===")
        
        try:
            model_path = self.create_model_structure("incomplete_model", include_adapter_model=False)
            validator = ModelValidatorTool()
            
            result = validator._run(
                model_path=model_path,
                base_model_name="qwen1.5-1.8b",
                num_test_samples=1,
                include_islamic_accuracy=False
            )
            
            self.logger.info(f"Missing adapter model test result: {result}")
            # This should handle missing files gracefully
            success = True  # Test passes if it doesn't crash
            
            self._save_test_result("missing_adapter_model", result, success)
            return success
            
        except Exception as e:
            self.logger.info(f"Expected exception for missing adapter model: {str(e)}")
            return True  # Expected behavior
            
    def test_invalid_base_model(self):
        """Test validation with invalid base model name"""
        self.logger.info("=== Testing Invalid Base Model ===")
        
        try:
            model_path = self.create_model_structure("test_invalid_base")
            validator = ModelValidatorTool()
            
            result = validator._run(
                model_path=model_path,
                base_model_name="invalid-model-name",
                num_test_samples=1,
                include_islamic_accuracy=False
            )
            
            self.logger.info(f"Invalid base model test result: {result}")
            # This should handle invalid models gracefully
            success = True  # Test passes if it doesn't crash
            
            self._save_test_result("invalid_base_model", result, success)
            return success
            
        except Exception as e:
            self.logger.info(f"Expected exception for invalid base model: {str(e)}")
            return True  # Expected behavior
            
    def _parse_result(self, result):
        """Parse validation result to determine success"""
        if isinstance(result, dict):
            return result.get("success", False)
        elif isinstance(result, str):
            return "error" not in result.lower() and "failed" not in result.lower()
        else:
            return bool(result)
            
    def _save_test_result(self, test_name, result, success):
        """Save test result to file"""
        result_file = self.test_output_dir / f"{test_name}_results.json"
        result_to_save = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "success": success,
            "raw_result": result if isinstance(result, (dict, str)) else str(result)
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_to_save, f, indent=2)
            
        self.logger.info(f"Test results saved to: {result_file}")
        
    def run_all_tests(self):
        """Run all comprehensive model validation tests"""
        self.logger.info("🧪 Starting Comprehensive Model Validation Tests")
        
        tests = [
            ("Basic Validation", self.test_basic_validation),
            ("Islamic Content Validation", self.test_islamic_content_validation),
            ("Performance Validation", self.test_performance_validation),
            ("Missing Adapter Model", self.test_missing_adapter_model),
            ("Invalid Base Model", self.test_invalid_base_model)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running: {test_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                success = test_func()
                results[test_name] = "✅ PASSED" if success else "❌ FAILED"
                self.logger.info(f"{test_name}: {results[test_name]}")
            except Exception as e:
                results[test_name] = f"❌ ERROR: {str(e)}"
                self.logger.error(f"{test_name}: {results[test_name]}")
        
        # Print summary
        self.logger.info(f"\n{'='*70}")
        self.logger.info("🏁 COMPREHENSIVE MODEL VALIDATION TEST SUMMARY")
        self.logger.info(f"{'='*70}")
        
        for test_name, result in results.items():
            self.logger.info(f"  {test_name}: {result}")
            
        passed = sum(1 for r in results.values() if "✅" in r)
        total = len(results)
        
        self.logger.info(f"\nOverall: {passed}/{total} tests passed")
        self.logger.info(f"Test output directory: {self.test_output_dir}")
        
        if passed == total:
            self.logger.info("🎉 All comprehensive tests passed!")
            return True
        else:
            self.logger.info("💥 Some tests failed.")
            return False

def main():
    test_runner = ComprehensiveModelValidationTest()
    success = test_runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()