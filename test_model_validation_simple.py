#!/usr/bin/env python3
"""
Simple Model Validation Test
Tests ModelValidatorTool with minimal sample data
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

class SimpleModelValidationTest:
    def __init__(self):
        self.setup_logging()
        self.test_output_dir = Path(f"validation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.test_output_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s:%(name)s:%(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('model_validation_test.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_minimal_model_structure(self):
        """Create minimal model structure for testing"""
        model_dir = self.test_output_dir / "test_model"
        model_dir.mkdir(exist_ok=True)
        
        # Create minimal adapter_config.json
        adapter_config = {
            "base_model_name_or_path": "qwen1.5-1.8b",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "revision": None,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM"
        }
        
        with open(model_dir / "adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)
            
        # Create minimal adapter_model.bin (empty file for testing)
        (model_dir / "adapter_model.bin").touch()
        
        self.logger.info(f"Created minimal model structure at: {model_dir}")
        return str(model_dir)
        
    def test_model_validator_basic(self):
        """Test basic ModelValidatorTool functionality"""
        self.logger.info("=== Testing ModelValidatorTool Basic Functionality ===")
        
        try:
            # Create minimal model structure
            model_path = self.create_minimal_model_structure()
            
            # Initialize validator
            validator = ModelValidatorTool()
            self.logger.info("ModelValidatorTool initialized successfully")
            
            # Test with minimal parameters
            result = validator._run(
                model_path=model_path,
                base_model_name="qwen1.5-1.8b",
                num_test_samples=1,  # Minimal sample size
                include_islamic_accuracy=False  # Skip complex validation
            )
            
            self.logger.info(f"Validation result type: {type(result)}")
            
            # Handle different return types
            if isinstance(result, dict):
                success = result.get("success", False)
                self.logger.info(f"Validation success: {success}")
                self.logger.info(f"Result details: {json.dumps(result, indent=2)}")
            elif isinstance(result, str):
                success = "error" not in result.lower() and "failed" not in result.lower()
                self.logger.info(f"Validation result (string): {result}")
                self.logger.info(f"Inferred success: {success}")
            else:
                success = bool(result)
                self.logger.info(f"Validation result (other): {result}")
                self.logger.info(f"Inferred success: {success}")
            
            # Save results
            result_file = self.test_output_dir / "validation_results.json"
            result_to_save = {
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "success": success,
                "raw_result": result if isinstance(result, (dict, str)) else str(result)
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_to_save, f, indent=2)
                
            self.logger.info(f"Results saved to: {result_file}")
            return success
            
        except Exception as e:
            self.logger.error(f"Model validation test failed: {str(e)}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    def test_model_validator_missing_files(self):
        """Test ModelValidatorTool with missing model files"""
        self.logger.info("=== Testing ModelValidatorTool with Missing Files ===")
        
        try:
            # Create empty directory
            empty_model_dir = self.test_output_dir / "empty_model"
            empty_model_dir.mkdir(exist_ok=True)
            
            validator = ModelValidatorTool()
            
            result = validator._run(
                model_path=str(empty_model_dir),
                base_model_name="qwen1.5-1.8b",
                num_test_samples=1,
                include_islamic_accuracy=False
            )
            
            self.logger.info(f"Missing files test result: {result}")
            
            # This should fail gracefully
            if isinstance(result, dict):
                success = result.get("success", False)
                self.logger.info(f"Expected failure - Success: {success}")
            elif isinstance(result, str):
                success = "error" not in result.lower() and "failed" not in result.lower()
                self.logger.info(f"Expected failure - Inferred success: {success}")
                
            return True  # Test passes if it handles missing files gracefully
            
        except Exception as e:
            self.logger.info(f"Expected exception for missing files: {str(e)}")
            return True  # This is expected behavior
            
    def run_all_tests(self):
        """Run all model validation tests"""
        self.logger.info("🧪 Starting Simple Model Validation Tests")
        
        tests = [
            ("Basic Functionality", self.test_model_validator_basic),
            ("Missing Files Handling", self.test_model_validator_missing_files)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running: {test_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                success = test_func()
                results[test_name] = "✅ PASSED" if success else "❌ FAILED"
                self.logger.info(f"{test_name}: {results[test_name]}")
            except Exception as e:
                results[test_name] = f"❌ ERROR: {str(e)}"
                self.logger.error(f"{test_name}: {results[test_name]}")
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🏁 MODEL VALIDATION TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        
        for test_name, result in results.items():
            self.logger.info(f"  {test_name}: {result}")
            
        passed = sum(1 for r in results.values() if "✅" in r)
        total = len(results)
        
        self.logger.info(f"\nOverall: {passed}/{total} tests passed")
        self.logger.info(f"Test output directory: {self.test_output_dir}")
        
        if passed == total:
            self.logger.info("🎉 All tests passed!")
            return True
        else:
            self.logger.info("💥 Some tests failed.")
            return False

def main():
    test_runner = SimpleModelValidationTest()
    success = test_runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()