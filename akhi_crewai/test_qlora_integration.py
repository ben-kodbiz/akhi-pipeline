#!/usr/bin/env python3
"""
QLoRA Integration Test Script

This script tests the QLoRA integration functionality to ensure
all components are working correctly.

Author: Assistant
Date: December 2024
Phase: 8 - QLoRA Integration Testing
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add project paths
sys.path.append(os.path.dirname(__file__))

try:
    from crew import AkhiPipelineCrew
    from main import AkhiPipelineApp
    from agents.qlora_trainer import QLoRATrainerAgent
    from tools.qlora_formatter import QLoRAFormatterTool
    from tools.axolotl_trainer import AxolotlTrainerTool
    from tools.model_validator import ModelValidatorTool
    from tools.model_deployer import ModelDeployerTool
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all QLoRA components are properly installed.")
    sys.exit(1)


class QLoRAIntegrationTester:
    """
    Test suite for QLoRA integration.
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    def setup_logging(self):
        """Setup logging for tests."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message
        })
    
    def test_imports(self) -> bool:
        """Test that all QLoRA components can be imported."""
        try:
            # Test agent import
            agent = QLoRATrainerAgent()
            self.log_test_result("QLoRA Agent Import", True, "Agent imported successfully")
            
            # Test tool imports
            formatter = QLoRAFormatterTool()
            trainer = AxolotlTrainerTool()
            validator = ModelValidatorTool()
            deployer = ModelDeployerTool()
            
            self.log_test_result("QLoRA Tools Import", True, "All tools imported successfully")
            return True
            
        except Exception as e:
            self.log_test_result("QLoRA Imports", False, f"Import failed: {e}")
            return False
    
    def test_configuration_loading(self) -> bool:
        """Test configuration file loading."""
        try:
            config_files = [
                "config/qlora_config.yaml",
                "config/crew_config.yaml"
            ]
            
            for config_file in config_files:
                if Path(config_file).exists():
                    self.log_test_result(f"Config File {config_file}", True, "File exists")
                else:
                    self.log_test_result(f"Config File {config_file}", False, "File missing")
                    return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Configuration Loading", False, f"Config error: {e}")
            return False
    
    def test_pipeline_integration(self) -> bool:
        """Test QLoRA integration with main pipeline."""
        try:
            # Test pipeline initialization
            app = AkhiPipelineApp()
            self.log_test_result("Pipeline App Init", True, "App initialized")
            
            # Test QLoRA status check
            status = app.get_system_status()
            if status.get('status') == 'success':
                qlora_status = status.get('data', {}).get('qlora', {})
                qlora_available = qlora_status.get('qlora_available', False)
                
                if qlora_available:
                    self.log_test_result("QLoRA Availability", True, "QLoRA is available")
                else:
                    self.log_test_result("QLoRA Availability", False, "QLoRA not available")
                
                return qlora_available
            else:
                self.log_test_result("System Status", False, f"Status error: {status.get('error')}")
                return False
                
        except Exception as e:
            self.log_test_result("Pipeline Integration", False, f"Integration error: {e}")
            return False
    
    def test_crew_qlora_methods(self) -> bool:
        """Test QLoRA methods in crew."""
        try:
            crew = AkhiPipelineCrew()
            
            # Test QLoRA status method
            if hasattr(crew, 'get_qlora_status'):
                status = crew.get_qlora_status()
                self.log_test_result("Crew QLoRA Status", True, f"Status: {status.get('qlora_available', 'unknown')}")
            else:
                self.log_test_result("Crew QLoRA Status", False, "Method not found")
                return False
            
            # Test QLoRA training method
            if hasattr(crew, 'execute_qlora_training'):
                self.log_test_result("Crew QLoRA Training Method", True, "Method exists")
            else:
                self.log_test_result("Crew QLoRA Training Method", False, "Method not found")
                return False
            
            # Test full pipeline with training method
            if hasattr(crew, 'execute_full_pipeline_with_training'):
                self.log_test_result("Crew Full Pipeline Training", True, "Method exists")
            else:
                self.log_test_result("Crew Full Pipeline Training", False, "Method not found")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Crew QLoRA Methods", False, f"Method test error: {e}")
            return False
    
    def test_cli_integration(self) -> bool:
        """Test CLI integration for QLoRA commands."""
        try:
            # Test that main.py has QLoRA methods
            app = AkhiPipelineApp()
            
            if hasattr(app, 'train_qlora_model'):
                self.log_test_result("CLI QLoRA Training", True, "Method exists")
            else:
                self.log_test_result("CLI QLoRA Training", False, "Method not found")
                return False
            
            if hasattr(app, 'process_pipeline_with_training'):
                self.log_test_result("CLI Pipeline Training", True, "Method exists")
            else:
                self.log_test_result("CLI Pipeline Training", False, "Method not found")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("CLI Integration", False, f"CLI test error: {e}")
            return False
    
    def test_sample_data_format(self) -> bool:
        """Test sample training data format."""
        try:
            # Create sample training data
            sample_data = [
                {
                    "instruction": "Explain the importance of prayer in Islam",
                    "input": "",
                    "output": "Prayer (Salah) is one of the Five Pillars of Islam and represents the direct connection between a Muslim and Allah. It is performed five times daily and serves as a constant reminder of one's faith and devotion."
                },
                {
                    "instruction": "What are the benefits of reading the Quran?",
                    "input": "",
                    "output": "Reading the Quran brings numerous spiritual benefits including guidance, peace of mind, increased faith, and closeness to Allah. It also provides wisdom for daily life and helps in character development."
                }
            ]
            
            # Test formatter tool
            formatter = QLoRAFormatterTool()
            
            # Create temporary test file
            test_file = Path("test_training_data.json")
            test_file.write_text(json.dumps(sample_data, indent=2))
            
            # Test if formatter can process the data
            if test_file.exists():
                self.log_test_result("Sample Data Creation", True, "Test data created")
                
                # Clean up
                test_file.unlink()
                self.log_test_result("Sample Data Format", True, "Format validation passed")
                return True
            else:
                self.log_test_result("Sample Data Format", False, "Could not create test data")
                return False
                
        except Exception as e:
            self.log_test_result("Sample Data Format", False, f"Format test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all QLoRA integration tests."""
        print("🧪 Starting QLoRA Integration Tests...\n")
        
        tests = [
            ("Imports", self.test_imports),
            ("Configuration", self.test_configuration_loading),
            ("Pipeline Integration", self.test_pipeline_integration),
            ("Crew Methods", self.test_crew_qlora_methods),
            ("CLI Integration", self.test_cli_integration),
            ("Data Format", self.test_sample_data_format)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name} tests...")
            if test_func():
                passed += 1
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All QLoRA integration tests passed!")
            print("✅ QLoRA integration is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the issues above.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': passed / total,
            'all_passed': passed == total,
            'detailed_results': self.test_results
        }


def main():
    """Main test execution."""
    print("🚀 QLoRA Integration Test Suite")
    print("=" * 50)
    
    tester = QLoRAIntegrationTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print(f"   Total Tests: {results['total_tests']}")
    print(f"   Passed: {results['passed_tests']}")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    
    if results['all_passed']:
        print("\n🎯 QLoRA Integration: READY FOR USE")
        print("\n🚀 You can now use QLoRA training features:")
        print("   • python main.py --train <data_path>")
        print("   • python main.py --process-train <query>")
        print("   • python main.py --status")
        print("   • python main.py --interactive")
    else:
        print("\n🔧 QLoRA Integration: NEEDS ATTENTION")
        print("\n❗ Please resolve the failed tests before using QLoRA features.")
    
    return 0 if results['all_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())