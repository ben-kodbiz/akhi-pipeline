#!/usr/bin/env python3
"""
Phase 7: Test Runner and Report Generator

Comprehensive test runner for Phase 7: Testing & Validation.
Executes all test suites and generates detailed reports.

Author: Assistant
Date: December 2024
Phase: 7 - Testing & Validation
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project paths
sys.path.append(os.path.dirname(__file__))


class Phase7TestRunner:
    """Comprehensive test runner for Phase 7 validation."""
    
    def __init__(self, output_dir="test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '7 - Testing & Validation',
            'test_suites': {},
            'summary': {},
            'recommendations': []
        }
    
    def run_test_suite(self, test_file: str, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run pytest with JSON output
            cmd = [
                'python', '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--json-report',
                f'--json-report-file={self.output_dir}/{suite_name.lower().replace(" ", "_")}_report.json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            duration = time.time() - start_time
            
            # Parse results
            suite_result = {
                'name': suite_name,
                'file': test_file,
                'duration': duration,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'status': 'PASSED' if result.returncode == 0 else 'FAILED'
            }
            
            # Try to load JSON report if available
            json_report_path = self.output_dir / f"{suite_name.lower().replace(' ', '_')}_report.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                        suite_result['detailed_results'] = json_report
                except Exception as e:
                    suite_result['json_parse_error'] = str(e)
            
            print(f"Status: {suite_result['status']}")
            print(f"Duration: {duration:.2f}s")
            
            if result.returncode != 0:
                print(f"Error Output: {result.stderr[:500]}...")
            
            return suite_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = {
                'name': suite_name,
                'file': test_file,
                'duration': duration,
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"ERROR: {str(e)}")
            return error_result
    
    def run_unit_tests(self):
        """Run existing unit tests for tools."""
        print("\n" + "="*80)
        print("PHASE 7: UNIT TESTING")
        print("="*80)
        
        tools_dir = Path("tools")
        if not tools_dir.exists():
            print("Tools directory not found. Skipping unit tests.")
            return
        
        # Find all test files in tools directory
        test_files = list(tools_dir.glob("test_*.py"))
        
        unit_test_results = []
        
        for test_file in test_files:
            tool_name = test_file.stem.replace('test_', '')
            suite_result = self.run_test_suite(
                str(test_file),
                f"Unit Tests - {tool_name.title()}"
            )
            unit_test_results.append(suite_result)
        
        self.results['test_suites']['unit_tests'] = unit_test_results
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n" + "="*80)
        print("PHASE 7: INTEGRATION TESTING")
        print("="*80)
        
        integration_file = "test_phase7_integration.py"
        if not Path(integration_file).exists():
            print(f"Integration test file {integration_file} not found.")
            return
        
        suite_result = self.run_test_suite(
            integration_file,
            "Integration Tests"
        )
        
        self.results['test_suites']['integration_tests'] = [suite_result]
    
    def run_quality_assurance_tests(self):
        """Run quality assurance tests."""
        print("\n" + "="*80)
        print("PHASE 7: QUALITY ASSURANCE")
        print("="*80)
        
        quality_file = "test_phase7_quality.py"
        if not Path(quality_file).exists():
            print(f"Quality test file {quality_file} not found.")
            return
        
        suite_result = self.run_test_suite(
            quality_file,
            "Quality Assurance Tests"
        )
        
        self.results['test_suites']['quality_assurance'] = [suite_result]
    
    def run_performance_tests(self):
        """Run performance benchmarking tests."""
        print("\n" + "="*80)
        print("PHASE 7: PERFORMANCE BENCHMARKING")
        print("="*80)
        
        performance_file = "test_phase7_performance.py"
        if not Path(performance_file).exists():
            print(f"Performance test file {performance_file} not found.")
            return
        
        suite_result = self.run_test_suite(
            performance_file,
            "Performance Benchmarking"
        )
        
        self.results['test_suites']['performance_tests'] = [suite_result]
    
    def check_system_status(self):
        """Check overall system status."""
        print("\n" + "="*80)
        print("SYSTEM STATUS CHECK")
        print("="*80)
        
        try:
            # Check main.py status
            result = subprocess.run(
                ['python', 'main.py', '--status'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            status_info = {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'status': 'HEALTHY' if result.returncode == 0 else 'UNHEALTHY'
            }
            
            print(f"System Status: {status_info['status']}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            
            self.results['system_status'] = status_info
            
        except Exception as e:
            print(f"Error checking system status: {e}")
            self.results['system_status'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def generate_summary(self):
        """Generate test summary and recommendations."""
        print("\n" + "="*80)
        print("GENERATING SUMMARY")
        print("="*80)
        
        total_suites = 0
        passed_suites = 0
        failed_suites = 0
        error_suites = 0
        total_duration = 0
        
        # Count results from all test categories
        for category, suites in self.results['test_suites'].items():
            if isinstance(suites, list):
                for suite in suites:
                    total_suites += 1
                    total_duration += suite.get('duration', 0)
                    
                    status = suite.get('status', 'UNKNOWN')
                    if status == 'PASSED':
                        passed_suites += 1
                    elif status == 'FAILED':
                        failed_suites += 1
                    else:
                        error_suites += 1
        
        # Calculate percentages
        pass_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0
        
        summary = {
            'total_test_suites': total_suites,
            'passed_suites': passed_suites,
            'failed_suites': failed_suites,
            'error_suites': error_suites,
            'pass_rate_percent': round(pass_rate, 1),
            'total_duration_seconds': round(total_duration, 2),
            'system_status': self.results.get('system_status', {}).get('status', 'UNKNOWN')
        }
        
        self.results['summary'] = summary
        
        # Generate recommendations
        recommendations = []
        
        if failed_suites > 0:
            recommendations.append("❌ Address failing test suites before proceeding to production")
        
        if error_suites > 0:
            recommendations.append("⚠️ Investigate test execution errors")
        
        if pass_rate < 80:
            recommendations.append("📈 Improve test pass rate to at least 80% before deployment")
        
        if summary['system_status'] != 'HEALTHY':
            recommendations.append("🔧 Fix system status issues identified in status check")
        
        if pass_rate >= 90 and failed_suites == 0:
            recommendations.append("✅ System is ready for production deployment")
        elif pass_rate >= 80:
            recommendations.append("⚡ System is ready for staging environment testing")
        else:
            recommendations.append("🚧 System requires additional development before deployment")
        
        # Add specific recommendations based on test categories
        unit_tests = self.results['test_suites'].get('unit_tests', [])
        if any(suite.get('status') == 'FAILED' for suite in unit_tests):
            recommendations.append("🔨 Fix failing unit tests for individual tools")
        
        integration_tests = self.results['test_suites'].get('integration_tests', [])
        if any(suite.get('status') == 'FAILED' for suite in integration_tests):
            recommendations.append("🔗 Address integration test failures for end-to-end workflows")
        
        quality_tests = self.results['test_suites'].get('quality_assurance', [])
        if any(suite.get('status') == 'FAILED' for suite in quality_tests):
            recommendations.append("📚 Improve Islamic content accuracy and answer quality")
        
        performance_tests = self.results['test_suites'].get('performance_tests', [])
        if any(suite.get('status') == 'FAILED' for suite in performance_tests):
            recommendations.append("⚡ Optimize system performance and resource usage")
        
        self.results['recommendations'] = recommendations
        
        # Print summary
        print(f"\nTest Summary:")
        print(f"  Total Test Suites: {total_suites}")
        print(f"  Passed: {passed_suites}")
        print(f"  Failed: {failed_suites}")
        print(f"  Errors: {error_suites}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        print(f"  Total Duration: {total_duration:.2f}s")
        print(f"  System Status: {summary['system_status']}")
        
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")
    
    def save_report(self):
        """Save comprehensive test report."""
        report_file = self.output_dir / f"phase7_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Also save a human-readable summary
        summary_file = self.output_dir / f"phase7_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AKHI CREWAI PIPELINE - PHASE 7 TEST REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n")
            f.write(f"Phase: {self.results['phase']}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*20 + "\n")
            summary = self.results['summary']
            f.write(f"Total Test Suites: {summary['total_test_suites']}\n")
            f.write(f"Passed: {summary['passed_suites']}\n")
            f.write(f"Failed: {summary['failed_suites']}\n")
            f.write(f"Errors: {summary['error_suites']}\n")
            f.write(f"Pass Rate: {summary['pass_rate_percent']}%\n")
            f.write(f"Duration: {summary['total_duration_seconds']}s\n")
            f.write(f"System Status: {summary['system_status']}\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-"*20 + "\n")
            for rec in self.results['recommendations']:
                f.write(f"{rec}\n")
            
            f.write("\nDETAILED RESULTS\n")
            f.write("-"*20 + "\n")
            for category, suites in self.results['test_suites'].items():
                f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                if isinstance(suites, list):
                    for suite in suites:
                        f.write(f"  {suite['name']}: {suite['status']} ({suite.get('duration', 0):.2f}s)\n")
        
        print(f"Summary report saved to: {summary_file}")
    
    def run_all_tests(self):
        """Run all Phase 7 test suites."""
        print("\n" + "="*80)
        print("AKHI CREWAI PIPELINE - PHASE 7: TESTING & VALIDATION")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check system status first
        self.check_system_status()
        
        # Run all test categories
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_quality_assurance_tests()
        self.run_performance_tests()
        
        # Generate summary and save reports
        self.generate_summary()
        self.save_report()
        
        print("\n" + "="*80)
        print("PHASE 7 TESTING COMPLETE")
        print("="*80)
        
        return self.results


def main():
    """Main entry point for Phase 7 testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 7: Testing & Validation Runner")
    parser.add_argument('--output-dir', default='test_reports', help='Output directory for reports')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--quality-only', action='store_true', help='Run only quality assurance tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    
    args = parser.parse_args()
    
    runner = Phase7TestRunner(args.output_dir)
    
    if args.unit_only:
        runner.check_system_status()
        runner.run_unit_tests()
    elif args.integration_only:
        runner.check_system_status()
        runner.run_integration_tests()
    elif args.quality_only:
        runner.check_system_status()
        runner.run_quality_assurance_tests()
    elif args.performance_only:
        runner.check_system_status()
        runner.run_performance_tests()
    else:
        runner.run_all_tests()
    
    if not (args.unit_only or args.integration_only or args.quality_only or args.performance_only):
        runner.generate_summary()
        runner.save_report()


if __name__ == "__main__":
    main()