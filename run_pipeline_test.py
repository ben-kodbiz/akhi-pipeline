#!/usr/bin/env python3
"""
Pipeline Test Runner
Simple script to execute end-to-end pipeline tests with proper setup and monitoring
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
from datetime import datetime

def check_environment():
    """Check if the environment is properly set up for testing"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if we're in the right directory
    if not Path("akhi_crewai").exists():
        print("❌ Please run from the project root directory")
        return False
    
    # Check for required config files
    required_files = [
        "test_end_to_end_pipeline.py",
        "test_config.yaml",
        "akhi_crewai/crew_config.yaml"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Required file missing: {file_path}")
            return False
    
    print("✅ Environment check passed")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "whisper",
        "yt-dlp",
        "crewai",
        "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def setup_test_environment():
    """Set up the test environment"""
    print("⚙️  Setting up test environment...")
    
    # Create necessary directories
    test_dirs = [
        "test_results",
        "test_results/audio",
        "test_results/transcripts",
        "test_results/qlora_data",
        "test_results/models"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables if needed
    if not os.getenv("YOUTUBE_API_KEY"):
        print("⚠️  YOUTUBE_API_KEY not set - YouTube search may be limited")
    
    print("✅ Test environment ready")
    return True

def run_test_phase(phase_name, test_script, args=None):
    """Run a specific test phase"""
    print(f"\n🚀 Running {phase_name}...")
    
    cmd = [sys.executable, test_script]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {phase_name} completed successfully")
            return True
        else:
            print(f"❌ {phase_name} failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {phase_name} timed out")
        return False
    except Exception as e:
        print(f"💥 {phase_name} error: {str(e)}")
        return False

def run_quick_test():
    """Run a quick test with minimal data"""
    print("⚡ Running quick test (1 video, minimal training)...")
    
    # Modify test config for quick run
    quick_config = {
        "max_videos_per_search": 1,
        "num_epochs": 1,
        "batch_size": 1,
        "max_seq_length": 512
    }
    
    return run_test_phase("Quick Test", "test_end_to_end_pipeline.py", ["--quick"])

def run_full_test():
    """Run the complete end-to-end test"""
    print("🔄 Running full end-to-end test...")
    return run_test_phase("Full Test", "test_end_to_end_pipeline.py")

def run_individual_phase(phase):
    """Run a specific test phase"""
    phase_map = {
        "search": "--phase=search",
        "download": "--phase=download",
        "format": "--phase=format",
        "train": "--phase=train",
        "validate": "--phase=validate"
    }
    
    if phase not in phase_map:
        print(f"❌ Unknown phase: {phase}")
        return False
    
    print(f"🎯 Running {phase} phase only...")
    return run_test_phase(f"{phase.title()} Phase", "test_end_to_end_pipeline.py", [phase_map[phase]])

def generate_test_report():
    """Generate a summary test report"""
    print("📊 Generating test report...")
    
    # Find the latest test results
    test_dirs = list(Path(".").glob("test_results_*"))
    if not test_dirs:
        print("❌ No test results found")
        return False
    
    latest_test_dir = max(test_dirs, key=lambda p: p.stat().st_mtime)
    report_file = latest_test_dir / "final_test_report.json"
    
    if not report_file.exists():
        print("❌ Test report not found")
        return False
    
    # Read and display summary
    import json
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    print(f"\n📋 Test Report Summary:")
    print(f"   Status: {'✅ PASSED' if report['success'] else '❌ FAILED'}")
    print(f"   Start: {report['start_time']}")
    print(f"   End: {report['end_time']}")
    
    print(f"\n📈 Phase Results:")
    for phase, result in report['test_phases'].items():
        status = '✅ PASSED' if result['success'] else '❌ FAILED'
        print(f"   {phase}: {status}")
    
    if report['errors']:
        print(f"\n❌ Errors:")
        for error in report['errors']:
            print(f"   - {error}")
    
    print(f"\n📁 Full results: {latest_test_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Akhi Pipeline Test Runner")
    parser.add_argument(
        "--mode", 
        choices=["quick", "full", "phase", "report"],
        default="full",
        help="Test mode to run"
    )
    parser.add_argument(
        "--phase",
        choices=["search", "download", "format", "train", "validate"],
        help="Specific phase to run (when mode=phase)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment and dependency checks"
    )
    
    args = parser.parse_args()
    
    print("🧪 Akhi CrewAI Pipeline Test Runner")
    print("=" * 50)
    
    # Environment checks
    if not args.skip_checks:
        if not check_environment():
            sys.exit(1)
        
        if not check_dependencies():
            sys.exit(1)
        
        if not setup_test_environment():
            sys.exit(1)
    
    # Run tests based on mode
    success = False
    
    if args.mode == "quick":
        success = run_quick_test()
    elif args.mode == "full":
        success = run_full_test()
    elif args.mode == "phase":
        if not args.phase:
            print("❌ --phase required when mode=phase")
            sys.exit(1)
        success = run_individual_phase(args.phase)
    elif args.mode == "report":
        success = generate_test_report()
    
    # Final status
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("💥 Test failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()