#!/usr/bin/env python3
"""
Quick Training Script - Simple wrapper for RAG to Axolotl pipeline

This is a simplified interface for the most common use cases:
- Quick dataset preparation
- Quick training with default settings
- Status checking

Usage:
    python quick_train.py prepare    # Prepare dataset only
    python quick_train.py train      # Prepare and start training
    python quick_train.py status     # Check training status

Author: Akhi Data Builder Team
Date: January 2025
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def prepare_dataset():
    """Prepare dataset for training"""
    print("🔄 Preparing dataset from RAG documents...")
    
    script_dir = Path(__file__).parent
    auto_train_script = script_dir / "auto_train_from_rag.py"
    
    cmd = [sys.executable, str(auto_train_script), "--prepare-only", "--verbose"]
    
    if run_command(cmd):
        print("✅ Dataset preparation completed!")
        print("\n📁 Next steps:")
        print("   1. Review the prepared dataset")
        print("   2. Run 'python quick_train.py train' to start training")
        return True
    else:
        print("❌ Dataset preparation failed!")
        return False

def start_training():
    """Prepare dataset and start training"""
    print("🚀 Starting full training pipeline...")
    
    script_dir = Path(__file__).parent
    auto_train_script = script_dir / "auto_train_from_rag.py"
    
    cmd = [sys.executable, str(auto_train_script), "--train", "--verbose"]
    
    if run_command(cmd):
        print("✅ Training pipeline started!")
        print("\n🔍 Monitor training:")
        print("   - Check logs in the axolotl_ready directory")
        print("   - Use 'python quick_train.py status' to check progress")
        return True
    else:
        print("❌ Training pipeline failed!")
        return False

def check_status():
    """Check training status"""
    print("📊 Checking training status...")
    
    # Look for common training directories
    base_dir = Path(__file__).parent.parent.parent
    crewai_dir = base_dir / "akhi_crewai"
    
    # Check for axolotl directories
    axolotl_dirs = list(crewai_dir.glob("axolotl_ready*"))
    
    if not axolotl_dirs:
        print("❌ No training directories found.")
        print("   Run 'python quick_train.py prepare' first.")
        return False
    
    print(f"📁 Found {len(axolotl_dirs)} training directory(ies):")
    
    for axolotl_dir in axolotl_dirs:
        print(f"\n📂 Directory: {axolotl_dir}")
        
        # Check for training files
        config_file = axolotl_dir / "axolotl_config.yml"
        train_script = axolotl_dir / "train_model.sh"
        
        if config_file.exists():
            print("   ✅ Configuration file found")
        else:
            print("   ❌ Configuration file missing")
        
        if train_script.exists():
            print("   ✅ Training script found")
        else:
            print("   ❌ Training script missing")
        
        # Check for training outputs
        output_dir = axolotl_dir / "outputs"
        if output_dir.exists():
            checkpoints = list(output_dir.glob("checkpoint-*"))
            if checkpoints:
                print(f"   🎯 Found {len(checkpoints)} training checkpoint(s)")
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                print(f"   📈 Latest checkpoint: {latest_checkpoint.name}")
            else:
                print("   ⏳ No checkpoints found (training may be starting)")
        else:
            print("   ⏳ No output directory (training not started)")
        
        # Check for logs
        log_files = list(axolotl_dir.glob("*.log"))
        if log_files:
            print(f"   📝 Found {len(log_files)} log file(s)")
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   📄 Latest log: {latest_log.name}")
            
            # Show last few lines of latest log
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("   📋 Recent log entries:")
                        for line in lines[-3:]:
                            print(f"      {line.strip()}")
            except Exception as e:
                print(f"   ⚠️  Could not read log: {e}")
    
    return True

def show_help():
    """Show help information"""
    print("""
🤖 Quick Training Script - RAG to Axolotl Pipeline

Commands:
  prepare    Prepare dataset from RAG documents
  train      Prepare dataset and start training
  status     Check training status and progress
  help       Show this help message

Examples:
  python quick_train.py prepare
  python quick_train.py train
  python quick_train.py status

Workflow:
  1. Upload documents to RAG system
  2. Run 'prepare' to convert documents to training format
  3. Run 'train' to start fine-tuning
  4. Use 'status' to monitor progress

For advanced options, use auto_train_from_rag.py directly.
""")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'prepare':
        success = prepare_dataset()
    elif command == 'train':
        success = start_training()
    elif command == 'status':
        success = check_status()
    elif command in ['help', '--help', '-h']:
        show_help()
        success = True
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python quick_train.py help' for available commands.")
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()