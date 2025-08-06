#!/usr/bin/env python3
"""
RAG to Axolotl Troubleshooting Script
Automated diagnostic tool for common issues

Usage:
    python troubleshoot.py              # Run all checks
    python troubleshoot.py --check gpu  # Check specific component
    python troubleshoot.py --fix        # Attempt automatic fixes

Author: Akhi Data Builder Team
Date: January 2025
"""

import os
import sys
import subprocess
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

class TroubleshootingTool:
    """Automated troubleshooting for RAG to Axolotl pipeline"""
    
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Auto-detect base directory
            current = Path.cwd()
            while current != current.parent:
                if (current / 'RAG').exists() and (current / 'akhi_crewai').exists():
                    self.base_dir = current
                    break
                current = current.parent
            else:
                self.base_dir = Path('/data/work/dev/akhi_data_builder')
        
        self.rag_dir = self.base_dir / 'RAG'
        self.axolotl_dir = self.base_dir / 'akhi_crewai' / 'axolotl_ready'
        self.issues = []
        self.fixes_applied = []
    
    def run_command(self, cmd: str) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        checks = {}
        
        # Check Python version
        python_version = sys.version_info
        checks['python_version'] = python_version >= (3, 8)
        if not checks['python_version']:
            self.issues.append(f"Python version {python_version.major}.{python_version.minor} < 3.8")
        
        # Check GPU availability
        exit_code, stdout, stderr = self.run_command("nvidia-smi")
        checks['gpu_available'] = exit_code == 0
        if not checks['gpu_available']:
            self.issues.append("NVIDIA GPU not detected or drivers not installed")
        
        # Check CUDA
        exit_code, stdout, stderr = self.run_command("nvcc --version")
        checks['cuda_available'] = exit_code == 0
        if not checks['cuda_available']:
            self.issues.append("CUDA toolkit not installed")
        
        # Check disk space
        exit_code, stdout, stderr = self.run_command(f"df -h {self.base_dir}")
        if exit_code == 0:
            lines = stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    available = parts[3]
                    # Extract numeric value (remove G, M, etc.)
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)([GM])', available)
                    if match:
                        value, unit = match.groups()
                        gb_available = float(value) if unit == 'G' else float(value) / 1024
                        checks['disk_space'] = gb_available >= 10
                        if not checks['disk_space']:
                            self.issues.append(f"Insufficient disk space: {available} available, need 10GB+")
                    else:
                        checks['disk_space'] = True
                else:
                    checks['disk_space'] = True
            else:
                checks['disk_space'] = True
        else:
            checks['disk_space'] = True
        
        # Check memory
        exit_code, stdout, stderr = self.run_command("free -h")
        if exit_code == 0:
            lines = stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 2:
                    total_mem = parts[1]
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)([GM])', total_mem)
                    if match:
                        value, unit = match.groups()
                        gb_memory = float(value) if unit == 'G' else float(value) / 1024
                        checks['memory'] = gb_memory >= 8
                        if not checks['memory']:
                            self.issues.append(f"Insufficient memory: {total_mem}, need 8GB+")
                    else:
                        checks['memory'] = True
                else:
                    checks['memory'] = True
            else:
                checks['memory'] = True
        else:
            checks['memory'] = True
        
        return checks
    
    def check_python_dependencies(self) -> Dict[str, bool]:
        """Check Python package dependencies"""
        print("🔍 Checking Python dependencies...")
        checks = {}
        
        required_packages = [
            'torch', 'transformers', 'peft', 'datasets', 'accelerate',
            'yaml', 'numpy', 'pandas'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                checks[package] = True
            except ImportError:
                checks[package] = False
                self.issues.append(f"Missing Python package: {package}")
        
        # Check PyTorch CUDA support
        try:
            import torch
            checks['torch_cuda'] = torch.cuda.is_available()
            if not checks['torch_cuda']:
                self.issues.append("PyTorch CUDA support not available")
        except ImportError:
            checks['torch_cuda'] = False
        
        return checks
    
    def check_directory_structure(self) -> Dict[str, bool]:
        """Check directory structure and required files"""
        print("🔍 Checking directory structure...")
        checks = {}
        
        # Check base directories
        checks['base_dir'] = self.base_dir.exists()
        checks['rag_dir'] = self.rag_dir.exists()
        checks['scripts_dir'] = (self.rag_dir / 'scripts').exists()
        
        if not checks['base_dir']:
            self.issues.append(f"Base directory not found: {self.base_dir}")
        if not checks['rag_dir']:
            self.issues.append(f"RAG directory not found: {self.rag_dir}")
        if not checks['scripts_dir']:
            self.issues.append(f"Scripts directory not found: {self.rag_dir / 'scripts'}")
        
        # Check required scripts
        required_scripts = [
            'api_server.py', 'rag_to_axolotl_bridge.py', 
            'auto_train_from_rag.py', 'quick_train.py'
        ]
        
        for script in required_scripts:
            script_path = self.rag_dir / 'scripts' / script
            checks[f'script_{script}'] = script_path.exists()
            if not checks[f'script_{script}']:
                self.issues.append(f"Required script not found: {script}")
        
        return checks
    
    def check_rag_configuration(self) -> Dict[str, bool]:
        """Check RAG system configuration"""
        print("🔍 Checking RAG configuration...")
        checks = {}
        
        config_path = self.rag_dir / 'config.yaml'
        checks['config_exists'] = config_path.exists()
        
        if checks['config_exists']:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                checks['config_valid'] = isinstance(config, dict)
                
                # Check for documents
                data_section = config.get('data', {})
                sources = data_section.get('sources', [])
                checks['has_documents'] = len(sources) > 0
                
                if not checks['has_documents']:
                    self.issues.append("No documents found in RAG configuration")
                
            except Exception as e:
                checks['config_valid'] = False
                self.issues.append(f"Invalid RAG configuration: {e}")
        else:
            checks['config_valid'] = False
            self.issues.append("RAG config.yaml not found")
        
        # Check uploads directory
        uploads_dir = self.rag_dir / 'uploads'
        checks['uploads_dir'] = uploads_dir.exists()
        if checks['uploads_dir']:
            uploaded_files = list(uploads_dir.glob('*'))
            checks['has_uploads'] = len(uploaded_files) > 0
            if not checks['has_uploads']:
                self.issues.append("No files found in uploads directory")
        else:
            checks['has_uploads'] = False
            self.issues.append("Uploads directory not found")
        
        return checks
    
    def check_training_data(self) -> Dict[str, bool]:
        """Check training data generation"""
        print("🔍 Checking training data...")
        checks = {}
        
        training_data_dir = self.rag_dir / 'training_data'
        checks['training_dir'] = training_data_dir.exists()
        
        if checks['training_dir']:
            jsonl_files = list(training_data_dir.glob('*.jsonl'))
            checks['has_training_data'] = len(jsonl_files) > 0
            
            if checks['has_training_data']:
                # Check file size
                for jsonl_file in jsonl_files:
                    file_size = jsonl_file.stat().st_size
                    checks[f'{jsonl_file.name}_size'] = file_size > 1000  # At least 1KB
                    if not checks[f'{jsonl_file.name}_size']:
                        self.issues.append(f"Training data file too small: {jsonl_file.name}")
            else:
                self.issues.append("No training data files found")
        else:
            checks['has_training_data'] = False
            self.issues.append("Training data directory not found")
        
        return checks
    
    def check_axolotl_setup(self) -> Dict[str, bool]:
        """Check Axolotl training setup"""
        print("🔍 Checking Axolotl setup...")
        checks = {}
        
        checks['axolotl_dir'] = self.axolotl_dir.exists()
        
        if checks['axolotl_dir']:
            # Check required files
            required_files = [
                'axolotl_config.yml', 'train_dataset.jsonl', 
                'val_dataset.jsonl', 'train_model.sh'
            ]
            
            for file_name in required_files:
                file_path = self.axolotl_dir / file_name
                checks[f'axolotl_{file_name}'] = file_path.exists()
                if not checks[f'axolotl_{file_name}']:
                    self.issues.append(f"Axolotl file not found: {file_name}")
            
            # Check training script permissions
            train_script = self.axolotl_dir / 'train_model.sh'
            if train_script.exists():
                checks['train_script_executable'] = os.access(train_script, os.X_OK)
                if not checks['train_script_executable']:
                    self.issues.append("Training script not executable")
        else:
            self.issues.append("Axolotl directory not found")
        
        return checks
    
    def check_training_status(self) -> Dict[str, bool]:
        """Check current training status"""
        print("🔍 Checking training status...")
        checks = {}
        
        # Check for running training processes
        exit_code, stdout, stderr = self.run_command("ps aux | grep train_model.sh | grep -v grep")
        checks['training_running'] = exit_code == 0 and stdout.strip()
        
        # Check for model checkpoints
        if self.axolotl_dir.exists():
            models_dir = self.axolotl_dir / 'models'
            checks['models_dir'] = models_dir.exists()
            
            if checks['models_dir']:
                checkpoints = list(models_dir.glob('checkpoint-*'))
                checks['has_checkpoints'] = len(checkpoints) > 0
                if checks['has_checkpoints']:
                    print(f"  ✅ Found {len(checkpoints)} training checkpoints")
            else:
                checks['has_checkpoints'] = False
        else:
            checks['models_dir'] = False
            checks['has_checkpoints'] = False
        
        return checks
    
    def attempt_fixes(self) -> List[str]:
        """Attempt automatic fixes for common issues"""
        print("🔧 Attempting automatic fixes...")
        fixes = []
        
        # Fix missing directories
        if not (self.rag_dir / 'uploads').exists():
            (self.rag_dir / 'uploads').mkdir(parents=True, exist_ok=True)
            fixes.append("Created uploads directory")
        
        if not (self.rag_dir / 'training_data').exists():
            (self.rag_dir / 'training_data').mkdir(parents=True, exist_ok=True)
            fixes.append("Created training_data directory")
        
        # Fix training script permissions
        train_script = self.axolotl_dir / 'train_model.sh'
        if train_script.exists() and not os.access(train_script, os.X_OK):
            os.chmod(train_script, 0o755)
            fixes.append("Made training script executable")
        
        # Create basic config if missing
        config_path = self.rag_dir / 'config.yaml'
        if not config_path.exists():
            basic_config = {
                'data': {'sources': []},
                'retrieval': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(basic_config, f, default_flow_style=False)
            fixes.append("Created basic RAG configuration")
        
        self.fixes_applied.extend(fixes)
        return fixes
    
    def generate_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        report = []
        report.append("\n" + "="*60)
        report.append("🔍 RAG TO AXOLOTL DIAGNOSTIC REPORT")
        report.append("="*60)
        
        # Run all checks
        all_checks = {}
        all_checks.update(self.check_system_requirements())
        all_checks.update(self.check_python_dependencies())
        all_checks.update(self.check_directory_structure())
        all_checks.update(self.check_rag_configuration())
        all_checks.update(self.check_training_data())
        all_checks.update(self.check_axolotl_setup())
        all_checks.update(self.check_training_status())
        
        # Summary
        passed = sum(1 for v in all_checks.values() if v)
        total = len(all_checks)
        report.append(f"\n📊 SUMMARY: {passed}/{total} checks passed")
        
        # Issues found
        if self.issues:
            report.append(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                report.append(f"  {i}. {issue}")
        else:
            report.append("\n✅ NO ISSUES FOUND")
        
        # Fixes applied
        if self.fixes_applied:
            report.append(f"\n🔧 FIXES APPLIED ({len(self.fixes_applied)}):")
            for i, fix in enumerate(self.fixes_applied, 1):
                report.append(f"  {i}. {fix}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        if self.issues:
            if any("GPU" in issue or "CUDA" in issue for issue in self.issues):
                report.append("  • Install NVIDIA drivers and CUDA toolkit")
            if any("package" in issue for issue in self.issues):
                report.append("  • Run: pip install torch transformers peft datasets accelerate")
            if any("documents" in issue for issue in self.issues):
                report.append("  • Upload documents via: python scripts/upload_cli.py --file document.pdf")
            if any("training data" in issue for issue in self.issues):
                report.append("  • Generate training data: python scripts/quick_train.py prepare")
            if any("Axolotl" in issue for issue in self.issues):
                report.append("  • Prepare Axolotl dataset: python prepare_axolotl_dataset.py")
        else:
            report.append("  • System appears ready for training!")
            report.append("  • Start training with: python scripts/quick_train.py train")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def run_specific_check(self, check_type: str) -> Dict[str, bool]:
        """Run specific type of check"""
        check_methods = {
            'system': self.check_system_requirements,
            'python': self.check_python_dependencies,
            'directories': self.check_directory_structure,
            'rag': self.check_rag_configuration,
            'training': self.check_training_data,
            'axolotl': self.check_axolotl_setup,
            'status': self.check_training_status
        }
        
        if check_type in check_methods:
            return check_methods[check_type]()
        else:
            print(f"❌ Unknown check type: {check_type}")
            print(f"Available checks: {', '.join(check_methods.keys())}")
            return {}

def main():
    parser = argparse.ArgumentParser(
        description='RAG to Axolotl troubleshooting tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python troubleshoot.py                    # Run all checks
  python troubleshoot.py --check system     # Check system requirements
  python troubleshoot.py --fix              # Attempt automatic fixes
  python troubleshoot.py --report           # Generate detailed report
"""
    )
    
    parser.add_argument(
        '--check', '-c',
        type=str,
        choices=['system', 'python', 'directories', 'rag', 'training', 'axolotl', 'status'],
        help='Run specific check only'
    )
    
    parser.add_argument(
        '--fix', '-f',
        action='store_true',
        help='Attempt automatic fixes'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate detailed report'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory path'
    )
    
    args = parser.parse_args()
    
    # Initialize troubleshooting tool
    tool = TroubleshootingTool(args.base_dir)
    
    if args.check:
        # Run specific check
        results = tool.run_specific_check(args.check)
        print(f"\n🔍 {args.check.upper()} CHECK RESULTS:")
        for key, value in results.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    elif args.fix:
        # Attempt fixes
        fixes = tool.attempt_fixes()
        if fixes:
            print("\n🔧 Applied fixes:")
            for fix in fixes:
                print(f"  ✅ {fix}")
        else:
            print("\n✅ No fixes needed")
    
    elif args.report:
        # Generate full report
        report = tool.generate_report()
        print(report)
    
    else:
        # Run all checks and show summary
        print("🚀 Running RAG to Axolotl diagnostics...")
        report = tool.generate_report()
        print(report)
        
        if tool.issues:
            print("\n🔧 Run with --fix to attempt automatic repairs")
            sys.exit(1)
        else:
            print("\n🎉 All systems ready!")
            sys.exit(0)

if __name__ == '__main__':
    main()