#!/usr/bin/env python3
"""
CLI Logger System for Akhi CrewAI Pipeline
Comprehensive logging, monitoring, and CLI utilities

Usage:
    python cli_logger.py --monitor-pipeline
    python cli_logger.py --analyze-logs --log-dir ./logs
    python cli_logger.py --dashboard

Author: Akhi CrewAI Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import threading
from collections import defaultdict, deque
import re

try:
    import psutil
except ImportError:
    print("psutil not installed. Install with: pip install psutil")
    psutil = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    print("Rich not installed. Install with: pip install rich")
    RICH_AVAILABLE = False
    Console = None

class PipelineLogger:
    """Advanced logging system for the QLoRA pipeline"""
    
    def __init__(self, log_dir: str = "./logs", max_log_files: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_log_files = max_log_files
        
        # Initialize console if Rich is available
        self.console = Console() if RICH_AVAILABLE else None
        
        # Log file paths
        self.main_log = self.log_dir / "pipeline.log"
        self.error_log = self.log_dir / "errors.log"
        self.performance_log = self.log_dir / "performance.log"
        self.audit_log = self.log_dir / "audit.log"
        
        # Setup logging
        self.setup_logging()
        
        # Monitoring data
        self.metrics = {
            'start_time': datetime.now(),
            'total_videos_processed': 0,
            'total_errors': 0,
            'current_stage': 'idle',
            'stages_completed': [],
            'performance_data': deque(maxlen=1000),
            'error_history': deque(maxlen=100),
            'resource_usage': deque(maxlen=100)
        }
        
        # Stage definitions
        self.pipeline_stages = [
            'search_collect',
            'download',
            'transcribe', 
            'summarize',
            'format_qlora',
            'prepare_dataset',
            'generate_config',
            'training'
        ]
        
        self.logger = logging.getLogger('PipelineLogger')
        self.logger.info("Pipeline logger initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Main logger
        main_handler = logging.FileHandler(self.main_log)
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(detailed_formatter)
        
        # Error logger
        error_handler = logging.FileHandler(self.error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Performance logger
        perf_handler = logging.FileHandler(self.performance_log)
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(simple_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add handlers
        root_logger.addHandler(main_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(perf_handler)
        root_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        root_logger.propagate = False
    
    def log_stage_start(self, stage: str, details: Dict[str, Any] = None):
        """Log the start of a pipeline stage"""
        self.metrics['current_stage'] = stage
        
        message = f"Starting stage: {stage}"
        if details:
            message += f" - {json.dumps(details)}"
        
        self.logger.info(message)
        
        # Audit log
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'stage_start',
            'stage': stage,
            'details': details or {}
        }
        
        self._write_audit_log(audit_entry)
    
    def log_stage_complete(self, stage: str, metrics: Dict[str, Any] = None):
        """Log the completion of a pipeline stage"""
        if stage not in self.metrics['stages_completed']:
            self.metrics['stages_completed'].append(stage)
        
        message = f"Completed stage: {stage}"
        if metrics:
            message += f" - {json.dumps(metrics)}"
        
        self.logger.info(message)
        
        # Performance tracking
        perf_entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'metrics': metrics or {}
        }
        
        self.metrics['performance_data'].append(perf_entry)
        
        # Audit log
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'stage_complete',
            'stage': stage,
            'metrics': metrics or {}
        }
        
        self._write_audit_log(audit_entry)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context"""
        self.metrics['total_errors'] += 1
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'current_stage': self.metrics['current_stage'],
            'context': context or {}
        }
        
        self.metrics['error_history'].append(error_info)
        
        self.logger.error(f"Error in {self.metrics['current_stage']}: {error}", exc_info=True)
        
        # Audit log
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'error',
            'error_info': error_info
        }
        
        self._write_audit_log(audit_entry)
    
    def log_video_processed(self, video_info: Dict[str, Any]):
        """Log when a video is processed"""
        self.metrics['total_videos_processed'] += 1
        
        self.logger.info(f"Video processed: {video_info.get('title', 'Unknown')} - "
                        f"Total: {self.metrics['total_videos_processed']}")
        
        # Audit log
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'video_processed',
            'video_info': video_info,
            'total_processed': self.metrics['total_videos_processed']
        }
        
        self._write_audit_log(audit_entry)
    
    def log_resource_usage(self):
        """Log current resource usage"""
        if not psutil:
            return
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resource_info = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
            
            self.metrics['resource_usage'].append(resource_info)
            
            # Log if resource usage is high
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                self.logger.warning(f"High resource usage detected: "
                                  f"CPU: {cpu_percent}%, Memory: {memory.percent}%, "
                                  f"Disk: {disk.percent}%")
        
        except Exception as e:
            self.logger.error(f"Failed to log resource usage: {e}")
    
    def _write_audit_log(self, entry: Dict[str, Any]):
        """Write entry to audit log"""
        try:
            with open(self.audit_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        runtime = datetime.now() - self.metrics['start_time']
        
        # Calculate progress
        completed_stages = len(self.metrics['stages_completed'])
        total_stages = len(self.pipeline_stages)
        progress_percent = (completed_stages / total_stages) * 100
        
        # Get latest resource usage
        latest_resources = None
        if self.metrics['resource_usage']:
            latest_resources = self.metrics['resource_usage'][-1]
        
        # Recent errors
        recent_errors = [e for e in self.metrics['error_history'] 
                        if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        return {
            'runtime_seconds': runtime.total_seconds(),
            'runtime_formatted': str(runtime).split('.')[0],
            'current_stage': self.metrics['current_stage'],
            'progress_percent': progress_percent,
            'stages_completed': self.metrics['stages_completed'],
            'total_videos_processed': self.metrics['total_videos_processed'],
            'total_errors': self.metrics['total_errors'],
            'recent_errors': len(recent_errors),
            'latest_resources': latest_resources,
            'is_running': self.metrics['current_stage'] != 'idle'
        }
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            log_files = list(self.log_dir.glob('*.log'))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(log_files) > self.max_log_files:
                for old_log in log_files[self.max_log_files:]:
                    old_log.unlink()
                    self.logger.info(f"Removed old log file: {old_log}")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")

class LogAnalyzer:
    """Analyze pipeline logs for insights"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.console = Console() if RICH_AVAILABLE else None
    
    def analyze_logs(self) -> Dict[str, Any]:
        """Analyze all log files"""
        analysis = {
            'summary': self._analyze_main_logs(),
            'errors': self._analyze_error_logs(),
            'performance': self._analyze_performance_logs(),
            'audit': self._analyze_audit_logs()
        }
        
        return analysis
    
    def _analyze_main_logs(self) -> Dict[str, Any]:
        """Analyze main pipeline logs"""
        main_log = self.log_dir / "pipeline.log"
        if not main_log.exists():
            return {'error': 'Main log file not found'}
        
        try:
            with open(main_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Count log levels
            level_counts = defaultdict(int)
            stage_counts = defaultdict(int)
            hourly_activity = defaultdict(int)
            
            for line in lines:
                # Extract log level
                if ' - INFO - ' in line:
                    level_counts['INFO'] += 1
                elif ' - ERROR - ' in line:
                    level_counts['ERROR'] += 1
                elif ' - WARNING - ' in line:
                    level_counts['WARNING'] += 1
                elif ' - DEBUG - ' in line:
                    level_counts['DEBUG'] += 1
                
                # Extract stage information
                if 'Starting stage:' in line or 'Completed stage:' in line:
                    stage_match = re.search(r'stage: (\w+)', line)
                    if stage_match:
                        stage_counts[stage_match.group(1)] += 1
                
                # Extract timestamp for hourly activity
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}):', line)
                if timestamp_match:
                    hour = timestamp_match.group(1)
                    hourly_activity[hour] += 1
            
            return {
                'total_lines': len(lines),
                'level_counts': dict(level_counts),
                'stage_counts': dict(stage_counts),
                'hourly_activity': dict(hourly_activity),
                'last_modified': datetime.fromtimestamp(main_log.stat().st_mtime).isoformat()
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze main logs: {e}'}
    
    def _analyze_error_logs(self) -> Dict[str, Any]:
        """Analyze error logs"""
        error_log = self.log_dir / "errors.log"
        if not error_log.exists():
            return {'total_errors': 0}
        
        try:
            with open(error_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            error_types = defaultdict(int)
            error_stages = defaultdict(int)
            recent_errors = []
            
            for line in lines:
                # Extract error type
                error_match = re.search(r'Error in (\w+):', line)
                if error_match:
                    stage = error_match.group(1)
                    error_stages[stage] += 1
                
                # Extract exception type
                exception_match = re.search(r'(\w+Error|\w+Exception):', line)
                if exception_match:
                    error_types[exception_match.group(1)] += 1
                
                # Recent errors (last 24 hours)
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    if timestamp > datetime.now() - timedelta(days=1):
                        recent_errors.append({
                            'timestamp': timestamp.isoformat(),
                            'message': line.strip()
                        })
            
            return {
                'total_errors': len(lines),
                'error_types': dict(error_types),
                'error_stages': dict(error_stages),
                'recent_errors': recent_errors[-10:],  # Last 10 recent errors
                'last_modified': datetime.fromtimestamp(error_log.stat().st_mtime).isoformat()
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze error logs: {e}'}
    
    def _analyze_performance_logs(self) -> Dict[str, Any]:
        """Analyze performance logs"""
        perf_log = self.log_dir / "performance.log"
        if not perf_log.exists():
            return {'no_performance_data': True}
        
        try:
            with open(perf_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            stage_durations = defaultdict(list)
            processing_rates = []
            
            for line in lines:
                # Extract stage completion times
                if 'Completed stage:' in line:
                    # This would need more sophisticated parsing
                    # based on actual log format
                    pass
            
            return {
                'total_entries': len(lines),
                'stage_durations': dict(stage_durations),
                'processing_rates': processing_rates,
                'last_modified': datetime.fromtimestamp(perf_log.stat().st_mtime).isoformat()
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze performance logs: {e}'}
    
    def _analyze_audit_logs(self) -> Dict[str, Any]:
        """Analyze audit logs"""
        audit_log = self.log_dir / "audit.log"
        if not audit_log.exists():
            return {'no_audit_data': True}
        
        try:
            with open(audit_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            event_counts = defaultdict(int)
            timeline = []
            
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    event_type = entry.get('event', 'unknown')
                    event_counts[event_type] += 1
                    
                    timeline.append({
                        'timestamp': entry.get('timestamp'),
                        'event': event_type,
                        'stage': entry.get('stage')
                    })
                except json.JSONDecodeError:
                    continue
            
            return {
                'total_events': len(lines),
                'event_counts': dict(event_counts),
                'timeline': timeline[-20:],  # Last 20 events
                'last_modified': datetime.fromtimestamp(audit_log.stat().st_mtime).isoformat()
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze audit logs: {e}'}
    
    def display_analysis(self, analysis: Dict[str, Any]):
        """Display log analysis results"""
        if not self.console:
            # Fallback to plain text
            print(json.dumps(analysis, indent=2))
            return
        
        # Create rich display
        layout = Layout()
        
        # Summary panel
        summary = analysis.get('summary', {})
        summary_text = f"""
Total Log Lines: {summary.get('total_lines', 0)}
Log Levels: {summary.get('level_counts', {})}
Stages: {summary.get('stage_counts', {})}
"""
        
        # Error panel
        errors = analysis.get('errors', {})
        error_text = f"""
Total Errors: {errors.get('total_errors', 0)}
Error Types: {errors.get('error_types', {})}
Error Stages: {errors.get('error_stages', {})}
"""
        
        # Display panels
        self.console.print(Panel(summary_text, title="📊 Log Summary", border_style="blue"))
        self.console.print(Panel(error_text, title="❌ Error Analysis", border_style="red"))
        
        # Recent errors
        recent_errors = errors.get('recent_errors', [])
        if recent_errors:
            self.console.print("\n🕒 Recent Errors:")
            for error in recent_errors[-5:]:
                self.console.print(f"  {error['timestamp']}: {error['message'][:100]}...")

class PipelineMonitor:
    """Real-time pipeline monitoring"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.console = Console() if RICH_AVAILABLE else None
        self.running = False
        self.logger = PipelineLogger(log_dir)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.console:
            print("Rich not available. Monitoring in text mode...")
            self._text_monitor()
            return
        
        self.running = True
        
        with Live(self._create_dashboard(), refresh_per_second=2) as live:
            while self.running:
                try:
                    # Update resource usage
                    self.logger.log_resource_usage()
                    
                    # Update dashboard
                    live.update(self._create_dashboard())
                    
                    time.sleep(1)
                
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    self.console.print(f"Monitoring error: {e}")
                    time.sleep(5)
    
    def _create_dashboard(self):
        """Create monitoring dashboard"""
        status = self.logger.get_pipeline_status()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        header_text = Text("🚀 Akhi CrewAI Pipeline Monitor", style="bold blue")
        layout["header"].update(Panel(header_text, border_style="blue"))
        
        # Status panel
        status_text = f"""
🔄 Current Stage: {status['current_stage']}
⏱️  Runtime: {status['runtime_formatted']}
📊 Progress: {status['progress_percent']:.1f}%
🎥 Videos Processed: {status['total_videos_processed']}
❌ Total Errors: {status['total_errors']}
🚨 Recent Errors: {status['recent_errors']}
"""
        
        layout["left"].update(Panel(status_text, title="📈 Status", border_style="green"))
        
        # Resource panel
        resources = status.get('latest_resources')
        if resources:
            resource_text = f"""
💻 CPU: {resources.get('cpu_percent', 0):.1f}%
🧠 Memory: {resources.get('memory_percent', 0):.1f}% ({resources.get('memory_used_gb', 0):.1f}GB)
💾 Disk: {resources.get('disk_percent', 0):.1f}% ({resources.get('disk_used_gb', 0):.1f}GB)
"""
        else:
            resource_text = "Resource monitoring not available"
        
        layout["right"].update(Panel(resource_text, title="🖥️  Resources", border_style="yellow"))
        
        # Footer
        footer_text = Text(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit", 
                          style="dim")
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        return layout
    
    def _text_monitor(self):
        """Text-based monitoring fallback"""
        self.running = True
        
        while self.running:
            try:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                status = self.logger.get_pipeline_status()
                
                print("=" * 60)
                print("🚀 AKHI CREWAI PIPELINE MONITOR")
                print("=" * 60)
                print(f"Current Stage: {status['current_stage']}")
                print(f"Runtime: {status['runtime_formatted']}")
                print(f"Progress: {status['progress_percent']:.1f}%")
                print(f"Videos Processed: {status['total_videos_processed']}")
                print(f"Total Errors: {status['total_errors']}")
                print(f"Recent Errors: {status['recent_errors']}")
                
                resources = status.get('latest_resources')
                if resources:
                    print(f"\nResources:")
                    print(f"  CPU: {resources.get('cpu_percent', 0):.1f}%")
                    print(f"  Memory: {resources.get('memory_percent', 0):.1f}%")
                    print(f"  Disk: {resources.get('disk_percent', 0):.1f}%")
                
                print(f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}")
                print("Press Ctrl+C to exit")
                print("=" * 60)
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CLI Logger and Monitor for Akhi CrewAI Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start real-time monitoring
  python cli_logger.py --monitor-pipeline
  
  # Analyze existing logs
  python cli_logger.py --analyze-logs --log-dir ./logs
  
  # Dashboard mode
  python cli_logger.py --dashboard --log-dir ./logs
  
  # Cleanup old logs
  python cli_logger.py --cleanup-logs --log-dir ./logs
"""
    )
    
    # Actions
    parser.add_argument('--monitor-pipeline', action='store_true', help='Start real-time pipeline monitoring')
    parser.add_argument('--analyze-logs', action='store_true', help='Analyze existing log files')
    parser.add_argument('--dashboard', action='store_true', help='Show dashboard with current status')
    parser.add_argument('--cleanup-logs', action='store_true', help='Clean up old log files')
    
    # Configuration
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory path')
    parser.add_argument('--max-log-files', type=int, default=100, help='Maximum number of log files to keep')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.monitor_pipeline:
            # Start monitoring
            monitor = PipelineMonitor(args.log_dir)
            print("Starting pipeline monitoring... Press Ctrl+C to exit")
            monitor.start_monitoring()
        
        elif args.analyze_logs:
            # Analyze logs
            analyzer = LogAnalyzer(args.log_dir)
            analysis = analyzer.analyze_logs()
            analyzer.display_analysis(analysis)
        
        elif args.dashboard:
            # Show dashboard
            logger = PipelineLogger(args.log_dir)
            status = logger.get_pipeline_status()
            
            if RICH_AVAILABLE:
                console = Console()
                console.print(json.dumps(status, indent=2))
            else:
                print(json.dumps(status, indent=2))
        
        elif args.cleanup_logs:
            # Cleanup logs
            logger = PipelineLogger(args.log_dir, args.max_log_files)
            logger.cleanup_old_logs()
            print("Log cleanup completed")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()