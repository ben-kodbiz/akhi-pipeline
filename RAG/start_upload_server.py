#!/usr/bin/env python3
"""
Startup script for RAG File Upload Server
Easily start the file upload API server with proper configuration.
"""

import os
import sys
import subprocess
import argparse
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'python-multipart',
        'pydantic',
        'requests',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_server_running(host: str, port: int) -> bool:
    """Check if server is already running"""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def wait_for_server(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for server to start"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if check_server_running(host, port):
            return True
        time.sleep(1)
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Start RAG File Upload Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                        # Start server on default port 8001
  %(prog)s --port 8080                           # Start server on port 8080
  %(prog)s --host 0.0.0.0 --port 8001            # Start server accessible from all interfaces
  %(prog)s --dev                                  # Start in development mode with auto-reload
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind the server to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Port to bind the server to (default: 8001)'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Run in development mode with auto-reload'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error'],
        default='info',
        help='Log level (default: info)'
    )
    
    parser.add_argument(
        '--upload-dir',
        default='uploads',
        help='Directory for uploaded files (default: uploads)'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='RAG configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if server is running, don\'t start it'
    )
    
    parser.add_argument(
        '--open-browser',
        action='store_true',
        help='Open web interface in browser after starting'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("🚀 RAG File Upload Server Launcher")
    print("="*60 + "\n")
    
    # Check if server is already running
    if check_server_running(args.host, args.port):
        print(f"✅ Server is already running at http://{args.host}:{args.port}")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/upload/docs")
        print(f"🌐 Web Interface: file://{Path.cwd()}/templates/upload.html")
        
        if args.check_only:
            return
        
        if args.open_browser:
            try:
                import webbrowser
                webbrowser.open(f"http://{args.host}:{args.port}/upload/docs")
            except:
                pass
        
        return
    
    if args.check_only:
        print(f"❌ Server is not running at http://{args.host}:{args.port}")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies are installed\n")
    
    # Check if required files exist
    script_path = Path("scripts/file_upload_api.py")
    config_path = Path(args.config)
    
    if not script_path.exists():
        print(f"❌ Upload API script not found: {script_path}")
        print("   Make sure you're running this from the RAG directory")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("   Make sure config.yaml exists in the current directory")
        sys.exit(1)
    
    # Create upload directory
    upload_dir = Path(args.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    print(f"📁 Upload directory: {upload_dir.absolute()}")
    
    # Prepare environment variables
    env = os.environ.copy()
    env['UPLOAD_DIR'] = str(upload_dir)
    env['CONFIG_PATH'] = str(config_path)
    
    # Prepare uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "scripts.file_upload_api:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.log_level
    ]
    
    if args.dev:
        cmd.extend(["--reload", "--reload-dir", "scripts"])
    
    print(f"🚀 Starting server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Mode: {'Development' if args.dev else 'Production'}")
    print(f"   Log Level: {args.log_level}")
    print(f"   Config: {config_path.absolute()}")
    print()
    
    try:
        # Start the server
        process = subprocess.Popen(cmd, env=env)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        if wait_for_server(args.host, args.port):
            print(f"\n✅ Server started successfully!")
            print(f"\n📡 Server URLs:")
            print(f"   API Base: http://{args.host}:{args.port}")
            print(f"   Health Check: http://{args.host}:{args.port}/health")
            print(f"   API Docs: http://{args.host}:{args.port}/upload/docs")
            print(f"   ReDoc: http://{args.host}:{args.port}/upload/redoc")
            
            # Web interface info
            web_interface_path = Path("templates/upload.html").absolute()
            if web_interface_path.exists():
                print(f"\n🌐 Web Interface:")
                print(f"   File: file://{web_interface_path}")
                print(f"   Note: Update API_BASE in upload.html if using different host/port")
            
            # CLI usage info
            cli_script = Path("scripts/upload_cli.py")
            if cli_script.exists():
                print(f"\n💻 CLI Usage:")
                print(f"   python scripts/upload_cli.py upload document.pdf")
                print(f"   python scripts/upload_cli.py list")
                print(f"   python scripts/upload_cli.py --help")
            
            print(f"\n🛑 Press Ctrl+C to stop the server")
            
            # Open browser if requested
            if args.open_browser:
                try:
                    import webbrowser
                    time.sleep(2)  # Give server a moment to fully start
                    webbrowser.open(f"http://{args.host}:{args.port}/upload/docs")
                except:
                    pass
            
            # Wait for the process
            process.wait()
            
        else:
            print("❌ Server failed to start within timeout")
            process.terminate()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")
        
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        if 'process' in locals():
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()