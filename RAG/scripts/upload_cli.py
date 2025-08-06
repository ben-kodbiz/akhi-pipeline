#!/usr/bin/env python3
"""
Command Line Interface for RAG Document Upload
Provides easy file upload functionality from the command line.
"""

import os
import sys
import argparse
import requests
import json
from pathlib import Path
from typing import List, Optional
import time

def print_banner():
    """Print CLI banner"""
    print("\n" + "="*60)
    print("📚 RAG Document Upload CLI")
    print("Upload documents to your Islamic RAG system")
    print("="*60 + "\n")

def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    if bytes_size == 0:
        return "0 B"
    
    sizes = ['B', 'KB', 'MB', 'GB']
    k = 1024
    i = 0
    while bytes_size >= k and i < len(sizes) - 1:
        bytes_size /= k
        i += 1
    
    return f"{bytes_size:.2f} {sizes[i]}"

def check_server_health(api_url: str) -> bool:
    """Check if the upload server is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_supported_formats(api_url: str) -> List[str]:
    """Get list of supported file formats"""
    try:
        response = requests.get(f"{api_url}/supported-formats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('supported_formats', [])
    except:
        pass
    return ['pdf', 'txt', 'docx', 'doc', 'html', 'json']

def validate_files(file_paths: List[str], supported_formats: List[str]) -> List[str]:
    """Validate file paths and formats"""
    valid_files = []
    
    for file_path in file_paths:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        # Check if it's a file
        if not path.is_file():
            print(f"❌ Not a file: {file_path}")
            continue
        
        # Check file extension
        file_ext = path.suffix.lower().lstrip('.')
        if file_ext not in supported_formats:
            print(f"❌ Unsupported format: {file_path} ({file_ext})")
            print(f"   Supported formats: {', '.join(supported_formats)}")
            continue
        
        # Check file size (50MB limit)
        file_size = path.stat().st_size
        if file_size > 50 * 1024 * 1024:
            print(f"❌ File too large: {file_path} ({format_file_size(file_size)})")
            print(f"   Maximum size: 50 MB")
            continue
        
        valid_files.append(file_path)
        print(f"✅ Valid file: {path.name} ({format_file_size(file_size)})")
    
    return valid_files

def upload_single_file(api_url: str, file_path: str, auto_process: bool = True, update_config: bool = True) -> dict:
    """Upload a single file"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
            data = {
                'auto_process': auto_process,
                'update_config': update_config
            }
            
            response = requests.post(f"{api_url}/upload", files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def upload_multiple_files(api_url: str, file_paths: List[str], auto_process: bool = True, update_config: bool = True) -> List[dict]:
    """Upload multiple files"""
    try:
        files = []
        for file_path in file_paths:
            files.append(('files', (Path(file_path).name, open(file_path, 'rb'), 'application/octet-stream')))
        
        data = {
            'auto_process': auto_process,
            'update_config': update_config
        }
        
        response = requests.post(f"{api_url}/upload/multiple", files=files, data=data, timeout=600)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            return response.json()
        else:
            return [{
                'success': False,
                'filename': Path(fp).name,
                'error': f"HTTP {response.status_code}: {response.text}"
            } for fp in file_paths]
    
    except Exception as e:
        return [{
            'success': False,
            'filename': Path(fp).name,
            'error': str(e)
        } for fp in file_paths]

def list_uploaded_files(api_url: str) -> dict:
    """List uploaded files"""
    try:
        response = requests.get(f"{api_url}/files", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'error': str(e)}

def delete_file(api_url: str, filename: str) -> dict:
    """Delete an uploaded file"""
    try:
        response = requests.delete(f"{api_url}/files/{filename}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def rebuild_index(api_url: str) -> dict:
    """Rebuild the RAG index"""
    try:
        response = requests.post(f"{api_url}/rebuild-index", timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(
        description="Upload documents to RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s upload document.pdf                    # Upload single file
  %(prog)s upload *.pdf *.txt                     # Upload multiple files
  %(prog)s upload docs/ --recursive               # Upload all files in directory
  %(prog)s list                                   # List uploaded files
  %(prog)s delete document.pdf                    # Delete uploaded file
  %(prog)s rebuild                                # Rebuild RAG index
  %(prog)s status                                 # Check server status
        """
    )
    
    parser.add_argument(
        '--api-url', 
        default='http://localhost:8001',
        help='API server URL (default: http://localhost:8001)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload files')
    upload_parser.add_argument('files', nargs='+', help='Files or directories to upload')
    upload_parser.add_argument('--recursive', '-r', action='store_true', help='Upload files recursively from directories')
    upload_parser.add_argument('--no-process', action='store_true', help='Skip automatic document processing')
    upload_parser.add_argument('--no-config', action='store_true', help='Skip configuration update')
    upload_parser.add_argument('--batch-size', type=int, default=5, help='Number of files to upload in each batch')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List uploaded files')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete uploaded file')
    delete_parser.add_argument('filename', help='Filename to delete')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild RAG index')
    rebuild_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check server status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print_banner()
    
    # Check server health
    print(f"🔍 Checking server at {args.api_url}...")
    if not check_server_health(args.api_url):
        print(f"❌ Server not available at {args.api_url}")
        print("   Make sure the upload API server is running:")
        print(f"   python scripts/file_upload_api.py")
        sys.exit(1)
    
    print(f"✅ Server is running at {args.api_url}\n")
    
    if args.command == 'status':
        # Get supported formats
        formats = get_supported_formats(args.api_url)
        print(f"📄 Supported formats: {', '.join(formats)}")
        
        # Get file list
        file_data = list_uploaded_files(args.api_url)
        if 'error' not in file_data:
            print(f"📁 Total files: {file_data['total_files']}")
            print(f"💾 Total size: {format_file_size(file_data['total_size'])}")
        
        print("\n✅ Server is healthy and ready for uploads")
    
    elif args.command == 'upload':
        # Get supported formats
        supported_formats = get_supported_formats(args.api_url)
        
        # Collect all files
        all_files = []
        for file_or_dir in args.files:
            path = Path(file_or_dir)
            
            if path.is_file():
                all_files.append(str(path))
            elif path.is_dir() and args.recursive:
                for ext in supported_formats:
                    all_files.extend([str(p) for p in path.rglob(f"*.{ext}")])
            elif path.is_dir():
                for ext in supported_formats:
                    all_files.extend([str(p) for p in path.glob(f"*.{ext}")])
            else:
                # Try glob pattern
                import glob
                matches = glob.glob(file_or_dir)
                all_files.extend(matches)
        
        if not all_files:
            print("❌ No files found to upload")
            sys.exit(1)
        
        # Remove duplicates
        all_files = list(set(all_files))
        
        print(f"📋 Found {len(all_files)} file(s) to validate...\n")
        
        # Validate files
        valid_files = validate_files(all_files, supported_formats)
        
        if not valid_files:
            print("\n❌ No valid files to upload")
            sys.exit(1)
        
        print(f"\n📤 Uploading {len(valid_files)} file(s)...\n")
        
        # Upload files in batches
        batch_size = args.batch_size
        auto_process = not args.no_process
        update_config = not args.no_config
        
        total_success = 0
        total_failed = 0
        
        for i in range(0, len(valid_files), batch_size):
            batch = valid_files[i:i+batch_size]
            print(f"📦 Batch {i//batch_size + 1}: Uploading {len(batch)} file(s)...")
            
            if len(batch) == 1:
                # Single file upload
                result = upload_single_file(args.api_url, batch[0], auto_process, update_config)
                results = [result]
            else:
                # Multiple file upload
                results = upload_multiple_files(args.api_url, batch, auto_process, update_config)
            
            # Process results
            for result in results:
                if result.get('success', False):
                    total_success += 1
                    filename = result.get('filename', 'Unknown')
                    chunks = result.get('chunks_processed', 'N/A')
                    print(f"   ✅ {filename} (chunks: {chunks})")
                else:
                    total_failed += 1
                    filename = result.get('filename', 'Unknown')
                    error = result.get('error', result.get('message', 'Unknown error'))
                    print(f"   ❌ {filename}: {error}")
            
            # Small delay between batches
            if i + batch_size < len(valid_files):
                time.sleep(1)
        
        print(f"\n📊 Upload Summary:")
        print(f"   ✅ Successful: {total_success}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   📁 Total: {len(valid_files)}")
        
        if total_success > 0:
            print(f"\n🎉 Upload completed! {total_success} file(s) are now available in your RAG system.")
    
    elif args.command == 'list':
        file_data = list_uploaded_files(args.api_url)
        
        if 'error' in file_data:
            print(f"❌ Error: {file_data['error']}")
            sys.exit(1)
        
        if args.format == 'json':
            print(json.dumps(file_data, indent=2))
        else:
            files = file_data['files']
            if not files:
                print("📁 No files uploaded yet")
            else:
                print(f"📁 {len(files)} uploaded file(s):\n")
                
                # Table header
                print(f"{'Filename':<30} {'Size':<10} {'Type':<6} {'Chunks':<8} {'Status':<10} {'Upload Date':<20}")
                print("-" * 90)
                
                for file in files:
                    filename = file['filename'][:28] + '..' if len(file['filename']) > 30 else file['filename']
                    size = format_file_size(file['file_size'])
                    file_type = file['file_type'].upper().lstrip('.')
                    chunks = str(file.get('chunks_count', 'N/A'))
                    status = file['status']
                    upload_date = file['upload_date'][:19].replace('T', ' ')
                    
                    print(f"{filename:<30} {size:<10} {file_type:<6} {chunks:<8} {status:<10} {upload_date:<20}")
                
                print(f"\n📊 Total: {file_data['total_files']} files, {format_file_size(file_data['total_size'])}")
    
    elif args.command == 'delete':
        if not args.force:
            confirm = input(f"Are you sure you want to delete '{args.filename}'? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Deletion cancelled")
                sys.exit(0)
        
        result = delete_file(args.api_url, args.filename)
        
        if result.get('success', False):
            print(f"✅ {result.get('message', 'File deleted successfully')}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    elif args.command == 'rebuild':
        if not args.force:
            confirm = input("Are you sure you want to rebuild the RAG index? This may take some time. (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Rebuild cancelled")
                sys.exit(0)
        
        print("🔨 Starting index rebuild...")
        result = rebuild_index(args.api_url)
        
        if result.get('success', False):
            print(f"✅ {result.get('message', 'Index rebuild completed')}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)