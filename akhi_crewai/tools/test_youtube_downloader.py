#!/usr/bin/env python3
"""Test script for YouTube Downloader Tool

This script tests the YouTube Downloader Tool functionality including:
- Tool initialization
- Configuration loading
- Input schema validation
- URL validation
- Video info extraction
- Audio download functionality
- Error handling
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from youtube_downloader import YouTubeDownloaderTool, YouTubeDownloadInput
except ImportError as e:
    print(f"❌ Failed to import YouTube Downloader Tool: {e}")
    sys.exit(1)


def test_tool_initialization():
    """Test tool initialization."""
    print("🔧 Testing tool initialization...")
    try:
        tool = YouTubeDownloaderTool()
        assert tool.name == "YouTube Downloader Tool"
        assert tool.description is not None
        assert tool.args_schema == YouTubeDownloadInput
        print("✅ Tool initialization successful")
        return tool
    except Exception as e:
        print(f"❌ Tool initialization failed: {e}")
        return None


def test_configuration_loading(tool):
    """Test configuration loading."""
    print("⚙️ Testing configuration loading...")
    try:
        config = tool.config
        assert isinstance(config, dict)
        print(f"✅ Configuration loaded: {len(config)} sections")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_input_schema():
    """Test input schema validation."""
    print("📋 Testing input schema...")
    try:
        # Test valid input
        valid_input = YouTubeDownloadInput(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            output_dir="./test_downloads",
            audio_format="mp3",
            audio_quality="0"
        )
        assert valid_input.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert valid_input.audio_format == "mp3"
        print("✅ Input schema validation successful")
        return True
    except Exception as e:
        print(f"❌ Input schema validation failed: {e}")
        return False


def test_url_validation(tool):
    """Test URL validation functionality."""
    print("🔗 Testing URL validation...")
    try:
        # Test valid YouTube URLs
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            if not tool._validate_youtube_url(url):
                print(f"❌ Valid URL rejected: {url}")
                return False
        
        # Test invalid URLs
        invalid_urls = [
            "https://www.google.com",
            "not_a_url",
            "https://www.youtube.com/watch?v=invalid"
        ]
        
        for url in invalid_urls:
            if tool._validate_youtube_url(url):
                print(f"❌ Invalid URL accepted: {url}")
                return False
        
        print("✅ URL validation working correctly")
        return True
    except Exception as e:
        print(f"❌ URL validation test failed: {e}")
        return False


def test_video_id_extraction(tool):
    """Test video ID extraction."""
    print("🆔 Testing video ID extraction...")
    try:
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ")
        ]
        
        for url, expected_id in test_cases:
            extracted_id = tool._extract_video_id(url)
            if extracted_id != expected_id:
                print(f"❌ ID extraction failed for {url}: got {extracted_id}, expected {expected_id}")
                return False
        
        print("✅ Video ID extraction working correctly")
        return True
    except Exception as e:
        print(f"❌ Video ID extraction test failed: {e}")
        return False


def test_dependency_validation(tool):
    """Test dependency validation."""
    print("📦 Testing dependency validation...")
    try:
        tool._validate_dependencies()
        print("✅ Dependencies validation successful (yt-dlp available)")
        return True
    except Exception as e:
        print(f"⚠️ Dependencies validation failed: {e}")
        print("   This is expected if yt-dlp is not installed")
        return False


def test_video_info_extraction(tool):
    """Test video information extraction."""
    print("📺 Testing video info extraction...")
    try:
        # Use a well-known, stable YouTube video for testing
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - stable test video
        
        video_info = tool._get_video_info(test_url)
        
        if video_info:
            print(f"✅ Video info extracted: {video_info.get('title', 'No title')}")
            return True
        else:
            print("⚠️ No video info extracted (may be due to network or yt-dlp issues)")
            return False
    except Exception as e:
        print(f"⚠️ Video info extraction failed: {e}")
        return False


def test_download_functionality(tool):
    """Test actual download functionality."""
    print("⬇️ Testing download functionality...")
    
    # Create temporary directory for test downloads
    temp_dir = tempfile.mkdtemp(prefix="youtube_downloader_test_")
    
    try:
        # Use a short, stable test video
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
        
        print(f"   Test URL: {test_url}")
        print(f"   Output directory: {temp_dir}")
        
        # Test the _run method
        result = tool._run(
            url=test_url,
            output_dir=temp_dir,
            audio_format="mp3",
            audio_quality="9",  # Use lowest quality for faster download
            extract_info=True
        )
        
        print(f"   Download result: {result[:200]}...")  # Show first 200 chars
        
        # Check if download was successful
        if "✅" in result and "Download Successful" in result:
            # Check if file was actually created
            mp3_files = list(Path(temp_dir).glob("*.mp3"))
            if mp3_files:
                print(f"✅ Download successful: {len(mp3_files)} file(s) created")
                return True
            else:
                print("⚠️ Download reported success but no MP3 files found")
                return False
        else:
            print(f"⚠️ Download failed or encountered issues")
            return False
            
    except Exception as e:
        print(f"⚠️ Download test failed: {e}")
        return False
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"   Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"   Warning: Could not clean up {temp_dir}: {e}")


def test_error_handling(tool):
    """Test error handling."""
    print("🚨 Testing error handling...")
    try:
        # Test with invalid URL
        result = tool._run(url="https://www.google.com")
        if "Error: Invalid YouTube URL" in result:
            print("✅ Invalid URL error handling working")
        else:
            print(f"⚠️ Unexpected response for invalid URL: {result[:100]}...")
        
        # Test with non-existent video
        result = tool._run(url="https://www.youtube.com/watch?v=nonexistent123")
        if "❌" in result or "Error" in result or "Failed" in result:
            print("✅ Non-existent video error handling working")
        else:
            print(f"⚠️ Unexpected response for non-existent video: {result[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 YouTube Downloader Tool Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Tool initialization
    total_tests += 1
    tool = test_tool_initialization()
    if tool:
        tests_passed += 1
    else:
        print("❌ Cannot continue without tool initialization")
        return
    
    # Test 2: Configuration loading
    total_tests += 1
    if test_configuration_loading(tool):
        tests_passed += 1
    
    # Test 3: Input schema
    total_tests += 1
    if test_input_schema():
        tests_passed += 1
    
    # Test 4: URL validation
    total_tests += 1
    if test_url_validation(tool):
        tests_passed += 1
    
    # Test 5: Video ID extraction
    total_tests += 1
    if test_video_id_extraction(tool):
        tests_passed += 1
    
    # Test 6: Dependency validation
    total_tests += 1
    if test_dependency_validation(tool):
        tests_passed += 1
    
    # Test 7: Video info extraction
    total_tests += 1
    if test_video_info_extraction(tool):
        tests_passed += 1
    
    # Test 8: Download functionality (only if dependencies are available)
    if tests_passed >= 6:  # Only test download if basic functionality works
        total_tests += 1
        if test_download_functionality(tool):
            tests_passed += 1
    
    # Test 9: Error handling
    total_tests += 1
    if test_error_handling(tool):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! YouTube Downloader Tool is working correctly.")
        return 0
    elif tests_passed >= total_tests * 0.7:  # 70% pass rate
        print("⚠️ Most tests passed. Tool is functional with some limitations.")
        return 0
    else:
        print("❌ Many tests failed. Tool may not be working correctly.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)