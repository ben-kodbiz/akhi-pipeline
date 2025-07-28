#!/usr/bin/env python3
"""Demo script for YouTube Downloader Tool

This script demonstrates the YouTube Downloader Tool functionality with
real examples of downloading Islamic educational content.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from youtube_downloader import YouTubeDownloaderTool
except ImportError as e:
    print(f"❌ Failed to import YouTube Downloader Tool: {e}")
    sys.exit(1)


def demo_basic_download():
    """Demonstrate basic download functionality."""
    print("\n🎬 Demo 1: Basic Audio Download")
    print("-" * 40)
    
    tool = YouTubeDownloaderTool()
    
    # Create a demo downloads directory
    demo_dir = "./demo_downloads"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Demo URL - a short Islamic lecture
    demo_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for demo
    
    print(f"Downloading from: {demo_url}")
    print(f"Output directory: {demo_dir}")
    
    result = tool._run(
        url=demo_url,
        output_dir=demo_dir,
        audio_format="mp3",
        audio_quality="0",  # Best quality
        extract_info=True
    )
    
    print("\nResult:")
    print(result)
    
    return demo_dir


def demo_custom_settings():
    """Demonstrate download with custom settings."""
    print("\n⚙️ Demo 2: Custom Download Settings")
    print("-" * 40)
    
    tool = YouTubeDownloaderTool()
    
    # Create a custom downloads directory
    custom_dir = "./custom_downloads"
    os.makedirs(custom_dir, exist_ok=True)
    
    # Demo URL
    demo_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print(f"Downloading with custom settings:")
    print(f"- URL: {demo_url}")
    print(f"- Format: WAV (higher quality)")
    print(f"- Quality: Best (0)")
    print(f"- Custom filename template")
    
    result = tool._run(
        url=demo_url,
        output_dir=custom_dir,
        audio_format="wav",
        audio_quality="0",
        filename_template="%(uploader)s - %(title)s.%(ext)s",
        extract_info=True
    )
    
    print("\nResult:")
    print(result)
    
    return custom_dir


def demo_batch_download():
    """Demonstrate downloading multiple videos."""
    print("\n📦 Demo 3: Batch Download")
    print("-" * 40)
    
    tool = YouTubeDownloaderTool()
    
    # Create batch downloads directory
    batch_dir = "./batch_downloads"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Demo URLs - multiple short videos
    demo_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
        "https://www.youtube.com/watch?v=oHg5SJYRHA0",  # Another demo video
    ]
    
    print(f"Downloading {len(demo_urls)} videos to: {batch_dir}")
    
    for i, url in enumerate(demo_urls, 1):
        print(f"\n📥 Downloading video {i}/{len(demo_urls)}")
        print(f"URL: {url}")
        
        result = tool._run(
            url=url,
            output_dir=batch_dir,
            audio_format="mp3",
            audio_quality="5",  # Medium quality for faster download
            extract_info=False  # Skip info extraction for faster processing
        )
        
        # Show abbreviated result
        if "✅" in result:
            print("✅ Download successful")
        else:
            print("❌ Download failed")
            print(result[:200] + "..." if len(result) > 200 else result)
    
    return batch_dir


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n🚨 Demo 4: Error Handling")
    print("-" * 40)
    
    tool = YouTubeDownloaderTool()
    
    error_cases = [
        {
            "name": "Invalid URL",
            "url": "https://www.google.com",
            "expected": "Invalid YouTube URL"
        },
        {
            "name": "Non-existent video",
            "url": "https://www.youtube.com/watch?v=nonexistent123",
            "expected": "Download Failed"
        },
        {
            "name": "Malformed URL",
            "url": "not_a_url",
            "expected": "Invalid YouTube URL"
        }
    ]
    
    for case in error_cases:
        print(f"\n🧪 Testing: {case['name']}")
        print(f"URL: {case['url']}")
        
        result = tool._run(url=case['url'])
        
        if case['expected'] in result:
            print(f"✅ Error handled correctly: {case['expected']}")
        else:
            print(f"⚠️ Unexpected response: {result[:100]}...")


def demo_islamic_content():
    """Demonstrate downloading Islamic educational content."""
    print("\n🕌 Demo 5: Islamic Educational Content")
    print("-" * 40)
    
    tool = YouTubeDownloaderTool()
    
    # Create Islamic content directory
    islamic_dir = "./islamic_downloads"
    os.makedirs(islamic_dir, exist_ok=True)
    
    print("This demo would download Islamic educational content.")
    print("For demonstration purposes, we'll use a placeholder URL.")
    print("\nIn a real scenario, you would use URLs like:")
    print("- Nouman Ali Khan lectures")
    print("- Yasir Qadhi Seerah series")
    print("- Omar Suleiman Islamic history")
    print("- Mufti Menk motivational talks")
    
    # For demo, we'll just show the tool configuration
    print(f"\nTool configuration for Islamic content:")
    print(f"- Output directory: {islamic_dir}")
    print(f"- Audio format: MP3 (compatible with most devices)")
    print(f"- Quality: Best (for clear audio)")
    print(f"- Filename template: Scholar - Topic.mp3")
    
    # Example of what the download would look like
    example_result = """
✅ **Download Successful**

**File:** ./islamic_downloads/Nouman Ali Khan - Quran Tafseer Surah Al-Fatiha.mp3
**Size:** 15,234,567 bytes
**Format:** mp3

**Video Information:**
Title: Quran Tafseer: Surah Al-Fatiha - Nouman Ali Khan
Channel: Bayyinah Institute
Duration: 45:23
Views: 125,432
Upload Date: 20231015

Successfully downloaded audio to ./islamic_downloads/Nouman Ali Khan - Quran Tafseer Surah Al-Fatiha.mp3
"""
    
    print("\nExample download result:")
    print(example_result)
    
    return islamic_dir


def show_directory_contents(directory):
    """Show contents of a directory."""
    if os.path.exists(directory):
        files = list(Path(directory).glob("*"))
        if files:
            print(f"\n📁 Contents of {directory}:")
            for file in files:
                if file.is_file():
                    size = file.stat().st_size
                    print(f"  📄 {file.name} ({size:,} bytes)")
        else:
            print(f"\n📁 {directory} is empty")
    else:
        print(f"\n📁 {directory} does not exist")


def cleanup_demo_directories():
    """Clean up demo directories."""
    demo_dirs = [
        "./demo_downloads",
        "./custom_downloads", 
        "./batch_downloads",
        "./islamic_downloads"
    ]
    
    print("\n🧹 Cleaning up demo directories...")
    for directory in demo_dirs:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"✅ Removed {directory}")
            except Exception as e:
                print(f"⚠️ Could not remove {directory}: {e}")
        else:
            print(f"ℹ️ {directory} does not exist")


def main():
    """Run all demos."""
    print("🎬 YouTube Downloader Tool Demo")
    print("=" * 50)
    print("This demo showcases the YouTube Downloader Tool capabilities")
    print("for downloading Islamic educational content.")
    
    try:
        # Demo 1: Basic download
        demo_dir = demo_basic_download()
        show_directory_contents(demo_dir)
        
        # Demo 2: Custom settings
        custom_dir = demo_custom_settings()
        show_directory_contents(custom_dir)
        
        # Demo 3: Batch download (commented out to avoid long downloads)
        # batch_dir = demo_batch_download()
        # show_directory_contents(batch_dir)
        
        # Demo 4: Error handling
        demo_error_handling()
        
        # Demo 5: Islamic content (conceptual)
        islamic_dir = demo_islamic_content()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nThe YouTube Downloader Tool provides:")
        print("✅ High-quality audio extraction")
        print("✅ Multiple format support (MP3, WAV, M4A)")
        print("✅ Customizable output settings")
        print("✅ Robust error handling")
        print("✅ Video metadata extraction")
        print("✅ Batch download capabilities")
        
        # Ask user if they want to clean up
        response = input("\n🧹 Clean up demo files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_demo_directories()
        else:
            print("Demo files preserved for inspection.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
        cleanup_demo_directories()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        cleanup_demo_directories()


if __name__ == "__main__":
    main()