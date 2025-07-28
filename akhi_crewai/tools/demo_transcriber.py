#!/usr/bin/env python3
"""Demo script for the Transcription Tool

This script demonstrates the TranscriptionTool functionality with various
configurations and use cases.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add the tools directory to the path
sys.path.append(os.path.dirname(__file__))

try:
    from transcriber import TranscriptionTool, TranscriptionInput
except ImportError as e:
    print(f"❌ Failed to import TranscriptionTool: {e}")
    sys.exit(1)


def create_demo_audio_file():
    """Create a demo audio file with speech for transcription."""
    print("🎵 Creating demo audio file...")
    
    demo_dir = os.path.join(os.path.dirname(__file__), "demo_audio")
    os.makedirs(demo_dir, exist_ok=True)
    
    demo_file = os.path.join(demo_dir, "islamic_demo.wav")
    
    # Try to create a demo audio file with text-to-speech
    try:
        # Use espeak if available to create speech audio
        demo_text = "Assalamu alaikum. This is a demonstration of the transcription tool for Islamic educational content."
        
        subprocess.run([
            "espeak", "-s", "150", "-v", "en", "-w", demo_file, demo_text
        ], check=True, capture_output=True)
        
        if os.path.exists(demo_file):
            print(f"✅ Demo audio file created: {demo_file}")
            return demo_file
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  espeak not available, trying ffmpeg...")
    
    # Fallback: Create a simple tone with ffmpeg
    try:
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-ar", "16000", "-ac", "1", "-y", demo_file
        ], check=True, capture_output=True)
        
        if os.path.exists(demo_file):
            print(f"✅ Demo tone file created: {demo_file}")
            return demo_file
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  ffmpeg not available, creating dummy file...")
    
    # Last resort: Create a dummy WAV file
    with open(demo_file, 'wb') as f:
        # Minimal WAV header for a 1-second file
        f.write(b'RIFF')
        f.write((36).to_bytes(4, 'little'))  # File size - 8
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))  # Subchunk1Size
        f.write((1).to_bytes(2, 'little'))   # AudioFormat (PCM)
        f.write((1).to_bytes(2, 'little'))   # NumChannels
        f.write((16000).to_bytes(4, 'little'))  # SampleRate
        f.write((32000).to_bytes(4, 'little'))  # ByteRate
        f.write((2).to_bytes(2, 'little'))   # BlockAlign
        f.write((16).to_bytes(2, 'little'))  # BitsPerSample
        f.write(b'data')
        f.write((0).to_bytes(4, 'little'))   # Subchunk2Size
    
    print(f"⚠️  Created dummy audio file: {demo_file}")
    print("   (Transcription will fail, but tool validation will work)")
    return demo_file


def demo_basic_transcription():
    """Demonstrate basic transcription functionality."""
    print("\n" + "=" * 60)
    print("🎯 Demo 1: Basic Transcription")
    print("=" * 60)
    
    # Create demo audio file
    demo_file = create_demo_audio_file()
    if not demo_file:
        print("❌ Could not create demo audio file")
        return
    
    try:
        # Initialize the tool
        tool = TranscriptionTool()
        
        print(f"\n📁 Input file: {demo_file}")
        print("⚙️  Configuration: Default settings (base model, auto device)")
        
        # Run transcription with default settings
        result = tool._run(
            audio_file_path=demo_file,
            model_size="tiny",  # Use tiny for speed
            device="cpu",
            include_timestamps=True,
            output_format="both"
        )
        
        print("\n📄 Transcription Result:")
        print("-" * 40)
        print(result)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        # Clean up
        if demo_file and os.path.exists(demo_file):
            try:
                os.unlink(demo_file)
                demo_dir = os.path.dirname(demo_file)
                if os.path.exists(demo_dir) and not os.listdir(demo_dir):
                    os.rmdir(demo_dir)
            except Exception:
                pass


def demo_advanced_configuration():
    """Demonstrate advanced configuration options."""
    print("\n" + "=" * 60)
    print("🎯 Demo 2: Advanced Configuration")
    print("=" * 60)
    
    tool = TranscriptionTool()
    
    # Show different model sizes
    print("\n🔧 Available Model Sizes:")
    models = ["tiny", "base", "small", "medium", "large"]
    for model in models:
        print(f"   • {model:8} - {'Fast, lower accuracy' if model == 'tiny' else 'Balanced' if model == 'base' else 'Better accuracy' if model in ['small', 'medium'] else 'Best accuracy, slower'}")
    
    # Show device options
    print("\n💻 Device Options:")
    print("   • auto  - Automatically detect best device (CPU/GPU)")
    print("   • cpu   - Force CPU usage")
    print("   • cuda  - Force GPU usage (if available)")
    
    # Show output formats
    print("\n📄 Output Formats:")
    print("   • text  - Plain text transcript only")
    print("   • json  - Structured JSON with timestamps and metadata")
    print("   • both  - Both text and JSON files")
    
    # Show language options
    print("\n🌍 Language Options:")
    print("   • auto  - Automatic language detection")
    print("   • en    - English")
    print("   • ar    - Arabic")
    print("   • ur    - Urdu")
    print("   • tr    - Turkish")
    print("   • ... and many more")


def demo_input_validation():
    """Demonstrate input validation features."""
    print("\n" + "=" * 60)
    print("🎯 Demo 3: Input Validation")
    print("=" * 60)
    
    tool = TranscriptionTool()
    
    print("\n🔍 Testing various input scenarios...")
    
    # Test 1: Missing file
    print("\n1. Missing audio file:")
    result = tool._run()
    print(f"   Result: {result[:100]}...")
    
    # Test 2: Non-existent file
    print("\n2. Non-existent file:")
    result = tool._run(audio_file_path="/non/existent/file.mp3")
    print(f"   Result: {result[:100]}...")
    
    # Test 3: Invalid file format
    print("\n3. Invalid file format:")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"test content")
        tmp_path = tmp.name
    
    try:
        result = tool._run(audio_file_path=tmp_path)
        print(f"   Result: {result[:100]}...")
    finally:
        os.unlink(tmp_path)
    
    # Test 4: Valid input schema
    print("\n4. Valid input schema:")
    try:
        valid_input = TranscriptionInput(
            audio_file_path="example.mp3",
            model_size="base",
            device="cpu",
            language="en",
            include_timestamps=True,
            word_timestamps=False,
            output_format="both"
        )
        print(f"   ✅ Schema validation passed")
        print(f"   Model: {valid_input.model_size}")
        print(f"   Device: {valid_input.device}")
        print(f"   Language: {valid_input.language}")
    except Exception as e:
        print(f"   ❌ Schema validation failed: {e}")


def demo_configuration_loading():
    """Demonstrate configuration loading from YAML."""
    print("\n" + "=" * 60)
    print("🎯 Demo 4: Configuration Loading")
    print("=" * 60)
    
    tool = TranscriptionTool()
    
    print("\n📋 Loaded Configuration:")
    config = tool.config
    
    if 'transcription' in config:
        transcription_config = config['transcription']
        print("\n🔧 Transcription Settings:")
        for key, value in transcription_config.items():
            print(f"   • {key:20}: {value}")
    else:
        print("   Using default configuration")
    
    print("\n📁 Output Directory:")
    output_dir = tool._get_output_directory()
    print(f"   Path: {output_dir}")
    print(f"   Exists: {os.path.exists(output_dir)}")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "=" * 60)
    print("🎯 Demo 5: Error Handling")
    print("=" * 60)
    
    tool = TranscriptionTool()
    
    print("\n🛡️  Error handling scenarios:")
    
    # Scenario 1: Missing dependencies (simulated)
    print("\n1. Dependency validation:")
    try:
        tool._validate_dependencies()
        print("   ✅ All dependencies available")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
    
    # Scenario 2: Invalid model size
    print("\n2. Invalid model size handling:")
    try:
        # This should be handled gracefully
        result = tool._run(
            audio_file_path="dummy.mp3",
            model_size="invalid_model"
        )
        print(f"   Result: {result[:100]}...")
    except Exception as e:
        print(f"   Handled exception: {e}")
    
    # Scenario 3: Device detection
    print("\n3. Device detection:")
    devices = ["auto", "cpu", "cuda"]
    for device in devices:
        detected = tool._determine_device(device)
        print(f"   {device:6} -> {detected}")


def demo_timestamp_features():
    """Demonstrate timestamp formatting features."""
    print("\n" + "=" * 60)
    print("🎯 Demo 6: Timestamp Features")
    print("=" * 60)
    
    tool = TranscriptionTool()
    
    print("\n⏰ Timestamp formatting examples:")
    
    test_times = [30, 65, 125, 3665, 7325]
    
    for seconds in test_times:
        formatted = tool._format_timestamp(seconds)
        print(f"   {seconds:5} seconds -> {formatted}")
    
    print("\n📝 Timestamp options:")
    print("   • include_timestamps: Add segment-level timestamps")
    print("   • word_timestamps: Add word-level timestamps (more detailed)")
    print("   • Format: MM:SS or HH:MM:SS")


def main():
    """Run all demos."""
    print("🚀 Transcription Tool Demo")
    print("=" * 60)
    print("This demo showcases the TranscriptionTool functionality")
    print("for Islamic educational content processing.")
    
    try:
        # Check if faster-whisper is available
        import faster_whisper
        whisper_available = True
    except ImportError:
        whisper_available = False
        print("\n⚠️  Note: faster-whisper not installed")
        print("   Some demos will show validation only")
    
    # Run demos
    demo_configuration_loading()
    demo_advanced_configuration()
    demo_input_validation()
    demo_error_handling()
    demo_timestamp_features()
    
    # Only run transcription demo if dependencies are available
    if whisper_available:
        demo_basic_transcription()
    else:
        print("\n⚠️  Skipping transcription demo (faster-whisper not available)")
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)
    
    print("\n📚 Next Steps:")
    print("1. Install faster-whisper: pip install faster-whisper")
    print("2. Test with real audio files")
    print("3. Integrate with CrewAI agents")
    print("4. Configure for your specific use case")
    
    print("\n🔧 Tool Integration:")
    print("```python")
    print("from transcriber import TranscriptionTool")
    print("")
    print("tool = TranscriptionTool()")
    print("result = tool._run(")
    print("    audio_file_path='path/to/audio.mp3',")
    print("    model_size='base',")
    print("    device='auto',")
    print("    include_timestamps=True")
    print(")")
    print("```")


if __name__ == "__main__":
    main()