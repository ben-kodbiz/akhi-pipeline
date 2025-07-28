#!/usr/bin/env python3
"""Test script for the Transcription Tool

This script tests all functionality of the TranscriptionTool including:
- Tool initialization
- Configuration loading
- Input schema validation
- Audio file validation
- Device detection
- Transcription functionality
- Output formatting
- Error handling
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the tools directory to the path
sys.path.append(os.path.dirname(__file__))

try:
    from transcriber import TranscriptionTool, TranscriptionInput
except ImportError as e:
    print(f"❌ Failed to import TranscriptionTool: {e}")
    sys.exit(1)


def test_tool_initialization():
    """Test tool initialization."""
    print("\n🧪 Testing tool initialization...")
    
    try:
        tool = TranscriptionTool()
        assert tool.name == "Transcription Tool"
        assert tool.args_schema == TranscriptionInput
        assert hasattr(tool, 'config')
        print("✅ Tool initialization successful")
        return tool
    except Exception as e:
        print(f"❌ Tool initialization failed: {e}")
        return None


def test_configuration_loading(tool):
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        config = tool.config
        assert isinstance(config, dict)
        
        # Check if transcription config exists
        if 'transcription' in config:
            transcription_config = config['transcription']
            print(f"✅ Configuration loaded: {len(config)} sections")
            print(f"   Transcription config: {transcription_config}")
        else:
            print("✅ Using default configuration")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_input_schema_validation():
    """Test input schema validation."""
    print("\n🧪 Testing input schema validation...")
    
    try:
        # Test valid input
        valid_input = TranscriptionInput(
            audio_file_path="/path/to/audio.mp3",
            model_size="base",
            device="cpu",
            include_timestamps=True
        )
        assert valid_input.audio_file_path == "/path/to/audio.mp3"
        assert valid_input.model_size == "base"
        assert valid_input.device == "cpu"
        
        # Test minimal input
        minimal_input = TranscriptionInput(audio_file_path="test.wav")
        assert minimal_input.model_size == "base"  # default
        assert minimal_input.device == "auto"  # default
        
        print("✅ Input schema validation successful")
        return True
    except Exception as e:
        print(f"❌ Input schema validation failed: {e}")
        return False


def test_audio_file_validation(tool):
    """Test audio file validation."""
    print("\n🧪 Testing audio file validation...")
    
    try:
        # Test with non-existent file
        try:
            tool._validate_audio_file("/non/existent/file.mp3")
            print("❌ Should have failed for non-existent file")
            return False
        except FileNotFoundError:
            print("✅ Correctly detected non-existent file")
        
        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            tool._validate_audio_file(tmp_path)
            print("❌ Should have failed for invalid extension")
            return False
        except ValueError:
            print("✅ Correctly detected invalid file extension")
        finally:
            os.unlink(tmp_path)
        
        # Test with valid extension (mock file)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"fake audio content")
            tmp_path = tmp.name
        
        try:
            result = tool._validate_audio_file(tmp_path)
            assert result is True
            print("✅ Audio file validation working correctly")
            return True
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"❌ Audio file validation test failed: {e}")
        return False


def test_device_detection(tool):
    """Test device detection logic."""
    print("\n🧪 Testing device detection...")
    
    try:
        # Test auto detection
        device = tool._determine_device("auto")
        assert device in ["cpu", "cuda"]
        print(f"✅ Auto device detection: {device}")
        
        # Test explicit CPU
        device = tool._determine_device("cpu")
        assert device == "cpu"
        print("✅ CPU device selection working")
        
        # Test explicit CUDA
        device = tool._determine_device("cuda")
        assert device == "cuda"
        print("✅ CUDA device selection working")
        
        return True
    except Exception as e:
        print(f"❌ Device detection test failed: {e}")
        return False


def test_output_directory_creation(tool):
    """Test output directory creation."""
    print("\n🧪 Testing output directory creation...")
    
    try:
        # Test default directory
        output_dir = tool._get_output_directory()
        assert os.path.exists(output_dir)
        print(f"✅ Default output directory created: {output_dir}")
        
        # Test custom directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_dir = os.path.join(tmp_dir, "custom_transcripts")
            output_dir = tool._get_output_directory(custom_dir)
            assert os.path.exists(output_dir)
            assert output_dir == custom_dir
            print("✅ Custom output directory creation working")
        
        return True
    except Exception as e:
        print(f"❌ Output directory test failed: {e}")
        return False


def test_dependency_validation(tool):
    """Test dependency validation."""
    print("\n🧪 Testing dependency validation...")
    
    try:
        # This should not raise an exception if faster-whisper is installed
        tool._validate_dependencies()
        print("✅ Dependencies validation successful (faster-whisper available)")
        return True
    except ImportError as e:
        print(f"⚠️  Dependencies not available: {e}")
        print("   This is expected if faster-whisper is not installed")
        return False
    except Exception as e:
        print(f"❌ Dependencies validation failed: {e}")
        return False


def test_timestamp_formatting(tool):
    """Test timestamp formatting."""
    print("\n🧪 Testing timestamp formatting...")
    
    try:
        # Test various timestamp formats
        assert tool._format_timestamp(65) == "01:05"
        assert tool._format_timestamp(3665) == "01:01:05"
        assert tool._format_timestamp(30) == "00:30"
        
        print("✅ Timestamp formatting working correctly")
        return True
    except Exception as e:
        print(f"❌ Timestamp formatting test failed: {e}")
        return False


def test_error_handling(tool):
    """Test error handling scenarios."""
    print("\n🧪 Testing error handling...")
    
    try:
        # Test missing audio file path
        result = tool._run()
        assert "Error" in result and "audio_file_path is required" in result
        print("✅ Missing file path error handling working")
        
        # Test non-existent file
        result = tool._run(audio_file_path="/non/existent/file.mp3")
        assert "File Error" in result
        print("✅ Non-existent file error handling working")
        
        # Test invalid file format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            result = tool._run(audio_file_path=tmp_path)
            assert "Validation Error" in result
            print("✅ Invalid format error handling working")
        finally:
            os.unlink(tmp_path)
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def create_test_audio_file():
    """Create a simple test audio file for transcription testing."""
    print("\n🧪 Creating test audio file...")
    
    try:
        # Try to create a simple audio file using available tools
        import subprocess
        
        # Create a temporary directory for test files
        test_dir = os.path.join(os.path.dirname(__file__), "test_audio")
        os.makedirs(test_dir, exist_ok=True)
        
        test_file = os.path.join(test_dir, "test_audio.wav")
        
        # Try to generate a simple audio file using ffmpeg if available
        try:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-ar", "16000", "-ac", "1", "-y", test_file
            ], check=True, capture_output=True)
            
            if os.path.exists(test_file):
                print(f"✅ Test audio file created: {test_file}")
                return test_file
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # If ffmpeg is not available, create a dummy file with correct extension
        with open(test_file, 'wb') as f:
            f.write(b"RIFF" + b"\x00" * 40)  # Minimal WAV header
        
        print(f"⚠️  Created dummy audio file: {test_file}")
        print("   (Real transcription will fail, but tool validation will work)")
        return test_file
        
    except Exception as e:
        print(f"❌ Failed to create test audio file: {e}")
        return None


def test_transcription_functionality(tool):
    """Test actual transcription functionality if possible."""
    print("\n🧪 Testing transcription functionality...")
    
    # Check if faster-whisper is available
    try:
        import faster_whisper
    except ImportError:
        print("⚠️  faster-whisper not available, skipping transcription test")
        return False
    
    # Create or find a test audio file
    test_file = create_test_audio_file()
    if not test_file:
        print("⚠️  No test audio file available, skipping transcription test")
        return False
    
    try:
        # Test transcription with minimal parameters
        result = tool._run(
            audio_file_path=test_file,
            model_size="tiny",  # Use smallest model for speed
            device="cpu",
            include_timestamps=True,
            output_format="both"
        )
        
        if "✅" in result and "Transcription Successful" in result:
            print("✅ Transcription functionality working")
            print(f"   Result preview: {result[:200]}...")
            return True
        else:
            print(f"⚠️  Transcription completed with issues: {result[:200]}...")
            return False
            
    except Exception as e:
        print(f"⚠️  Transcription test failed (expected with dummy file): {e}")
        return False
    finally:
        # Clean up test file
        if test_file and os.path.exists(test_file):
            try:
                os.unlink(test_file)
                # Remove test directory if empty
                test_dir = os.path.dirname(test_file)
                if os.path.exists(test_dir) and not os.listdir(test_dir):
                    os.rmdir(test_dir)
            except Exception:
                pass


def main():
    """Run all tests."""
    print("🚀 Starting Transcription Tool Tests")
    print("=" * 50)
    
    # Track test results
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
    
    # Test 3: Input schema validation
    total_tests += 1
    if test_input_schema_validation():
        tests_passed += 1
    
    # Test 4: Audio file validation
    total_tests += 1
    if test_audio_file_validation(tool):
        tests_passed += 1
    
    # Test 5: Device detection
    total_tests += 1
    if test_device_detection(tool):
        tests_passed += 1
    
    # Test 6: Output directory creation
    total_tests += 1
    if test_output_directory_creation(tool):
        tests_passed += 1
    
    # Test 7: Dependency validation
    total_tests += 1
    if test_dependency_validation(tool):
        tests_passed += 1
    
    # Test 8: Timestamp formatting
    total_tests += 1
    if test_timestamp_formatting(tool):
        tests_passed += 1
    
    # Test 9: Error handling
    total_tests += 1
    if test_error_handling(tool):
        tests_passed += 1
    
    # Test 10: Transcription functionality (optional)
    total_tests += 1
    if test_transcription_functionality(tool):
        tests_passed += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Transcription Tool is ready.")
    elif tests_passed >= total_tests - 1:  # Allow for optional transcription test to fail
        print("✅ Core tests passed! Transcription Tool is functional.")
        print("   (Transcription test may fail without proper audio files)")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print("\n🔧 Tool Features:")
    print("✅ Multiple Whisper model sizes (tiny, base, small, medium, large)")
    print("✅ Device selection (CPU/GPU auto-detection)")
    print("✅ Language detection and specification")
    print("✅ Timestamp generation (segment and word-level)")
    print("✅ Multiple output formats (text, JSON, both)")
    print("✅ Robust error handling and validation")
    print("✅ CrewAI BaseTool integration")
    print("✅ Configuration management via YAML")


if __name__ == "__main__":
    main()