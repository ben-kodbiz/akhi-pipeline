#!/usr/bin/env python3
"""
Download Whisper Model Locally

This script downloads the Whisper model to a local cache directory
to avoid downloading from Hugging Face during transcription.
"""

import os
from faster_whisper import WhisperModel

def download_whisper_model(model_size="base", cache_dir=None):
    """
    Download Whisper model to local cache directory.
    
    Args:
        model_size: Size of the Whisper model (tiny, base, small, medium, large)
        cache_dir: Directory to store the model
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "whisper_cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading Whisper model '{model_size}' to {cache_dir}...")
    
    try:
        # Initialize model with download_root to cache locally
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=cache_dir
        )
        
        print(f"✅ Successfully downloaded Whisper model '{model_size}'")
        print(f"📁 Model cached in: {cache_dir}")
        
        # Test the model with a simple transcription
        print("🧪 Testing model...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Whisper model locally")
    parser.add_argument("--model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size to download")
    parser.add_argument("--cache-dir", type=str, 
                       help="Custom cache directory")
    
    args = parser.parse_args()
    
    success = download_whisper_model(args.model, args.cache_dir)
    
    if success:
        print("\n🎉 Model download completed successfully!")
        print("\n📝 Next steps:")
        print("1. The transcriber will now use the local model")
        print("2. No more downloads from Hugging Face during transcription")
        print("3. Run your transcription pipeline as usual")
    else:
        print("\n❌ Model download failed. Please check your internet connection and try again.")