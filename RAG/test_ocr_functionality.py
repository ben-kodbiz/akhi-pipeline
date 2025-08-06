#!/usr/bin/env python3
"""
Test script to verify OCR functionality in the RAG system
"""

import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add scripts to path
sys.path.append('scripts')

from document_loader import EnhancedDocumentLoader

def create_test_image_with_text(text: str, output_path: Path):
    """Create a test image with text for OCR testing"""
    # Create a white image
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a larger font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Add text to image
    draw.text((50, 50), text, fill='black', font=font)
    
    # Save image
    img.save(output_path)
    print(f"✅ Created test image: {output_path}")

def test_ocr_on_image():
    """Test OCR functionality on a generated image"""
    print("=" * 60)
    print("Testing OCR on Image Files")
    print("=" * 60)
    
    try:
        # Initialize document loader
        loader = EnhancedDocumentLoader(chunk_size=512, chunk_overlap=50)
        
        # Create test image with text
        test_text = "This is a test document for OCR.\nIt contains multiple lines of text.\nThe RAG system should be able to extract this text."
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image_path = Path(tmp_file.name)
        
        create_test_image_with_text(test_text, test_image_path)
        
        # Test OCR extraction
        print(f"📄 Processing image with OCR...")
        documents = loader.load_document(test_image_path)
        
        if documents:
            print(f"✅ Successfully extracted text using OCR")
            print(f"📊 Extracted {len(documents)} chunks")
            print(f"📖 First chunk preview:")
            print(f"   Content: {documents[0].page_content[:200]}...")
            print(f"   Metadata: {documents[0].metadata}")
            
            # Check if extracted text contains expected content
            extracted_text = documents[0].page_content.lower()
            if "test document" in extracted_text and "ocr" in extracted_text:
                print("✅ OCR successfully extracted expected text content")
            else:
                print("⚠️  OCR extracted text but content doesn't match expected")
        else:
            print("❌ No text extracted from image")
            return False
        
        # Clean up
        test_image_path.unlink()
        return True
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def test_supported_formats():
    """Test that OCR formats are properly supported"""
    print("\n" + "=" * 60)
    print("Testing Supported Formats")
    print("=" * 60)
    
    try:
        loader = EnhancedDocumentLoader()
        formats = loader.get_supported_formats()
        
        print("📋 Supported formats:")
        for ext, available in formats.items():
            status = "✅" if available else "❌"
            print(f"   {ext}: {status}")
        
        # Check OCR formats
        ocr_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
        ocr_supported = all(formats.get(fmt, False) for fmt in ocr_formats)
        
        if ocr_supported:
            print("✅ All OCR image formats are supported")
        else:
            print("❌ Some OCR image formats are not supported")
            
        return ocr_supported
        
    except Exception as e:
        print(f"❌ Format test failed: {e}")
        return False

def test_dependency_check():
    """Test dependency installation guidance"""
    print("\n" + "=" * 60)
    print("Testing Dependency Check")
    print("=" * 60)
    
    try:
        loader = EnhancedDocumentLoader()
        install_cmd = loader.install_missing_dependencies()
        
        print(f"📦 Dependency status:")
        print(install_cmd)
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return False

def main():
    """Run all OCR tests"""
    print("🔍 Starting OCR Functionality Tests")
    print("=" * 80)
    
    tests = [
        ("Supported Formats", test_supported_formats),
        ("Dependency Check", test_dependency_check),
        ("OCR on Image", test_ocr_on_image),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All OCR functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())