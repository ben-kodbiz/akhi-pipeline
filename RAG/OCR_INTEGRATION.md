# OCR Integration in RAG System

This document describes the OCR (Optical Character Recognition) capabilities integrated into the RAG system, enabling processing of scanned documents and image-based files.

## Overview

The RAG system now supports OCR functionality using **Tesseract + pytesseract** with a hybrid approach:

1. **Text Extraction First**: Attempts to extract text directly from PDFs
2. **OCR Fallback**: If no text is found, uses OCR to process the document as images
3. **Native Image Support**: Directly processes image files using OCR

## Supported Formats

### Enhanced PDF Processing
- **Text-based PDFs**: Extracted using PyPDF2 or LangChain loaders
- **Scanned PDFs**: Processed using OCR when no extractable text is found
- **Mixed PDFs**: Handles documents with both text and scanned pages

### Image Files
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **TIFF** (.tiff)
- **BMP** (.bmp)

## Installation

### Python Dependencies
```bash
pip install pytesseract Pillow pdf2image
```

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ara
```

#### macOS
```bash
brew install tesseract tesseract-lang
```

#### Windows
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## Language Support

The OCR is configured to support:
- **English** (eng)
- **Arabic** (ara)

Configuration: `--oem 3 --psm 6 -l eng+ara`

## Usage Examples

### Processing Scanned PDFs
```python
from scripts.document_loader import EnhancedDocumentLoader

loader = EnhancedDocumentLoader()
documents = loader.load_document('scanned_document.pdf')

# The system will automatically:
# 1. Try text extraction first
# 2. Fall back to OCR if no text found
# 3. Log which method was used
```

### Processing Image Files
```python
from scripts.document_loader import EnhancedDocumentLoader

loader = EnhancedDocumentLoader()
documents = loader.load_document('document_image.png')

# Directly processes the image using OCR
```

### Checking OCR Availability
```python
from scripts.document_loader import EnhancedDocumentLoader

loader = EnhancedDocumentLoader()
formats = loader.get_supported_formats()

print(f"OCR available: {formats['png']}")
print(f"PDF with OCR: {formats['pdf']}")
```

## API Integration

The file upload API automatically supports OCR-enabled formats:

### Upload Scanned Documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@scanned_document.pdf"
```

### Upload Image Files
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document_image.png"
```

## Performance Considerations

### OCR Processing Time
- **Text extraction**: Fast (milliseconds)
- **OCR processing**: Slower (seconds per page)
- **Hybrid approach**: Optimal balance between speed and capability

### Quality Factors
- **Image resolution**: Higher resolution = better OCR accuracy
- **Text clarity**: Clear, high-contrast text works best
- **Language**: Arabic text recognition may require specific fonts

## Logging and Monitoring

The system provides detailed logging:

```
INFO:document_loader:Successfully extracted text from PDF using LangChain
INFO:document_loader:No extractable text found in PDF. Attempting OCR...
INFO:document_loader:Successfully extracted text from PDF using OCR (5 pages processed)
INFO:document_loader:Successfully extracted text from image using OCR
```

## Error Handling

### Common Issues

1. **Tesseract not installed**
   ```
   Error: No text could be extracted from PDF and OCR is not available
   Solution: Install Tesseract system dependency
   ```

2. **Poor OCR quality**
   ```
   Warning: OCR failed for page 3: No text could be extracted
   Solution: Check image quality, resolution, or text clarity
   ```

3. **Language not supported**
   ```
   Error: Tesseract language 'xyz' not found
   Solution: Install required language pack
   ```

## Testing

Run the OCR functionality tests:

```bash
python test_ocr_functionality.py
```

This will test:
- OCR availability
- Image processing
- Format support
- Dependency status

## Configuration

### Tesseract Configuration
The OCR uses optimized settings:
- **OEM 3**: LSTM OCR Engine Mode
- **PSM 6**: Uniform block of text
- **Languages**: English + Arabic

### Custom Configuration
To modify OCR settings, edit the `custom_config` in:
- `_extract_text_with_ocr_from_pdf()`
- `_load_image()`

## Integration with RAG Pipeline

### Automatic Processing
1. Documents uploaded through any interface
2. System detects file type
3. Applies appropriate processing method
4. OCR used when needed
5. Text chunked and indexed normally

### Metadata Enhancement
OCR-processed documents include metadata:
```python
{
    'source': 'document_name',
    'file_path': '/path/to/file',
    'file_type': '.pdf',
    'processing_method': 'ocr',  # Added for OCR-processed content
    'chunk_id': 0,
    'total_chunks': 5
}
```

## Best Practices

### Document Preparation
1. **Scan at high resolution** (300+ DPI)
2. **Ensure good contrast** between text and background
3. **Avoid skewed or rotated text** when possible
4. **Use clear, readable fonts**

### Performance Optimization
1. **Batch processing** for multiple documents
2. **Monitor processing time** for large documents
3. **Consider preprocessing** images for better OCR results

### Quality Assurance
1. **Review OCR output** for accuracy
2. **Test with sample documents** before bulk processing
3. **Monitor error logs** for processing issues

## Troubleshooting

### Check Dependencies
```python
from scripts.document_loader import EnhancedDocumentLoader
loader = EnhancedDocumentLoader()
print(loader.install_missing_dependencies())
```

### Verify Tesseract Installation
```bash
tesseract --version
tesseract --list-langs
```

### Test OCR Functionality
```bash
python test_ocr_functionality.py
```

## Future Enhancements

### Potential Improvements
1. **Additional language support**
2. **Image preprocessing** (deskewing, noise reduction)
3. **OCR confidence scoring**
4. **Batch processing optimization**
5. **Alternative OCR engines** (EasyOCR, PaddleOCR)

### Configuration Options
1. **Configurable OCR settings**
2. **Quality thresholds**
3. **Processing timeouts**
4. **Custom language models**

---

## Summary

The OCR integration significantly enhances the RAG system's document processing capabilities:

✅ **Hybrid approach** ensures optimal performance
✅ **Automatic fallback** to OCR when needed
✅ **Multi-language support** (English + Arabic)
✅ **Image file processing** for various formats
✅ **Seamless API integration** with existing endpoints
✅ **Comprehensive error handling** and logging
✅ **Easy testing and validation** tools

This enables the RAG system to handle a much broader range of document types, including historical texts, scanned books, and image-based documents that were previously inaccessible.