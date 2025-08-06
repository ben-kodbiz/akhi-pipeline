# RAG Document Upload Guide

## Overview
This guide explains the multiple ways users can upload PDF, TXT, DOC, and other document formats to the RAG (Retrieval-Augmented Generation) system.

## Upload Methods

### 1. Web Interface (Recommended)
**File:** `upload_interface.html`

- **Access:** Open `upload_interface.html` in your browser
- **Features:**
  - Drag & drop file upload
  - Multiple file selection
  - Real-time upload progress
  - File management (view, delete)
  - Automatic RAG index updates
  - Support for all document formats

**Supported Formats:**
- PDF (.pdf)
- Text files (.txt)
- Word documents (.docx, .doc)
- HTML files (.html, .htm)
- JSON files (.json)

### 2. API Endpoints
**Base URL:** `http://localhost:8000`

#### Upload Single File
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "process_immediately=true" \
  -F "update_config=true"
```

#### Upload Multiple Files
```bash
curl -X POST "http://localhost:8000/upload/multiple" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt" \
  -F "process_immediately=true" \
  -F "update_config=true"
```

#### List Uploaded Files
```bash
curl -X GET "http://localhost:8000/files"
```

#### Delete File
```bash
curl -X DELETE "http://localhost:8000/files/document.pdf"
```

#### Rebuild RAG Index
```bash
curl -X POST "http://localhost:8000/rebuild-index"
```

### 3. Command Line Interface
**File:** `upload_cli.py`

```bash
# Upload single file
python upload_cli.py upload document.pdf

# Upload multiple files
python upload_cli.py upload document1.pdf document2.txt document3.docx

# List files
python upload_cli.py list

# Delete file
python upload_cli.py delete document.pdf

# Rebuild index
python upload_cli.py rebuild
```

### 4. Manual Method (Legacy)
1. Place files in the RAG directory
2. Update `config.yaml` to include file paths in `data.sources`
3. Restart the RAG pipeline

## API Documentation

### Interactive API Docs
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Response Models

#### Upload Response
```json
{
  "success": true,
  "filename": "document.pdf",
  "file_path": "/path/to/uploads/document.pdf",
  "file_size": 1024000,
  "file_type": ".pdf",
  "chunks_processed": 150,
  "message": "File uploaded successfully"
}
```

#### File List Response
```json
{
  "files": [
    {
      "filename": "document.pdf",
      "file_path": "/path/to/uploads/document.pdf",
      "file_size": 1024000,
      "file_type": ".pdf",
      "upload_date": "2024-01-01T12:00:00",
      "chunks_count": 150,
      "status": "processed"
    }
  ],
  "total_files": 1,
  "total_size": 1024000,
  "supported_formats": [".pdf", ".txt", ".docx", ".doc", ".html", ".htm", ".json"]
}
```

## File Processing

### Automatic Processing
- Files are automatically chunked into smaller segments
- Embeddings are generated for vector search
- Content is indexed in the RAG system
- Configuration is updated automatically

### Processing Options
- `process_immediately`: Process file right after upload (default: true)
- `update_config`: Add file to config.yaml (default: true)

### File Storage
- Uploaded files are stored in the `uploads/` directory
- Original filenames are preserved
- Duplicate names get automatic suffixes (_1, _2, etc.)

## Security Considerations

### File Validation
- Only supported file types are accepted
- Maximum file size: 50MB
- File content is validated during processing

### Access Control
- API runs on localhost by default
- CORS is enabled for web interface
- No authentication required (development setup)

## Troubleshooting

### Common Issues

1. **"Unsupported file type" error**
   - Check that your file has a supported extension
   - Ensure the file is not corrupted

2. **"File too large" error**
   - Maximum file size is 50MB
   - Consider splitting large documents

3. **Processing fails**
   - Check server logs for detailed error messages
   - Ensure the file content is readable
   - Try rebuilding the index

4. **Upload interface not working**
   - Ensure the API server is running on port 8000
   - Check browser console for JavaScript errors
   - Verify CORS settings

### Server Status
Check if the RAG API server is running:
```bash
curl http://localhost:8000/health
```

### Logs
Monitor server logs for upload and processing status:
```bash
# If running the server directly
python scripts/api_server.py

# Check the terminal output for detailed logs
```

## Integration Examples

### Python Integration
```python
import requests

# Upload file
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {'process_immediately': True, 'update_config': True}
    response = requests.post('http://localhost:8000/upload', files=files, data=data)
    print(response.json())

# List files
response = requests.get('http://localhost:8000/files')
print(response.json())
```

### JavaScript Integration
```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('process_immediately', 'true');
formData.append('update_config', 'true');

fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Best Practices

1. **Use the Web Interface** for interactive uploads
2. **Use the API** for programmatic integration
3. **Use the CLI** for batch operations
4. **Monitor file processing** status before querying
5. **Rebuild the index** after bulk uploads
6. **Keep file names descriptive** for better organization
7. **Regular cleanup** of unused files to save space

## Support

For issues or questions:
1. Check the server logs for error details
2. Verify API endpoint availability
3. Test with smaller files first
4. Ensure all dependencies are installed