#!/usr/bin/env python3
"""
File Upload API for RAG Pipeline
Provides endpoints for uploading and managing documents in the RAG system.
"""

import os
import shutil
import yaml
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our document loader
from document_loader import EnhancedDocumentLoader
from rag_pipeline import IslamicRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class UploadResponse(BaseModel):
    """Response model for file uploads"""
    success: bool = Field(..., description="Upload success status")
    filename: str = Field(..., description="Uploaded filename")
    file_path: str = Field(..., description="File storage path")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File extension")
    chunks_processed: Optional[int] = Field(None, description="Number of chunks processed")
    message: str = Field(..., description="Status message")

class FileInfo(BaseModel):
    """File information model"""
    filename: str = Field(..., description="File name")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File extension")
    upload_date: str = Field(..., description="Upload timestamp")
    chunks_count: Optional[int] = Field(None, description="Number of chunks")
    status: str = Field(..., description="Processing status")

class FileListResponse(BaseModel):
    """Response model for file listing"""
    files: List[FileInfo] = Field(..., description="List of uploaded files")
    total_files: int = Field(..., description="Total number of files")
    total_size: int = Field(..., description="Total size in bytes")
    supported_formats: List[str] = Field(..., description="Supported file formats")

class DeleteResponse(BaseModel):
    """Response model for file deletion"""
    success: bool = Field(..., description="Deletion success status")
    filename: str = Field(..., description="Deleted filename")
    message: str = Field(..., description="Status message")

class RebuildResponse(BaseModel):
    """Response model for index rebuild"""
    success: bool = Field(..., description="Rebuild success status")
    total_documents: int = Field(..., description="Total documents processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    message: str = Field(..., description="Status message")

class FileUploadManager:
    """Manages file uploads and RAG integration"""
    
    def __init__(self, upload_dir: str = "uploads", config_path: str = "config.yaml"):
        self.upload_dir = Path(upload_dir)
        self.config_path = Path(config_path)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Initialize document loader
        self.document_loader = EnhancedDocumentLoader()
        
        # Supported file extensions
        self.supported_extensions = {
            '.txt', '.pdf', '.docx', '.doc', '.html', '.htm', '.json'
        }
        
        # Maximum file size (50MB)
        self.max_file_size = 50 * 1024 * 1024
        
        logger.info(f"File upload manager initialized with upload directory: {self.upload_dir}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        formats = self.document_loader.get_supported_formats()
        return [fmt for fmt, available in formats.items() if available]
    
    def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.supported_extensions:
            return {
                'valid': False,
                'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(self.supported_extensions)}'
            }
        
        # Check file size (if available)
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            return {
                'valid': False,
                'error': f'File too large: {file.size / (1024*1024):.1f}MB. Maximum: {self.max_file_size / (1024*1024):.1f}MB'
            }
        
        return {'valid': True}
    
    async def save_file(self, file: UploadFile) -> Dict[str, Any]:
        """Save uploaded file to disk"""
        try:
            # Generate unique filename if needed
            original_name = file.filename
            file_path = self.upload_dir / original_name
            
            # Handle duplicate filenames
            counter = 1
            while file_path.exists():
                name_parts = Path(original_name).stem, Path(original_name).suffix
                new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                file_path = self.upload_dir / new_name
                counter += 1
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_size = len(content)
            
            logger.info(f"File saved: {file_path} ({file_size} bytes)")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'filename': file_path.name,
                'file_size': file_size,
                'file_type': file_path.suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process document and extract chunks"""
        try:
            documents = self.document_loader.load_documents_from_paths([str(file_path)])
            
            return {
                'success': True,
                'chunks_count': len(documents),
                'total_chars': sum(len(doc.page_content) for doc in documents)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_config(self, file_path: str) -> bool:
        """Update config.yaml to include new file"""
        try:
            # Load current config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add file to sources if not already present
            if 'data' not in config:
                config['data'] = {'sources': []}
            elif 'sources' not in config['data']:
                config['data']['sources'] = []
            
            # Convert to relative path from RAG directory
            relative_path = str(Path(file_path).relative_to(Path.cwd()))
            
            if relative_path not in config['data']['sources']:
                config['data']['sources'].append(relative_path)
                
                # Save updated config
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                logger.info(f"Added {relative_path} to config.yaml")
                return True
            else:
                logger.info(f"File {relative_path} already in config.yaml")
                return True
                
        except Exception as e:
            logger.error(f"Error updating config: {str(e)}")
            return False
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of uploaded files with metadata"""
        files = []
        
        for file_path in self.upload_dir.glob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    stat = file_path.stat()
                    
                    # Try to get chunk count
                    chunks_count = None
                    try:
                        docs = self.document_loader.load_documents_from_paths([str(file_path)])
                        chunks_count = len(docs)
                    except:
                        pass
                    
                    files.append({
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'file_size': stat.st_size,
                        'file_type': file_path.suffix.lower(),
                        'upload_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'chunks_count': chunks_count,
                        'status': 'processed' if chunks_count else 'uploaded'
                    })
                except Exception as e:
                    logger.error(f"Error getting file info for {file_path}: {str(e)}")
        
        return sorted(files, key=lambda x: x['upload_date'], reverse=True)
    
    def delete_file(self, filename: str) -> Dict[str, Any]:
        """Delete uploaded file and remove from config"""
        try:
            file_path = self.upload_dir / filename
            
            if not file_path.exists():
                return {
                    'success': False,
                    'error': f'File not found: {filename}'
                }
            
            # Remove file
            file_path.unlink()
            
            # Remove from config
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'data' in config and 'sources' in config['data']:
                    # Remove various possible path formats
                    sources_to_remove = [
                        str(file_path),
                        str(file_path.relative_to(Path.cwd())),
                        filename,
                        f"uploads/{filename}"
                    ]
                    
                    original_count = len(config['data']['sources'])
                    config['data']['sources'] = [
                        src for src in config['data']['sources'] 
                        if src not in sources_to_remove
                    ]
                    
                    if len(config['data']['sources']) < original_count:
                        with open(self.config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        logger.info(f"Removed {filename} from config.yaml")
            
            except Exception as e:
                logger.warning(f"Could not update config after deleting {filename}: {str(e)}")
            
            return {
                'success': True,
                'message': f'File {filename} deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize upload manager
upload_manager = FileUploadManager()

# Create FastAPI app
app = FastAPI(
    title="RAG File Upload API",
    description="File upload and management API for RAG Pipeline",
    version="1.0.0",
    docs_url="/upload/docs",
    redoc_url="/upload/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    auto_process: bool = Form(True),
    update_config: bool = Form(True)
):
    """Upload a document file to the RAG system"""
    
    # Validate file
    validation = upload_manager.validate_file(file)
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation['error'])
    
    # Save file
    save_result = await upload_manager.save_file(file)
    if not save_result['success']:
        raise HTTPException(status_code=500, detail=save_result['error'])
    
    response_data = {
        'success': True,
        'filename': save_result['filename'],
        'file_path': save_result['file_path'],
        'file_size': save_result['file_size'],
        'file_type': save_result['file_type'],
        'message': f"File {save_result['filename']} uploaded successfully"
    }
    
    # Process document if requested
    if auto_process:
        process_result = upload_manager.process_document(Path(save_result['file_path']))
        if process_result['success']:
            response_data['chunks_processed'] = process_result['chunks_count']
            response_data['message'] += f" and processed into {process_result['chunks_count']} chunks"
        else:
            response_data['message'] += f" but processing failed: {process_result['error']}"
    
    # Update config if requested
    if update_config:
        if upload_manager.update_config(save_result['file_path']):
            response_data['message'] += " and added to configuration"
        else:
            response_data['message'] += " but failed to update configuration"
    
    return UploadResponse(**response_data)

@app.post("/upload/multiple", response_model=List[UploadResponse])
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    auto_process: bool = Form(True),
    update_config: bool = Form(True)
):
    """Upload multiple document files"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per upload")
    
    results = []
    
    for file in files:
        try:
            # Validate file
            validation = upload_manager.validate_file(file)
            if not validation['valid']:
                results.append(UploadResponse(
                    success=False,
                    filename=file.filename,
                    file_path="",
                    file_size=0,
                    file_type="",
                    message=f"Validation failed: {validation['error']}"
                ))
                continue
            
            # Save file
            save_result = await upload_manager.save_file(file)
            if not save_result['success']:
                results.append(UploadResponse(
                    success=False,
                    filename=file.filename,
                    file_path="",
                    file_size=0,
                    file_type="",
                    message=f"Save failed: {save_result['error']}"
                ))
                continue
            
            response_data = {
                'success': True,
                'filename': save_result['filename'],
                'file_path': save_result['file_path'],
                'file_size': save_result['file_size'],
                'file_type': save_result['file_type'],
                'message': f"File {save_result['filename']} uploaded successfully"
            }
            
            # Process document if requested
            if auto_process:
                process_result = upload_manager.process_document(Path(save_result['file_path']))
                if process_result['success']:
                    response_data['chunks_processed'] = process_result['chunks_count']
                    response_data['message'] += f" and processed into {process_result['chunks_count']} chunks"
            
            # Update config if requested
            if update_config:
                if upload_manager.update_config(save_result['file_path']):
                    response_data['message'] += " and added to configuration"
            
            results.append(UploadResponse(**response_data))
            
        except Exception as e:
            results.append(UploadResponse(
                success=False,
                filename=file.filename,
                file_path="",
                file_size=0,
                file_type="",
                message=f"Upload failed: {str(e)}"
            ))
    
    return results

@app.get("/files", response_model=FileListResponse)
async def list_files():
    """Get list of uploaded files"""
    files = upload_manager.get_uploaded_files()
    total_size = sum(f['file_size'] for f in files)
    
    return FileListResponse(
        files=[FileInfo(**f) for f in files],
        total_files=len(files),
        total_size=total_size,
        supported_formats=upload_manager.get_supported_formats()
    )

@app.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str):
    """Delete an uploaded file"""
    result = upload_manager.delete_file(filename)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return DeleteResponse(
        success=True,
        filename=filename,
        message=result['message']
    )

@app.post("/rebuild-index", response_model=RebuildResponse)
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the RAG index with all uploaded files"""
    
    def rebuild_task():
        try:
            import time
            start_time = time.time()
            
            # Initialize RAG pipeline
            pipeline = IslamicRAGPipeline(str(upload_manager.config_path))
            pipeline.initialize_pipeline(rebuild_index=True)
            
            processing_time = time.time() - start_time
            
            # Get document count
            total_docs = 0
            if pipeline.vector_store:
                try:
                    # Try to get collection info
                    collection = pipeline.vector_store._collection
                    total_docs = collection.count()
                except:
                    pass
            
            logger.info(f"Index rebuilt successfully in {processing_time:.2f}s with {total_docs} documents")
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {str(e)}")
    
    # Start rebuild in background
    background_tasks.add_task(rebuild_task)
    
    return RebuildResponse(
        success=True,
        total_documents=0,  # Will be updated by background task
        processing_time=0.0,
        message="Index rebuild started in background"
    )

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    formats = upload_manager.document_loader.get_supported_formats()
    return {
        'supported_formats': [fmt for fmt, available in formats.items() if available],
        'all_formats': formats,
        'install_command': upload_manager.document_loader.install_missing_dependencies()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'upload_directory': str(upload_manager.upload_dir),
        'config_path': str(upload_manager.config_path),
        'supported_formats': len(upload_manager.get_supported_formats())
    }

def main():
    """Run the upload API server"""
    uvicorn.run(
        "file_upload_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()