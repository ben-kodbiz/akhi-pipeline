#!/usr/bin/env python3
"""
FastAPI Server for Islamic RAG Pipeline
Provides REST API endpoints for querying the RAG system.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import yaml
import time
from datetime import datetime

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import IslamicRAGPipeline
from document_loader import EnhancedDocumentLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str = Field(..., description="The Islamic question to ask")
    max_sources: int = Field(default=3, description="Maximum number of source documents to return")
    include_sources: bool = Field(default=True, description="Whether to include source documents in response")

class SourceDocument(BaseModel):
    """Model for source document information"""
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source name")
    chunk_id: int = Field(..., description="Chunk identifier")
    relevance_score: Optional[float] = Field(None, description="Relevance score if available")

class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default=[], description="Source documents used")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Dict[str, str] = Field(default={}, description="Model information")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    vector_store_ready: bool = Field(..., description="Whether vector store is ready")
    total_documents: int = Field(default=0, description="Total documents in vector store")

class ConversationRequest(BaseModel):
    """Request model for conversation-based queries"""
    question: str = Field(..., description="Current question")
    conversation_history: List[Dict[str, str]] = Field(default=[], description="Previous Q&A pairs")
    session_id: Optional[str] = Field(None, description="Session identifier")

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

# Global RAG pipeline instance
rag_pipeline: Optional[IslamicRAGPipeline] = None
conversation_memory: Dict[str, List[Dict[str, str]]] = {}
document_loader: Optional[EnhancedDocumentLoader] = None

class FileUploadManager:
    """Manages file uploads and RAG integration"""
    
    def __init__(self, upload_dir: str = "uploads", config_path: str = "config.yaml"):
        self.upload_dir = Path(upload_dir)
        self.config_path = Path(config_path)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Initialize document loader
        global document_loader
        if document_loader is None:
            document_loader = EnhancedDocumentLoader()
        self.document_loader = document_loader
        
        # Supported file extensions
        self.supported_extensions = {
            '.txt', '.pdf', '.docx', '.doc', '.html', '.htm', '.json'
        }
        
        # Maximum file size (50MB)
        self.max_file_size = 50 * 1024 * 1024
        
        logger.info(f"File upload manager initialized with upload directory: {self.upload_dir}")
    
    def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.supported_extensions:
            return {
                'valid': False,
                'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(self.supported_extensions)}'
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
            file_path_obj = Path(file_path)
            if file_path_obj.is_absolute():
                try:
                    relative_path = str(file_path_obj.relative_to(Path.cwd()))
                except ValueError:
                    # If file is not in current directory tree, use absolute path
                    relative_path = str(file_path_obj)
            else:
                relative_path = str(file_path_obj)
            
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
    title="Islamic RAG API",
    description="Retrieval-Augmented Generation API for Islamic Question Answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the upload interface HTML file
@app.get("/upload_interface.html")
async def serve_upload_interface():
    """Serve the upload interface HTML file"""
    upload_interface_path = Path(__file__).parent.parent / "upload_interface.html"
    if upload_interface_path.exists():
        return FileResponse(upload_interface_path)
    else:
        raise HTTPException(status_code=404, detail="Upload interface not found")

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline
    
    logger.info("Starting Islamic RAG API server...")
    
    try:
        # Initialize RAG pipeline
        config_path = Path(__file__).parent.parent / "config.yaml"
        rag_pipeline = IslamicRAGPipeline(str(config_path))
        rag_pipeline.initialize_pipeline(rebuild_index=False)
        
        logger.info("RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Islamic RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global rag_pipeline
    
    if rag_pipeline is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            vector_store_ready=False,
            total_documents=0
        )
    
    try:
        # Check if vector store has documents
        total_docs = 0
        if rag_pipeline.vector_store:
            try:
                # Try to get collection info
                collection = rag_pipeline.vector_store._collection
                total_docs = collection.count()
            except:
                total_docs = 0
        
        return HealthResponse(
            status="healthy",
            model_loaded=rag_pipeline.model is not None,
            vector_store_ready=rag_pipeline.vector_store is not None,
            total_documents=total_docs
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            vector_store_ready=False,
            total_documents=0
        )

# File Upload Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    process_immediately: bool = Form(True),
    update_config: bool = Form(True)
):
    """Upload a single file to the RAG system"""
    try:
        # Validate file
        validation = upload_manager.validate_file(file)
        if not validation['valid']:
            return JSONResponse(
                status_code=400,
                content=UploadResponse(
                    success=False,
                    filename=file.filename,
                    file_path="",
                    file_size=0,
                    file_type="",
                    message=validation['error']
                ).dict()
            )
        
        # Save file
        save_result = await upload_manager.save_file(file)
        if not save_result['success']:
            return JSONResponse(
                status_code=500,
                content=UploadResponse(
                    success=False,
                    filename=file.filename,
                    file_path="",
                    file_size=0,
                    file_type="",
                    message=f"Failed to save file: {save_result['error']}"
                ).dict()
            )
        
        chunks_processed = None
        
        # Process document if requested
        if process_immediately:
            process_result = upload_manager.process_document(Path(save_result['file_path']))
            if process_result['success']:
                chunks_processed = process_result['chunks_count']
                logger.info(f"Processed {save_result['filename']}: {process_result['chunks_count']} chunks")
            else:
                logger.warning(f"Failed to process {save_result['filename']}: {process_result['error']}")
        
        # Update config if requested
        if update_config:
            config_updated = upload_manager.update_config(save_result['file_path'])
            if not config_updated:
                logger.warning(f"Failed to update config for {save_result['filename']}")
        
        return UploadResponse(
            success=True,
            filename=save_result['filename'],
            file_path=save_result['file_path'],
            file_size=save_result['file_size'],
            file_type=save_result['file_type'],
            chunks_processed=chunks_processed,
            message=f"File '{save_result['filename']}' uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=UploadResponse(
                success=False,
                filename=file.filename if file else "unknown",
                file_path="",
                file_size=0,
                file_type="",
                message=f"Upload failed: {str(e)}"
            ).dict()
        )

@app.post("/upload/multiple", response_model=List[UploadResponse])
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    process_immediately: bool = Form(True),
    update_config: bool = Form(True)
):
    """Upload multiple files to the RAG system"""
    try:
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
                        message=validation['error']
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
                        message=f"Failed to save file: {save_result['error']}"
                    ))
                    continue
                
                chunks_processed = None
                
                # Process document if requested
                if process_immediately:
                    process_result = upload_manager.process_document(Path(save_result['file_path']))
                    if process_result['success']:
                        chunks_processed = process_result['chunks_count']
                    else:
                        logger.warning(f"Failed to process {file.filename}: {process_result['error']}")
                
                # Update config if requested
                if update_config:
                    upload_manager.update_config(save_result['file_path'])
                
                results.append(UploadResponse(
                    success=True,
                    filename=save_result['filename'],
                    file_path=save_result['file_path'],
                    file_size=save_result['file_size'],
                    file_type=save_result['file_type'],
                    chunks_processed=chunks_processed,
                    message=f"File '{save_result['filename']}' uploaded successfully"
                ))
                
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
        
    except Exception as e:
        logger.error(f"Multiple upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multiple upload failed: {str(e)}")

@app.get("/files", response_model=FileListResponse)
async def list_files():
    """List all uploaded files"""
    try:
        files_data = upload_manager.get_uploaded_files()
        files = [FileInfo(**file_data) for file_data in files_data]
        
        total_size = sum(f.file_size for f in files)
        
        return FileListResponse(
            files=files,
            total_files=len(files),
            total_size=total_size,
            supported_formats=list(upload_manager.supported_extensions)
        )
        
    except Exception as e:
        logger.error(f"List files error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str):
    """Delete an uploaded file"""
    try:
        result = upload_manager.delete_file(filename)
        
        if result['success']:
            return DeleteResponse(
                success=True,
                filename=filename,
                message=result['message']
            )
        else:
            status_code = 404 if 'not found' in result['error'].lower() else 500
            raise HTTPException(status_code=status_code, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_rag_index():
    """Rebuild the RAG index with all uploaded files"""
    try:
        global rag_pipeline
        
        # Reinitialize the RAG pipeline to pick up new files
        config_path = Path("config.yaml")
        if config_path.exists():
            rag_pipeline = IslamicRAGPipeline(str(config_path))
            rag_pipeline.initialize_pipeline(rebuild_index=True)
            
            return {
                "success": True,
                "message": "RAG index rebuilt successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Config file not found")
            
    except Exception as e:
        logger.error(f"Rebuild index error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG pipeline"""
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Process the query
        result = rag_pipeline.query(request.question)
        
        processing_time = time.time() - start_time
        
        # Format source documents
        sources = []
        if request.include_sources and result.get("source_documents"):
            for doc in result["source_documents"][:request.max_sources]:
                source_doc = SourceDocument(
                    content=doc["content"],
                    source=doc["metadata"].get("source", "unknown"),
                    chunk_id=doc["metadata"].get("chunk_id", 0)
                )
                sources.append(source_doc)
        
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            processing_time=processing_time,
            model_info={
                "model": "qwen3-1.7b-qlora",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/conversation", response_model=QueryResponse)
async def conversation_query(request: ConversationRequest):
    """Handle conversation-based queries with memory"""
    global rag_pipeline, conversation_memory
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import time
        start_time = time.time()
        
        session_id = request.session_id or "default"
        
        # Get or create conversation history
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        # Add previous history if provided
        if request.conversation_history:
            conversation_memory[session_id].extend(request.conversation_history)
        
        # Build context from conversation history
        context_parts = []
        for qa_pair in conversation_memory[session_id][-5:]:  # Last 5 exchanges
            if "question" in qa_pair and "answer" in qa_pair:
                context_parts.append(f"Q: {qa_pair['question']}\nA: {qa_pair['answer']}")
        
        # Enhance question with context if available
        enhanced_question = request.question
        if context_parts:
            context_str = "\n\n".join(context_parts)
            enhanced_question = f"Previous conversation:\n{context_str}\n\nCurrent question: {request.question}"
        
        # Process the query
        result = rag_pipeline.query(enhanced_question)
        
        processing_time = time.time() - start_time
        
        # Store in conversation memory
        conversation_memory[session_id].append({
            "question": request.question,
            "answer": result["answer"]
        })
        
        # Limit conversation history size
        if len(conversation_memory[session_id]) > 20:
            conversation_memory[session_id] = conversation_memory[session_id][-15:]
        
        # Format source documents
        sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"][:3]:
                source_doc = SourceDocument(
                    content=doc["content"],
                    source=doc["metadata"].get("source", "unknown"),
                    chunk_id=doc["metadata"].get("chunk_id", 0)
                )
                sources.append(source_doc)
        
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            processing_time=processing_time,
            model_info={
                "model": "qwen3-1.7b-qlora",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "session_id": session_id
            }
        )
        
    except Exception as e:
        logger.error(f"Conversation query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation query failed: {str(e)}")

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    global conversation_memory
    
    if session_id in conversation_memory:
        del conversation_memory[session_id]
        return {"message": f"Conversation history cleared for session {session_id}"}
    else:
        return {"message": f"No conversation history found for session {session_id}"}

@app.get("/conversations")
async def list_conversations():
    """List active conversation sessions"""
    global conversation_memory
    
    sessions = []
    for session_id, history in conversation_memory.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(history),
            "last_activity": history[-1] if history else None
        })
    
    return {"active_sessions": sessions}

@app.post("/rebuild_index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the vector index (background task)"""
    global rag_pipeline
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    def rebuild_task():
        try:
            logger.info("Starting index rebuild...")
            documents = rag_pipeline.load_and_process_documents()
            rag_pipeline.build_vector_store(documents)
            rag_pipeline.setup_retriever()
            logger.info("Index rebuild completed")
        except Exception as e:
            logger.error(f"Index rebuild failed: {str(e)}")
    
    background_tasks.add_task(rebuild_task)
    return {"message": "Index rebuild started in background"}

def main():
    """Run the FastAPI server"""
    import yaml
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()