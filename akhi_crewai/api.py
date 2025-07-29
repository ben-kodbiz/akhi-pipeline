#!/usr/bin/env python3
"""
Akhi CrewAI Pipeline - FastAPI Web Interface

RESTful API and WebSocket interface for the Islamic Content Processing Pipeline.
Provides web integration for:
- Video research and processing
- Question answering on indexed content
- Content summarization
- Real-time updates via WebSocket

Author: Assistant
Date: December 2024
Phase: 6 - Main Application & CLI
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Add project paths
sys.path.append(os.path.dirname(__file__))

from main import AkhiPipelineApp


# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for videos")
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=50)


class ProcessRequest(BaseModel):
    query: str = Field(..., description="Research topic for pipeline processing")
    max_videos: int = Field(5, description="Maximum number of videos to process", ge=1, le=20)
    full_process: bool = Field(True, description="Whether to run full pipeline or just search")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about indexed content")
    index_name: str = Field("default", description="Name of the FAISS index to query")
    max_results: int = Field(5, description="Maximum number of context chunks", ge=1, le=10)


class SummarizeRequest(BaseModel):
    content_type: str = Field("recent", description="Type of content to summarize")
    limit: int = Field(5, description="Number of items to include", ge=1, le=20)


class APIResponse(BaseModel):
    status: str = Field(..., description="Response status: success, error, warning")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: str = Field(..., description="Response timestamp")


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Failed to send personal message: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


# Initialize FastAPI app
app = FastAPI(
    title="Akhi CrewAI Pipeline API",
    description="RESTful API for Islamic Content Processing Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize components
pipeline_app = None
websocket_manager = WebSocketManager()


def get_pipeline_app() -> AkhiPipelineApp:
    """Get or initialize the pipeline application."""
    global pipeline_app
    if pipeline_app is None:
        pipeline_app = AkhiPipelineApp()
    return pipeline_app


def create_response(status: str, data: Any = None, message: str = None) -> APIResponse:
    """Create a standardized API response."""
    return APIResponse(
        status=status,
        data=data,
        message=message,
        timestamp=datetime.now().isoformat()
    )


# API Endpoints

@app.get("/")
async def root():
    """Serve the web interface."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(str(static_path))
    else:
        # Fallback to API info if no web interface
        return create_response(
            status="success",
            data={
                "name": "Akhi CrewAI Pipeline API",
                "version": "1.0.0",
                "description": "Islamic Content Processing Pipeline",
                "endpoints": {
                    "search": "/api/search",
                    "process": "/api/process",
                    "query": "/api/query",
                    "summarize": "/api/summarize",
                    "status": "/api/status",
                    "websocket": "/ws",
                    "docs": "/docs"
                }
            },
            message="Welcome to Akhi CrewAI Pipeline API"
        )


@app.get("/api", response_model=APIResponse)
async def api_info():
    """API information endpoint."""
    return create_response(
        status="success",
        data={
            "name": "Akhi CrewAI Pipeline API",
            "version": "1.0.0",
            "description": "Islamic Content Processing Pipeline",
            "endpoints": {
                "search": "/api/search",
                "process": "/api/process",
                "query": "/api/query",
                "summarize": "/api/summarize",
                "status": "/api/status",
                "websocket": "/ws",
                "docs": "/docs"
            }
        },
        message="Welcome to Akhi CrewAI Pipeline API"
    )


@app.get("/api/status", response_model=APIResponse)
async def get_status():
    """Get system status and health information."""
    try:
        app_instance = get_pipeline_app()
        status_data = app_instance.get_system_status()
        
        return create_response(
            status="success",
            data=status_data,
            message="System status retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=APIResponse)
async def search_videos(request: SearchRequest):
    """Search for videos on a given topic."""
    try:
        app_instance = get_pipeline_app()
        result = app_instance.search_videos(request.query, request.max_results)
        
        if result.get('error'):
            return create_response(
                status="error",
                data=result,
                message=f"Search failed: {result['error']}"
            )
        
        # Extract video count from the formatted result string
        video_count = 0
        if 'result' in result and "Found" in result['result']:
            import re
            match = re.search(r'Found (\d+) videos', result['result'])
            video_count = int(match.group(1)) if match else 0
        
        return create_response(
            status="success",
            data=result,
            message=f"Found {video_count} videos"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process", response_model=APIResponse)
async def process_pipeline(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Execute the full pipeline for a given query."""
    try:
        app_instance = get_pipeline_app()
        
        # For long-running processes, we'll run in background
        if request.full_process:
            # Start background task
            background_tasks.add_task(
                run_pipeline_background,
                request.query,
                request.max_videos,
                request.full_process
            )
            
            return create_response(
                status="success",
                data={
                    "query": request.query,
                    "status": "processing",
                    "message": "Pipeline started in background. Check WebSocket for updates."
                },
                message="Pipeline processing started"
            )
        else:
            # Quick search only
            result = app_instance.process_pipeline(request.query, request.full_process)
            return create_response(
                status="success" if result.get('status') == 'success' else "error",
                data=result,
                message="Pipeline execution completed"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=APIResponse)
async def query_content(request: QueryRequest):
    """Ask questions about indexed content."""
    try:
        app_instance = get_pipeline_app()
        result = app_instance.query_content(request.question, request.index_name)
        
        if result.get('status') == 'error':
            return create_response(
                status="error",
                data=result,
                message=f"Query failed: {result.get('error')}"
            )
        
        return create_response(
            status="success",
            data=result,
            message="Query processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize", response_model=APIResponse)
async def summarize_content(request: SummarizeRequest):
    """Generate summaries of content."""
    try:
        app_instance = get_pipeline_app()
        result = app_instance.summarize_content(request.content_type, request.limit)
        
        if result.get('status') == 'error':
            return create_response(
                status="error",
                data=result,
                message=f"Summarization failed: {result.get('error')}"
            )
        
        return create_response(
            status="success",
            data=result,
            message="Summary generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket_manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to Akhi CrewAI Pipeline",
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get('type', 'unknown')
                
                if message_type == 'ping':
                    # Respond to ping
                    await websocket_manager.send_personal_message(
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                
                elif message_type == 'query':
                    # Handle real-time query
                    question = message.get('question', '')
                    if question:
                        app_instance = get_pipeline_app()
                        result = app_instance.query_content(question)
                        
                        await websocket_manager.send_personal_message(
                            json.dumps({
                                "type": "query_result",
                                "question": question,
                                "result": result,
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
                
                else:
                    # Echo unknown message types
                    await websocket_manager.send_personal_message(
                        json.dumps({
                            "type": "echo",
                            "original_message": message,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                    
            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


async def run_pipeline_background(query: str, max_videos: int, full_process: bool):
    """Run pipeline processing in background and send updates via WebSocket."""
    try:
        # Notify start
        await websocket_manager.broadcast(
            json.dumps({
                "type": "pipeline_update",
                "status": "started",
                "query": query,
                "message": f"Starting pipeline processing for: {query}",
                "timestamp": datetime.now().isoformat()
            })
        )
        
        # Execute pipeline
        app_instance = get_pipeline_app()
        result = app_instance.process_pipeline(query, full_process)
        
        # Notify completion
        await websocket_manager.broadcast(
            json.dumps({
                "type": "pipeline_update",
                "status": "completed",
                "query": query,
                "result": result,
                "message": f"Pipeline processing completed for: {query}",
                "timestamp": datetime.now().isoformat()
            })
        )
        
    except Exception as e:
        # Notify error
        await websocket_manager.broadcast(
            json.dumps({
                "type": "pipeline_update",
                "status": "error",
                "query": query,
                "error": str(e),
                "message": f"Pipeline processing failed for: {query}",
                "timestamp": datetime.now().isoformat()
            })
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Akhi CrewAI Pipeline API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Akhi CrewAI Pipeline API Server")
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔌 WebSocket: ws://{args.host}:{args.port}/ws")
    
    run_api_server(host=args.host, port=args.port, reload=args.reload)