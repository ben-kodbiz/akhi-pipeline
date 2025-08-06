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
from utils.config_loader import get_config

# Add project paths
sys.path.append(os.path.dirname(__file__))

from main import AkhiPipelineApp
from agents.qlora_trainer import QLoRATrainerAgent


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


# QLoRA-specific request models
class QLoRATrainingRequest(BaseModel):
    model_name: str = Field("akhi-islamic-assistant", description="Name for the trained model")
    training_data_path: str = Field(..., description="Path to training data file")
    base_model: str = Field("", description="Base model to fine-tune")
    num_epochs: int = Field(3, description="Number of training epochs", ge=1, le=10)
    learning_rate: float = Field(0.0002, description="Learning rate", gt=0, le=0.01)
    batch_size: int = Field(1, description="Training batch size", ge=1, le=8)
    lora_rank: int = Field(16, description="LoRA rank", ge=1, le=64)
    lora_alpha: int = Field(32, description="LoRA alpha", ge=1, le=128)
    max_seq_length: int = Field(512, description="Maximum sequence length", ge=128, le=2048)


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


# Load supported models from config
try:
    config = get_config()
    SUPPORTED_MODELS = config.get('api', {}).get('supported_models', [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small", 
        "Qwen/Qwen-1_8B-Chat",
        "microsoft/DialoGPT-large"
    ])
except Exception:
    SUPPORTED_MODELS = [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small",
        "Qwen/Qwen-1_8B-Chat", 
        "microsoft/DialoGPT-large"
    ]

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
qlora_trainer = None
websocket_manager = WebSocketManager()


def get_pipeline_app() -> AkhiPipelineApp:
    """Get or create pipeline app instance."""
    global pipeline_app
    if pipeline_app is None:
        pipeline_app = AkhiPipelineApp()
    return pipeline_app


def get_qlora_trainer() -> QLoRATrainerAgent:
    """Get or create QLoRA trainer instance."""
    global qlora_trainer
    if qlora_trainer is None:
        qlora_trainer = QLoRATrainerAgent()
    return qlora_trainer


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


# QLoRA API Endpoints
@app.get("/api/qlora/configs", response_model=APIResponse)
async def get_qlora_configs():
    """Get QLoRA training configurations."""
    try:
        trainer = get_qlora_trainer()
        config = trainer.get_config()
        
        return create_response(
            status="success",
            data={
                "default_config": config,
                "available_models": SUPPORTED_MODELS,
                "training_parameters": {
                    "num_epochs": {"min": 1, "max": 10, "default": 3},
                    "learning_rate": {"min": 0.0001, "max": 0.01, "default": 0.0002},
                    "batch_size": {"min": 1, "max": 8, "default": 1},
                    "lora_rank": {"min": 1, "max": 64, "default": 16},
                    "lora_alpha": {"min": 1, "max": 128, "default": 32},
                    "max_seq_length": {"min": 128, "max": 2048, "default": 512}
                }
            },
            message="QLoRA configurations retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qlora/train", response_model=APIResponse)
async def start_qlora_training(request: QLoRATrainingRequest, background_tasks: BackgroundTasks):
    """Start QLoRA training job."""
    try:
        trainer = get_qlora_trainer()
        
        # Create job configuration
        job_config = {
            "model_name": request.model_name,
            "training_data_path": request.training_data_path,
            "base_model": request.base_model,
            "num_epochs": request.num_epochs,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "lora_rank": request.lora_rank,
            "lora_alpha": request.lora_alpha,
            "max_seq_length": request.max_seq_length
        }
        
        # Start training job
        job_id = trainer.start_training_job(job_config)
        
        # Notify via WebSocket
        await websocket_manager.broadcast(
            json.dumps({
                "type": "qlora_training_started",
                "job_id": job_id,
                "model_name": request.model_name,
                "timestamp": datetime.now().isoformat()
            })
        )
        
        return create_response(
            status="success",
            data={
                "job_id": job_id,
                "status": "started",
                "config": job_config
            },
            message=f"QLoRA training job {job_id} started successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qlora/training/jobs", response_model=APIResponse)
async def get_training_jobs():
    """Get all QLoRA training jobs."""
    try:
        trainer = get_qlora_trainer()
        
        # Get current training status
        status = trainer.get_training_status()
        
        jobs = []
        if status.get('current_job'):
            jobs.append({
                "id": status['current_job'],
                "status": status.get('status', 'unknown'),
                "progress": status.get('progress', 0),
                "model_path": status.get('model_path'),
                "last_checkpoint": status.get('last_checkpoint'),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
        
        return create_response(
            status="success",
            data={"jobs": jobs},
            message=f"Retrieved {len(jobs)} training jobs"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qlora/status/{job_id}", response_model=APIResponse)
async def get_training_status(job_id: str):
    """Get status of a specific training job."""
    try:
        trainer = get_qlora_trainer()
        status = trainer.get_training_status()
        
        if status.get('current_job') != job_id:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        return create_response(
            status="success",
            data={
                "job_id": job_id,
                "status": status.get('status', 'unknown'),
                "progress": status.get('progress', 0),
                "model_path": status.get('model_path'),
                "last_checkpoint": status.get('last_checkpoint'),
                "validation_results": status.get('validation_results'),
                "updated_at": datetime.now().isoformat()
            },
            message=f"Training status for job {job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qlora/metrics/{job_id}", response_model=APIResponse)
async def get_training_metrics(job_id: str):
    """Get training metrics for a specific job."""
    try:
        trainer = get_qlora_trainer()
        status = trainer.get_training_status()
        
        if status.get('current_job') != job_id:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        # Mock metrics data - in a real implementation, this would come from training logs
        metrics = {
            "loss": [2.5, 2.1, 1.8, 1.6, 1.4],
            "learning_rate": [0.0002, 0.00018, 0.00016, 0.00014, 0.00012],
            "epoch": [1, 2, 3, 4, 5],
            "step": [100, 200, 300, 400, 500],
            "validation_accuracy": [0.65, 0.72, 0.78, 0.82, 0.85],
            "training_time": "2h 30m",
            "gpu_utilization": 85,
            "memory_usage": "12.5GB"
        }
        
        return create_response(
            status="success",
            data={
                "job_id": job_id,
                "metrics": metrics,
                "last_updated": datetime.now().isoformat()
            },
            message=f"Training metrics for job {job_id}"
        )
        
    except HTTPException:
        raise
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