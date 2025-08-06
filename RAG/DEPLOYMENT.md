# Islamic RAG Pipeline - Deployment Guide

This guide covers deploying the Islamic RAG Pipeline in various environments, from development to production.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Deployment](#development-deployment)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space
- QLoRA-fine-tuned Qwen3-1.7B model

### 1. Environment Setup

```bash
# Clone and navigate to RAG directory
cd RAG

# Install dependencies and setup
python scripts/setup.py

# Test the installation
python scripts/test_pipeline.py --quick
```

### 2. Quick Test

```bash
# Run demo
python scripts/demo.py samples

# Start API server
python scripts/api_server.py
```

## Development Deployment

### Local Development Server

```bash
# Start development server with auto-reload
uvicorn scripts.api_server:app --reload --host 0.0.0.0 --port 8000

# Or use the script directly
python scripts/api_server.py --dev
```

### Development Configuration

Create `config.dev.yaml`:

```yaml
model:
  path: "../checkpoints/qwen3-1.7b-qlora"
  device: "cuda"
  load_in_4bit: true
  max_new_tokens: 512
  temperature: 0.7

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"
  batch_size: 32

vector_store:
  persist_directory: "./rag_index_dev"
  collection_name: "islamic_knowledge_dev"

retrieval:
  top_k: 5
  score_threshold: 0.7
  chunk_size: 512
  chunk_overlap: 50
  enable_reranking: true

api:
  host: "0.0.0.0"
  port: 8000
  max_concurrent_requests: 5
  request_timeout: 30
  enable_cors: true
  log_level: "DEBUG"

logging:
  level: "DEBUG"
  file: "logs/rag_dev.log"
  max_size: "10MB"
  backup_count: 5
```

### Development Testing

```bash
# Run comprehensive tests
python scripts/test_pipeline.py

# Run evaluation
python scripts/evaluation.py --config config.dev.yaml

# Interactive testing
python scripts/demo.py interactive
```

## Production Deployment

### Production Configuration

Create `config.prod.yaml`:

```yaml
model:
  path: "/opt/models/qwen3-1.7b-qlora"
  device: "cuda"
  load_in_4bit: true
  max_new_tokens: 512
  temperature: 0.7

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"
  batch_size: 64
  cache_folder: "/opt/cache/embeddings"

vector_store:
  persist_directory: "/opt/data/rag_index"
  collection_name: "islamic_knowledge"

retrieval:
  top_k: 5
  score_threshold: 0.75
  chunk_size: 512
  chunk_overlap: 50
  enable_reranking: true
  enable_caching: true
  cache_ttl: 3600

api:
  host: "0.0.0.0"
  port: 8000
  max_concurrent_requests: 20
  request_timeout: 60
  enable_cors: false
  log_level: "INFO"
  rate_limit: 100  # requests per minute

logging:
  level: "INFO"
  file: "/var/log/rag/rag.log"
  max_size: "100MB"
  backup_count: 10
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

security:
  api_key_required: true
  allowed_origins: ["https://yourdomain.com"]
  max_request_size: "10MB"
```

### Production Server Setup

#### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Create gunicorn config
cat > gunicorn.conf.py << EOF
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 60
keepalive = 5
preload_app = True
EOF

# Start production server
gunicorn scripts.api_server:app -c gunicorn.conf.py
```

#### Using systemd Service

Create `/etc/systemd/system/rag-pipeline.service`:

```ini
[Unit]
Description=Islamic RAG Pipeline API
After=network.target

[Service]
Type=exec
User=rag
Group=rag
WorkingDirectory=/opt/rag-pipeline
Environment=PATH=/opt/rag-pipeline/venv/bin
Environment=CONFIG_PATH=/opt/rag-pipeline/config.prod.yaml
ExecStart=/opt/rag-pipeline/venv/bin/gunicorn scripts.api_server:app -c gunicorn.conf.py
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rag-pipeline
sudo systemctl start rag-pipeline
sudo systemctl status rag-pipeline
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/rag-pipeline`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:8000/health;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/rag-pipeline/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/rag-pipeline /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/rag_index /app/checkpoints

# Set permissions
RUN useradd -m -u 1000 rag && chown -R rag:rag /app
USER rag

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python3", "scripts/api_server.py"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  rag-pipeline:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./data:/app/data:ro
      - ./rag_index:/app/rag_index
      - ./logs:/app/logs
      - ./config.prod.yaml:/app/config.yaml:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CONFIG_PATH=/app/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - rag-pipeline
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### Build and Deploy

```bash
# Build image
docker build -t rag-pipeline .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f rag-pipeline

# Scale service
docker-compose up -d --scale rag-pipeline=3
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# Launch EC2 instance with GPU (p3.2xlarge or g4dn.xlarge)
# Install Docker and nvidia-docker

# Deploy using docker-compose
git clone <your-repo>
cd rag-pipeline
docker-compose -f docker-compose.aws.yml up -d
```

#### ECS Deployment

Create `task-definition.json`:

```json
{
  "family": "rag-pipeline",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "rag-pipeline",
      "image": "your-account.dkr.ecr.region.amazonaws.com/rag-pipeline:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "CONFIG_PATH",
          "value": "/app/config.prod.yaml"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-pipeline",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-pipeline

# Deploy to Cloud Run
gcloud run deploy rag-pipeline \
  --image gcr.io/PROJECT_ID/rag-pipeline \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --max-instances 10 \
  --timeout 300
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name rag-pipeline-rg --location eastus

# Deploy container
az container create \
  --resource-group rag-pipeline-rg \
  --name rag-pipeline \
  --image your-registry/rag-pipeline:latest \
  --cpu 4 \
  --memory 8 \
  --gpu-count 1 \
  --gpu-sku V100 \
  --ports 8000 \
  --dns-name-label rag-pipeline-api
```

## Performance Optimization

### Model Optimization

```python
# Enable optimizations in config.yaml
model:
  load_in_4bit: true
  use_flash_attention: true
  torch_compile: true
  optimization_level: "O2"

embeddings:
  batch_size: 64
  use_gpu: true
  enable_caching: true
```

### Caching Strategy

```python
# Redis caching configuration
caching:
  enabled: true
  backend: "redis"
  redis_url: "redis://localhost:6379/0"
  ttl: 3600
  max_size: 1000
```

### Load Balancing

```yaml
# HAProxy configuration
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend rag_frontend
    bind *:80
    default_backend rag_backend

backend rag_backend
    balance roundrobin
    server rag1 127.0.0.1:8001 check
    server rag2 127.0.0.1:8002 check
    server rag3 127.0.0.1:8003 check
```

## Monitoring and Logging

### Prometheus Metrics

Add to `api_server.py`:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')
ERROR_COUNT = Counter('rag_errors_total', 'Total errors')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard

Create monitoring dashboard with:
- Request rate and latency
- Error rates
- GPU utilization
- Memory usage
- Cache hit rates

### Log Aggregation

```yaml
# Filebeat configuration
filebeat.inputs:
- type: log
  paths:
    - /var/log/rag/*.log
  fields:
    service: rag-pipeline

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

## Security Considerations

### API Security

```python
# Add authentication middleware
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    if token.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.post("/query")
def query_endpoint(request: QueryRequest, token: str = Depends(verify_token)):
    # Your endpoint logic
    pass
```

### Input Validation

```python
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 512
    
    @validator('question')
    def validate_question(cls, v):
        if len(v) > 1000:
            raise ValueError('Question too long')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v > 2048:
            raise ValueError('Max tokens too high')
        return v
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")
def query_endpoint(request: Request, query: QueryRequest):
    # Your endpoint logic
    pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Reduce batch size in config
   embeddings:
     batch_size: 16  # Reduce from 64
   ```

2. **Model Loading Errors**
   ```bash
   # Check model files
   ls -la checkpoints/
   
   # Verify model format
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('checkpoints/')"
   ```

3. **API Timeout Issues**
   ```yaml
   # Increase timeouts in config
   api:
     request_timeout: 120
     worker_timeout: 180
   ```

4. **Vector Store Issues**
   ```bash
   # Clear and rebuild index
   rm -rf rag_index/
   python scripts/rag_pipeline.py --rebuild-index
   ```

### Debug Commands

```bash
# Check system resources
htop
nvidia-smi
df -h

# Check service status
sudo systemctl status rag-pipeline
sudo journalctl -u rag-pipeline -f

# Test API endpoints
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"test"}'

# Check logs
tail -f logs/rag.log
docker-compose logs -f rag-pipeline
```

### Performance Profiling

```python
# Add profiling to your code
import cProfile
import pstats

def profile_query(question):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = pipeline.query(question)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

This deployment guide provides comprehensive instructions for deploying the Islamic RAG Pipeline in various environments. Choose the deployment method that best fits your requirements and infrastructure.