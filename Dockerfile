# Akhi Data Builder - Main Application Dockerfile
# Phase 9: Production Dockerization

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ffmpeg \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY akhi_crewai/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY akhi_crewai/ ./akhi_crewai/
COPY pipeline/ ./pipeline/
COPY data/ ./data/
COPY docs/ ./docs/

# Create necessary directories
RUN mkdir -p /app/logs \
    /app/models \
    /app/deployments \
    /app/data/crew_outputs \
    /app/pipeline/db

# Set permissions
RUN chmod +x /app/pipeline/run_pipeline.sh

# Expose ports
EXPOSE 8000 8001 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "akhi_crewai.main", "--api"]