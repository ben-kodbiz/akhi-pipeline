#!/bin/bash
# Akhi Data Builder - Docker Deployment Script
# Phase 9: Production Dockerization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="akhi-data-builder"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "All requirements satisfied"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment file from template..."
        cp .env.docker "$ENV_FILE"
        log_warning "Please edit $ENV_FILE with your configuration before proceeding"
        read -p "Press Enter to continue after editing $ENV_FILE..."
    fi
    
    # Create necessary directories
    mkdir -p logs models deployments data/crew_outputs pipeline/db docker/nginx/ssl
    
    # Set permissions
    chmod +x pipeline/run_pipeline.sh
    
    log_success "Environment setup complete"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build main application
    docker-compose build --no-cache akhi-crewai
    
    # Build frontend
    docker-compose build --no-cache akhi-frontend
    
    log_success "Docker images built successfully"
}

start_services() {
    local profile="${1:-default}"
    
    log_info "Starting services with profile: $profile"
    
    if [ "$profile" = "monitoring" ]; then
        docker-compose --profile monitoring up -d
    else
        docker-compose up -d akhi-crewai akhi-frontend redis nginx
    fi
    
    log_success "Services started successfully"
}

stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting services..."
    docker-compose restart
    log_success "Services restarted"
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    echo ""
    log_info "Service logs (last 20 lines):"
    docker-compose logs --tail=20
}

show_logs() {
    local service="${1:-}"
    
    if [ -n "$service" ]; then
        log_info "Showing logs for service: $service"
        docker-compose logs -f "$service"
    else
        log_info "Showing logs for all services:"
        docker-compose logs -f
    fi
}

backup_data() {
    log_info "Creating backup..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/akhi_backup_$timestamp.tar.gz"
    
    mkdir -p "$BACKUP_DIR"
    
    # Create backup
    tar -czf "$backup_file" \
        --exclude="logs/*.log" \
        --exclude="models/*/pytorch_model.bin" \
        data/ pipeline/db/ models/*/config.json models/*/tokenizer* \
        2>/dev/null || true
    
    log_success "Backup created: $backup_file"
}

restore_data() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_info "Restoring from backup: $backup_file"
    
    # Stop services
    docker-compose down
    
    # Restore data
    tar -xzf "$backup_file"
    
    # Start services
    start_services
    
    log_success "Data restored successfully"
}

cleanup() {
    log_info "Cleaning up..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    log_success "Cleanup complete"
}

update() {
    log_info "Updating application..."
    
    # Pull latest changes (if using git)
    if [ -d ".git" ]; then
        git pull
    fi
    
    # Rebuild images
    build_images
    
    # Restart services
    restart_services
    
    log_success "Update complete"
}

show_help() {
    echo "Akhi Data Builder - Docker Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup           Setup environment and build images"
    echo "  start [profile] Start services (default|monitoring)"
    echo "  stop            Stop all services"
    echo "  restart         Restart all services"
    echo "  status          Show service status and logs"
    echo "  logs [service]  Show logs for all services or specific service"
    echo "  backup          Create data backup"
    echo "  restore <file>  Restore from backup file"
    echo "  update          Update application and restart"
    echo "  cleanup         Clean up unused Docker resources"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Initial setup"
    echo "  $0 start                    # Start core services"
    echo "  $0 start monitoring         # Start with monitoring"
    echo "  $0 logs akhi-crewai         # Show logs for main app"
    echo "  $0 backup                   # Create backup"
    echo "  $0 restore backup.tar.gz    # Restore from backup"
}

# Main script
case "${1:-help}" in
    setup)
        check_requirements
        setup_environment
        build_images
        log_success "Setup complete! Run '$0 start' to start services."
        ;;
    start)
        check_requirements
        start_services "$2"
        echo ""
        log_info "Services are starting up. Use '$0 status' to check progress."
        log_info "Frontend will be available at: http://localhost:80"
        log_info "API will be available at: http://localhost:80/api"
        if [ "$2" = "monitoring" ]; then
            log_info "Grafana will be available at: http://localhost:3001 (admin/admin123)"
            log_info "Prometheus will be available at: http://localhost:9090"
        fi
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    backup)
        backup_data
        ;;
    restore)
        if [ -z "$2" ]; then
            log_error "Please specify backup file to restore"
            exit 1
        fi
        restore_data "$2"
        ;;
    update)
        update
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac