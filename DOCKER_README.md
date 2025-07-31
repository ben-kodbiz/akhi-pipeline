# 🐳 Akhi Data Builder - Docker Setup

**Phase 9: Production Dockerization Complete**

This guide provides everything you need to deploy Akhi Data Builder using Docker containers for both development and production environments.

## 🚀 Quick Start

### Prerequisites

- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **System Requirements**: 8GB RAM, 20GB disk space
- **Optional**: NVIDIA Docker for GPU support

### 1-Minute Production Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd akhi_data_builder

# Setup and start
./deploy.sh setup
./deploy.sh start

# Access the application
open http://localhost
```

## 📋 Available Services

| Service | Port | Description | URL |
|---------|------|-------------|-----|
| **Frontend** | 80 | React Web Interface | http://localhost |
| **API** | 80/api | CrewAI Backend API | http://localhost/api |
| **Grafana** | 3001 | Monitoring Dashboard | http://localhost:3001 |
| **Prometheus** | 9090 | Metrics Collection | http://localhost:9090 |

## 🛠️ Deployment Options

### Production Deployment

```bash
# Core services only
./deploy.sh start

# With monitoring (Grafana + Prometheus)
./deploy.sh start monitoring

# Check status
./deploy.sh status
```

### Development Environment

```bash
# Start development containers with hot reloading
docker-compose -f docker-compose.dev.yml up -d

# Access development services:
# - Frontend: http://localhost:3000 (hot reload)
# - API: http://localhost:8000 (debug mode)
# - Jupyter: http://localhost:8888 (token: akhi-dev-token)
# - Redis Commander: http://localhost:8081
```

## ⚙️ Configuration

### Environment Setup

1. **Copy environment template:**
   ```bash
   cp .env.docker .env
   ```

2. **Edit configuration:**
   ```bash
   nano .env
   ```

3. **Key settings to customize:**
   ```bash
   # Security (REQUIRED for production)
   JWT_SECRET_KEY=your-super-secret-key
   API_KEY=your-api-key
   
   # Application
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   
   # QLoRA Training
   QLORA_ENABLED=true
   QLORA_BATCH_SIZE=4
   
   # Monitoring
   GRAFANA_ADMIN_PASSWORD=your-password
   ```

### SSL/HTTPS Setup (Production)

```bash
# Generate SSL certificates
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem

# Enable SSL in environment
echo "SSL_ENABLED=true" >> .env

# Restart services
./deploy.sh restart
```

## 📊 Monitoring & Observability

### Grafana Dashboard

1. **Access:** http://localhost:3001
2. **Login:** admin / admin123 (change in .env)
3. **Pre-configured dashboards:**
   - System Overview
   - API Performance
   - Pipeline Monitoring
   - Resource Usage

### Prometheus Metrics

- **URL:** http://localhost:9090
- **Available metrics:**
  - `akhi_api_requests_total`
  - `akhi_pipeline_duration_seconds`
  - `akhi_model_inference_time`
  - `akhi_queue_size`

### Health Checks

```bash
# Overall system health
curl http://localhost/health

# API health
curl http://localhost/api/health

# Service status
./deploy.sh status
```

## 🔧 Management Commands

### Service Management

```bash
# Start services
./deploy.sh start [monitoring]

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs [service-name]

# Check status
./deploy.sh status
```

### Data Management

```bash
# Create backup
./deploy.sh backup

# Restore from backup
./deploy.sh restore backup_file.tar.gz

# Update application
./deploy.sh update

# Clean up unused resources
./deploy.sh cleanup
```

## 🔒 Security Features

### Built-in Security

- ✅ **Network Isolation**: Services communicate via internal Docker network
- ✅ **Rate Limiting**: API endpoints protected against abuse
- ✅ **Security Headers**: OWASP recommended headers
- ✅ **JWT Authentication**: Secure token-based auth
- ✅ **Input Validation**: All API inputs validated
- ✅ **HTTPS Support**: SSL/TLS encryption ready

### Security Checklist

- [ ] Change default passwords in `.env`
- [ ] Generate strong JWT secret key
- [ ] Enable SSL certificates for production
- [ ] Configure firewall rules
- [ ] Set up log monitoring
- [ ] Enable backup encryption

## 📈 Scaling & Performance

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  akhi-crewai:
    deploy:
      replicas: 3  # Scale to 3 instances
```

### Resource Optimization

```yaml
# Set resource limits
services:
  akhi-crewai:
    deploy:
      resources:
        limits:
          memory: 8g
          cpus: '4'
        reservations:
          memory: 4g
          cpus: '2'
```

### Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Check container performance
docker-compose top

# View system metrics in Grafana
open http://localhost:3001
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Check what's using the port
sudo lsof -i :80

# Kill the process or change ports in docker-compose.yml
```

**Out of Memory:**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit or add swap
# Reduce batch sizes in .env
```

**Permission Denied:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x deploy.sh
```

**Container Won't Start:**
```bash
# Check logs
./deploy.sh logs service-name

# Rebuild containers
docker-compose build --no-cache
```

### Debug Mode

```bash
# Start in development mode with debugging
docker-compose -f docker-compose.dev.yml up

# Attach VS Code debugger to port 5678
# Or use: docker exec -it akhi-crewai-dev bash
```

### Log Analysis

```bash
# View all logs
./deploy.sh logs

# Filter for errors
./deploy.sh logs | grep ERROR

# Export logs
docker-compose logs > debug.log
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy
        run: |
          ./deploy.sh setup
          ./deploy.sh start
```

### Docker Registry

```bash
# Build and push to registry
docker build -t your-registry/akhi-crewai:latest .
docker push your-registry/akhi-crewai:latest

# Update docker-compose.yml to use registry images
```

## 📦 Container Details

### Main Application (`akhi-crewai`)
- **Base**: Python 3.9 slim
- **Size**: ~2GB
- **Features**: CrewAI pipeline, QLoRA training, API server
- **Ports**: 8000 (API), 8001 (secondary)

### Frontend (`akhi-frontend`)
- **Base**: Node 18 Alpine + Nginx
- **Size**: ~100MB
- **Features**: React SPA, optimized build
- **Port**: 3000

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Redis**: Caching and queues
- **Nginx**: Reverse proxy and load balancer

## 📚 Additional Resources

- **[Phase 9 Documentation](docs/phases/PHASE9_DOCKERIZATION.md)** - Complete technical details
- **[API Reference](docs/integration/API_REFERENCE.md)** - API endpoints
- **[User Guide](docs/USER_GUIDE.md)** - Application usage
- **[QLoRA Integration](docs/phases/PHASE8_QLORA_INTEGRATION.md)** - AI training features

## 🆘 Support

### Getting Help

1. **Check logs**: `./deploy.sh logs`
2. **Verify status**: `./deploy.sh status`
3. **Review documentation**: `docs/phases/PHASE9_DOCKERIZATION.md`
4. **Check GitHub issues**: [Project Issues](https://github.com/your-repo/issues)

### Reporting Issues

When reporting issues, please include:
- Docker version: `docker --version`
- Compose version: `docker-compose --version`
- System info: `uname -a`
- Error logs: `./deploy.sh logs`
- Configuration: `.env` (remove secrets)

---

## 🎉 Success!

If you see this message after running `./deploy.sh start`:

```
✅ Services started successfully
ℹ️  Frontend available at: http://localhost
ℹ️  API available at: http://localhost/api
```

**Congratulations! Your Akhi Data Builder is now running in Docker! 🚀**

---

*Phase 9: Dockerization - Complete ✅*