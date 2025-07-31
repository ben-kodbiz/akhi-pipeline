# Phase 9: Production Dockerization

## Overview

Phase 9 implements comprehensive Docker containerization for the Akhi Data Builder project, providing production-ready deployment capabilities with monitoring, security, and scalability features.

## 🐳 Docker Architecture

### Core Services

1. **akhi-crewai** - Main CrewAI application
2. **akhi-frontend** - React web interface
3. **redis** - Caching and task queues
4. **nginx** - Reverse proxy and load balancer
5. **prometheus** - Metrics collection (optional)
6. **grafana** - Monitoring dashboards (optional)

### Network Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Internet      │    │   Docker Host   │
│                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          │              ┌───────▼───────┐
          │              │     Nginx     │
          │              │   (Port 80)   │
          │              └───────┬───────┘
          │                      │
          │              ┌───────▼───────┐
          │              │   Frontend    │
          │              │  (Port 3000)  │
          │              └───────────────┘
          │                      │
          │              ┌───────▼───────┐
          │              │   CrewAI API  │
          │              │  (Port 8000)  │
          │              └───────┬───────┘
          │                      │
          │              ┌───────▼───────┐
          │              │     Redis     │
          │              │  (Port 6379)  │
          │              └───────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ disk space

### Production Deployment

```bash
# Clone and setup
git clone <repository>
cd akhi_data_builder

# Initial setup
./deploy.sh setup

# Start core services
./deploy.sh start

# Start with monitoring
./deploy.sh start monitoring
```

### Development Setup

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access services:
# - Frontend: http://localhost:3000
# - API: http://localhost:8000
# - Jupyter: http://localhost:8888 (token: akhi-dev-token)
# - Redis Commander: http://localhost:8081
```

## 📁 File Structure

```
akhi_data_builder/
├── Dockerfile                    # Main application container
├── Dockerfile.dev               # Development container
├── Dockerfile.jupyter           # Jupyter notebook container
├── docker-compose.yml           # Production orchestration
├── docker-compose.dev.yml       # Development orchestration
├── deploy.sh                    # Deployment automation script
├── .env.docker                  # Environment template
├── docker/
│   ├── nginx/
│   │   └── nginx.conf          # Nginx configuration
│   ├── prometheus/
│   │   └── prometheus.yml      # Metrics configuration
│   └── grafana/
│       ├── datasources/
│       │   └── prometheus.yml  # Grafana datasource
│       └── dashboards/
│           └── dashboard.yml   # Dashboard configuration
└── frontend_web/
    ├── Dockerfile              # Frontend production container
    ├── Dockerfile.dev          # Frontend development container
    └── nginx.conf              # Frontend nginx config
```

## 🔧 Configuration

### Environment Variables

Copy `.env.docker` to `.env` and customize:

```bash
# Application Settings
APP_NAME=akhi-data-builder
ENVIRONMENT=production
API_PORT=8000

# Security
JWT_SECRET_KEY=your-secret-key
API_KEY=your-api-key

# Model Configuration
MODEL_PATH=/app/models
QLORA_ENABLED=true

# Monitoring
METRICS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin123
```

### Resource Limits

Configure in `docker-compose.yml`:

```yaml
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

## 🛠️ Deployment Commands

### Basic Operations

```bash
# Setup and build
./deploy.sh setup

# Start services
./deploy.sh start
./deploy.sh start monitoring  # With monitoring

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# Check status
./deploy.sh status

# View logs
./deploy.sh logs
./deploy.sh logs akhi-crewai  # Specific service
```

### Data Management

```bash
# Create backup
./deploy.sh backup

# Restore from backup
./deploy.sh restore backup_20231201_120000.tar.gz

# Update application
./deploy.sh update

# Cleanup unused resources
./deploy.sh cleanup
```

## 📊 Monitoring

### Prometheus Metrics

Access at `http://localhost:9090`

**Available Metrics:**
- Application performance
- API response times
- Resource usage
- Error rates
- Pipeline progress

### Grafana Dashboards

Access at `http://localhost:3001` (admin/admin123)

**Pre-configured Dashboards:**
- System Overview
- API Performance
- Pipeline Monitoring
- Resource Usage

### Health Checks

```bash
# Check service health
curl http://localhost/health

# Check API status
curl http://localhost/api/status

# Check metrics
curl http://localhost/metrics
```

## 🔒 Security Features

### Network Security
- Isolated Docker networks
- Rate limiting on API endpoints
- Nginx security headers
- Internal service communication

### Authentication
- JWT token authentication
- API key protection
- Configurable token expiration
- Secure secret management

### SSL/TLS (Production)

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem

# Update nginx.conf to enable HTTPS
# Set SSL_ENABLED=true in .env
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Akhi Data Builder

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          ./deploy.sh setup
          ./deploy.sh start
```

### Docker Registry

```bash
# Build and push images
docker build -t your-registry/akhi-crewai:latest .
docker push your-registry/akhi-crewai:latest

# Update docker-compose.yml to use registry images
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  akhi-crewai:
    deploy:
      replicas: 3
    
  nginx:
    depends_on:
      - akhi-crewai
```

### Load Balancing

Nginx automatically load balances between multiple backend instances.

### Resource Optimization

```bash
# Monitor resource usage
docker stats

# Adjust resource limits based on usage
# Update docker-compose.yml accordingly
```

## 🐛 Troubleshooting

### Common Issues

**Port Conflicts:**
```bash
# Check port usage
sudo netstat -tulpn | grep :8000

# Update ports in docker-compose.yml
```

**Memory Issues:**
```bash
# Check memory usage
docker stats

# Increase memory limits
# Add swap if needed
```

**Permission Issues:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER data/ logs/ models/
chmod +x deploy.sh pipeline/run_pipeline.sh
```

### Debug Mode

```bash
# Start in debug mode
docker-compose -f docker-compose.dev.yml up

# Attach debugger (VS Code)
# Connect to localhost:5678
```

### Log Analysis

```bash
# View detailed logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs > application.log

# Monitor in real-time
docker-compose logs -f akhi-crewai | grep ERROR
```

## 🔄 Backup and Recovery

### Automated Backups

```bash
# Setup cron job for daily backups
echo "0 2 * * * cd /path/to/akhi_data_builder && ./deploy.sh backup" | crontab -
```

### Disaster Recovery

```bash
# Full system restore
./deploy.sh stop
./deploy.sh restore latest_backup.tar.gz
./deploy.sh start
```

## 📋 Maintenance

### Regular Tasks

```bash
# Weekly cleanup
./deploy.sh cleanup

# Update dependencies
docker-compose pull
./deploy.sh update

# Security updates
docker system prune -a
```

### Performance Tuning

1. **Monitor resource usage**
2. **Adjust container limits**
3. **Optimize database queries**
4. **Configure caching**
5. **Update hardware as needed**

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Log rotation configured
- [ ] Security headers enabled
- [ ] Rate limiting configured
- [ ] Health checks working
- [ ] Documentation updated

## 🔗 Related Documentation

- [Phase 8: QLoRA Integration](PHASE8_QLORA_INTEGRATION.md)
- [Production Testing Plan](../project/QLoRA_Production_Testing_Plan.md)
- [User Guide](../USER_GUIDE.md)
- [API Documentation](../integration/API_REFERENCE.md)

## 📞 Support

For deployment issues:
1. Check logs: `./deploy.sh logs`
2. Verify configuration: `./deploy.sh status`
3. Review documentation
4. Contact support team

---

**Phase 9 Status: ✅ Complete**

**Next Steps:**
- Production deployment
- Performance optimization
- User training
- Community feedback
- Advanced monitoring features