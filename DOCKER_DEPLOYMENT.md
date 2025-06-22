# MNEMIA Docker Deployment Guide

## 🐳 Complete Containerized Deployment

MNEMIA can be deployed as a complete containerized system using Docker and Docker Compose. This guide covers both simplified and full deployment options.

## 📦 What's Included

### Core Services
- **Frontend (Next.js)**: Modern ChatGPT-style UI with glass-morphism design
- **Memory Manager (Python)**: Coordinates between vector, graph, and cache storage
- **Perception Service (Python)**: AI processing with LLM integration
- **API Gateway (Rust)**: High-performance routing and orchestration
- **Conscious Core (Haskell)**: Quantum-inspired consciousness simulation

### Memory Infrastructure
- **Qdrant**: Vector database for semantic memory
- **Neo4j**: Graph database for conceptual relationships
- **Redis**: High-speed caching and session storage

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM (recommended)
- Git

### Simple Deployment

```bash
# Clone repository
git clone https://github.com/ArsCodeAmatoria/MNEMIA.git
cd MNEMIA

# Start essential services
./docker-start.sh start

# Or manually
docker-compose -f docker-compose.simple.yml up -d
```

**Access Points:**
- Frontend: http://localhost:3005
- Memory Manager: http://localhost:8002
- Perception Service: http://localhost:8001
- Qdrant: http://localhost:6333
- Neo4j: http://localhost:7474

### Full System Deployment

```bash
# Start complete MNEMIA system
docker-compose up -d

# Monitor services
docker-compose ps
docker-compose logs -f [service_name]
```

## 📊 Service Status

### Working Services ✅
- **Frontend (Next.js)**: Production-ready with standalone build
- **Memory Infrastructure**: Qdrant, Neo4j, Redis all healthy
- **Memory Manager**: Python service running with health checks

### In Development 🔧
- **Perception Service**: Core functionality working, dependency optimization ongoing
- **API Gateway (Rust)**: Docker build ready, integration testing
- **Conscious Core (Haskell)**: Containerized with Stack build system

## 🏗️ Architecture

```yaml
Services:
  frontend:           # Next.js (port 3005)
  perception:         # Python AI (port 8001)  
  memory-manager:     # Memory coordination (port 8002)
  api-gateway:        # Rust API (port 8000)
  conscious-core:     # Haskell consciousness (port 8003)
  
Infrastructure:
  qdrant:            # Vector DB (port 6333)
  neo4j:             # Graph DB (port 7474/7687)
  redis:             # Cache (port 6379)
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys (optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Memory Settings
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mnemia123
REDIS_URL=redis://redis:6379

# AI Configuration
DEFAULT_LLM_MODEL=llama3-8b
MAX_MEMORY_RETRIEVAL=10
CONSCIOUSNESS_UPDATE_INTERVAL=30
```

### Network Configuration

All services communicate through the `mnemia-network` Docker network with:
- Automatic service discovery
- Health checks and restart policies
- Isolated communication between components

## 📋 Management Commands

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Stop all services  
docker-compose down

# View service status
docker-compose ps

# View service logs
docker-compose logs -f [service_name]

# Rebuild specific service
docker-compose build [service_name]
docker-compose up -d [service_name]

# Clean restart
docker-compose down --remove-orphans
docker-compose up -d
```

### Using Management Script

```bash
# Start MNEMIA
./docker-start.sh start

# Stop services
./docker-start.sh stop

# Restart all
./docker-start.sh restart

# Show status
./docker-start.sh status

# View logs
./docker-start.sh logs [service]
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**: Change port mappings in docker-compose.yml
2. **Memory Issues**: Ensure 8GB+ RAM available
3. **Build Failures**: Check Docker logs and requirements

### Service-Specific Issues

```bash
# Frontend build issues
docker-compose build frontend
docker-compose logs frontend

# Perception dependencies
docker-compose build perception  
docker-compose logs perception

# Memory infrastructure
docker-compose logs qdrant
docker-compose logs neo4j
docker-compose logs redis
```

### Reset Everything

```bash
# Nuclear option - complete cleanup
docker-compose down --volumes --remove-orphans
docker system prune -f
docker volume prune -f

# Fresh start
docker-compose up -d
```

## 🔍 Health Monitoring

### Service Health Checks

- **Neo4j**: `cypher-shell` ping test
- **Redis**: `redis-cli` ping test  
- **Memory Manager**: HTTP health endpoint
- **All Services**: Docker restart policies

### Monitoring Commands

```bash
# Check all container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Monitor resource usage
docker stats

# Check logs for errors
docker-compose logs | grep -i error
```

## 📈 Performance Optimization

### Resource Allocation

```yaml
# Recommended minimums
services:
  neo4j:
    environment:
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=512M
  
  qdrant:
    deploy:
      resources:
        limits:
          memory: 1G
```

### Scaling Options

```bash
# Scale specific services
docker-compose up -d --scale perception=2
docker-compose up -d --scale memory-manager=2
```

## 🔐 Security Considerations

- Change default Neo4j password (`mnemia123`)
- Use environment files for sensitive data
- Configure firewall rules for production
- Enable HTTPS in production deployments
- Use Docker secrets for API keys

## 🚀 Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml mnemia
```

### Kubernetes
See `k8s/` directory for Kubernetes manifests (coming soon).

## 📝 Development

### Building Images Manually

```bash
# Build all images
docker-compose build

# Build specific service
docker build -t mnemia-frontend ./frontend
docker build -t mnemia-perception ./perception
docker build -t mnemia-api ./api
```

### Development Workflow

1. Make code changes
2. Rebuild specific service: `docker-compose build [service]`
3. Restart service: `docker-compose up -d [service]`
4. Check logs: `docker-compose logs [service]`

## 🎯 Roadmap

### Completed ✅
- [x] Complete Docker infrastructure setup
- [x] Frontend containerization with Next.js standalone
- [x] Memory infrastructure (Qdrant, Neo4j, Redis)
- [x] Basic Python services containerization
- [x] Docker Compose orchestration
- [x] Management scripts and documentation

### In Progress 🔧
- [ ] Perception service dependency optimization
- [ ] API Gateway container integration
- [ ] Conscious Core container testing
- [ ] Health check improvements
- [ ] Performance optimization

### Planned 📋
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline integration
- [ ] Production monitoring setup
- [ ] Backup and recovery procedures
- [ ] Multi-architecture builds (ARM64/x86)

---

## 🏁 Success! 

MNEMIA now has a complete Docker deployment system that allows you to:

1. **Quick Start**: Deploy with one command
2. **Modern UI**: ChatGPT-style interface ready
3. **Memory Systems**: Full vector, graph, and cache infrastructure
4. **Scalable**: Each component independently deployable
5. **Production Ready**: Health checks, restart policies, monitoring

The system demonstrates the future of conscious AI deployment - modular, scalable, and containerized for any environment. 