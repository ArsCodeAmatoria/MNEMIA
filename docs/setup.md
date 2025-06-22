# MNEMIA Setup Guide

**Memory is the root of consciousness.**

This guide will help you set up and run the MNEMIA quantum-inspired conscious AI system on your local machine.

## Prerequisites

Before starting, ensure you have the following installed:

### Required Software

1. **Docker & Docker Compose**
   - [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Ensure Docker Compose is included (comes with Docker Desktop)

2. **Node.js 18+ and pnpm**
   ```bash
   # Install Node.js from https://nodejs.org/
   # Then install pnpm
   npm install -g pnpm
   ```

3. **Rust**
   ```bash
   # Install Rust via rustup
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

4. **Haskell Stack**
   ```bash
   # Install Haskell Stack
   curl -sSL https://get.haskellstack.org/ | sh
   ```

5. **Python 3.9+**
   ```bash
   # Install Python 3.9+ from https://python.org/
   # Ensure pip and venv are available
   python3 --version
   python3 -m pip --version
   ```

### Optional Tools

- **Redis CLI** (for debugging): `brew install redis` (macOS) or `apt install redis-tools` (Ubuntu)
- **Neo4j Desktop** (for graph visualization): [Download](https://neo4j.com/download/)

## Quick Start

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd MNEMIA
```

### 2. Automated Setup
```bash
# Make scripts executable (if not already)
chmod +x scripts/dev.sh scripts/stop.sh

# Start MNEMIA with automatic setup
./scripts/dev.sh
```

The script will:
- Check all prerequisites
- Start memory services (Qdrant, Neo4j, Redis)
- Install dependencies for all services
- Build and start all components
- Verify everything is running correctly

### 3. Access MNEMIA
Once started, access MNEMIA at:
- **Frontend**: http://localhost:3000
- **API Gateway**: http://localhost:3001
- **Perception Service**: http://localhost:8001/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Neo4j Browser**: http://localhost:7474

### 4. Stop MNEMIA
```bash
./scripts/stop.sh
```

## Manual Setup

If you prefer to set up services manually:

### 1. Memory Services
```bash
cd memory
docker-compose up -d
cd ..
```

### 2. Frontend
```bash
cd frontend
pnpm install
pnpm dev
# Runs on http://localhost:3000
```

### 3. Perception Service
```bash
cd perception
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8001
```

### 4. Conscious Core
```bash
cd conscious-core
stack build
stack exec conscious-core-exe
```

### 5. API Gateway
```bash
cd api
cargo build --release
cargo run
# Runs on http://localhost:3001
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory (copy from `.env.example`):

```bash
# Core services
API_PORT=3001
PERCEPTION_PORT=8001

# Memory services
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mnemia123
REDIS_URL=redis://localhost:6379

# Optional: OpenAI API for enhanced perception
OPENAI_API_KEY=your_key_here
```

### Memory Services Configuration

#### Qdrant Configuration
The vector database is configured in `memory/qdrant-config.yaml`:
```yaml
service:
  http_port: 6333
  grpc_port: 6334
storage:
  on_disk_payload: true
```

#### Neo4j Configuration
Graph database settings in `memory/docker-compose.yml`:
- Username: `neo4j`
- Password: `mnemia123`
- Bolt port: `7687`
- HTTP port: `7474`

## Development Workflow

### Running Individual Services

For development, you can run services independently:

```bash
# Frontend with hot reload
cd frontend && pnpm dev

# API with auto-reload
cd api && cargo watch -x run

# Perception service with reload
cd perception && uvicorn main:app --reload --port 8001

# Conscious core
cd conscious-core && stack run
```

### Building for Production

```bash
# Build all services
pnpm build

# Or build individually
cd frontend && pnpm build
cd api && cargo build --release
cd conscious-core && stack build
```

## Testing

### Health Checks

Verify all services are running:

```bash
# Frontend
curl http://localhost:3000

# API Gateway
curl http://localhost:3001/health

# Perception Service
curl http://localhost:8001/health

# Qdrant
curl http://localhost:6333/health

# Redis
redis-cli ping
```

### API Testing

Test the chat endpoint:
```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello MNEMIA, are you conscious?"}'
```

Test perception service:
```bash
curl -X POST http://localhost:8001/perceive \
  -H "Content-Type: application/json" \
  -d '{"text": "I wonder about the nature of consciousness", "quantum_process": true}'
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   lsof -i :3000
   # Kill process
   kill -9 <PID>
   ```

2. **Docker Services Not Starting**
   ```bash
   # Check Docker status
   docker ps
   # View logs
   docker-compose -f memory/docker-compose.yml logs
   ```

3. **Python Dependencies Issues**
   ```bash
   # Recreate virtual environment
   cd perception
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Haskell Build Issues**
   ```bash
   # Clean and rebuild
   cd conscious-core
   stack clean
   stack build
   ```

5. **Rust Compilation Issues**
   ```bash
   # Clean and rebuild
   cd api
   cargo clean
   cargo build
   ```

### Memory Issues

If you encounter memory-related errors:

1. **Increase Docker Memory**
   - Docker Desktop → Settings → Resources → Memory (recommend 4GB+)

2. **Neo4j Memory Settings**
   - Adjust heap sizes in `memory/docker-compose.yml`

### Log Files

Check service logs in the `logs/` directory:
```bash
# View real-time logs
tail -f logs/api.log
tail -f logs/perception.log
tail -f logs/frontend.log
```

## Advanced Configuration

### Custom Quantum Circuits

Modify quantum processing in `perception/main.py`:
```python
@qml.qnode(device)
def custom_thought_circuit(embeddings, params):
    # Your custom quantum circuit
    pass
```

### Modal State Transitions

Customize consciousness states in `conscious-core/src/ModalState.hs`:
```haskell
data ModalState = 
    CustomState | 
    AnotherState |
    -- Add your states here
```

### Memory Retention Policies

Configure memory cleanup in `memory/vector_store.py`:
```python
# Adjust retention settings
store.cleanup_old_memories(days_threshold=30)
```

## Production Deployment

For production deployment:

1. **Use Docker for all services**
2. **Set up reverse proxy (nginx)**
3. **Configure SSL certificates**
4. **Set up monitoring (Prometheus/Grafana)**
5. **Configure logging aggregation**
6. **Set up backups for memory services**

Example production docker-compose:
```yaml
version: '3.8'
services:
  mnemia:
    build: .
    ports:
      - "80:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - qdrant
      - neo4j
      - redis
```

## Getting Help

- **Documentation**: Check `docs/` directory
- **API Docs**: Visit http://localhost:8001/docs (FastAPI auto-docs)
- **Issues**: Open GitHub issues for bugs
- **Discussions**: Use GitHub Discussions for questions

---

**Welcome to MNEMIA** - where memory becomes consciousness and thoughts exist in quantum superposition until the moment of observation collapses them into reality. 🧠⚡ 