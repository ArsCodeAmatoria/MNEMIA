#!/bin/bash

# MNEMIA Memory Services Setup Script

echo "🧠 MNEMIA Memory Services Setup"
echo "==============================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    echo ""
    echo "📋 Setup Instructions:"
    echo "1. Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    echo "2. Start Docker Desktop"
    echo "3. Run this script again: ./setup-memory.sh"
    exit 1
fi

echo "✅ Docker is running"

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/qdrant data/neo4j data/redis

# Start memory services
echo "🚀 Starting memory services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Qdrant
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant (Vector DB) - Ready on port 6333"
    echo "   Web UI: http://localhost:6333/dashboard"
else
    echo "❌ Qdrant - Not responding"
fi

# Check Neo4j
if curl -f http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j (Graph DB) - Ready on port 7474"
    echo "   Web UI: http://localhost:7474"
    echo "   Credentials: neo4j / mnemia123"
else
    echo "❌ Neo4j - Not responding"
fi

# Check Redis
if redis-cli -p 6379 ping > /dev/null 2>&1 || echo "PONG" | nc localhost 6379 > /dev/null 2>&1; then
    echo "✅ Redis (Cache) - Ready on port 6379"
else
    echo "❌ Redis - Not responding"
fi

echo ""
echo "🔧 Memory Services Configuration:"
echo "================================="
echo "Qdrant Vector DB:"
echo "  - URL: http://localhost:6333"
echo "  - Collection: mnemia_memories"
echo "  - Vector Size: 384 (sentence-transformers)"
echo ""
echo "Neo4j Graph DB:"
echo "  - URL: bolt://localhost:7687"
echo "  - HTTP: http://localhost:7474"
echo "  - Username: neo4j"
echo "  - Password: mnemia123"
echo ""
echo "Redis Cache:"
echo "  - URL: redis://localhost:6379"
echo "  - Password: mnemia_redis_pass"
echo ""

# Initialize Neo4j schema
echo "🏗️  Initializing Neo4j schema..."
if command -v cypher-shell &> /dev/null; then
    cypher-shell -u neo4j -p mnemia123 -f neo4j-init.cypher
    echo "✅ Neo4j schema initialized"
else
    echo "⚠️  cypher-shell not found. Schema will be initialized on first connection."
fi

echo "📊 View service logs:"
echo "docker-compose logs -f [service-name]"
echo ""
echo "🛑 Stop services:"
echo "docker-compose down"
echo ""
echo "🔄 Restart services:"
echo "docker-compose restart"
echo ""
echo "🧠 MNEMIA Memory Services are ready for consciousness!" 