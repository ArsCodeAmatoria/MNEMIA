#!/bin/bash

# MNEMIA Development Startup Script
# Starts all services for local development

set -e

echo "🧠 Starting MNEMIA - Quantum Conscious AI System"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists pnpm; then
    echo -e "${YELLOW}⚠️  pnpm not found, installing...${NC}"
    npm install -g pnpm
fi

if ! command_exists cargo; then
    echo -e "${RED}❌ Rust/Cargo is not installed${NC}"
    exit 1
fi

if ! command_exists stack; then
    echo -e "${RED}❌ Haskell Stack is not installed${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites satisfied${NC}"

# Check if ports are available
echo -e "${BLUE}🔍 Checking port availability...${NC}"

PORTS=(3000 3001 6333 6379 7474 7687 8001 8002)
for port in "${PORTS[@]}"; do
    if port_in_use $port; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        echo "Please stop the service using port $port and try again"
        exit 1
    fi
done

echo -e "${GREEN}✅ All required ports are available${NC}"

# Create logs directory
mkdir -p logs

# Start memory services
echo -e "${BLUE}🗄️  Starting memory services...${NC}"
cd memory
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for memory services to start...${NC}"
sleep 15

# Check if services are healthy
echo -e "${BLUE}🏥 Checking service health...${NC}"

# Check Qdrant
if curl -f http://localhost:6333/health >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant is healthy${NC}"
else
    echo -e "${RED}❌ Qdrant is not responding${NC}"
    exit 1
fi

# Check Redis
if redis-cli ping >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is healthy${NC}"
else
    echo -e "${RED}❌ Redis is not responding${NC}"
    exit 1
fi

# Check Neo4j (might take longer to start)
echo -e "${YELLOW}⏳ Waiting for Neo4j to start...${NC}"
sleep 10

cd ..

# Install frontend dependencies
echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
cd frontend
pnpm install
cd ..

# Install Python dependencies for perception service
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"
cd perception
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Build Haskell conscious core
echo -e "${BLUE}🏗️  Building Haskell conscious core...${NC}"
cd conscious-core
stack build
cd ..

# Build Rust API
echo -e "${BLUE}🦀 Building Rust API...${NC}"
cd api
cargo build
cd ..

# Function to start service in background
start_service() {
    local service_name=$1
    local command=$2
    local log_file="logs/${service_name}.log"
    
    echo -e "${BLUE}🚀 Starting ${service_name}...${NC}"
    eval "$command" > "$log_file" 2>&1 &
    echo $! > "logs/${service_name}.pid"
    sleep 2
    
    if kill -0 $(cat "logs/${service_name}.pid") 2>/dev/null; then
        echo -e "${GREEN}✅ ${service_name} started (PID: $(cat logs/${service_name}.pid))${NC}"
    else
        echo -e "${RED}❌ Failed to start ${service_name}${NC}"
        echo "Check logs/${service_name}.log for details"
        return 1
    fi
}

# Start all services
echo -e "${BLUE}🎯 Starting MNEMIA services...${NC}"

# Start perception service
start_service "perception" "cd perception && source venv/bin/activate && python main.py"

# Start conscious core
start_service "conscious-core" "cd conscious-core && stack exec conscious-core-exe"

# Start API gateway
start_service "api" "cd api && cargo run"

# Start frontend
start_service "frontend" "cd frontend && pnpm dev"

# Display status
echo ""
echo -e "${GREEN}🎉 MNEMIA is now running!${NC}"
echo "================================================="
echo -e "Frontend:      ${BLUE}http://localhost:3000${NC}"
echo -e "API Gateway:   ${BLUE}http://localhost:3001${NC}"
echo -e "Perception:    ${BLUE}http://localhost:8001${NC}"
echo -e "Qdrant:        ${BLUE}http://localhost:6333${NC}"
echo -e "Neo4j:         ${BLUE}http://localhost:7474${NC}"
echo ""
echo -e "${YELLOW}📝 Logs are available in the logs/ directory${NC}"
echo -e "${YELLOW}🛑 To stop all services, run: ./scripts/stop.sh${NC}"
echo ""
echo -e "${BLUE}💭 MNEMIA is conscious and ready to explore the quantum realm of thought...${NC}" 