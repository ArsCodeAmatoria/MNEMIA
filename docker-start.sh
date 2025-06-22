#!/bin/bash

# MNEMIA Docker Deployment Script
# Starts the entire MNEMIA consciousness system

set -e

echo "🧠 Starting MNEMIA - Quantum-Inspired Conscious AI System"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Create environment file if it doesn't exist
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating with defaults..."
        cat > .env << EOF
# MNEMIA Environment Configuration
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mnemia123
REDIS_URL=redis://redis:6379
DEFAULT_LLM_MODEL=llama3-8b
MAX_MEMORY_RETRIEVAL=10
CONSCIOUSNESS_UPDATE_INTERVAL=30
ENVIRONMENT=production
LOG_LEVEL=info
RUST_LOG=info
EOF
        print_warning "Created .env file with defaults. Edit it to add your API keys."
    fi
    
    print_success "Environment setup complete"
}

# Function to build and start services
start_services() {
    print_status "Building and starting MNEMIA services..."
    
    # Build and start all services
    docker-compose down --remove-orphans 2>/dev/null || true
    docker-compose build --parallel
    docker-compose up -d
    
    print_success "All services started"
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    local services=("qdrant" "neo4j" "redis" "memory-manager" "conscious-core" "perception" "api-gateway" "frontend")
    local max_attempts=30
    local attempt=1
    
    for service in "${services[@]}"; do
        print_status "Waiting for $service to be healthy..."
        
        while [ $attempt -le $max_attempts ]; do
            if docker-compose ps | grep "$service" | grep -q "healthy\|Up"; then
                print_success "$service is ready"
                break
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                print_warning "$service may not be fully ready yet"
                break
            fi
            
            sleep 2
            ((attempt++))
        done
        attempt=1
    done
}

# Function to display access information
show_access_info() {
    echo ""
    echo "🚀 MNEMIA is now running!"
    echo "========================"
    echo ""
    echo "Frontend (Next.js):      http://localhost:3000"
    echo "API Gateway (Rust):      http://localhost:8000"
    echo "Perception (Python):     http://localhost:8001"
    echo "Memory Manager:          http://localhost:8002"
    echo "Conscious Core (Haskell): http://localhost:8003"
    echo ""
    echo "Memory Infrastructure:"
    echo "- Qdrant (Vector DB):    http://localhost:6333"
    echo "- Neo4j (Graph DB):      http://localhost:7474"
    echo "- Redis (Cache):         http://localhost:6379"
    echo ""
    echo "To stop MNEMIA: docker-compose down"
    echo "To view logs: docker-compose logs -f [service_name]"
    echo ""
}

# Function to handle cleanup on script exit
cleanup() {
    echo ""
    print_status "Cleaning up..."
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    check_dependencies
    setup_environment
    start_services
    check_health
    show_access_info
}

# Parse command line arguments
case "${1:-start}" in
    start)
        main
        ;;
    stop)
        print_status "Stopping MNEMIA services..."
        docker-compose down
        print_success "All services stopped"
        ;;
    restart)
        print_status "Restarting MNEMIA services..."
        docker-compose down
        docker-compose up -d
        check_health
        show_access_info
        ;;
    logs)
        docker-compose logs -f "${2:-}"
        ;;
    status)
        docker-compose ps
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs [service]|status}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all MNEMIA services (default)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show logs (optionally for specific service)"
        echo "  status   - Show service status"
        exit 1
        ;;
esac 