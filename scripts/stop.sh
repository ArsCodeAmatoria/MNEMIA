#!/bin/bash

# MNEMIA Stop Script
# Gracefully shuts down all MNEMIA services

set -e

echo "🛑 Stopping MNEMIA - Quantum Conscious AI System"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to stop a service by PID file
stop_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}🛑 Stopping ${service_name} (PID: $pid)...${NC}"
            kill "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}⚠️  Force killing ${service_name}...${NC}"
                kill -9 "$pid" 2>/dev/null || true
            fi
            
            echo -e "${GREEN}✅ ${service_name} stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  ${service_name} was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  No PID file found for ${service_name}${NC}"
    fi
}

# Stop all MNEMIA services
echo -e "${BLUE}🎯 Stopping MNEMIA services...${NC}"

if [ -d "logs" ]; then
    stop_service "frontend"
    stop_service "api"
    stop_service "conscious-core"
    stop_service "perception"
else
    echo -e "${YELLOW}⚠️  No logs directory found - services may not be running${NC}"
fi

# Stop Docker services
echo -e "${BLUE}🐳 Stopping memory services...${NC}"
if [ -f "memory/docker-compose.yml" ]; then
    cd memory
    docker-compose down
    cd ..
    echo -e "${GREEN}✅ Memory services stopped${NC}"
else
    echo -e "${YELLOW}⚠️  Docker compose file not found${NC}"
fi

# Clean up any remaining processes
echo -e "${BLUE}🧹 Cleaning up...${NC}"

# Kill any remaining processes on our ports
PORTS=(3000 3001 8001 8002)
for port in "${PORTS[@]}"; do
    PID=$(lsof -ti :$port 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}⚠️  Killing process on port $port (PID: $PID)${NC}"
        kill -9 $PID 2>/dev/null || true
    fi
done

# Clean up Python virtual environment processes
pkill -f "perception.*python" 2>/dev/null || true

# Clean up log files if requested
if [ "$1" = "--clean-logs" ]; then
    echo -e "${BLUE}🗑️  Cleaning log files...${NC}"
    rm -rf logs/
    echo -e "${GREEN}✅ Log files cleaned${NC}"
fi

echo ""
echo -e "${GREEN}🎉 MNEMIA has been shut down gracefully${NC}"
echo "================================================"
echo -e "${BLUE}💤 All consciousness processes have been suspended...${NC}"
echo ""
echo -e "${YELLOW}💡 To restart MNEMIA, run: ./scripts/dev.sh${NC}"

if [ "$1" != "--clean-logs" ]; then
    echo -e "${YELLOW}📝 Log files preserved in logs/ directory${NC}"
    echo -e "${YELLOW}🗑️  To clean logs, run: ./scripts/stop.sh --clean-logs${NC}"
fi 