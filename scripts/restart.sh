#!/bin/bash

# Open WebUI Chat Analyzer Restart Script
# Restarts the application based on the detected deployment method

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check for Docker setup
check_docker_setup() {
    if [ -f "docker-compose.yml" ] && [ -f "Dockerfile" ]; then
        if command -v docker &> /dev/null && docker info &> /dev/null; then
            if command -v make &> /dev/null; then
                return 0  # Docker with Make
            elif docker compose version &> /dev/null || command -v docker-compose &> /dev/null; then
                return 1  # Docker without Make
            fi
        fi
    fi
    return 2  # No Docker setup
}

# Restart with Docker (Make)
restart_docker_make() {
    print_info "🐳 Restarting services with Make..."

    if make restart; then
        print_status "🔄 Application restarted successfully!"
        return 0
    else
        print_error "Failed to restart services with Make"
        return 1
    fi
}

# Restart with Docker Compose
restart_docker_compose() {
    print_info "🐳 Restarting services with Docker Compose..."

    if docker compose version &> /dev/null; then
        COMPOSE_CMD=(docker compose)
    else
        COMPOSE_CMD=(docker-compose)
    fi

    local restarted=0

    if docker ps --filter name=openwebui-chat-analyzer --format "{{.Names}}" | grep -q openwebui-chat-analyzer; then
        if "${COMPOSE_CMD[@]}" restart openwebui-chat-analyzer; then
            restarted=1
        fi
    fi

    if docker ps --filter name=openwebui-chat-analyzer-dev --format "{{.Names}}" | grep -q openwebui-chat-analyzer-dev; then
        if "${COMPOSE_CMD[@]}" restart openwebui-chat-analyzer-dev; then
            restarted=1
        fi
    fi

    if [ "$restarted" -eq 0 ]; then
        print_warning "No running containers detected. Starting services instead..."
        if "${COMPOSE_CMD[@]}" up -d; then
            print_status "🚀 Application started!"
            return 0
        else
            print_error "Failed to start services with Docker Compose"
            return 1
        fi
    else
        print_status "🔄 Application restarted successfully!"
        return 0
    fi
}

echo "🔄 Open WebUI Chat Analyzer Restart"
echo "==================================="

DOCKER_STATUS=$(check_docker_setup; echo $?)

case $DOCKER_STATUS in
    0)
        restart_docker_make
        ;;
    1)
        restart_docker_compose
        ;;
    2)
        print_error "No Docker setup detected!"
        echo ""
        echo "If you are running via Python virtual environment:"
        echo "  1. Stop the current Streamlit process (Ctrl+C)"
        echo "  2. Re-run ./run.sh to start it again"
        exit 1
        ;;
esac

