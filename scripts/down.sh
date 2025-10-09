#!/bin/bash

# Open WebUI Chat Analyzer Shutdown Script
# Stops the application based on the detected deployment method

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

# Stop with Docker (Make)
stop_docker_make() {
    print_info "🐳 Stopping services with Make..."

    if make down; then
        print_status "🛑 Application stopped successfully!"
        return 0
    else
        print_error "Failed to stop services with Make"
        return 1
    fi
}

# Stop with Docker Compose
stop_docker_compose() {
    print_info "🐳 Stopping services with Docker Compose..."

    if docker compose version &> /dev/null; then
        COMPOSE_CMD=(docker compose)
    else
        COMPOSE_CMD=(docker-compose)
    fi

    if "${COMPOSE_CMD[@]}" down; then
        print_status "🛑 Application stopped successfully!"
        return 0
    else
        print_error "Failed to stop services with Docker Compose"
        return 1
    fi
}

echo "🛑 Open WebUI Chat Analyzer Shutdown"
echo "===================================="

DOCKER_STATUS=$(check_docker_setup; echo $?)

case $DOCKER_STATUS in
    0)
        stop_docker_make
        ;;
    1)
        stop_docker_compose
        ;;
    2)
        print_error "No Docker setup detected!"
        echo ""
        echo "If you are running via Python virtual environment:"
        echo "  1. Use Ctrl+C in the terminal where the app is running"
        echo "  2. Or kill the Streamlit process manually"
        exit 1
        ;;
esac
