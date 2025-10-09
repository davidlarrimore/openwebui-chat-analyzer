#!/bin/bash

# Open WebUI Chat Analyzer Launcher
# This script detects the installation method and starts the application appropriately

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

# Check for Python setup
check_python_setup() {
    if [ -d "venv" ] && [ -f "openwebui_chat_analyzer.py" ]; then
        return 0
    fi
    return 1
}

# Start with Docker (Make)
start_docker_make() {
    print_info "🐳 Starting with Docker using Make..."
    
    # Check if already running
    if docker ps --filter name=openwebui-chat-analyzer --format "table {{.Names}}" | grep -q openwebui-chat-analyzer; then
        print_warning "Container already running. Checking status..."
        make logs
        return 0
    fi
    
    # Start the application
    make up
    
    if [ $? -eq 0 ]; then
        print_status "🚀 Application started successfully!"
        print_info "📊 Dashboard: http://localhost:8501"
        print_info "🔍 Health check: http://localhost:8501/_stcore/health"
        echo ""
        print_info "Useful commands:"
        echo "  make logs    # View application logs"
        echo "  make shell   # Open container shell"
        echo "  make down    # Stop the application"
        echo "  make help    # Show all commands"
    else
        print_error "Failed to start with Docker"
        return 1
    fi
}

# Start with Docker Compose
start_docker_compose() {
    print_info "🐳 Starting with Docker Compose..."
    
    # Check if already running
    if docker ps --filter name=openwebui-chat-analyzer --format "table {{.Names}}" | grep -q openwebui-chat-analyzer; then
        print_warning "Container already running at http://localhost:8501"
        docker-compose logs --tail=20 openwebui-chat-analyzer
        return 0
    fi
    
    # Start the application
    if docker compose version &> /dev/null; then
        docker compose up -d
    else
        docker-compose up -d
    fi
    
    if [ $? -eq 0 ]; then
        print_status "🚀 Application started successfully!"
        print_info "📊 Dashboard: http://localhost:8501"
        print_info "🔍 Health check: http://localhost:8501/_stcore/health"
        echo ""
        print_info "Useful commands:"
        echo "  docker-compose logs -f     # View logs"
        echo "  docker-compose down        # Stop application"
        echo "  docker-compose restart     # Restart application"
    else
        print_error "Failed to start with Docker Compose"
        return 1
    fi
}

# Start with Python
start_python() {
    print_info "🐍 Starting with Python virtual environment..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found!"
        echo "Please run setup.sh first to create the environment."
        echo "Or use Docker: make up"
        exit 1
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate

    # Check if the main Python file exists
    if [ ! -f "openwebui_chat_analyzer.py" ]; then
        print_error "openwebui_chat_analyzer.py not found!"
        echo "Please ensure the analyzer file is in the same directory as this script."
        exit 1
    fi

    # Check if required packages are installed
    print_info "📦 Checking dependencies..."
    python -c "import streamlit, pandas, plotly" 2>/dev/null || {
        print_warning "Required packages not installed! Installing..."
        pip install -r requirements.txt
    }

    # Check if port is available
    if lsof -Pi :8501 -sTCP:LISTEN -t &> /dev/null; then
        print_warning "Port 8501 is already in use. Streamlit will find an available port."
    fi

    # Start the Streamlit app
    print_status "🚀 Starting Open WebUI Chat Analyzer..."
    print_info "📊 Dashboard will open at: http://localhost:8501"
    print_info "🛑 Press Ctrl+C to stop the server"
    echo ""

    streamlit run openwebui_chat_analyzer.py \
        --server.address=localhost \
        --server.port=8501 \
        --server.headless=true \
        --browser.gatherUsageStats=false
}

# Main execution logic
echo "🚀 Open WebUI Chat Analyzer Launcher"
echo "===================================="

# Detect installation method
DOCKER_STATUS=$(check_docker_setup; echo $?)
PYTHON_STATUS=$(check_python_setup; echo $?)

case $DOCKER_STATUS in
    0)  # Docker with Make available
        start_docker_make
        ;;
    1)  # Docker without Make
        start_docker_compose
        ;;
    2)  # No Docker setup
        if [ $PYTHON_STATUS -eq 0 ]; then
            start_python
        else
            print_error "No valid installation found!"
            echo ""
            echo "Available options:"
            echo "  🐳 Docker: Run 'setup.sh' and choose Docker option"
            echo "  🐍 Python: Run 'setup.sh' and choose Python option"
            echo ""
            echo "Or run setup manually:"
            echo "  Docker: make up"
            echo "  Python: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
            exit 1
        fi
        ;;
esac