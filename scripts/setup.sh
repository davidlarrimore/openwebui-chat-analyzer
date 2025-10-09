#!/bin/bash

# Open WebUI Chat Analyzer - Complete Setup Script
# Works on macOS, Linux, and Windows (via Git Bash/WSL)
# Supports both Docker and Python installation methods

set -e

echo "🚀 Setting up Open WebUI Chat Analyzer..."
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check for Docker
check_docker() {
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_status "Docker found: $DOCKER_VERSION"
        return 0
    else
        return 1
    fi
}

# Check for Docker Compose
check_docker_compose() {
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "v2+")
        print_status "Docker Compose found: $COMPOSE_VERSION"
        return 0
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        print_status "Docker Compose found: $COMPOSE_VERSION"  
        return 0
    else
        return 1
    fi
}

# Docker setup
setup_docker() {
    print_info "Setting up Docker environment..."
    
    # Create data directory
    mkdir -p data
    touch data/.gitkeep
    print_status "Data directory created"
    
    # Create configuration if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        print_info "Creating default configuration..."
        cat > config.yaml << 'EOF'
# Open WebUI Chat Analyzer Configuration
analytics:
  sentiment_threshold: 0.1
  max_wordcloud_words: 100
  default_timezone: "UTC"

visualization:
  theme: "plotly_white"
  color_palette: "viridis"
  chart_height: 400

performance:
  max_file_size_mb: 500
  chunk_size: 1000
EOF
        print_status "Configuration file created"
    fi
    
    # Build and start with Docker Compose
    print_info "Building and starting Docker containers..."
    if command -v make &> /dev/null; then
        make up
    else
        docker-compose up -d --build
    fi
    
    if [ $? -eq 0 ]; then
        print_status "Docker setup completed successfully!"
        print_info "🌐 Access the application at: http://localhost:8501"
        
        # Wait a moment and test connectivity
        print_info "Testing connectivity..."
        sleep 5
        if curl -s -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            print_status "Application is running and healthy!"
        else
            print_warning "Application is starting up... may take a few more seconds"
        fi
    else
        print_error "Docker setup failed!"
        return 1
    fi
}

# Python setup (original functionality)
setup_python() {
    # Check if Python 3 is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed!"
        echo "Please install Python 3.8+ from:"
        echo "  • macOS: https://python.org or 'brew install python'"
        echo "  • Linux: 'sudo apt install python3 python3-pip python3-venv'"
        echo "  • Windows: https://python.org"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python 3 found: $PYTHON_VERSION"

    # Check Python version (minimum 3.8)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    # Create virtual environment
    print_info "Creating virtual environment..."
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi

    python3 -m venv venv
    print_status "Virtual environment created"

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet

    # Install required packages
    print_info "Installing required packages..."
    pip install -r requirements.txt --quiet

    print_status "Python packages installed"

    # Download NLTK data for TextBlob
    print_info "Downloading language data for sentiment analysis..."
    python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
    print('Language data downloaded successfully')
except Exception as e:
    print(f'Note: Some language data may not be available: {e}')
" 2>/dev/null || print_warning "Some language data downloads failed (not critical)"

    # Create improved launcher script
    print_info "Creating launcher script..."
    cat > run.sh << 'EOF'
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
EOF

    # Make launcher executable
    chmod +x run.sh
    print_status "Launcher script created and made executable"

    # Create data directory with .gitkeep
    mkdir -p data
    touch data/.gitkeep
    print_status "Data directory created"

    # Create sample config if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        cat > config.yaml << 'EOF'
# Open WebUI Chat Analyzer Configuration
analytics:
  sentiment_threshold: 0.1
  max_wordcloud_words: 100
  default_timezone: "UTC"

visualization:
  theme: "plotly_white"
  color_palette: "viridis"
  chart_height: 400

performance:
  max_file_size_mb: 500
  chunk_size: 1000
EOF
        print_status "Configuration file created"
    fi

    print_status "Python setup completed successfully!"
}

# Detect installation preference
print_info "Detecting available installation methods..."

DOCKER_AVAILABLE=false
PYTHON_AVAILABLE=false

if check_docker && check_docker_compose; then
    DOCKER_AVAILABLE=true
fi

if command -v python3 &> /dev/null; then
    PYTHON_AVAILABLE=true
fi

# Installation method selection
if [ "$DOCKER_AVAILABLE" = true ] && [ "$PYTHON_AVAILABLE" = true ]; then
    echo ""
    print_info "Both Docker and Python are available. Choose your preferred method:"
    echo "  1) 🐳 Docker (Recommended - Zero configuration, production-ready)"
    echo "  2) 🐍 Python (Traditional setup, good for development)"
    echo ""
    read -p "Enter your choice (1 or 2): " choice
    
    case $choice in
        1)
            setup_docker
            ;;
        2)
            setup_python
            ;;
        *)
            print_warning "Invalid choice. Defaulting to Docker..."
            setup_docker
            ;;
    esac
elif [ "$DOCKER_AVAILABLE" = true ]; then
    print_info "Docker detected. Using Docker installation..."
    setup_docker
elif [ "$PYTHON_AVAILABLE" = true ]; then
    print_info "Python detected. Using Python installation..."
    setup_python
else
    print_error "Neither Docker nor Python 3 found!"
    echo ""
    echo "Please install one of the following:"
    echo ""
    echo "🐳 Docker (Recommended):"
    echo "  • macOS/Windows: Download Docker Desktop from https://docker.com"
    echo "  • Linux: sudo apt install docker.io docker-compose"
    echo ""
    echo "🐍 Python:"
    echo "  • macOS: brew install python3 or download from https://python.org"
    echo "  • Linux: sudo apt install python3 python3-pip python3-venv"
    echo "  • Windows: Download from https://python.org"
    exit 1
fi

# Summary
echo ""
if [ "$DOCKER_AVAILABLE" = true ] && ([ "$choice" = "1" ] || [ "$PYTHON_AVAILABLE" = false ]); then
    echo "🎉 Docker setup completed successfully!"
    echo "===================================="
    echo ""
    print_info "🐳 Docker containers are running:"
    echo "  📊 Web Interface: http://localhost:8501"
    echo "  🔍 Health Check: http://localhost:8501/_stcore/health"
    echo ""
    print_status "Docker Management Commands:"
    echo "  📋 make help          # Show all available commands"
    echo "  📊 make logs          # View application logs"
    echo "  🐚 make shell         # Open container shell"
    echo "  🛑 make down          # Stop all services"
    echo "  🔄 make restart       # Restart services"
    echo "  💾 make backup        # Backup data"
    echo ""
else
    echo "🎉 Python setup completed successfully!"
    echo "====================================="
    echo ""
    print_info "Project structure:"
    echo "  📁 $(pwd)"
    echo "  ├── 🐍 venv/              (Python virtual environment)"
    echo "  ├── 📊 openwebui_chat_analyzer.py (Main application)"
    echo "  ├── 🚀 run.sh    (Launcher script)"
    echo "  ├── 📋 requirements.txt   (Dependencies)"
    echo "  ├── ⚙️  config.yaml       (Configuration)"
    echo "  └── 📂 data/              (Upload directory)"
    echo ""
    print_status "Next steps:"
    echo "  1️⃣  Run the analyzer: ./run.sh"
    echo "  2️⃣  Open browser to: http://localhost:8501"
    echo "  3️⃣  Upload your Open WebUI JSON export"
fi

echo ""
print_info "To get your chat data from Open WebUI:"
echo "  • Go to Settings → Data & Privacy"
echo "  • Click 'Export All Chats'"
echo "  • Upload the downloaded JSON file to the web interface"
echo ""
print_info "Troubleshooting:"
echo "  • If port 8501 is busy, the app will find another port"
echo "  • Check terminal output for the actual URL"
echo "  • Ensure JSON file is valid Open WebUI export"
echo "  • For Docker issues: check that Docker Desktop is running"
echo ""
print_info "Need help? Check the README.md or create an issue on GitHub"
echo ""
echo "Happy analyzing! 🚀📈"