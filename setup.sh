#!/bin/bash

# Open WebUI Chat Analyzer - Complete Setup Script
# Works on macOS, Linux, and Windows (via Git Bash/WSL)

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
cat > run_analyzer.sh << 'EOF'
#!/bin/bash

# Open WebUI Chat Analyzer Launcher
# This script activates the virtual environment and starts the Streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Check if the main Python file exists
if [ ! -f "openwebui_analyzer.py" ]; then
    echo "❌ openwebui_analyzer.py not found!"
    echo "Please ensure the analyzer file is in the same directory as this script."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, pandas, plotly" 2>/dev/null || {
    echo "❌ Required packages not installed!"
    echo "Installing requirements..."
    pip install -r requirements.txt
}

# Start the Streamlit app
echo "🚀 Starting Open WebUI Chat Analyzer..."
echo "📊 Dashboard will open at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run openwebui_analyzer.py \
    --server.address=localhost \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
EOF

# Make launcher executable
chmod +x run_analyzer.sh
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

# Summary
echo ""
echo "🎉 Setup completed successfully!"
echo "================================="
echo ""
print_info "Project structure:"
echo "  📁 $(pwd)"
echo "  ├── 🐍 venv/              (Python virtual environment)"
echo "  ├── 📊 openwebui_analyzer.py (Main application)"
echo "  ├── 🚀 run_analyzer.sh    (Launcher script)"
echo "  ├── 📋 requirements.txt   (Dependencies)"
echo "  ├── ⚙️  config.yaml       (Configuration)"
echo "  └── 📂 data/              (Upload directory)"
echo ""
print_status "Next steps:"
echo "  1️⃣  Run the analyzer: ./run_analyzer.sh"
echo "  2️⃣  Open browser to: http://localhost:8501"
echo "  3️⃣  Upload your Open WebUI JSON export"
echo ""
print_info "To get your chat data from Open WebUI:"
echo "  • Go to Settings → Data & Privacy"
echo "  • Click 'Export All Chats'"
echo "  • Upload the downloaded JSON file"
echo ""
print_info "Troubleshooting:"
echo "  • If port 8501 is busy, the app will find another port"
echo "  • Check terminal output for the actual URL"
echo "  • Ensure JSON file is valid Open WebUI export"
echo ""
echo "Happy analyzing! 🚀📈"