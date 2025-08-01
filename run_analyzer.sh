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
