#!/bin/bash

# Open WebUI Chat Analyzer - Quick Setup for macOS
# This script will set up everything you need in a few minutes

set -e

echo "🚀 Setting up Open WebUI Chat Analyzer on macOS..."
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create project directory
PROJECT_DIR="$HOME/openwebui-analyzer"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "📁 Created project directory: $PROJECT_DIR"

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing required packages..."

# Create requirements.txt
cat > requirements.txt << EOF
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
wordcloud>=1.9.2
textblob>=0.17.1
networkx>=3.1
numpy>=1.24.0
Pillow>=10.0.0
EOF

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data for TextBlob
echo "📚 Downloading language data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('brown')" 2>/dev/null || true

# Create the main analyzer file (this will be created separately)
echo "📄 Creating analyzer application..."

# Create a simple launcher script
cat > run_analyzer.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run openwebui_analyzer.py --server.address=localhost --server.port=8501
EOF

chmod +x run_analyzer.sh

# Create README
cat > README.md << 'EOF'
# Open WebUI Chat Analyzer

A powerful dashboard for analyzing your Open WebUI chat exports.

## Quick Start

1. Run the analyzer:
   ```bash
   ./run_analyzer.sh
   ```

2. Open your browser to: http://localhost:8501

3. Upload your Open WebUI JSON export file

## Features

- 📊 Overview metrics and statistics
- 📈 Time-based analysis and trends
- 🤖 Model usage distribution
- 💭 Content analysis with word clouds
- 😊 Sentiment analysis
- 🔍 Full-text search across messages
- 📤 Export processed data

## Getting Your Data

1. In Open WebUI, go to Settings → Data & Privacy
2. Click "Export All Chats"
3. Upload the downloaded JSON file to this analyzer

## Troubleshooting

- If port 8501 is busy, the app will automatically try the next available port
- Check the terminal output for the actual URL if different
- Ensure your JSON file is a valid Open WebUI export

Enjoy analyzing your conversations! 🚀
EOF

echo ""
echo "🎉 Setup complete!"
echo "=================================================="
echo ""
echo "📁 Project location: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "1. Save the analyzer code to: $PROJECT_DIR/openwebui_analyzer.py"
echo "2. Run: cd $PROJECT_DIR && ./run_analyzer.sh"
echo "3. Open your browser to: http://localhost:8501"
echo "4. Upload your Open WebUI JSON export file"
echo ""
echo "💡 To get your data from Open WebUI:"
echo "   Settings → Data & Privacy → Export All Chats"
echo ""
echo "Happy analyzing! 🚀"