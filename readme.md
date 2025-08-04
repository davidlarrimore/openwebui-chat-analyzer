# 💬 Open WebUI Chat Analyzer

A simple dashboard for analyzing Open WebUI chat log data. Transform your conversation data into actionable insights with beautiful visualizations and comprehensive analytics.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## ✨ Features

### 📊 **Comprehensive Analytics**

- **Overview Metrics**: Total chats, messages, unique users, and engagement statistics
- **Time Analysis**: Daily activity trends, peak usage hours, and conversation patterns
- **Model Performance**: AI model usage distribution and performance insights
- **Content Analysis**: Word clouds, message length analysis, and topic identification
- **Sentiment Analysis**: User satisfaction tracking with positive/negative/neutral ratios
- **Search & Filter**: Full-text search across all messages with advanced filtering

### 🚀 **Easy to Use**

- **Zero Configuration**: Just upload your JSON export and start analyzing
- **Interactive Dashboards**: Real-time charts and visualizations
- **Export Ready**: Download processed data as CSV for further analysis

## 🎯 Quick Start

### Prerequisites

**Choose your preferred method:**
- **🐳 Docker** (Recommended): Docker Desktop installed
- **🐍 Python**: Python 3.8+ (macOS, Linux, or Windows)

### Option A: Docker (Recommended - Zero Configuration)

```bash
# 1. Clone the repository
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer

# 2. Start with Docker (one command!)
make up
# OR: docker-compose up -d

# 3. Access the application
open http://localhost:8501
```

**That's it!** Skip to step 4 below.

### Option B: Python Setup

```bash
# 1. Clone & Setup
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the analyzer
streamlit run openwebui_chat_analyzer.py
```

### Get Your Data (Both Methods)

1. Open your Open WebUI instance
2. Navigate to **Settings** → **Data & Privacy**
3. Click **"Export All Chats"**
4. Save the downloaded JSON file

### Analyze Your Data

1. Upload your JSON export file to the web interface
2. Explore the interactive dashboards
3. Export insights as needed

## 📸 Screenshots

### Overview Dashboard

Get instant insights into your chat activity with key metrics and trends.

### Time Analysis

Discover your usage patterns with activity heatmaps and trend analysis.

### Content Analysis

Explore conversation topics with word clouds and message analysis.

### Sentiment Tracking

Monitor user satisfaction and sentiment trends over time.

## 🛠️ Installation Options

### Option 1: Docker (Recommended)

**Why Docker?** Zero configuration, consistent environment, easy updates, and production-ready.

```bash
# Quick start
git clone https://github.com/yourusername/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
make up

# Alternative Docker commands
docker-compose up -d
# OR
docker build -t openwebui-chat-analyzer . && docker run -d -p 8501:8501 openwebui-chat-analyzer
```

**Docker Features:**
- 🚀 One-command deployment
- 🔒 Security hardened (non-root user, read-only filesystem)
- 📊 Resource management and health checks
- 🔄 Easy updates and backups
- 🌐 Production-ready with nginx proxy option

**Docker Management:**
```bash
make help        # Show all available commands
make logs        # View application logs
make shell       # Open container shell
make backup      # Backup data
make clean       # Clean up containers
```

### Option 2: Automated Setup (Python)

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/openwebui-chat-analyzer/main/setup.sh | bash
```

### Option 3: Manual Installation (Python)

```bash
# Clone repository
git clone https://github.com/yourusername/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install streamlit pandas plotly wordcloud textblob networkx

# Download language data
python -c "import nltk; nltk.download('punkt')"

# Run the application
streamlit run openwebui_chat_analyzer.py
```

## 📊 Analytics Features

### **Overview Metrics**

- Total conversations and messages
- Unique user count
- Average messages per conversation
- User vs AI message ratios

### **Time-Based Analysis**

- Daily, weekly, and monthly activity trends
- Peak usage hours and days
- Conversation duration patterns
- Activity heatmaps

### **Model Performance**

- AI model usage distribution
- Response time analysis (if available)
- Model preference trends
- Performance comparisons

### **Content Intelligence**

- Word frequency analysis
- Topic modeling and clustering
- Message length distributions
- Content category identification

### **Sentiment Analysis**

- Positive, negative, neutral message classification
- Sentiment trends over time
- User satisfaction indicators
- Emotional journey mapping

### **Search & Discovery**

- Full-text search across all messages
- Advanced filtering by date, user, model
- Conversation thread exploration
- Export search results

## 🔧 Configuration

### Environment Variables

**Docker:**
```bash
# Customize Docker deployment
docker run -d \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e TZ=America/New_York \
  -p 8501:8501 \
  openwebui-chat-analyzer
```

**Python:**

```bash
# Optional: Customize port and host
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost

# Optional: Enable debug mode
export STREAMLIT_LOGGER_LEVEL=debug
```

### Config File

Create `config.yaml` for advanced settings:

```yaml
analytics:
  sentiment_threshold: 0.1
  max_wordcloud_words: 100
  default_timezone: "UTC"

visualization:
  theme: "plotly_white"
  color_palette: "viridis"
  chart_height: 400
```

## 📦 Dependencies

### Core Requirements

```text
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
wordcloud>=1.9.2
textblob>=0.17.1
networkx>=3.1
numpy>=1.24.0
```

### Optional Enhancements

```text
jupyter>=1.0.0          # For notebook analysis
scikit-learn>=1.3.0     # Advanced ML features
spacy>=3.6.0           # Enhanced NLP
```

## 🚀 Usage Examples

### Basic Analysis

```python
# Load your data
python openwebui_chat_analyzer.py

# Upload JSON file through the web interface
# Explore the generated dashboards
```

### Advanced Search

```python
# Search for specific topics
# Filter by date ranges
# Export filtered results
```

### Custom Analysis

```python
# Extend the analyzer with custom metrics
# Add new visualization types
# Integrate with external tools
```

## 📁 Project Structure

```text
openwebui-chat-analyzer/
├── 🐳 Docker Files
│   ├── Dockerfile              # Multi-stage production build
│   ├── docker-compose.yml      # Orchestration configuration
│   ├── .dockerignore           # Docker ignore rules
│   └── Makefile               # Docker management commands
├── 📊 Application
│   ├── openwebui_chat_analyzer.py   # Main Streamlit application
│   ├── requirements.txt        # Python dependencies
│   └── config.yaml            # Configuration settings
├── 🚀 Setup Scripts
│   ├── setup.sh               # Automated Python setup
│   └── run.sh        # Python launcher script
├── 📁 Data Directories
│   ├── data/                  # Upload directory (gitignored)
│   └── sample_data/           # Example data for testing
├── 📖 Documentation
│   ├── README.md              # This file
│   └── docs/                  # Additional documentation
└── 🧪 Testing
    └── tests/                 # Test files
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork the repository
git clone https://github.com/yourusername/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions
- Write tests for new features

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

#### Docker Issues

**Docker not running:**

```bash
# Start Docker Desktop (Mac/Windows) or Docker service (Linux)
# Mac: Open Docker Desktop app
# Linux: sudo systemctl start docker
```

**Port already in use:**

```bash
# Docker will automatically find an available port
# Check logs: make logs or docker-compose logs
```

**Permission errors with volumes:**

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data
# Or use Docker: make backup && make restore BACKUP=filename
```

#### Python Issues

##### Port Already in Use

```bash
# Streamlit will automatically find an available port
# Check terminal output for the actual URL
```

##### Memory Issues with Large Files

```bash
# For very large exports (>100MB), consider:
# 1. Use Docker with increased memory: docker run --memory="4g"
# 2. Split the JSON file into smaller chunks
# 3. Use the streaming processor mode
```

##### Missing Dependencies

```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Update pip
pip install --upgrade pip
```

##### TextBlob Data Missing

```bash
# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

### Performance Tips

- Use smaller date ranges for faster processing
- Close unused browser tabs to free memory
- Consider using the batch processing mode for large datasets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing app framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [Open WebUI](https://openwebui.com/) for the inspiration and data format
- The open-source community for invaluable tools and libraries

## 📞 Support

### Getting Help

- 📖 Check the [Documentation](docs/)
- 🐛 Report issues on [GitHub Issues](https://github.com/yourusername/openwebui-chat-analyzer/issues)
- 💬 Join our [Discord Community](https://discord.gg/your-server)
- 📧 Email: dlarrimore@amivero.com

### FAQ

**Q: Which installation method should I choose?**
A: Docker is recommended for most users - it's faster to set up and more reliable. Use Python setup if you need to modify the code or don't have Docker.

**Q: Is my chat data secure?**
A: Yes! All processing happens locally on your machine. Docker adds an extra security layer with isolated containers and non-root users.

**Q: Can I run this on a server?**
A: Yes! Use `make deploy-prod` for production deployment with nginx proxy and SSL support.

**Q: How do I update to the latest version?**
A: Docker: `make update` | Python: `git pull && pip install -r requirements.txt`

**Q: What if I have a very large JSON file?**
A: Docker handles large files better due to optimized memory management. Use `docker run --memory="4g"` for files >500MB.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the Open WebUI community

[🚀 Get Started](#-quick-start) • [📊 Features](#-features) • [🤝 Contribute](#-contributing) • [📞 Support](#-support)

</div>