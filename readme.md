# 💬 Open WebUI Chat Analyzer

A simple dashboard for analyzing Open WebUI chat log data. Transform your conversation data into actionable insights with beautiful visualizations and comprehensive analytics.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red.svg)

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

- Python 3.8 or higher
- macOS, Linux, or Windows

### 1. Clone & Setup

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Your Data

1. Open your Open WebUI instance
2. Navigate to **Settings** → **Data & Privacy**
3. Click **"Export All Chats"**
4. Save the downloaded JSON file

### 3. Run the Analyzer

```bash
streamlit run openwebui_analyzer.py
```

Your dashboard will open at `http://localhost:8501`

### 4. Analyze Your Data

1. Upload your JSON export file
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

### Option 1: Automated Setup (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/openwebui-chat-analyzer/main/setup.sh | bash
```

### Option 2: Manual Installation

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
streamlit run openwebui_analyzer.py
```

### Option 3: Docker (Coming Soon)

```bash
docker run -p 8501:8501 -v $(pwd)/data:/app/data openwebui-analyzer
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
python openwebui_analyzer.py

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
├── openwebui_analyzer.py      # Main Streamlit application
├── requirements.txt           # Python dependencies
├── setup.sh                  # Automated setup script
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── config.yaml              # Configuration settings
├── data/                    # Data directory (gitignored)
│   └── sample_export.json   # Sample data for testing
├── docs/                    # Documentation
│   ├── user_guide.md
│   └── api_reference.md
└── tests/                   # Test files
    └── test_analyzer.py
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

#### Port Already in Use

```bash
# Streamlit will automatically find an available port
# Check terminal output for the actual URL
```

#### Memory Issues with Large Files

```bash
# For very large exports (>100MB), consider:
# 1. Splitting the JSON file
# 2. Using the streaming processor
# 3. Increasing system memory
```

**Missing Dependencies**

```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Update pip
pip install --upgrade pip
```

**TextBlob Data Missing**
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
- 📧 Email: support@yourproject.com

### FAQ

**Q: Is my chat data secure?**
A: Yes! All processing happens locally on your machine. Your data never leaves your computer.

**Q: What file formats are supported?**
A: Currently supports Open WebUI JSON exports. More formats coming soon.

**Q: Can I analyze data from other chat platforms?**
A: The current version is optimized for Open WebUI. We're working on supporting more platforms.

**Q: How large files can I process?**
A: Tested with files up to 500MB. Performance depends on your system specifications.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the Open WebUI community

[🚀 Get Started](#-quick-start) • [📊 Features](#-features) • [🤝 Contribute](#-contributing) • [📞 Support](#-support)

</div>
