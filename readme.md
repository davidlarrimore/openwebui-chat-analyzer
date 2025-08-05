# 💬 Open WebUI Chat Analyzer

**Privacy-first analytics dashboard for Open WebUI chat exports. Transform your conversation data into actionable insights with interactive visualizations.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## ✨ Key Features

### **📊 Comprehensive Analytics**

- **Engagement Metrics**: Total chats, messages, users, token usage, and file uploads
- **Time Analysis**: Daily activity trends, usage heatmaps, peak conversation periods
- **Model Performance**: AI model usage distribution and effectiveness tracking
- **Content Intelligence**: Word clouds, message length patterns, topic identification
- **Sentiment Analysis**: User satisfaction tracking with temporal trends

### **🔍 Advanced Search & Navigation**

- **Thread-based Search**: Full-text search with conversation context and highlighted results
- **Smart Filtering**: Filter by role (user/assistant), date ranges, and message attributes
- **Individual Thread Export**: Download specific conversations as JSON with metadata
- **Browse Interface**: Chronological conversation browsing with file attachment indicators

### **📤 Export Capabilities**

- **CSV Export**: Processed chat and message data for external analysis
- **Thread Export**: Individual conversations with complete metadata
- **Filtered Results**: Export search results and custom datasets

## 🚀 Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
make up
# Access at http://localhost:8501
```

### Python Setup

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run openwebui_chat_analyzer.py
```

### Getting Your Data

1. Open WebUI → Settings → Data & Privacy → Export All Chats
2. Upload the JSON file to the analyzer interface
3. Start exploring your conversation insights

## 📈 Use Cases

### **Personal Productivity Analysis**

- Track your AI usage patterns and peak productivity hours
- Identify which models work best for different types of tasks
- Monitor conversation quality and satisfaction trends
- Analyze file upload patterns and multimodal interaction trends

### **Content Research & Discovery**

- Search across all conversations to find specific information
- Export relevant conversation threads for documentation
- Identify recurring topics and conversation themes
- Track how your questions and interaction style evolve

### **Usage Optimization**

- Understand token usage patterns for cost optimization
- Compare model performance across different conversation types
- Identify underutilized features (file uploads, specific models)
- Optimize conversation strategies based on sentiment analysis

### **Data Export & Integration**

- Export conversation data for external analysis tools
- Create backups of important conversation threads
- Generate reports for team or organizational insights
- Integrate with existing workflow and documentation systems

## 🐳 Docker Management

```bash
make help           # Show all available commands
make up             # Start services
make down           # Stop services
make logs           # View application logs
make dev            # Development mode with live reload
make backup         # Create data backup
make clean          # Clean up containers
```

## 📊 Analytics Overview

The analyzer processes your Open WebUI JSON export to provide:

- **Overview Dashboard**: Key metrics, token usage, and engagement statistics
- **Time Analysis**: Activity trends, usage heatmaps, conversation patterns
- **Model Usage**: Distribution charts and performance comparisons
- **Content Analysis**: Word clouds, message patterns, length distributions
- **Sentiment Tracking**: User satisfaction trends and conversation quality
- **Search Interface**: Full-text search with thread context and export
- **Browse Data**: Chronological conversation exploration

## 🔧 Technical Details

### **Data Processing**

- Handles large JSON files (500MB+) efficiently
- Local processing only - no external API calls
- Pandas-based data transformation with caching
- Support for chat metadata, message content, and file attachments

### **Supported Data Format**

Open WebUI JSON export containing:
```json
[{
  "id": "chat_id", "user_id": "user_id", "title": "Chat Title",
  "chat": {
    "messages": [{
      "role": "user|assistant", "content": "text", 
      "timestamp": epoch, "model": "model_name"
    }]
  }
}]
```

### **Deployment Options**

- **Docker**: Production-ready with health checks and resource management
- **Development Mode**: Live code reloading for customization
- **Python**: Traditional virtual environment setup
- **Production**: Nginx proxy support for server deployment

## 🔒 Privacy & Security

- **Local Processing**: All analysis happens on your machine
- **No External Dependencies**: No API calls or data transmission
- **Data Control**: Full control over data storage and cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with various data sizes and formats
4. Submit a pull request with clear description

## 📄 License

MIT License - see LICENSE file for details.

---

**Made for privacy-conscious Open WebUI users who want to understand their AI interaction patterns.**