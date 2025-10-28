# 💬 Open WebUI Chat Analyzer

> **A local-first analytics platform for exploring your Open WebUI conversations**

Transform your Open WebUI chat history into actionable insights with this comprehensive analytics stack. Featuring a FastAPI backend paired with a modern Next.js dashboard, your conversation data never leaves your environment—making it perfect for privacy-conscious teams and individual power users.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ✨ Key Features

### 🔒 **Privacy-First Architecture**
- **100% Local Processing** – All data stays on your machine
- **No External Services** – Dashboard communicates only with your local backend
- **Self-Hosted by Design** – Complete control over your conversation analytics
- **Adaptive Alias System** – Stable pseudonyms are stored in the database with an optional real-name override

### 📊 **Comprehensive Analytics**
- **📈 Time Analysis** – Daily trends, conversation patterns, hour-by-day heatmaps
- **📝 Content Analysis** – Word clouds, message length distributions, sentiment breakdown
- **💬 Chat Browser** – Full-text search, filters, and detailed conversation views
- **🔍 Advanced Search** – Query across all messages with powerful filtering options

### 🚀 **Intelligent Data Loading**
- **Direct Connect** – Sync live from your Open WebUI instance with one click
- **File Import** – Drop exports into `data/` or upload through the UI
- **Instant Metrics** – Dashboard updates immediately while summaries process in background
- **Incremental Sync** – Smart updates that only fetch new conversations

### 🤖 **AI-Powered Summaries**
- **Local LLM Integration** – Uses Ollama for automatic chat summarization
- **Incremental Persistence** – Summaries saved as each chat completes (no data loss)
- **Smart Context** – Sentence transformers identify salient utterances for better summaries
- **Fallback Support** – Can use Open WebUI completions endpoint if needed

### 🎨 **Modern UI/UX**
- **Next.js 14 App Router** – Fast, responsive single-page application
- **Tailwind + shadcn/ui** – Beautiful, accessible component library
- **Real-Time Updates** – Live processing logs and progress tracking
- **Multi-User Support** – Auth.js with credentials and GitHub OAuth

---

## 🎯 Quick Start

### Option A: Docker (Recommended)

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
cp .env.example .env
make up
```

**Access Points:**
- 🎨 **Dashboard**: http://localhost:8503
- 🔌 **API**: http://localhost:8502
- 📖 **API Docs**: http://localhost:8502/docs

**Useful Commands:**
```bash
make logs    # View combined logs
make down    # Stop all services
make restart # Restart services
make help    # See all available commands
```

### Option B: Local Development

**Backend:**
```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env

# First run only
python -m textblob.download_corpora

# Start backend
uvicorn backend.app:app --reload --port 8502
```

**Frontend** (in a new terminal):
```bash
cd openwebui-chat-analyzer/frontend-next
pnpm install
pnpm dev  # Runs on http://localhost:3000
```

### Option C: Guided Setup

```bash
scripts/setup.sh  # Interactive wizard for Docker or local setup
```

---

## 📖 Dashboard Overview

### 📊 Overview
- Total conversations, messages, and user activity
- Model usage statistics and file upload tracking
- Approximate token volume (derived from character counts)
- User and model breakdowns with visual charts

### 📈 Time Analysis
- **Daily Trends** – Message volume over time
- **Conversation Length** – Distribution of chat durations
- **Heatmaps** – Activity by hour and day of week
- **Filters** – Segment by user and model

### 📝 Content Analysis
- **Word Clouds** – Most frequently used terms
- **Message Length** – Histograms by role and model
- **Sentiment Breakdown** – Positive, neutral, negative classification
- **Per-User Insights** – Individual communication patterns

### 🔍 Search
- **Full-Text Search** – Query across all messages
- **Advanced Filters** – By user, model, date range, sentiment
- **Export Results** – Download filtered data as CSV or JSON

### 💬 Browse Chats
- **Paginated View** – Browse all conversations
- **Rich Metadata** – Timestamps, participants, model info
- **AI Summaries** – One-line headlines for each chat
- **Quick Actions** – Download individual threads as JSON

### ⚙️ Configuration
- **Data Source Management** – Connect to Open WebUI or upload exports
- **Sync Settings** – Configure full vs incremental sync modes
- **Automated Scheduler** – Set up periodic data refreshes
- **Summarizer Settings** – Choose Ollama model for AI summaries
- **Identity Privacy** – Toggle between pseudonyms and real names on user-facing charts
- **Real-Time Logs** – Monitor sync and processing operations
- **System Status** – View connection health and data freshness

---

## 🔧 Configuration

### Environment Setup

Copy `.env.example` to `.env` and configure:

#### Backend Connectivity
```bash
OWUI_API_BASE_URL=http://localhost:8502       # Backend URL for dashboard
OWUI_API_ALLOWED_ORIGINS=http://localhost:3000 # CORS origins
OWUI_DATA_DIR=./data                           # Default export directory
```

#### Direct Connect Defaults
```bash
OWUI_DIRECT_HOST=http://localhost:3000         # Open WebUI base URL
OWUI_DIRECT_API_KEY=                          # Optional prefill API key
OWUI_EXPOSE_REAL_NAMES=false                  # Set true to expose real names by default
```

#### AI & Summarization
```bash
# Sentence Transformers
EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
SALIENT_K=10                                   # Number of salient utterances

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=180
OLLAMA_DEFAULT_MODEL=llama3.2:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

#### Frontend Settings
```bash
FRONTEND_NEXT_PORT=8503                        # Published dashboard port
FRONTEND_NEXT_PUBLIC_URL=http://localhost:8503 # External URL
FRONTEND_NEXT_BACKEND_BASE_URL=http://backend:8502 # Internal backend URL

# Auth.js
NEXTAUTH_SECRET=your-secret-here
NEXTAUTH_URL=http://localhost:8503

# Optional GitHub OAuth
GITHUB_OAUTH_ENABLED=false
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
```

### Identity Privacy

- **Pseudonym Catalog** – `backend/data/pseudonyms.json` contains the canonical alias list used when ingesting users.
- **Stable Assignments** – Pseudonyms are persisted in the database and refreshed automatically on every sync.
- **Configurable Exposure** – Toggle between pseudonyms and real names from the ⚙️ Configuration page or set `OWUI_EXPOSE_REAL_NAMES=true` to default to real names.

---

## 📥 Loading Data

### Method 1: Direct Connect (Recommended)

1. Navigate to **⚙️ Configuration** in the dashboard
2. Click **Edit Credentials** and enter:
   - Your Open WebUI base URL (e.g., `http://localhost:3000`)
   - An API key with read permissions
3. Click **Test Connection** to verify
4. Click **Sync Data Now** to import

**Benefits:**
- ✅ Automatic incremental updates
- ✅ Always in sync with your Open WebUI instance
- ✅ No manual export/import workflow
- ✅ Scheduler support for automated syncs

### Method 2: File Upload

1. Export from Open WebUI:
   - **Settings → Data & Privacy → Export All Chats** (`all-chats-export-*.json`)
   - **Settings → Database → Export Users** (`users.csv`, optional)
   - Capture `/api/v1/models` as `models.json` (optional, for friendly names)

2. Import options:
   - Drop files in the `data/` directory (auto-loaded on startup)
   - Upload through **⚙️ Configuration** page
   - Files stored in `uploads/` directory

---

## 🤖 AI Summaries

### How It Works

The analyzer automatically generates one-line summaries for each conversation:

1. **Salient Extraction** – Uses sentence-transformers to identify key utterances
2. **LLM Summarization** – Feeds context to your configured Ollama model
3. **Incremental Persistence** – Saves each summary immediately (no data loss)
4. **Background Processing** – Metrics update instantly; summaries generate async

### Configuration

Choose your summarization model in **⚙️ Configuration → Summarizer Settings**:
- Select from available Ollama models
- Settings persist to database
- Changes apply to future summarization jobs

### Rebuilding Summaries

Regenerate all summaries anytime:
- Click **⚙️ Configuration → Quick Actions → Rebuild Summaries**
- Or via API: `POST /api/v1/summaries/rebuild`
- Monitor progress in the **Processing Log**

---

## 🔄 Sync Modes

### Full Sync
- **When to Use**: First sync, changing data sources, or recovering from issues
- **What It Does**: Replaces all local data with fresh import from Open WebUI
- **Recommended**: When `has_data: false` or hostname changes

### Incremental Sync
- **When to Use**: Regular updates from the same Open WebUI instance
- **What It Does**: Fetches only new conversations since last sync
- **Recommended**: When `has_data: true` and source matches
- **Benefits**: Faster, preserves local summaries, efficient

The dashboard **automatically recommends** the appropriate mode based on your current dataset state.

---

## 🔌 Backend API

Key endpoints for integration and automation:

### Dataset & Metadata
```bash
GET  /api/v1/datasets/meta          # Current dataset stats
GET  /api/v1/chats                  # Chat metadata
GET  /api/v1/messages               # Message content
GET  /api/v1/users                  # User directory
POST /api/v1/datasets/reset         # Delete all data
```

### Direct Connect
```bash
POST /api/v1/openwebui/sync         # Sync from Open WebUI
POST /api/v1/openwebui/test         # Test connection
GET  /api/v1/sync/status            # Sync status & freshness
```

### File Uploads
```bash
POST /api/v1/uploads/chat-export    # Upload all-chats-export.json
POST /api/v1/uploads/users          # Upload users.csv
POST /api/v1/uploads/models         # Upload models.json
```

### Summaries
```bash
GET  /api/v1/summaries/status       # Current summarizer status
POST /api/v1/summaries/rebuild      # Regenerate all summaries
GET  /api/v1/summaries/events       # Stream summary events
```

### Admin Settings
```bash
GET  /api/v1/admin/settings/direct-connect     # Get Direct Connect settings
PUT  /api/v1/admin/settings/direct-connect     # Update settings
GET  /api/v1/sync/scheduler                    # Get scheduler config
POST /api/v1/sync/scheduler                    # Update scheduler
```

**Interactive API Docs**: Visit http://localhost:8502/docs when backend is running

---

## 📊 Data Export

### CSV Downloads
- **What's Included**: Same columns shown in dashboard tables
- **Use Cases**: Analysis in Excel, pandas, Tableau, Power BI
- **Fields**: Timestamps, participants, sentiment scores, token estimates

### JSON Downloads
- **What's Included**: Complete conversation metadata and messages
- **Format**: ISO timestamps, attachments, role information
- **Use Cases**: Backup, data migration, custom processing

### Notes
- Token estimates are **heuristic** (based on character counts)
- Sentiment scores use **TextBlob** polarity scale (−1 to 1)
- Exports reflect current filter/search state

---

## 🧪 Sample Data

Explore the dashboard instantly with sample data:

```bash
cp sample_data/sample_data_extract.json data/
cp sample_data/sample_users.csv data/
# Restart backend to auto-load, or upload via Configuration page
```

---

## 🏗️ Project Structure

```
openwebui-chat-analyzer/
├── backend/                 # FastAPI application
│   ├── app.py              # Main application entry point
│   ├── routes.py           # API endpoint definitions
│   ├── services.py         # Business logic & data processing
│   ├── db.py               # SQLite database layer
│   ├── models.py           # Pydantic models
│   ├── summarizer.py       # AI summarization pipeline
│   └── tests/              # Backend test suite
│
├── frontend-next/          # Next.js 14 dashboard
│   ├── app/                # App Router pages & layouts
│   ├── components/         # React components
│   ├── lib/                # Utilities, types, API client
│   └── tests/              # Frontend test suite
│
├── data/                   # Default export directory
├── uploads/                # User-uploaded files
├── scripts/                # Setup & utility scripts
├── sample_data/            # Example datasets
├── .env.example            # Environment template
└── docker-compose.yml      # Container orchestration
```

---

## 🔐 Privacy & Security

### Data Handling
- ✅ **100% Local** – All processing happens on your machine
- ✅ **No External Calls** – Dashboard only talks to local backend
- ✅ **No Telemetry** – Zero tracking or analytics collection
- ✅ **File-Based Storage** – SQLite database in your project directory

### Credential Management
- 🔒 API keys stored in database with quote-safe normalization
- 🔒 Password fields in UI (`type="password"`)
- 🔒 Redacted logging (shows `supe...2345` instead of full key)
- 🔒 Keys never appear in processing logs or responses

### Authentication
- Auth.js (NextAuth) with local credentials provider
- Optional GitHub OAuth integration
- Session-based authentication
- Protected API routes

---

## 🧩 Advanced Features

### Automatic Sync Scheduler
- Configure periodic incremental syncs (5 min to 24 hours)
- Enable/disable via **⚙️ Configuration → Scheduler**
- Settings persist across restarts
- Runs in background without blocking dashboard

### Data Freshness Indicators
- **Staleness Threshold**: Configurable via `SYNC_STALENESS_THRESHOLD_HOURS` (default: 6 hours)
- **Visual Pills**: Green "Current" / Amber "Stale" indicators
- **Last Sync Display**: Human-readable timestamps with relative time

### Processing Log Viewer
- **Real-Time Streaming**: Polls `/api/v1/logs` every 2 seconds
- **Auto-Scroll**: Follows new entries (disable by scrolling up)
- **Structured Logs**: Timestamp, level, phase, job ID, message, details
- **Circular Buffer**: Retains last 200 events
- **Color-Coded Levels**: Debug, info, warning, error

### WAL Mode & Performance
- SQLite Write-Ahead Logging for better concurrency
- Foreign key enforcement for data integrity
- Normal synchronous mode for speed with safety
- Prevents long locks during large syncs

---

## 🛠️ Development

### Frontend Development
```bash
cd frontend-next
pnpm dev        # Start dev server (http://localhost:3000)
pnpm build      # Production build
pnpm test       # Run test suite
pnpm lint       # ESLint check
```

### Backend Development
```bash
source venv/bin/activate
uvicorn backend.app:app --reload --port 8502  # Auto-reload on changes
pytest backend/tests/                          # Run tests
```

### Docker Development
```bash
make dev        # Hot-reload for both frontend and backend
make logs       # Tail all container logs
make shell      # Access backend container shell
```

### Testing
```bash
# Backend
pytest backend/tests/ -v

# Frontend
cd frontend-next && pnpm test
```

---

## 🐛 Troubleshooting

### Dashboard Won't Start
- ✅ Verify Node.js 20+ is installed: `node --version`
- ✅ Verify pnpm is installed: `pnpm --version`
- ✅ Clear cache: `pnpm store prune`

### Backend Connection Issues
- ✅ Confirm backend is running: `curl http://localhost:8502/health`
- ✅ Check `OWUI_API_BASE_URL` in `.env`
- ✅ Verify no port conflicts: `lsof -i :8502`

### Summarizer Failing
- ✅ Confirm Ollama is running: `ollama list`
- ✅ Verify model is available: `ollama run llama3.2:latest`
- ✅ Check `OLLAMA_BASE_URL` in `.env`
- ✅ Increase timeout: `OLLAMA_TIMEOUT=300`

### Word Clouds Not Rendering
- ✅ Install system fonts: `sudo apt-get install fonts-dejavu`
- ✅ Restart backend after font installation

### Database Locked Errors
- ✅ Ensure only one backend instance is running
- ✅ Check for stale lock files in database directory
- ✅ Verify WAL mode is enabled (automatic in recent versions)

### Sync Shows "Stale" Data
- ✅ Run manual sync from **⚙️ Configuration**
- ✅ Adjust `SYNC_STALENESS_THRESHOLD_HOURS` if needed
- ✅ Enable automatic scheduler for regular updates

---

## 🤝 Contributing

Contributions are welcome! Please review [AGENTS.md](AGENTS.md) for:
- Coding standards and conventions
- Development workflow guidelines
- Testing requirements
- Release procedures

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS
- [shadcn/ui](https://ui.shadcn.com/) - Component library
- [Auth.js](https://authjs.dev/) - Authentication
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [TextBlob](https://textblob.readthedocs.io/) - Sentiment analysis

---

## 📬 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/davidlarrimore/openwebui-chat-analyzer/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/davidlarrimore/openwebui-chat-analyzer/discussions)
- 📖 **Documentation**: This README and inline code comments

---

<div align="center">

**Made with ❤️ for the Open WebUI community**

[⬆ Back to Top](#-open-webui-chat-analyzer)

</div>
