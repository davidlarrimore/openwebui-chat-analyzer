# 💬 Open WebUI Chat Analyzer

Streamlit dashboard for exploring Open WebUI chat exports locally.

## Highlights

- Local FastAPI backend plus Streamlit UI—your exports never leave your machine
- Backend auto-loads the latest `all-chats-export-*.json` from `data/` and supports an optional `users.csv` for friendly names
- Overview metrics for chats, messages, per-role activity, file uploads, and approximate token volume
- Filters every visualization by Open WebUI user and model
- Time analysis (daily trend, conversation length, hour-by-day heatmap) and content analysis (word cloud, message length)
- Sentiment breakdown with TextBlob plus full-text search, paginated browsing, and per-thread JSON downloads
- CSV exports for both chat metadata and individual messages

## Configuration

1. Copy `.env.example` to `.env`.
2. Adjust values as needed:
   - `OWUI_API_BASE_URL` – URL the Streamlit UI should use. Use `http://localhost:8502` for local runs or `http://backend:8502` when using Docker Compose.
   - `OWUI_API_ALLOWED_ORIGINS` – Comma-separated list of origins permitted to call the FastAPI backend.
   - `OWUI_DATA_DIR` – Directory where default exports live (relative to the project root).
3. Restart the backend (and Streamlit) after changing environment variables.

## Input Data

- `all-chats-export-*.json` from Open WebUI: Admin Panel → **Settings → Data & Privacy → Export All Chats**
- Optional `users.csv` from Admin Panel → **Settings → Database → Export Users** — needs `user_id` and a name column
- Place files in `data/` for the backend to auto-load on startup or upload them through the interface; uploads are forwarded to the API and stored under `uploads/`

## Quick Start

### Option A – Docker (recommended)

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
cp .env.example .env          # optional: set OWUI_API_BASE_URL to http://backend:8502 for docker compose
make up            # or: docker compose up -d
```

The Streamlit UI listens on `http://localhost:8501` and the FastAPI backend on `http://localhost:8502`. Use `make down` to stop, `make logs` to tail the container, and `make help` for the complete command catalog.
The compose stack now bind-mounts the `frontend/` directory, so Streamlit changes land immediately without rebuilding.

### Handy Make Commands

- `make help` – List every available helper target with a short description.
- `make up` / `make down` – Start or stop both backend and frontend services.
- `make up-frontend` / `make up-backend` – Launch a single service.
- `make build` / `make rebuild` – Build images (all) or rebuild and restart.
- `make destroy` – Remove all services, volumes, and orphan containers.
- `make logs` / `make logs-frontend` / `make logs-backend` – Tail logs.
- `make restart` / `make restart-frontend` / `make restart-backend` – Restart running services.
- `make dev` – Start the backend plus the hot-reload frontend profile.

Run `make help` for the full list (build, deploy, debug, tooling helpers, etc.).

### Option B – Local Python environment

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m textblob.download_corpora   # first run only
uvicorn backend.app:app --reload --port 8502  # terminal 1
streamlit run frontend/app.py
```

Run the backend and Streamlit UI in separate terminals (or background the FastAPI process) so the dashboard can reach `http://localhost:8502`.

### Option C – Guided setup scripts

Run `scripts/setup.sh` for an interactive wizard that can prepare either Docker or the virtual environment. After setup, use the `make` targets (`make up`, `make down`, `make logs`, etc.) for day-to-day lifecycle commands.

## Backend API

The Streamlit front-end now talks to a FastAPI service that normalizes and serves the chat exports. Key endpoints:

- `GET /api/v1/datasets/meta` – current dataset identifier, row counts, source label, and last updated timestamp.
- `GET /api/v1/chats` / `GET /api/v1/messages` / `GET /api/v1/users` – hydrated chat metadata, messages, and optional user directory.
- `POST /api/v1/uploads/chat-export` – upload a new `all-chats-export*.json`; replaces the in-memory dataset and bumps the dataset id.
- `POST /api/v1/uploads/users` – upload a companion `users.csv` for friendly display names.

Run `uvicorn backend.app:app --reload` during development to keep the API available to the dashboard.

## Dashboard Tour

- **File uploader**: Detects the latest export under `data/` automatically and lets you add a `users.csv` to replace raw IDs with friendly names.
- **Overview metrics**: Totals and averages for chats, messages, per-role counts, file uploads, and approximate input/output token volumes (derived from character length).
- **Model usage**: Horizontal bar chart plus quick stats for each model encountered across the filtered dataset.
- **Filters**: Slice all visuals by Open WebUI user and model; filter changes reset pagination so the browse experience stays predictable.
- **Tabs**:
  - `🧾 Overview`: Metrics recomputed for the active filters.
  - `📈 Time Analysis`: Daily activity timeline, conversation-length histogram, and hour-by-day heatmap.
  - `💭 Content Analysis`: Word cloud for user messages, average message length by role, and length distribution.
  - `😊 Sentiment`: TextBlob polarity grouped into positive/neutral/negative segments with a time series and supporting metrics.
  - `🔍 Search`: Full-text search with role filter, highlighted matches, attachment badges, and per-thread JSON downloads.
  - `🗂 Browse Data`: Paginated conversation browser with expanders, attachment indicators, and download buttons.
- **Export section**: Download enriched chat metadata and message tables as CSV for external analysis.

## Working With the Data

- CSV downloads contain the same columns the dashboard uses, making follow-on analysis in pandas, spreadsheets, or BI tools straightforward.
- Per-thread JSON downloads include metadata, ISO timestamps, attachments, and every message shown in the interface.
- Sentiment scores and token estimates are heuristic: tokens are inferred from character counts, and sentiment uses TextBlob’s polarity scale (−1 to 1).

## Sample Data

`sample_data/sample_data_extract.json` and `sample_data/sample_users.csv` let you explore the dashboard without waiting for a fresh export. Copy them to `data/` or upload them through the UI to see the charts populate immediately.

## Development Notes

- `docker-compose.yml` defines production, development (live reload), and optional Nginx proxy profiles. Use `docker compose --profile development up frontend-dev` or `make dev` for auto-reload.
- The `Makefile` centralizes build and lifecycle commands — start with `make help`.
- Python dependencies are split between `backend/requirements.txt` and `frontend/requirements.txt` (aggregate via root `requirements.txt`). The multi-stage Dockerfile builds dedicated images for each service and downloads the TextBlob corpora for the Streamlit frontend.

## Privacy & Storage

All requests stay on your machine—the Streamlit UI only talks to the bundled FastAPI service. Uploaded files remain under the repository (`data/` and `uploads/`) until you remove them.

## Troubleshooting

- If Streamlit crashes during sentiment analysis, install the TextBlob corpora with `python -m textblob.download_corpora`.
- Some environments need a font package for `wordcloud`; installing system fonts (for example `sudo apt-get install fonts-dejavu`) fixes blank visuals.
- Adjust `STREAMLIT_SERVER_PORT` or the Docker port mapping if 8501 is already in use.
- Seeing “Unable to connect to the backend API”? Make sure `uvicorn backend.app:app --port 8502` (or the Docker `backend` service) is running and reachable.

## License

MIT — see `LICENSE` for the full text.
