# 💬 Open WebUI Chat Analyzer

Streamlit dashboard for exploring Open WebUI chat exports locally.

## Highlights

- Works entirely offline; your exports never leave your machine
- Auto-loads the latest `all-chats-export-*.json` from `data/` and supports an optional `users.csv` for friendly names
- Overview metrics for chats, messages, per-role activity, file uploads, and approximate token volume
- Filters every visualization by Open WebUI user and model
- Time analysis (daily trend, conversation length, hour-by-day heatmap) and content analysis (word cloud, message length)
- Sentiment breakdown with TextBlob plus full-text search, paginated browsing, and per-thread JSON downloads
- CSV exports for both chat metadata and individual messages

## Input Data

- `all-chats-export-*.json` from Open WebUI: Admin Panel → **Settings → Data & Privacy → Export All Chats**
- Optional `users.csv` from Admin Panel → **Settings → Database → Export Users** — needs `user_id` and a name column
- Place files in `data/` to auto-load on startup or upload them through the interface; uploads land under `uploads/`

## Quick Start

### Option A – Docker (recommended)

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
make up            # or: docker compose up -d
```

The app listens on `http://localhost:8501`. Use `make down` to stop, `make logs` to tail the container, and `make help` for the complete command catalog.
The compose stack now bind-mounts `openwebui_chat_analyzer.py`, so code changes land immediately without rebuilding the image.

### Handy Make Commands

- `make help` – List every available helper target with a short description.
- `make up` / `make down` – Start or stop the docker compose stack.
- `make logs` / `make logs-dev` – Follow logs for the production or dev containers.
- `make dev` / `make dev-detached` – Launch the live-reload development profile (foreground/background).
- `make restart` / `make quick-fix` – Restart containers or rebuild + restart when things get stuck.
- `make status` / `make ps` – Show container status at a glance.
- `make backup` / `make restore BACKUP=...` – Snapshot and restore the mounted data directory.
- `make clean` / `make clean-all` – Remove containers, with the latter also pruning volumes and images.
- `make update` – Pull latest sources and rebuild the compose stack.
- `make fix-permissions` – Reset `matplotlib` and data directory ownership inside running containers.

Run `make help` for the full list (build, deploy, debug, tooling helpers, etc.).

### Option B – Local Python environment

```bash
git clone https://github.com/davidlarrimore/openwebui-chat-analyzer.git
cd openwebui-chat-analyzer
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m textblob.download_corpora   # first run only
streamlit run openwebui_chat_analyzer.py
```

### Option C – Guided setup scripts

Run `scripts/setup.sh` for an interactive wizard that can prepare either Docker or the virtual environment. Companion scripts (`scripts/up.sh`, `scripts/down.sh`, `scripts/restart.sh`) wrap the common lifecycle commands.

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

- `docker-compose.yml` defines production, development (live reload), and optional Nginx proxy profiles. Use `docker compose --profile development up openwebui-chat-analyzer-dev` or `make dev` for auto-reload.
- The `Makefile` centralizes build, run, backup, and diagnostic commands — start with `make help`.
- Python dependencies live in `requirements.txt`. The Dockerfile pre-installs TextBlob corpora and runs the app as a non-root user.

## Privacy & Storage

The analyzer never makes network calls; everything happens on your machine. Uploaded files stay under the repository (`data/` and `uploads/`) until you remove them.

## Troubleshooting

- If Streamlit crashes during sentiment analysis, install the TextBlob corpora with `python -m textblob.download_corpora`.
- Some environments need a font package for `wordcloud`; installing system fonts (for example `sudo apt-get install fonts-dejavu`) fixes blank visuals.
- Adjust `STREAMLIT_SERVER_PORT` or the Docker port mapping if 8501 is already in use.

## License

MIT — see `LICENSE` for the full text.
