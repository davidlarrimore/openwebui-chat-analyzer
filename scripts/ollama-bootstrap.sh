#!/bin/sh
set -eu

# Launch the Ollama server in the background so we can optionally preload models.
CLI_HOST=${OLLAMA_CLI_HOST:-http://127.0.0.1:11434}

ollama serve &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID"
  fi
}
trap cleanup INT TERM

# Preload configured models to avoid cold-start latency when the container begins serving.
if [ -n "${OLLAMA_PRELOAD_MODELS:-}" ]; then
  ready=0
  for attempt in $(seq 1 40); do
    if OLLAMA_HOST="$CLI_HOST" ollama list >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done

  if [ "$ready" -ne 1 ]; then
    echo "Warning: Ollama server did not become ready in time; skipping preload." >&2
  else
    sleep 15
    for model in $OLLAMA_PRELOAD_MODELS; do
      echo "Preloading model: $model"
      pulled=0
      for _ in $(seq 1 10); do
        if OLLAMA_HOST="$CLI_HOST" ollama pull "$model"; then
          pulled=1
          break
        fi
        sleep 5
      done
      if [ "$pulled" -ne 1 ]; then
        echo "Warning: failed to pull model $model" >&2
      fi
    done
  fi
fi

wait "$SERVER_PID"
