#!/bin/sh
set -eu

default_url="${BACKEND_URL:-http://localhost:8502}"
printf "Backend base URL [%s]: " "${default_url}"
IFS= read -r base_url
base_url="${base_url:-$default_url}"

printf "Email/Username: "
IFS= read -r login_email

printf "Password: "
if [ -t 0 ]; then
    stty -echo
fi
IFS= read -r login_password
if [ -t 0 ]; then
    stty echo
fi
printf "\n"

echo "Select summarizer mode:"
echo "  1) Incremental (only missing summaries)"
echo "  2) Full (overwrite all summaries)"
printf "Mode [1/2]: "
IFS= read -r mode_choice

case "${mode_choice}" in
    1) mode="incremental" ;;
    2) mode="full" ;;
    *) echo "Invalid mode selection." ; exit 1 ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="summarizer_run_${timestamp}.log"
fifo="$(mktemp -u)"
mkfifo "${fifo}"
tee -a "${log_file}" < "${fifo}" &
tee_pid=$!
exec > "${fifo}" 2>&1

cookie_file="$(mktemp)"
cleanup() {
    rm -f "${fifo}" "${cookie_file}"
    kill "${tee_pid}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Log file: ${log_file}"
echo "Backend URL: ${base_url}"
echo "Mode: ${mode}"
echo "Starting login..."

py_cmd=""
if command -v python3 >/dev/null 2>&1; then
    py_cmd="python3"
elif command -v python >/dev/null 2>&1; then
    py_cmd="python"
else
    echo "Python not found. Activate your venv or install python3."
    exit 1
fi

export LOGIN_EMAIL="${login_email}"
export LOGIN_PASSWORD="${login_password}"
login_payload="$(${py_cmd} - <<'PY'
import json
import os

print(json.dumps({"email": os.environ["LOGIN_EMAIL"], "password": os.environ["LOGIN_PASSWORD"]}))
PY
)"
unset LOGIN_EMAIL LOGIN_PASSWORD

login_response="$(curl -sS -w "\n%{http_code}" \
    -X POST "${base_url}/api/backend/auth/login" \
    -H "Content-Type: application/json" \
    -d "${login_payload}" \
    -c "${cookie_file}" \
    -b "${cookie_file}")"

login_body="$(printf '%s' "${login_response}" | sed '$d')"
login_code="$(printf '%s' "${login_response}" | tail -n 1)"

echo "Login response code: ${login_code}"
echo "Login response body: ${login_body}"

if [ "${login_code}" != "200" ]; then
    echo "Login failed; aborting."
    exit 1
fi

api_prefix=""
for candidate in "/api/v1" "/api/backend/api/v1"; do
    run_response="$(curl -sS -w "\n%{http_code}" \
        -X POST "${base_url}${candidate}/summaries/run?mode=${mode}" \
        -b "${cookie_file}")"
    run_code="$(printf '%s' "${run_response}" | tail -n 1)"
    if [ "${run_code}" != "404" ]; then
        api_prefix="${candidate}"
        break
    fi
done

if [ -z "${api_prefix}" ]; then
    echo "Summarizer run endpoint not found on /api/v1 or /api/backend/api/v1."
    echo "Restart the backend to load the new route, then retry."
    exit 1
fi

echo "Using API prefix: ${api_prefix}"
echo "Triggering summarizer run..."
run_body="$(printf '%s' "${run_response}" | sed '$d')"
run_code="$(printf '%s' "${run_response}" | tail -n 1)"

echo "Run response code: ${run_code}"
echo "Run response body: ${run_body}"

if [ "${run_code}" != "200" ]; then
    echo "Summarizer run request failed; aborting."
    exit 1
fi

last_event_id=""
while true; do
    status_json="$(curl -sS "${base_url}${api_prefix}/summaries/status" -b "${cookie_file}")"
    echo "STATUS $(date +%H:%M:%S): ${status_json}"

    after_query=""
    if [ -n "${last_event_id}" ]; then
        after_query="&after=${last_event_id}"
    fi
    events_json="$(curl -sS "${base_url}${api_prefix}/summaries/events?limit=200${after_query}" -b "${cookie_file}")"
    echo "EVENTS $(date +%H:%M:%S): ${events_json}"

    last_event_id="$(printf '%s' "${events_json}" | ${py_cmd} - <<'PY'
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print("")
    raise SystemExit(0)
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)
events = data.get("events") or []
if events:
    print(events[-1].get("event_id", ""))
PY
)"

    state="$(printf '%s' "${status_json}" | ${py_cmd} - <<'PY'
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print("")
    raise SystemExit(0)
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)
print(data.get("state", ""))
PY
)"

    if [ "${state}" = "done" ] || [ "${state}" = "error" ] || [ "${state}" = "cancelled" ]; then
        echo "Summarizer finished with state: ${state}"
        break
    fi

    sleep 2
done
