#!/bin/bash

# Open WebUI Chat Analyzer – environment bootstrapper
# Supports Docker-based deployments or a local Python + Next.js setup.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

fail() {
    print_error "$1"
    exit 1
}

# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------
check_docker() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

check_docker_compose() {
    if docker compose version >/dev/null 2>&1; then
        return 0
    elif command -v docker-compose >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

ensure_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        fail "Python 3.10+ is required. Install it and re-run this script."
    fi

    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        PY_VER="$(python3 --version 2>/dev/null | cut -d' ' -f2)"
        fail "Python 3.10+ is required. Detected ${PY_VER:-unknown}."
    fi

    print_status "Python $(python3 --version | cut -d' ' -f2) detected"
}

ensure_node() {
    if ! command -v node >/dev/null 2>&1; then
        fail "Node.js 20+ is required for the Next.js dashboard."
    fi

    NODE_MAJOR="$(node --version | sed 's/v//' | cut -d'.' -f1)"
    if [ "$NODE_MAJOR" -lt 20 ]; then
        fail "Node.js 20+ is required. Detected $(node --version)."
    fi

    print_status "Node $(node --version) detected"
}

ensure_pnpm() {
    if command -v pnpm >/dev/null 2>&1; then
        print_status "pnpm $(pnpm --version) detected"
        return
    fi

    if command -v corepack >/dev/null 2>&1; then
        print_info "Activating pnpm via corepack..."
        corepack prepare pnpm@8.15.4 --activate
        return
    fi

    fail "pnpm not found. Install pnpm 8 (via corepack or https://pnpm.io/) and re-run."
}

# -----------------------------------------------------------------------------
# Installation helpers
# -----------------------------------------------------------------------------
install_backend_dependencies() {
    print_info "Creating/refreshing Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_status "Existing virtual environment detected"
    fi

    # shellcheck disable=SC1091
    source venv/bin/activate

    print_info "Upgrading pip and installing backend requirements..."
    pip install --upgrade pip >/dev/null
    pip install -r requirements.txt >/dev/null
    print_status "Backend dependencies installed"

    print_info "Ensuring TextBlob corpora are available (used for sentiment analysis)..."
    python -m textblob.download_corpora >/dev/null
    print_status "TextBlob corpora present"

    deactivate
}

install_frontend_dependencies() {
    ensure_node
    ensure_pnpm

    print_info "Installing Next.js dependencies with pnpm..."
    (cd frontend-next && pnpm install >/dev/null)
    print_status "Frontend dependencies installed"
}

create_launcher_script() {
    print_info "Writing ./run.sh helper (starts backend + Next.js dev server)..."
    cat > run.sh <<'RUNEOF'
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

if [ ! -d "venv" ]; then
    echo "Backend virtualenv not found. Run scripts/setup.sh first."
    exit 1
fi

if ! command -v pnpm >/dev/null 2>&1; then
    echo "pnpm is required to run the frontend. Install pnpm and try again."
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

trap 'trap - INT TERM EXIT; kill 0' INT TERM EXIT

uvicorn backend.app:app --reload --host 0.0.0.0 --port 8502 &
BACKEND_PID=$!

echo "Backend running with PID ${BACKEND_PID}. Starting Next.js dev server..."

cd frontend-next
pnpm dev
RUNEOF

    chmod +x run.sh
    print_status "Launcher script created at ./run.sh"
}

setup_local() {
    print_info "Setting up local development environment (Python + Next.js)..."
    ensure_python
    install_backend_dependencies
    install_frontend_dependencies
    create_launcher_script

    echo ""
    print_status "Local development environment prepared."
    print_info "To start developing:"
    echo "  1. source venv/bin/activate"
    echo "  2. uvicorn backend.app:app --reload --port 8502"
    echo "  3. In another terminal: cd frontend-next && pnpm dev"
    echo ""
    print_info "Alternatively run ./run.sh to launch both services together."
}

setup_docker() {
    print_info "Setting up Docker environment..."

    if ! check_docker; then
        fail "Docker does not appear to be running. Start Docker Desktop or your daemon and retry."
    fi

    if ! check_docker_compose; then
        fail "Docker Compose v2 is required. Upgrade Docker or install docker-compose."
    fi

    mkdir -p data
    touch data/.gitkeep

    if command -v make >/dev/null 2>&1; then
        print_info "Using Makefile helpers to start the stack..."
        make up
    else
        print_warning "make not found – falling back to docker compose directly"
        docker compose up -d backend frontend-next
    fi

    print_status "Docker services launched"

    FRONTEND_PORT="${FRONTEND_NEXT_PORT:-8503}"
    print_info "Dashboard: http://localhost:${FRONTEND_PORT}"
    print_info "Backend health: http://localhost:8502/health"
    print_info "Use 'make logs' or 'docker compose logs -f' to monitor container startup."
}

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

echo "🚀 Open WebUI Chat Analyzer Setup"
echo "================================"
echo "Project root: ${PROJECT_ROOT}"
echo ""

OPTIONS=()
if check_docker; then
    OPTIONS+=("docker")
fi
OPTIONS+=("local")

echo "Select an installation option:"
if check_docker; then
    echo "  1) Docker containers (recommended for production/testing)"
    echo "  2) Local development (Python backend + Next.js frontend)"
else
    echo "  1) Local development (Python backend + Next.js frontend)"
fi

read -rp "Enter choice: " choice

echo ""

if check_docker; then
    case "$choice" in
        1)
            setup_docker
            ;;
        2)
            setup_local
            ;;
        *)
            print_warning "Unrecognised choice. Defaulting to Docker..."
            setup_docker
            ;;
    esac
else
    case "$choice" in
        1)
            setup_local
            ;;
        *)
            print_warning "Unrecognised choice. Defaulting to local setup..."
            setup_local
            ;;
    esac
fi

echo ""
print_status "Setup complete. Happy analysing!"
