# Open WebUI Chat Analyzer – Docker helpers

COMPOSE ?= docker compose
BACKEND ?= backend
PROXY ?= nginx
FRONTEND_NEXT ?= frontend-next
BACKEND_PORT ?= 8502
SERVICES := $(BACKEND) $(FRONTEND_NEXT)
DEV_COMPOSE_FILES ?= -f docker-compose.yml -f docker-compose.dev.yml

.PHONY: help up up-frontend-next up-backend down down-frontend-next \
	down-backend build build-frontend-next build-backend rebuild \
	restart restart-frontend-next restart-backend destroy logs \
	logs-frontend-next logs-backend ps status dev dev-down dev-restart \
	shell-frontend-next shell-frontend-next-dev shell-backend \
	health-backend

help: ## Show this help message
	@echo "Open WebUI Chat Analyzer – Docker Compose Commands"
	@echo "================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -------------------------------------------------------------------
# Core lifecycle management
# -------------------------------------------------------------------
up: ## Start frontend and backend services
	$(COMPOSE) up -d $(SERVICES)

up-frontend-next: ## Start only the Next.js frontend service
	$(COMPOSE) up -d $(FRONTEND_NEXT)

up-backend: ## Start only the backend service
	$(COMPOSE) up -d $(BACKEND)

down: ## Stop all services
	$(COMPOSE) down

down-frontend-next: ## Stop and remove the Next.js frontend service
	$(COMPOSE) stop $(FRONTEND_NEXT)
	$(COMPOSE) rm -f $(FRONTEND_NEXT)

down-backend: ## Stop and remove the backend service
	$(COMPOSE) stop $(BACKEND)
	$(COMPOSE) rm -f $(BACKEND)

destroy: ## Stop and remove all services, volumes, and orphans
	$(COMPOSE) down --volumes --remove-orphans

# -------------------------------------------------------------------
# Build and rebuild images
# -------------------------------------------------------------------
build: ## Build images for all services
	$(COMPOSE) build $(SERVICES)

build-frontend-next: ## Build only the Next.js frontend image
	$(COMPOSE) build $(FRONTEND_NEXT)

build-backend: ## Build only the backend image
	$(COMPOSE) build $(BACKEND)

rebuild: ## Rebuild images and restart all services
	$(COMPOSE) up -d --build $(SERVICES)

# -------------------------------------------------------------------
# Restart and status helpers
# -------------------------------------------------------------------
restart: ## Restart all services
	$(COMPOSE) restart $(SERVICES)

restart-frontend-next: ## Restart only the Next.js frontend service
	$(COMPOSE) restart $(FRONTEND_NEXT)

restart-backend: ## Restart only the backend service
	$(COMPOSE) restart $(BACKEND)

ps status: ## Show container status
	$(COMPOSE) ps

logs: ## Tail logs for all services
	$(COMPOSE) logs -f $(SERVICES)

logs-frontend-next: ## Tail logs for the Next.js frontend
	$(COMPOSE) logs -f $(FRONTEND_NEXT)

logs-backend: ## Tail logs for the backend
	$(COMPOSE) logs -f $(BACKEND)

# -------------------------------------------------------------------
# Development conveniences
# -------------------------------------------------------------------
dev: ## Start backend and frontend with hot-reload enabled
	$(COMPOSE) $(DEV_COMPOSE_FILES) up -d $(SERVICES)

dev-down: ## Stop development profile services
	$(COMPOSE) $(DEV_COMPOSE_FILES) down

dev-restart: ## Restart development profile services
	$(COMPOSE) $(DEV_COMPOSE_FILES) restart $(SERVICES)

shell: ## Open a shell in the Next.js frontend container
	$(COMPOSE) exec $(FRONTEND_NEXT) /bin/sh

shell-frontend-next: ## Open a shell in the Next.js frontend container
	$(COMPOSE) exec $(FRONTEND_NEXT) /bin/sh

shell-frontend-next-dev: ## Open a shell in the Next.js frontend dev container
	$(COMPOSE) $(DEV_COMPOSE_FILES) exec $(FRONTEND_NEXT) /bin/sh

shell-backend: ## Open a shell in the backend container
	$(COMPOSE) exec $(BACKEND) /bin/bash

health-backend: ## Query the backend health endpoint
	@curl --fail --silent --show-error http://localhost:$(BACKEND_PORT)/health && echo
