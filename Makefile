# Open WebUI Chat Analyzer – Docker helpers

COMPOSE ?= docker compose
FRONTEND ?= frontend
BACKEND ?= backend
DEV_FRONTEND ?= frontend-dev
DEV_FRONTEND_NEXT ?= frontend-next-dev
PROXY ?= nginx
FRONTEND_NEXT ?= frontend-next
DB ?= postgres
SERVICES := $(FRONTEND) $(FRONTEND_NEXT) $(BACKEND) $(DB)

.PHONY: help up up-frontend up-frontend-next up-backend down down-frontend \
	down-frontend-next down-backend build build-frontend build-frontend-next \
	build-backend rebuild restart restart-frontend restart-frontend-next \
	restart-backend restart-db destroy logs logs-frontend logs-frontend-next \
	logs-backend logs-db ps status dev dev-down dev-restart shell shell-frontend \
	shell-frontend-next shell-frontend-next-dev shell-backend shell-dev shell-db \
	up-db down-db

help: ## Show this help message
	@echo "Open WebUI Chat Analyzer – Docker Compose Commands"
	@echo "================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -------------------------------------------------------------------
# Core lifecycle management
# -------------------------------------------------------------------
up: ## Start both frontends and backend services
	$(COMPOSE) up -d $(SERVICES)

up-frontend: ## Start only the frontend service
	$(COMPOSE) up -d $(FRONTEND)

up-frontend-next: ## Start only the Next.js frontend service
	$(COMPOSE) up -d $(FRONTEND_NEXT)

up-backend: ## Start only the backend service
	$(COMPOSE) up -d $(BACKEND)

up-db: ## Start only the database service
	$(COMPOSE) up -d $(DB)

down: ## Stop all services
	$(COMPOSE) down

down-frontend: ## Stop and remove the frontend service
	$(COMPOSE) stop $(FRONTEND)
	$(COMPOSE) rm -f $(FRONTEND)

down-frontend-next: ## Stop and remove the Next.js frontend service
	$(COMPOSE) stop $(FRONTEND_NEXT)
	$(COMPOSE) rm -f $(FRONTEND_NEXT)

down-backend: ## Stop and remove the backend service
	$(COMPOSE) stop $(BACKEND)
	$(COMPOSE) rm -f $(BACKEND)

down-db: ## Stop and remove the database service
	$(COMPOSE) stop $(DB)
	$(COMPOSE) rm -f $(DB)

destroy: ## Stop and remove all services, volumes, and orphans
	$(COMPOSE) down --volumes --remove-orphans

# -------------------------------------------------------------------
# Build and rebuild images
# -------------------------------------------------------------------
build: ## Build images for all services
	$(COMPOSE) build $(SERVICES)

build-frontend: ## Build only the frontend image
	$(COMPOSE) build $(FRONTEND)

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

restart-frontend: ## Restart only the frontend service
	$(COMPOSE) restart $(FRONTEND)

restart-frontend-next: ## Restart only the Next.js frontend service
	$(COMPOSE) restart $(FRONTEND_NEXT)

restart-backend: ## Restart only the backend service
	$(COMPOSE) restart $(BACKEND)

restart-db: ## Restart only the database service
	$(COMPOSE) restart $(DB)

ps status: ## Show container status
	$(COMPOSE) ps

logs: ## Tail logs for all services
	$(COMPOSE) logs -f $(SERVICES)

logs-frontend: ## Tail logs for the frontend
	$(COMPOSE) logs -f $(FRONTEND)

logs-frontend-next: ## Tail logs for the Next.js frontend
	$(COMPOSE) logs -f $(FRONTEND_NEXT)

logs-backend: ## Tail logs for the backend
	$(COMPOSE) logs -f $(BACKEND)

logs-db: ## Tail logs for the database
	$(COMPOSE) logs -f $(DB)

# -------------------------------------------------------------------
# Development conveniences
# -------------------------------------------------------------------
dev: ## Start backend plus hot-reload frontend profiles
	$(COMPOSE) stop $(FRONTEND) $(FRONTEND_NEXT)
	$(COMPOSE) up -d $(DB) $(BACKEND) $(DEV_FRONTEND) $(DEV_FRONTEND_NEXT)

dev-down: ## Stop development profile services
	$(COMPOSE) stop $(DEV_FRONTEND) $(DEV_FRONTEND_NEXT)
	$(COMPOSE) rm -f $(DEV_FRONTEND) $(DEV_FRONTEND_NEXT)

dev-restart: ## Restart development profile services
	$(COMPOSE) restart $(DEV_FRONTEND) $(DEV_FRONTEND_NEXT)

shell: ## Open a shell in the long-running frontend container
	$(COMPOSE) exec $(FRONTEND) /bin/bash

shell-frontend: ## Open a shell in the frontend container
	$(COMPOSE) exec $(FRONTEND) /bin/bash

shell-frontend-next: ## Open a shell in the Next.js frontend container
	$(COMPOSE) exec $(FRONTEND_NEXT) /bin/sh

shell-frontend-next-dev: ## Open a shell in the Next.js frontend dev container
	$(COMPOSE) exec $(DEV_FRONTEND_NEXT) /bin/sh

shell-backend: ## Open a shell in the backend container
	$(COMPOSE) exec $(BACKEND) /bin/bash

shell-db: ## Open a shell in the database container
	$(COMPOSE) exec $(DB) /bin/sh

shell-dev: ## Open a shell in the hot-reload frontend container
	$(COMPOSE) exec $(DEV_FRONTEND) /bin/bash
