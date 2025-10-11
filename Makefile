# Open WebUI Chat Analyzer – Docker helpers

COMPOSE ?= docker compose
FRONTEND ?= frontend
BACKEND ?= backend
DEV_FRONTEND ?= frontend-dev
PROXY ?= nginx
SERVICES := $(FRONTEND) $(BACKEND)

.PHONY: help up up-frontend up-backend down down-frontend down-backend \
	build build-frontend build-backend rebuild restart restart-frontend \
	restart-backend destroy logs logs-frontend logs-backend ps status dev \
	dev-down dev-restart shell shell-frontend shell-backend shell-dev

help: ## Show this help message
	@echo "Open WebUI Chat Analyzer – Docker Compose Commands"
	@echo "================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -------------------------------------------------------------------
# Core lifecycle management
# -------------------------------------------------------------------
up: ## Start frontend and backend services
	$(COMPOSE) up -d $(SERVICES)

up-frontend: ## Start only the frontend service
	$(COMPOSE) up -d $(FRONTEND)

up-backend: ## Start only the backend service
	$(COMPOSE) up -d $(BACKEND)

down: ## Stop all services
	$(COMPOSE) down

down-frontend: ## Stop and remove the frontend service
	$(COMPOSE) stop $(FRONTEND)
	$(COMPOSE) rm -f $(FRONTEND)

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

build-frontend: ## Build only the frontend image
	$(COMPOSE) build $(FRONTEND)

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

restart-backend: ## Restart only the backend service
	$(COMPOSE) restart $(BACKEND)

ps status: ## Show container status
	$(COMPOSE) ps

logs: ## Tail logs for all services
	$(COMPOSE) logs -f $(SERVICES)

logs-frontend: ## Tail logs for the frontend
	$(COMPOSE) logs -f $(FRONTEND)

logs-backend: ## Tail logs for the backend
	$(COMPOSE) logs -f $(BACKEND)

# -------------------------------------------------------------------
# Development conveniences
# -------------------------------------------------------------------
dev: ## Start backend plus hot-reload frontend profile
	$(COMPOSE) up -d $(BACKEND) $(DEV_FRONTEND)

dev-down: ## Stop development profile services
	$(COMPOSE) stop $(DEV_FRONTEND)
	$(COMPOSE) rm -f $(DEV_FRONTEND)

dev-restart: ## Restart development profile services
	$(COMPOSE) restart $(DEV_FRONTEND)

shell: ## Open a shell in the long-running frontend container
	$(COMPOSE) exec $(FRONTEND) /bin/bash

shell-frontend: ## Open a shell in the frontend container
	$(COMPOSE) exec $(FRONTEND) /bin/bash

shell-backend: ## Open a shell in the backend container
	$(COMPOSE) exec $(BACKEND) /bin/bash

shell-dev: ## Open a shell in the hot-reload frontend container
	$(COMPOSE) exec $(DEV_FRONTEND) /bin/bash
