# OpenWebUI Chat Analyzer - Docker Makefile
# Provides convenient commands for Docker operations

.PHONY: help build run stop clean logs shell test deploy backup restore dev

# Default target
help: ## Show this help message
	@echo "OpenWebUI Chat Analyzer - Docker Commands"
	@echo "========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development commands
build: ## Build the Docker image
	@echo "🔨 Building OpenWebUI Chat Analyzer image..."
	docker build -t openwebui-chat-analyzer:latest .

build-dev: ## Build development image with debug info
	@echo "🔨 Building development image..."
	docker build --build-arg DEBUG=true -t openwebui-chat-analyzer:dev .

run: ## Run the container (detached)
	@echo "🚀 Starting OpenWebUI Chat Analyzer..."
	docker run -d \
		--name openwebui-chat-analyzer \
		-p 8501:8501 \
		-v $$(pwd)/data:/app/data \
		--restart unless-stopped \
		openwebui-chat-analyzer:latest
	@echo "✅ Container started! Access: http://localhost:8501"

# NEW: Development mode with live reloading
dev: ## Run in development mode with live code reloading
	@echo "🔧 Starting development environment with live reloading..."
	@echo "📝 Changes to openwebui_chat_analyzer.py will be reflected immediately"
	docker-compose --profile development up --build openwebui-chat-analyzer-dev
	@echo "✅ Development container started! Access: http://localhost:8501"

dev-detached: ## Run development mode in background
	@echo "🔧 Starting development environment (detached)..."
	docker-compose --profile development up -d --build openwebui-chat-analyzer-dev
	@echo "✅ Development container started! Access: http://localhost:8501"
	@echo "📋 Use 'make logs-dev' to view logs"

logs-dev: ## Show development container logs
	@echo "📋 Development container logs:"
	docker-compose --profile development logs -f openwebui-chat-analyzer-dev

stop-dev: ## Stop development container
	@echo "🛑 Stopping development container..."
	docker-compose --profile development down
	@echo "✅ Development environment stopped"

restart-dev: ## Restart development container
	@echo "🔄 Restarting development container..."
	docker-compose --profile development restart openwebui-chat-analyzer-dev
	@echo "✅ Development container restarted!"

run-dev: ## Run in development mode with source code mounted (legacy)
	@echo "🔧 Starting in development mode..."
	docker run -d \
		--name openwebui-chat-analyzer-dev \
		-p 8501:8501 \
		-v $$(pwd)/openwebui_chat_analyzer.py:/app/openwebui_chat_analyzer.py \
		-v $$(pwd)/data:/app/data \
		-v $$(pwd)/.streamlit:/app/.streamlit:ro \
		-e STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto \
		-e STREAMLIT_SERVER_RUN_ON_SAVE=true \
		openwebui-chat-analyzer:latest
	@echo "✅ Development container started! Access: http://localhost:8501"

# Docker Compose commands
up: ## Start with docker-compose
	@echo "🚀 Starting with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started! Access: http://localhost:8501"

up-prod: ## Start production environment with proxy
	@echo "🚀 Starting production environment..."
	docker-compose --profile production up -d
	@echo "✅ Production services started!"

down: ## Stop docker-compose services
	@echo "🛑 Stopping services..."
	docker-compose down

status: ## Show container status
	@echo "📊 Container Status:"
	@echo "==================="
	@if docker ps -a --filter name=openwebui --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep openwebui; then \
		echo ""; \
		echo "🔍 Detailed info:"; \
		docker ps -a --filter name=openwebui --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "❌ No OpenWebUI analyzer containers found"; \
		echo ""; \
		echo "All containers:"; \
		docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	fi

stop: ## Stop the container
	@echo "🛑 Stopping OpenWebUI Chat Analyzer..."
	@# Try different possible container names
	@for name in openwebui-chat-analyzer openwebui-chat-analyzer-dev openwebui-chat-analyzer-openwebui-chat-analyzer; do \
		if docker ps --filter name=$$name --format "{{.Names}}" | grep -q $$name; then \
			echo "Stopping $$name..."; \
			docker stop $$name; \
			docker rm $$name; \
		fi; \
	done
	@echo "✅ Cleanup complete"

# Monitoring and debugging
logs: ## Show container logs
	@echo "📋 Container logs:"
	@if docker ps -a --filter name=openwebui-chat-analyzer --format "table {{.Names}}" | grep -q openwebui-chat-analyzer; then \
		docker logs -f openwebui-chat-analyzer; \
	elif docker ps -a --filter name=openwebui-chat-analyzer-dev --format "table {{.Names}}" | grep -q openwebui-chat-analyzer-dev; then \
		docker logs -f openwebui-chat-analyzer-dev; \
	else \
		echo "❌ No OpenWebUI analyzer container found"; \
		echo "Available containers:"; \
		docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
		echo ""; \
		echo "Try: make dev"; \
	fi

logs-compose: ## Show docker-compose logs
	@echo "📋 Docker Compose logs:"
	@if docker-compose ps | grep -q openwebui; then \
		docker-compose logs -f; \
	else \
		echo "❌ No docker-compose services running"; \
		echo "Try: make up or make dev"; \
	fi

shell: ## Open shell in running container
	@echo "🐚 Opening shell in container..."
	@if docker ps --filter name=openwebui-chat-analyzer-dev --format "{{.Names}}" | grep -q openwebui-chat-analyzer-dev; then \
		docker exec -it openwebui-chat-analyzer-dev /bin/bash; \
	elif docker ps --filter name=openwebui-chat-analyzer --format "{{.Names}}" | grep -q openwebui-chat-analyzer; then \
		docker exec -it openwebui-chat-analyzer /bin/bash; \
	else \
		echo "❌ No running OpenWebUI analyzer container found"; \
		echo "Try: make dev"; \
	fi

shell-dev: ## Open shell in development container
	@echo "🐚 Opening shell in development container..."
	@if docker ps --filter name=openwebui-chat-analyzer-dev --format "{{.Names}}" | grep -q openwebui-chat-analyzer-dev; then \
		docker exec -it openwebui-chat-analyzer-dev /bin/bash; \
	else \
		echo "❌ No running development container found"; \
		echo "Try: make dev"; \
	fi

inspect: ## Inspect the container
	@echo "🔍 Container inspection:"
	@if docker ps -a --filter name=openwebui-chat-analyzer --format "{{.Names}}" | grep -q openwebui-chat-analyzer; then \
		docker inspect openwebui-chat-analyzer; \
	elif docker ps -a --filter name=openwebui-chat-analyzer-dev --format "{{.Names}}" | grep -q openwebui-chat-analyzer-dev; then \
		docker inspect openwebui-chat-analyzer-dev; \
	else \
		echo "❌ No OpenWebUI analyzer container found"; \
	fi

stats: ## Show container resource usage
	@echo "📊 Container stats:"
	@if docker ps --filter name=openwebui --format "{{.Names}}" | grep -q openwebui; then \
		docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $$(docker ps --filter name=openwebui --format "{{.Names}}"); \
	else \
		echo "❌ No running OpenWebUI analyzer containers found"; \
	fi

health: ## Check container health
	@echo "🏥 Health check:"
	@if docker ps --filter name=openwebui --format "{{.Names}}" | grep -q openwebui; then \
		for container in $$(docker ps --filter name=openwebui --format "{{.Names}}"); do \
			echo "Checking $$container..."; \
			docker exec $$container curl -f http://localhost:8501/_stcore/health 2>/dev/null && echo "✅ $$container is healthy" || echo "❌ $$container health check failed"; \
		done; \
	else \
		echo "❌ No running OpenWebUI analyzer containers found"; \
	fi

# Testing
test: ## Run basic connectivity test
	@echo "🧪 Testing container connectivity..."
	@timeout 30 bash -c 'until curl -f http://localhost:8501/_stcore/health; do sleep 1; done' && echo "✅ Test passed!" || echo "❌ Test failed!"

test-upload: ## Test with sample data (requires sample.json)
	@echo "🧪 Testing file upload functionality..."
	@if [ -f "sample.json" ]; then \
		echo "Sample file found, manual test required via web interface"; \
	else \
		echo "❌ sample.json not found. Create a sample Open WebUI export for testing."; \
	fi

# Cleanup
clean: ## Remove containers and images
	@echo "🧹 Cleaning up..."
	-docker stop openwebui-chat-analyzer openwebui-chat-analyzer-dev
	-docker rm openwebui-chat-analyzer openwebui-chat-analyzer-dev
	-docker rmi openwebui-chat-analyzer:latest openwebui-chat-analyzer:dev
	docker system prune -f

clean-all: ## Remove everything including volumes
	@echo "🧹 Deep cleaning (including volumes)..."
	docker-compose down -v
	-docker stop openwebui-chat-analyzer openwebui-chat-analyzer-dev
	-docker rm openwebui-chat-analyzer openwebui-chat-analyzer-dev
	-docker rmi openwebui-chat-analyzer:latest openwebui-chat-analyzer:dev
	docker system prune -a -f --volumes

# Data management
backup: ## Backup data volume
	@echo "💾 Backing up data..."
	@mkdir -p backups
	docker run --rm \
		-v $$(pwd)/data:/source:ro \
		-v $$(pwd)/backups:/backup \
		alpine tar czf /backup/data-backup-$$(date +%Y%m%d-%H%M%S).tar.gz -C /source .
	@echo "✅ Backup complete!"

restore: ## Restore data from backup (usage: make restore BACKUP=filename)
	@if [ -z "$(BACKUP)" ]; then \
		echo "❌ Usage: make restore BACKUP=data-backup-YYYYMMDD-HHMMSS.tar.gz"; \
		exit 1; \
	fi
	@echo "📥 Restoring data from $(BACKUP)..."
	@mkdir -p data
	docker run --rm \
		-v $$(pwd)/data:/target \
		-v $$(pwd)/backups:/backup:ro \
		alpine tar xzf /backup/$(BACKUP) -C /target
	@echo "✅ Restore complete!"

list-backups: ## List available backups
	@echo "📦 Available backups:"
	@ls -la backups/data-backup-*.tar.gz 2>/dev/null || echo "No backups found"

# Deployment helpers
deploy-local: build run ## Build and run locally
	@echo "✅ Local deployment complete!"

deploy-prod: ## Deploy to production (requires setup)
	@echo "🚀 Production deployment..."
	@echo "⚠️  Make sure to configure:"
	@echo "   - SSL certificates in nginx/ssl/"
	@echo "   - Domain name in nginx/nginx.conf"
	@echo "   - Environment variables"
	docker-compose --profile production up -d --build
	@echo "✅ Production deployment complete!"

# Quick development workflow
dev-quick: stop-dev dev-detached ## Stop any existing dev container and start fresh
	@echo "✅ Quick development restart complete!"

# Diagnostic commands
debug: ## Show debugging information
	@echo "🔍 Debug Information"
	@echo "==================="
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose version: $$(docker-compose --version 2>/dev/null || docker compose version 2>/dev/null || echo 'Not installed')"
	@echo "Make version: $$(make --version | head -1 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "📦 All containers:"
	@docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers found"
	@echo ""
	@echo "🌐 Docker Compose services:"
	@docker-compose ps 2>/dev/null || echo "No compose services found"
	@echo ""
	@echo "🗂️ Project files:"
	@ls -la | grep -E "(Dockerfile|docker-compose|Makefile|\.py$$)" || echo "No relevant files found"

ps: ## Show all containers (alias for status)
	@$(MAKE) status

# Update workflow
update: ## Pull latest changes and rebuild
	@echo "🔄 Updating application..."
	git pull
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Update complete!"

# Maintenance
restart: ## Restart the container
	@echo "🔄 Restarting container..."
	@if docker ps --filter name=openwebui-chat-analyzer-dev --format "{{.Names}}" | grep -q openwebui-chat-analyzer-dev; then \
		docker restart openwebui-chat-analyzer-dev; \
		echo "✅ Development container restarted!"; \
	elif docker ps --filter name=openwebui-chat-analyzer --format "{{.Names}}" | grep -q openwebui-chat-analyzer; then \
		docker restart openwebui-chat-analyzer; \
		echo "✅ Production container restarted!"; \
	else \
		echo "❌ No running containers found"; \
	fi

quick-fix: ## Rebuild and restart to fix common issues
	@echo "🔧 Quick fix: rebuilding containers..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Quick fix complete!"
	@echo "📊 Access: http://localhost:8501"

fix-permissions: ## Fix matplotlib and data directory permissions
	@echo "🔧 Fixing permissions..."
	@if docker ps --filter name=openwebui --format "{{.Names}}" | grep -q openwebui; then \
		for container in $$(docker ps --filter name=openwebui --format "{{.Names}}"); do \
			echo "Fixing permissions in $$container..."; \
			docker exec -u root $$container mkdir -p /tmp/matplotlib /app/data; \
			docker exec -u root $$container chown -R appuser:appuser /tmp/matplotlib /app/data; \
		done; \
		echo "✅ Permissions fixed!"; \
	else \
		echo "❌ No running containers found"; \
	fi

# Information
info: ## Show system information
	@echo "ℹ️  System Information:"
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"  
	@echo "Available images:"
	@docker images openwebui-chat-analyzer
	@echo ""
	@echo "Running containers:"
	@docker ps --filter name=openwebui-chat-analyzer

urls: ## Show application URLs
	@echo "🌐 Application URLs:"
	@echo "Main app: http://localhost:8501"
	@echo "Health check: http://localhost:8501/_stcore/health"
	@if docker ps --filter name=nginx --format "table {{.Names}}" | grep -q nginx; then \
		echo "Nginx proxy: http://localhost"; \
	fi

# VS Code integration helpers
code: ## Open project in VS Code
	@echo "📝 Opening project in VS Code..."
	code .

edit: ## Quick edit the main Python file
	@echo "📝 Opening main analyzer file..."
	code openwebui_chat_analyzer.py