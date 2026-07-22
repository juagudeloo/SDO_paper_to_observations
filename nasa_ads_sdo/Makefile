# NASA ADS SDO API Makefile
# Simplifies common development tasks

.PHONY: help install clean run dev check-db

# Default target
help:
	@echo "NASA ADS SDO API - Available commands:"
	@echo ""
	@echo "  make install       - Install dependencies and set up virtual environment"
	@echo "  make run           - Run the API in production mode"
	@echo "  make dev           - Run the API in development mode with auto-reload"
	@echo "  make clean         - Remove virtual environment and cache files"
	@echo "  make check-db      - Check if database exists and show stats"
	@echo "  make help          - Show this help message"

# Install dependencies and set up environment
install:
	@echo "🚀 Setting up NASA ADS SDO API..."
	@bash setup.sh

# Run the API in production mode
run:
	@echo "🚀 Starting API server..."
	@bash run_api.sh

# Run the API in development mode
dev:
	@echo "🚀 Starting API server in development mode..."
	@bash run_dev.sh

# Clean up generated files and virtual environment
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf venv/
	@rm -rf api/__pycache__/
	@rm -rf api/modules/__pycache__/
	@rm -rf api/scripts/__pycache__/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Check if database exists and show basic info
check-db:
	@if [ -f "api/database/sdo_papers_2010_2024.db" ]; then \
		echo "✅ Database found: api/database/sdo_papers_2010_2024.db"; \
		echo "📊 Database size: $$(du -h api/database/sdo_papers_2010_2024.db | cut -f1)"; \
	else \
		echo "❌ Database not found at api/database/sdo_papers_2010_2024.db"; \
		echo "Please ensure the database file exists before running the API."; \
	fi