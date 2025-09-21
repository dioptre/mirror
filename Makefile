# Avatar Mirror System Makefile
# Requires uv: https://github.com/astral-sh/uv

.PHONY: help install install-dev install-gpu install-cpu install-face-swap run test lint format clean setup-models setup-face-swap godot demo

# Default target
help:
	@echo "Avatar Mirror System - Make Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install basic dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-gpu      Install with GPU acceleration support"
	@echo "  install-cpu      Install CPU-only version (for development)"
	@echo "  install-face-swap Install with face swapping support"
	@echo "  install-all      Install everything (GPU + face swap + dev)"
	@echo ""
	@echo "Development Commands:"
	@echo "  run              Run the avatar mirror system"
	@echo "  run-debug        Run with debug logging and visualization"
	@echo "  test             Run tests"
	@echo "  lint             Run linting (flake8, mypy)"
	@echo "  format           Format code (black, isort)"
	@echo "  clean            Clean up generated files"
	@echo ""
	@echo "Special Setup:"
	@echo "  setup-models     Download and setup all AI models"
	@echo "  setup-face-swap  Setup Deep-Live-Cam face swapping (interactive)"
	@echo "  setup-gpu        Setup for GPU acceleration" 
	@echo "  setup-cpu        Setup for CPU-only development"
	@echo ""
	@echo "Godot Integration:"
	@echo "  godot            Open Godot client project"
	@echo "  demo             Run avatar mirror + open Godot demo"

# Basic installation
install:
	uv pip install -e .

# Development installation
install-dev:
	uv pip install -e ".[dev]"

# GPU installation
install-gpu:
	uv pip install -e ".[gpu,dev]"
	
# CPU-only installation (for development)
install-cpu:
	uv pip install -e ".[cpu,dev]"
	uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade

# Face swapping installation  
install-face-swap:
	uv pip install -e ".[face-swap,dev]"

# Install everything
install-all:
	uv pip install -e ".[all]"

# Quick setups
setup-gpu: install-gpu
	@echo "✅ GPU setup complete"
	@echo "Run with: make run"

setup-cpu: 
	uv pip install -e ".[cpu,dev]"
	uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade
	@echo "✅ CPU-only setup complete"  
	@echo "Run with: make run"

# Model setup
setup-models:
	uv run python scripts/setup_models.py
	@echo "✅ All AI models setup complete"

# Face swapping setup (interactive)
setup-face-swap:
	uv pip install -e ".[face-swap,dev]"
	uv run python scripts/setup_face_swap.py --enable
	@echo "✅ Face swapping setup complete"

# Run the system
run:
	uv run python -m src.main

# Run with debug logging
run-debug:
	LOG_LEVEL=DEBUG uv run python -m src.main

# Run tests
test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=html

# Linting
lint:
	uv run flake8 src tests
	uv run mypy src

# Formatting
format:
	uv run black src tests scripts
	uv run isort src tests scripts

# Check formatting
format-check:
	uv run black --check src tests scripts
	uv run isort --check-only src tests scripts

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf cache/ *.log

# Development workflow
dev-setup: install-dev
	uv run pre-commit install
	@echo "✅ Development environment ready"

# Check everything before commit
check: format-check lint test
	@echo "✅ All checks passed"

# Initialize project (first time setup)
init:
	@echo "🚀 Initializing Avatar Mirror System..."
	@echo "Creating virtual environment..."
	uv venv
	@echo "Installing dependencies..."
	$(MAKE) install-dev
	@echo "✅ Project initialized!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate environment: source .venv/bin/activate"
	@echo "2. For GPU support: make setup-gpu"
	@echo "3. For CPU-only: make setup-cpu" 
	@echo "4. For face swapping: make setup-face-swap"
	@echo "5. Run system: make run"

# Quick start for different use cases
quick-start-gpu:
	uv venv
	$(MAKE) setup-gpu
	@echo "🎉 Ready for GPU development!"

quick-start-cpu:
	uv venv
	@echo "Installing CPU-optimized dependencies..."
	uv pip install -e ".[cpu,dev]"
	uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade
	@echo "🎉 Ready for CPU development!"
	@echo "Activate with: source .venv/bin/activate"
	@echo "Then run: make run"

quick-start-full:
	uv venv
	$(MAKE) install-all
	$(MAKE) setup-face-swap
	@echo "🎉 Full system ready!"

# Show system info
# Godot integration
godot:
	@echo "🎮 Opening Godot client project..."
	@if command -v godot >/dev/null 2>&1; then \
		godot --path godot_client; \
	else \
		echo "❌ Godot not found in PATH"; \
		echo "Please install Godot 4.3+ or open godot_client/ manually"; \
		open godot_client; \
	fi

demo:
	@echo "🎬 Starting Avatar Mirror demo..."
	@echo "1. Starting Avatar Mirror System..."
	$(MAKE) run &
	@sleep 3
	@echo "2. Opening Godot demo..."
	$(MAKE) godot

info:
	@echo "Avatar Mirror System Information"
	@echo "==============================="
	uv --version
	@echo "Python version:"
	uv run python --version  
	@echo "Virtual environment:"
	@echo "${VIRTUAL_ENV}"
	@echo "Dependencies:"
	uv pip list | head -20