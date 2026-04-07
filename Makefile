#!make
.PHONY: help deps activate test test/unit test/integration lint format install dev-install clean/venv reinstall

# If someone runs just 'make', show the help
.DEFAULT_GOAL := help

# --- OS Detection and Configuration ---
SHELL := /bin/bash
.ONESHELL:

# Windows specific overrides
ifeq ($(OS),Windows_NT)
    SHELL := cmd.exe
    VENV_DIR = Scripts
    PYTHON_EXE = python.exe
    SET_ENV = set
else
    VENV_DIR = bin
    PYTHON_EXE = python
    SET_ENV = export
endif

IN_CONTAINER := $(shell (test -f /.dockerenv || test -n "$$KUBERNETES_SERVICE_HOST") && echo 1 || echo 0)

ifeq ($(IN_CONTAINER),1)
  # Inside container: use system Python (packages already installed in Docker image)
  PYTHON ?= python
else
  # Outside container: use virtual environment
  VENV ?= env
  PYTHON ?= $(VENV)/$(VENV_DIR)/$(PYTHON_EXE)
endif

# --- Core Targets ---
help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_\/-]+:.*?## / {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

ifeq ($(IN_CONTAINER),1)
# Inside container: no venv needed, packages are in system Python
.PHONY: activate
activate: ## No-op inside container (packages already installed)
	@echo "Running inside container - using system Python with pre-installed packages."

clean/venv: ## Remove virtual environment (no-op in container)
	@echo "No venv to clean inside container."

reinstall: ## No-op inside container
	@echo "No venv to reinstall inside container."
else
# Outside container: use virtual environment
.PHONY: $(VENV)/$(VENV_DIR)/activate
$(VENV)/$(VENV_DIR)/activate: pyproject.toml ## Create virtualenv and install dependencies
	@echo "Creating virtual environment at $(VENV)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv venv "$(VENV)" --clear || python3 -m venv "$(VENV)"; \
		uv pip install --python "$(VENV)/$(VENV_DIR)/$(PYTHON_EXE)" -e ".[dev,clipboard]"; \
	else \
		python3 -m venv "$(VENV)"; \
		"$(VENV)/$(VENV_DIR)/pip" install -U pip; \
		"$(VENV)/$(VENV_DIR)/pip" install -e ".[dev,clipboard]"; \
	fi
	@echo "Virtual environment setup complete."

clean/venv: ## Remove virtual environment
	rm -rf $(VENV)

reinstall: clean/venv activate ## Clean and reinstall everything

activate: $(VENV)/$(VENV_DIR)/activate ## Activate the virtual environment
endif

# --- Installation ---
install: activate ## Install anton in production mode
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python "$(PYTHON)" -e .; \
	else \
		$(PYTHON) -m pip install -e .; \
	fi

dev-install: activate ## Install anton in development mode with all extras
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python "$(PYTHON)" -e ".[dev,clipboard]"; \
	else \
		$(PYTHON) -m pip install -e ".[dev,clipboard]"; \
	fi

# --- Testing ---
test/unit: activate ## Run unit tests
	$(PYTHON) -m pytest tests/

test/unit/coverage: activate ## Run unit tests with coverage
	$(PYTHON) -m pytest --cov=anton tests/ --cov-report=term-missing

test/unit/live: activate ## Run unit tests with live API calls (requires API keys)
	$(PYTHON) -m pytest tests/ --live

test: test/unit ## Run standard tests

coverage/html: activate ## Generate HTML coverage report
	$(PYTHON) -m pytest --cov=anton tests/ --cov-report html
	@echo "Coverage report generated in htmlcov/"

# --- Linting and Formatting ---
lint: check/lint format/check ## Run all linting and formatting checks

check/lint: activate ## Check code style with Ruff
	$(PYTHON) -m ruff check anton tests

format/check: activate ## Check code formatting with Ruff
	$(PYTHON) -m ruff format anton tests --check

format: activate ## Format code with Ruff
	$(PYTHON) -m ruff format anton tests

check/fix: activate ## Fix linting issues with Ruff
	$(PYTHON) -m ruff check anton tests --fix

# --- Development ---
run: activate ## Run anton CLI
	$(PYTHON) -m anton

demo: activate ## Run anton with demo bug report
	$(PYTHON) -m anton demo_bug_report.md

# --- Build ---
build: activate ## Build distribution packages
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python "$(PYTHON)" build; \
		$(PYTHON) -m build; \
	else \
		$(PYTHON) -m pip install build; \
		$(PYTHON) -m build; \
	fi

clean/build: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info

# --- Documentation ---
docs: activate ## Generate documentation (placeholder for future)
	@echo "Documentation generation not yet implemented"

# --- Release ---
release/check: lint test ## Check if ready for release
	@echo "✓ All checks passed - ready for release"

clean: clean/venv clean/build ## Clean all generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
