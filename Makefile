#!make
.PHONY: help deps activate test test/all lint format

# If someone runs just 'make', show the help
.DEFAULT_GOAL := help

# --- OS Detection and Configuration ---
# Default to Unix-like shell unless on Windows
SHELL := /bin/bash
SET_ENV = export

SHELL := /bin/bash
.ONESHELL:

# Windows specific overrides
ifeq ($(OS),Windows_NT)
    SHELL := cmd.exe
    VENV_DIR = Scripts
    PYTHON_EXE = python.exe
    PIP_EXE = pip.exe
    SET_ENV = set
else
    VENV_DIR = bin
    PYTHON_EXE = python
    PIP_EXE = pip
    SET_ENV = export
endif

IN_CONTAINER := $(shell test -f /.dockerenv && echo 1 || echo 0)

ifeq ($(IN_CONTAINER),1)
  VENV ?= /opt/venv
else
  VENV ?= env
endif

PYTHON ?= $(VENV)/$(VENV_DIR)/$(PYTHON_EXE)
PIP ?= $(VENV)/$(VENV_DIR)/$(PIP_EXE)
DOCKER_COMMAND := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
MINDS_CONTAINER = minds

# --- Core Targets ---
help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: $(VENV)/$(VENV_DIR)/activate
$(VENV)/$(VENV_DIR)/activate: requirements/requirements-dev.txt ## Create virtualenv and install dependencies
	@echo "Creating virtual environment at $(VENV)..."
	python -m venv "$(VENV)"
	@echo "Virtual environment created. Installing requirements..."
	@ls -la $(VENV)/$(VENV_DIR)/ || echo "$(VENV_DIR) directory not found"
	$(PIP) install -r requirements/requirements-dev.txt
	$(PIP) install -e .
	@echo "Virtual environment setup complete."

activate: $(VENV)/$(VENV_DIR)/activate ## Activate the virtual environment

# Check if a docker-compose command was found. Print a help message if not
deps:
ifndef DOCKER_COMMAND
	@echo "Docker compose not found. Please install either docker-compose (the tool) or the docker compose plugin."
	exit 1
endif

# --- Testing ---
test/unit: activate ## Run unit tests
	$(PYTHON) -m pytest tests/unit/

test/integration: activate 
	$(PYTHON) -m pytest tests/integration/ -m "happy_path"


test/integration/all: activate ## Run all integration tests
	$(PYTHON) -m pytest tests/integration/

test: test/unit test/integration ## Run standard tests (unit and happy_path integration)

test/all: test/unit test/integration/all ## Run all tests (unit, integration, and contract)

# --- Coverage ---
test/unit/coverage: activate ## Run unit tests with coverage
	$(PYTHON) -m pytest --cov=minds tests/unit/ --cov-fail-under=75

coverage/html: activate ## Generate HTML coverage report
	$(PYTHON) -m pytest --cov=minds tests/unit/ --cov-report html

# --- Development ---
run: activate docker/deps ## Run the development server with auto-reload
	$(PYTHON) -m watchfiles --filter python '$(PYTHON) -m uvicorn minds.server:app --host 0.0.0.0 --port 9010' .

migrate: activate ## Run Alembic database migrations
	$(PYTHON) -m alembic upgrade head

# --- Docker ---
docker/deps: ## Start docker dependencies (Postgres, Redis, etc.)
	$(DOCKER_COMMAND) up -d postgres redis langfuse-web langfuse-worker migrate

docker/build: ## Build the docker image
	$(SET_ENV) DOCKER_BUILDKIT=1 && $(DOCKER_COMMAND) build

docker/run: docker/deps ## Run the full application in Docker
	$(SET_ENV) DOCKER_BUILDKIT=1 && $(DOCKER_COMMAND) up

docker/stop: ## Stop the docker containers
	$(DOCKER_COMMAND) down

# --- Linting and Formatting ---
lint: check/lint format/check ## Run all linting and formatting checks

check/lint: activate ## Check code style with Ruff
	$(PYTHON) -m ruff check minds tests

format/check: activate ## Check code formatting with Ruff
	$(PYTHON) -m ruff format minds tests --check

format: activate ## Format code with Ruff
	$(PYTHON) -m ruff format minds tests

check/fix: activate ## Format code with ruff
	$(PYTHON) -m ruff check minds tests --fix
