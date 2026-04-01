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
PREFECT_CONFIG ?= prefect.yaml

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
	$(PYTHON) -m pytest --cov=minds tests/unit/ --cov-fail-under=80

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

prefect/secrets: ## Deploy Prefect secrets from local settings
	$(PYTHON) -c "from minds.jobs.settings import create_prefect_settings; create_prefect_settings(); print('✓ Prefect secrets deployed successfully')"

prefect/deploy: ## Deploy all flows to Prefect (uses PREFECT_CONFIG, default: prefect.yaml)
	@echo "Deploying all flows using $(PREFECT_CONFIG)..."
	@echo "Auto-rejecting all deployment prompts..."
	@if command -v $(VENV)/bin/prefect >/dev/null 2>&1; then \
		yes n | $(VENV)/bin/prefect deploy --all --prefect-file $(PREFECT_CONFIG); \
	else \
		echo "Error: Prefect not found. Please ensure it's installed in your virtual environment."; \
		echo "Try running: $(PIP) install prefect"; \
		exit 1; \
	fi

# Set dynamic image for CI/CD deployments
prefect/set-image-tag: ## Set the image tag dynamically (usage: make prefect/set-image-tag IMAGE_TAG=development-abc123 or set IMAGE_TAG env var)
	@# Set IMAGE_TAG from environment variable if not provided as make variable
	$(eval IMAGE_TAG ?= $(shell echo $$IMAGE_TAG))
	@# Check if IMAGE_TAG is still empty after trying environment variable
	@if [ -z "$(IMAGE_TAG)" ]; then \
		echo "Error: IMAGE_TAG is required. Set it as environment variable or pass as make variable: make prefect/set-image IMAGE_TAG=development-abc123"; \
		exit 1; \
	fi
	@echo "Setting image tag to: $(IMAGE_TAG) in $(PREFECT_CONFIG)"
	sed -i.bak 's|IMAGE_TAG_PLACEHOLDER|$(IMAGE_TAG)|g' $(PREFECT_CONFIG)
	@echo "✓ Image tag updated in $(PREFECT_CONFIG)"

# Set dynamic environment name for deployment names
prefect/set-env: ## Set the environment name for deployment names (usage: make prefect/set-env ENV=dev or set ENV env var)
	@# Set ENV from environment variable if not provided as make variable
	$(eval ENV ?= $(shell echo $$ENV))
	@# Check if ENV is still empty after trying environment variable
	@if [ -z "$(ENV)" ]; then \
		echo "Error: ENV is required. Set it as environment variable or pass as make variable: make prefect/set-env ENV=dev"; \
		exit 1; \
	fi
	@echo "Setting environment name to: $(ENV) in $(PREFECT_CONFIG)"
	sed -i.bak 's|ENV_PLACEHOLDER|$(ENV)|g' $(PREFECT_CONFIG)
	@echo "✓ Environment name updated in $(PREFECT_CONFIG)"


prefect/set-config: ## Set image, environment if provided (usage: make prefect/set-config IMAGE_TAG=dev-123 ENV=dev)
	@# Set image tag if IMAGE_TAG is provided
	$(MAKE) prefect/set-image-tag;

	@# Set environment name if ENV is provided
	$(MAKE) prefect/set-env;

prefect/deploy/full: activate prefect/secrets prefect/set-config prefect/deploy ## Set config (image/API URL if provided), and then deploy all flows

# --- Docker-specific Prefect Commands ---
prefect/docker/deploy: ## Deploy all flows using prefect.docker.yaml
	$(MAKE) prefect/deploy PREFECT_CONFIG=prefect.docker.yaml

prefect/docker/set-image-tag: ## Set image tag in prefect.docker.yaml (usage: make prefect/docker/set-image-tag IMAGE_TAG=dev-123)
	$(MAKE) prefect/set-image-tag PREFECT_CONFIG=prefect.docker.yaml

prefect/docker/set-env: ## Set environment name in prefect.docker.yaml (usage: make prefect/docker/set-env ENV=dev)
	$(MAKE) prefect/set-env PREFECT_CONFIG=prefect.docker.yaml

prefect/docker/set-config: ## Set image and environment in prefect.docker.yaml (usage: make prefect/docker/set-config IMAGE_TAG=dev-123 ENV=dev)
	$(MAKE) prefect/set-config PREFECT_CONFIG=prefect.docker.yaml

prefect/docker/deploy/full: activate prefect/secrets ## Set config and deploy all flows using prefect.docker.yaml
	$(MAKE) prefect/set-config PREFECT_CONFIG=prefect.docker.yaml
	$(MAKE) prefect/deploy PREFECT_CONFIG=prefect.docker.yaml
