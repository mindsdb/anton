#!make
.PHONY: help deps activate test test/all lint format

# If someone runs just 'make', show the help
.DEFAULT_GOAL := help

# --- OS Detection and Configuration ---
# Default to Unix-like shell unless on Windows
SHELL := /bin/bash
SET_ENV = export

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
endif

# --- Variables ---
VENV = env
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

$(VENV)/$(VENV_DIR)/activate: requirements/requirements-dev.txt ## Create virtualenv and install dependencies
	python -m venv "$(VENV)"
	$(PIP) install -r requirements/requirements-dev.txt
	$(PIP) install -e .

activate: $(VENV)/$(VENV_DIR)/activate ## Activate the virtual environment

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
	$(PYTHON) -m pytest --cov=minds tests/unit/ --cov-fail-under=85

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

prefect/deploy: activate ## Deploy all flows to Prefect
	@echo "Deploying all flows..."
	@echo "Auto-rejecting all deployment prompts..."
	@if command -v $(VENV)/bin/prefect >/dev/null 2>&1; then \
		yes n | $(VENV)/bin/prefect deploy --all; \
	else \
		echo "Error: Prefect not found. Please ensure it's installed in your virtual environment."; \
		echo "Try running: $(PIP) install prefect"; \
		exit 1; \
	fi

# Set dynamic image for CI/CD deployments
prefect/set-image: ## Set the image tag dynamically (usage: make prefect/set-image IMAGE_TAG=development-abc123 or set IMAGE_TAG env var)
	@# Set IMAGE_TAG from environment variable if not provided as make variable
	$(eval IMAGE_TAG ?= $(shell echo $$IMAGE_TAG))
	@# Check if IMAGE_TAG is still empty after trying environment variable
	@if [ -z "$(IMAGE_TAG)" ]; then \
		echo "Error: IMAGE_TAG is required. Set it as environment variable or pass as make variable: make prefect/set-image IMAGE_TAG=development-abc123"; \
		exit 1; \
	fi
	@echo "Setting image tag to: $(IMAGE_TAG)"
	sed -i.bak 's|IMAGE_TAG_PLACEHOLDER|$(IMAGE_TAG)|g' prefect.yaml
	@echo "✓ Image tag updated in prefect.yaml"

# Set dynamic environment name for deployment names
prefect/set-env: ## Set the environment name for deployment names (usage: make prefect/set-env ENV=dev or set ENV env var)
	@# Set ENV from environment variable if not provided as make variable
	$(eval ENV ?= $(shell echo $$ENV))
	@# Check if ENV is still empty after trying environment variable
	@if [ -z "$(ENV)" ]; then \
		echo "Error: ENV is required. Set it as environment variable or pass as make variable: make prefect/set-env ENV=dev"; \
		exit 1; \
	fi
	@echo "Setting environment name to: $(ENV)"
	sed -i.bak 's|ENV_PLACEHOLDER|$(ENV)|g' prefect.yaml
	@echo "✓ Environment name updated in prefect.yaml"

prefect/set-config: ## Set image, environment, and API URL if provided (usage: make prefect/set-config IMAGE_TAG=dev-123 ENV=dev PREFECT_API_URL=http://prefect-server.dev.svc.cluster.local:4200/api)
	@# Set image tag if IMAGE_TAG is provided
	@if [ ! -z "$(IMAGE_TAG)" ] || [ ! -z "$$IMAGE_TAG" ]; then \
		$(MAKE) prefect/set-image; \
	fi
	@# Set environment name if ENV is provided
	@if [ ! -z "$(ENV)" ] || [ ! -z "$$ENV" ]; then \
		$(MAKE) prefect/set-env; \
	fi

prefect/deploy/full: activate prefect/secrets prefect/set-config prefect/deploy ## Set config (image/API URL if provided), and then deploy all flows
