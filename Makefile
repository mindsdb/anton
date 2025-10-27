#!make
.PHONY: help deps

# If someone runs just 'make', show the help
.DEFAULT_GOAL := help

# Find the docker-compose command on the current system
DOCKER_COMMAND :=   $(shell docker-compose -v > /dev/null 2>&1; \
					if [ $$? -eq 0 ]; then \
						echo "docker-compose"; \
					else \
						docker compose -v > /dev/null 2>&1; \
						if [ $$? -eq 0 ]; then \
							echo "docker compose"; \
						fi; \
					fi;)

VENV = env
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

MINDS_CONTAINER = minds

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: $(VENV)/bin/activate
$(VENV)/bin/activate: requirements/requirements-dev.txt # Create virtualenv and install dependencies
	python3 -m venv "$(VENV)"
	$(PIP) install -r requirements/requirements-dev.txt
	$(PIP) install -e .

activate: $(VENV)/bin/activate # Activate virtualenv
	@echo "activate virtualenv"

# Check if a docker-compose command was found. Print a help message if not
deps:
ifndef DOCKER_COMMAND
	@echo "Docker compose not found. Please install either docker-compose (the tool) or the docker compose plugin."
	exit 1
endif

# Run unit tests
test/unit: activate
	$(PYTHON) -m pytest tests/unit/

# Run integration tests
test/integration: activate
	$(PYTHON) -m pytest tests/integration/

# Run all tests
test: test/unit test/integration

# Coverage
test/unit/coverage: activate
	$(PYTHON) -m pytest --cov=minds tests/unit/ --cov-fail-under=85

coverage/html: activate
	$(PYTHON) -m pytest --cov=minds tests/unit/ --cov-report html

# Run the server
run: activate docker/deps
	$(PYTHON) -m watchfiles --filter python '$(PYTHON) -m uvicorn minds.server:app --host 0.0.0.0 --port 9010' .

# Run docker deps
docker/deps:
	$(DOCKER_COMMAND) up -d postgres redis langfuse-web langfuse-worker migrate

# Build the docker image
docker/build: deps
	export DOCKER_BUILDKIT=1; \
	$(DOCKER_COMMAND) build

# Run the docker container
docker/run: deps
	export DOCKER_BUILDKIT=1; \
	$(DOCKER_COMMAND) up

# Stop the docker container
docker/stop: deps
	$(DOCKER_COMMAND) down

# Run database migrations
migrate: activate ## Run alembic database migrations
	$(PYTHON) -m alembic upgrade head

lint: check/lint format/check

check/lint: activate ## Check code style with ruff
	$(PYTHON) -m ruff check minds tests

format/check: activate ## Format code with ruff
	$(PYTHON) -m ruff format minds tests --check

format: activate ## Format code with ruff
	$(PYTHON) -m ruff format minds tests

check/fix: activate ## Format code with ruff
	$(PYTHON) -m ruff check minds tests --fix

prefect/secrets: ## Deploy Prefect secrets from local settings
	$(PYTHON) -c "from minds.jobs.settings import create_prefect_settings; create_prefect_settings(); print('✓ Prefect secrets deployed successfully')"

prefect/deploy: ## Deploy all flows to Prefect (requires secrets to be deployed first)
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
