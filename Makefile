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

# Run the server
run: activate docker/deps
	$(PYTHON) -m watchfiles --filter python '$(PYTHON) -m uvicorn minds.server:app --host 0.0.0.0 --port 9010' .

# Run docker deps
docker/deps:
	$(DOCKER_COMMAND) up -d postgres redis langfuse migrate

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
