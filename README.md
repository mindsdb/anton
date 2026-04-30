# Minds

Minds is a FastAPI-based service that provides an OpenAI-compatible chat completions endpoint. It supports both streaming and non-streaming responses, with comprehensive logging and observability features.

## Features

- OpenAI-compatible `/chat/completions` endpoint
- Streaming and non-streaming response modes
- Built-in logging and performance monitoring
- Langfuse integration for observability
- CORS support for web applications
- Docker support for containerized deployment
- PostgreSQL database integration with SQLModel
- Alembic database migrations
- UUID-based primary keys with automatic generation
- MindsDB SDK integration for AI model management and data operations


## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment support
- PostgreSQL 16+ (or use Docker Compose for local development)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd minds
```

2. Create and activate a virtual environment:
```bash
make activate
```

This will:
- Create a Python virtual environment in the `env/` directory
- Install all dependencies from `requirements/requirements-dev.txt`
- Install the project in development mode

### Environment Configuration

Create a `.env` file in the root directory with your configuration:

```env
# Database Configuration
DATABASE__URI=postgresql://minds:minds@localhost:35432/minds
DATABASE__POOL_SIZE=20
DATABASE__MAX_OVERFLOW=20
DATABASE__POOL_TIMEOUT=300
DATABASE__POOL_RECYCLE=300
DATABASE__POOL_PRE_PING=true
DATABASE__QUERY_TIMEOUT=300
DATABASE__STATEMENT_TIMEOUT=300000

# Docker Compose Database URIs (for containerized deployment)
MINDS_DB_URI_DOCKER=postgresql://minds:minds@postgres:5432/minds
LANGFUSE_DB_URI=postgresql://langfuse:langfuse@postgres:5432/langfuse

# PostgreSQL Configuration (for Docker Compose)
PG_USER=postgres
PG_PASSWORD=postgres
PG_PORT=35432

# Redis Configuration (for Docker Compose)
REDIS_PORT=36379

# Service Ports
MIND_PORT=9010
LANGFUSE_PORT=3001

# OpenAI Configuration
OPENAI__API_URL=https://api.openai.com/v1
OPENAI__API_KEY=your-api-key-here
OPENAI__MODEL_NAME=gpt-4o
OPENAI__MAX_TOKENS=400000

# MindsDB Configuration
MINDSDB__URL=https://cloud.mindsdb.com

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=false
LOG_DIR=logs

# Langfuse Configuration
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_INIT_PROJECT_ID=your-project-uuid
LANGFUSE_INIT_PROJECT_NAME=MindsDB
LANGFUSE_INIT_USER_EMAIL=admin@example.com
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=admin123
```

### Database Setup

The application uses PostgreSQL with SQLModel for database operations and Alembic for migrations.

#### Local Development with Docker Compose

The easiest way to get started is using Docker Compose, which provides PostgreSQL, Redis, and Langfuse services:

```bash
# Start all services (PostgreSQL, Redis, Langfuse, and the application)
make docker/run

# Stop all services
make docker/stop
```

This will:
- Start PostgreSQL on port 35432
- Start Redis on port 36379  
- Start Langfuse on port 3001
- Run database migrations automatically
- Start the Minds application on port 9010

#### Manual Database Setup

If you prefer to use an external PostgreSQL instance:

1. **Create the database and user:**
   ```sql
   CREATE DATABASE minds;
   CREATE USER minds WITH PASSWORD 'minds';
   GRANT ALL PRIVILEGES ON DATABASE minds TO minds;
   ```

2. **Set the DATABASE__URI environment variable:**
   ```bash
   export DATABASE__URI="postgresql://minds:minds@localhost:5432/minds"
   ```

3. **Run migrations:**
   ```bash
   make migrate
   ```

#### Database Migrations

The project uses Alembic for database schema management:

```bash
# Run all pending migrations
make migrate

# Or run alembic directly
python -m alembic upgrade head

# Create a new migration (after modifying models)
python -m alembic revision --autogenerate -m "Description of changes"

# Check migration status
python -m alembic current

# View migration history
python -m alembic history
```

#### Database Models

The application uses SQLModel (built on SQLAlchemy) for database models:

- **BaseSQLModel**: Base class with UUID primary key, created_at, and modified_at fields
- **Mind**: Example model representing a "mind" entity

Example usage:
```python
from minds.db.pg_session import get_session
from minds.model.mind import Mind

# Create a new mind
session = get_session()
mind = Mind(name="My AI Mind")
session.add(mind)
session.commit()
session.refresh(mind)  # Get the generated UUID and timestamps
print(f"Created mind with ID: {mind.id}")
session.close()
```

## Available Commands

The project includes a comprehensive Makefile with the following commands:

### Development Commands

- `make help` - Display all available commands with descriptions
- `make activate` - Create and activate the Python virtual environment
- `make run` - Start the development server with auto-reload on `http://0.0.0.0:9010`
- `make migrate` - Run Alembic database migrations

#### Auto-Reload Development Server

The `make run` command uses `watchfiles` to automatically restart the server when Python files change:

```bash
make run
```

This command will:
1. Start the required Docker services (PostgreSQL, Redis, Langfuse) if not already running
2. Start the FastAPI server with auto-reload enabled
3. Monitor all Python files for changes and restart the server automatically
4. Serve the application on `http://0.0.0.0:9010`

The auto-reload feature makes development faster by eliminating the need to manually restart the server after code changes.

### Testing Commands

- `make test/unit` - Run unit tests only
- `make test/integration` - Run integration tests only  
- `make test` - Run all tests (unit + integration)
- `make test/unit/coverage` - Run unit tests with coverage reporting (requires 85% coverage)
- `make coverage/html` - Generate HTML coverage report in `htmlcov/` directory

#### Test Coverage

The project maintains high test coverage standards:

- **Minimum Coverage**: 85% coverage required for all unit tests
- **Coverage Tools**: Uses `pytest-cov` for coverage reporting
- **HTML Reports**: Generate detailed HTML coverage reports with `make coverage/html`
- **CI/CD Integration**: Coverage is automatically checked in GitHub Actions workflows

The test suite includes:
- **Unit Tests**: Comprehensive testing of individual components, handlers, models, and utilities
- **Integration Tests**: End-to-end testing of API endpoints and system integration
- **Mock Testing**: Extensive use of mocks for external dependencies (MindsDB SDK, OpenAI, database)
- **Async Testing**: Full support for testing async operations and streaming responses

Coverage reports help identify untested code paths and ensure reliability. The HTML coverage report provides detailed line-by-line coverage information and is generated in the `htmlcov/` directory.

### Docker Commands

- `make docker/build` - Build the Docker image
- `make docker/run` - Run the full stack (PostgreSQL, Redis, Langfuse, and Minds application)
- `make docker/stop` - Stop and remove all Docker containers

The Docker Compose setup includes:
- **PostgreSQL 16**: Database server on port 35432
- **Redis 7.2**: Cache server on port 36379
- **Langfuse 2.87**: Observability platform on port 3001
- **Minds Application**: Main service on port 9010
- **Migration Service**: Automatically runs database migrations on startup
- **Autoheal**: Automatically restarts unhealthy containers

### Usage Examples

```bash
# Start development server (requires external PostgreSQL)
make run

# Run database migrations
make migrate

# Run tests
make test

# Build and run full stack with Docker (recommended for development)
make docker/build
make docker/run

# Quick start with Docker (includes database setup)
make docker/run  # This will start PostgreSQL, run migrations, and start the app
```

## API Endpoints

### Health Check

- **GET** `/healthz` - Returns service health status

### Chat Completions

- **POST** `/chat/completions`
- **POST** `/v1/chat/completions`

OpenAI-compatible chat completions endpoint that supports both streaming and non-streaming modes.

#### Request Format

**Non-Streaming Request:**
```json
{
    "model": "minds",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ],
    "metadata": {
        "mdb_completions_session_id": 1748503096170
    }
}
```

**Streaming Request:**
```json
{
    "model": "minds",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ],
    "metadata": {
        "mdb_completions_session_id": 1748503096170
    },
    "stream": true
}
```

#### Request Parameters

- `model` (string, required): The model identifier
- `messages` (array, required): Array of message objects with `role` and `content`
- `metadata` (object, optional): Additional metadata including session ID
- `stream` (boolean, optional): Enable streaming responses (default: false)

#### Message Roles

- `system`: System-level instructions or context
- `user`: User input messages
- `assistant`: AI assistant responses
- `function`: Function call responses

#### Response Formats

**Non-Streaming Response:**
Returns a standard OpenAI-compatible JSON response with the complete chat completion.

**Streaming Response:**
Returns Server-Sent Events (SSE) with `text/event-stream` content type. Each event contains a JSON chunk following the OpenAI streaming format.

#### Example Usage

**cURL Examples:**

Non-streaming:
```bash
curl -X POST http://localhost:9010/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minds",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"mdb_completions_session_id": 123}
  }'
```

Streaming:
```bash
curl -X POST http://localhost:9010/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minds",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "metadata": {"mdb_completions_session_id": 123}
  }'
```

## Streaming Architecture

The service uses a queue-based streaming architecture that allows for real-time message delivery:

### MessageStreamer Interface

The core of the streaming system is the `MessageStreamer` interface, which provides:

```python
await streamer.push(role=Role.system, content=f"Using model: {self.model}")
```

This method allows handlers to push messages to a queue that gets consumed by the streaming response system.

### How Streaming Works

1. **Message Queue**: Each streaming request creates a `Streamer` instance with an internal `asyncio.Queue`

2. **Producer Pattern**: Chat handlers use `streamer.push()` to add messages to the queue:
   ```python
   await streamer.push(role=Role.system, content="System message")
   await streamer.push(role=Role.assistant, content="Response content")
   ```

3. **Consumer Pattern**: The streaming response consumes messages from the queue and formats them as Server-Sent Events

4. **Real-time Delivery**: Messages are delivered to the client as soon as they're pushed to the queue, enabling real-time streaming

### Streaming vs Non-Streaming

- **Streaming Mode**: Uses `Streamer` class with asyncio.Queue for real-time message delivery
- **Non-Streaming Mode**: Uses `StreamerCollector` class to collect all messages and return them as a single JSON response

This architecture allows the same handler code to work for both streaming and non-streaming requests, with the only difference being the streamer implementation used.

## MindsDB Client Integration

The service includes a robust MindsDB SDK integration that provides seamless access to MindsDB's AI capabilities and data management features.

### MindsDB Client Features

The MindsDB client integration offers:

- **Authentication**: Secure API key-based authentication with MindsDB Cloud or on-premise instances
- **Model Management**: Access to AI models, training, and inference capabilities
- **Database Operations**: Query and manage databases connected to MindsDB
- **Error Handling**: Comprehensive error handling and validation
- **Request Context**: Automatic client creation from FastAPI request context

### Client Creation

The system provides two main functions for creating MindsDB clients:

```python
from minds.client.mindsdb import create_mindsdb_client_from_request, create_mindsdb_client

# Create client from FastAPI request (extracts Bearer token automatically)
client = create_mindsdb_client_from_request(request)

# Create client with explicit API key
client = create_mindsdb_client("your-api-key")
```

### Usage in Handlers

Chat completion handlers automatically receive a configured MindsDB client:

```python
class ChatCompletionsHandler:
    def __init__(self, session: Session, mindsdb_client: Server, messages: List[Message], model: str, stream: bool):
        self.mindsdb_client = mindsdb_client
    
    async def chat_completions(self, streamer: MessageStreamer):
        # Access MindsDB models
        models = self.mindsdb_client.models.list()
        
        # Access databases
        databases = self.mindsdb_client.databases.list()
        
        # Use MindsDB for AI operations
        # ... your AI logic here
```

### Configuration

Configure the MindsDB connection in your `.env` file:

```env
# MindsDB Configuration
MINDSDB_URL=https://cloud.mindsdb.com  # or your on-premise URL
```

### Authentication

The client uses Bearer token authentication extracted from request headers:

```bash
curl -X POST http://localhost:9010/chat/completions \
  -H "Authorization: Bearer your-mindsdb-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "minds", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Error Handling

The client includes comprehensive error handling:

- **Invalid API Key**: Validates API key format and presence
- **Connection Errors**: Handles network and authentication failures
- **Service Errors**: Graceful handling of MindsDB service errors

## Project Structure

```
minds/
├── minds/
│   ├── client/           # Client implementations (OpenAI and MindsDB)
│   ├── common/           # Shared utilities (logging, variables)
│   ├── db/              # Database session management
│   ├── handlers/         # Request handlers
│   ├── model/           # SQLModel database models
│   ├── requests/         # Request/response schemas and streaming
│   └── server.py         # FastAPI application
├── alembic/             # Database migration scripts
│   └── versions/        # Migration version files
├── scripts/             # Utility scripts (database initialization)
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── requirements/        # Python dependencies
├── deployment/          # Kubernetes deployment configs
├── docker-compose.yml   # Docker Compose configuration
├── alembic.ini         # Alembic migration configuration
├── example_usage.py    # Database usage examples
└── Makefile            # Build and development commands
```

## Development

The service is built with:
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous programming support
- **SQLModel**: Modern SQL database toolkit (built on SQLAlchemy)
- **PostgreSQL**: Primary database with UUID support
- **Alembic**: Database migration management
- **Langfuse**: Observability and tracing
- **OpenAI Client**: Integration with OpenAI-compatible APIs
- **MindsDB SDK**: Integration with MindsDB for AI model management and data operations
- **Docker Compose**: Local development environment with all services


## Logging and Observability

The service includes comprehensive logging with:
- Structured logging with multiple output formats
- Performance monitoring and timing
- Request/response logging
- Error tracking and stack traces
- Optional file-based logging with rotation
- Langfuse integration for distributed tracing

## Feature Flags

The service uses LaunchDarkly for feature flag management, enabling dynamic feature control without code deployments. Feature flags allow you to enable or disable features for specific users, environments, or contexts in real-time.

### LaunchDarkly Integration

The application integrates with LaunchDarkly to manage feature flags across different environments. Feature flags are configured through the [LaunchDarkly dashboard](https://app.launchdarkly.com/projects/default/flags/disable-langfuse/targeting) and evaluated at runtime based on user context.

### Configuration

Configure LaunchDarkly in your `.env` file:

```env
# LaunchDarkly Configuration
LAUNCHDARKLY__SDK_KEY=your-sdk-key-here
LAUNCHDARKLY__OFFLINE_MODE=false
```

**Configuration Options:**

- `LAUNCHDARKLY__SDK_KEY`: Your LaunchDarkly SDK key for the environment
- `LAUNCHDARKLY__OFFLINE_MODE`: Set to `true` for local development without LaunchDarkly connectivity (uses default values)

### Available Feature Flags

#### disable-langfuse

Controls whether Langfuse observability tracing is enabled for specific users or environments.

**Configuration:**

```env
# Feature Flag Configuration
FEATURE_FLAG_DISABLE_LANGFUSE__NAME=disable-langfuse
FEATURE_FLAG_DISABLE_LANGFUSE__DEFAULT_VALUE=false
```

**Usage:**

The flag is evaluated on each chat completion request based on the user's email:

```python
from minds.common.launch_darkly.disable_langfuse import is_langfuse_disabled

# Check if Langfuse should be disabled for this user
langfuse_disabled = is_langfuse_disabled(context=context)

if langfuse_disabled:
    # Use non-instrumented handler
    handler = chat_completions_request_handler.__wrapped__
else:
    # Use instrumented handler with Langfuse tracing
    handler = chat_completions_request_handler
```

**Management:**

Feature flags are managed through the LaunchDarkly dashboard, where you can:
- Enable/disable flags per environment (dev, staging, production, etc.)
- Target specific users by email
- Create percentage rollouts
- Set up rules based on custom attributes
- Monitor flag usage and changes

Access the LaunchDarkly dashboard: [https://app.launchdarkly.com/projects/default/flags/disable-langfuse/targeting](https://app.launchdarkly.com/projects/default/flags/disable-langfuse/targeting)

### Adding New Feature Flags

To add a new feature flag:

1. **Create the flag in LaunchDarkly dashboard** with appropriate targeting rules

2. **Add configuration to `app_settings.py`:**

```python
class AppSettings(Settings):
    feature_flag_your_feature: FeatureFlagSettings = Field(
        default=FeatureFlagSettings(name="your-feature-name", default_value=False)
    )
```

3. **Create a helper function** in `minds/common/launch_darkly/`:

```python
from ldclient.context import Context as LDContext

from minds.common.launch_darkly import get_client
from minds.requests.context import Context

def is_your_feature_enabled(context: Context) -> bool:
    """Check if your feature is enabled."""
    settings = get_app_settings()
    
    ld_context = (
        LDContext.builder(str(context.user_email))
        .kind("user")
        .name(context.user_email)
        .set("email", context.user_email)
        .build()
    )
    
    return get_client().variation(
        settings.feature_flag_your_feature.name,
        ld_context,
        settings.feature_flag_your_feature.default_value
    )
```

4. **Use the flag** in your code to conditionally enable/disable features

### Local Development

For local development without LaunchDarkly connectivity:

```env
LAUNCHDARKLY__OFFLINE_MODE=true
```

When offline mode is enabled, all feature flags will use their configured default values.

## Resource Usage Limits

The service enforces per-user resource consumption limits to prevent abuse and support billing. Limits are fetched dynamically from [Statsig](https://statsig.com/) via the `mind-usage-limits` dynamic config and are evaluated on every resource-creating request.

### How It Works

1. **Limit retrieval** – `LimitsService.get_mind_limits()` reads the current limit thresholds from the Statsig dynamic config (`mind-usage-limits`) and computes usage counts from the database. In self-hosted mode, all limits default to **unlimited**.

2. **Guard system** – A centralised guard module (`minds/common/guards/usage.py`) provides the `require_usage_available` function. Endpoints call this guard before performing any resource-creating operation:

```python
from minds.common.guards import require_usage_available, ResourceType

await require_usage_available(limits_service, ResourceType.MINDS)
```

3. **HTTP 429 rejection** – When usage meets or exceeds the configured limit (monthly *or* lifetime), the guard raises `UsageLimitExceededError`, returning an **HTTP 429 Too Many Requests** response with a descriptive message.

### Protected Endpoints

| Endpoint | Resource Type | Guard checks |
|---|---|---|
| `POST /chat/completions` | `QUESTIONS` | Monthly & lifetime question limits |
| `POST /responses` | `QUESTIONS` | Monthly & lifetime question limits |
| `POST /minds` (create) | `MINDS` | Monthly & lifetime mind limits |
| `POST /datasources` (create) | `DATASOURCES` | Monthly & lifetime datasource limits |

### ResourceType Enum

The `ResourceType` enum identifies the four tracked resource categories:

- `MINDS` – number of minds created
- `DATASOURCES` – number of datasources created
- `TOKENS` – total tokens consumed
- `QUESTIONS` – number of questions asked

### Usage Tracking: Lifetime vs. Billing Cycle

Each resource tracks two usage counters via the `UsageConfig` schema:

- **`lifetime`** – total count across all time
- **`billing_cycle`** – count since the start of the current billing period

The billing period start is communicated by the upstream gateway via the `X-Billing-Period-Start` request header in **ISO-8601** format (e.g. `2025-07-01T00:00:00Z`). When this header is present, the service filters resource counts to only include records created on or after that timestamp. When absent, billing-cycle usage defaults to the lifetime count.

### Configuration

Limits are configured per user/organisation in the Statsig dashboard under the `mind-usage-limits` dynamic config. Each resource section has the shape:

```json
{
  "minds": {
    "limit": { "lifetime": 10, "monthly": 5 }
  },
  "datasources": {
    "limit": { "lifetime": 30, "monthly": 10 }
  },
  "tokens": {
    "limit": { "lifetime": -1, "monthly": 1000000 }
  },
  "questions": {
    "limit": { "lifetime": -1, "monthly": 250 }
  }
}
```

A limit value of `-1` means **unlimited** (the resource is uncapped).

### Extending with New Guard Types

The `minds/common/guards/` package is designed to be generic. To add a new guard category (e.g. permissions, feature access):

1. Create a new module under `minds/common/guards/` (e.g. `permissions.py`).
2. Implement a guard function following the same pattern as `require_usage_available`.
3. Export it from `minds/common/guards/__init__.py`.
4. Inject the guard as a dependency in the relevant endpoints.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## License

[Add your license information here]
