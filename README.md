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
DATABASE_URI=postgresql://minds:minds@localhost:35432/minds
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=300
DB_POOL_RECYCLE=300
DB_POOL_PRE_PING=true
DB_QUERY_TIMEOUT=300
DB_STATEMENT_TIMEOUT=300000

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
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL_NAME=gpt-4o
OPENAI_MAX_TOKENS=400000

# MindsDB Configuration
MINDSDB_URL=https://cloud.mindsdb.com

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

2. **Set the DATABASE_URI environment variable:**
   ```bash
   export DATABASE_URI="postgresql://minds:minds@localhost:5432/minds"
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## License

[Add your license information here]
