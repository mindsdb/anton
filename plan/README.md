# Anton Web Frontend — Plan

## Goal

A web app that makes Anton accessible to non-technical users. The user opens a browser, picks a task or types a request, and Anton does the work. The scratchpad mechanics, tool calls, and memory system stay invisible — the user just sees a conversation and results.

## Target Users

- Business analysts who need data crunched or visualized
- Product managers who want quick reports or competitive analysis
- Researchers who need web scraping, API integrations, or data processing
- Anyone who would benefit from an AI assistant that *does* things, not just writes code

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Frontend (React)                     │
│                                                      │
│  Chat UI  │  Output Viewer  │  Templates  │  Sessions│
└────────┬─────────────────────────────────────────────┘
         │  WebSocket (streaming) + REST
         ▼
┌──────────────────────────────────────────────────────┐
│                API Server (FastAPI)                   │
│                                                      │
│  /api/chat       — WebSocket: stream turn events     │
│  /api/sessions   — REST: list, resume                │
│  /api/memory     — REST: read-only summary           │
│  /api/outputs    — REST: serve generated HTML/files  │
│  /api/files      — REST: upload files for analysis   │
│  /api/shares     — REST: create, get, list shares    │
│  /api/settings   — REST: get/set config              │
│                                                      │
│  Wraps ChatSession.turn_stream() over WebSocket      │
└────────┬─────────────────────────────────────────────┘
         │  Direct Python import
         ▼
┌──────────────────────────────────────────────────────┐
│             Anton Core (existing package)             │
│                                                      │
│  ChatSession  │  ScratchpadManager  │  Cortex  │ ... │
└──────────────────────────────────────────────────────┘
```

The API server is a thin FastAPI wrapper. It does NOT reimplement any agent logic — it imports `ChatSession`, `AntonSettings`, `Cortex`, etc. directly from the `anton` package.

## Features

### 1. Chat Interface (core)

The main screen. A clean conversation thread where the user types what they need and Anton responds with streaming text.

- Real-time text streaming (typewriter effect from `StreamTextDelta` events)
- File upload via drag-and-drop or button (images, CSVs, spreadsheets)
- Inline rendering of Anton's outputs (HTML dashboards, charts, tables) directly in the conversation
- Download button for any generated files
- "Open in new tab" for full-screen dashboards
- A subtle working indicator when Anton is executing scratchpad code (spinner + one-line status like "Fetching stock prices..." from `StreamTaskProgress`). No code is shown — just the status message.

### 2. Task Templates

The landing experience for non-technical users. Instead of a blank prompt, the user picks from a gallery of common tasks, each with a guided form.

| Template | Inputs | What Anton Does |
|---|---|---|
| **Analyze a spreadsheet** | File upload | Summary stats, charts, insights |
| **Track stock portfolio** | Ticker symbols, quantities | Live prices, performance dashboard |
| **Research a topic** | Topic description, optional URLs | Web research + structured summary |
| **Compare products** | Product names or URLs | Side-by-side comparison table |
| **Generate a report** | Data source description, format | Structured report with visuals |
| **Scrape a website** | URL, what to extract | Structured data extraction |
| **Retrieve Sales Data** | Credentials for connection | Connect to the data source and retrieve data from questions |

Each template is a form that collects inputs and constructs a well-formed prompt for Anton. The form submission drops the user into the chat interface with the prompt pre-filled and sent.

Templates are defined as simple JSON/config — easy to add new ones without code changes.

### 3. Session Sidebar

A collapsible sidebar listing past conversations.

- Session list with auto-generated titles and dates
- Click to resume any previous session
- Delete sessions
- That's it — no search, no export, keep it simple

### 4. Memory (light)

A small settings-adjacent panel where the user can see what Anton has learned. Read-only.

- List of rules Anton follows (always/never/when)
- List of lessons Anton has learned
- User profile (what Anton knows about the user)
- Memory mode toggle (autopilot/copilot/off)
- No editing, no episodic timeline — just a summary view

### 5. Sharing (basic)

Share outputs and conversations via public links (anyone with the link can view).

**MVP scope:**
- "Copy link" button on individual outputs (HTML dashboards, charts) and on full conversations
- Links are public, token-based URLs (`/s/{token}`) — no auth required to view
- Shared outputs are snapshots: the output file is copied to durable storage at share time so it survives scratchpad cleanup
- Shared conversations are read-only rendered views of the message thread at time of sharing (not live)
- Simple JSON-file–based share index on the server (token → metadata + path). No database yet.
- No expiration, no revocation in the MVP — links live forever

**Post-MVP roadmap:**
- Authentication system, so sharing can be restricted to logged-in users or specific people
- Granular permissions (view-only, view + download)
- Link expiration and revocation UI
- Live shared sessions that update as the conversation continues
- SQLite or Postgres backing store to replace the JSON share index
- "Manage shares" panel listing all active share links

### 6. Settings

Minimal configuration screen.

- API key entry (Anthropic or OpenAI)
- Model selection (planning + coding)

## WebSocket Protocol

Client → Server:
```json
{
  "type": "message",
  "content": "Analyze this CSV and make a dashboard",
  "files": ["upload_id_123"]
}
```

Server → Client (one JSON per streaming event):
```json
{"type": "text_delta", "text": "Let me analyze that..."}
{"type": "status", "message": "Parsing CSV data..."}
{"type": "text_delta", "text": "Here's what I found..."}
{"type": "output", "kind": "html", "url": "/api/outputs/abc123.html"}
{"type": "output", "kind": "image", "url": "/api/outputs/chart.png"}
{"type": "complete", "usage": {"input_tokens": 5000, "output_tokens": 2000}}
```

The server translates internal `StreamEvent` types into this simplified protocol. Tool-level details (tool_start, tool_end, tool_delta) are collapsed into user-friendly `status` messages. The frontend never sees tool internals.

## Tech Stack

**API Server:**
- Python 3.11+
- FastAPI + uvicorn
- WebSockets (built into FastAPI)
- Direct import of `anton` package

**Frontend:**
- React 19 + TypeScript
- Tailwind CSS
- shadcn/ui components
- react-markdown + remark-gfm for message rendering
- WebSocket client for streaming

## Implementation Phases

### Phase 1 — Chat MVP

Get a working chat loop in the browser.

- FastAPI server with WebSocket endpoint wrapping `ChatSession.turn_stream()`
- Session manager (create, store, resume sessions)
- React chat UI: message thread, input area, text streaming
- Working indicator during tool execution
- File upload (store server-side, pass to Anton as multimodal input)
- Basic output rendering (HTML in iframe, images inline)

### Phase 2 — Templates + Sessions + Share

Make it useful for non-technical users.

- Task template gallery (landing page)
- Template forms that construct prompts
- Session sidebar with history and resume
- Download buttons for generated files
- "Open in new tab" for dashboards
- Basic sharing: `POST /api/shares` creates a token, copies output to `shares/` directory, stores metadata in a JSON index. `GET /s/{token}` serves a read-only view.
- Share button on outputs and conversations in the chat UI

### Phase 3 — Polish

- Light memory view (read-only)
- Settings screen (API key, model, theme)
- Mobile-responsive layout
- Error states and edge cases

### Phase 4 — Auth + Advanced Sharing

- User authentication (session-based or OAuth)
- Restricted share links (require login, specific users)
- Share expiration and revocation
- "Manage shares" UI
- Migrate share index and sessions to SQLite

## Integration Notes

**HTML output detection:** Anton creates HTML files via scratchpad and opens them with `webbrowser.open()`. The API server should watch for file creation in the scratchpad working directory and serve those files via `/api/outputs` instead of trying to open a browser.

**Multimodal input:** `turn_stream()` accepts `str` or `list[dict]` for multimodal content. The API server handles file uploads, stores them, and constructs the appropriate content blocks.

**Context overflow:** Handled internally by `ChatSession`. The frontend receives a `StreamContextCompacted` event — show a brief toast notification.

**Memory confirmations:** In `copilot` mode, some memories need user confirmation (stored in `_pending_memory_confirmations`). For the MVP, we run in `autopilot` mode and skip this. Can add later as a simple accept/dismiss dialog.

**Sharing & output durability:** Scratchpad venvs and their output files can be reset or removed. When creating a share link, the server copies the referenced output file(s) into a persistent `shares/` directory alongside a JSON metadata file. This decouples shared content from scratchpad lifecycle. Conversation shares serialize the message thread as JSON at share time (snapshot, not live). The `SharedView` frontend component renders these without needing a WebSocket or active session.

## File Structure

```
anton-web/
├── api/
│   ├── __init__.py
│   ├── main.py                FastAPI app, CORS, static file serving
│   ├── routes/
│   │   ├── chat.py            WebSocket endpoint
│   │   ├── sessions.py        Session list + resume
│   │   ├── memory.py          Read-only memory summary
│   │   ├── outputs.py         Serve generated files
│   │   ├── files.py           File upload
│   │   ├── shares.py          Create + serve share links
│   │   └── settings.py        Config get/set
│   ├── session_manager.py     Pool of active ChatSessions
│   ├── share_store.py         JSON-file share index + snapshot logic
│   └── ws_protocol.py         StreamEvent → JSON translation
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Chat.tsx           Main chat view
│   │   │   ├── MessageThread.tsx  Message list with streaming
│   │   │   ├── InputArea.tsx      Text input + file upload
│   │   │   ├── OutputEmbed.tsx    Inline HTML/image rendering
│   │   │   ├── ShareButton.tsx    "Copy link" share action
│   │   │   ├── SharedView.tsx     Public read-only view for /s/{token}
│   │   │   ├── SessionSidebar.tsx Past sessions
│   │   │   ├── TemplateGallery.tsx Template cards
│   │   │   ├── TemplateForm.tsx   Guided input form
│   │   │   ├── MemoryView.tsx     Read-only memory summary
│   │   │   └── Settings.tsx       Config screen
│   │   ├── hooks/
│   │   │   ├── useAnton.ts        WebSocket + state management
│   │   │   └── useSession.ts      Session CRUD
│   │   └── lib/
│   │       └── types.ts           Message and event types
│   ├── package.json
│   └── vite.config.ts
├── Dockerfile
└── docker-compose.yml
```
