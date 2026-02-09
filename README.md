# AgentWatch

**Production-grade observability platform for AI agents.** Track token usage, latency, error rates, and cost per task across your AI agents in real-time.

AgentWatch is a lightweight, self-hosted monitoring tool that gives you complete visibility into your AI agent workflows—without vendor lock-in or external dependencies.

## Features

### 📊 Real-Time Monitoring
- **Live event stream** via Server-Sent Events (SSE)
- **Interactive dashboards** with Chart.js visualizations
- **Cost tracking** with per-model pricing support
- **Latency insights** (p50, p95, averages)
- **Error rate monitoring** per agent, model, and task

### 🚀 Zero-Code Integration
- **Auto-patching** for OpenAI and Anthropic clients (no code changes needed)
- **Decorator-based** instrumentation for selective tracking
- **Manual logging** API for full control
- **Batch event sending** with background thread (no blocking)

### 💾 Self-Hosted & Minimal
- **SQLite database** (no Postgres setup required)
- **Single FastAPI server** (~364 lines)
- **Vanilla HTML/CSS/JS dashboard** (no build step, no Node.js)
- **Pip installable** with minimal dependencies

### 📈 Production-Ready
- **Async/await throughout** (async SQLite, FastAPI)
- **Parameterized SQL queries** (no injection vulnerabilities)
- **Graceful error handling** (tracking failures never break your app)
- **Reconnection logic** for SSE streams

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/agentwatch.git
cd agentwatch
pip install -e .
```

### Run the Server

```bash
agentwatch serve
```

Server starts on `http://localhost:8787` with the dashboard at `/`.

### Populate Demo Data

In another terminal:

```bash
# Backfill 7 days of realistic simulated data
python demo.py --backfill

# Or stream live simulated events (Ctrl+C to stop)
python demo.py --live
```

### Use the SDK

```python
import agentwatch

# Initialize (zero-code auto-patching)
agentwatch.init(
    server_url="http://localhost:8787",
    agent="research-bot",
    auto_patch=True  # Auto-instrument OpenAI/Anthropic
)

# All API calls are now tracked automatically
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
# ✅ Token usage, latency, cost automatically recorded
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Your AI Agent Code                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
                   ┌──────────────────┐
                   │  AgentWatch SDK  │
                   │ (auto-patching   │
                   │  decorator,      │
                   │  manual log)     │
                   └──────────────────┘
                              ↓
                   ┌──────────────────┐
                   │  Background      │
                   │  Event Sender    │
                   │  (batching,      │
                   │   threading)     │
                   └──────────────────┘
                              ↓
                ┌─────────────────────────┐
                │  FastAPI Server (8787)  │
                │  ├─ REST API endpoints  │
                │  ├─ SSE streaming       │
                │  └─ Static dashboard    │
                └─────────────────────────┘
                              ↓
                ┌─────────────────────────┐
                │   SQLite Database       │
                │  (events, metrics)      │
                └─────────────────────────┘
                              ↓
                ┌─────────────────────────┐
                │  Dashboard (Browser)    │
                │  ├─ Metric cards        │
                │  ├─ Charts (cost, lat)  │
                │  ├─ Live event feed     │
                │  └─ Breakdown tables    │
                └─────────────────────────┘
```

### Components

| Component | Purpose | Tech |
|---|---|---|
| **SDK** (`agentwatch/sdk.py`) | Client instrumentation library | Python threading, httpx |
| **Server** (`agentwatch/server.py`) | REST API + SSE, async event handling | FastAPI, aiosqlite |
| **Database** (`agentwatch/db.py`) | Event storage & metrics aggregation | SQLite, async queries |
| **Dashboard** (`agentwatch/static/index.html`) | Real-time monitoring UI | Vanilla JS, Chart.js, SSE |

---

## SDK Usage

### 1. Auto-Patching (Zero Code Changes)

**The easiest way.** Initialize once, and all OpenAI/Anthropic API calls are tracked automatically:

```python
import agentwatch

agentwatch.init(
    server_url="http://localhost:8787",
    agent="my-agent-name",
    task="default-task",
    auto_patch=True  # <-- enabled by default
)

# Now use OpenAI or Anthropic normally
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
# ✅ Automatically recorded:
#    - model: "gpt-4o"
#    - input_tokens: X
#    - output_tokens: Y
#    - latency_ms: Z
#    - cost_usd: calculated
#    - status: "success" or "error"
```

**Supported libraries:**
- `openai` (sync and async)
- `anthropic` (sync and async)

### 2. Decorator (Selective Instrumentation)

Label specific functions as "tasks" for fine-grained tracking:

```python
from agentwatch import track

@track(task="summarize-documents", agent="research-bot")
def summarize(text):
    # This function is now tracked
    response = client.chat.completions.create(...)
    return response.choices[0].message.content

result = summarize("Long document...")
# ✅ Recorded as a task event with the function's execution time
```

**Works with async functions too:**

```python
@track(task="process-batch", agent="pipeline")
async def process_batch(items):
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

### 3. Manual Logging (Full Control)

For non-API-call events or custom metrics:

```python
import agentwatch

agentwatch.log_event(
    task_name="custom-analysis",
    agent_name="analytics-bot",
    model="gpt-4o-mini",
    input_tokens=500,
    output_tokens=100,
    latency_ms=1200.0,
    status="success",
    error_message=None,
    metadata={"batch_size": 10, "user_id": "123"}
)
```

### 4. Class-Based API

Prefer instances over module-level state?

```python
from agentwatch import AgentWatch

watch = AgentWatch(
    server_url="http://localhost:8787",
    agent="my-agent",
    task="my-task"
)

watch.log_event(
    task_name="query",
    model="gpt-4o",
    input_tokens=100,
    output_tokens=50,
    latency_ms=500.0,
    status="success"
)

# Or use the decorator
@watch.track(task="analyze")
def analyze(data):
    return do_something(data)
```

---

## API Endpoints

All endpoints return JSON and support CORS.

### POST /api/events

**Ingest a batch of events.**

Request:
```json
{
  "events": [
    {
      "id": "uuid",
      "task_name": "summarize",
      "agent_name": "research-bot",
      "model": "gpt-4o",
      "input_tokens": 1000,
      "output_tokens": 200,
      "latency_ms": 2340.5,
      "cost_usd": 0.0085,
      "status": "success",
      "error_message": null,
      "timestamp": "2026-02-09T15:30:00+00:00",
      "metadata": {"source": "batch_1"}
    }
  ]
}
```

Response:
```json
{
  "status": "ok",
  "count": 1
}
```

### GET /api/events

**Query events with optional filters.**

Query parameters:
- `agent_name` (string, optional)
- `task_name` (string, optional)
- `model` (string, optional)
- `status` (string: "success" or "error", optional)
- `since` (ISO datetime, optional)
- `until` (ISO datetime, optional)
- `limit` (integer, default 100)
- `offset` (integer, default 0)

Example: `GET /api/events?agent_name=research-bot&status=error&limit=50`

Response:
```json
{
  "events": [
    {
      "id": "uuid",
      "task_name": "...",
      "agent_name": "...",
      ...
    }
  ]
}
```

### GET /api/metrics/summary

**Get aggregated metrics for a time period.**

Query parameters:
- `period` (string: "1h", "24h", "7d", "30d", "all", default "24h")

Example: `GET /api/metrics/summary?period=7d`

Response:
```json
{
  "total_cost": 145.32,
  "total_requests": 523,
  "success_rate": 96.5,
  "avg_latency_ms": 1250.5,
  "latency_p50": 820.0,
  "latency_p95": 3100.0,
  "active_agents": ["research-bot", "support-agent"],
  "period": "7d"
}
```

### GET /api/metrics/timeseries

**Get time-bucketed metrics for charting.**

Query parameters:
- `period` (string: "1h", "24h", "7d", "30d", default "24h")
- `bucket` (integer minutes, optional; auto-calculated if not provided)

Example: `GET /api/metrics/timeseries?period=24h`

Response:
```json
{
  "timestamps": [
    "2026-02-08T00:00:00",
    "2026-02-08T01:00:00",
    ...
  ],
  "cost": [0.5, 1.2, 0.8, ...],
  "tokens": [1000, 2500, 1800, ...],
  "requests": [10, 25, 18, ...],
  "avg_latency": [500, 750, 600, ...],
  "latency_p50": [400, 600, 480, ...],
  "latency_p95": [900, 1400, 1080, ...],
  "errors": [0, 1, 0, ...]
}
```

### GET /api/metrics/breakdown

**Get metrics grouped by agent, model, or task.**

Query parameters:
- `group_by` (string: "agent_name", "model", "task_name", default "agent_name")
- `period` (string: "1h", "24h", "7d", "30d", default "24h")

Example: `GET /api/metrics/breakdown?group_by=model&period=7d`

Response:
```json
[
  {
    "name": "gpt-4o",
    "requests": 1250,
    "avg_latency": 1500.0,
    "total_cost": 85.50,
    "error_rate": 2.4
  },
  {
    "name": "claude-sonnet-4-20250514",
    "requests": 950,
    "avg_latency": 1200.0,
    "total_cost": 15.30,
    "error_rate": 1.2
  }
]
```

### GET /api/stream

**Server-Sent Events (SSE) stream of live events.**

Connects to a persistent SSE stream. Client receives events as they're ingested:

```
data: {"id":"uuid","task_name":"...","agent_name":"...","status":"success",...}

data: {"id":"uuid2","task_name":"...","agent_name":"...","status":"error",...}
```

Connection drops? The client automatically reconnects after 3 seconds (implemented in dashboard JS).

---

## Dashboard

Navigate to `http://localhost:8787` to open the dashboard.

### Layout

**Header**
- AgentWatch logo with animated pulse indicator
- Time range buttons: 1h, 24h, 7d, 30d (click to change)
- "Updated X ago" timestamp
- Connection status (Connected / Reconnecting)

**Metric Cards (4 columns)**
1. **Total Cost** — Dollar amount for the selected period, mini sparkline trend
2. **Requests** — Request count with success rate badge (green/amber/red)
3. **Latency** — p50 latency prominent, p95 below it
4. **Active Agents** — Count with agent name tags

**Charts (2 columns)**
- **Left: Cost & Tokens Over Time** — Dual Y-axis (cost in $, tokens count), cost as filled gradient line, tokens as dashed purple line
- **Right: Latency Over Time** — p50 and p95 with shaded band between them

**Bottom Panels (2 columns)**
- **Left: Live Event Feed** — Real-time SSE stream, newest events at top. Shows: timestamp, agent (color-coded badge), task, model, tokens in/out, cost, latency, status dot
- **Right: Breakdown Tables** — Three tabs (By Agent, By Model, By Task). Sortable columns. Error rate shown as colored bar (green < 5%, amber 5-15%, red > 15%)

### Features

- **Dark theme** — Easy on the eyes for long monitoring sessions
- **Real-time updates** — Live event feed via SSE, metrics refresh every 30 seconds
- **Responsive** — Works on desktop and tablet
- **No external CSS framework** — Pure CSS, minimal bundle size (~48 KB HTML)
- **Chart.js visualizations** — Smooth animations, hover tooltips

---

## Configuration

### Server

Environment variables:

```bash
# Set custom database path
export AGENTWATCH_DB="/path/to/agentwatch.db"

# Run server
agentwatch serve --host 0.0.0.0 --port 8787
```

Command-line arguments:

```
agentwatch serve --help

usage: agentwatch serve [-h] [--host HOST] [--port PORT] [--db DB]

options:
  --host HOST   Server host (default: 0.0.0.0)
  --port PORT   Server port (default: 8787)
  --db DB       Database path (default: agentwatch.db)
```

### SDK

Initialization parameters:

```python
agentwatch.init(
    server_url="http://localhost:8787",  # AgentWatch server URL
    agent="my-agent-name",                # Default agent name for events
    task="default-task",                  # Default task name for events
    auto_patch=True                       # Auto-patch OpenAI/Anthropic
)
```

---

## Demo Script

Generate realistic simulated data for testing and demos.

### Backfill Mode

Generate 7 days of historical data (~3,500 events):

```bash
python demo.py --backfill
```

Features:
- Reproducible (`random.Random(42)`)
- Realistic daily patterns (10% overnight, peak 09:00-17:00)
- 5 agent profiles with diverse tasks and models
- Progress bar during upload
- ~500 events per day

### Live Mode

Stream simulated events in real-time:

```bash
python demo.py --live
```

Features:
- 1-3 events every 2-5 seconds
- Colored terminal output
- Simulated latency and errors
- Ctrl+C for session summary

### Options

```bash
python demo.py --help

usage: demo.py [--live | --backfill] [--server URL] [--speed MULTIPLIER]

options:
  --live, --backfill   Operation mode (default: --live)
  --server URL         AgentWatch server (default: http://localhost:8787)
  --speed MULTIPLIER   Speed multiplier for live mode (default: 1.0)
```

---

## Pricing & Cost Tracking

AgentWatch includes built-in pricing for popular models:

| Model | Input Price | Output Price |
|---|---|---|
| gpt-4o | $2.50/M | $10.00/M |
| gpt-4o-mini | $0.15/M | $0.60/M |
| gpt-3.5-turbo | $0.50/M | $1.50/M |
| claude-opus-4-20250514 | $15.00/M | $75.00/M |
| claude-sonnet-4-20250514 | $3.00/M | $15.00/M |
| claude-haiku-3-5-20241022 | $0.80/M | $4.00/M |

**Unsupported model?** Falls back to `$0.003/1k input` and `$0.015/1k output`.

---

## Performance

- **Event ingestion**: ~5,000 events/second (batch mode)
- **Query latency**: <50ms for typical queries (last 24h)
- **Dashboard refresh**: ~500ms (fetches summary + timeseries + breakdown in parallel)
- **Storage**: ~50 KB per 1,000 events
- **Memory**: ~50 MB baseline (SQLite in memory, event queue)

---

## Database Schema

Single `events` table:

```sql
CREATE TABLE events (
  id TEXT PRIMARY KEY,
  task_name TEXT NOT NULL,
  agent_name TEXT NOT NULL,
  model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  latency_ms REAL NOT NULL,
  cost_usd REAL NOT NULL,
  status TEXT NOT NULL,
  error_message TEXT,
  timestamp TEXT NOT NULL,
  metadata TEXT
);

CREATE INDEX idx_events_timestamp ON events (timestamp);
CREATE INDEX idx_events_agent_name ON events (agent_name);
CREATE INDEX idx_events_task_name ON events (task_name);
CREATE INDEX idx_events_status ON events (status);
```

---

## Development

### Project Structure

```
agentwatch/
├── agentwatch/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Dataclasses, pricing
│   ├── db.py                    # SQLite async layer
│   ├── sdk.py                   # Client SDK (auto-patch, decorator, manual)
│   ├── server.py                # FastAPI server
│   ├── cli.py                   # CLI entry point
│   └── static/
│       └── index.html           # Dashboard (1,627 lines)
├── tests/
│   └── __init__.py
├── demo.py                      # Demo data generator
├── pyproject.toml               # Package metadata
├── README.md                    # This file
└── docs/
    └── plans/
        └── 2026-02-09-agentwatch-design.md
```

### Running Tests

```bash
pytest tests/
```

### Running Locally

```bash
# Terminal 1: Start server
agentwatch serve

# Terminal 2: Backfill demo data
python demo.py --backfill

# Terminal 3: Stream live events (optional)
python demo.py --live --speed 2

# Browser: Open http://localhost:8787
```

---

## Troubleshooting

### Server won't start on port 8787

Port already in use. Either:

```bash
# Use a different port
agentwatch serve --port 8888

# Or kill the process on 8787
lsof -ti:8787 | xargs kill -9
```

### Dashboard shows no data

1. Ensure server is running: `curl http://localhost:8787/api/metrics/summary`
2. Load demo data: `python demo.py --backfill`
3. Check browser console for errors (F12)
4. Verify SSE connection: open DevTools → Network → look for `/api/stream`

### Events not being tracked

Check SDK initialization:

```python
import agentwatch
agentwatch.init()  # Must call init() once at startup

# Verify auto-patching is enabled
import openai
client = openai.OpenAI()
# Make an API call
```

Missing `agentwatch.init()`? Events are silently dropped (with a warning logged).

### High memory usage

SQLite keeps recent queries in memory. For long-running servers, consider:

```python
# Periodically clean old events (optional)
# await db.delete_events_before(datetime.now() - timedelta(days=30))
```

---

## License

MIT

---

## Contributing

Contributions welcome! Areas for enhancement:

- [ ] Alerting rules (e.g., "cost > $X/day")
- [ ] Export to CSV/JSON
- [ ] Custom cost models
- [ ] PostgreSQL backend option
- [ ] Multi-server aggregation
- [ ] API key authentication

---

## Acknowledgments

AgentWatch tracks production AI concerns that actually matter:
- **Cost control** — Know what each agent costs
- **Reliability** — Monitor error rates and latency
- **Visibility** — Real-time insights without external dependencies

Built for developers who care about observability without vendor lock-in.
