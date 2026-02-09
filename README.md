# AgentWatch

Lightweight AI agent observability platform. Track token usage, latency, error rates, and cost per task across your AI agents.

## Quick Start

```bash
pip install -e .
agentwatch serve
```

Open http://localhost:8787 to view the dashboard.

### Populate with demo data

```bash
# Backfill 7 days of simulated data
python demo.py --backfill

# Or stream live simulated events
python demo.py --live
```

## SDK Usage

### Zero-code instrumentation (auto-patching)

```python
import agentwatch
agentwatch.init(server_url="http://localhost:8787", agent="my-agent")

# All OpenAI and Anthropic API calls are now automatically tracked
import openai
client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

### Decorator

```python
from agentwatch import track

@track(task="summarize", agent="research-bot")
def summarize(text):
    return client.chat.completions.create(model="gpt-4o", messages=[...])
```

### Manual

```python
import agentwatch
agentwatch.log_event(
    task_name="classify",
    agent_name="support-bot",
    model="gpt-4o-mini",
    input_tokens=500,
    output_tokens=50,
    latency_ms=340.0,
    status="success",
)
```

## Architecture

```
Agent code --> SDK auto-capture --> POST /api/events --> SQLite --> Dashboard (SSE + REST)
```

- **SDK**: Python client with auto-patching, decorator, and manual APIs
- **Server**: FastAPI + SQLite + SSE for real-time updates
- **Dashboard**: Single-page HTML/CSS/JS with Chart.js (no build step)

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/events` | Ingest event batch |
| `GET /api/events` | Query events with filters |
| `GET /api/metrics/summary` | Aggregated metrics (cost, latency, success rate) |
| `GET /api/metrics/timeseries` | Time-bucketed data for charts |
| `GET /api/metrics/breakdown` | Breakdown by agent, model, or task |
| `GET /api/stream` | SSE stream for live events |
