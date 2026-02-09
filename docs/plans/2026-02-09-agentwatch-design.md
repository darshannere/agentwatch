# AgentWatch — AI Agent Observability Platform

## Architecture

Three components:
1. **Python SDK** (`agentwatch`) — auto-patches openai/anthropic clients, captures tokens, latency, cost, errors
2. **FastAPI Backend** — REST API + SSE, SQLite storage, serves dashboard
3. **Single-page Dashboard** — vanilla HTML/CSS/JS, Chart.js, dark theme

Data flow: Agent → SDK auto-capture → POST /api/events → SQLite → Dashboard (SSE + REST)

## SDK

Three tiers:
- `agentwatch.init(server_url)` — monkey-patches openai/anthropic, zero code changes
- `@track(task, agent)` decorator — labels specific functions
- `agentwatch.log_event(...)` — manual control

Captured per event: task_name, agent_name, model, input_tokens, output_tokens, latency_ms, cost_usd, status, error_message, timestamp.

Built-in pricing table for common models. Batches events, flushes every 5s or on exit.

## API Endpoints

- `POST /api/events` — ingest events (batch)
- `GET /api/events` — query events (with filters)
- `GET /api/metrics/summary` — aggregated metrics for dashboard cards
- `GET /api/metrics/timeseries` — time-bucketed data for charts
- `GET /api/metrics/breakdown` — cost/errors by agent/model/task
- `GET /api/stream` — SSE for live event feed
- `GET /` — serves dashboard HTML

## Dashboard Layout

- Top: 4 metric cards (total cost, requests + success rate, avg latency p50/p95, active agents)
- Middle left: Cost & tokens over time (line chart)
- Middle right: Latency distribution (histogram + trend)
- Bottom left: Live event feed (SSE)
- Bottom right: Breakdown tables (by agent, model, task)
- Time range selector: 1h / 24h / 7d / 30d

## Tech

- Python 3.10+, FastAPI, uvicorn, SQLite (aiosqlite)
- No frontend build step
- Single `pip install agentwatch` for everything
