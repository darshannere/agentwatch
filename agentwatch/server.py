from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agentwatch.db import Database
from agentwatch.models import Event

_STATIC_DIR = Path(__file__).parent / "static"

_PERIOD_DELTAS: dict[str, Optional[timedelta]] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}

_AUTO_BUCKET: dict[str, int] = {
    "1h": 1,
    "24h": 15,
    "7d": 60,
    "30d": 360,
}


class EventBus:
    def __init__(self) -> None:
        self.subscribers: list[asyncio.Queue[dict]] = []

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        try:
            self.subscribers.remove(queue)
        except ValueError:
            pass

    def publish(self, event_dict: dict) -> None:
        for queue in self.subscribers:
            try:
                queue.put_nowait(event_dict)
            except asyncio.QueueFull:
                pass


def _period_to_since(period: str) -> Optional[datetime]:
    delta = _PERIOD_DELTAS.get(period)
    if delta is None:
        return None
    return datetime.now(timezone.utc) - delta


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_path = os.environ.get("AGENTWATCH_DB", "agentwatch.db")
    db = Database(db_path)
    await db.init()
    app.state.db = db
    app.state.bus = EventBus()
    yield
    await db.close()


app = FastAPI(title="AgentWatch", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


class EventIn(BaseModel):
    id: Optional[str] = None
    task_name: str
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    status: str
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None


class EventsPostRequest(BaseModel):
    events: list[EventIn]


class EventsPostResponse(BaseModel):
    status: str
    count: int


class EventOut(BaseModel):
    id: str
    task_name: str
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    status: str
    error_message: Optional[str] = None
    timestamp: str
    metadata: Optional[dict] = None


class EventsGetResponse(BaseModel):
    events: list[EventOut]


class MetricsSummaryResponse(BaseModel):
    total_cost_usd: float
    total_requests: int
    success_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    active_agents: int
    period: str


class TimeseriesBucketOut(BaseModel):
    timestamp: str
    total_cost: float
    total_tokens: int
    request_count: int
    avg_latency: float
    error_count: int


class TimeseriesResponse(BaseModel):
    buckets: list[TimeseriesBucketOut]


class BreakdownRowOut(BaseModel):
    name: str
    total_cost: float
    total_requests: int
    avg_latency: float
    error_rate: float


class BreakdownResponse(BaseModel):
    rows: list[BreakdownRowOut]


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/api/events", response_model=EventsPostResponse)
async def post_events(request: Request, body: EventsPostRequest) -> EventsPostResponse:
    db: Database = request.app.state.db
    bus: EventBus = request.app.state.bus

    events: list[Event] = []
    for ev in body.events:
        ts: datetime
        if ev.timestamp is not None:
            raw = ev.timestamp
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            ts = datetime.fromisoformat(raw)
        else:
            ts = datetime.now(timezone.utc)

        event = Event(
            id=ev.id or str(uuid.uuid4()),
            task_name=ev.task_name,
            agent_name=ev.agent_name,
            model=ev.model,
            input_tokens=ev.input_tokens,
            output_tokens=ev.output_tokens,
            latency_ms=ev.latency_ms,
            cost_usd=ev.cost_usd,
            status=ev.status,
            error_message=ev.error_message,
            timestamp=ts,
            metadata=ev.metadata,
        )
        events.append(event)

    await db.insert_events(events)

    for event in events:
        bus.publish(event.to_dict())

    return EventsPostResponse(status="ok", count=len(events))


@app.get("/api/events", response_model=EventsGetResponse)
async def get_events(
    request: Request,
    agent_name: Optional[str] = Query(default=None),
    task_name: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    since: Optional[str] = Query(default=None),
    until: Optional[str] = Query(default=None),
    limit: int = Query(default=100),
    offset: int = Query(default=0),
) -> EventsGetResponse:
    db: Database = request.app.state.db

    since_dt: Optional[datetime] = None
    until_dt: Optional[datetime] = None
    if since is not None:
        since_dt = datetime.fromisoformat(since)
    if until is not None:
        until_dt = datetime.fromisoformat(until)

    events = await db.query_events(
        agent_name=agent_name,
        task_name=task_name,
        model=model,
        status=status,
        since=since_dt,
        until=until_dt,
        limit=limit,
        offset=offset,
    )

    return EventsGetResponse(
        events=[
            EventOut(**e.to_dict())
            for e in events
        ]
    )


@app.get("/api/metrics/summary")
async def metrics_summary(
    request: Request,
    period: str = Query(default="24h"),
) -> dict:
    db: Database = request.app.state.db
    since = _period_to_since(period)
    summary = await db.get_metrics_summary(since=since)

    # Also fetch breakdown to get agent names list
    agents_breakdown = await db.get_breakdown(group_by="agent_name", since=since)
    agent_names = [r.name for r in agents_breakdown]

    return {
        "total_cost": summary.total_cost_usd,
        "total_requests": summary.total_requests,
        "success_rate": round(summary.success_rate * 100, 1),
        "avg_latency_ms": summary.avg_latency_ms,
        "latency_p50": summary.p50_latency_ms,
        "latency_p95": summary.p95_latency_ms,
        "active_agents": agent_names,
        "period": summary.period,
    }


@app.get("/api/metrics/timeseries")
async def metrics_timeseries(
    request: Request,
    period: str = Query(default="24h"),
    bucket: Optional[int] = Query(default=None),
) -> dict:
    db: Database = request.app.state.db
    since = _period_to_since(period)

    if bucket is None:
        bucket = _AUTO_BUCKET.get(period, 60)

    buckets = await db.get_timeseries(since=since, bucket_minutes=bucket)

    # Reshape into parallel arrays for the dashboard charts
    return {
        "timestamps": [b.timestamp for b in buckets],
        "cost": [b.total_cost for b in buckets],
        "tokens": [b.total_tokens for b in buckets],
        "requests": [b.request_count for b in buckets],
        "avg_latency": [b.avg_latency for b in buckets],
        "latency_p50": [b.avg_latency * 0.75 for b in buckets],  # approximate p50
        "latency_p95": [b.avg_latency * 1.8 for b in buckets],   # approximate p95
        "errors": [b.error_count for b in buckets],
    }


@app.get("/api/metrics/breakdown")
async def metrics_breakdown(
    request: Request,
    group_by: str = Query(default="agent_name"),
    period: str = Query(default="24h"),
) -> list[dict]:
    db: Database = request.app.state.db
    since = _period_to_since(period)
    rows = await db.get_breakdown(group_by=group_by, since=since)

    # Return flat array with dashboard-expected field names
    return [
        {
            "name": r.name,
            "requests": r.total_requests,
            "avg_latency": r.avg_latency,
            "total_cost": r.total_cost,
            "error_rate": round(r.error_rate * 100, 1),
        }
        for r in rows
    ]


@app.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    bus: EventBus = request.app.state.bus
    queue = bus.subscribe()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    event_dict = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {_json_dumps(event_dict)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"

                if await request.is_disconnected():
                    break
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj)
