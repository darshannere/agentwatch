from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from .models import (
    BreakdownRow,
    Event,
    MetricsSummary,
    TimeseriesBucket,
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
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
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_events_agent_name ON events (agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_events_task_name ON events (task_name)",
    "CREATE INDEX IF NOT EXISTS idx_events_status ON events (status)",
]


def _serialize_metadata(metadata: Optional[dict]) -> Optional[str]:
    if metadata is None:
        return None
    import json
    return json.dumps(metadata)


def _deserialize_metadata(raw: Optional[str]) -> Optional[dict]:
    if raw is None:
        return None
    import json
    return json.loads(raw)


def _row_to_event(row: aiosqlite.Row) -> Event:
    ts = row["timestamp"]
    if isinstance(ts, str):
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        ts = datetime.fromisoformat(ts)

    return Event(
        id=row["id"],
        task_name=row["task_name"],
        agent_name=row["agent_name"],
        model=row["model"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        latency_ms=row["latency_ms"],
        cost_usd=row["cost_usd"],
        status=row["status"],
        error_message=row["error_message"],
        timestamp=ts,
        metadata=_deserialize_metadata(row["metadata"]),
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


class Database:
    def __init__(self, db_path: str = "agentwatch.db") -> None:
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def init(self) -> None:
        conn = await self._get_conn()
        await conn.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await conn.execute(idx_sql)
        await conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def insert_events(self, events: list[Event]) -> None:
        conn = await self._get_conn()
        await conn.executemany(
            """
            INSERT OR REPLACE INTO events
                (id, task_name, agent_name, model, input_tokens, output_tokens,
                 latency_ms, cost_usd, status, error_message, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.id,
                    e.task_name,
                    e.agent_name,
                    e.model,
                    e.input_tokens,
                    e.output_tokens,
                    e.latency_ms,
                    e.cost_usd,
                    e.status,
                    e.error_message,
                    e.timestamp.isoformat(),
                    _serialize_metadata(e.metadata),
                )
                for e in events
            ],
        )
        await conn.commit()

    async def query_events(
        self,
        agent_name: Optional[str] = None,
        task_name: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list = []

        if agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        if task_name is not None:
            clauses.append("task_name = ?")
            params.append(task_name)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = await self._get_conn()
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_event(row) for row in rows]

    async def get_metrics_summary(self, since: Optional[datetime] = None) -> MetricsSummary:
        params: list = []
        where = ""
        if since is not None:
            where = "WHERE timestamp >= ?"
            params.append(since.isoformat())

        conn = await self._get_conn()

        agg_query = f"""
            SELECT
                COALESCE(SUM(cost_usd), 0) AS total_cost,
                COUNT(*) AS total_requests,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_count,
                COALESCE(AVG(latency_ms), 0) AS avg_latency,
                COUNT(DISTINCT agent_name) AS active_agents
            FROM events {where}
        """
        cursor = await conn.execute(agg_query, params)
        row = await cursor.fetchone()

        total_requests = row["total_requests"]
        success_rate = row["success_count"] / total_requests if total_requests > 0 else 0.0

        latency_query = f"SELECT latency_ms FROM events {where} ORDER BY latency_ms"
        cursor = await conn.execute(latency_query, params)
        latency_rows = await cursor.fetchall()
        latencies = [r["latency_ms"] for r in latency_rows]

        period = f"since {since.isoformat()}" if since else "all_time"

        return MetricsSummary(
            total_cost_usd=row["total_cost"],
            total_requests=total_requests,
            success_rate=round(success_rate, 4),
            avg_latency_ms=round(row["avg_latency"], 2),
            p50_latency_ms=round(_percentile(latencies, 50), 2),
            p95_latency_ms=round(_percentile(latencies, 95), 2),
            active_agents=row["active_agents"],
            period=period,
        )

    async def get_timeseries(
        self,
        since: Optional[datetime] = None,
        bucket_minutes: int = 60,
    ) -> list[TimeseriesBucket]:
        params: list = []
        where = ""
        if since is not None:
            where = "WHERE timestamp >= ?"
            params.append(since.isoformat())

        bucket_seconds = bucket_minutes * 60

        query = f"""
            SELECT
                strftime('%Y-%m-%dT%H:%M:00',
                    datetime(
                        (CAST(strftime('%s', timestamp) AS INTEGER) / {bucket_seconds}) * {bucket_seconds},
                        'unixepoch'
                    )
                ) AS bucket,
                COALESCE(SUM(cost_usd), 0) AS total_cost,
                COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
                COUNT(*) AS request_count,
                COALESCE(AVG(latency_ms), 0) AS avg_latency,
                COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS error_count
            FROM events {where}
            GROUP BY bucket
            ORDER BY bucket
        """

        conn = await self._get_conn()
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        return [
            TimeseriesBucket(
                timestamp=row["bucket"],
                total_cost=round(row["total_cost"], 6),
                total_tokens=row["total_tokens"],
                request_count=row["request_count"],
                avg_latency=round(row["avg_latency"], 2),
                error_count=row["error_count"],
            )
            for row in rows
        ]

    async def get_breakdown(
        self,
        group_by: str = "agent_name",
        since: Optional[datetime] = None,
    ) -> list[BreakdownRow]:
        allowed_columns = {"agent_name", "task_name", "model", "status"}
        if group_by not in allowed_columns:
            raise ValueError(f"group_by must be one of {allowed_columns}")

        params: list = []
        where = ""
        if since is not None:
            where = "WHERE timestamp >= ?"
            params.append(since.isoformat())

        query = f"""
            SELECT
                {group_by} AS name,
                COALESCE(SUM(cost_usd), 0) AS total_cost,
                COUNT(*) AS total_requests,
                COALESCE(AVG(latency_ms), 0) AS avg_latency,
                COALESCE(
                    SUM(CASE WHEN status = 'error' THEN 1.0 ELSE 0.0 END) / COUNT(*),
                    0
                ) AS error_rate
            FROM events {where}
            GROUP BY {group_by}
            ORDER BY total_cost DESC
        """

        conn = await self._get_conn()
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()

        return [
            BreakdownRow(
                name=row["name"],
                total_cost=round(row["total_cost"], 6),
                total_requests=row["total_requests"],
                avg_latency=round(row["avg_latency"], 2),
                error_rate=round(row["error_rate"], 4),
            )
            for row in rows
        ]

    async def get_latest_event_id(self) -> Optional[str]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT id FROM events ORDER BY timestamp DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row["id"] if row else None
