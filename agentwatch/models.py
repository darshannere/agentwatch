from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4-turbo": (0.0100, 0.0300),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-opus-4-20250514": (0.0150, 0.0750),
    "claude-sonnet-4-20250514": (0.0030, 0.0150),
    "claude-haiku-3-5-20241022": (0.0008, 0.0040),
    "claude-3-5-sonnet-20241022": (0.0030, 0.0150),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}

_DEFAULT_INPUT_PRICE = 0.0030
_DEFAULT_OUTPUT_PRICE = 0.0150


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = PRICING.get(model, (_DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE))
    return (input_tokens * input_price + output_tokens * output_price) / 1000.0


@dataclass
class Event:
    task_name: str
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    status: str
    error_message: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        data = dict(data)
        ts = data.get("timestamp")
        if isinstance(ts, str):
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**data)


@dataclass
class EventBatch:
    events: list[Event]


@dataclass
class MetricsSummary:
    total_cost_usd: float
    total_requests: int
    success_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    active_agents: int
    period: str


@dataclass
class TimeseriesBucket:
    timestamp: str
    total_cost: float
    total_tokens: int
    request_count: int
    avg_latency: float
    error_count: int


@dataclass
class BreakdownRow:
    name: str
    total_cost: float
    total_requests: int
    avg_latency: float
    error_rate: float
