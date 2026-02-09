#!/usr/bin/env python3
"""AgentWatch Demo — Generate realistic simulated AI agent events.

Simulates multiple AI agents running various tasks, generating realistic
event data to populate the AgentWatch dashboard.

Usage:
    python demo.py                          # live mode (default)
    python demo.py --backfill               # backfill 7 days of history
    python demo.py --live --speed 2         # live mode, 2x speed
    python demo.py --server http://host:port
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Agent profiles
# ---------------------------------------------------------------------------

AGENTS: dict[str, dict[str, Any]] = {
    "research-bot": {
        "tasks": ["web-search", "summarize-article", "extract-entities", "fact-check"],
        "models": ["gpt-4o", "gpt-4o-mini"],
        "error_rate": 0.03,
    },
    "support-agent": {
        "tasks": ["classify-ticket", "draft-response", "sentiment-analysis"],
        "models": ["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022"],
        "error_rate": 0.05,
    },
    "code-reviewer": {
        "tasks": ["analyze-pr", "suggest-fixes", "generate-tests"],
        "models": ["claude-opus-4-20250514", "gpt-4o"],
        "error_rate": 0.02,
    },
    "data-pipeline": {
        "tasks": ["clean-data", "generate-embeddings", "classify-documents"],
        "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
        "error_rate": 0.08,
    },
    "content-writer": {
        "tasks": ["draft-blog", "write-summary", "generate-headlines"],
        "models": ["claude-sonnet-4-20250514", "gpt-4o"],
        "error_rate": 0.04,
    },
}

# ---------------------------------------------------------------------------
# Token ranges by task complexity
# ---------------------------------------------------------------------------

SIMPLE_TASKS = {
    "classify-ticket", "sentiment-analysis", "extract-entities",
    "classify-documents", "generate-headlines", "fact-check",
}

MEDIUM_TASKS = {
    "summarize-article", "draft-response", "suggest-fixes",
    "write-summary", "clean-data", "web-search",
    "generate-embeddings",
}

HEAVY_TASKS = {
    "analyze-pr", "draft-blog", "generate-tests",
}

TOKEN_RANGES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
for _t in SIMPLE_TASKS:
    TOKEN_RANGES[_t] = ((200, 800), (50, 200))
for _t in MEDIUM_TASKS:
    TOKEN_RANGES[_t] = ((500, 3000), (200, 1000))
for _t in HEAVY_TASKS:
    TOKEN_RANGES[_t] = ((2000, 8000), (500, 3000))

# ---------------------------------------------------------------------------
# Pricing table (per 1M tokens) — used for cost calculation in the demo
# ---------------------------------------------------------------------------

PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

ERROR_MESSAGES = [
    "Rate limit exceeded (429)",
    "Context length exceeded: input too long",
    "API timeout after 30s",
    "Invalid response format",
    "Content filter triggered",
    "Service unavailable (503)",
]

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"

AGENT_COLORS: dict[str, str] = {
    "research-bot": _CYAN,
    "support-agent": _MAGENTA,
    "code-reviewer": _BLUE,
    "data-pipeline": _YELLOW,
    "content-writer": _GREEN,
}


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD given per-1M-token pricing."""
    input_price, output_price = PRICING.get(model, (3.00, 15.00))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def generate_event(
    timestamp: datetime | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Generate a single realistic simulated event."""
    r = rng or random.Random()

    # Pick an agent and task
    agent_name = r.choice(list(AGENTS.keys()))
    profile = AGENTS[agent_name]
    task_name = r.choice(profile["tasks"])
    model = r.choice(profile["models"])

    # Token counts
    (in_lo, in_hi), (out_lo, out_hi) = TOKEN_RANGES.get(
        task_name, ((500, 3000), (200, 1000))
    )
    input_tokens = r.randint(in_lo, in_hi)
    output_tokens = r.randint(out_lo, out_hi)

    # Determine status
    is_error = r.random() < profile["error_rate"]
    if is_error:
        status = "error"
        error_message = r.choice(ERROR_MESSAGES)
        # Errors often produce fewer output tokens
        output_tokens = r.randint(0, max(1, out_lo // 2))
    else:
        status = "success"
        error_message = None

    # Cost
    cost_usd = calculate_cost(model, input_tokens, output_tokens)

    # Latency: ~1ms per output token + 200-500ms overhead
    base_overhead_ms = r.uniform(200.0, 500.0)
    base_latency_ms = output_tokens * 1.0 + base_overhead_ms

    # Random jitter +/- 30%
    jitter_factor = r.uniform(0.7, 1.3)
    latency_ms = base_latency_ms * jitter_factor

    # Occasional slow requests (5% chance of 2-5x normal latency)
    if r.random() < 0.05:
        latency_ms *= r.uniform(2.0, 5.0)

    # For errors, latency can vary: timeouts are long, others may be short
    if is_error and error_message == "API timeout after 30s":
        latency_ms = r.uniform(28000.0, 32000.0)

    latency_ms = round(latency_ms, 1)
    cost_usd = round(cost_usd, 6)

    ts = timestamp or datetime.now(timezone.utc)

    return {
        "id": str(uuid.uuid4()),
        "task_name": task_name,
        "agent_name": agent_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "status": status,
        "error_message": error_message,
        "timestamp": ts.isoformat(),
    }


# ---------------------------------------------------------------------------
# Daily activity pattern
# ---------------------------------------------------------------------------

def hour_weight(hour: int) -> float:
    """Return a relative activity weight for a given hour (0-23).

    Simulates realistic daily patterns:
      - 00:00-06:00  low activity (10% of peak)
      - 06:00-09:00  ramp up
      - 09:00-17:00  peak
      - 17:00-22:00  wind down
      - 22:00-00:00  low activity
    """
    if 0 <= hour < 6:
        return 0.10
    elif 6 <= hour < 9:
        # Linear ramp from 0.10 to 1.0
        return 0.10 + 0.90 * ((hour - 6) / 3.0)
    elif 9 <= hour < 17:
        return 1.0
    elif 17 <= hour < 22:
        # Linear wind-down from 1.0 to 0.10
        return 1.0 - 0.90 * ((hour - 17) / 5.0)
    else:  # 22-24
        return 0.10


# ---------------------------------------------------------------------------
# HTTP sending with retry
# ---------------------------------------------------------------------------

def send_events(
    client: httpx.Client,
    server_url: str,
    events: list[dict[str, Any]],
    max_retries: int = 5,
) -> bool:
    """Send a batch of events to the server. Returns True on success."""
    url = f"{server_url}/api/events"
    payload = {"events": events}
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            resp = client.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
            return True
        except httpx.ConnectError:
            if attempt < max_retries - 1:
                print(
                    f"  {_YELLOW}Connection refused — "
                    f"retrying in {backoff:.0f}s (attempt {attempt + 1}/{max_retries}){_RESET}"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                print(
                    f"  {_RED}Failed to connect to {server_url} "
                    f"after {max_retries} attempts{_RESET}"
                )
                return False
        except httpx.HTTPStatusError as exc:
            print(f"  {_RED}Server error: {exc.response.status_code}{_RESET}")
            return False
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                print(
                    f"  {_YELLOW}Request timeout — "
                    f"retrying in {backoff:.0f}s{_RESET}"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                print(f"  {_RED}Request timed out after {max_retries} attempts{_RESET}")
                return False

    return False


# ---------------------------------------------------------------------------
# Formatted event printing
# ---------------------------------------------------------------------------

def format_event(event: dict[str, Any]) -> str:
    """Return a colored, human-readable single-line representation of an event."""
    agent = event["agent_name"]
    task = event["task_name"]
    model = event["model"]
    status = event["status"]
    latency = event["latency_ms"]
    cost = event["cost_usd"]
    in_tok = event["input_tokens"]
    out_tok = event["output_tokens"]

    agent_color = AGENT_COLORS.get(agent, _CYAN)

    if status == "success":
        status_str = f"{_GREEN}OK{_RESET}"
    else:
        err = event.get("error_message", "unknown error")
        status_str = f"{_RED}ERR{_RESET} {_DIM}{err}{_RESET}"

    # Format latency
    if latency >= 1000:
        latency_str = f"{latency / 1000:.1f}s"
    else:
        latency_str = f"{latency:.0f}ms"

    # Format cost
    if cost < 0.001:
        cost_str = f"${cost * 1000:.3f}m"  # millicents display
    else:
        cost_str = f"${cost:.4f}"

    return (
        f"  {agent_color}{_BOLD}{agent:>16}{_RESET}  "
        f"{task:<22}  "
        f"{_DIM}{model:<30}{_RESET}  "
        f"[{status_str}]  "
        f"{in_tok:>5}in/{out_tok:>5}out  "
        f"{latency_str:>7}  "
        f"{cost_str:>10}"
    )


# ---------------------------------------------------------------------------
# Mode 1: Backfill historical data
# ---------------------------------------------------------------------------

def run_backfill(server_url: str) -> None:
    """Generate ~500 events/day over the last 7 days and send in batches."""
    print(f"\n{_BOLD}AgentWatch Demo — Backfill Mode{_RESET}")
    print(f"Server: {server_url}")
    print(f"Generating 7 days of historical data...\n")

    rng = random.Random(42)  # reproducible
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)

    # Pre-compute all events
    all_events: list[dict[str, Any]] = []

    target_per_day = 500
    # Compute the sum of hourly weights to distribute events properly
    total_weight = sum(hour_weight(h) for h in range(24))

    for day_offset in range(7):
        day_start = start + timedelta(days=day_offset)
        day_label = day_start.strftime("%Y-%m-%d")
        day_events: list[dict[str, Any]] = []

        for hour in range(24):
            # Number of events in this hour, proportional to weight
            w = hour_weight(hour)
            events_this_hour = max(1, round(target_per_day * w / total_weight))

            for _ in range(events_this_hour):
                # Random minute and second within the hour
                minute = rng.randint(0, 59)
                second = rng.randint(0, 59)
                microsecond = rng.randint(0, 999999)
                ts = day_start.replace(
                    hour=hour, minute=minute, second=second,
                    microsecond=microsecond,
                )
                event = generate_event(timestamp=ts, rng=rng)
                day_events.append(event)

        # Sort by timestamp within the day
        day_events.sort(key=lambda e: e["timestamp"])
        all_events.extend(day_events)
        print(f"  {_DIM}{day_label}{_RESET}  generated {len(day_events)} events")

    print(f"\n  Total: {_BOLD}{len(all_events)}{_RESET} events")
    print(f"  Sending in batches of 50...\n")

    # Send in batches
    batch_size = 50
    sent = 0
    errors = 0

    with httpx.Client() as client:
        for i in range(0, len(all_events), batch_size):
            batch = all_events[i : i + batch_size]
            ok = send_events(client, server_url, batch)
            if ok:
                sent += len(batch)
            else:
                errors += len(batch)
                print(f"  {_RED}Failed to send batch at offset {i}{_RESET}")

            # Progress
            pct = (i + len(batch)) / len(all_events) * 100
            bar_width = 30
            filled = int(bar_width * pct / 100)
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  ({sent} sent, {errors} failed)"
            )
            sys.stdout.flush()

    print(f"\n\n  {_GREEN}Done!{_RESET} Sent {sent} events, {errors} failed.\n")


# ---------------------------------------------------------------------------
# Mode 2: Live simulation
# ---------------------------------------------------------------------------

def run_live(server_url: str, speed: float) -> None:
    """Generate events in real-time and send them individually."""
    print(f"\n{_BOLD}AgentWatch Demo — Live Mode{_RESET}")
    print(f"Server: {server_url}")
    print(f"Speed:  {speed}x")
    print(f"Press Ctrl+C to stop.\n")

    # Header
    print(
        f"  {_BOLD}{'Agent':>16}  "
        f"{'Task':<22}  "
        f"{'Model':<30}  "
        f"{'Status':>5}  "
        f"{'Tokens':>14}  "
        f"{'Latency':>7}  "
        f"{'Cost':>10}{_RESET}"
    )
    print(f"  {'=' * 120}")

    total_events = 0
    total_cost = 0.0
    total_errors = 0

    rng = random.Random()

    with httpx.Client() as client:
        try:
            while True:
                # Generate 1-3 events per cycle
                num_events = rng.randint(1, 3)

                for _ in range(num_events):
                    event = generate_event(rng=rng)
                    print(format_event(event))

                    ok = send_events(client, server_url, [event], max_retries=2)
                    if not ok:
                        print(f"    {_RED}^ failed to send{_RESET}")

                    total_events += 1
                    total_cost += event["cost_usd"]
                    if event["status"] == "error":
                        total_errors += 1

                # Wait 2-5 seconds (adjusted by speed multiplier)
                delay = rng.uniform(2.0, 5.0) / speed
                time.sleep(delay)

        except KeyboardInterrupt:
            print(f"\n\n  {_BOLD}Session Summary{_RESET}")
            print(f"  Events sent: {total_events}")
            print(f"  Total cost:  ${total_cost:.4f}")
            print(f"  Errors:      {total_errors}")
            error_rate = (total_errors / total_events * 100) if total_events else 0
            print(f"  Error rate:  {error_rate:.1f}%")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentWatch Demo — generate simulated AI agent events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python demo.py                         # live mode\n"
            "  python demo.py --backfill              # backfill 7 days\n"
            "  python demo.py --live --speed 3        # 3x speed\n"
            "  python demo.py --server http://host:80 # custom server\n"
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Run in live simulation mode (default)",
    )
    mode_group.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill 7 days of historical data",
    )

    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8787",
        help="AgentWatch server URL (default: http://localhost:8787)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for live mode (default: 1.0)",
    )

    args = parser.parse_args()

    if args.backfill:
        run_backfill(args.server)
    else:
        run_live(args.server, args.speed)


if __name__ == "__main__":
    main()
