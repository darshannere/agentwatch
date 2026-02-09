"""AgentWatch SDK — Three-tier instrumentation for AI agent observability."""

from __future__ import annotations

import atexit
import functools
import importlib
import logging
import queue
import threading
import time
import warnings
from typing import Any, Callable, Optional, TypeVar, overload

import httpx

from agentwatch.models import Event, calculate_cost

logger = logging.getLogger("agentwatch")

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_server_url: str = "http://localhost:8787"
_sender: Optional[BackgroundSender] = None
_initialized: bool = False
_default_agent: str = "default"
_default_task: str = "default"


# ---------------------------------------------------------------------------
# Background event sender
# ---------------------------------------------------------------------------

class BackgroundSender:
    """Thread-safe batching sender that flushes events over HTTP."""

    _FLUSH_INTERVAL: float = 5.0
    _BATCH_SIZE: int = 50

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public api --

    def enqueue(self, event_dict: dict[str, Any]) -> None:
        self._queue.put(event_dict)

    def flush(self) -> None:
        batch = self._drain()
        if batch:
            self._send(batch)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        self.flush()

    # -- internals --

    def _drain(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._FLUSH_INTERVAL)
            self.flush()

    def _send(self, batch: list[dict[str, Any]]) -> None:
        url = f"{self._server_url}/api/events"
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(url, json={"events": batch})
        except Exception:
            logger.debug("AgentWatch: failed to send %d events", len(batch), exc_info=True)


# ---------------------------------------------------------------------------
# Forward reference for module-level type hint (BackgroundSender used above)
# ---------------------------------------------------------------------------
# (BackgroundSender is defined before use in _sender, but the annotation at
#  module scope is evaluated lazily thanks to `from __future__ import annotations`.)


# ---------------------------------------------------------------------------
# Response extractors
# ---------------------------------------------------------------------------

def _extract_openai_response(response: Any) -> dict[str, Any]:
    """Extract model and token counts from an OpenAI ChatCompletion response."""
    model: str = getattr(response, "model", "unknown")
    usage = getattr(response, "usage", None)
    input_tokens: int = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens: int = getattr(usage, "completion_tokens", 0) if usage else 0
    return {"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens}


def _extract_anthropic_response(response: Any) -> dict[str, Any]:
    """Extract model and token counts from an Anthropic Message response."""
    model: str = getattr(response, "model", "unknown")
    usage = getattr(response, "usage", None)
    input_tokens: int = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens: int = getattr(usage, "output_tokens", 0) if usage else 0
    return {"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens}


def _try_extract_response(response: Any) -> Optional[dict[str, Any]]:
    """Attempt to detect and extract from an OpenAI or Anthropic response object."""
    cls_name = type(response).__name__
    module = type(response).__module__ or ""

    if "openai" in module or cls_name == "ChatCompletion":
        return _extract_openai_response(response)
    if "anthropic" in module or cls_name == "Message":
        return _extract_anthropic_response(response)
    return None


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------

def _ensure_initialized() -> bool:
    if not _initialized:
        warnings.warn(
            "AgentWatch SDK is not initialized. Call agentwatch.init() first.",
            stacklevel=3,
        )
        return False
    return True


def _record_event(
    *,
    task_name: str,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    status: str,
    error_message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    if _sender is None:
        return
    cost = calculate_cost(model, input_tokens, output_tokens)
    event = Event(
        task_name=task_name,
        agent_name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost,
        status=status,
        error_message=error_message,
        metadata=metadata,
    )
    _sender.enqueue(event.to_dict())


# ---------------------------------------------------------------------------
# Tier 3: Manual logging
# ---------------------------------------------------------------------------

def log_event(
    *,
    task_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: str = "unknown",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    status: str = "success",
    error_message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Manually log an observability event (Tier 3)."""
    if not _ensure_initialized():
        return
    _record_event(
        task_name=task_name or _default_task,
        agent_name=agent_name or _default_agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        status=status,
        error_message=error_message,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Tier 2: Decorator
# ---------------------------------------------------------------------------

@overload
def track(fn: F) -> F: ...


@overload
def track(
    *,
    task: Optional[str] = None,
    agent: Optional[str] = None,
) -> Callable[[F], F]: ...


def track(
    fn: Optional[F] = None,
    *,
    task: Optional[str] = None,
    agent: Optional[str] = None,
) -> Any:
    """Decorator that tracks a function call as an observability event (Tier 2).

    Can be used as ``@track`` or ``@track(task="x", agent="y")``.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _initialized:
                return func(*args, **kwargs)

            start = time.perf_counter()
            status = "success"
            error_message: Optional[str] = None
            result: Any = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                status = "error"
                error_message = str(exc)
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                extracted = _try_extract_response(result) if result is not None else None
                model = extracted["model"] if extracted else "unknown"
                input_tokens = extracted["input_tokens"] if extracted else 0
                output_tokens = extracted["output_tokens"] if extracted else 0
                _record_event(
                    task_name=task or _default_task,
                    agent_name=agent or _default_agent,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _initialized:
                return await func(*args, **kwargs)

            start = time.perf_counter()
            status = "success"
            error_message: Optional[str] = None
            result: Any = None
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as exc:
                status = "error"
                error_message = str(exc)
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                extracted = _try_extract_response(result) if result is not None else None
                model = extracted["model"] if extracted else "unknown"
                input_tokens = extracted["input_tokens"] if extracted else 0
                output_tokens = extracted["output_tokens"] if extracted else 0
                _record_event(
                    task_name=task or _default_task,
                    agent_name=agent or _default_agent,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# Tier 1: Auto-patching
# ---------------------------------------------------------------------------

def _patch_openai() -> None:
    """Monkey-patch OpenAI client to auto-track chat completion calls."""
    try:
        openai_mod = importlib.import_module("openai.resources.chat.completions")
    except (ImportError, ModuleNotFoundError):
        logger.debug("AgentWatch: openai not installed, skipping patch")
        return

    completions_cls = getattr(openai_mod, "Completions", None)
    if completions_cls is None:
        return

    _patch_sync_openai(completions_cls)
    _patch_async_openai()


def _patch_sync_openai(completions_cls: type) -> None:
    original = completions_cls.create

    @functools.wraps(original)
    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        error_message: Optional[str] = None
        response: Any = None
        try:
            response = original(self, *args, **kwargs)
            return response
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            try:
                latency_ms = (time.perf_counter() - start) * 1000.0
                if response is not None:
                    extracted = _extract_openai_response(response)
                else:
                    extracted = {
                        "model": kwargs.get("model", "unknown"),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                _record_event(
                    task_name=_default_task,
                    agent_name=_default_agent,
                    model=extracted["model"],
                    input_tokens=extracted["input_tokens"],
                    output_tokens=extracted["output_tokens"],
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
            except Exception:
                logger.debug("AgentWatch: error in OpenAI sync tracking", exc_info=True)

    completions_cls.create = patched_create  # type: ignore[assignment]


def _patch_async_openai() -> None:
    try:
        async_mod = importlib.import_module("openai.resources.chat.completions")
    except (ImportError, ModuleNotFoundError):
        return

    async_cls = getattr(async_mod, "AsyncCompletions", None)
    if async_cls is None:
        return

    original = async_cls.create

    @functools.wraps(original)
    async def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        error_message: Optional[str] = None
        response: Any = None
        try:
            response = await original(self, *args, **kwargs)
            return response
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            try:
                latency_ms = (time.perf_counter() - start) * 1000.0
                if response is not None:
                    extracted = _extract_openai_response(response)
                else:
                    extracted = {
                        "model": kwargs.get("model", "unknown"),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                _record_event(
                    task_name=_default_task,
                    agent_name=_default_agent,
                    model=extracted["model"],
                    input_tokens=extracted["input_tokens"],
                    output_tokens=extracted["output_tokens"],
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
            except Exception:
                logger.debug("AgentWatch: error in OpenAI async tracking", exc_info=True)

    async_cls.create = patched_create  # type: ignore[assignment]


def _patch_anthropic() -> None:
    """Monkey-patch Anthropic client to auto-track message creation calls."""
    try:
        messages_mod = importlib.import_module("anthropic.resources.messages")
    except (ImportError, ModuleNotFoundError):
        logger.debug("AgentWatch: anthropic not installed, skipping patch")
        return

    messages_cls = getattr(messages_mod, "Messages", None)
    if messages_cls is not None:
        _patch_sync_anthropic(messages_cls)

    async_cls = getattr(messages_mod, "AsyncMessages", None)
    if async_cls is not None:
        _patch_async_anthropic(async_cls)


def _patch_sync_anthropic(messages_cls: type) -> None:
    original = messages_cls.create

    @functools.wraps(original)
    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        error_message: Optional[str] = None
        response: Any = None
        try:
            response = original(self, *args, **kwargs)
            return response
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            try:
                latency_ms = (time.perf_counter() - start) * 1000.0
                if response is not None:
                    extracted = _extract_anthropic_response(response)
                else:
                    extracted = {
                        "model": kwargs.get("model", "unknown"),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                _record_event(
                    task_name=_default_task,
                    agent_name=_default_agent,
                    model=extracted["model"],
                    input_tokens=extracted["input_tokens"],
                    output_tokens=extracted["output_tokens"],
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
            except Exception:
                logger.debug("AgentWatch: error in Anthropic sync tracking", exc_info=True)

    messages_cls.create = patched_create  # type: ignore[assignment]


def _patch_async_anthropic(async_cls: type) -> None:
    original = async_cls.create

    @functools.wraps(original)
    async def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        status = "success"
        error_message: Optional[str] = None
        response: Any = None
        try:
            response = await original(self, *args, **kwargs)
            return response
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            raise
        finally:
            try:
                latency_ms = (time.perf_counter() - start) * 1000.0
                if response is not None:
                    extracted = _extract_anthropic_response(response)
                else:
                    extracted = {
                        "model": kwargs.get("model", "unknown"),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                _record_event(
                    task_name=_default_task,
                    agent_name=_default_agent,
                    model=extracted["model"],
                    input_tokens=extracted["input_tokens"],
                    output_tokens=extracted["output_tokens"],
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
            except Exception:
                logger.debug("AgentWatch: error in Anthropic async tracking", exc_info=True)

    async_cls.create = patched_create  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init(
    *,
    server_url: str = "http://localhost:8787",
    agent: str = "default",
    task: str = "default",
    auto_patch: bool = True,
) -> None:
    """Initialize the AgentWatch SDK.

    Args:
        server_url: The AgentWatch server URL to send events to.
        agent: Default agent name attached to events.
        task: Default task name attached to events.
        auto_patch: If True, monkey-patch openai and anthropic clients.
    """
    global _server_url, _sender, _initialized, _default_agent, _default_task

    if _initialized:
        logger.warning("AgentWatch SDK already initialized; re-initializing")
        if _sender is not None:
            _sender.shutdown()

    _server_url = server_url
    _default_agent = agent
    _default_task = task
    _sender = BackgroundSender(server_url)
    _initialized = True

    atexit.register(_shutdown)

    if auto_patch:
        _patch_openai()
        _patch_anthropic()

    logger.info("AgentWatch SDK initialized (server=%s)", server_url)


def _shutdown() -> None:
    global _sender, _initialized
    if _sender is not None:
        _sender.shutdown()
        _sender = None
    _initialized = False


# ---------------------------------------------------------------------------
# Convenience wrapper class
# ---------------------------------------------------------------------------

class AgentWatch:
    """Object-oriented wrapper around the module-level SDK functions."""

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8787",
        agent: str = "default",
        task: str = "default",
        auto_patch: bool = True,
    ) -> None:
        init(server_url=server_url, agent=agent, task=task, auto_patch=auto_patch)

    @staticmethod
    def log_event(
        *,
        task_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        model: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        status: str = "success",
        error_message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        log_event(
            task_name=task_name,
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
            metadata=metadata,
        )

    @staticmethod
    def track(
        fn: Optional[F] = None,
        *,
        task: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> Any:
        return track(fn=fn, task=task, agent=agent)
