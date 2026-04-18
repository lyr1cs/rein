"""rein memory provider for Hermes Agent.

Bridges rein's MCP server as a Hermes MemoryProvider plugin, giving the agent
cross-validated memory with adaptive search, temporal knowledge graph, and
multi-level fusion (Tantivy BM25 + HNSW + Gemini embedding).

Connects to rein via MCP stdio transport (spawns ``rein serve`` subprocess).
All 28 rein tools are available; a curated subset is exposed as agent-callable
tools, while lifecycle hooks (prefetch, sync_turn, etc.) run automatically.

Install:
    ln -s /path/to/rein/integrations/hermes ~/.hermes/plugins/memory/rein
    # or copy this directory to plugins/memory/rein/

Activate:
    hermes memory setup   # select "rein"
    # or set memory.provider: rein in ~/.hermes/config.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import textwrap
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP SDK import (Hermes already depends on mcp)
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
except ImportError:
    logger.debug("mcp package not installed — rein provider unavailable")


# ---------------------------------------------------------------------------
# Tool schemas exposed to the agent (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_REIN_RECALL_SCHEMA = {
    "name": "rein_recall",
    "description": (
        "Search rein memory by semantic query. Returns cross-validated results "
        "with confidence scores. Use before answering questions about the user, "
        "past conversations, or project context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "topic": {"type": "string", "description": "Optional topic filter."},
            "limit": {"type": "integer", "description": "Max results (default 10)."},
        },
        "required": ["query"],
    },
}

_REIN_STORE_SCHEMA = {
    "name": "rein_store",
    "description": (
        "Store an important fact, preference, or decision to rein memory. "
        "Use for information the user would expect you to remember later."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Topic/category."},
            "content": {"type": "string", "description": "Content to store."},
            "importance": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Importance level.",
            },
            "keywords": {"type": "string", "description": "Comma-separated keywords."},
        },
        "required": ["topic", "content"],
    },
}

_REIN_TIMELINE_SCHEMA = {
    "name": "rein_timeline",
    "description": (
        "View temporal knowledge graph — episodes and concept changes over time. "
        "Use for 'what happened', 'when did X change' questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "Start date (YYYY-MM-DD)."},
            "to": {"type": "string", "description": "End date (YYYY-MM-DD)."},
            "limit": {"type": "integer", "description": "Max entries (default 20)."},
        },
    },
}

_REIN_STATS_SCHEMA = {
    "name": "rein_stats",
    "description": "Show rein memory statistics — total memories, topics, health.",
    "parameters": {"type": "object", "properties": {}},
}

_EXPOSED_TOOLS = [
    _REIN_RECALL_SCHEMA,
    _REIN_STORE_SCHEMA,
    _REIN_TIMELINE_SCHEMA,
    _REIN_STATS_SCHEMA,
]

_EXPOSED_TOOL_NAMES = {t["name"] for t in _EXPOSED_TOOLS}


# ---------------------------------------------------------------------------
# Async event loop thread (manages MCP connection lifetime)
# ---------------------------------------------------------------------------


class _McpBridge:
    """Manages a persistent MCP stdio connection to ``rein serve`` on a
    dedicated background event loop thread."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._ready = threading.Event()
        self._shutdown = False
        self._conn_task: Optional[asyncio.Task] = None

    # -- public (thread-safe, blocking from caller) -------------------------

    def start(self) -> bool:
        """Start the background loop and connect to rein. Returns True on success."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="rein-mcp"
        )
        self._thread.start()
        # Wait up to 15s for MCP handshake
        return self._ready.wait(timeout=15)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a rein MCP tool synchronously. Returns result text."""
        if not self._session or not self._loop or self._shutdown:
            return json.dumps({"error": "rein not connected"})
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_call_tool(name, arguments), self._loop
            )
            result = future.result(timeout=30)
            return result
        except Exception as e:
            logger.warning("rein tool call %s failed: %s", name, e)
            return json.dumps({"error": str(e)})

    def stop(self):
        """Shut down the MCP connection and background loop."""
        self._shutdown = True
        if self._loop and self._conn_task:
            self._loop.call_soon_threadsafe(self._conn_task.cancel)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    # -- internal -----------------------------------------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._conn_task = self._loop.create_task(self._connect())
        try:
            self._loop.run_forever()
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _connect(self):
        """Establish MCP stdio connection to ``rein serve``."""
        rein_bin = shutil.which("rein")
        if not rein_bin:
            logger.error("rein binary not found on PATH")
            self._ready.set()  # unblock caller (will see _session is None)
            return

        server_params = StdioServerParameters(
            command=rein_bin,
            args=["serve"],
            env={
                **os.environ,
                # Ensure rein doesn't try to read stdin for interactive prompts
                "REIN_LOG": os.environ.get("REIN_LOG", "warn"),
            },
        )

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    logger.info("rein MCP connected (stdio)")
                    self._ready.set()
                    # Keep connection alive until shutdown
                    while not self._shutdown:
                        await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("rein MCP connection failed: %s", e)
            self._ready.set()
        finally:
            self._session = None

    async def _async_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._session:
            return json.dumps({"error": "no session"})
        result = await self._session.call_tool(name, arguments)
        # MCP tool result is a list of content blocks
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else json.dumps({"ok": True})


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

# Import ABC — handle both direct import and Hermes's own location
try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    # Fallback: define a minimal stub so the module can be imported outside Hermes
    class MemoryProvider:  # type: ignore[no-redef]
        name = ""
        def is_available(self): return False
        def initialize(self, *a, **kw): pass
        def get_tool_schemas(self): return []


class ReinMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider backed by rein's MCP server."""

    # Batch-ingest accumulated turns every N turns
    _INGEST_INTERVAL = 15

    def __init__(self):
        self._bridge: Optional[_McpBridge] = None
        self._session_id: str = ""
        self._turn_buffer: List[Dict[str, str]] = []
        self._buffer_lock = threading.Lock()
        self._flush_in_flight = threading.Event()
        self._flush_in_flight.set()  # starts as "no flush running"
        self._prefetch_cache: str = ""
        self._prefetch_query: str = ""
        self._prefetch_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "rein"

    # -- Core lifecycle -----------------------------------------------------

    def is_available(self) -> bool:
        return _MCP_AVAILABLE and shutil.which("rein") is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._bridge = _McpBridge()
        ok = self._bridge.start()
        if ok and self._bridge._session:
            logger.info("rein memory provider ready (session=%s)", session_id)
        else:
            logger.warning("rein memory provider failed to connect")

    def system_prompt_block(self) -> str:
        return textwrap.dedent("""\
            You have access to rein, a cross-validated memory system with semantic
            search and temporal knowledge graph. Memories are automatically stored
            and recalled each turn. For explicit operations use the rein_recall,
            rein_store, rein_timeline, and rein_stats tools.""")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Auto-recall relevant memories before each turn."""
        if not self._bridge:
            return ""
        # Use cached result from queue_prefetch if it matches this query
        with self._prefetch_lock:
            if self._prefetch_query == query and self._prefetch_cache:
                cached = self._prefetch_cache
                self._prefetch_cache = ""
                self._prefetch_query = ""
            else:
                cached = ""
                self._prefetch_cache = ""
                self._prefetch_query = ""
        if cached:
            return cached
        # Fallback: synchronous recall
        result = self._bridge.call_tool("rein_recall", {
            "query": query,
            "limit": 5,
        })
        if not result or result.startswith('{"error'):
            return ""
        return f"<rein-recall>\n{result}\n</rein-recall>"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the next turn's recall in background."""
        if not self._bridge:
            return

        def _warm():
            result = self._bridge.call_tool("rein_recall", {
                "query": query,
                "limit": 5,
            })
            if result and not result.startswith('{"error'):
                with self._prefetch_lock:
                    self._prefetch_cache = f"<rein-recall>\n{result}\n</rein-recall>"
                    self._prefetch_query = query

        threading.Thread(target=_warm, daemon=True).start()

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Accumulate turns and batch-ingest periodically.

        Primary ingestion happens via on_session_end. This method buffers
        turns and flushes every _INGEST_INTERVAL turns as a safety net
        for long sessions.
        """
        with self._buffer_lock:
            self._turn_buffer.append({"role": "user", "content": user_content[:2000]})
            self._turn_buffer.append({"role": "assistant", "content": assistant_content[:2000]})
            should_flush = len(self._turn_buffer) >= self._INGEST_INTERVAL * 2

        if should_flush:
            self._flush_turn_buffer()

    @staticmethod
    def _ingest_ok(result: str) -> bool:
        """Check if an ingest result indicates success (not an error)."""
        if not result:
            return False
        return not (result.startswith('{"error') or result.lower().startswith("error"))

    def _flush_turn_buffer(self) -> None:
        """Batch-ingest buffered turns via rein_ingest_session.

        Uses _flush_in_flight as an atomic gate: only one flush can run at
        a time. On success, removes the sent turns from the buffer.
        """
        with self._buffer_lock:
            # Atomic claim: check + clear in one lock scope
            if not self._flush_in_flight.is_set():
                return
            if not self._bridge or not self._turn_buffer:
                return
            self._flush_in_flight.clear()
            turns = self._turn_buffer[:]
            n = len(turns)

        def _do_ingest():
            try:
                result = self._bridge.call_tool("rein_ingest_session", {
                    "turns": turns,
                    "session_id": self._session_id,
                    "title": f"hermes-mid-session-{self._session_id[:8]}",
                })
                if self._ingest_ok(result):
                    with self._buffer_lock:
                        del self._turn_buffer[:n]
            finally:
                self._flush_in_flight.set()

        threading.Thread(target=_do_ingest, daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_EXPOSED_TOOLS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        if not self._bridge:
            return json.dumps({"error": "rein not connected"})
        if tool_name not in _EXPOSED_TOOL_NAMES:
            return json.dumps({"error": f"unknown tool: {tool_name}"})
        return self._bridge.call_tool(tool_name, args)

    def shutdown(self) -> None:
        # Flush remaining turns and wait for completion before stopping
        self._flush_turn_buffer()
        self._flush_in_flight.wait(timeout=35)
        if self._bridge:
            self._bridge.stop()
            self._bridge = None

    # -- Optional hooks -----------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush remaining buffered turns and create a session Episode.

        Waits for any in-flight background flush to complete first, then
        sends only the unflushed remainder. Clears buffer only on success.
        """
        if not self._bridge:
            return
        # Wait for any background _flush_turn_buffer to finish.
        # If timed out, a flush is still running — skip to avoid duplicates.
        if not self._flush_in_flight.wait(timeout=35):
            logger.warning("rein: in-flight flush still running at session end, skipping final ingest")
            return

        with self._buffer_lock:
            remaining = self._turn_buffer[:]

        if not remaining:
            return

        result = self._bridge.call_tool("rein_ingest_session", {
            "turns": remaining,
            "session_id": self._session_id,
            "title": f"hermes-session-{self._session_id[:8]}",
        })
        if self._ingest_ok(result):
            with self._buffer_lock:
                del self._turn_buffer[:len(remaining)]

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Save context about to be discarded by compression via rein_ingest_session.

        Does NOT touch _turn_buffer — the buffer contains recent turns that
        Hermes is keeping live, not the old prefix being compressed.
        """
        if not self._bridge:
            return ""
        turns = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if content:
                turns.append({"role": role, "content": content[:2000]})

        if not turns:
            return ""

        self._bridge.call_tool("rein_ingest_session", {
            "turns": turns,
            "session_id": self._session_id,
            "title": f"hermes-pre-compress-{self._session_id[:8]}",
        })
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Record subagent delegation results."""
        if not self._bridge:
            return
        self._bridge.call_tool("rein_ingest_session", {
            "content": f"Delegated task: {task}\n\nResult: {result[:3000]}",
            "session_id": child_session_id or self._session_id,
            "title": f"delegation-{child_session_id[:8] if child_session_id else 'unknown'}",
        })

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to rein."""
        if not self._bridge or action == "remove":
            return
        self._bridge.call_tool("rein_store", {
            "topic": f"hermes-{target}",
            "content": content,
            "importance": "high" if target == "user" else "medium",
            "keywords": f"hermes,{target},{action}",
        })

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "rein_http_token",
                "description": "HTTP token for rein server (optional, only for remote mode)",
                "secret": True,
                "required": False,
                "env_var": "REIN_HTTP_TOKEN",
            },
        ]


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx):
    ctx.register_memory_provider(ReinMemoryProvider())
