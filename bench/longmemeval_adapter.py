#!/usr/bin/env python3
"""
LongMemEval benchmark adapter for rein.

Usage:
    # Start rein in Docker first:
    #   docker run --rm -e REIN_HTTP_TOKEN=bench -p 8680:8680 -v rein-bench:/data rein:latest
    #
    # Run benchmark (oracle, 500 questions):
    #   python3 bench/longmemeval_adapter.py --data /tmp/LongMemEval/data/longmemeval_oracle.json --out bench/results.jsonl
    #
    # Evaluate:
    #   cd /tmp/LongMemEval/src/evaluation
    #   python3 evaluate_qa.py gpt-4o ../../bench/results.jsonl ../../data/longmemeval_oracle.json
"""

import argparse
import json
import os
import time
import requests
import uuid


REIN_URL = os.environ.get("REIN_URL", "http://localhost:8680/mcp")
REIN_TOKEN = os.environ.get("REIN_HTTP_TOKEN", "bench")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QA_MODEL = os.environ.get("QA_MODEL", "gpt-4o")  # Model for answering questions

# MCP session state
SESSION_ID = None


def mcp_request(method: str, params: dict | None = None) -> dict | None:
    """Send an MCP JSON-RPC request to rein via Streamable HTTP."""
    global SESSION_ID
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
    }
    if params:
        payload["params"] = params

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {REIN_TOKEN}",
    }
    if SESSION_ID:
        headers["Mcp-Session-Id"] = SESSION_ID

    resp = requests.post(REIN_URL, json=payload, headers=headers, stream=True, timeout=30)

    # Extract session ID from response headers
    if "Mcp-Session-Id" in resp.headers:
        SESSION_ID = resp.headers["Mcp-Session-Id"]

    # Parse SSE response
    result = None
    for raw_line in resp.iter_lines():
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line
        if line and line.startswith("data: "):
            data = line[6:]
            if data.strip():
                try:
                    msg = json.loads(data)
                    if "result" in msg:
                        result = msg["result"]
                except json.JSONDecodeError:
                    pass
    return result


def mcp_init():
    """Initialize MCP session."""
    global SESSION_ID
    result = mcp_request("initialize", {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "longmemeval-bench", "version": "1.0"},
    })
    # Send initialized notification (no id for notifications)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {REIN_TOKEN}",
    }
    if SESSION_ID:
        headers["Mcp-Session-Id"] = SESSION_ID
    requests.post(REIN_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                  headers=headers, timeout=5)
    return result


def rein_store(content: str, topic: str = "longmemeval", keywords: str = "") -> str | None:
    """Store a memory in rein via MCP tools/call."""
    args = {"content": content, "topic": topic}
    if keywords:
        args["keywords"] = keywords
    result = mcp_request("tools/call", {
        "name": "rein_store",
        "arguments": args,
    })
    if result and "content" in result:
        for item in result["content"]:
            if item.get("type") == "text":
                return item["text"]
    return None


def rein_ingest_session(content: str, agent_label: str = "longmemeval") -> str | None:
    """Ingest a full session in rein via MCP tools/call."""
    result = mcp_request("tools/call", {
        "name": "rein_ingest_session",
        "arguments": {
            "content": content,
            "agent_label": agent_label,
            "is_subagent": False,
        },
    })
    if result and "content" in result:
        for item in result["content"]:
            if item.get("type") == "text":
                return item["text"]
    return None


def rein_recall(query: str, limit: int = 10) -> str | None:
    """Recall memories from rein via MCP tools/call."""
    result = mcp_request("tools/call", {
        "name": "rein_recall",
        "arguments": {"query": query, "limit": limit},
    })
    if result and "content" in result:
        for item in result["content"]:
            if item.get("type") == "text":
                return item["text"]
    return None


def rein_forget_all():
    """Clear all memories by forgetting each one. Uses rein_stats to check count."""
    # Get topics, then forget by listing recent
    result = mcp_request("tools/call", {
        "name": "rein_recent",
        "arguments": {"limit": 1000},
    })
    if result and "content" in result:
        for item in result["content"]:
            if item.get("type") == "text":
                text = item["text"]
                # Extract IDs from "id: XXXX" lines
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("id: "):
                        mem_id = line[4:].strip()
                        mcp_request("tools/call", {
                            "name": "rein_forget",
                            "arguments": {"id": mem_id},
                        })


def llm_answer(question: str, context: str, question_date: str) -> str:
    """Use LLM (GPT-4o or Gemini) to answer the question given recalled context."""
    system_prompt = "You are a helpful assistant with access to the user's conversation history."
    user_prompt = f"""Based on the following recalled memories from past conversations, answer the user's question.
Today's date: {question_date}

=== Recalled Memories ===
{context}
=== End Memories ===

Question: {question}

Answer concisely and directly. If the information is not available in the memories, say "I don't have that information."
"""

    if OPENAI_API_KEY and QA_MODEL.startswith("gpt"):
        return _openai_chat(system_prompt, user_prompt)
    elif GEMINI_API_KEY:
        return _gemini_chat(system_prompt + "\n\n" + user_prompt)
    else:
        return f"[no API key configured] context: {context[:500]}"


def _openai_chat(system: str, user: str) -> str:
    """Call OpenAI chat completion with rate limit handling."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": QA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    for attempt in range(6):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 5:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            return f"[error: {e}]"
    return "[error: max retries exceeded]"


def _gemini_chat(prompt: str) -> str:
    """Call Gemini generateContent."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"[error: {e}]"


def format_session(session: list[dict], date: str) -> str:
    """Format a chat session into storable text."""
    lines = [f"[Session date: {date}]"]
    for turn in session:
        role = turn["role"].capitalize()
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def run_benchmark(data_path: str, output_path: str, limit: int | None = None,
                  skip_existing: bool = True):
    """Run LongMemEval benchmark against rein."""
    with open(data_path) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    # Load existing results to skip
    existing_ids = set()
    if skip_existing and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj["question_id"])
                except:
                    pass
        print(f"Loaded {len(existing_ids)} existing results, skipping those")

    print(f"Running {len(dataset)} questions against rein at {REIN_URL}")
    print(f"QA model: {QA_MODEL}" + (f" (OpenAI)" if OPENAI_API_KEY and QA_MODEL.startswith("gpt") else f" (Gemini)"))

    # Initialize MCP session
    mcp_init()

    results = []
    for i, item in enumerate(dataset):
        qid = item["question_id"]
        if qid in existing_ids:
            continue

        qtype = item["question_type"]
        question = item["question"]
        answer = item["answer"]
        question_date = item["question_date"]
        sessions = item["haystack_sessions"]
        dates = item["haystack_dates"]

        print(f"\n[{i+1}/{len(dataset)}] {qid} ({qtype})")

        # Clear previous question's memories
        rein_forget_all()
        time.sleep(0.1)

        # Store all sessions through full ingestion so the benchmark exercises
        # memories + concepts + links + episodes, not just flat store().
        for j, (session, date) in enumerate(zip(sessions, dates)):
            text = format_session(session, date)
            rein_ingest_session(text, "longmemeval")

        # Recall with the question
        recalled = rein_recall(question, limit=15)
        if not recalled:
            recalled = "No memories found."

        # Generate answer
        hypothesis = llm_answer(question, recalled, question_date)

        print(f"  Q: {question[:100]}")
        print(f"  A: {answer[:100]}")
        print(f"  H: {hypothesis[:100]}")

        result = {"question_id": qid, "hypothesis": hypothesis}
        results.append(result)

        # Append to output file incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"\nDone. {len(results)} new results written to {output_path}")
    total = len(results) + len(existing_ids)
    print(f"Total results: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval adapter for rein")
    parser.add_argument("--data", required=True, help="Path to longmemeval JSON file")
    parser.add_argument("--out", default="bench/results.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--rein-url", default=None, help="rein MCP URL")
    parser.add_argument("--rein-token", default=None, help="rein HTTP token")
    args = parser.parse_args()

    if args.rein_url:
        REIN_URL = args.rein_url
    if args.rein_token:
        REIN_TOKEN = args.rein_token

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run_benchmark(args.data, args.out, args.limit)
