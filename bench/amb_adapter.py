#!/usr/bin/env python3
"""
MemoryAgentBench (AMB) adapter for rein.
Evaluates rein's memory capabilities across 4 competencies:
  AR (Accurate Retrieval), TTL (Test-Time Learning),
  LRU (Long-Range Understanding), CR (Conflict Resolution).

Requires: pip install datasets requests

Usage:
    python3 bench/amb_adapter.py --out bench/amb_results.jsonl
    python3 bench/amb_adapter.py --out bench/amb_results.jsonl --workers 4 --competency AR
    python3 bench/amb_adapter.py --eval bench/amb_results.jsonl  # Score existing results

Dataset: automatically downloaded from HuggingFace (ai-hyz/MemoryAgentBench)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
QA_MODEL = os.environ.get("QA_MODEL", "gpt-4o")
REIN_BIN = os.environ.get("REIN_BIN", "rein")


# ---------------------------------------------------------------------------
# rein CLI helpers
# ---------------------------------------------------------------------------

def rein_store(db_path: str, content: str, topic: str = "amb") -> bool:
    try:
        result = subprocess.run(
            [REIN_BIN, "store", "-t", topic, "-c", content],
            env={**os.environ, "REIN_DB": db_path},
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def rein_recall(db_path: str, query: str, limit: int = 15) -> str:
    try:
        result = subprocess.run(
            [REIN_BIN, "recall", query, "-l", str(limit)],
            env={**os.environ, "REIN_DB": db_path},
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def rein_update(db_path: str, memory_id: str, content: str) -> bool:
    try:
        result = subprocess.run(
            [REIN_BIN, "update", memory_id, "-c", content],
            env={**os.environ, "REIN_DB": db_path},
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LLM QA (answer questions using recalled context)
# ---------------------------------------------------------------------------

def ask_llm(query: str, context: str, model: str = QA_MODEL) -> str:
    """Use LLM to answer a question given retrieved context."""
    if "gpt" in model or "o1" in model or "o3" in model:
        return ask_openai(query, context, model)
    elif "gemini" in model:
        return ask_gemini(query, context, model)
    return ""


def ask_openai(query: str, context: str, model: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer the question based on the retrieved memories. Be concise and factual. If the memories don't contain the answer, say 'I don't know'."},
            {"role": "user", "content": f"Memories:\n{context}\n\nQuestion: {query}"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  OpenAI error: {e}", file=sys.stderr)
        return ""


def ask_gemini(query: str, context: str, model: str) -> str:
    if not GEMINI_API_KEY:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": f"Answer the question based on the retrieved memories. Be concise and factual.\n\nMemories:\n{context}\n\nQuestion: {query}"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    try:
        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"  Gemini error: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_amb_dataset(competency: str = None):
    """Load MemoryAgentBench from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("ai-hyz/MemoryAgentBench")
    items = []

    for split_name in ds:
        for item in ds[split_name]:
            if competency and item.get("competency", "") != competency:
                continue
            items.append(item)

    return items


# ---------------------------------------------------------------------------
# Process a single task
# ---------------------------------------------------------------------------

def process_task(task: dict, task_idx: int, total: int) -> dict:
    """Process one AMB task: store chunks → ask question → return result."""
    task_id = task.get("id", task.get("task_id", f"task_{task_idx}"))
    competency = task.get("competency", "unknown")
    question = task.get("question", task.get("query", ""))
    expected = task.get("answer", task.get("expected", ""))
    chunks = task.get("chunks", task.get("interactions", []))

    print(f"  [{task_idx+1}/{total}] {competency}: {question[:60]}...", file=sys.stderr)

    # Create temp DB
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memories.db")

        # Store all chunks incrementally
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                content = chunk
            elif isinstance(chunk, dict):
                content = chunk.get("content", chunk.get("text", json.dumps(chunk)))
            else:
                content = str(chunk)

            if content.strip():
                rein_store(db_path, content, topic=f"amb_{competency}")

        # Recall and answer
        context = rein_recall(db_path, question, limit=15)
        hypothesis = ask_llm(question, context) if context else ""

    return {
        "task_id": task_id,
        "competency": competency,
        "question": question,
        "expected": expected,
        "hypothesis": hypothesis,
        "context_retrieved": bool(context),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_results(results_path: str):
    """Score AMB results using SubEM and LLM judge."""
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Group by competency
    by_comp = {}
    for r in results:
        comp = r.get("competency", "unknown")
        by_comp.setdefault(comp, []).append(r)

    print(f"\n{'Competency':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 55)

    total_correct = 0
    total_count = 0

    for comp, items in sorted(by_comp.items()):
        correct = 0
        for item in items:
            expected = str(item.get("expected", "")).lower().strip()
            hypothesis = str(item.get("hypothesis", "")).lower().strip()
            # SubEM: expected substring in hypothesis
            if expected and expected in hypothesis:
                correct += 1
        total_correct += correct
        total_count += len(items)
        acc = correct / len(items) * 100 if items else 0
        print(f"{comp:<25} {correct:<10} {len(items):<10} {acc:.1f}%")

    overall = total_correct / total_count * 100 if total_count else 0
    print("-" * 55)
    print(f"{'Overall':<25} {total_correct:<10} {total_count:<10} {overall:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MemoryAgentBench adapter for rein")
    parser.add_argument("--out", type=str, default="bench/amb_results.jsonl", help="Output JSONL file")
    parser.add_argument("--competency", type=str, help="Filter by competency: AR, TTL, LRU, CR")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks (0=all)")
    parser.add_argument("--eval", type=str, help="Evaluate existing results file instead of running")
    args = parser.parse_args()

    if args.eval:
        evaluate_results(args.eval)
        return

    print(f"Loading MemoryAgentBench dataset...", file=sys.stderr)
    tasks = load_amb_dataset(args.competency)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks", file=sys.stderr)

    if not tasks:
        print("No tasks found. Check competency filter.", file=sys.stderr)
        return

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_task, task, i, len(tasks)): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

    # Sort by original order
    results.sort(key=lambda r: r.get("task_id", ""))

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"\nCompleted {len(results)} tasks in {elapsed:.1f}s", file=sys.stderr)
    print(f"Results: {out_path}", file=sys.stderr)

    # Auto-evaluate
    evaluate_results(str(out_path))


if __name__ == "__main__":
    main()
