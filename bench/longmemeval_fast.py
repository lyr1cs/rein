#!/usr/bin/env python3
"""
Fast LongMemEval adapter for rein.
Uses local rein CLI with per-question temp DBs + parallel workers.

Usage:
    python3 bench/longmemeval_fast.py --data /tmp/LongMemEval/data/longmemeval_oracle.json --out bench/results.jsonl
    python3 bench/longmemeval_fast.py --data /tmp/LongMemEval/data/longmemeval_oracle.json --out bench/results.jsonl --workers 8
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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
QA_MODEL = os.environ.get("QA_MODEL", "gpt-4o")
REIN_BIN = os.environ.get("REIN_BIN", "rein")
USE_LLM_EXTRACT = True  # Use Gemini Flash Lite to extract structured memories


def rein_store(db_path: str, content: str, topic: str = "longmemeval") -> bool:
    """Store a memory using rein CLI."""
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
    """Recall memories using rein CLI."""
    try:
        result = subprocess.run(
            [REIN_BIN, "recall", query, "-l", str(limit)],
            env={**os.environ, "REIN_DB": db_path},
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def format_session(session: list, date: str) -> str:
    """Format a chat session into storable text."""
    lines = [f"[Session date: {date}]"]
    for turn in session:
        role = turn["role"].capitalize()
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def llm_extract(session_text: str, date: str) -> list[str]:
    """Use Gemini Flash Lite to extract structured memories from a conversation session.
    Returns a list of extracted memory strings."""
    if not GEMINI_API_KEY:
        return [session_text]  # fallback: store raw

    model = "gemini-3.1-flash-lite-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""Extract the key facts, preferences, events, and decisions from this conversation.
Output each memory as a separate line, prefixed with "- ".
Include dates, names, specific details, and user preferences.
Keep each memory concise (1-2 sentences) but include all important details.

Session date: {date}

{session_text[:6000]}

Extracted memories:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Parse bullet points
            memories = []
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    mem = f"[{date}] {line[2:]}"
                    memories.append(mem)
            return memories if memories else [session_text[:4000]]
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return [session_text[:4000]]  # fallback


def openai_answer(system: str, user: str) -> str:
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
                wait = min(2 ** (attempt + 1), 60)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) and attempt < 5:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            return f"[error: {e}]"
        except Exception as e:
            if attempt < 5:
                time.sleep(2)
                continue
            return f"[error: {e}]"
    return "[error: max retries exceeded]"


def gemini_answer(prompt: str) -> str:
    """Call Gemini as fallback."""
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
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
                time.sleep(2 ** (attempt + 1))
            else:
                return f"[error: {e}]"
    return "[error: max retries exceeded]"


def llm_answer(question: str, context: str, question_date: str) -> str:
    """Answer question with recalled context."""
    system = "You are a helpful assistant with access to the user's conversation history."
    user = f"""Based on the following recalled memories from past conversations, answer the user's question.
Today's date: {question_date}

=== Recalled Memories ===
{context}
=== End Memories ===

Question: {question}

Answer concisely and directly. If the information is not available in the memories, say "I don't have that information."
"""
    if OPENAI_API_KEY and QA_MODEL.startswith("gpt"):
        return openai_answer(system, user)
    elif GEMINI_API_KEY:
        return gemini_answer(system + "\n\n" + user)
    else:
        return f"[no API key] {context[:300]}"


def process_question(item: dict, idx: int, total: int) -> dict:
    """Process a single question with a temp DB."""
    qid = item["question_id"]
    qtype = item["question_type"]
    question = item["question"]
    answer = item["answer"]
    question_date = item["question_date"]
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]

    # Create temp DB for this question
    tmp = tempfile.mktemp(suffix=".db", prefix=f"rein_bench_{qid}_")

    try:
        # Store all sessions (with LLM extraction if enabled)
        for session, date in zip(sessions, dates):
            text = format_session(session, date)
            if USE_LLM_EXTRACT:
                memories = llm_extract(text, date)
                for mem in memories:
                    rein_store(tmp, mem)
            else:
                if len(text) > 8000:
                    text = text[:8000] + "\n[truncated]"
                rein_store(tmp, text)

        # Recall
        recalled = rein_recall(tmp, question, limit=15)
        if not recalled:
            recalled = "No memories found."

        # Answer
        hypothesis = llm_answer(question, recalled, question_date)

        print(f"  [{idx+1}/{total}] {qid} ({qtype})")
        print(f"    Q: {question[:80]}")
        print(f"    A: {answer[:80]}")
        print(f"    H: {hypothesis[:80]}")

        return {"question_id": qid, "hypothesis": hypothesis}

    finally:
        # Cleanup temp DB and related files
        for ext in ["", "-wal", "-shm"]:
            p = Path(tmp + ext)
            if p.exists():
                p.unlink()
        # Cleanup tantivy/hnsw index dirs
        base = Path(tmp).parent
        for pattern in [f"rein_bench_{qid}_*"]:
            for f in base.glob(pattern):
                if f.is_file():
                    f.unlink()


def run_benchmark(data_path: str, output_path: str, limit: int | None = None,
                  workers: int = 4):
    """Run LongMemEval benchmark with parallel workers."""
    with open(data_path) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    # Load existing results
    existing_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["question_id"])
                except:
                    pass
        print(f"Skipping {len(existing_ids)} existing results")

    pending = [item for item in dataset if item["question_id"] not in existing_ids]
    total = len(pending)

    if total == 0:
        print("All questions already processed")
        return

    print(f"Processing {total} questions with {workers} workers")
    print(f"QA model: {QA_MODEL}")
    print(f"rein binary: {REIN_BIN}")

    start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_question, item, i, total): item
            for i, item in enumerate(pending)
        }

        with open(output_path, "a") as out:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    out.write(json.dumps(result) + "\n")
                    out.flush()
                    completed += 1
                except Exception as e:
                    item = futures[future]
                    print(f"  ERROR {item['question_id']}: {e}")

    elapsed = time.time() - start
    print(f"\nDone. {completed}/{total} in {elapsed:.0f}s ({elapsed/max(completed,1):.1f}s/q)")
    print(f"Total results: {completed + len(existing_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast LongMemEval adapter for rein")
    parser.add_argument("--data", required=True, help="Path to longmemeval JSON file")
    parser.add_argument("--out", default="bench/results_fast.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--no-extract", action="store_true", help="Disable LLM extraction (store raw)")
    args = parser.parse_args()

    if args.no_extract:
        USE_LLM_EXTRACT = False  # module-level var, no global needed in __main__

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run_benchmark(args.data, args.out, args.limit, args.workers)
