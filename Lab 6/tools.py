"""
tools.py — Agent-callable tools for the Snowflake + RAG project
================================================================
Five tools the AI agent can invoke:
  1. query_database     — SQL against Snowflake
  2. search_documents   — RAG retrieval from PDF documents
  3. search_images      — Multimodal RAG for images
  4. generate_chart     — Create bar/line/pie charts
  5. summarize_text     — Summarize text using HuggingFace LLM
"""

import os
import sys
import json
import time
import base64
import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from scripts.sf_connect import get_conn

# ── Config ──
DB = os.getenv("SNOWFLAKE_DATABASE", "INSTRUCTOR2_DB")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "MY_SCHEMA")
HF_TOKEN = os.getenv("HF_TOKEN", "")
VALID_TABLES = ["EVENTS", "USERS"]
CHART_DIR = os.path.join(os.path.dirname(__file__), "data", "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def _fqn(table: str) -> str:
    """Return fully-qualified table name."""
    return f"{DB}.{SCHEMA}.{table}"


# ── Snowflake connection caching ──
_cached_conn = None


def _get_cached_conn():
    global _cached_conn
    if _cached_conn is None or _cached_conn.is_closed():
        _cached_conn = get_conn()
    return _cached_conn


def _run_sql(sql: str) -> tuple:
    """Execute SQL via cursor, return (DataFrame, latency_ms)."""
    t0 = time.time()
    conn = _get_cached_conn()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=columns)
    finally:
        cur.close()
    return df, int((time.time() - t0) * 1000)


# ═══════════════════════════════════════════════════════════════
# TOOL 1: query_database
# ═══════════════════════════════════════════════════════════════

def query_database(table_name: str, limit: int = 100) -> dict:
    """
    Run a SQL query against Snowflake and return results as a table.

    Args:
        table_name: Table to query (EVENTS or USERS).
        limit: Max rows to return (1-500, default 100).

    Returns:
        dict with columns, rows, row_count, latency_ms or error.
    """
    try:
        table_name = table_name.upper().strip()
        if table_name not in VALID_TABLES:
            return {"error": f"Invalid table '{table_name}'. Valid: {VALID_TABLES}"}
        limit = min(max(1, int(limit)), 500)

        sql = f"SELECT * FROM {_fqn(table_name)} LIMIT {limit};"
        df, latency = _run_sql(sql)
        return {
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "row_count": len(df),
            "latency_ms": latency,
        }
    except Exception as e:
        return {"error": f"query_database failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════
# TOOL 2: search_documents
# ═══════════════════════════════════════════════════════════════

def search_documents(query: str, top_k: int = 3) -> dict:
    """
    Retrieve relevant documents from the PDF collection using RAG.

    Args:
        query: Natural language search query.
        top_k: Number of document chunks to return (1-10, default 3).

    Returns:
        dict with query, results (list of text chunks with sources and scores).
    """
    try:
        from rag_pipeline import get_rag_pipeline
        rag = get_rag_pipeline()
        top_k = min(max(1, int(top_k)), 10)
        results = rag.search(query, top_k=top_k)
        return {
            "query": query,
            "results": results,
            "num_results": len(results),
        }
    except Exception as e:
        return {"error": f"search_documents failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════
# TOOL 3: search_images
# ═══════════════════════════════════════════════════════════════

def search_images(query: str, top_k: int = 3) -> dict:
    """
    Find relevant images via multimodal RAG (caption-based search).

    Args:
        query: Natural language description of desired image.
        top_k: Number of images to return (1-5, default 3).

    Returns:
        dict with query, results (list of image filenames, captions, scores).
    """
    try:
        from multimodal_rag import get_image_rag
        rag = get_image_rag()
        top_k = min(max(1, int(top_k)), 5)
        results = rag.search(query, top_k=top_k)
        # Remove full path from results (keep filename and caption only)
        clean_results = []
        for r in results:
            clean_results.append({
                "filename": r.get("filename", ""),
                "caption": r.get("caption", ""),
                "score": r.get("score", 0),
            })
        return {
            "query": query,
            "results": clean_results,
            "num_results": len(clean_results),
        }
    except Exception as e:
        return {"error": f"search_images failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════
# TOOL 4: generate_chart
# ═══════════════════════════════════════════════════════════════

def generate_chart(
    chart_type: str,
    title: str,
    labels: list,
    values: list,
) -> dict:
    """
    Create a bar, line, or pie chart from provided data.

    Args:
        chart_type: One of "bar", "line", "pie".
        title: Chart title.
        labels: List of category labels.
        values: List of numeric values (same length as labels).

    Returns:
        dict with chart_path (saved PNG path) and chart_type.
    """
    try:
        if chart_type not in ("bar", "line", "pie"):
            return {"error": f"Invalid chart_type '{chart_type}'. Use 'bar', 'line', or 'pie'."}
        if len(labels) != len(values):
            return {"error": f"labels ({len(labels)}) and values ({len(values)}) must be same length."}
        if not labels:
            return {"error": "labels and values cannot be empty."}

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63",
                  "#00BCD4", "#FF5722", "#607D8B", "#795548", "#3F51B5"]

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                        str(val), ha="center", fontsize=10, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif chart_type == "line":
            ax.plot(labels, values, marker="o", linewidth=2, color="#2196F3", markersize=8)
            for i, val in enumerate(values):
                ax.text(i, val + max(values) * 0.03, str(val), ha="center", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif chart_type == "pie":
            ax.pie(values, labels=labels, colors=colors[:len(labels)],
                   autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})

        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        # Save chart
        chart_filename = f"chart_{int(time.time())}.png"
        chart_path = os.path.join(CHART_DIR, chart_filename)
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {
            "chart_path": chart_path,
            "chart_filename": chart_filename,
            "chart_type": chart_type,
            "title": title,
            "message": f"Chart saved as {chart_filename}",
        }
    except Exception as e:
        return {"error": f"generate_chart failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════
# TOOL 5: summarize_text
# ═══════════════════════════════════════════════════════════════

def summarize_text(text: str, max_length: int = 150) -> dict:
    """
    Summarize a long text using a HuggingFace LLM.

    Args:
        text: The text to summarize.
        max_length: Approximate max words for the summary (default 150).

    Returns:
        dict with original_length, summary, and summary_length.
    """
    try:
        if not text or len(text.strip()) < 20:
            return {"error": "Text is too short to summarize."}

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a concise summarizer. Summarize the following text clearly and briefly."},
                {"role": "user", "content": f"Summarize this text in about {max_length} words:\n\n{text[:3000]}"},
            ],
            "max_tokens": max_length * 2,
        }

        resp = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers, json=payload, timeout=60,
        )
        if resp.status_code != 200:
            return {"error": f"Summarization API error {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        summary = data["choices"][0]["message"]["content"]

        return {
            "original_length": len(text.split()),
            "summary": summary,
            "summary_length": len(summary.split()),
        }
    except Exception as e:
        return {"error": f"summarize_text failed: {str(e)}"}


# ── Tool Registry ──
TOOL_REGISTRY = {
    "query_database": query_database,
    "search_documents": search_documents,
    "search_images": search_images,
    "generate_chart": generate_chart,
    "summarize_text": summarize_text,
}
