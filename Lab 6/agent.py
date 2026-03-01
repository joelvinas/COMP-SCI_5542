"""
agent.py — AI Agent with iterative tool-calling via HuggingFace Inference API
==============================================================================
Uses a HuggingFace-hosted LLM (Qwen2.5) with function-calling to interpret
user questions, decide which Snowflake tools to invoke, perform multi-step
reasoning, and produce a final natural-language answer.

Run standalone:
    python agent.py "Show me all events"
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools import TOOL_REGISTRY
from tool_schemas import TOOL_SCHEMAS

# ── Config ──
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
API_URL = "https://router.huggingface.co/v1/chat/completions"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a helpful Snowflake Data Assistant for the CS 5542 Week 5 project.

You have access to these 5 tools:

1. **query_database** — Run SQL queries against Snowflake tables (EVENTS and USERS).
   - EVENTS: EVENT_ID, EVENT_TIME, TEAM, CATEGORY, VALUE
   - USERS: USER_ID, TEAM, ROLE, CREATED_AT

2. **search_documents** — Search through PDF documents using RAG.

3. **search_images** — Find relevant images (diagrams, charts, dashboards) via multimodal RAG.

4. **generate_chart** — Create bar, line, or pie charts from data.

5. **summarize_text** — Summarize long text using an LLM.

CRITICAL RULES:
- ALWAYS call a tool before answering. NEVER answer from your own knowledge.
- ONLY use information that comes directly from tool results. Quote or paraphrase the retrieved text.
- NEVER fabricate numbers, statistics, scores, or percentages that are not in the tool results.
- If a tool returns no useful results, say so honestly. Do not guess.
- When reporting document content, attribute it clearly (e.g., "According to the document...").
- For data visualization, query data FIRST with query_database, then use generate_chart.
- For knowledge questions, ALWAYS use search_documents first.
- For image requests, use search_images.
"""


def _call_llm(messages: list, tools: list) -> dict:
    """Make a direct HTTP POST to the HuggingFace Inference API."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1024,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HF API error {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def run_agent(user_query: str, chat_history: list = None) -> dict:
    """
    Run the AI agent loop: interpret the user query, call tools as needed,
    and return a final answer with reasoning steps.

    Args:
        user_query: The user's natural-language question.
        chat_history: Optional list of prior messages (dicts with "role" and "content").

    Returns:
        dict with keys:
            "answer" (str): The agent's final response.
            "steps" (list[dict]): Each reasoning step with tool calls and results.
    """
    # Build message list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_query})

    steps = []

    for iteration in range(MAX_ITERATIONS):
        try:
            data = _call_llm(messages, TOOL_SCHEMAS)
        except Exception as e:
            return {
                "answer": f"⚠️ LLM call failed: {str(e)}",
                "steps": steps,
            }

        choice = data["choices"][0]
        message = choice["message"]

        # Check if the model wants to call tools
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # Add assistant message with tool calls to history
            messages.append(message)

            # Execute each tool call
            for tc in tool_calls:
                func = tc["function"]
                tool_name = func["name"]
                try:
                    tool_args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]
                except json.JSONDecodeError:
                    tool_args = {}

                step = {
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "args": tool_args,
                }

                # Call the tool
                if tool_name in TOOL_REGISTRY:
                    try:
                        result = TOOL_REGISTRY[tool_name](**tool_args)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {str(e)}"}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                step["result_summary"] = (
                    f"{result.get('row_count', '?')} rows"
                    if isinstance(result, dict) and "row_count" in result
                    else f"{result.get('num_results', '?')} results"
                    if isinstance(result, dict) and "num_results" in result
                    else f"chart: {result.get('chart_filename', '')}"
                    if isinstance(result, dict) and "chart_path" in result
                    else "completed" if isinstance(result, dict) and "error" not in result
                    else result.get("error", "unknown") if isinstance(result, dict)
                    else "unknown"
                )

                # Attach extra data for UI rendering
                extra = {}
                if isinstance(result, dict):
                    if "chart_path" in result:
                        extra["chart_path"] = result["chart_path"]
                    if tool_name == "search_images" and "results" in result:
                        extra["image_results"] = result["results"]
                step["extra"] = extra

                steps.append(step)

                # Truncate large results for the LLM context
                result_str = json.dumps(result, default=str)
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + '... [truncated]"}'

                # Add tool result to message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        else:
            # No tool calls — the model produced a final answer
            return {
                "answer": message.get("content", "I couldn't generate a response."),
                "steps": steps,
            }

    # Max iterations reached
    return {
        "answer": message.get("content", "I reached the maximum number of reasoning steps."),
        "steps": steps,
    }


# ── CLI entry point ──
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"🤖 Agent processing: {query}\n")
    result = run_agent(query)
    print(f"📝 Answer:\n{result['answer']}\n")
    if result["steps"]:
        print("🔧 Tool calls:")
        for s in result["steps"]:
            print(f"  Step {s['iteration']}: {s['tool']}({s['args']}) → {s['result_summary']}")
