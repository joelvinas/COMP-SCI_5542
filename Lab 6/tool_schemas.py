"""
tool_schemas.py — Function-calling schemas for the 5 agent tools
=================================================================
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run a SQL query against Snowflake and return results as a table. "
                "Use this to retrieve raw data from the EVENTS or USERS table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table to query. Must be 'EVENTS' or 'USERS'.",
                        "enum": ["EVENTS", "USERS"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (1-500, default 100).",
                        "default": 100,
                    },
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Retrieve relevant documents using the RAG pipeline. "
                "Searches through PDF documents about Snowflake, data pipelines, "
                "SQL analytics, team performance, and data quality. "
                "Use this when the user asks conceptual or knowledge questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of document chunks to return (1-10, default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": (
                "Find relevant images via multimodal RAG. "
                "Searches through data visualization images including pipeline diagrams, "
                "performance charts, architecture diagrams, dashboards, and quality reports. "
                "Use this when the user asks for visual content or diagrams."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of desired image.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of images to return (1-5, default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": (
                "Create a bar, line, or pie chart from data and save as PNG. "
                "Use this after querying data to visualize results. "
                "Provide labels and numeric values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "description": "Type of chart to create.",
                        "enum": ["bar", "line", "pie"],
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Category labels for the chart.",
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numeric values for each label.",
                    },
                },
                "required": ["chart_type", "title", "labels", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": (
                "Summarize a long text using an LLM. "
                "Use this to create concise summaries of retrieved documents, "
                "query results, or any other long text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize.",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Approximate max words for the summary (default 150).",
                        "default": 150,
                    },
                },
                "required": ["text"],
            },
        },
    },
]
