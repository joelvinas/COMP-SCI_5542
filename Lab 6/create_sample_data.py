"""
create_sample_data.py — Generate sample PDFs and images for RAG pipelines
==========================================================================
Creates 5 PDF documents and 5 images with captions in data/documents/
and data/images/ respectively.

Run once:
    python create_sample_data.py
"""

import os
import json
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Directories ──
DOC_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
IMG_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# PDF GENERATION
# ═══════════════════════════════════════════════════════════════

PDF_CONTENT = {
    "snowflake_overview.pdf": {
        "title": "Snowflake Cloud Data Platform Overview",
        "body": (
            "Snowflake is a cloud-based data warehousing platform that provides data storage, "
            "processing, and analytic solutions. It runs on AWS, Azure, and Google Cloud.\n\n"
            "Key Features:\n"
            "- Separation of storage and compute for independent scaling\n"
            "- Automatic clustering and micro-partitioning for query optimization\n"
            "- Support for structured and semi-structured data (JSON, Avro, Parquet)\n"
            "- Time Travel for accessing historical data up to 90 days\n"
            "- Zero-copy cloning for instant database copies without extra storage\n"
            "- Role-based access control (RBAC) for fine-grained security\n\n"
            "Architecture:\n"
            "Snowflake uses a unique multi-cluster shared data architecture. The storage layer "
            "holds all data in a columnar format. The compute layer consists of virtual warehouses "
            "that independently process queries. The cloud services layer manages authentication, "
            "metadata, query parsing, and optimization.\n\n"
            "Virtual Warehouses:\n"
            "Virtual warehouses are compute clusters that execute SQL queries. They can be resized "
            "(XS to 6XL) and auto-suspended when idle. Multiple warehouses can query the same data "
            "concurrently without contention. This makes Snowflake ideal for mixed workloads."
        ),
    },
    "data_pipeline_guide.pdf": {
        "title": "Data Pipeline Best Practices Guide",
        "body": (
            "A data pipeline is a series of steps that move data from source systems to a "
            "target destination, transforming it along the way.\n\n"
            "Pipeline Types:\n"
            "1. ETL (Extract, Transform, Load) - transform data before loading\n"
            "2. ELT (Extract, Load, Transform) - load raw data, transform in-place\n"
            "3. Streaming pipelines - process data in real-time\n\n"
            "Best Practices:\n"
            "- Idempotency: running the pipeline twice produces the same result\n"
            "- Monitoring: track latency, row counts, and error rates\n"
            "- Schema evolution: handle new columns gracefully\n"
            "- Data validation: check constraints after each stage\n"
            "- Incremental loading: process only new or changed records\n\n"
            "Snowflake-specific tips:\n"
            "- Use COPY INTO for bulk loading from stages\n"
            "- Use Snowpipe for continuous micro-batch ingestion\n"
            "- Use Tasks and Streams for change data capture (CDC)\n"
            "- Monitor with QUERY_HISTORY and WAREHOUSE_METERING_HISTORY views"
        ),
    },
    "sql_analytics_reference.pdf": {
        "title": "SQL Analytical Query Patterns Reference",
        "body": (
            "This reference covers common SQL analytical patterns used in data warehousing.\n\n"
            "1. Aggregation Queries:\n"
            "SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary\n"
            "FROM employees GROUP BY department ORDER BY avg_salary DESC;\n\n"
            "2. Window Functions:\n"
            "SELECT name, salary, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rank\n"
            "FROM employees;\n\n"
            "3. Common Table Expressions (CTEs):\n"
            "WITH monthly_sales AS (\n"
            "  SELECT DATE_TRUNC('month', sale_date) as month, SUM(amount) as total\n"
            "  FROM sales GROUP BY 1\n"
            ") SELECT month, total, LAG(total) OVER (ORDER BY month) as prev_month FROM monthly_sales;\n\n"
            "4. Pivot Tables:\n"
            "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1','Q2','Q3','Q4'));\n\n"
            "5. Time-Series Analysis:\n"
            "Use DATEADD, DATEDIFF, and DATE_TRUNC for time-based calculations.\n"
            "Use QUALIFY with ROW_NUMBER() for deduplication."
        ),
    },
    "team_performance_report.pdf": {
        "title": "Team Performance Analysis Report",
        "body": (
            "Quarterly Performance Summary for Q1 2026\n\n"
            "Team A (Developers + Analysts):\n"
            "- Total events generated: 2,450\n"
            "- Primary categories: search (45%), upload (30%), analysis (25%)\n"
            "- Average event value: 12.8\n"
            "- Peak activity: Tuesdays and Thursdays\n"
            "- Recommendation: Increase analysis capacity\n\n"
            "Team B (Developers):\n"
            "- Total events generated: 1,890\n"
            "- Primary categories: search (60%), upload (40%)\n"
            "- Average event value: 9.7\n"
            "- Peak activity: Mondays\n"
            "- Recommendation: Diversify event categories\n\n"
            "Team C (Managers):\n"
            "- Total events generated: 980\n"
            "- Primary categories: analysis (70%), search (30%)\n"
            "- Average event value: 20.0\n"
            "- Peak activity: Fridays\n"
            "- Recommendation: Higher value events, maintain quality\n\n"
            "Overall trends: Total pipeline throughput increased 15% over last quarter. "
            "Data quality score: 97.2%. Average query latency: 245ms."
        ),
    },
    "data_quality_guidelines.pdf": {
        "title": "Data Quality Monitoring Guidelines",
        "body": (
            "Data quality is critical for trustworthy analytics. This document outlines "
            "key dimensions and monitoring practices.\n\n"
            "Six Dimensions of Data Quality:\n"
            "1. Completeness - are all required fields populated?\n"
            "2. Accuracy - does the data reflect reality?\n"
            "3. Consistency - is data consistent across systems?\n"
            "4. Timeliness - is data available when needed?\n"
            "5. Validity - does data conform to business rules?\n"
            "6. Uniqueness - are there unwanted duplicates?\n\n"
            "Monitoring Strategies:\n"
            "- Row count checks after each load\n"
            "- NULL percentage thresholds per column\n"
            "- Referential integrity validation\n"
            "- Statistical distribution monitoring (z-score anomaly detection)\n"
            "- Schema drift detection\n\n"
            "Snowflake Tools:\n"
            "- Use INFORMATION_SCHEMA for metadata validation\n"
            "- Use RESULT_SCAN() to inspect recent query results\n"
            "- Set up Alerts for automated threshold monitoring\n"
            "- Use DATA_QUALITY_MONITORING_RESULTS (if enabled)"
        ),
    },
}


def create_pdfs():
    """Generate 5 PDF files."""
    for filename, content in PDF_CONTENT.items():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, content["title"], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, content["body"])
        path = os.path.join(DOC_DIR, filename)
        pdf.output(path)
        print(f"  Created: {path}")


# ═══════════════════════════════════════════════════════════════
# IMAGE GENERATION
# ═══════════════════════════════════════════════════════════════

def create_images():
    """Generate 5 data-themed images with matplotlib."""
    captions = {}

    # 1. Data pipeline diagram
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    stages = ["Extract", "Transform", "Load", "Analyze", "Visualize"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63"]
    for i, (stage, color) in enumerate(zip(stages, colors)):
        ax.barh(0, 1, left=i, color=color, edgecolor="white", height=0.5)
        ax.text(i + 0.5, 0, stage, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.set_xlim(0, 5)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    ax.set_title("Data Pipeline Architecture", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "data_pipeline_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    captions["data_pipeline_diagram.png"] = "Data pipeline architecture diagram showing Extract Transform Load Analyze and Visualize stages"
    print(f"  Created: {path}")

    # 2. Team performance chart
    fig, ax = plt.subplots(figsize=(8, 5))
    teams = ["Team A", "Team B", "Team C"]
    events = [2450, 1890, 980]
    bars = ax.bar(teams, events, color=["#4CAF50", "#2196F3", "#FF9800"], edgecolor="white", width=0.6)
    for bar, val in zip(bars, events):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(val), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Events")
    ax.set_title("Team Performance — Total Events (Q1 2026)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 3000)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "team_performance_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    captions["team_performance_chart.png"] = "Bar chart showing team performance metrics with Team A leading at 2450 events followed by Team B and Team C"
    print(f"  Created: {path}")

    # 3. Snowflake architecture
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = ["Cloud Services\n(Auth, Metadata, Optimizer)", "Compute\n(Virtual Warehouses)", "Storage\n(Micro-partitions)"]
    layer_colors = ["#29B6F6", "#0288D1", "#01579B"]
    for i, (layer, color) in enumerate(zip(layers, layer_colors)):
        y = 2 - i
        rect = plt.Rectangle((0.5, y - 0.35), 5, 0.7, facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(3, y, layer, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 3)
    ax.axis("off")
    ax.set_title("Snowflake Cloud Data Platform Architecture", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "snowflake_architecture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    captions["snowflake_architecture.png"] = "Snowflake cloud data platform three layer architecture showing Cloud Services Compute and Storage layers"
    print(f"  Created: {path}")

    # 4. Analytics dashboard
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # KPI card simulation
    kpis = [("Total Events", "5,320"), ("Avg Latency", "245ms"), ("Data Quality", "97.2%"), ("Active Users", "12")]
    kpi_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for ax_i, (label, value), color in zip(axes.flat, kpis, kpi_colors):
        ax_i.text(0.5, 0.6, value, ha="center", va="center", fontsize=24, fontweight="bold", color=color, transform=ax_i.transAxes)
        ax_i.text(0.5, 0.25, label, ha="center", va="center", fontsize=12, color="#666", transform=ax_i.transAxes)
        ax_i.set_xlim(0, 1)
        ax_i.set_ylim(0, 1)
        ax_i.axis("off")
        for spine in ax_i.spines.values():
            spine.set_visible(True)
            spine.set_color("#ddd")
    fig.suptitle("Analytics Dashboard — Key Performance Indicators", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "analytics_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    captions["analytics_dashboard.png"] = "Analytics dashboard showing key performance indicators including total events average latency data quality score and active users"
    print(f"  Created: {path}")

    # 5. Data quality report
    fig, ax = plt.subplots(figsize=(8, 5))
    dimensions = ["Completeness", "Accuracy", "Consistency", "Timeliness", "Validity", "Uniqueness"]
    scores = [98.5, 96.2, 97.8, 94.1, 99.0, 97.5]
    colors_dq = ["#4CAF50" if s >= 97 else "#FF9800" if s >= 95 else "#F44336" for s in scores]
    bars = ax.barh(dimensions, scores, color=colors_dq, edgecolor="white", height=0.6)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{score}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(90, 101)
    ax.set_title("Data Quality Report — Six Dimensions", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "data_quality_report.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    captions["data_quality_report.png"] = "Data quality monitoring report showing scores across six dimensions completeness accuracy consistency timeliness validity and uniqueness"
    print(f"  Created: {path}")

    # Save captions
    captions_path = os.path.join(IMG_DIR, "captions.json")
    with open(captions_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"  Created: {captions_path}")


if __name__ == "__main__":
    print("Creating PDF documents...")
    create_pdfs()
    print("\nCreating images...")
    create_images()
    print("\nDone! All sample data created.")
