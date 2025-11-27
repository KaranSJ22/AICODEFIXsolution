#!/usr/bin/env python3
"""
db_analyzer.py

AI CODEFIX 2025 - Medium Challenge
----------------------------------
- Connects to a SQLite database
- Discovers schema and analyzes data
- Generates 3+ charts (PNG)
- Builds a PDF report
- Sends an email with summary + PDF + charts

Dependencies (put these in requirements.txt):
    python-dotenv
    pandas
    matplotlib
    seaborn
    reportlab
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import sqlite3
from pathlib import Path
from datetime import datetime
import traceback

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# --------- GLOBAL CONFIG ---------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CHART1_PATH = OUTPUT_DIR / "chart1_table_counts.png"
CHART2_PATH = OUTPUT_DIR / "chart2_trend.png"
CHART3_PATH = OUTPUT_DIR / "chart3_correlation.png"
PDF_REPORT_PATH = OUTPUT_DIR / "report.pdf"

# Seaborn style for professional look
sns.set(style="whitegrid")


# --------- DATABASE EXPLORATION ---------
def connect_db(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database at db_path.
    Raises FileNotFoundError if the file does not exist.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return sqlite3.connect(str(db_file))


def get_tables(conn: sqlite3.Connection):
    """
    Return a list of table names in the SQLite database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = cursor.fetchall()
    tables = [r[0] for r in rows]
    return tables


def get_table_schema(conn: sqlite3.Connection, table_name: str):
    """
    Return schema information for a given table.

    Note: Currently not used in the main flow, but kept for completeness.
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    cols = cursor.fetchall()
    # cols: cid, name, type, notnull, dflt_value, pk
    schema = [
        {
            "cid": c[0],
            "name": c[1],
            "type": c[2],
            "notnull": bool(c[3]),
            "default": c[4],
            "pk": bool(c[5]),
        }
        for c in cols
    ]
    return schema


def load_table_df(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """
    Load a table into a pandas DataFrame.
    Returns an empty DataFrame on failure (with a warning).
    """
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df
    except Exception as e:
        print(f"[WARN] Failed to load table {table_name}: {e}")
        return pd.DataFrame()


def analyze_table(df: pd.DataFrame):
    """
    Return basic stats + data quality metrics for a dataframe.

    Returns a dict with:
      - row_count
      - basic_stats (pandas describe result or None)
      - null_counts (Series or None)
      - duplicate_rows (int)
      - numeric_corr (DataFrame or None)
    """
    if df.empty:
        return {
            "row_count": 0,
            "basic_stats": None,
            "null_counts": None,
            "duplicate_rows": 0,
            "numeric_corr": None,
        }

    row_count = len(df)

    try:
        basic_stats = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        # For older pandas without datetime_is_numeric
        basic_stats = df.describe(include="all")

    null_counts = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()

    numeric_df = df.select_dtypes(include="number")
    corr = None
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

    return {
        "row_count": row_count,
        "basic_stats": basic_stats,
        "null_counts": null_counts,
        "duplicate_rows": duplicate_rows,
        "numeric_corr": corr,
    }


# --------- VISUALIZATIONS ---------
def plot_table_counts(table_stats):
    """
    Chart 1: Bar chart of row counts per table.
    Saves to CHART1_PATH and returns the path, or None if no data.
    """
    if not table_stats:
        print("[INFO] No table stats to plot for chart1.")
        return None

    tables = list(table_stats.keys())
    counts = [table_stats[t]["row_count"] for t in tables]

    if not tables:
        print("[INFO] No tables available for chart1.")
        return None

    plt.figure(figsize=(8, 5))
    sns.barplot(x=tables, y=counts)
    plt.title("Row Counts per Table")
    plt.xlabel("Table")
    plt.ylabel("Row Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CHART1_PATH, dpi=300)
    plt.close()
    return CHART1_PATH


def find_time_series_candidate(table_dfs):
    """
    Find a table/column pair suitable for time series:

      - one datetime-like column (dt_col)
      - one *different* numeric column (num_col)

    Returns: (tname, df, dt_col, num_col) or (None, None, None, None)
    """
    for tname, df in table_dfs.items():
        if df.empty:
            continue

        dt_cols = []
        for col in df.columns:
            # Try to parse column as datetime
            try:
                pd.to_datetime(df[col], errors="raise")
                dt_cols.append(col)
            except Exception:
                continue

        if not dt_cols:
            continue

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        # Ensure numeric columns are not the same as datetime candidates
        numeric_cols = [c for c in numeric_cols if c not in dt_cols]

        if numeric_cols:
            return tname, df, dt_cols[0], numeric_cols[0]

    return None, None, None, None


def plot_trend_chart(table_dfs):
    """
    Chart 2: Time series chart if possible.
    Fallback: line plot of a numeric column vs row index.

    Saves to CHART2_PATH and returns the path, or None if no data.
    """
    # Try time series
    tname, df, dt_col, num_col = find_time_series_candidate(table_dfs)

    if tname is not None:
        print(f"[INFO] Using table '{tname}' for time-series chart.")
        df = df.copy()
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])

        if not df.empty:
            ts = df.groupby(df[dt_col].dt.date)[num_col].sum()
            if not ts.empty:
                plt.figure(figsize=(8, 5))
                plt.plot(ts.index, ts.values, marker="o")
                plt.title(f"Trend of {num_col} over Time ({tname})")
                plt.xlabel("Date")
                plt.ylabel(num_col)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(CHART2_PATH, dpi=300)
                plt.close()
                return CHART2_PATH

    # Fallback: line chart for first numeric column in any table
    for tname, df in table_dfs.items():
        if df.empty:
            continue

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col = num_cols[0]
            subset = df[col].head(50)
            if subset.empty:
                continue

            plt.figure(figsize=(8, 5))
            plt.plot(subset.index, subset.values, marker="o")
            plt.title(f"Trend of {col} by Row Index ({tname})")
            plt.xlabel("Row Index")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(CHART2_PATH, dpi=300)
            plt.close()
            return CHART2_PATH

    print("[INFO] No suitable numeric/time-series data for chart2.")
    return None


def plot_correlation_heatmap(table_stats):
    """
    Chart3: Correlation heatmap from the largest table
    that has >1 numeric column.

    Produces a cleaner plot:
      - shows only lower triangle
      - auto-adjusts figure size
      - disables annotations if too many variables
    """
    # Find table with largest row_count that has corr matrix
    candidate = None
    for tname, stats in table_stats.items():
        corr = stats["numeric_corr"]
        if corr is not None and corr.shape[0] > 1:
            if candidate is None or stats["row_count"] > table_stats[candidate]["row_count"]:
                candidate = tname

    if candidate is None:
        print("[INFO] No numeric correlations found for chart3.")
        return None

    corr = table_stats[candidate]["numeric_corr"]

    # ---- Layout & appearance tweaks ----
    n_vars = corr.shape[0]

    # If there are many variables, don't annotate every cell
    annot = n_vars <= 10

    # Dynamic figure size (cap it so it doesn't explode)
    fig_width = min(1.0 * n_vars + 2, 18)   # inches
    fig_height = min(0.8 * n_vars + 2, 18)  # inches

    plt.figure(figsize=(fig_width, fig_height))

    # Mask upper triangle to reduce clutter
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f" if annot else "",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.6},
        linewidths=0.5,
        linecolor="white"
    )

    plt.title(f"Correlation Heatmap ({candidate})", pad=20)

    # Smaller font & rotated x labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    plt.savefig(CHART3_PATH, dpi=300)
    plt.close()
    return CHART3_PATH

# --------- INSIGHTS GENERATION ---------
def generate_insights(table_stats):
    """
    Generate high-level textual insights from table_stats.

    Returns a list of insight strings.
    """
    insights = []

    if not table_stats:
        insights.append("No tables found in the database.")
        return insights

    total_records = sum(s["row_count"] for s in table_stats.values())
    if total_records == 0:
        insights.append("Tables are present but contain no records.")
        return insights

    # Largest table
    largest_table = max(table_stats.items(), key=lambda kv: kv[1]["row_count"])[0]
    largest_count = table_stats[largest_table]["row_count"]
    proportion = largest_count / max(total_records, 1)

    insights.append(
        (
            f"Largest table is '{largest_table}' with {largest_count} records, "
            f"representing approximately {proportion:.0%} of all rows."
        )
    )

    # Missing data
    max_nulls = 0
    max_null_table = None
    max_null_col = None
    for tname, stats in table_stats.items():
        nulls = stats["null_counts"]
        if nulls is not None:
            for col, cnt in nulls.items():
                if cnt > max_nulls:
                    max_nulls = cnt
                    max_null_table = tname
                    max_null_col = col

    if max_nulls > 0 and max_null_table is not None and max_null_col is not None:
        insights.append(
            (
                f"Highest missing data observed in '{max_null_table}.{max_null_col}' "
                f"with {max_nulls} null values, indicating a potential data quality issue."
            )
        )

    # Strong correlations
    best_corr_val = 0
    best_corr_pair = None
    best_corr_table = None
    for tname, stats in table_stats.items():
        corr = stats["numeric_corr"]
        if corr is not None:
            for c1 in corr.columns:
                for c2 in corr.columns:
                    # Avoid duplicate pairs and diagonal
                    if c1 >= c2:
                        continue
                    val = abs(corr.loc[c1, c2])
                    if val > best_corr_val:
                        best_corr_val = val
                        best_corr_pair = (c1, c2)
                        best_corr_table = tname

    if best_corr_pair and best_corr_val >= 0.5:
        c1, c2 = best_corr_pair
        insights.append(
            (
                f"Strong relationship detected between '{c1}' and '{c2}' in table '{best_corr_table}' "
                f"(correlation ≈ {best_corr_val:.2f}), which may be useful for predictive modeling."
            )
        )

    # Duplicates
    total_dups = sum(s["duplicate_rows"] for s in table_stats.values())
    if total_dups > 0:
        insights.append(
            (
                f"Detected {total_dups} duplicated rows across all tables, suggesting an opportunity "
                f"for data deduplication."
            )
        )

    if not insights:
        insights.append("Data appears clean with no major issues or strong correlations detected.")

    return insights


# --------- PDF REPORT GENERATION ---------
def wrap_text(text, max_width=90):
    """
    Simple word-wrapping helper for report text.
    max_width is approximate character length per line.
    """
    words = text.split()
    lines = []
    current = []
    length = 0

    for w in words:
        extra = len(w) + (1 if current else 0)
        if length + extra > max_width:
            lines.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
            length += extra

    if current:
        lines.append(" ".join(current))

    return lines


def create_pdf_report(team_name, db_path, tables, table_stats, insights):
    """
    Generate a PDF report summarizing the analysis and charts.
    Saves to PDF_REPORT_PATH and returns the path.
    """
    c = canvas.Canvas(str(PDF_REPORT_PATH), pagesize=A4)
    width, height = A4

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Page 1: Executive summary
    y = height - 50
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, f"Database Analysis Report - {team_name}")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Database: {db_path}")
    y -= 15
    c.drawString(40, y, f"Analysis Date: {now_str}")
    y -= 30

    # Executive summary section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "1. Executive Summary")
    y -= 20
    c.setFont("Helvetica", 11)

    total_tables = len(tables)
    total_records = sum(s["row_count"] for s in table_stats.values()) if table_stats else 0
    largest_table = (
        max(table_stats, key=lambda t: table_stats[t]["row_count"])
        if table_stats
        else "N/A"
    )

    summary_lines = [
        f"- Total Tables: {total_tables}",
        f"- Total Records: {total_records}",
        f"- Largest Table: {largest_table}",
    ]

    for line in summary_lines:
        c.drawString(50, y, line)
        y -= 15

    # Key insights section
    y -= 10
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "2. Key Insights")
    y -= 20
    c.setFont("Helvetica", 11)

    if not insights:
        insights = ["No key insights generated."]

    for i, ins in enumerate(insights, start=1):
        wrapped = wrap_text(ins, max_width=90)
        for idx, wline in enumerate(wrapped):
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
            prefix = f"{i}. " if idx == 0 else "   "
            c.drawString(50, y, prefix + wline)
            y -= 15

    # New page for charts
    c.showPage()
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "3. Visualizations")
    y -= 30

    def draw_chart(path, title):
        nonlocal y
        if not path or not Path(path).exists():
            return

        if y < 250:  # Ensure space for image
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, "Visualizations (contd.)")
            y -= 30

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, title)
        y -= 20

        img = ImageReader(str(path))
        img_width, img_height = img.getSize()
        scale = min((width - 80) / img_width, 300 / img_height)
        disp_w = img_width * scale
        disp_h = img_height * scale

        c.drawImage(img, 40, y - disp_h, width=disp_w, height=disp_h)
        y -= disp_h + 40

    draw_chart(CHART1_PATH, "Chart 1: Row Counts per Table")
    draw_chart(CHART2_PATH, "Chart 2: Trend / Time Series")
    draw_chart(CHART3_PATH, "Chart 3: Correlation Heatmap")

    c.save()
    print(f"[INFO] PDF report generated at: {PDF_REPORT_PATH}")
    return PDF_REPORT_PATH


# --------- EMAIL SENDING ---------
def send_email(recipient, team_name, db_summary, insights, attachments):
    """
    Send an email with text body and attachments.

    SMTP config must be provided via environment variables:
      SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print("[WARN] SMTP_USER or SMTP_PASSWORD missing. Skipping email sending.")
        return

    subject = f"Database Analysis Report - {team_name}"

    body_lines = [
        "Dear Recipient,\n",
        "Please find the automated database analysis report below.\n",
        "=== DATABASE SUMMARY ===",
        f"- Total Tables: {db_summary['total_tables']}",
        f"- Total Records: {db_summary['total_records']}",
        f"- Analysis Date: {db_summary['analysis_date']}",
        "",
        "=== KEY INSIGHTS ===",
    ]

    for i, ins in enumerate(insights[:5], start=1):
        body_lines.append(f"{i}. {ins}")

    body_lines += [
        "",
        "PDF report with charts has been attached.",
        "",
        "Best regards,",
        f"{team_name}",
        "AI CODEFIX 2025",
    ]

    body_text = "\n".join(body_lines)

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body_text, "plain"))

    # Attach files (PDF + charts)
    for fpath in attachments:
        if not fpath or not Path(fpath).exists():
            continue

        filename = Path(fpath).name
        with open(fpath, "rb") as f:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                part = MIMEImage(f.read())
            else:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)

            part.add_header("Content-Disposition", f"attachment; filename={filename}")
            msg.attach(part)

    print("[INFO] Sending email...")
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

    print(f"[INFO] Email sent to {recipient}.")


# --------- MAIN PIPELINE ---------
def main():
    """
    Main pipeline:
      - Parse arguments
      - Connect to DB
      - Analyze tables
      - Generate charts and PDF
      - Send email with summary + attachments
    """
    parser = argparse.ArgumentParser(
        description="Automatic Database Insights & Email Agent"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to SQLite database file (e.g., data.db)",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Recipient email address",
    )
    parser.add_argument(
        "--team-name",
        default="AI Team",
        help="Your team name for report & subject",
    )
    args = parser.parse_args()

    db_path = args.db
    recipient = args.email
    team_name = args.team_name

    print(f"[INFO] Connecting to database: {db_path}")
    try:
        conn = connect_db(db_path)
    except Exception as e:
        print(f"[ERROR] Could not connect to database: {e}")
        return

    try:
        # Discover tables
        tables = get_tables(conn)
        print(f"[INFO] Found tables: {tables}")

        table_dfs = {}
        table_stats = {}

        # Load and analyze each table
        for t in tables:
            print(f"[INFO] Loading and analyzing table: {t}")
            df = load_table_df(conn, t)
            table_dfs[t] = df
            stats = analyze_table(df)
            table_stats[t] = stats

        conn.close()

        # Generate charts
        print("[INFO] Generating charts...")
        chart1 = plot_table_counts(table_stats)
        chart2 = plot_trend_chart(table_dfs)
        chart3 = plot_correlation_heatmap(table_stats)

        # Generate insights
        print("[INFO] Generating insights...")
        insights = generate_insights(table_stats)

        # Create PDF report
        print("[INFO] Creating PDF report...")
        pdf_path = create_pdf_report(
            team_name, db_path, tables, table_stats, insights
        )

        # Prepare email summary
        total_tables = len(tables)
        total_records = sum(s["row_count"] for s in table_stats.values())
        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db_summary = {
            "total_tables": total_tables,
            "total_records": total_records,
            "analysis_date": analysis_date,
        }

        # Send email (if SMTP config available)
        attachments = [pdf_path, chart1, chart2, chart3]
        send_email(recipient, team_name, db_summary, insights, attachments)

        print("[INFO] Done. All outputs saved in 'output/' directory.")

    except Exception as e:
        print("[ERROR] Unexpected error:")
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()

