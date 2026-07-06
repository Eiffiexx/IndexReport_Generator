"""
    python3 main.py \
        --input "/Users/brain/Downloads/NHIndexMonthly.csv" \
        --output "index_report.pdf"
"""


import argparse
import math
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


REQUIRED_COLUMNS = {"date", "ror", "type", "cnt", "aumbn"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)
MARGIN = 0.38 * inch

NAVY = colors.HexColor("#174A63")
DARK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#667085")
GRID = colors.HexColor("#D9DEE5")
PANEL = colors.HexColor("#F6F8FA")
GREEN = colors.HexColor("#1F8A5B")
RED = colors.HexColor("#B42318")
ORANGE = colors.HexColor("#FF4B00")

PALETTE = [
    "#174A63", "#C05A2B", "#33865C", "#7A4A9E", "#D09A2D",
    "#477FB7", "#9C3B3B", "#4D8D91", "#5A5F6B", "#8E6A35",
]

STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(
    name="Tiny",
    parent=STYLES["Normal"],
    fontName="Helvetica",
    fontSize=6.2,
    leading=7.1,
    textColor=DARK,
))
STYLES.add(ParagraphStyle(
    name="TinyRight",
    parent=STYLES["Tiny"],
    alignment=2,
))


def load_index_data(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the monthly index dataset."""
    df = pd.read_csv(input_path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["type"] = df["type"].astype(str).str.strip()
    df = df.dropna(subset=["date", "type"])
    df = df.sort_values(["type", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def cumulative_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return math.nan
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return math.nan
    years = len(returns) / 12
    return float((1 + returns).prod() ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(12))


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return math.nan
    cumulative_index = (1 + returns).cumprod()
    drawdowns = cumulative_index / cumulative_index.cummax() - 1
    return float(drawdowns.min())


def format_percent(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def format_number(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "-"
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def build_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create one summary row per index."""
    latest_date = df["date"].max()
    rows = []

    for index_name, group in df.groupby("type", sort=True):
        group = group.sort_values("date")
        latest_row = group[group["date"] == latest_date].tail(1)
        january_row = group[(group["year"] == latest_date.year) & (group["month"] == 1)].tail(1)

        ytd_returns = group[group["year"] == latest_date.year]["ror"]
        trailing_12m_returns = group[group["date"] > latest_date - pd.DateOffset(months=12)]["ror"]
        all_returns = group["ror"]

        ann_return = annualized_return(all_returns)
        ann_vol = annualized_volatility(all_returns)
        latest_obs = latest_row["cnt"].iloc[0] if not latest_row.empty else math.nan
        january_obs = january_row["cnt"].iloc[0] if not january_row.empty else math.nan

        rows.append({
            "index": index_name,
            "start_date": group["date"].min(),
            "latest_date": latest_date,
            "latest_month_return": latest_row["ror"].iloc[0] if not latest_row.empty else math.nan,
            "ytd_return": cumulative_return(ytd_returns),
            "trailing_12m_return": cumulative_return(trailing_12m_returns),
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_simple": ann_return / ann_vol if ann_vol and not pd.isna(ann_vol) else math.nan,
            "max_drawdown": max_drawdown(all_returns),
            "latest_observations": latest_obs,
            "january_observations": january_obs,
            "observation_change_ytd": latest_obs - january_obs
            if not pd.isna(latest_obs) and not pd.isna(january_obs) else math.nan,
            "latest_aum_bn": latest_row["aumbn"].iloc[0] if not latest_row.empty else math.nan,
            "months_in_history": len(group),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("ytd_return", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank_ytd", range(1, len(summary) + 1))
    return summary


def build_monthly_return_table(group: pd.DataFrame) -> list[list[str]]:
    """Build a calendar-year monthly return table for one index."""
    rows = [["Year"] + MONTHS + ["YTD", "Avg Obs"]]

    for year in sorted(group["year"].unique(), reverse=True):
        year_group = group[group["year"] == year].sort_values("month")
        row = [str(year)]

        for month_number in range(1, 13):
            month_return = year_group.loc[year_group["month"] == month_number, "ror"]
            row.append(format_percent(month_return.iloc[0]) if not month_return.empty else "-")

        row.append(format_percent(cumulative_return(year_group["ror"])))
        row.append(format_number(year_group["cnt"].mean()))
        rows.append(row)

    return rows


def cumulative_index(group: pd.DataFrame) -> pd.Series:
    return (1 + group.sort_values("date")["ror"]).cumprod() * 100


def draw_header(pdf: canvas.Canvas, title: str, subtitle: str, label: str) -> None:
    pdf.setFillColor(colors.white)
    pdf.rect(0, PAGE_HEIGHT - 0.48 * inch, PAGE_WIDTH, 0.48 * inch, stroke=0, fill=1)
    pdf.setStrokeColor(colors.HexColor("#E5E7EB"))
    pdf.line(0, PAGE_HEIGHT - 0.48 * inch, PAGE_WIDTH, PAGE_HEIGHT - 0.48 * inch)

    logo_x = MARGIN
    logo_y = PAGE_HEIGHT - 0.34 * inch
    pdf.setStrokeColor(DARK)
    pdf.setLineWidth(0.8)
    pdf.rect(logo_x, logo_y - 0.12 * inch, 0.22 * inch, 0.22 * inch, stroke=1, fill=0)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawCentredString(logo_x + 0.11 * inch, logo_y - 0.055 * inch, "N")

    wordmark_x = logo_x + 0.31 * inch
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.black)
    nilsson_text = "NILSSON"
    pdf.drawString(wordmark_x, logo_y - 0.05 * inch, nilsson_text)
    pdf.setFillColor(ORANGE)
    hedge_x = wordmark_x + pdf.stringWidth(nilsson_text, "Helvetica-Bold", 11) + 0.03 * inch
    pdf.drawString(hedge_x, logo_y - 0.05 * inch, "HEDGE")

    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.28 * inch, label.upper())

    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 0.72 * inch, title)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 0.91 * inch, subtitle)


def draw_footer(pdf: canvas.Canvas, page_number: int, latest_date: pd.Timestamp) -> None:
    pdf.setStrokeColor(GRID)
    pdf.line(MARGIN, 0.28 * inch, PAGE_WIDTH - MARGIN, 0.28 * inch)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.5)
    pdf.drawString(
        MARGIN,
        0.16 * inch,
        f"Source: monthly index dataset. Returns are monthly. Observation counts represent reporting coverage. Data through {latest_date:%B %Y}.",
    )
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 0.16 * inch, f"Page {page_number}")


def draw_card(pdf: canvas.Canvas, x: float, y: float, width: float, height: float,
              label: str, value: str, accent=NAVY) -> None:
    pdf.setFillColor(PANEL)
    pdf.setStrokeColor(colors.HexColor("#E7EAEE"))
    pdf.roundRect(x, y, width, height, 4, stroke=1, fill=1)
    pdf.setFillColor(accent)
    pdf.rect(x, y + height - 0.06 * inch, width, 0.06 * inch, stroke=0, fill=1)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.8)
    pdf.drawString(x + 0.08 * inch, y + height - 0.21 * inch, label.upper())
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(x + 0.08 * inch, y + 0.15 * inch, value[:28])


def draw_section_label(pdf: canvas.Canvas, x: float, y: float, text: str) -> None:
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(x, y, text)


def draw_line_chart(pdf: canvas.Canvas, x: float, y: float, width: float, height: float,
                    series: list[tuple[str, list[float]]], title: str) -> None:
    """Draw a simple line chart directly in the PDF."""
    draw_section_label(pdf, x, y + height + 0.12 * inch, title)

    all_values = []
    for _, values in series:
        all_values.extend([v for v in values if not pd.isna(v)])
    if not all_values:
        return

    low, high = min(all_values), max(all_values)
    padding = (high - low) * 0.08 if high != low else max(abs(high) * 0.1, 1)
    low -= padding
    high += padding

    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.4)
    for i in range(5):
        yy = y + height * i / 4
        pdf.line(x, yy, x + width, yy)

    def scale_x(index: int, count: int) -> float:
        return x + width * index / max(count - 1, 1)

    def scale_y(value: float) -> float:
        return y + (value - low) / (high - low) * height

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.2)
    for i in range(5):
        value = low + (high - low) * i / 4
        pdf.drawRightString(x - 0.05 * inch, y + height * i / 4 - 2, f"{value:.0f}")

    for series_number, (name, values) in enumerate(series):
        color = colors.HexColor(PALETTE[series_number % len(PALETTE)])
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.3 if series_number == 0 else 0.9)
        last_point = None

        for index, value in enumerate(values):
            if pd.isna(value):
                last_point = None
                continue
            point = (scale_x(index, len(values)), scale_y(value))
            if last_point:
                pdf.line(last_point[0], last_point[1], point[0], point[1])
            last_point = point

        legend_x = x + width - 1.25 * inch
        legend_y = y + height - series_number * 0.13 * inch
        pdf.setFillColor(color)
        pdf.rect(legend_x, legend_y, 0.06 * inch, 0.06 * inch, stroke=0, fill=1)
        pdf.setFillColor(DARK)
        pdf.setFont("Helvetica", 6.3)
        pdf.drawString(legend_x + 0.09 * inch, legend_y - 1, name[:22])

    pdf.setStrokeColor(DARK)
    pdf.setLineWidth(0.6)
    pdf.line(x, y, x + width, y)
    pdf.line(x, y, x, y + height)


def paragraph_cell(value: str, right_align: bool = False) -> Paragraph:
    style = STYLES["TinyRight"] if right_align else STYLES["Tiny"]
    return Paragraph(str(value), style)


def draw_table(pdf: canvas.Canvas, rows: list[list[str]], x: float, top_y: float,
               column_widths: list[float], row_height: float) -> float:
    formatted_rows = []
    for row_index, row in enumerate(rows):
        formatted_rows.append([
            paragraph_cell(cell, right_align=col_index != 1 and row_index != 0)
            for col_index, cell in enumerate(row)
        ])

    table = Table(
        formatted_rows,
        colWidths=column_widths,
        rowHeights=[row_height] * len(formatted_rows),
    )

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDE2E7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]

    for row_index in range(1, len(formatted_rows)):
        if row_index % 2 == 0:
            style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#FAFBFC")))

    table.setStyle(TableStyle(style))
    table_height = row_height * len(formatted_rows)
    table.wrapOn(pdf, sum(column_widths), table_height)
    table.drawOn(pdf, x, top_y - table_height)
    return table_height


def summary_table_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = [[
        "Rank", "Index", "Latest", "YTD", "12M", "Ann.", "Vol.",
        "Obs", "Obs Chg", "AUM bn",
    ]]

    for _, row in summary.iterrows():
        rows.append([
            str(int(row["rank_ytd"])),
            row["index"],
            format_percent(row["latest_month_return"]),
            format_percent(row["ytd_return"]),
            format_percent(row["trailing_12m_return"]),
            format_percent(row["annualized_return"]),
            format_percent(row["annualized_volatility"]),
            format_number(row["latest_observations"]),
            format_number(row["observation_change_ytd"]),
            format_number(row["latest_aum_bn"], 1),
        ])

    return rows


def draw_summary_page(pdf: canvas.Canvas, summary: pd.DataFrame,
                      latest_date: pd.Timestamp, page_number: int) -> None:
    draw_header(
        pdf,
        "Index Summary Dashboard",
        f"All indices ranked by year-to-date return through {latest_date:%B %Y}",
        "Summary",
    )

    best = summary.sort_values("ytd_return", ascending=False).iloc[0]
    worst = summary.sort_values("ytd_return", ascending=True).iloc[0]

    top_y = PAGE_HEIGHT - 1.72 * inch
    card_width = (PAGE_WIDTH - 2 * MARGIN - 0.36 * inch) / 4
    draw_card(pdf, MARGIN, top_y, card_width, 0.48 * inch,
              "Best YTD", f"{best['index']} {format_percent(best['ytd_return'])}", GREEN)
    draw_card(pdf, MARGIN + card_width + 0.12 * inch, top_y, card_width, 0.48 * inch,
              "Weakest YTD", f"{worst['index']} {format_percent(worst['ytd_return'])}", RED)
    draw_card(pdf, MARGIN + 2 * (card_width + 0.12 * inch), top_y, card_width, 0.48 * inch,
              "Indices Covered", format_number(len(summary)), NAVY)
    draw_card(pdf, MARGIN + 3 * (card_width + 0.12 * inch), top_y, card_width, 0.48 * inch,
              "Latest Observations", format_number(summary["latest_observations"].sum()), NAVY)

    draw_section_label(pdf, MARGIN, PAGE_HEIGHT - 2.02 * inch, "Summary Metrics")
    column_widths = [
        0.32 * inch, 1.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch,
        0.55 * inch, 0.50 * inch, 0.42 * inch, 0.48 * inch, 0.55 * inch,
    ]
    draw_table(pdf, summary_table_rows(summary), MARGIN, PAGE_HEIGHT - 2.12 * inch,
               column_widths, 0.137 * inch)

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7)
    pdf.drawString(
        MARGIN,
        0.50 * inch,
        "Note: observation-count changes may reflect reporting coverage and survivorship, not only strategy-level fund launches or closures.",
    )
    draw_footer(pdf, page_number, latest_date)


def observation_series_by_year(group: pd.DataFrame) -> list[tuple[str, list[float]]]:
    series = []
    for year in sorted(group["year"].unique(), reverse=True)[:8]:
        year_group = group[group["year"] == year].sort_values("month")
        values = [math.nan] * 12
        for _, row in year_group.iterrows():
            values[int(row["month"]) - 1] = row["cnt"]
        series.append((str(year), values))
    return series


def draw_index_page(pdf: canvas.Canvas, group: pd.DataFrame, metrics: pd.Series,
                    latest_date: pd.Timestamp, page_number: int) -> None:
    index_name = metrics["index"]
    start_date = metrics["start_date"]
    draw_header(
        pdf,
        index_name,
        f"One-page index view | data from {start_date:%B %Y} to {latest_date:%B %Y}",
        "Index Detail",
    )

    top_y = PAGE_HEIGHT - 1.38 * inch
    card_width = (PAGE_WIDTH - 2 * MARGIN - 0.60 * inch) / 6
    cards = [
        ("Latest", format_percent(metrics["latest_month_return"]),
         GREEN if metrics["latest_month_return"] >= 0 else RED),
        ("YTD", format_percent(metrics["ytd_return"]),
         GREEN if metrics["ytd_return"] >= 0 else RED),
        ("12M", format_percent(metrics["trailing_12m_return"]),
         GREEN if metrics["trailing_12m_return"] >= 0 else RED),
        ("Ann. Return", format_percent(metrics["annualized_return"]), NAVY),
        ("Max DD", format_percent(metrics["max_drawdown"]), RED),
        ("Latest Obs", format_number(metrics["latest_observations"]), NAVY),
    ]

    for card_number, (label, value, color) in enumerate(cards):
        draw_card(
            pdf,
            MARGIN + card_number * (card_width + 0.12 * inch),
            top_y,
            card_width,
            0.46 * inch,
            label,
            value,
            color,
        )

    ordered_group = group.sort_values("date")
    performance = cumulative_index(ordered_group).tolist()
    draw_line_chart(
        pdf,
        MARGIN + 0.34 * inch,
        2.65 * inch,
        4.55 * inch,
        2.28 * inch,
        [(index_name, performance)],
        "Cumulative Performance (rebased to 100)",
    )

    draw_line_chart(
        pdf,
        5.78 * inch,
        2.65 * inch,
        4.75 * inch,
        2.28 * inch,
        observation_series_by_year(ordered_group),
        "Observation Count by Month",
    )

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.2)
    for month_index, month in enumerate(MONTHS):
        pdf.drawCentredString(5.78 * inch + 4.75 * inch * month_index / 11,
                              2.49 * inch, month[0])

    draw_section_label(pdf, MARGIN, 1.58 * inch, "Monthly Returns and Average Observation Count")
    table_rows = build_monthly_return_table(ordered_group[ordered_group["date"] >= pd.Timestamp(2021, 1, 1)])
    column_widths = [0.42 * inch] + [0.47 * inch] * 12 + [0.50 * inch, 0.50 * inch]
    draw_table(pdf, table_rows, MARGIN, 1.47 * inch, column_widths, 0.145 * inch)

    obs_change = metrics["observation_change_ytd"]
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.5)
    pdf.drawRightString(
        PAGE_WIDTH - MARGIN,
        0.46 * inch,
        f"YTD observation change: {format_number(metrics['january_observations'])} to "
        f"{format_number(metrics['latest_observations'])} ({format_number(obs_change)})",
    )
    draw_footer(pdf, page_number, latest_date)


def generate_pdf_report(input_path: str | Path, output_path: str | Path) -> None:
    df = load_index_data(input_path)
    summary = build_summary_metrics(df)
    latest_date = df["date"].max()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=landscape(letter))
    page_number = 1

    draw_summary_page(pdf, summary, latest_date, page_number)
    pdf.showPage()

    for _, metrics in summary.sort_values("index").iterrows():
        page_number += 1
        group = df[df["type"] == metrics["index"]]
        draw_index_page(pdf, group, metrics, latest_date, page_number)
        pdf.showPage()

    pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an automated index report PDF.")
    parser.add_argument("--input", required=True, help="Path to monthly index CSV")
    parser.add_argument("--output", default="index_report.pdf", help="Output PDF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_pdf_report(args.input, args.output)
    print(f"Index report generated: {args.output}")


if __name__ == "__main__":
    main()
