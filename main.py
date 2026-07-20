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


CANONICAL_COLUMNS = ["date", "ror", "type", "cnt", "aumbn"]
COLUMN_ALIASES = {
    "date": ["date", "month", "period", "as_of_date", "month_end", "monthend"],
    "ror": ["ror", "return", "returns", "monthly_return", "monthly_returns", "performance"],
    "type": ["type", "index", "index_name", "strategy", "category", "name"],
    "cnt": ["cnt", "count", "observations", "observation_count", "num_observations", "managers", "manager_count"],
    "aumbn": ["aumbn", "aum_bn", "aum", "assets_bn", "assets", "aum_billion"],
}
REQUIRED_CANONICAL_COLUMNS = {"date", "ror", "type"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MAIN_INDICES = ["CTA", "Crypto", "Equity LS", "Market Neutral", "Risk Parity"]
LOGO_URL = "https://front.nilssonhedge.com/"

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
    fontSize=6.7,
    leading=7.6,
    textColor=DARK,
))
STYLES.add(ParagraphStyle(
    name="TinyRight",
    parent=STYLES["Tiny"],
    alignment=2,
))
STYLES.add(ParagraphStyle(
    name="HeaderTiny",
    parent=STYLES["Tiny"],
    textColor=colors.white,
))
STYLES.add(ParagraphStyle(
    name="HeaderTinyRight",
    parent=STYLES["HeaderTiny"],
    alignment=2,
))
STYLES.add(ParagraphStyle(
    name="Commentary",
    parent=STYLES["Normal"],
    fontName="Helvetica",
    fontSize=7.4,
    leading=9,
    textColor=DARK,
))


def normalize_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_column(columns: list[str], canonical_name: str) -> str | None:
    normalized_to_original = {normalize_column_name(column): column for column in columns}
    for alias in COLUMN_ALIASES[canonical_name]:
        if alias in normalized_to_original:
            return normalized_to_original[alias]
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canonical_name in CANONICAL_COLUMNS:
        source_column = find_column(list(df.columns), canonical_name)
        if source_column:
            rename_map[source_column] = canonical_name

    df = df.rename(columns=rename_map).copy()
    missing_required = REQUIRED_CANONICAL_COLUMNS.difference(df.columns)
    if missing_required:
        expected = {
            name: COLUMN_ALIASES[name]
            for name in sorted(missing_required)
        }
        raise ValueError(
            "Input file is missing required index-report columns. "
            f"Missing: {sorted(missing_required)}. Accepted column names: {expected}"
        )

    if "cnt" not in df.columns:
        df["cnt"] = math.nan
    if "aumbn" not in df.columns:
        df["aumbn"] = math.nan
    return df[CANONICAL_COLUMNS]


def load_index_data(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the monthly index dataset."""
    df = pd.read_csv(input_path)
    df = standardize_columns(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ror"] = pd.to_numeric(df["ror"], errors="coerce")
    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce")
    df["aumbn"] = pd.to_numeric(df["aumbn"], errors="coerce")

    # If returns look like percentages, e.g. 2.1 instead of 0.021, convert them.
    if df["ror"].abs().dropna().median() > 1:
        df["ror"] = df["ror"] / 100

    df["type"] = df["type"].astype(str).str.strip()
    df = df.dropna(subset=["date", "type", "ror"])
    df = df.sort_values(["type", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def load_descriptions(description_path: str | Path | None) -> dict[str, str]:
    """Load optional index descriptions from a CSV.

    Expected structure is flexible. Preferred columns are:
        index, description
    Common alternatives such as type/index_name/name and text/commentary
    are also accepted.
    """
    if not description_path:
        return {}

    path = Path(description_path)
    if not path.exists():
        raise FileNotFoundError(f"Description CSV not found: {path}")

    df = pd.read_csv(path)
    index_column = None
    description_column = None
    normalized = {normalize_column_name(column): column for column in df.columns}

    for candidate in ["index", "index_name", "type", "strategy", "name"]:
        if candidate in normalized:
            index_column = normalized[candidate]
            break

    for candidate in ["description", "text", "commentary", "comment", "notes"]:
        if candidate in normalized:
            description_column = normalized[candidate]
            break

    if not index_column or not description_column:
        raise ValueError("Description CSV must include index/name and description/text columns.")

    descriptions = {}
    for _, row in df.dropna(subset=[index_column]).iterrows():
        descriptions[str(row[index_column]).strip()] = str(row.get(description_column, "")).strip()
    return descriptions


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
        prior_row = group[group["date"] < latest_date].tail(1)
        january_row = group[(group["year"] == latest_date.year) & (group["month"] == 1)].tail(1)

        ytd_returns = group[group["year"] == latest_date.year]["ror"]
        trailing_12m_returns = group[group["date"] > latest_date - pd.DateOffset(months=12)]["ror"]
        all_returns = group["ror"]

        ann_return = annualized_return(all_returns)
        ann_vol = annualized_volatility(all_returns)
        latest_obs = latest_row["cnt"].iloc[0] if not latest_row.empty else math.nan
        prior_obs = prior_row["cnt"].iloc[0] if not prior_row.empty else math.nan
        january_obs = january_row["cnt"].iloc[0] if not january_row.empty else math.nan
        latest_aum = latest_row["aumbn"].iloc[0] if not latest_row.empty else math.nan
        prior_aum = prior_row["aumbn"].iloc[0] if not prior_row.empty else math.nan
        manager_reporting_pct = latest_obs / prior_obs - 1 if prior_obs and not pd.isna(prior_obs) else math.nan
        aum_reporting_pct = latest_aum / prior_aum - 1 if prior_aum and not pd.isna(prior_aum) else math.nan

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
            "prior_observations": prior_obs,
            "january_observations": january_obs,
            "manager_reporting_pct": manager_reporting_pct,
            "observation_change_ytd": latest_obs - january_obs
            if not pd.isna(latest_obs) and not pd.isna(january_obs) else math.nan,
            "latest_aum_bn": latest_aum,
            "prior_aum_bn": prior_aum,
            "aum_reporting_pct": aum_reporting_pct,
            "months_in_history": len(group),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("ytd_return", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank_ytd", range(1, len(summary) + 1))
    return summary


def build_monthly_return_table(group: pd.DataFrame) -> list[list[str]]:
    """Build a calendar-year monthly return table for one index."""
    rows = [["Year"] + MONTHS + ["YTD", "Avg Manager"]]

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


def build_short_correlation_lookup(df: pd.DataFrame, months: int = 12, limit: int = 5) -> dict:
    """Calculate closest and lowest short-term correlations for each index."""
    returns = df.pivot_table(index="date", columns="type", values="ror").sort_index()
    recent_returns = returns.tail(months)
    correlation_matrix = recent_returns.corr(min_periods=max(4, months // 2))
    lookup = {}

    for index_name in correlation_matrix.columns:
        correlations = correlation_matrix[index_name].drop(index_name, errors="ignore").dropna()
        lookup[index_name] = {
            "closest": correlations.sort_values(ascending=False).head(limit),
            "lowest": correlations.sort_values(ascending=True).head(limit),
        }

    return lookup


def draw_header(pdf: canvas.Canvas, title: str, subtitle: str, label: str,
                logo_path: str | Path | None = None) -> None:
    pdf.setFillColor(colors.white)
    pdf.rect(0, PAGE_HEIGHT - 0.48 * inch, PAGE_WIDTH, 0.48 * inch, stroke=0, fill=1)
    pdf.setStrokeColor(colors.HexColor("#E5E7EB"))
    pdf.line(0, PAGE_HEIGHT - 0.48 * inch, PAGE_WIDTH, PAGE_HEIGHT - 0.48 * inch)

    logo_x = MARGIN
    logo_y = PAGE_HEIGHT - 0.34 * inch
    logo_right = logo_x + 1.25 * inch
    if logo_path and Path(logo_path).exists():
        pdf.drawImage(str(logo_path), logo_x, logo_y - 0.16 * inch,
                      width=1.22 * inch, height=0.26 * inch,
                      preserveAspectRatio=True, mask="auto")
    else:
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
        logo_right = hedge_x + 0.43 * inch
    pdf.linkURL(
        LOGO_URL,
        (logo_x, logo_y - 0.15 * inch, logo_right, logo_y + 0.12 * inch),
        relative=0,
    )

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
        f"Source: monthly index dataset. Returns are monthly. Manager counts represent reporting coverage. Data through {latest_date:%B %Y}.",
    )
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 0.16 * inch, f"Page {page_number}")


def draw_card(pdf: canvas.Canvas, x: float, y: float, width: float, height: float,
              label: str, value: str, accent=NAVY, value_align: str = "right") -> None:
    label_font_size = 6.1
    value_font_size = 10.2
    label_y = y + height - 0.198 * inch
    value_y = y + 0.100 * inch

    pdf.setFillColor(PANEL)
    pdf.setStrokeColor(colors.HexColor("#E7EAEE"))
    pdf.roundRect(x, y, width, height, 4, stroke=1, fill=1)
    pdf.setFillColor(accent)
    pdf.rect(x, y + height - 0.06 * inch, width, 0.06 * inch, stroke=0, fill=1)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", label_font_size)
    pdf.drawString(x + 0.08 * inch, label_y, label.upper())
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", value_font_size)
    if value_align == "right":
        pdf.drawRightString(x + width - 0.08 * inch, value_y, value[:28])
    else:
        pdf.drawString(x + 0.08 * inch, value_y, value[:28])


def draw_section_label(pdf: canvas.Canvas, x: float, y: float, text: str) -> None:
    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(x, y, text)


def draw_line_chart(pdf: canvas.Canvas, x: float, y: float, width: float, height: float,
                    series: list[tuple[str, list[float]]], title: str,
                    y_axis_format: str = "number", show_legend: bool = True,
                    x_labels: list[str] | None = None) -> None:
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
        label = f"{value:.0f}"
        pdf.drawRightString(x - 0.05 * inch, y + height * i / 4 - 2, label)

    if x_labels:
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 6.1)
        count = len(x_labels)
        for index, label in enumerate(x_labels):
            if not label:
                continue
            pdf.drawCentredString(scale_x(index, count), y - 0.16 * inch, label)

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

        if show_legend:
            legend_x = x + series_number * 0.57 * inch
            legend_y = y - 0.21 * inch
            pdf.setFillColor(color)
            pdf.rect(legend_x, legend_y, 0.055 * inch, 0.055 * inch, stroke=0, fill=1)
            pdf.setFillColor(DARK)
            pdf.setFont("Helvetica", 5.9)
            pdf.drawString(legend_x + 0.075 * inch, legend_y - 1, name[:7])

    pdf.setStrokeColor(DARK)
    pdf.setLineWidth(0.6)
    pdf.line(x, y, x + width, y)
    pdf.line(x, y, x, y + height)


def paragraph_cell(value: str, right_align: bool = False, header: bool = False) -> Paragraph:
    if header:
        style = STYLES["HeaderTinyRight"] if right_align else STYLES["HeaderTiny"]
    else:
        style = STYLES["TinyRight"] if right_align else STYLES["Tiny"]
    return Paragraph(str(value), style)


def draw_table(pdf: canvas.Canvas, rows: list[list[str]], x: float, top_y: float,
               column_widths: list[float], row_height: float) -> float:
    formatted_rows = []
    for row_index, row in enumerate(rows):
        formatted_rows.append([
            paragraph_cell(cell, right_align=col_index != 1 and row_index != 0, header=row_index == 0)
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
        "Managers", "Mgrs %", "Prior AUM", "AUM %",
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
            format_percent(row["manager_reporting_pct"]),
            format_number(row["prior_aum_bn"], 1),
            format_percent(row["aum_reporting_pct"]),
        ])

    return rows


def select_main_index_summary(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[summary["index"].isin(MAIN_INDICES)].copy()
    if selected.empty:
        selected = summary.head(len(MAIN_INDICES)).copy()
    selected = selected.sort_values("ytd_return", ascending=False).reset_index(drop=True)
    selected["rank_ytd"] = range(1, len(selected) + 1)
    return selected


def draw_correlation_triangle_page(pdf: canvas.Canvas, df: pd.DataFrame,
                                   latest_date: pd.Timestamp, page_number: int,
                                   logo_path: str | Path | None = None) -> None:
    draw_header(
        pdf,
        "Main Index Correlation Triangle",
        f"Latest 12 months through {latest_date:%B %Y}",
        "Correlation",
        logo_path,
    )

    returns = df.pivot_table(index="date", columns="type", values="ror").sort_index()
    columns = [name for name in MAIN_INDICES if name in returns.columns]
    if len(columns) < 2:
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(MARGIN, PAGE_HEIGHT - 1.35 * inch, "Not enough main-index data for a correlation triangle.")
        draw_footer(pdf, page_number, latest_date)
        return

    corr = returns[columns].tail(12).corr(min_periods=6)
    rows = [[""] + columns]
    for row_name in columns:
        row = [row_name]
        for col_name in columns:
            if columns.index(col_name) > columns.index(row_name):
                row.append("")
            else:
                value = corr.loc[row_name, col_name]
                row.append(f"{value:.2f}" if not pd.isna(value) else "-")
        rows.append(row)

    table_rows = []
    for row_index, row in enumerate(rows):
        table_rows.append([
            paragraph_cell(cell, right_align=col_index > 0, header=row_index == 0)
            for col_index, cell in enumerate(row)
        ])

    col_widths = [1.25 * inch] + [1.02 * inch] * len(columns)
    table = Table(table_rows, colWidths=col_widths, rowHeights=[0.30 * inch] * len(table_rows))
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#EEF3F6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE2E7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    table.setStyle(TableStyle(style))
    table_width = sum(col_widths)
    table_height = 0.30 * inch * len(table_rows)
    table.wrapOn(pdf, table_width, table_height)
    table.drawOn(pdf, MARGIN, PAGE_HEIGHT - 1.45 * inch - table_height)

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(
        MARGIN,
        PAGE_HEIGHT - 1.62 * inch - table_height,
        "Lower triangle shows pairwise return correlations among main website indices using the latest 12 monthly returns.",
    )
    draw_footer(pdf, page_number, latest_date)


def draw_summary_page(pdf: canvas.Canvas, summary: pd.DataFrame,
                      latest_date: pd.Timestamp, page_number: int,
                      logo_path: str | Path | None = None) -> None:
    display_summary = select_main_index_summary(summary)
    draw_header(
        pdf,
        "Index Summary Dashboard",
        f"Main indices through {latest_date:%B %Y}",
        "Summary",
        logo_path,
    )

    best = display_summary.sort_values("ytd_return", ascending=False).iloc[0]
    worst = display_summary.sort_values("ytd_return", ascending=True).iloc[0]

    top_y = PAGE_HEIGHT - 1.72 * inch
    card_width = (PAGE_WIDTH - 2 * MARGIN - 0.36 * inch) / 4
    draw_card(pdf, MARGIN, top_y, card_width, 0.54 * inch,
              "Best YTD", f"{best['index']} {format_percent(best['ytd_return'])}", GREEN)
    draw_card(pdf, MARGIN + card_width + 0.12 * inch, top_y, card_width, 0.54 * inch,
              "Weakest YTD", f"{worst['index']} {format_percent(worst['ytd_return'])}", RED)
    draw_card(pdf, MARGIN + 2 * (card_width + 0.12 * inch), top_y, card_width, 0.54 * inch,
              "Main Indices", format_number(len(display_summary)), NAVY)
    draw_card(pdf, MARGIN + 3 * (card_width + 0.12 * inch), top_y, card_width, 0.54 * inch,
              "Managers Reporting", format_number(display_summary["latest_observations"].sum()), NAVY)

    draw_section_label(pdf, MARGIN, PAGE_HEIGHT - 2.02 * inch, "Main Index Summary")
    column_widths = [
        0.34 * inch, 1.34 * inch, 0.54 * inch, 0.54 * inch, 0.54 * inch,
        0.54 * inch, 0.50 * inch, 0.56 * inch, 0.54 * inch, 0.58 * inch, 0.52 * inch,
    ]
    draw_table(pdf, summary_table_rows(display_summary), MARGIN, PAGE_HEIGHT - 2.12 * inch,
               column_widths, 0.19 * inch)

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7)
    pdf.drawString(
        MARGIN,
        0.50 * inch,
        "Note: manager-reporting changes may reflect reporting coverage and survivorship, not only strategy-level fund launches or closures.",
    )
    draw_footer(pdf, page_number, latest_date)


def normalized_observation_series_by_year(group: pd.DataFrame) -> list[tuple[str, list[float]]]:
    """Return yearly observation-count series normalized to January = 100.

    Raw observation counts are useful, but they can make each year hard to
    compare when database coverage changes. Normalizing each year to January
    highlights within-year survivorship/reporting coverage changes.
    """
    series = []
    for year in sorted(group["year"].unique(), reverse=True)[:8]:
        year_group = group[group["year"] == year].sort_values("month")
        january_count = year_group.loc[year_group["month"] == 1, "cnt"]
        if january_count.empty or january_count.iloc[0] == 0:
            continue

        base_count = january_count.iloc[0]
        values = [math.nan] * 12
        for _, row in year_group.iterrows():
            values[int(row["month"]) - 1] = row["cnt"] / base_count * 100
        series.append((str(year), values))
    return series


def normalized_observation_values_by_year(group: pd.DataFrame) -> dict[int, list[float]]:
    values_by_year = {}
    for year in sorted(group["year"].unique()):
        year_group = group[group["year"] == year].sort_values("month")
        january_count = year_group.loc[year_group["month"] == 1, "cnt"]
        if january_count.empty or january_count.iloc[0] == 0 or pd.isna(january_count.iloc[0]):
            continue

        values = [math.nan] * 12
        base_count = january_count.iloc[0]
        for _, row in year_group.iterrows():
            if not pd.isna(row["cnt"]):
                values[int(row["month"]) - 1] = row["cnt"] / base_count * 100
        values_by_year[int(year)] = values
    return values_by_year


def trim_recent_values(values: list[float], count: int = 2) -> list[float]:
    trimmed = list(values)
    valid_positions = [index for index, value in enumerate(trimmed) if not pd.isna(value)]
    if len(valid_positions) > count + 2:
        for index in valid_positions[-count:]:
            trimmed[index] = math.nan
    return trimmed


def draw_observation_cone_chart(pdf: canvas.Canvas, x: float, y: float,
                                width: float, height: float, group: pd.DataFrame) -> None:
    draw_section_label(pdf, x, y + height + 0.12 * inch, "Manager Reporting Cone (Jan = 100)")
    values_by_year = normalized_observation_values_by_year(group)
    if not values_by_year:
        return

    current_year = max(values_by_year)
    current_values = trim_recent_values(values_by_year[current_year], count=2)
    historical_years = [year for year in values_by_year if year != current_year]
    historical_values = [values_by_year[year] for year in historical_years]

    cone_min = [math.nan] * 12
    cone_max = [math.nan] * 12
    for month_index in range(12):
        month_values = [
            values[month_index]
            for values in historical_values
            if not pd.isna(values[month_index])
        ]
        if month_values:
            cone_min[month_index] = min(month_values)
            cone_max[month_index] = max(month_values)

    all_values = [
        value
        for values in [cone_min, cone_max, current_values]
        for value in values
        if not pd.isna(value)
    ]
    if not all_values:
        return

    low, high = min(all_values), max(all_values)
    padding = (high - low) * 0.10 if high != low else 5
    low -= padding
    high += padding

    def scale_x(index: int) -> float:
        return x + width * index / 11

    def scale_y(value: float) -> float:
        return y + (value - low) / (high - low) * height

    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.4)
    for i in range(5):
        yy = y + height * i / 4
        pdf.line(x, yy, x + width, yy)
        value = low + (high - low) * i / 4
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 6.1)
        pdf.drawRightString(x - 0.05 * inch, yy - 2, f"{value:.0f}")

    upper_points = [
        (scale_x(index), scale_y(value))
        for index, value in enumerate(cone_max)
        if not pd.isna(value)
    ]
    lower_points = [
        (scale_x(index), scale_y(value))
        for index, value in enumerate(cone_min)
        if not pd.isna(value)
    ]

    if upper_points and lower_points:
        polygon = upper_points + list(reversed(lower_points))
        pdf.setFillColor(colors.Color(0.09, 0.29, 0.39, alpha=0.12))
        path = pdf.beginPath()
        path.moveTo(polygon[0][0], polygon[0][1])
        for point in polygon[1:]:
            path.lineTo(point[0], point[1])
        path.close()
        pdf.drawPath(path, stroke=0, fill=1)

    for values, color, line_width in [
        (cone_max, colors.HexColor("#93A4AD"), 0.8),
        (cone_min, colors.HexColor("#93A4AD"), 0.8),
        (current_values, NAVY, 1.6),
    ]:
        pdf.setStrokeColor(color)
        pdf.setLineWidth(line_width)
        last_point = None
        for index, value in enumerate(values):
            if pd.isna(value):
                last_point = None
                continue
            point = (scale_x(index), scale_y(value))
            if last_point:
                pdf.line(last_point[0], last_point[1], point[0], point[1])
            last_point = point

    legend_y = y + height + 0.14 * inch
    legend_x = x + width - 1.36 * inch
    pdf.setFont("Helvetica", 6.0)
    pdf.setFillColor(NAVY)
    pdf.rect(legend_x, legend_y - 0.01 * inch, 0.055 * inch, 0.055 * inch, stroke=0, fill=1)
    pdf.setFillColor(DARK)
    pdf.drawString(legend_x + 0.075 * inch, legend_y - 0.01 * inch, f"{current_year}")
    pdf.setFillColor(colors.HexColor("#93A4AD"))
    pdf.rect(legend_x + 0.42 * inch, legend_y - 0.01 * inch, 0.055 * inch, 0.055 * inch, stroke=0, fill=1)
    pdf.setFillColor(DARK)
    pdf.drawString(legend_x + 0.50 * inch, legend_y - 0.01 * inch, "Historical cone")

    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.1)
    for index, month in enumerate(MONTHS):
        pdf.drawCentredString(scale_x(index), y - 0.15 * inch, month)

    pdf.setStrokeColor(DARK)
    pdf.setLineWidth(0.6)
    pdf.line(x, y, x + width, y)
    pdf.line(x, y, x, y + height)


def format_correlation_items(items: pd.Series) -> str:
    if items.empty:
        return "Not enough short-term data"
    return ", ".join(f"{name} ({value:.2f})" for name, value in items.items())


def draw_short_correlation_summary(pdf: canvas.Canvas, index_name: str,
                                   correlation_lookup: dict, x: float, y: float,
                                   width: float) -> None:
    correlation_data = correlation_lookup.get(index_name, {})
    closest = correlation_data.get("closest", pd.Series(dtype=float))
    lowest = correlation_data.get("lowest", pd.Series(dtype=float))

    draw_section_label(pdf, x, y + 0.43 * inch, "Short-Term Correlation (latest 12 months)")
    pdf.setFillColor(PANEL)
    pdf.setStrokeColor(colors.HexColor("#E7EAEE"))
    pdf.roundRect(x, y - 0.12 * inch, width, 0.48 * inch, 4, stroke=1, fill=1)

    pdf.setFillColor(DARK)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawString(x + 0.08 * inch, y + 0.21 * inch, "Closest:")
    closest_text = Paragraph(format_correlation_items(closest), STYLES["Commentary"])
    closest_text.wrapOn(pdf, width - 0.72 * inch, 0.18 * inch)
    closest_text.drawOn(pdf, x + 0.58 * inch, y + 0.14 * inch)

    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawString(x + 0.08 * inch, y - 0.01 * inch, "Lowest:")
    lowest_text = Paragraph(format_correlation_items(lowest), STYLES["Commentary"])
    lowest_text.wrapOn(pdf, width - 0.72 * inch, 0.18 * inch)
    lowest_text.drawOn(pdf, x + 0.58 * inch, y - 0.08 * inch)


def draw_description_box(pdf: canvas.Canvas, index_name: str, descriptions: dict[str, str],
                         x: float, y: float, width: float) -> None:
    description = descriptions.get(index_name, "").strip()
    if not description:
        return

    pdf.setFillColor(PANEL)
    pdf.setStrokeColor(colors.HexColor("#E7EAEE"))
    pdf.roundRect(x, y, width, 0.40 * inch, 4, stroke=1, fill=1)
    draw_section_label(pdf, x + 0.08 * inch, y + 0.25 * inch, "Index Description")
    paragraph = Paragraph(description, STYLES["Commentary"])
    paragraph.wrapOn(pdf, width - 0.16 * inch, 0.18 * inch)
    paragraph.drawOn(pdf, x + 0.08 * inch, y + 0.06 * inch)


def draw_index_page(pdf: canvas.Canvas, group: pd.DataFrame, metrics: pd.Series,
                    correlation_lookup: dict, descriptions: dict[str, str],
                    latest_date: pd.Timestamp, page_number: int,
                    logo_path: str | Path | None = None) -> None:
    index_name = metrics["index"]
    start_date = metrics["start_date"]
    draw_header(
        pdf,
        index_name,
        f"One-page index view | data from {start_date:%B %Y} to {latest_date:%B %Y}",
        "Index Detail",
        logo_path,
    )

    top_y = PAGE_HEIGHT - 1.56 * inch
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
        ("Managers", format_number(metrics["latest_observations"]), NAVY),
    ]

    for card_number, (label, value, color) in enumerate(cards):
        draw_card(
            pdf,
            MARGIN + card_number * (card_width + 0.12 * inch),
            top_y,
            card_width,
            0.52 * inch,
            label,
            value,
            color,
        )

    ordered_group = group.sort_values("date")
    performance = cumulative_index(ordered_group).tolist()
    performance_labels = [
        str(date.year) if date.month == 1 else ""
        for date in ordered_group["date"]
    ]
    if performance_labels:
        performance_labels[0] = str(ordered_group["date"].iloc[0].year)
        performance_labels[-1] = str(ordered_group["date"].iloc[-1].year)

    has_description = bool(descriptions.get(index_name, "").strip())
    if has_description:
        draw_description_box(
            pdf,
            index_name,
            descriptions,
            MARGIN,
            6.30 * inch,
            PAGE_WIDTH - 2 * MARGIN,
        )

    draw_short_correlation_summary(
        pdf,
        index_name,
        correlation_lookup,
        MARGIN,
        5.68 * inch if has_description else 6.08 * inch,
        PAGE_WIDTH - 2 * MARGIN,
    )

    draw_line_chart(
        pdf,
        MARGIN + 0.34 * inch,
        2.65 * inch if has_description else 2.90 * inch,
        4.55 * inch,
        2.28 * inch,
        [(index_name, performance)],
        "Cumulative Performance (rebased to 100)",
        show_legend=False,
        x_labels=performance_labels,
    )

    draw_observation_cone_chart(
        pdf,
        5.78 * inch,
        2.65 * inch if has_description else 2.90 * inch,
        4.75 * inch,
        2.28 * inch,
        ordered_group,
    )

    draw_section_label(pdf, MARGIN, 1.58 * inch, "Monthly Returns and Average Manager Count")
    table_rows = build_monthly_return_table(ordered_group[ordered_group["date"] >= pd.Timestamp(2021, 1, 1)])
    column_widths = [0.42 * inch] + [0.47 * inch] * 12 + [0.50 * inch, 0.50 * inch]
    draw_table(pdf, table_rows, MARGIN, 1.47 * inch, column_widths, 0.145 * inch)
    draw_footer(pdf, page_number, latest_date)


def generate_pdf_report(input_path: str | Path, output_path: str | Path,
                        descriptions_path: str | Path | None = None,
                        logo_path: str | Path | None = None) -> None:
    df = load_index_data(input_path)
    summary = build_summary_metrics(df)
    correlation_lookup = build_short_correlation_lookup(df)
    descriptions = load_descriptions(descriptions_path)
    latest_date = df["date"].max()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=landscape(letter))
    page_number = 1

    draw_summary_page(pdf, summary, latest_date, page_number, logo_path)
    pdf.showPage()

    page_number += 1
    draw_correlation_triangle_page(pdf, df, latest_date, page_number, logo_path)
    pdf.showPage()

    for _, metrics in summary.sort_values("index").iterrows():
        page_number += 1
        group = df[df["type"] == metrics["index"]]
        draw_index_page(pdf, group, metrics, correlation_lookup, descriptions, latest_date, page_number, logo_path)
        pdf.showPage()

    pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an automated index report PDF.")
    parser.add_argument("--input", required=True, help="Path to monthly index CSV")
    parser.add_argument("--output", default="index_report.pdf", help="Output PDF path")
    parser.add_argument("--descriptions", default=None, help="Optional CSV with columns: index, description")
    parser.add_argument("--logo", default=None, help="Optional logo image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_pdf_report(args.input, args.output, args.descriptions, args.logo)
    print(f"Index report generated: {args.output}")


if __name__ == "__main__":
    main()
