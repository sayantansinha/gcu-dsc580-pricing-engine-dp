from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional, Any

import matplotlib.pyplot as plt
import pytz
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
    Preformatted,
)

from src.services.analytics.visual_tools import (
    chart_actual_vs_pred,
    chart_residuals,
    chart_residuals_qq,
)
from src.utils.data_io_utils import save_report_pdf
from src.utils.explain_utils import permutation_importance_scores, shap_summary_df
from ui.common import APP_NAME


@dataclass
class Section:
    title: str
    html: str  # HTML fragment for this section


# -------------------------------------------------------------------
# Basic HTML helpers
# -------------------------------------------------------------------

def _format_human_ts() -> str:
    """Return a localized timestamp like 'Nov 30, 2025 07:15 pm' in LA time."""
    tz = pytz.timezone("America/Los_Angeles")
    now = datetime.now(tz)
    return now.strftime("%b %d, %Y %I:%M %p").lower()


def _html_escape(text: str) -> str:
    import html
    return html.escape(str(text), quote=True)


def _html_table_from_mapping(mapping: Dict[str, Any], table_class: str = "tbl") -> str:
    """Render a simple 2-column HTML table from a dict."""
    if not mapping:
        return "<div class='muted'>No data available.</div>"

    rows = []
    for k, v in mapping.items():
        key = _html_escape(k)
        if isinstance(v, (dict, list)):
            val = _html_escape(json.dumps(v, indent=2))
            cell = f"<pre>{val}</pre>"
        else:
            cell = _html_escape(v)
        rows.append(f"<tr><th>{key}</th><td>{cell}</td></tr>")

    return f"<table class='{table_class}'><tbody>{''.join(rows)}</tbody></table>"


def _html_table_from_records(records: List[Dict[str, Any]], table_class: str = "tbl") -> str:
    """Render an HTML table from a list of dicts."""
    if not records:
        return "<div class='muted'>No data available.</div>"

    cols: List[str] = []
    for rec in records:
        for k in rec:
            if k not in cols:
                cols.append(k)

    head = "".join(f"<th>{_html_escape(c)}</th>" for c in cols)
    body_rows = []
    for rec in records:
        tds = []
        for c in cols:
            v = rec.get(c, "")
            if isinstance(v, (dict, list)):
                val = _html_escape(json.dumps(v, indent=2))
                cell = f"<pre>{val}</pre>"
            else:
                cell = _html_escape(v)
            tds.append(f"<td>{cell}</td>")
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    return f"<table class='{table_class}'><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_html_report(title: str, meta: Dict[str, Any], sections: List[Section]) -> str:
    """
    Build a full HTML page used both for on-screen display (if needed)
    and for conversion into PDF.
    """
    gen_ts = _format_human_ts()
    meta_tbl = _html_table_from_mapping(meta)

    head = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{_html_escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; color:#222; line-height:1.45; }}
h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
h2 {{ margin: 20px 0 8px 0; font-size: 18px; border-bottom: 1px solid #e5e5e5; padding-bottom: 4px; }}
h3 {{ margin: 14px 0 6px 0; font-size: 16px; }}
.meta {{ margin: 8px 0 16px 0; font-size: 12px; color:#555; }}
table.tbl {{ border-collapse: collapse; border-spacing:0; margin: 6px 0 12px 0; font-size: 12px; }}
table.tbl th, table.tbl td {{ border: 1px solid #ddd; padding: 4px 6px; vertical-align: top; }}
table.tbl th {{ background:#f7f7f7; text-align:left; font-weight: 600; }}
pre {{ background:#f8f8f8; padding: 6px 8px; font-size: 11px; overflow-x:auto; }}
img {{ max-width: 100%; height:auto; margin: 6px 0 10px 0; }}
.muted {{ color:#777; font-style:italic; }}
</style>
</head>
<body>
"""
    body_parts = [
        f"<h1>{_html_escape(title)}</h1>",
        f"<div class='meta'><strong>Generated:</strong> {gen_ts}</div>",
        "<h2>Run information</h2>",
        meta_tbl,
    ]
    for sec in sections:
        body_parts.append(f"<h2>{_html_escape(sec.title)}</h2>")
        body_parts.append(sec.html or "")

    tail = "</body></html>"
    return head + "".join(body_parts) + tail


# -------------------------------------------------------------------
# HTML → PDF helpers
# -------------------------------------------------------------------

_RE_BODY = re.compile(r"<body[^>]*>(.*?)</body>", re.I | re.S)
_RE_STYLE = re.compile(r"<style.*?>.*?</style>", re.I | re.S)
_RE_HEAD = re.compile(r"<head.*?>.*?</head>", re.I | re.S)
_RE_BR = re.compile(r"<br\s*/?>", re.I | re.S)
_RE_P = re.compile(r"<p.*?>(.*?)</p>", re.I | re.S)
_RE_H1 = re.compile(r"<h1.*?>(.*?)</h1>", re.I | re.S)
_RE_H2 = re.compile(r"<h2.*?>(.*?)</h2>", re.I | re.S)
_RE_H3 = re.compile(r"<h3.*?>(.*?)</h3>", re.I | re.S)
_RE_PRE = re.compile(r"<pre.*?>(.*?)</pre>", re.I | re.S)
_RE_TABLE = re.compile(r"<table.*?>.*?</table>", re.I | re.S)
_RE_TH = re.compile(r"<th.*?>(.*?)</th>", re.I | re.S)
_RE_TD = re.compile(r"<td.*?>(.*?)</td>", re.I | re.S)

# Match <img src="data:image/...base64,...">
_RE_IMG = re.compile(
    r"<img[^>]*src=['\"](data:image/[^,]+,[^'\"]+)['\"][^>]*>",
    re.I | re.S,
)

# ✅ FIXED: use \d+, not \\d+ (was treating \d literally before)
_PLACEHOLDER_RE = re.compile(r"^__IMG_PLACEHOLDER_(\d+)__$")


def _normalize_breaks(text: str) -> str:
    return _RE_BR.sub("<br/>", (text or "").strip())


def _placeholder_index(seg: str) -> Optional[int]:
    m = _PLACEHOLDER_RE.match(seg)
    return int(m.group(1)) if m else None


def _data_uri_to_image_flowable(data_uri: str, max_width: float) -> Image:
    """Convert a base64 data URI into a ReportLab Image flowable."""
    # Accept both full data URI and bare base64
    if data_uri.startswith("data:") and "," in data_uri:
        _, b64 = data_uri.split(",", 1)
    else:
        b64 = data_uri

    img_bytes = base64.b64decode(b64)
    buf = BytesIO(img_bytes)

    # Use ImageReader only to get size, then reset buffer
    ir = ImageReader(buf)
    iw, ih = ir.getSize()
    scale = min(max_width / float(iw), 1.0)

    # IMPORTANT: reset buffer so Image() reads from the start
    buf.seek(0)

    # ReportLab's Image flowable expects a filename or file-like, not ImageReader
    return Image(buf, width=iw * scale, height=ih * scale)


_STYLES = getSampleStyleSheet()
_STYLES.add(ParagraphStyle(name="Body", parent=_STYLES["Normal"], fontSize=10, leading=12))
_STYLES.add(
    ParagraphStyle(
        name="H2",
        parent=_STYLES["Heading2"],
        fontSize=14,
        leading=16,
        spaceBefore=12,
        spaceAfter=4,
    )
)
_STYLES.add(
    ParagraphStyle(
        name="H3",
        parent=_STYLES["Heading3"],
        fontSize=12,
        leading=14,
        spaceBefore=10,
        spaceAfter=2,
    )
)


def _parse_table_to_flowable(tbl_html: str) -> Optional[Table]:
    """Convert basic HTML table to a ReportLab Table."""
    head_match = re.search(r"<thead.*?>(.*?)</thead>", tbl_html, re.I | re.S)
    body_match = re.search(r"<tbody.*?>(.*?)</tbody>", tbl_html, re.I | re.S)
    html_head = head_match.group(1) if head_match else ""
    html_body = body_match.group(1) if body_match else tbl_html

    headers = _RE_TH.findall(html_head)
    rows = []
    for row_html in re.findall(r"<tr.*?>(.*?)</tr>", html_body, re.I | re.S):
        cells = _RE_TD.findall(row_html)
        if cells:
            rows.append(cells)

    if not headers and not rows:
        return None

    data = []
    if headers:
        data.append(headers)
    data.extend(rows)

    tbl = Table(data, repeatRows=1 if headers else 0)
    style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )
    tbl.setStyle(style)
    return tbl


def _html_to_story(html: str, page_width: float) -> List[Any]:
    """
    Convert a subset of HTML to ReportLab flowables:
      - h1/h2/h3
      - p
      - pre
      - table
      - img (data URI)
    """
    from reportlab.platypus import Paragraph
    import html as _html_mod

    story: List[Any] = []
    if not html:
        return story

    m = _RE_BODY.search(html)
    html = m.group(1) if m else html
    html = _RE_STYLE.sub("", html)
    html = _RE_HEAD.sub("", html)

    imgs: List[str] = []

    def _img_repl(m: re.Match) -> str:
        # store the full data URI (data:image/...base64,...)
        imgs.append(m.group(1))
        return f"__IMG_PLACEHOLDER_{len(imgs) - 1}__"

    # Replace <img> tags with placeholders, keep URIs in imgs[]
    html = _RE_IMG.sub(_img_repl, html)

    # use \d+ in the split so it matches the placeholders we generate
    chunks = re.split(
        r"(?i)(<h1.*?>.*?</h1>|<h2.*?>.*?</h2>|<h3.*?>.*?</h3>|<p.*?>.*?</p>|<pre.*?>.*?</pre>|<table.*?>.*?</table>)",
        html,
    )

    def _append_chunk(chunk: str):
        chunk = chunk.strip()
        if not chunk:
            return

        if _RE_H1.match(chunk):
            txt = _RE_H1.findall(chunk)[0]
            story.append(Paragraph(_normalize_breaks(txt), _STYLES["Heading1"]))
            story.append(Spacer(1, 8))
            return

        if _RE_H2.match(chunk):
            txt = _RE_H2.findall(chunk)[0]
            story.append(Paragraph(_normalize_breaks(txt), _STYLES["H2"]))
            story.append(Spacer(1, 6))
            return

        if _RE_H3.match(chunk):
            txt = _RE_H3.findall(chunk)[0]
            story.append(Paragraph(_normalize_breaks(txt), _STYLES["H3"]))
            story.append(Spacer(1, 4))
            return

        if _RE_PRE.match(chunk):
            txt = _RE_PRE.findall(chunk)[0]
            txt = _html_mod.unescape(txt)
            story.append(Preformatted(_normalize_breaks(txt), _STYLES["Code"]))
            story.append(Spacer(1, 6))
            return

        if _RE_TABLE.match(chunk):
            tbl = _parse_table_to_flowable(chunk)
            if tbl:
                story.append(tbl)
                story.append(Spacer(1, 8))
                return

        if _RE_P.match(chunk):
            body = _RE_P.findall(chunk)[0]
            segments = re.split(r"(__IMG_PLACEHOLDER_\d+__)", body)
            for seg in segments:
                if seg.startswith("__IMG_PLACEHOLDER_"):
                    idx = _placeholder_index(seg)
                    if idx is None or idx < 0 or idx >= len(imgs):
                        continue
                    story.append(_data_uri_to_image_flowable(imgs[idx], page_width))
                    story.append(Spacer(1, 6))
                else:
                    txt = _normalize_breaks(seg)
                    if txt.strip():
                        story.append(Paragraph(txt, _STYLES["Body"]))
                        story.append(Spacer(1, 4))
            return

        # fallback
        segments = re.split(r"(__IMG_PLACEHOLDER_\d+__)", chunk)
        for seg in segments:
            if seg.startswith("__IMG_PLACEHOLDER_"):
                idx = _placeholder_index(seg)
                if idx is None or idx < 0 or idx >= len(imgs):
                    continue
                story.append(_data_uri_to_image_flowable(imgs[idx], page_width))
                story.append(Spacer(1, 6))
            else:
                txt = _normalize_breaks(seg)
                if txt.strip():
                    story.append(Paragraph(txt, _STYLES["Body"]))
                    story.append(Spacer(1, 4))

    for c in chunks:
        _append_chunk(c)

    return story


def df_to_table(df, max_rows: int = 25) -> List[Any]:
    """Convert a DataFrame to a ReportLab table."""
    import pandas as pd

    if df is None:
        return []

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.head(max_rows)
    data = [list(df.columns)]
    data.extend(df.values.tolist())

    tbl = Table(data, repeatRows=1)
    style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )
    tbl.setStyle(style)
    return [tbl, Spacer(1, 8)]


def build_pdf_from_html(
        html_report: str,
        report_name: str,
        base_dir: str,
        df_preview=None,
) -> str:
    """
    Convert HTML to PDF and persist it via save_report_pdf (LOCAL or S3).
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40,
    )

    story: List[Any] = []
    page_width = doc.width
    story += _html_to_story(html_report, page_width)

    if df_preview is not None:
        story.append(PageBreak())
        story.append(Paragraph("Dataset preview (first rows)", _STYLES["H2"]))
        story += df_to_table(df_preview, max_rows=25)

    def _numbered(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(doc_.pagesize[0] - 36, 20, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_numbered, onLaterPages=_numbered)
    pdf_bytes = buf.getvalue()
    buf.close()

    return save_report_pdf(base_dir=base_dir, name=report_name, pdf_bytes=pdf_bytes)


# -------------------------------------------------------------------
# PPE-specific report sections
# -------------------------------------------------------------------

def _build_eda_section(df) -> Section:
    """Quantitative data exploration section, with stats + basic charts."""
    html_parts: List[str] = []
    html_parts.append(
        "<p>This section summarizes the main characteristics of the dataset "
        "used by the Predictive Pricing Engine data product.</p>"
    )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    try:
        desc = df.describe(include="all").transpose()
        desc_html = desc.to_html(classes="tbl", border=0)
    except Exception:
        desc_html = "<div class='muted'>Summary statistics could not be computed.</div>"

    html_parts.append("<h3>Summary statistics</h3>")
    html_parts.append(desc_html)

    # ------------------------------------------------------------------
    # Numeric distribution (first numeric column)
    # ------------------------------------------------------------------
    num_cols = list(df.select_dtypes(include="number").columns) if df is not None else []
    if num_cols:
        num_col = num_cols[0]
        try:
            fig, ax = plt.subplots()
            df[num_col].dropna().hist(ax=ax, bins=30)
            ax.set_title(f"Distribution of {num_col}")
            ax.set_xlabel(num_col)
            ax.set_ylabel("Count")
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            data_b64 = base64.b64encode(buf.read()).decode("ascii")
            plt.close(fig)

            uri = f"data:image/png;base64,{data_b64}"
            html_parts.append(f"<h3>Numeric distribution – {_html_escape(num_col)}</h3>")
            html_parts.append(f"<img src='{uri}'/>")
        except Exception:
            # don't break the report if plotting fails
            plt.close("all")

    # ------------------------------------------------------------------
    # Categorical frequency (non-numeric column)
    # ------------------------------------------------------------------
    cat_cols = list(df.select_dtypes(exclude="number").columns) if df is not None else []
    if cat_cols:
        cat_col = cat_cols[0]
        try:
            vc = df[cat_col].astype("string").value_counts().head(20)

            fig, ax = plt.subplots()
            vc.plot(kind="bar", ax=ax)
            ax.set_title(f"Top categories in {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel("Count")
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            data_b64 = base64.b64encode(buf.read()).decode("ascii")
            plt.close(fig)

            uri = f"data:image/png;base64,{data_b64}"
            html_parts.append(f"<h3>Categorical distribution – {_html_escape(cat_col)}</h3>")
            html_parts.append(f"<img src='{uri}'/>")
        except Exception:
            plt.close("all")

    return Section("Quantitative Data Exploration", "".join(html_parts))


def _build_visual_section(df) -> Section:
    """Visual exploration of numeric relationships (scatter + optional correlation)."""
    html_parts: List[str] = []
    html_parts.append(
        "<p>This section provides visual exploration of relationships between numeric variables.</p>"
    )

    num_cols = list(df.select_dtypes(include="number").columns)
    # Simple scatter (first two numeric columns)
    if len(num_cols) >= 2:
        x_col, y_col = num_cols[0], num_cols[1]
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], alpha=0.35)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        data_b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        uri = f"data:image/png;base64,{data_b64}"
        html_parts.append(f"<h3>{_html_escape(y_col)} vs {_html_escape(x_col)}</h3>")
        html_parts.append(f"<img src='{uri}'/>")

    # Optionally add a small correlation heatmap (first few numeric columns)
    if len(num_cols) >= 2:
        cols = num_cols[:8]
        corr = df[cols].corr()
        fig, ax = plt.subplots()
        cax = ax.imshow(corr, interpolation="nearest")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.set_title("Correlation heatmap")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        data_b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        uri = f"data:image/png;base64,{data_b64}"
        html_parts.append("<h3>Correlation heatmap</h3>")
        html_parts.append(f"<img src='{uri}'/>")

    return Section("Visual Exploration", "".join(html_parts))


def _build_model_section(
        per_model_metrics: Optional[List[Dict[str, Any]]] = None,
        ensemble_avg_metrics: Optional[Dict[str, Any]] = None,
        ensemble_wgt_metrics: Optional[Dict[str, Any]] = None,
        bp_results: Optional[Dict[str, Any]] = None,
        y_true=None,
        y_pred=None,
        model=None,
        x_valid=None,
        y_valid=None,
        x_sample=None,
) -> Section:
    """
    Model performance & analytics section.
    Uses your visual_tools charts for residual diagnostics and explain_utils
    for permutation importance + SHAP.
    """
    html_parts: List[str] = []
    html_parts.append(
        "<p>This section summarizes model performance metrics, residual diagnostics, "
        "and feature-level explainability for the Predictive Pricing Engine.</p>"
    )

    if per_model_metrics:
        html_parts.append("<h3>Per-model metrics</h3>")
        html_parts.append(_html_table_from_records(per_model_metrics))

    if ensemble_avg_metrics:
        html_parts.append("<h3>Ensemble (average) metrics</h3>")
        html_parts.append(_html_table_from_mapping(ensemble_avg_metrics))

    if ensemble_wgt_metrics:
        html_parts.append("<h3>Ensemble (weighted by 1/RMSE) metrics</h3>")
        html_parts.append(_html_table_from_mapping(ensemble_wgt_metrics))

    if bp_results:
        html_parts.append("<h3>Breusch–Pagan test</h3>")
        html_parts.append(_html_table_from_mapping(bp_results))

    # --- permutation importance ---
    if model is not None and x_valid is not None and y_valid is not None:
        try:
            pi_df = permutation_importance_scores(model, x_valid, y_valid, n_repeats=5)
            if not pi_df.empty:
                html_parts.append("<h3>Permutation importance (validation set)</h3>")
                html_parts.append(
                    pi_df.head(20).to_html(index=False, classes="tbl", border=0)
                )
        except Exception:
            # fail quietly, report still works
            pass

    # --- SHAP mean |SHAP| summary ---
    if model is not None and x_sample is not None:
        try:
            shap_df = shap_summary_df(model, x_sample)
            if not shap_df.empty:
                html_parts.append("<h3>SHAP mean |SHAP| (top features)</h3>")
                html_parts.append(
                    shap_df.head(20).to_html(index=False, classes="tbl", border=0)
                )
        except Exception:
            pass

    # Residual diagnostics using existing visual_tools helpers
    if y_true is not None and y_pred is not None:
        import pandas as pd

        y_true_s = pd.Series(y_true)
        y_pred_s = pd.Series(y_pred)

        uri1 = chart_actual_vs_pred(y_true_s, y_pred_s)
        uri2 = chart_residuals(y_true_s, y_pred_s)
        uri3 = chart_residuals_qq(y_true_s, y_pred_s)

        html_parts.append("<h3>Actual vs Predicted</h3>")
        html_parts.append(f"<img src='{uri1}'/>")
        html_parts.append("<h3>Residuals vs Predicted</h3>")
        html_parts.append(f"<img src='{uri2}'/>")
        html_parts.append("<h3>Residuals Q–Q plot</h3>")
        html_parts.append(f"<img src='{uri3}'/>")

    return Section("Model Performance & Analytics", "".join(html_parts))


# -------------------------------------------------------------------
# Public report generators (one per report type)
# -------------------------------------------------------------------

def generate_eda_report(run_id: str, df, report_name: str = "eda_report") -> str:
    """Quantitative data exploration report."""
    row_count = int(len(df)) if df is not None else 0
    meta = {
        "Run id": run_id,
        "Rows in dataset": row_count,
    }

    if df is not None and row_count > 0:
        sections = [_build_eda_section(df)]
        df_preview = df.head(25)
    else:
        sections = [
            Section(
                "Quantitative Data Exploration",
                "<p class='muted'>No dataset available.</p>",
            )
        ]
        df_preview = None

    html = build_html_report(f"{APP_NAME} – Quantitative Data Exploration", meta, sections)
    return build_pdf_from_html(html, report_name, base_dir=run_id, df_preview=df_preview)


def generate_visualization_report(run_id: str, df, report_name: str = "visual_report") -> str:
    """Visual exploration report."""
    row_count = int(len(df)) if df is not None else 0
    meta = {
        "Run id": run_id,
        "Rows in dataset": row_count,
    }

    if df is not None and row_count > 0:
        sections = [_build_visual_section(df)]
        df_preview = df.head(25)
    else:
        sections = [
            Section("Visual Exploration", "<p class='muted'>No dataset available.</p>")
        ]
        df_preview = None

    html = build_html_report(f"{APP_NAME} – Visual Exploration", meta, sections)
    return build_pdf_from_html(html, report_name, base_dir=run_id, df_preview=df_preview)


def generate_model_analytics_report(
        run_id: str,
        *,
        per_model_metrics: Optional[List[Dict[str, Any]]] = None,
        ensemble_avg_metrics: Optional[Dict[str, Any]] = None,
        ensemble_wgt_metrics: Optional[Dict[str, Any]] = None,
        bp_results: Optional[Dict[str, Any]] = None,
        y_true=None,
        y_pred=None,
        model=None,
        x_valid=None,
        y_valid=None,
        x_sample=None,
        report_name: str = "model_analytics_report",
) -> str:
    """Model performance & analytics report."""
    meta = {
        "Run id": run_id,
    }

    sections = [
        _build_model_section(
            per_model_metrics=per_model_metrics,
            ensemble_avg_metrics=ensemble_avg_metrics,
            ensemble_wgt_metrics=ensemble_wgt_metrics,
            bp_results=bp_results,
            y_true=y_true,
            y_pred=y_pred,
            model=model,
            x_valid=x_valid,
            y_valid=y_valid,
            x_sample=x_sample,
        )
    ]

    html = build_html_report(f"{APP_NAME} – Model Analytics", meta, sections)
    return build_pdf_from_html(html, report_name, base_dir=run_id, df_preview=None)


def generate_full_technical_report(
        run_id: str,
        *,
        df,
        per_model_metrics: Optional[List[Dict[str, Any]]] = None,
        ensemble_avg_metrics: Optional[Dict[str, Any]] = None,
        ensemble_wgt_metrics: Optional[Dict[str, Any]] = None,
        bp_results: Optional[Dict[str, Any]] = None,
        y_true=None,
        y_pred=None,
        report_name: str = "technical_summary_report",
) -> str:
    """
    Combined technical summary (EDA + visual + model analytics) for the run.
    """
    row_count = int(len(df)) if df is not None else 0
    meta = {
        "Run id": run_id,
        "Rows in dataset": row_count,
    }

    sections: List[Section] = []
    df_preview = None
    if df is not None and row_count > 0:
        sections.append(_build_eda_section(df))
        sections.append(_build_visual_section(df))
        df_preview = df.head(25)

    sections.append(
        _build_model_section(
            per_model_metrics=per_model_metrics,
            ensemble_avg_metrics=ensemble_avg_metrics,
            ensemble_wgt_metrics=ensemble_wgt_metrics,
            bp_results=bp_results,
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    html = build_html_report(f"{APP_NAME} – Technical Summary", meta, sections)
    return build_pdf_from_html(html, report_name, base_dir=run_id, df_preview=df_preview)
