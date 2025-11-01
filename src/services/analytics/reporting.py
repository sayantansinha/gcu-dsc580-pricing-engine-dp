from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
import json
from services.data_io import save_report_local  # saves under /data/public

@dataclass
class Section:
    title: str
    html: str  # trusted HTML fragments

def build_html_report(title: str, meta: Dict, sections: List[Section]) -> str:
    head = f"""<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title></head><body>
    <h1>{title}</h1>
    <p><strong>Generated:</strong> {datetime.utcnow().isoformat()}Z</p>
    <pre>{json.dumps(meta, indent=2)}</pre><hr/>"""
    body = "".join([f"<h2>{s.title}</h2>\n{s.html}" for s in sections])
    return head + body + "</body></html>"

def save_html_report(html: str, report_name: str) -> str:
    """Persist to /data/public/<report_name>.html and return the path."""
    return save_report_local(html, report_name)
