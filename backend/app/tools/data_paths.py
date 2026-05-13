from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = APP_DIR / "data"


def data_dir() -> Path:
    return Path(os.getenv("CAMPAIGN_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()


def output_dir(kind: str = "pdfs") -> Path:
    root = Path(os.getenv("EXPORT_DIR", str(APP_DIR.parents[1] / "outputs" / "pdfs"))).resolve()
    if kind == "json":
        root = root.parent / "json"
    root.mkdir(parents=True, exist_ok=True)
    return root
