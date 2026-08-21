"""JSON export for aligned lyrics."""

from __future__ import annotations

import json
from pathlib import Path


def write_alignment(path: str | Path, document: dict, duration: float) -> None:
	destination = Path(path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	document["duration"] = round(duration, 3)
	document["timingStatus"] = "synced"
	destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

