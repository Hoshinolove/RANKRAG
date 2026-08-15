from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml

from rankrag.models import RankingResult


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def write_jsonl(path: str | Path, results: Iterable[RankingResult]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, destination)


def iter_results(path: str | Path) -> Iterable[RankingResult]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield RankingResult.from_dict(json.loads(line))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        temporary = handle.name
    os.replace(temporary, destination)
