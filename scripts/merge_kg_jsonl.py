#!/usr/bin/env python3
"""Merge and validate sharded KG JSONL files without loading records in memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="kg_extractions.part-*.jsonl")
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {input_dir}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with output.open("w", encoding="utf-8") as destination:
        for source in files:
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    record = json.loads(line)
                    record_id = str(record["id"])
                    if record_id in seen:
                        raise ValueError(f"Duplicate record id {record_id} in {source}:{line_number}")
                    if not record.get("entities") or "relationships" not in record:
                        raise ValueError(f"Invalid KG schema in {source}:{line_number}")
                    seen.add(record_id)
                    destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
    if args.expected_count is not None and count != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} records but merged {count}")
    print(f"merged_files={len(files)} records={count} output={output}")


if __name__ == "__main__":
    main()
