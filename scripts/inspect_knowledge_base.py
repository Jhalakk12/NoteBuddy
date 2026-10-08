#!/usr/bin/env python3
"""Print stored chunks and citation metadata for Exercise 2 evidence."""

import argparse
import json
from pathlib import Path
import sys

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma",
        help="Persistent Chroma directory.",
    )
    args = parser.parse_args()
    client = chromadb.PersistentClient(path=str(args.database))
    collection = client.get_collection("course_materials")
    result = collection.get(limit=args.limit, include=["documents", "metadatas"])
    rows = [
        {"id": item_id, "text": text, "metadata": metadata}
        for item_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    ]
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
