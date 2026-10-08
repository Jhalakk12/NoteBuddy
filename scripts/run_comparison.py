#!/usr/bin/env python3
"""Call the Exercise 3 comparison endpoint and print both answers."""

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="When is assignment 2 due?",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    response = httpx.post(
        f"{args.url.rstrip('/')}/compare",
        json={"question": args.question},
        timeout=300,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

