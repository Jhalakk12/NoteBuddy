#!/usr/bin/env python3
"""Verify NoteBuddy's Week 3 workflows and evaluation dashboard."""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict | None = None, timeout: int = 360) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    request = Request(url, data=body, headers=headers, method="POST" if body else "GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def request_text(url: str, timeout: int = 30) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify website, private services, knowledge base, Ollama, RAG, and citations."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Public NoteBuddy website origin",
    )
    parser.add_argument(
        "--question",
        default="When is assignment 2 due?",
        help="Question with a known answer in the indexed course material",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        with urlopen(f"{base_url}/healthz", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"website health returned HTTP {response.status}")

        application_ui = request_text(f"{base_url}/")
        retrieval_ui = request_text(f"{base_url}/retrieval.html")
        llm_ui = request_text(f"{base_url}/llm.html")
        evaluation_ui = request_text(f"{base_url}/evaluation.html")
        guardrails_ui = request_text(f"{base_url}/guardrails.html")
        if "Ask your course" not in application_ui:
            raise RuntimeError("Application UI marker is missing")
        if "Knowledge + Retrieval" not in retrieval_ui:
            raise RuntimeError("Retrieval UI marker is missing")
        if "Raw LLM Workspace" not in llm_ui:
            raise RuntimeError("LLM UI marker is missing")
        if "Model &amp; RAG Evaluation" not in evaluation_ui:
            raise RuntimeError("Evaluation UI marker is missing")
        if "AI Guardrails" not in guardrails_ui:
            raise RuntimeError("Guardrails UI marker is missing")

        system = request_json(f"{base_url}/api/system/status", timeout=30)
        if system.get("overall") != "ok":
            raise RuntimeError(f"system is not healthy: {system}")

        knowledge = request_json(f"{base_url}/api/knowledge/status", timeout=30)
        if knowledge.get("collection_total", 0) < 1:
            raise RuntimeError("knowledge base has no indexed chunks")

        answer = request_json(
            f"{base_url}/api/ask",
            {"question": args.question},
        )
        if answer.get("orchestration") != ["application", "retrieval", "llm"]:
            raise RuntimeError(f"unexpected orchestration: {answer.get('orchestration')}")
        if not answer.get("sources"):
            raise RuntimeError("grounded answer returned no citations")

        retrieval = request_json(
            f"{base_url}/retrieval-api/retrieve",
            {"question": args.question, "top_k": 2},
        )
        if not retrieval.get("sources"):
            raise RuntimeError("Retrieval UI API returned no evidence")

        raw = request_json(
            f"{base_url}/llm-api/generate",
            {"prompt": "Reply with the single word READY."},
        )
        if not raw.get("answer") or not raw.get("model"):
            raise RuntimeError("LLM UI API returned an invalid response")

        dataset = request_json(f"{base_url}/api/evaluation/dataset", timeout=30)
        if dataset.get("count") != 25:
            raise RuntimeError(f"expected 25 Week 4 questions, got {dataset.get('count')}")

        evaluation = request_json(f"{base_url}/api/evaluation/report", timeout=30)
        if evaluation.get("evaluation_runs") != 75 or len(evaluation.get("models", [])) != 3:
            raise RuntimeError("Week 4 report must contain 25 questions × 3 models")

        repository = request_json(f"{base_url}/data-api/repository/status", timeout=30)
        if repository.get("collection_total", 0) < 1:
            raise RuntimeError("repository knowledge base has no indexed chunks")

        repository_search = request_json(
            f"{base_url}/retrieval-api/retrieve/repository",
            {"question": "Which files connect the browser to the backend services?", "top_k": 4},
        )
        if not repository_search.get("sources"):
            raise RuntimeError("repository retrieval returned no source files")

        guardrails = request_json(f"{base_url}/api/guardrails/overview", timeout=30)
        if guardrails.get("status") != "active" or guardrails.get("rules_count", 0) < 1:
            raise RuntimeError("Guardrails subsystem is not active")

        print("PASS: website health")
        print("PASS: Application, Retrieval/Knowledge, LLM, Evaluation, and Guardrails UI pages")
        print("PASS: Application, Retrieval, LLM/Ollama, and Data services")
        print(
            "PASS: knowledge base "
            f"({knowledge['document_count']} documents, {knowledge['collection_total']} chunks)"
        )
        print("PASS: browser-origin /api gateway")
        print("PASS: browser-origin /retrieval-api and /llm-api gateways")
        print("PASS: application -> retrieval -> llm orchestration")
        print("PASS: Week 4 dataset (25 questions), report (75 runs), and three-model results")
        print(
            "PASS: repository understanding "
            f"({repository['files']} files, {repository['collection_total']} chunks)"
        )
        print(f"PASS: AI Guardrails & Output Testing ({guardrails['rules_count']} rules active, {guardrails['test_dataset_size']} test cases)")
        print(f"PASS: citation {answer['sources'][0]['citation']}")
        print(f"ANSWER: {answer['answer']}")
        return 0
    except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
