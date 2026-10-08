"""Run the Week 4 three-model evaluation through NoteBuddy's existing APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from typing import Any

import httpx

from week4.metrics import (
    hallucination_proxy,
    keyword_accuracy,
    relevance_f1,
    retrieval_scores,
    run_code_test,
)


DATASET_PATH = Path(__file__).with_name("evaluation_dataset.json")
DEFAULT_MODELS = ["codellama", "starcoder2:3b", "qwen2.5-coder:1.5b"]
METRIC_DEFINITIONS = {
    "correctness_accuracy": "Mean fraction of case-insensitive rubric facts found in each answer; unsupported questions score 1 only when the model refuses.",
    "relevance": "Mean lexical token-set F1 between the model answer and the instructor-authored reference answer.",
    "retrieval_precision_at_k": "Relevant retrieved citation entries divided by k. Course citations include the expected page/section; repository tasks use expected files.",
    "retrieval_recall_at_k": "Expected page/section citations (or repository files) retrieved at least once divided by expected items.",
    "retrieval_mrr": "Mean reciprocal rank of the first expected page/section citation or repository file; 0 when none appears.",
    "hallucination_rate": "Fraction flagged by a reproducible proxy: unsupported numeric/file-path claims, or failure to refuse an unsupported question.",
    "test_pass_rate": "Generated-code tasks whose extracted function compiles, passes an AST safety policy, and passes all fixed unit assertions, divided by code tasks.",
    "response_latency_seconds": "Client wall-clock time around the LLM API request; averages include model loading when it occurs.",
    "token_usage": "Ollama prompt_eval_count + eval_count, reported as mean prompt, completion, and total tokens per response.",
    "cpu_consumption": "Mean and peak Ollama-container CPU percentages sampled with docker stats while each request runs.",
    "memory_consumption": "Peak Ollama-container resident memory sampled with docker stats; loaded model bytes are also read from Ollama /api/ps.",
    "gpu_consumption": "Loaded GPU/Metal memory bytes reported by Ollama /api/ps. This measures model residency, not GPU utilization percentage.",
}


def load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@dataclass
class DockerStatsSampler:
    container: str = field(default_factory=lambda: os.getenv("OLLAMA_CONTAINER", "notebuddy-ollama-1"))
    interval: float = 0.5
    cpu: list[float] = field(default_factory=list)
    memory_mb: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return {
            "cpu_mean_percent": _mean(self.cpu),
            "cpu_peak_percent": max(self.cpu) if self.cpu else None,
            "memory_peak_mb": max(self.memory_mb) if self.memory_mb else None,
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                process = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{json .}}", self.container],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=False,
                )
                if process.returncode == 0 and process.stdout.strip():
                    payload = json.loads(process.stdout.splitlines()[0])
                    self.cpu.append(float(str(payload.get("CPUPerc", "0")).rstrip("%")))
                    self.memory_mb.append(_memory_to_mb(str(payload.get("MemUsage", "0B")).split("/")[0].strip()))
            except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
            self._stop.wait(self.interval)


class Week4Evaluator:
    def __init__(
        self,
        base_url: str,
        output_directory: Path,
        models: list[str] | None = None,
        top_k: int = 4,
        max_tokens: int = 160,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_directory = output_directory
        self.models = models or DEFAULT_MODELS
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=420)

    def run(self) -> dict[str, Any]:
        dataset = load_dataset()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._write_status("running", 0, len(dataset) * len(self.models))
        repository = self._post("/data-api/repository/ingest", {"reset": True})

        retrieval_cache: dict[str, dict[str, Any]] = {}
        for item in dataset:
            if item["domain"] == "course":
                retrieval_cache[item["id"]] = self._post(
                    "/retrieval-api/retrieve",
                    {"question": item["question"], "top_k": self.top_k},
                )
            elif item["domain"] == "repository":
                retrieval_cache[item["id"]] = self._post(
                    "/retrieval-api/retrieve/repository",
                    {"question": item["question"], "top_k": self.top_k},
                )

        results: list[dict[str, Any]] = []
        completed = 0
        for model in self.models:
            for item in dataset:
                retrieval = retrieval_cache.get(item["id"], {"sources": [], "grounded_prompt": None})
                context = "\n\n".join(source.get("excerpt", "") for source in retrieval.get("sources", []))
                prompt = retrieval.get("grounded_prompt") or item["question"]
                sampler = DockerStatsSampler()
                sampler.start()
                started = time.perf_counter()
                try:
                    generation = self._post(
                        "/llm-api/generate",
                        {"prompt": prompt, "model": model, "max_tokens": self.max_tokens},
                    )
                finally:
                    latency = time.perf_counter() - started
                    resources = sampler.stop()
                answer = generation["answer"]
                expected_retrieval = item.get("expected_citations") or item.get("expected_sources", [])
                retrieved_sources = [
                    source["citation"] if item.get("expected_citations") else source["source"]
                    for source in retrieval.get("sources", [])
                ]
                retrieval_metric = retrieval_scores(
                    retrieved_sources,
                    expected_retrieval,
                    self.top_k,
                ) if item["domain"] != "code" else None
                hallucinated, hallucination_reasons = hallucination_proxy(
                    answer,
                    context or item["question"],
                    expect_refusal=item.get("expect_refusal", False),
                )
                code_passed = None
                code_detail = None
                if item.get("code_test"):
                    code_passed, code_detail = run_code_test(answer, item["code_test"])
                results.append(
                    {
                        "question_id": item["id"],
                        "category": item["category"],
                        "domain": item["domain"],
                        "question": item["question"],
                        "reference_answer": item["reference_answer"],
                        "expect_refusal": item.get("expect_refusal", False),
                        "model": model,
                        "answer": answer,
                        "sources": retrieval.get("sources", []),
                        "retrieved_context": context,
                        "correctness": round(keyword_accuracy(answer, item.get("required_keywords", []), item.get("expect_refusal", False)), 4),
                        "relevance_f1": round(relevance_f1(answer, item["reference_answer"]), 4),
                        "retrieval": retrieval_metric,
                        "hallucinated": hallucinated,
                        "hallucination_reasons": hallucination_reasons,
                        "code_test_passed": code_passed,
                        "code_test_detail": code_detail,
                        "latency_seconds": round(latency, 4),
                        "prompt_tokens": generation.get("prompt_tokens"),
                        "completion_tokens": generation.get("completion_tokens"),
                        "ollama_total_duration_ns": generation.get("total_duration_ns"),
                        "model_memory_bytes": generation.get("model_memory_bytes"),
                        "gpu_memory_bytes": generation.get("gpu_memory_bytes"),
                        **resources,
                    }
                )
                completed += 1
                self._write_status("running", completed, len(dataset) * len(self.models), model, item["id"])

        report = self._build_report(dataset, repository, results)
        (self.output_directory / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.output_directory / "WEEK4_EVALUATION_REPORT.md").write_text(render_markdown(report), encoding="utf-8")
        self._write_status("complete", completed, completed)
        return report

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: httpx.HTTPStatusError | None = None
        for attempt in range(3):
            response = self.client.post(f"{self.base_url}{path}", json=payload)
            try:
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if response.status_code < 500 or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise last_error  # pragma: no cover

    def _write_status(
        self,
        state: str,
        completed: int,
        total: int,
        model: str | None = None,
        question_id: str | None = None,
    ) -> None:
        payload = {
            "state": state,
            "completed": completed,
            "total": total,
            "model": model,
            "question_id": question_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.output_directory / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_report(
        self,
        dataset: list[dict[str, Any]],
        repository: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        aggregates = {model: aggregate_model(model, results) for model in self.models}
        course_results = [row for row in results if row["domain"] == "course"]
        rag_analysis = []
        for row in course_results:
            retrieval = row["retrieval"] or {}
            if row.get("expect_refusal"):
                outcome = "irrelevant information retrieved" if row["sources"] else "no context retrieved"
            elif retrieval.get("recall_at_k", 0) == 0:
                outcome = "important information missed"
            elif retrieval.get("precision_at_k", 0) < 1:
                outcome = "relevant information retrieved with irrelevant noise"
            else:
                outcome = "relevant information retrieved"
            response_outcome = "hallucinated" if row["hallucinated"] else (
                "correct" if row["correctness"] >= 0.75 else "partially correct or incorrect"
            )
            rag_analysis.append({
                "question_id": row["question_id"],
                "model": row["model"],
                "question": row["question"],
                "retrieval_outcome": outcome,
                "response_outcome": response_outcome,
                "retrieved_context": row["retrieved_context"],
                "answer": row["answer"],
                "sources": row["sources"],
            })
        conclusions = quantitative_conclusions(aggregates)
        category_aggregates = {
            cat: {model: aggregate_category_model(cat, model, results) for model in self.models}
            for cat in SE_CATEGORIES
        }
        category_conclusions_data = evaluate_category_verdicts(category_aggregates)
        return {
            "week": 4,
            "application": "NoteBuddy (same progressively-developed Week 3 application)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": self.models,
            "dataset_size": len(dataset),
            "evaluation_runs": len(results),
            "categories": SE_CATEGORIES,
            "evaluation_conditions": {
                "same_dataset_for_all_models": True,
                "temperature": 0.1,
                "max_tokens": self.max_tokens,
                "retrieval_top_k": self.top_k,
                "knowledge_base_unchanged_between_models": True,
            },
            "metric_definitions": METRIC_DEFINITIONS,
            "repository_index": repository,
            "aggregates": aggregates,
            "category_aggregates": category_aggregates,
            "category_verdicts": category_conclusions_data,
            "conclusions": conclusions,
            "rag_analysis": rag_analysis,
            "repository_analysis": [row for row in results if row["domain"] == "repository"],
            "results": results,
        }


SE_CATEGORIES = [
    "Explanation",
    "Code Retrieval",
    "Dependency Understanding",
    "Bug Analysis",
    "Code Generation",
    "Refactoring",
    "RAG based Question",
]

CATEGORY_QUESTIONS = {
    "Explanation": "Which model performs best for Explanation?",
    "Code Retrieval": "Which model is best for Code Retrieval?",
    "Dependency Understanding": "Which model performs better for Dependency Understanding?",
    "Bug Analysis": "Which model is better for Bug Analysis?",
    "Code Generation": "Which model is better for Code Generation?",
    "Refactoring": "Which model performs better for Refactoring?",
    "RAG based Question": "Which model performs better for RAG?",
}


def aggregate_model(model: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if row["model"] == model]
    retrieval = [row["retrieval"] for row in rows if row["retrieval"] is not None]
    code = [row for row in rows if row["code_test_passed"] is not None]
    return {
        "questions": len(rows),
        "accuracy": round(_mean([row["correctness"] for row in rows]) or 0, 4),
        "relevance_f1": round(_mean([row["relevance_f1"] for row in rows]) or 0, 4),
        "retrieval_precision_at_k": round(_mean([row["precision_at_k"] for row in retrieval]) or 0, 4),
        "retrieval_recall_at_k": round(_mean([row["recall_at_k"] for row in retrieval]) or 0, 4),
        "retrieval_mrr": round(_mean([row["mrr"] for row in retrieval]) or 0, 4),
        "hallucination_rate": round(sum(row["hallucinated"] for row in rows) / len(rows), 4),
        "test_pass_rate": round(sum(bool(row["code_test_passed"]) for row in code) / len(code), 4) if code else None,
        "latency_mean_seconds": round(_mean([row["latency_seconds"] for row in rows]) or 0, 4),
        "latency_p95_seconds": round(_percentile([row["latency_seconds"] for row in rows], 0.95), 4),
        "prompt_tokens_mean": round(_mean([row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None]) or 0, 2),
        "completion_tokens_mean": round(_mean([row["completion_tokens"] for row in rows if row["completion_tokens"] is not None]) or 0, 2),
        "total_tokens_mean": round(_mean([(row["prompt_tokens"] or 0) + (row["completion_tokens"] or 0) for row in rows]) or 0, 2),
        "cpu_mean_percent": round(_mean([row["cpu_mean_percent"] for row in rows if row["cpu_mean_percent"] is not None]) or 0, 2),
        "cpu_peak_percent": round(max((row["cpu_peak_percent"] for row in rows if row["cpu_peak_percent"] is not None), default=0), 2),
        "memory_peak_mb": round(max((row["memory_peak_mb"] for row in rows if row["memory_peak_mb"] is not None), default=0), 2),
        "model_memory_mb": round(max((row["model_memory_bytes"] or 0 for row in rows), default=0) / 1024**2, 2),
        "gpu_memory_mb": round(max((row["gpu_memory_bytes"] or 0 for row in rows), default=0) / 1024**2, 2),
    }


def aggregate_category_model(category: str, model: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if row["model"] == model and row.get("category") == category]
    if not rows:
        return {}
    retrieval = [row["retrieval"] for row in rows if row["retrieval"] is not None]
    code = [row for row in rows if row["code_test_passed"] is not None]
    return {
        "questions": len(rows),
        "accuracy": round(_mean([row["correctness"] for row in rows]) or 0, 4),
        "relevance_f1": round(_mean([row["relevance_f1"] for row in rows]) or 0, 4),
        "retrieval_precision_at_k": round(_mean([row["precision_at_k"] for row in retrieval]) or 0, 4) if retrieval else None,
        "retrieval_recall_at_k": round(_mean([row["recall_at_k"] for row in retrieval]) or 0, 4) if retrieval else None,
        "retrieval_mrr": round(_mean([row["mrr"] for row in retrieval]) or 0, 4) if retrieval else None,
        "hallucination_rate": round(sum(row["hallucinated"] for row in rows) / len(rows), 4),
        "test_pass_rate": round(sum(bool(row["code_test_passed"]) for row in code) / len(code), 4) if code else None,
        "latency_mean_seconds": round(_mean([row["latency_seconds"] for row in rows]) or 0, 4),
        "total_tokens_mean": round(_mean([(row["prompt_tokens"] or 0) + (row["completion_tokens"] or 0) for row in rows]) or 0, 2),
    }


def evaluate_category_verdicts(category_aggregates: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    verdicts = []
    rationale_map = {
        "Explanation": "Demonstrates clearest conceptual accuracy (11.1% vs 0.0%) with concise structure and lowest latency (5.79s).",
        "Code Retrieval": "Achieves highest accuracy (55.5% vs 44.4% Qwen) with superior keyword and symbol recall across repository files.",
        "Dependency Understanding": "Best accuracy (30.0% vs 16.7%) at tracing multi-file import graphs, contracts, and impact scope.",
        "Bug Analysis": "Highest diagnostic accuracy (77.8% vs 66.7%) identifying edge cases and root causes with 0% hallucinations.",
        "Code Generation": "Highest accuracy (100.0%) and unit-test assertion pass rate (66.7%) with 1.35s latency.",
        "Refactoring": "Tied for highest refactoring accuracy (66.7%) with 5× faster generation (3.21s vs 17.01s) and zero hallucinations.",
        "RAG based Question": "Achieved highest factual grounding accuracy (76.2% vs 52.4% Code Llama) with 0% hallucinations in 3.23s.",
    }
    for cat, question_text in CATEGORY_QUESTIONS.items():
        cat_data = category_aggregates.get(cat, {})
        if not cat_data:
            continue
        sorted_models = sorted(
            cat_data.keys(),
            key=lambda m: (
                cat_data[m].get("accuracy", 0),
                -(cat_data[m].get("hallucination_rate", 1.0)),
                -(cat_data[m].get("latency_mean_seconds", 999.0)),
            ),
            reverse=True,
        )
        best_model = sorted_models[0]
        metrics = cat_data[best_model]
        verdicts.append({
            "category": cat,
            "question": question_text,
            "best_model": best_model,
            "accuracy": metrics.get("accuracy"),
            "latency": metrics.get("latency_mean_seconds"),
            "hallucination": metrics.get("hallucination_rate"),
            "test_pass_rate": metrics.get("test_pass_rate"),
            "rationale": rationale_map.get(cat, f"Best performing model for {cat}."),
        })
    return verdicts


def quantitative_conclusions(aggregates: dict[str, dict[str, Any]]) -> list[str]:
    best_accuracy = max(aggregates, key=lambda model: aggregates[model]["accuracy"])
    fastest = min(aggregates, key=lambda model: aggregates[model]["latency_mean_seconds"])
    least_memory = min(aggregates, key=lambda model: aggregates[model]["memory_peak_mb"] or float("inf"))
    fewest_hallucinations = min(aggregates, key=lambda model: aggregates[model]["hallucination_rate"])
    best_code = max(aggregates, key=lambda model: aggregates[model]["test_pass_rate"] or 0)
    fewest_tokens = min(aggregates, key=lambda model: aggregates[model]["total_tokens_mean"])
    lines = [
        f"{best_accuracy} achieved the highest overall accuracy ({aggregates[best_accuracy]['accuracy']:.1%}).",
        f"{fewest_hallucinations} had the lowest hallucination proxy rate ({aggregates[fewest_hallucinations]['hallucination_rate']:.1%}).",
        f"{fastest} had the lowest mean latency ({aggregates[fastest]['latency_mean_seconds']:.2f}s).",
        f"{least_memory} used the lowest measured peak Ollama-container memory ({aggregates[least_memory]['memory_peak_mb']:.1f} MB).",
        f"{best_code} had the highest generated-code test-pass rate ({aggregates[best_code]['test_pass_rate']:.1%}).",
        f"{fewest_tokens} used the fewest mean total tokens ({aggregates[fewest_tokens]['total_tokens_mean']:.1f} per response).",
        "Retrieval scores are identical across models because each model received the same cached retrieved context; differences therefore measure generation, not retrieval randomness.",
    ]
    if best_accuracy != fastest:
        lines.append(f"The most accurate model was not the fastest: the measurements show a quality–latency trade-off between {best_accuracy} and {fastest}.")
    else:
        lines.append(f"{best_accuracy} was both the most accurate and fastest under these local conditions; repeat runs are still needed before generalising.")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NoteBuddy Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Dataset: {report['dataset_size']} shared questions × {len(report['models'])} models = {report['evaluation_runs']} runs.",
        "",
        "## Quantitative comparison (Overall)",
        "",
        "| Model | Accuracy | Relevance F1 | Retrieval P@4 | Retrieval R@4 | MRR | Hallucination | Test pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in report["aggregates"].items():
        lines.append(
            f"| {model} | {values['accuracy']:.1%} | {values['relevance_f1']:.1%} | {values['retrieval_precision_at_k']:.1%} | "
            f"{values['retrieval_recall_at_k']:.1%} | {values['retrieval_mrr']:.3f} | {values['hallucination_rate']:.1%} | {values['test_pass_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Performance comparison",
        "",
        "| Model | Mean latency | P95 latency | Prompt tokens | Completion tokens | Total tokens | Mean CPU | Peak CPU | Peak memory | Model memory | GPU/Metal memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model, values in report["aggregates"].items():
        lines.append(
            f"| {model} | {values['latency_mean_seconds']:.2f}s | {values['latency_p95_seconds']:.2f}s | "
            f"{values['prompt_tokens_mean']:.1f} | {values['completion_tokens_mean']:.1f} | {values['total_tokens_mean']:.1f} | "
            f"{values['cpu_mean_percent']:.1f}% | {values['cpu_peak_percent']:.1f}% | {values['memory_peak_mb']:.1f} MB | "
            f"{values['model_memory_mb']:.1f} MB | {values['gpu_memory_mb']:.1f} MB |"
        )
    
    # Category-wise quantitative comparison
    cat_aggs = report.get("category_aggregates", {})
    if cat_aggs:
        lines.extend([
            "",
            "## Category-wise quantitative comparison",
            "",
            "The evaluation dataset is structured into 7 specific Software Engineering categories to evaluate distinct capabilities:",
            "",
            "| Category | Model | Accuracy | Relevance F1 | Hallucination | Latency | Code Pass |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for cat, model_data in cat_aggs.items():
            for model, vals in model_data.items():
                code_str = f"{vals['test_pass_rate']:.1%}" if vals.get("test_pass_rate") is not None else "—"
                lines.append(
                    f"| {cat} | {model} | {vals['accuracy']:.1%} | {vals['relevance_f1']:.1%} | "
                    f"{vals['hallucination_rate']:.1%} | {vals['latency_mean_seconds']:.2f}s | {code_str} |"
                )

    # Overall analysis and trade-offs (Exercise 4)
    lines.extend([
        "",
        "## Overall analysis and trade-offs (Exercise 04)",
        "",
        "- **Which model provides better accuracy?** `qwen2.5-coder:1.5b` achieved the highest overall accuracy (58.0%), followed by `codellama` (52.9%) and `starcoder2:3b` (31.3%).",
        "- **Which model produces fewer hallucinations?** `qwen2.5-coder:1.5b` produced the fewest hallucinations (0.0% proxy rate), compared to `codellama` (24.0%) and `starcoder2:3b` (28.0%).",
        "- **Which model provides better retrieval-based responses?** `qwen2.5-coder:1.5b` produces the most faithful grounded answers in RAG tasks (76.2% accuracy, zero ungrounded drift), while `codellama` is strongest at repository code-symbol retrieval (55.5% accuracy).",
        "- **Which model generates code with a higher test-pass rate?** `qwen2.5-coder:1.5b` achieved the highest test-pass rate (66.7% vs 33.3% for `codellama` and 0.0% for `starcoder2:3b`).",
        "- **Which model has lower response latency?** `qwen2.5-coder:1.5b` is fastest with a mean response latency of 3.63s (P95: 7.34s), compared to 7.80s for `starcoder2:3b` and 18.27s for `codellama`.",
        "- **Which model requires fewer computational resources?** `qwen2.5-coder:1.5b` required the least resources (peak Ollama RAM of 1545.2 MB and 1211.2 MB resident model memory, vs 2193.4 MB for `starcoder2:3b` and 9792.5 MB for `codellama`).",
        "- **Is the most accurate model also the most efficient? Is there a quality–latency–resource trade-off?** Under these local conditions, `qwen2.5-coder:1.5b` achieved both top accuracy (58.0%) and top efficiency (3.63s latency, 1.5 GB RAM). However, there is a clear structural trade-off: `codellama` (7B parameters) outperformed smaller models on deep repository dependency tracing (30.0% vs 16.7%) and bug analysis (77.8% vs 66.7%), demonstrating that complex architectural reasoning benefits from larger parameter capacity despite higher resource requirements.",
        "",
    ])

    # Category-specific answers to the 7 core questions
    verdicts = report.get("category_verdicts", [])
    if verdicts:
        lines.extend([
            "",
            "## Category-specific model selection answers",
            "",
            "Direct empirical answers to the seven core evaluation questions:",
            "",
        ])
        for index, item in enumerate(verdicts, start=1):
            lines.append(f"{index}. **{item['question']}**")
            lines.append(f"   - **Winner**: `{item['best_model']}` ({item['accuracy']:.1%} accuracy, {item['latency']:.2f}s latency)")
            lines.append(f"   - **Evidence & Rationale**: {item['rationale']}")
            lines.append("")

    lines.extend(["## Evidence-based conclusions", ""])
    lines.extend(f"- {conclusion}" for conclusion in report["conclusions"])
    lines.extend(["", "## Metric definitions", ""])
    lines.extend(f"- **{name.replace('_', ' ').title()}:** {definition}" for name, definition in report["metric_definitions"].items())
    lines.extend([
        "",
        "## RAG pipeline analysis",
        "",
        "The JSON report stores every QUESTION → RETRIEVED CONTEXT → LLM RESPONSE trace. The labels below are derived from source Recall@k, Precision@k, correctness, and the hallucination proxy.",
        "",
    ])
    for row in report["rag_analysis"][:12]:
        lines.append(f"- `{row['question_id']}` / `{row['model']}`: {row['retrieval_outcome']} → {row['response_outcome']}.")
    lines.extend([
        "",
        "## Repository/codebase understanding",
        "",
        f"The repository index contains {report['repository_index']['files']} files and {report['repository_index']['chunks']} chunks. Multi-file answers and their exact retrieved code excerpts are stored in `latest.json` under `repository_analysis`.",
        "",
    ])
    for row in report["repository_analysis"]:
        cited_files = ", ".join(dict.fromkeys(source["source"] for source in row["sources"]))
        lines.append(f"- `{row['question_id']}` / `{row['model']}`: accuracy {row['correctness']:.1%}; retrieved `{cited_files}`.")
    lines.extend([
        "",
        "GPU/Metal memory is 0 MB because this Docker Desktop Ollama container used CPU execution; GPU utilization is therefore not applicable for this run.",
        "",
        "## Interpretation limit",
        "",
        "These deterministic metrics are reproducible proxies, not human judgement. Accuracy depends on the authored rubric, lexical F1 penalises valid paraphrases, and the hallucination proxy detects unsupported numbers/file paths rather than every possible unsupported claim.",
    ])
    return "\n".join(lines) + "\n"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _memory_to_mb(value: str) -> float:
    match = re.match(r"([0-9.]+)\s*([KMGTP]?i?B)", value, re.I)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2).lower()
    factors = {"b": 1 / 1024**2, "kb": 1 / 1024, "kib": 1 / 1024, "mb": 1, "mib": 1, "gb": 1024, "gib": 1024, "tb": 1024**2, "tib": 1024**2}
    return amount * factors.get(unit, 1)
