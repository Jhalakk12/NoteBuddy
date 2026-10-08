# NoteBuddy Evaluation Report

Generated: 2026-09-06T12:18:37.890139+00:00
Dataset: 25 shared questions × 3 models = 75 runs.

## Quantitative comparison (Overall)

| Model | Accuracy | Relevance F1 | Retrieval P@4 | Retrieval R@4 | MRR | Hallucination | Test pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| codellama | 52.9% | 42.6% | 23.9% | 64.1% | 0.530 | 24.0% | 33.3% |
| starcoder2:3b | 31.3% | 24.7% | 23.9% | 64.1% | 0.530 | 28.0% | 0.0% |
| qwen2.5-coder:1.5b | 58.0% | 43.0% | 23.9% | 64.1% | 0.530 | 0.0% | 66.7% |

## Performance comparison

| Model | Mean latency | P95 latency | Prompt tokens | Completion tokens | Total tokens | Mean CPU | Peak CPU | Peak memory | Model memory | GPU/Metal memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codellama | 18.27s | 31.07s | 880.7 | 68.5 | 949.2 | 542.4% | 620.4% | 9792.5 MB | 4282.7 MB | 0.0 MB |
| starcoder2:3b | 7.80s | 14.57s | 799.8 | 96.9 | 896.7 | 507.0% | 618.7% | 2193.4 MB | 1786.2 MB | 0.0 MB |
| qwen2.5-coder:1.5b | 3.63s | 7.34s | 737.7 | 54.0 | 791.6 | 424.9% | 609.8% | 1545.2 MB | 1211.2 MB | 0.0 MB |

## Category-wise quantitative comparison

The evaluation dataset is structured into 7 specific Software Engineering categories to evaluate distinct capabilities:

| Category | Model | Accuracy | Relevance F1 | Hallucination | Latency | Code Pass |
|---|---|---:|---:|---:|---:|---:|
| Explanation | codellama | 0.0% | 18.6% | 33.3% | 24.81s | — |
| Explanation | starcoder2:3b | 0.0% | 10.6% | 0.0% | 9.16s | — |
| Explanation | qwen2.5-coder:1.5b | 11.1% | 18.8% | 0.0% | 5.79s | — |
| Code Retrieval | codellama | 55.5% | 46.1% | 0.0% | 15.76s | — |
| Code Retrieval | starcoder2:3b | 33.3% | 21.8% | 33.3% | 8.46s | — |
| Code Retrieval | qwen2.5-coder:1.5b | 44.4% | 44.4% | 0.0% | 3.30s | — |
| Dependency Understanding | codellama | 30.0% | 23.6% | 33.3% | 28.56s | — |
| Dependency Understanding | starcoder2:3b | 16.7% | 15.6% | 33.3% | 10.61s | — |
| Dependency Understanding | qwen2.5-coder:1.5b | 16.7% | 23.9% | 0.0% | 6.78s | — |
| Bug Analysis | codellama | 77.8% | 62.6% | 0.0% | 13.62s | — |
| Bug Analysis | starcoder2:3b | 66.7% | 50.6% | 33.3% | 5.46s | — |
| Bug Analysis | qwen2.5-coder:1.5b | 66.7% | 46.8% | 0.0% | 2.32s | — |
| Code Generation | codellama | 88.9% | 21.1% | 66.7% | 9.00s | 33.3% |
| Code Generation | starcoder2:3b | 55.6% | 39.3% | 33.3% | 2.95s | 0.0% |
| Code Generation | qwen2.5-coder:1.5b | 100.0% | 21.1% | 0.0% | 1.35s | 66.7% |
| Refactoring | codellama | 66.7% | 39.4% | 0.0% | 17.01s | — |
| Refactoring | starcoder2:3b | 0.0% | 18.4% | 0.0% | 6.96s | — |
| Refactoring | qwen2.5-coder:1.5b | 66.7% | 42.4% | 0.0% | 3.21s | — |
| RAG based Question | codellama | 52.4% | 61.7% | 28.6% | 18.63s | — |
| RAG based Question | starcoder2:3b | 38.1% | 21.3% | 42.9% | 9.18s | — |
| RAG based Question | qwen2.5-coder:1.5b | 76.2% | 69.0% | 0.0% | 3.23s | — |

## Overall analysis and trade-offs (Exercise 04)

- **Which model provides better accuracy?** `qwen2.5-coder:1.5b` achieved the highest overall accuracy (58.0%), followed by `codellama` (52.9%) and `starcoder2:3b` (31.3%).
- **Which model produces fewer hallucinations?** `qwen2.5-coder:1.5b` produced the fewest hallucinations (0.0% proxy rate), compared to `codellama` (24.0%) and `starcoder2:3b` (28.0%).
- **Which model provides better retrieval-based responses?** `qwen2.5-coder:1.5b` produces the most faithful grounded answers in RAG tasks (76.2% accuracy, zero ungrounded drift), while `codellama` is strongest at repository code-symbol retrieval (55.5% accuracy).
- **Which model generates code with a higher test-pass rate?** `qwen2.5-coder:1.5b` achieved the highest test-pass rate (66.7% vs 33.3% for `codellama` and 0.0% for `starcoder2:3b`).
- **Which model has lower response latency?** `qwen2.5-coder:1.5b` is fastest with a mean response latency of 3.63s (P95: 7.34s), compared to 7.80s for `starcoder2:3b` and 18.27s for `codellama`.
- **Which model requires fewer computational resources?** `qwen2.5-coder:1.5b` required the least resources (peak Ollama RAM of 1545.2 MB and 1211.2 MB resident model memory, vs 2193.4 MB for `starcoder2:3b` and 9792.5 MB for `codellama`).
- **Is the most accurate model also the most efficient? Is there a quality–latency–resource trade-off?** Under these local conditions, `qwen2.5-coder:1.5b` achieved both top accuracy (58.0%) and top efficiency (3.63s latency, 1.5 GB RAM). However, there is a clear structural trade-off: `codellama` (7B parameters) outperformed smaller models on deep repository dependency tracing (30.0% vs 16.7%) and bug analysis (77.8% vs 66.7%), demonstrating that complex architectural reasoning benefits from larger parameter capacity despite higher resource requirements.


## Category-specific model selection answers

Direct empirical answers to the seven core evaluation questions:

1. **Which model performs best for Explanation?**
   - **Winner**: `qwen2.5-coder:1.5b` (11.1% accuracy, 5.79s latency)
   - **Evidence & Rationale**: Demonstrates clearest conceptual accuracy (11.1% vs 0.0%) with concise structure and lowest latency (5.79s).

2. **Which model is best for Code Retrieval?**
   - **Winner**: `codellama` (55.5% accuracy, 15.76s latency)
   - **Evidence & Rationale**: Achieves highest accuracy (55.5% vs 44.4% Qwen) with superior keyword and symbol recall across repository files.

3. **Which model performs better for Dependency Understanding?**
   - **Winner**: `codellama` (30.0% accuracy, 28.56s latency)
   - **Evidence & Rationale**: Best accuracy (30.0% vs 16.7%) at tracing multi-file import graphs, contracts, and impact scope.

4. **Which model is better for Bug Analysis?**
   - **Winner**: `codellama` (77.8% accuracy, 13.62s latency)
   - **Evidence & Rationale**: Highest diagnostic accuracy (77.8% vs 66.7%) identifying edge cases and root causes with 0% hallucinations.

5. **Which model is better for Code Generation?**
   - **Winner**: `qwen2.5-coder:1.5b` (100.0% accuracy, 1.35s latency)
   - **Evidence & Rationale**: Highest accuracy (100.0%) and unit-test assertion pass rate (66.7%) with 1.35s latency.

6. **Which model performs better for Refactoring?**
   - **Winner**: `qwen2.5-coder:1.5b` (66.7% accuracy, 3.21s latency)
   - **Evidence & Rationale**: Tied for highest refactoring accuracy (66.7%) with 5× faster generation (3.21s vs 17.01s) and zero hallucinations.

7. **Which model performs better for RAG?**
   - **Winner**: `qwen2.5-coder:1.5b` (76.2% accuracy, 3.23s latency)
   - **Evidence & Rationale**: Achieved highest factual grounding accuracy (76.2% vs 52.4% Code Llama) with 0% hallucinations in 3.23s.

## Evidence-based conclusions

- qwen2.5-coder:1.5b achieved the highest accuracy (58.0%).
- qwen2.5-coder:1.5b had the lowest hallucination proxy rate (0.0%).
- qwen2.5-coder:1.5b had the lowest mean latency (3.63s).
- qwen2.5-coder:1.5b used the lowest measured peak Ollama-container memory (1545.2 MB).
- qwen2.5-coder:1.5b had the highest generated-code test-pass rate (66.7%).
- qwen2.5-coder:1.5b used the fewest mean total tokens (791.6 per response).
- Retrieval scores are identical across models because each model received the same cached retrieved context; differences therefore measure generation, not retrieval randomness.
- qwen2.5-coder:1.5b was both the most accurate and fastest under these local conditions; repeat runs are still needed before generalising.

## Metric definitions

- **Correctness Accuracy:** Mean fraction of case-insensitive rubric facts found in each answer; unsupported questions score 1 only when the model refuses.
- **Relevance:** Mean lexical token-set F1 between the model answer and the instructor-authored reference answer.
- **Retrieval Precision At K:** Relevant retrieved citation entries divided by k. Course citations include the expected page/section; repository tasks use expected files.
- **Retrieval Recall At K:** Expected page/section citations (or repository files) retrieved at least once divided by expected items.
- **Retrieval Mrr:** Mean reciprocal rank of the first expected page/section citation or repository file; 0 when none appears.
- **Hallucination Rate:** Fraction flagged by a reproducible proxy: unsupported numeric/file-path claims, or failure to refuse an unsupported question.
- **Test Pass Rate:** Generated-code tasks whose extracted function compiles, passes an AST safety policy, and passes all fixed unit assertions, divided by code tasks.
- **Response Latency Seconds:** Client wall-clock time around the LLM API request; averages include model loading when it occurs.
- **Token Usage:** Ollama prompt_eval_count + eval_count, reported as mean prompt, completion, and total tokens per response.
- **Cpu Consumption:** Mean and peak Ollama-container CPU percentages sampled with docker stats while each request runs.
- **Memory Consumption:** Peak Ollama-container resident memory sampled with docker stats; loaded model bytes are also read from Ollama /api/ps.
- **Gpu Consumption:** Loaded GPU/Metal memory bytes reported by Ollama /api/ps. This measures model residency, not GPU utilization percentage.

## RAG pipeline analysis

The JSON report stores every QUESTION → RETRIEVED CONTEXT → LLM RESPONSE trace. The labels below are derived from source Recall@k, Precision@k, correctness, and the hallucination proxy.

- `course-01` / `codellama`: relevant information retrieved with irrelevant noise → correct.
- `course-02` / `codellama`: relevant information retrieved with irrelevant noise → partially correct or incorrect.
- `course-03` / `codellama`: important information missed → partially correct or incorrect.
- `course-04` / `codellama`: relevant information retrieved with irrelevant noise → correct.
- `course-05` / `codellama`: important information missed → partially correct or incorrect.
- `course-06` / `codellama`: important information missed → partially correct or incorrect.
- `course-07` / `codellama`: relevant information retrieved with irrelevant noise → correct.
- `course-08` / `codellama`: relevant information retrieved with irrelevant noise → correct.
- `course-09` / `codellama`: relevant information retrieved with irrelevant noise → partially correct or incorrect.
- `course-10` / `codellama`: relevant information retrieved with irrelevant noise → partially correct or incorrect.
- `course-11` / `codellama`: important information missed → partially correct or incorrect.
- `course-12` / `codellama`: relevant information retrieved with irrelevant noise → correct.

## Repository/codebase understanding

The repository index contains 56 files and 91 chunks. Multi-file answers and their exact retrieved code excerpts are stored in `latest.json` under `repository_analysis`.

- `repo-01` / `codellama`: accuracy 0.0%; retrieved `data/documents/example-syllabus.md, week4/evaluator.py, docs/IDEA_ARCHITECTURE_AND_PROCESS.md`.
- `repo-02` / `codellama`: accuracy 40.0%; retrieved `knowledge/ingest.py, scripts/ingest_documents.py, services/data/main.py, knowledge/loaders.py`.
- `repo-03` / `codellama`: accuracy 0.0%; retrieved `services/llm/main.py, tests/test_services.py, week4/metrics.py`.
- `repo-04` / `codellama`: accuracy 0.0%; retrieved `services/application/main.py, docs/PROJECT_WORK_AND_SETUP.md, scripts/dev_server.py, services/retrieval/main.py`.
- `repo-01` / `starcoder2:3b`: accuracy 0.0%; retrieved `data/documents/example-syllabus.md, week4/evaluator.py, docs/IDEA_ARCHITECTURE_AND_PROCESS.md`.
- `repo-02` / `starcoder2:3b`: accuracy 0.0%; retrieved `knowledge/ingest.py, scripts/ingest_documents.py, services/data/main.py, knowledge/loaders.py`.
- `repo-03` / `starcoder2:3b`: accuracy 0.0%; retrieved `services/llm/main.py, tests/test_services.py, week4/metrics.py`.
- `repo-04` / `starcoder2:3b`: accuracy 0.0%; retrieved `services/application/main.py, docs/PROJECT_WORK_AND_SETUP.md, scripts/dev_server.py, services/retrieval/main.py`.
- `repo-01` / `qwen2.5-coder:1.5b`: accuracy 0.0%; retrieved `data/documents/example-syllabus.md, week4/evaluator.py, docs/IDEA_ARCHITECTURE_AND_PROCESS.md`.
- `repo-02` / `qwen2.5-coder:1.5b`: accuracy 0.0%; retrieved `knowledge/ingest.py, scripts/ingest_documents.py, services/data/main.py, knowledge/loaders.py`.
- `repo-03` / `qwen2.5-coder:1.5b`: accuracy 0.0%; retrieved `services/llm/main.py, tests/test_services.py, week4/metrics.py`.
- `repo-04` / `qwen2.5-coder:1.5b`: accuracy 0.0%; retrieved `services/application/main.py, docs/PROJECT_WORK_AND_SETUP.md, scripts/dev_server.py, services/retrieval/main.py`.

GPU/Metal memory is 0 MB because this Docker Desktop Ollama container used CPU execution; GPU utilization is therefore not applicable for this run.

## Interpretation limit

These deterministic metrics are reproducible proxies, not human judgement. Accuracy depends on the authored rubric, lexical F1 penalises valid paraphrases, and the hallucination proxy detects unsupported numbers/file paths rather than every possible unsupported claim.
