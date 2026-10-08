"""Deterministic Week 4 metric definitions and safe code-test execution."""

import ast
import re
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import tempfile
from pathlib import Path


REFUSAL_MARKERS = (
    "couldn't find relevant information",
    "do not contain",
    "does not contain",
    "not provided",
    "insufficient",
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.I)
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?(?!\w|\.\d)")
FILE_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.(?:py|js|html|css|ya?ml|toml|md)", re.I)


def keyword_accuracy(answer: str, required: list[str], expect_refusal: bool = False) -> float:
    """Fraction of rubric facts present; refusals are binary."""
    lowered = answer.casefold()
    if expect_refusal:
        return float(any(marker in lowered for marker in REFUSAL_MARKERS))
    if not required:
        return 1.0
    return sum(keyword.casefold() in lowered for keyword in required) / len(required)


def relevance_f1(answer: str, reference: str) -> float:
    """Lexical token-set F1 between answer and the instructor-authored reference."""
    answer_tokens = set(TOKEN_RE.findall(answer.casefold()))
    reference_tokens = set(TOKEN_RE.findall(reference.casefold()))
    if not answer_tokens or not reference_tokens:
        return 0.0
    overlap = len(answer_tokens & reference_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def retrieval_scores(retrieved_sources: list[str], expected_sources: list[str], top_k: int) -> dict[str, float]:
    """Compute expected citation/source Precision@k, Recall@k, and reciprocal rank."""
    expected = set(expected_sources)
    if not expected:
        return {"precision_at_k": 1.0 if not retrieved_sources else 0.0, "recall_at_k": 1.0, "mrr": 1.0}
    relevant = [source for source in retrieved_sources[:top_k] if source in expected]
    first_rank = next(
        (index for index, source in enumerate(retrieved_sources[:top_k], start=1) if source in expected),
        None,
    )
    return {
        "precision_at_k": len(relevant) / top_k,
        "recall_at_k": len(set(relevant)) / len(expected),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def hallucination_proxy(
    answer: str,
    context: str,
    *,
    expect_refusal: bool = False,
) -> tuple[bool, list[str]]:
    """Flag unsupported numbers/file paths, or failure to refuse an unsupported question."""
    lowered = answer.casefold()
    if expect_refusal:
        refused = any(marker in lowered for marker in REFUSAL_MARKERS)
        return (not refused, [] if refused else ["failed_to_refuse"])

    support = context.casefold()
    supported_numbers = set(NUMBER_RE.findall(context))
    unsupported_numbers = sorted(set(NUMBER_RE.findall(answer)) - supported_numbers)
    unsupported_files = sorted({value for value in FILE_RE.findall(answer) if value.casefold() not in support})
    reasons = [f"unsupported_number:{value}" for value in unsupported_numbers]
    reasons.extend(f"unsupported_file:{value}" for value in unsupported_files)
    return bool(reasons), reasons


CODE_TESTS = {
    "normalize_question": """
assert normalize_question('  When   is\\n A2 due?  ') == 'When is A2 due?'
assert normalize_question('') == ''
assert normalize_question('one') == 'one'
""",
    "citation_label": """
assert citation_label('lecture.pdf', page=12) == 'lecture.pdf, page/slide 12'
assert citation_label('syllabus.md', section='Assessment') == 'syllabus.md, section: Assessment'
""",
    "reciprocal_rank": """
assert reciprocal_rank(['a', 'b', 'c'], 'a') == 1.0
assert reciprocal_rank(['a', 'b', 'c'], 'b') == 0.5
assert reciprocal_rank(['a', 'b', 'c'], 'x') == 0.0
""",
}


def run_code_test(answer: str, test_name: str) -> tuple[bool, str]:
    """Compile and execute a tightly restricted generated function against fixed tests."""
    code = _extract_python(answer)
    if not code:
        return False, "no Python function found"
    try:
        tree = ast.parse(code)
        _validate_generated_ast(tree)
    except (SyntaxError, ValueError) as exc:
        return False, str(exc)

    test_code = CODE_TESTS[test_name]
    with tempfile.TemporaryDirectory(prefix="notebuddy-week4-") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(f"{code}\n{test_code}\n", encoding="utf-8")
        try:
            extra_kwargs = {}
            if sys.platform != "win32" and resource is not None:
                extra_kwargs["preexec_fn"] = _limit_child
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                **extra_kwargs,
            )
        except subprocess.TimeoutExpired:
            return False, "test timed out"
    if result.returncode == 0:
        return True, "all fixed tests passed"
    detail = (result.stderr or result.stdout).strip().splitlines()
    return False, detail[-1][:300] if detail else f"exit code {result.returncode}"


def _extract_python(answer: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", answer, re.I | re.S)
    candidate = fenced.group(1) if fenced else answer
    start = candidate.find("def ")
    return candidate[start:].strip() if start >= 0 else ""


def _validate_generated_ast(tree: ast.AST) -> None:
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.With, ast.AsyncWith, ast.Try, ast.Lambda, ast.Global, ast.Nonlocal)
    allowed_calls = {"len", "range", "enumerate", "float", "str", "int"}
    allowed_methods = {"split", "join", "strip"}
    if not tree.body or any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError("candidate must contain only function definitions")
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr not in allowed_methods):
            raise ValueError(f"method is not allowed: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in allowed_calls:
            if not any(isinstance(parent, ast.FunctionDef) and parent.name == node.func.id for parent in tree.body):
                raise ValueError(f"call is not allowed: {node.func.id}")


def _limit_child() -> None:
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    # RLIMIT_AS is unreliable for framework Python processes on macOS. Linux
    # containers use it to cap generated-code memory without weakening tests.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
