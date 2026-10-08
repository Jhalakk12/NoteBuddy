#!/usr/bin/env python3
"""Run NoteBuddy's four APIs and static website for local development."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CHROMA_PORT = 8010


def verify_ollama() -> None:
    try:
        with urlopen("http://127.0.0.1:11434/api/version", timeout=3) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama returned HTTP {response.status}")
    except (URLError, TimeoutError) as exc:
        raise SystemExit(
            "Ollama is not reachable at http://127.0.0.1:11434. "
            "Start it with `brew services start ollama` or `ollama serve`."
        ) from exc


def wait_for_url(url: str, service_name: str, timeout_seconds: int = 30) -> None:
    """Wait until a local dependency accepts HTTP requests."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError(f"{service_name} did not become ready at {url}")


def command(module: str, port: int) -> list[str]:
    return [
        PYTHON,
        "-m",
        "uvicorn",
        module,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def main() -> None:
    verify_ollama()
    base_env = os.environ.copy()
    base_env.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    base_env.setdefault("CHROMA_PATH", "data/chroma")
    base_env.setdefault("CHROMA_HOST", "127.0.0.1")
    base_env.setdefault("CHROMA_PORT", str(CHROMA_PORT))
    base_env.setdefault("DOCUMENT_PATH", "data/documents")

    services = [
        ("Retrieval", command("services.retrieval.main:app", 8001), base_env),
        ("LLM", command("services.llm.main:app", 8002), base_env),
        ("Data", command("services.data.main:app", 8003), base_env),
        (
            "Application",
            command("services.application.main:app", 8000),
            {
                **base_env,
                "RETRIEVAL_SERVICE_URL": "http://127.0.0.1:8001",
                "LLM_SERVICE_URL": "http://127.0.0.1:8002",
                "DATA_SERVICE_URL": "http://127.0.0.1:8003",
            },
        ),
        (
            "Website",
            [PYTHON, "scripts/dev_server.py"],
            base_env,
        ),
    ]

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        chroma_exe = (
            shutil.which("chroma")
            or str(Path(PYTHON).parent / "Scripts" / "chroma.exe")
            or str(Path(PYTHON).with_name("chroma"))
        )
        chroma_command = [
            chroma_exe,
            "run",
            "--path",
            str(PROJECT_ROOT / base_env["CHROMA_PATH"]),
            "--host",
            base_env["CHROMA_HOST"],
            "--port",
            base_env["CHROMA_PORT"],
        ]
        chroma = subprocess.Popen(chroma_command, cwd=PROJECT_ROOT, env=base_env)
        processes.append(("Chroma", chroma))
        wait_for_url(
            f"http://{base_env['CHROMA_HOST']}:{base_env['CHROMA_PORT']}/api/v2/heartbeat",
            "Chroma",
        )

        for name, args, environment in services:
            process = subprocess.Popen(args, cwd=PROJECT_ROOT, env=environment)
            processes.append((name, process))
        print("\nNoteBuddy is starting:")
        print("  Website: http://127.0.0.1:8080")
        print("  Backend: Application, Retrieval, LLM, Data, and Chroma are local-only")
        print("\nPress Ctrl+C to stop all NoteBuddy processes.\n")

        while True:
            for name, process in processes:
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"{name} stopped unexpectedly with code {return_code}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
