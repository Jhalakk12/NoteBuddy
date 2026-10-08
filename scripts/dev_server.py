#!/usr/bin/env python3
"""Serve the static UI and proxy /api to the local Application Service."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "ui"
APPLICATION_URL = os.getenv("APPLICATION_URL", "http://127.0.0.1:8000").rstrip("/")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://127.0.0.1:8001").rstrip("/")
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8002").rstrip("/")
DATA_URL = os.getenv("DATA_URL", "http://127.0.0.1:8003").rstrip("/")
HOST = os.getenv("WEBSITE_HOST", "127.0.0.1")
PORT = int(os.getenv("WEBSITE_PORT", "8080"))
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class NoteBuddyHandler(SimpleHTTPRequestHandler):
    """Static-file handler with a same-origin Application API gateway."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if self._is_api_path():
            self._proxy()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self._is_api_path():
            self._proxy()
            return
        self.send_error(404)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if self._is_api_path():
            self._proxy()
            return
        self.send_error(404)

    def _proxy(self) -> None:
        prefixes = {
            "/api/": APPLICATION_URL,
            "/retrieval-api/": RETRIEVAL_URL,
            "/llm-api/": LLM_URL,
            "/data-api/": DATA_URL,
        }
        prefix, base_url = next(
            (item for item in prefixes.items() if self.path.startswith(item[0]))
        )
        target = f"{base_url}/{self.path.removeprefix(prefix)}"
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        forwarded_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        request = Request(
            target,
            data=body,
            headers=forwarded_headers,
            method=self.command,
        )
        try:
            with urlopen(request, timeout=360) as response:
                self._forward_response(response.status, response.headers, response.read())
        except HTTPError as exc:
            self._forward_response(exc.code, exc.headers, exc.read())
        except (URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", str(exc))
            message = f'{{"detail":"Application Service unavailable: {reason}"}}'.encode()
            self._forward_response(
                502,
                {"Content-Type": "application/json", "Content-Length": str(len(message))},
                message,
            )

    def _is_api_path(self) -> bool:
        return self.path.startswith(("/api/", "/retrieval-api/", "/llm-api/", "/data-api/"))

    def _forward_response(self, status: int, headers, body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-length":
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    handler = partial(NoteBuddyHandler, directory=str(UI_ROOT))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    print(f"NoteBuddy website: http://{HOST}:{PORT}")
    print(f"Same-origin API proxy: /api -> {APPLICATION_URL}")
    print(f"Retrieval UI proxy: /retrieval-api -> {RETRIEVAL_URL}")
    print(f"LLM UI proxy: /llm-api -> {LLM_URL}")
    print(f"Data UI proxy: /data-api -> {DATA_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
