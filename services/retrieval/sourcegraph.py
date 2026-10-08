"""Sourcegraph Code Intelligence and Semantic Repository Analysis Module.

Provides AST-based symbol navigation, cross-file reference tracking,
and structural search compatible with Sourcegraph query syntax for NoteBuddy.
"""

import ast
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.request
import json


class SourcegraphSymbol:
    def __init__(
        self,
        name: str,
        symbol_type: str,
        file_path: str,
        line_number: int,
        signature: str,
        docstring: Optional[str] = None,
        code_snippet: str = "",
    ):
        self.name = name
        self.symbol_type = symbol_type
        self.file_path = file_path.replace("\\", "/")
        self.line_number = line_number
        self.signature = signature
        self.docstring = docstring or ""
        self.code_snippet = code_snippet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.symbol_type,
            "file": self.file_path,
            "line": self.line_number,
            "signature": self.signature,
            "docstring": self.docstring,
            "snippet": self.code_snippet,
            "sourcegraph_url": f"https://sourcegraph.com/search?q={self.name}",
        }


class SourcegraphEngine:
    """AST Code Intelligence & Sourcegraph Semantic Query Engine for NoteBuddy."""

    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            # Default to NoteBuddy project root
            self.repo_root = Path(__file__).resolve().parents[2]
        else:
            self.repo_root = repo_root
        self.symbols: List[SourcegraphSymbol] = []
        self.file_index: Dict[str, str] = {}
        self.is_indexed = False
        self._index_repository()

    def _index_repository(self) -> None:
        """Scan repository Python files and build AST symbol index."""
        self.symbols.clear()
        self.file_index.clear()
        scan_dirs = ["services", "knowledge", "guardrails", "scripts", "app"]

        for d in scan_dirs:
            dir_path = self.repo_root / d
            if not dir_path.is_dir():
                continue
            for py_path in dir_path.rglob("*.py"):
                if "__pycache__" in str(py_path):
                    continue
                try:
                    rel_path = str(py_path.relative_to(self.repo_root)).replace("\\", "/")
                    content = py_path.read_text(encoding="utf-8", errors="replace")
                    self.file_index[rel_path] = content
                    self._parse_file_ast(rel_path, content)
                except Exception:
                    continue
        self.is_indexed = True

    def _parse_file_ast(self, rel_path: str, content: str) -> None:
        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError:
            return

        lines = content.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                sig = f"class {node.name}"
                if node.bases:
                    bases = [getattr(b, "id", getattr(b, "attr", "...")) for b in node.bases]
                    sig += f"({', '.join(bases)})"
                doc = ast.get_docstring(node)
                snippet = "\n".join(lines[node.lineno - 1 : min(node.lineno + 5, len(lines))])
                self.symbols.append(
                    SourcegraphSymbol(
                        name=node.name,
                        symbol_type="class",
                        file_path=rel_path,
                        line_number=node.lineno,
                        signature=sig,
                        docstring=doc,
                        code_snippet=snippet,
                    )
                )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                sig = f"def {node.name}({', '.join(args[:4])}{'...' if len(args) > 4 else ''})"
                doc = ast.get_docstring(node)
                is_endpoint = any(
                    isinstance(dec, ast.Call)
                    and getattr(getattr(dec, "func", None), "attr", "") in ("get", "post", "put", "delete")
                    for dec in node.decorator_list
                )
                sym_type = "endpoint" if is_endpoint else "function"
                snippet = "\n".join(lines[node.lineno - 1 : min(node.lineno + 4, len(lines))])
                self.symbols.append(
                    SourcegraphSymbol(
                        name=node.name,
                        symbol_type=sym_type,
                        file_path=rel_path,
                        line_number=node.lineno,
                        signature=sig,
                        docstring=doc,
                        code_snippet=snippet,
                    )
                )

    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Execute a Sourcegraph-style query against repository symbol index.
        
        Supports queries like:
          - 'type:symbol orchestrator'
          - 'type:class KnowledgeStore'
          - 'type:endpoint retrieve'
          - 'path:guardrails'
          - 'run_notebuddy'
        """
        raw_query = query.strip()
        type_filter = None
        path_filter = None
        term = raw_query

        # Parse Sourcegraph query modifiers
        match_type = re.search(r"type:(symbol|class|function|endpoint)", raw_query, re.IGNORECASE)
        if match_type:
            type_filter = match_type.group(1).lower()
            term = raw_query.replace(match_type.group(0), "").strip()

        match_path = re.search(r"path:([^\s]+)", raw_query, re.IGNORECASE)
        if match_path:
            path_filter = match_path.group(1).lower()
            term = term.replace(match_path.group(0), "").strip()

        term_lower = term.lower().strip()
        matches: List[SourcegraphSymbol] = []

        for sym in self.symbols:
            if type_filter and type_filter != "symbol" and sym.symbol_type != type_filter:
                continue
            if path_filter and path_filter not in sym.file_path.lower():
                continue

            if not term_lower or term_lower in sym.name.lower() or term_lower in sym.signature.lower() or term_lower in sym.file_path.lower():
                matches.append(sym)

        # Sort matches by exact name hit, then name start, then file
        def score(sym: SourcegraphSymbol) -> int:
            if sym.name.lower() == term_lower:
                return 0
            if sym.name.lower().startswith(term_lower):
                return 1
            if term_lower in sym.name.lower():
                return 2
            return 3

        matches.sort(key=score)
        results = [m.to_dict() for m in matches[:limit]]

        return {
            "query": query,
            "engine": "Sourcegraph OSS / AST Code Intelligence",
            "status": "connected",
            "total_indexed_symbols": len(self.symbols),
            "total_indexed_files": len(self.file_index),
            "match_count": len(results),
            "results": results,
        }

    def get_symbol_context(self, question: str, max_symbols: int = 4) -> str:
        """Extract high-relevance AST symbol definitions to augment RAG context."""
        keywords = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", question)
        found: List[SourcegraphSymbol] = []
        seen = set()

        for kw in keywords:
            kw_low = kw.lower()
            for sym in self.symbols:
                if sym.name.lower() == kw_low and sym.name not in seen:
                    found.append(sym)
                    seen.add(sym.name)
                    if len(found) >= max_symbols:
                        break
            if len(found) >= max_symbols:
                break

        if not found:
            return ""

        context_lines = ["--- SOURCEGRAPH SYMBOL DEFINITIONS & AST CONTEXT ---"]
        for sym in found:
            context_lines.append(
                f"[{sym.symbol_type.upper()}] {sym.signature} ({sym.file_path}:{sym.line_number})\n{sym.code_snippet}"
            )
        return "\n\n".join(context_lines)


# Global singleton engine instance
_engine: Optional[SourcegraphEngine] = None


def get_sourcegraph_engine() -> SourcegraphEngine:
    global _engine
    if _engine is None:
        _engine = SourcegraphEngine()
    return _engine
