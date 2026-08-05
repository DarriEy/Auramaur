"""Mechanical AST sweep for defect classes, applied uniformly to every module.

Judgment-driven review goes where the money is and leaves quiet code unread.
Both real findings in the 2026-08-05 audit were the same shape: a defense the
codebase had already written, documented, and applied — missed at one sibling
call site. That class is invisible to attention-ordered reading and trivial for
a per-file pass.

Each dimension below is grounded in an actual finding, not a generic lint rule:

  unbounded-parse    CPU-bound parse on the event loop with no offload.
                     rss.py caps + threads + deadlines its feedparser call
                     after a 25-minute parse froze exits (2026-07-23);
                     google_trends.py parsed the same way with none of it.
  untrusted-repr     A prompt template fed a bare ``str()``/f-string instead of
                     the scrubbing helper. nlp/prompts.py routes evidence and
                     market context through format_untrusted_text(); the
                     cross-exchange matcher interpolated ``str(list_a)``.
  blocking-in-async  Sync network/sleep inside a coroutine — same event-loop
                     stall as unbounded-parse, different cause.
  naive-datetime     Local-time day boundary where the ledger uses UTC.
                     transfers.py:_today() is UTC; broker/pnl.py is not.
  swallowed-error    ``except: pass`` on a money path, where a silent failure
                     is indistinguishable from success.

Usage:
    uv run python scripts/sweep_invariants.py auramaur
    uv run python scripts/sweep_invariants.py auramaur --baseline .sweep-baseline.json
    uv run python scripts/sweep_invariants.py auramaur --write-baseline .sweep-baseline.json

Exits 1 when findings appear that are not in the baseline, so the sweep can be
adopted on a large codebase without fixing everything first.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Parses that are CPU-bound enough that a hostile or malformed body stalls the
# loop for minutes. json.loads is C-fast and deliberately excluded.
_CPU_PARSERS = {
    "fromstring", "parse", "parseString", "BeautifulSoup", "read_html",
}
_CPU_PARSER_ROOTS = {"ET", "etree", "ElementTree", "feedparser", "minidom", "bs4"}

# Calls that move work off the event loop.
_OFFLOADERS = {"to_thread", "run_in_executor", "wait_for"}

_BLOCKING_ROOTS = {"requests", "urllib", "httpx"}
_BLOCKING_ATTRS = {"sleep", "get", "post", "put", "delete", "request", "urlopen"}

_NAIVE_TIME = {"utcnow", "today"}

# Directories where a silently swallowed exception is a money-correctness issue.
_MONEY_PATHS = ("broker/", "treasury/", "exchange/", "risk/")


@dataclass(frozen=True, order=True)
class Finding:
    file: str
    line: int
    dimension: str
    detail: str

    def key(self) -> str:
        """Line-independent identity, so unrelated edits don't churn baselines."""
        return f"{self.file}::{self.dimension}::{self.detail}"


def _root_name(node: ast.AST) -> str:
    """Leftmost identifier of a dotted call, e.g. ``ET`` in ``ET.fromstring``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


class _Sweeper(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.rel = relpath
        self.findings: list[Finding] = []
        self._async_depth = 0
        self._offload_depth = 0

    # -- context tracking ------------------------------------------------

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._async_depth += 1
        self.generic_visit(node)
        self._async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A sync def nested in a coroutine is its own context; the parse inside
        # it only blocks if that def is itself called inline. Treat as sync.
        outer, self._async_depth = self._async_depth, 0
        self.generic_visit(node)
        self._async_depth = outer

    # -- dimensions ------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        attr = _attr_name(func)
        root = _root_name(func)

        offload = attr in _OFFLOADERS
        if offload:
            self._offload_depth += 1

        if (
            self._async_depth
            and not self._offload_depth
            and attr in _CPU_PARSERS
            and root in _CPU_PARSER_ROOTS
        ):
            self._add(node, "unbounded-parse",
                      f"{root}.{attr} on the event loop with no to_thread/wait_for")

        if (
            self._async_depth
            and not self._offload_depth
            and root in _BLOCKING_ROOTS
            and attr in _BLOCKING_ATTRS
        ):
            self._add(node, "blocking-in-async", f"sync {root}.{attr} inside a coroutine")

        if root == "time" and attr == "sleep" and self._async_depth:
            self._add(node, "blocking-in-async", "time.sleep inside a coroutine")

        if attr in _NAIVE_TIME and root in {"datetime", "date"}:
            self._add(node, "naive-datetime", f"{root}.{attr}() has no timezone")
        if attr == "now" and root == "datetime" and not node.args and not node.keywords:
            self._add(node, "naive-datetime", "datetime.now() has no tz argument")

        self._check_prompt_format(node, attr)

        self.generic_visit(node)
        if offload:
            self._offload_depth -= 1

    def _check_prompt_format(self, node: ast.Call, attr: str) -> None:
        """``SOME_PROMPT.format(x=str(untrusted))`` — bypasses the scrubbers."""
        if attr != "format" or not isinstance(node.func, ast.Attribute):
            return
        target = node.func.value
        name = target.id if isinstance(target, ast.Name) else ""
        if "PROMPT" not in name.upper():
            return
        for kw in node.keywords:
            for sub in ast.walk(kw.value):
                if isinstance(sub, ast.JoinedStr):
                    self._add(node, "untrusted-repr",
                              f"{name}.format({kw.arg}=) interpolates an f-string")
                    break
                if isinstance(sub, ast.Call) and _attr_name(sub.func) in {"str", "repr"}:
                    self._add(node, "untrusted-repr",
                              f"{name}.format({kw.arg}=) passes a bare "
                              f"{_attr_name(sub.func)}() instead of a format_* scrubber")
                    break

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if any(self.rel.replace("\\", "/").find(p) >= 0 for p in _MONEY_PATHS):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                kind = "bare except" if node.type is None else "except"
                self._add(node, "swallowed-error", f"{kind}: pass on a money path")
        self.generic_visit(node)

    def _add(self, node: ast.AST, dimension: str, detail: str) -> None:
        self.findings.append(
            Finding(self.rel, getattr(node, "lineno", 0), dimension, detail)
        )


def sweep(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if parts & {".venv", "node_modules", "__pycache__", "tests"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        sweeper = _Sweeper(str(path))
        sweeper.visit(tree)
        findings.extend(sweeper.findings)
    return sorted(findings)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, help="package directory to sweep")
    ap.add_argument("--baseline", type=Path, help="JSON of accepted finding keys")
    ap.add_argument("--write-baseline", type=Path, help="record current findings and exit 0")
    ap.add_argument("--dimension", action="append", help="restrict to these dimensions")
    args = ap.parse_args(argv)

    findings = sweep(args.root)
    if args.dimension:
        findings = [f for f in findings if f.dimension in set(args.dimension)]

    if args.write_baseline:
        args.write_baseline.write_text(
            json.dumps(sorted(f.key() for f in findings), indent=2) + "\n"
        )
        print(f"baseline: {len(findings)} findings -> {args.write_baseline}")
        return 0

    accepted: set[str] = set()
    if args.baseline and args.baseline.exists():
        accepted = set(json.loads(args.baseline.read_text()))

    new = [f for f in findings if f.key() not in accepted]

    by_dim: dict[str, int] = {}
    for f in findings:
        by_dim[f.dimension] = by_dim.get(f.dimension, 0) + 1
    for dim in sorted(by_dim):
        print(f"{dim:20} {by_dim[dim]:4}")
    print(f"{'TOTAL':20} {len(findings):4}   ({len(new)} not in baseline)")

    if new:
        print()
        for f in new:
            print(f"{f.file}:{f.line}  [{f.dimension}]  {f.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
