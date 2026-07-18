#!/usr/bin/env python3
"""Env-schema drift checker: manifest vs .env.example vs local .env.

Compares NAMES ONLY. Values are never read into memory as values — each line
is split at the first '=' and everything right of it is discarded immediately.
Nothing right of an '=' is ever printed.

Checks:
  1. every manifest entry (unless in_example: false) appears in .env.example
  2. every .env.example name appears in the manifest
  3. if .env exists: every name in it is known to the manifest
     (unknown names are usually typos or no-ops, e.g. KALSHI_ENABLED)
  4. with --profile live: every `required: live` name is present in .env

Exit 0 clean, 1 on drift.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Nested pydantic-settings overrides (env_nested_delimiter="__") are an open
# namespace mapping onto config; any FOO__BAR name is legitimate.
NESTED = "__"

ROOT = Path(__file__).resolve().parent.parent


def env_names(path: Path, include_commented: bool) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if include_commented and line.startswith("#"):
            line = line.lstrip("# ")
        if "=" not in line or line.startswith("#"):
            continue
        name = line.split("=", 1)[0].strip()
        if NAME_RE.match(name):
            names.add(name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=["paper", "live"], default="paper",
                    help="live additionally requires the `required: live` set in .env")
    args = ap.parse_args()

    manifest_path = ROOT / "deploy" / "env.manifest.yaml"
    manifest: dict[str, dict] = yaml.safe_load(manifest_path.read_text())
    bad_names = [n for n in manifest if not NAME_RE.match(n)]
    if bad_names:
        print(f"DRIFT: malformed manifest names: {sorted(bad_names)}")
        return 1

    example = env_names(ROOT / ".env.example", include_commented=True)
    problems: list[str] = []

    for name, spec in manifest.items():
        if spec.get("in_example", True) and name not in example:
            problems.append(f"manifest name missing from .env.example: {name}")
    for name in sorted(example - manifest.keys()):
        if NESTED in name:
            continue
        problems.append(f".env.example name missing from manifest: {name}")

    dotenv = ROOT / ".env"
    if dotenv.exists():
        local = env_names(dotenv, include_commented=False)
        for name in sorted(local):
            if name in manifest or NESTED in name:
                continue
            problems.append(f"unknown name in .env (typo or no-op?): {name}")
        if args.profile == "live":
            required = {n for n, s in manifest.items() if s.get("required") == "live"}
            for name in sorted(required - local):
                problems.append(f"required-for-live name missing from .env: {name}")
    elif args.profile == "live":
        problems.append("no .env present but --profile live requested")

    if problems:
        print(f"DRIFT ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("env schema clean: manifest, .env.example"
          + (", .env" if dotenv.exists() else "") + " agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
