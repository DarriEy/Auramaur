"""Enforce dependency-audit findings against explicit, expiring exceptions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_SEVERITY = {"info": 0, "low": 1, "moderate": 2, "medium": 2, "high": 3, "critical": 4}


@dataclass(frozen=True, order=True)
class Finding:
    ecosystem: str
    package: str
    advisory: str
    severity: str = ""


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable audit JSON: {exc}") from exc


def pip_findings(report: object) -> set[Finding]:
    if isinstance(report, dict):
        dependencies = report.get("dependencies")
    else:
        dependencies = report
    if not isinstance(dependencies, list):
        raise ValueError("pip-audit report must contain a dependencies list")
    findings: set[Finding] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise ValueError("pip-audit dependency entries must be objects")
        package = str(dependency.get("name", "")).lower()
        vulns = dependency.get("vulns", [])
        if not package or not isinstance(vulns, list):
            raise ValueError("pip-audit dependency is missing name/vulns")
        for vulnerability in vulns:
            if not isinstance(vulnerability, dict) or not vulnerability.get("id"):
                raise ValueError(f"pip-audit vulnerability for {package} has no id")
            findings.add(Finding("pypi", package, str(vulnerability["id"])))
    return findings


def _npm_advisory(via: object, package: str) -> str:
    if isinstance(via, str):
        return f"dependency:{via}"
    if not isinstance(via, dict):
        raise ValueError(f"npm vulnerability for {package} has malformed via entry")
    url = str(via.get("url", "")).rstrip("/")
    if url:
        return url.rsplit("/", 1)[-1]
    if via.get("source") is not None:
        return f"source:{via['source']}"
    raise ValueError(f"npm vulnerability for {package} has no advisory identity")


def npm_findings(report: object, min_severity: str = "high") -> set[Finding]:
    if not isinstance(report, dict) or not isinstance(report.get("vulnerabilities"), dict):
        raise ValueError("npm audit report must contain a vulnerabilities object")
    if min_severity not in _SEVERITY:
        raise ValueError(f"unknown npm severity threshold: {min_severity}")
    findings: set[Finding] = set()
    for key, vulnerability in report["vulnerabilities"].items():
        if not isinstance(vulnerability, dict):
            raise ValueError(f"npm vulnerability {key} must be an object")
        package = str(vulnerability.get("name") or key).lower()
        severity = str(vulnerability.get("severity", "")).lower()
        if severity not in _SEVERITY:
            raise ValueError(f"npm vulnerability for {package} has unknown severity {severity!r}")
        if _SEVERITY[severity] < _SEVERITY[min_severity]:
            continue
        via = vulnerability.get("via", [])
        if not isinstance(via, list):
            raise ValueError(f"npm vulnerability for {package} has malformed via")
        for advisory in via or [f"package:{package}"]:
            findings.add(Finding(
                "npm", package, _npm_advisory(advisory, package), severity))
    return findings


def policy_exceptions(policy: object, today: date) -> dict[tuple[str, str, str], dict]:
    if not isinstance(policy, dict) or policy.get("version") != 1:
        raise ValueError("policy must be an object with version 1")
    exceptions = policy.get("exceptions")
    if not isinstance(exceptions, list):
        raise ValueError("policy exceptions must be a list")
    approved: dict[tuple[str, str, str], dict] = {}
    required = {"ecosystem", "package", "advisory", "expires", "owner", "reason"}
    for item in exceptions:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"policy exception must contain {sorted(required)}")
        key = (
            str(item["ecosystem"]).lower(),
            str(item["package"]).lower(),
            str(item["advisory"]),
        )
        if key in approved:
            raise ValueError(f"duplicate policy exception: {key}")
        try:
            expires = date.fromisoformat(str(item["expires"]))
        except ValueError as exc:
            raise ValueError(f"invalid expiry for {key}: {item['expires']}") from exc
        if expires < today:
            raise ValueError(f"expired policy exception: {key} expired {expires}")
        if not str(item["owner"]).strip() or not str(item["reason"]).strip():
            raise ValueError(f"policy exception lacks owner/reason: {key}")
        approved[key] = item
    return approved


def evaluate(
    policy: object,
    findings: set[Finding],
    *,
    today: date | None = None,
) -> tuple[list[Finding], list[tuple[str, str, str]]]:
    approved = policy_exceptions(policy, today or date.today())
    finding_keys = {(f.ecosystem, f.package, f.advisory) for f in findings}
    unapproved = sorted(f for f in findings if (
        f.ecosystem, f.package, f.advisory) not in approved)
    unused = sorted(key for key in approved if key not in finding_keys)
    return unapproved, unused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--pip-report", type=Path, required=True)
    parser.add_argument("--npm-report", type=Path, required=True)
    parser.add_argument("--npm-min-severity", default="high")
    args = parser.parse_args(argv)
    try:
        policy = _load_json(args.policy)
        findings = pip_findings(_load_json(args.pip_report))
        findings |= npm_findings(
            _load_json(args.npm_report), args.npm_min_severity)
        unapproved, unused = evaluate(policy, findings)
    except ValueError as exc:
        print(f"dependency audit policy error: {exc}", file=sys.stderr)
        return 2
    for key in unused:
        print(f"warning: unused dependency exception: {key}", file=sys.stderr)
    if unapproved:
        print("unapproved dependency vulnerabilities:", file=sys.stderr)
        for finding in unapproved:
            suffix = f" ({finding.severity})" if finding.severity else ""
            print(
                f"  {finding.ecosystem}:{finding.package}:"
                f"{finding.advisory}{suffix}",
                file=sys.stderr,
            )
        return 1
    print(f"dependency audit policy passed ({len(findings)} finding(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
