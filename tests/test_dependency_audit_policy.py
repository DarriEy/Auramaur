from datetime import date

import pytest

from scripts.check_dependency_audit import (
    Finding,
    evaluate,
    npm_findings,
    pip_findings,
    policy_exceptions,
)


def _policy(*exceptions):
    return {"version": 1, "exceptions": list(exceptions)}


def _exception(**overrides):
    item = {
        "ecosystem": "pypi",
        "package": "demo",
        "advisory": "GHSA-demo",
        "expires": "2026-09-01",
        "owner": "repository-maintainers",
        "reason": "Upgrade is blocked by an upstream compatibility issue.",
    }
    item.update(overrides)
    return item


def test_pip_report_extracts_advisory_identity():
    report = {"dependencies": [
        {"name": "Demo", "version": "1", "vulns": [{"id": "GHSA-demo"}]},
        {"name": "clean", "version": "2", "vulns": []},
    ]}
    assert pip_findings(report) == {Finding("pypi", "demo", "GHSA-demo")}


def test_npm_report_filters_below_threshold_and_extracts_url_id():
    report = {"vulnerabilities": {
        "direct": {
            "name": "Direct", "severity": "high",
            "via": [{"url": "https://github.com/advisories/GHSA-node"}],
        },
        "low": {"name": "low", "severity": "low", "via": ["transitive"]},
    }}
    assert npm_findings(report) == {
        Finding("npm", "direct", "GHSA-node", "high")}


def test_unapproved_finding_fails_policy():
    finding = Finding("pypi", "demo", "GHSA-demo")
    unapproved, unused = evaluate(_policy(), {finding}, today=date(2026, 8, 5))
    assert unapproved == [finding]
    assert unused == []


def test_current_owned_exception_approves_exact_finding():
    finding = Finding("pypi", "demo", "GHSA-demo")
    unapproved, unused = evaluate(
        _policy(_exception()), {finding}, today=date(2026, 8, 5))
    assert unapproved == []
    assert unused == []


def test_expired_exception_is_configuration_error():
    with pytest.raises(ValueError, match="expired policy exception"):
        policy_exceptions(
            _policy(_exception(expires="2026-08-04")), date(2026, 8, 5))


@pytest.mark.parametrize("field", ["owner", "reason"])
def test_exception_requires_accountability_fields(field):
    with pytest.raises(ValueError, match="lacks owner/reason"):
        policy_exceptions(
            _policy(_exception(**{field: ""})), date(2026, 8, 5))


def test_unused_exception_is_reported_for_cleanup():
    unapproved, unused = evaluate(
        _policy(_exception()), set(), today=date(2026, 8, 5))
    assert unapproved == []
    assert unused == [("pypi", "demo", "GHSA-demo")]
