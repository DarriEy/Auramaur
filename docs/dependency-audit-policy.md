# Dependency audit policy

CI audits the locked Python runtime set and the web lockfile. Findings are
blocking unless they match an entry in
`.github/dependency-audit-policy.json`.

Prefer upgrading or removing the dependency. An exception is a temporary
risk acceptance and must contain:

- `ecosystem`: `pypi` or `npm`
- `package`: normalized package name
- `advisory`: the exact advisory identity printed by the policy check
- `expires`: ISO date after which CI fails until the exception is removed or
  renewed
- `owner`: person or team responsible for remediation
- `reason`: concrete upgrade blocker and compensating control

Example:

```json
{
  "ecosystem": "npm",
  "package": "example-package",
  "advisory": "GHSA-abcd-1234-5678",
  "expires": "2026-09-01",
  "owner": "repository-maintainers",
  "reason": "Patched release breaks the production bundler; input is not user-controlled."
}
```

Expired or malformed exceptions fail CI even when the advisory has
disappeared. Unused current exceptions produce a warning so they can be
removed. npm findings below `high` remain visible in the JSON report but do
not block; all pip-audit findings block because that report does not expose a
stable severity field.
