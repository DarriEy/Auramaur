"""Credentials passed as query parameters must never reach a log line.

aiohttp's ClientResponseError stringifies to include the full request URL. For
a metered API the triggering condition is the *normal* failure — 403 over-quota
or 401 after rotation — so `log.warning(..., error=str(exc)[:120])` on a source
that puts its key first in the query string writes the key to auramaur.log in
cleartext. Truncation is not a mitigation; the key is early in the URL.
"""

from aiohttp import ClientResponseError, RequestInfo
from yarl import URL

from auramaur.data_sources.base import redact_error

_KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCD"


def _client_error(url: str) -> ClientResponseError:
    u = URL(url)
    return ClientResponseError(
        RequestInfo(url=u, method="GET", headers={}, real_url=u),
        (), status=403, message="Forbidden",
    )


def test_congress_key_is_not_logged():
    exc = _client_error(
        f"https://api.congress.gov/v3/bill?api_key={_KEY}&format=json&limit=20"
    )
    out = redact_error(exc, 120)
    assert _KEY not in out
    assert "<redacted>" in out
    # The diagnostic value survives: status, host and non-secret params remain.
    assert "403" in out and "api.congress.gov" in out


def test_bea_userid_and_eia_key_are_not_logged():
    guid = "1A2B3C4D-5E6F-7A8B-9C0D-1E2F3A4B5C6D"
    bea = redact_error(_client_error(
        f"https://apps.bea.gov/api/data?UserID={guid}&method=GetData"), 120)
    assert guid not in bea
    eia = redact_error(_client_error(
        f"https://api.eia.gov/v2/seriesid/X?api_key={_KEY}"), 120)
    assert _KEY not in eia


def test_unknown_parameter_names_are_covered_by_the_shape_backstop():
    """A credential under a name we never enumerated is still redacted."""
    out = redact_error(_client_error(
        f"https://vendor.test/v1/data?subscription_credential={_KEY}&fmt=json"), 200)
    assert _KEY not in out


def test_redaction_precedes_truncation():
    """A short limit must not slice into a key the redactor would have covered."""
    out = redact_error(_client_error(
        f"https://api.congress.gov/v3/bill?api_key={_KEY}"), 80)
    assert _KEY not in out
    assert out[:20] not in _KEY


def test_short_ordinary_values_are_left_readable():
    """Redaction must not destroy the diagnostic content of ordinary params."""
    out = redact_error(_client_error(
        "https://api.example.test/v1/x?format=json&limit=20"), 200)
    assert "format=json" in out and "limit=20" in out


def test_accepts_a_plain_string_or_a_non_http_exception():
    assert redact_error(ValueError("boom"), 50) == "boom"
    assert redact_error("plain text", 50) == "plain text"
