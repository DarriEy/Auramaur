"""Credentials passed as query parameters must never reach a log line.

aiohttp's ClientResponseError stringifies to include the full request URL. For
a metered API the triggering condition is the *normal* failure — 403 over-quota
or 401 after rotation — so `log.warning(..., error=str(exc)[:120])` on a source
that puts its key first in the query string writes the key to auramaur.log in
cleartext. Truncation is not a mitigation; the key is early in the URL.
"""

from types import SimpleNamespace
import pytest
from aiohttp import ClientResponseError, RequestInfo
from yarl import URL

from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.base import redact_error

_KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCD"


def _client_error(url: str, *, encoded: bool = False) -> ClientResponseError:
    # encoded=True pins the wire form: yarl decodes %XX escapes when it parses a
    # plain string, and it is the escaped form that reaches the log line.
    u = URL(url, encoded=encoded)
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


def test_percent_encoded_secrets_do_not_escape_the_shape_backstop():
    """A standard-base64 credential is percent-encoded on the wire.

    aiohttp turns the reserved characters of a base64 key into %XX escapes
    ('+' -> %2B, '=' -> %3D). If '%' is absent from the opaque-value class the
    escape cuts the value into runs shorter than the 20-char threshold and an
    un-enumerated parameter name carries the whole secret through intact.
    """
    key = "AbCdEfGhIj+KlMnOpQrStUvWxYz01="  # standard-base64 alphabet
    on_the_wire = str(URL("https://vendor.test/v1/data").with_query(
        {"cred": key, "format": "json"}))
    assert "%2B" in on_the_wire and "%3D" in on_the_wire, "aiohttp escapes these"

    out = redact_error(_client_error(on_the_wire, encoded=True), 200)
    assert "KlMnOpQrStUvWxYz01" not in out
    assert "<redacted>" in out
    # The diagnostic content of an ordinary parameter is untouched.
    assert "format=json" in out

    # Same defect when the caller percent-encodes the key itself.
    verbatim = redact_error(_client_error(
        "https://vendor.test/v1/data?cred=AbCdEfGhIj%2FKlMnOpQrStUvWxYz0123",
        encoded=True), 200)
    assert "KlMnOpQrStUvWxYz0123" not in verbatim
    assert "<redacted>" in verbatim


def test_redaction_precedes_truncation():
    """A short limit must not slice into a key the redactor would have covered."""
    out = redact_error(_client_error(
        f"https://api.congress.gov/v3/bill?api_key={_KEY}"), 80)
    assert _KEY not in out
    assert out[:20] not in _KEY


def test_short_ordinary_values_are_left_readable():
    """Redaction must not destroy the diagnostic content of ordinary params."""
    out = redact_error(_client_error(
        "https://api.example.test/v1/x?format=json&limit=20&series=UNRATE"), 200)
    assert "format=json" in out and "limit=20" in out and "series=UNRATE" in out


def test_accepts_a_plain_string_or_a_non_http_exception():
    assert redact_error(ValueError("boom"), 50) == "boom"
    assert redact_error("plain text", 50) == "plain text"


@pytest.mark.asyncio
async def test_aggregator_catch_all_redacts_before_the_row_is_persisted():
    """The aggregator is the widest sink and the only one that outlives the log.

    Whatever a source lets escape lands here with a 500-char window and is
    written to source_fetches.error, which the read-only dashboard renders.
    """
    class LeakySource:
        source_name = "leaky"
        categories = None

        async def fetch(self, query, limit=20):
            raise _client_error(
                f"https://vendor.test/v1/data?api_key={_KEY}&format=json")

        async def close(self):
            pass

    captured: dict = {}
    observer = SimpleNamespace(ingestion=lambda **kwargs: captured.update(kwargs))
    await Aggregator([LeakySource()], observer=observer).gather("question")

    error_column = [row[5] for row in captured["fetch_rows"] if row[2] == "error"]
    assert error_column, "the failing source should have produced an error row"
    assert _KEY not in error_column[0]
    assert "<redacted>" in error_column[0]
    # Still diagnostic: the venue and the ordinary parameter survive.
    assert "vendor.test" in error_column[0] and "format=json" in error_column[0]
# ---------------------------------------------------------------------------
# The opaque-value backstop must terminate at a delimiter, not enumerate the
# characters a secret is allowed to contain. A base64 credential defeats a
# character class: '/', '+' and a percent-encoded '%2F' all fall outside
# [A-Za-z0-9_-], so the run before the first one is under the length threshold
# and nothing matches at all.
# ---------------------------------------------------------------------------

_B64_SLASH = "aB3dEf/GhIjKlMnOpQrStUvWxYz0123456789"
_B64_PLUS = "aB3dEf+GhIjKlMnOpQrStUvWxYz0123456789"
_PCT_ENC = "aB3dEf%2FGhIjKlMnOpQrStUvW0123456789"
_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"


@pytest.mark.parametrize("secret", [_B64_SLASH, _B64_PLUS, _PCT_ENC, _JWT])
def test_base64_and_jwt_secrets_are_redacted_under_unknown_param_names(secret):
    """The param name is deliberately one we never enumerated."""
    out = redact_error(
        _client_error(f"https://vendor.test/v1/data?subscription_credential={secret}&fmt=json"),
        300,
    )
    assert secret not in out
    assert "<redacted>" in out
    # Partial exposure is still exposure — no run of the secret may survive.
    assert secret[:12] not in out


def test_backstop_does_not_swallow_short_diagnostic_params():
    """Over-redaction is safe but useless; short ordinary params must survive."""
    out = redact_error(
        _client_error("https://api.example.test/v1/x?format=json&limit=20&page=3"), 300)
    assert "format=json" in out and "limit=20" in out and "page=3" in out
