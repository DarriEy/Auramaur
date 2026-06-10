"""SEC EDGAR client — tender-offer filing discovery for the odd-lot pillar.

Polls EDGAR full-text search for recent issuer tender offers (form SC TO-I
and amendments) and fetches filing documents for the LLM fine-print read.

SEC fair-access rules: requests MUST carry a descriptive User-Agent with a
contact address, and stay well under 10 req/s — this client is polled every
few hours and touches a handful of filings per cycle, far below the limit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import aiohttp
import structlog

log = structlog.get_logger()

_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
_USER_AGENT = "Auramaur research bot (contact: dae5@hi.is)"

TENDER_FORMS = ["SC TO-I", "SC TO-I/A", "SC 13E4", "SC 13E-4"]


@dataclass
class TenderFiling:
    accession: str          # e.g. 0001193125-26-123456
    cik: str
    company: str            # display name (often "Name (TICKER) (CIK ...)")
    form: str
    filed_at: str           # YYYY-MM-DD
    primary_doc: str        # filename of the main document

    @property
    def ticker(self) -> str | None:
        m = re.search(r"\(([A-Z]{1,5})\)", self.company or "")
        return m.group(1) if m else None

    @property
    def doc_url(self) -> str:
        return _ARCHIVE_URL.format(
            cik=str(int(self.cik)),
            accession=self.accession.replace("-", ""),
            filename=self.primary_doc,
        )


def parse_search_hits(payload: dict) -> list[TenderFiling]:
    """Parse the EDGAR full-text-search JSON into TenderFiling rows.

    Defensive: EDGAR's schema has drifted before; rows missing any required
    field are skipped rather than crashing the cycle.
    """
    out: list[TenderFiling] = []
    for hit in (payload.get("hits", {}) or {}).get("hits", []) or []:
        try:
            src = hit.get("_source", {}) or {}
            # _id is "<accession>:<filename>"
            hit_id = hit.get("_id", "")
            accession, _, filename = hit_id.partition(":")
            ciks = src.get("ciks") or []
            names = src.get("display_names") or []
            if not (accession and filename and ciks):
                continue
            out.append(TenderFiling(
                accession=accession,
                cik=str(ciks[0]),
                company=str(names[0]) if names else "",
                form=str(src.get("file_type") or src.get("form") or ""),
                filed_at=str(src.get("file_date") or ""),
                primary_doc=filename,
            ))
        except Exception as e:  # pragma: no cover - defensive
            log.debug("edgar.parse_hit_error", error=str(e))
    return out


class EdgarClient:
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": _USER_AGENT})
        return self._session

    async def recent_tender_filings(self, days: int = 7) -> list[TenderFiling]:
        """Recent SC TO-I (+amendment) filings via full-text search."""
        session = await self._get_session()
        params = {
            "q": "\"odd lot\"",
            "forms": ",".join(TENDER_FORMS),
            "dateRange": "custom",
            # full-text search accepts ISO dates; widen by `days`
        }
        from datetime import date, timedelta
        params["startdt"] = (date.today() - timedelta(days=days)).isoformat()
        params["enddt"] = date.today().isoformat()
        try:
            async with session.get(
                _SEARCH_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as e:
            log.warning("edgar.search_error", error=str(e)[:120])
            return []
        filings = parse_search_hits(payload)
        log.info("edgar.search_done", hits=len(filings), days=days)
        return filings

    async def fetch_document(self, filing: TenderFiling,
                             max_chars: int = 60000) -> str:
        """Fetch the primary filing document text (HTML tags stripped)."""
        session = await self._get_session()
        try:
            async with session.get(
                filing.doc_url, timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                body = await resp.text(errors="replace")
        except Exception as e:
            log.warning("edgar.fetch_error", accession=filing.accession,
                        error=str(e)[:120])
            return ""
        # crude but adequate tag strip for LLM consumption
        text = re.sub(r"<[^>]+>", " ", body)
        text = re.sub(r"&nbsp;?", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
