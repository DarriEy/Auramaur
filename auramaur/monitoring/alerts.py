"""Optional alert notifications via Telegram and Discord."""

from __future__ import annotations

import aiohttp
import structlog

log = structlog.get_logger()


class AlertManager:
    """Send alerts via Telegram and/or Discord webhooks."""

    def __init__(self, telegram_bot_token: str = "", telegram_chat_id: str = "", discord_webhook_url: str = ""):
        self.telegram_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_url = discord_webhook_url
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def send(self, message: str, level: str = "info") -> None:
        """Send alert to all configured channels."""
        if self.telegram_token and self.telegram_chat_id:
            await self._send_telegram(message)
        if self.discord_url:
            await self._send_discord(message)

    async def _send_telegram(self, message: str) -> None:
        session = await self._ensure_session()
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            async with session.post(url, json={"chat_id": self.telegram_chat_id, "text": message, "parse_mode": "Markdown"}) as resp:
                if resp.status != 200:
                    log.warning("alert.telegram_error", status=resp.status)
        except Exception as e:
            log.error("alert.telegram_error", error=str(e))

    async def _send_discord(self, message: str) -> None:
        session = await self._ensure_session()
        try:
            async with session.post(self.discord_url, json={"content": message}) as resp:
                if resp.status not in (200, 204):
                    log.warning("alert.discord_error", status=resp.status)
        except Exception as e:
            log.error("alert.discord_error", error=str(e))
