"""Web dashboard — a read-only FastAPI service over the bot's SQLite state.

Runs as a separate process from the bot (``auramaur web``). Opens the database
with SQLite's ``mode=ro`` URI so it can never contend with — or corrupt — the
live daemon's writes, and reuses the cockpit's query layer so the numbers can
never diverge from ``auramaur cockpit``.

Kept out of the bot's runtime deps on purpose (the ``web`` extra), same as the
agent MCP bridge: the bot never imports this package.
"""
