FROM node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3 AS node_runtime

# Web dashboard SPA — built here so the runtime image serves static files only.
FROM node_runtime AS webbuild
WORKDIR /webbuild
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY web ./
RUN npm run build

FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS runtime

ARG CLAUDE_CODE_VERSION=2.1.214

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    PATH=/app/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gosu passwd sqlite3 tini \
    && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && pip install --no-cache-dir "uv==0.8.15" \
    && rm -rf /var/lib/apt/lists/* /root/.cache

WORKDIR /app
# Dependencies first, in their own layer: source edits must not invalidate
# the expensive torch/sentence-transformers install. The cache mount keeps
# downloaded wheels across builds even when uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra kalshi --extra ibkr --extra web \
    && uv pip install --python .venv/bin/python \
       --index-url https://download.pytorch.org/whl/cpu "torch==2.12.0" \
    && uv pip install --python .venv/bin/python "sentence-transformers==5.5.1"
COPY README.md ./
COPY auramaur ./auramaur
COPY config ./config
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra kalshi --extra ibkr --extra web
COPY --from=webbuild /webbuild/dist /app/web/dist
ENV AURAMAUR_WEB_DIST=/app/web/dist

RUN groupadd --gid 10001 auramaur \
    && useradd --uid 10001 --gid 10001 --create-home auramaur \
    && mkdir -p /app/state /app/logs /home/auramaur/.claude \
    && chown -R auramaur:auramaur /app /home/auramaur

COPY --chown=auramaur:auramaur deploy/docker/auramaur-entrypoint.sh \
     deploy/docker/auramaur-web-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/auramaur-entrypoint.sh"]
CMD ["auramaur", "run", "--hybrid"]

HEALTHCHECK --interval=60s --timeout=15s --start-period=90s --retries=3 \
    CMD ["python", "-m", "auramaur.health"]
