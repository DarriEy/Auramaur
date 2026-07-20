import { useEffect, useRef, useState } from "react";
import type { Envelope } from "./types";

/** What the connection banner should say. Derived, never stored. */
export type Phase =
  | "connecting" // nothing received yet
  | "live" // fresh data, service healthy
  | "degraded" // service reachable but reporting an error (e.g. bad DB)
  | "disconnected"; // no fresh data — API unreachable or stream stalled

export interface DashboardStatus {
  envelope: Envelope | null;
  phase: Phase;
  /** Seconds since the last envelope arrived over ANY transport. */
  receivedAgo: number | null;
}

const POLL_FALLBACK_MS = 5000;
const STALE_AFTER_S = 8; // > server refresh (2s), < long enough to worry

/**
 * Data over two transports: an SSE stream for liveness, plus a one-shot fetch
 * at mount (fast first paint, and proof the API is reachable even where a
 * proxy breaks SSE) that continues as a poll whenever the stream is down.
 * EventSource auto-reconnects on its own.
 */
export function useDashboardState(): DashboardStatus {
  const [envelope, setEnvelope] = useState<Envelope | null>(null);
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const [, setTick] = useState(0); // 1s re-render so ages/staleness move
  const sseUp = useRef(false);

  useEffect(() => {
    let disposed = false;

    const apply = (env: Envelope) => {
      if (disposed) return;
      setEnvelope(env);
      setReceivedAt(Date.now());
    };

    const poll = async () => {
      try {
        const resp = await fetch("/api/state");
        if (resp.ok) apply(await resp.json());
      } catch {
        /* API unreachable — phase derivation reports it */
      }
    };
    void poll();

    const es = new EventSource("/api/stream");
    es.addEventListener("state", (ev) => {
      sseUp.current = true;
      apply(JSON.parse((ev as MessageEvent).data));
    });
    es.onerror = () => {
      sseUp.current = false;
    };

    const pollTimer = setInterval(() => {
      if (!sseUp.current) void poll();
    }, POLL_FALLBACK_MS);
    const tickTimer = setInterval(() => setTick((t) => t + 1), 1000);

    return () => {
      disposed = true;
      es.close();
      clearInterval(pollTimer);
      clearInterval(tickTimer);
    };
  }, []);

  const receivedAgo = receivedAt === null ? null : (Date.now() - receivedAt) / 1000;
  let phase: Phase;
  if (envelope === null) {
    phase = "connecting";
  } else if (receivedAgo !== null && receivedAgo > STALE_AFTER_S) {
    phase = "disconnected";
  } else if (!envelope.ok) {
    phase = "degraded";
  } else {
    phase = "live";
  }

  return { envelope, phase, receivedAgo };
}
