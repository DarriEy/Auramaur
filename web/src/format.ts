export function money(v: number | null | undefined, signed = false): string {
  if (v === null || v === undefined) return "n/a";
  const sign = signed && v > 0 ? "+" : v < 0 ? "-" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function price(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(3);
}

export function deltaClass(value: number | null): "up" | "down" | "" {
  if (value === null || value === 0) return "";
  return value > 0 ? "up" : "down";
}

/** "4s ago" / "3m ago" / "2h ago", matching the cockpit's buckets. */
export function ago(ageSeconds: number | null): string {
  if (ageSeconds === null) return "—";
  if (ageSeconds < 60) return `${Math.floor(ageSeconds)}s ago`;
  if (ageSeconds < 3600) return `${Math.floor(ageSeconds / 60)}m ago`;
  return `${Math.floor(ageSeconds / 3600)}h ago`;
}

/** Cockpit liveness thresholds: fresh < 90s, stale < 600s, dead beyond.
 *  Never-seen (no log line in the tail) is "unknown", not "dead" — a pillar
 *  that hasn't spoken isn't the same signal as one that went silent. */
export function liveness(
  ageSeconds: number | null,
): "fresh" | "stale" | "dead" | "unknown" {
  if (ageSeconds === null) return "unknown";
  if (ageSeconds < 90) return "fresh";
  if (ageSeconds < 600) return "stale";
  return "dead";
}

export function utcClock(iso: string): string {
  return `${iso.slice(11, 19)} UTC`;
}
