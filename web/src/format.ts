export function money(v: number | null | undefined, signed = false): string {
  if (v === null || v === undefined) return "n/a";
  const rounded = Math.round(v * 100) / 100;
  const sign = signed && rounded > 0 ? "+" : rounded < 0 ? "-" : "";
  return `${sign}$${Math.abs(rounded).toLocaleString("en-US", {
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

export function compactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function exactTime(value?: string | null): string {
  if (!value) return "Unknown time";
  const normalized = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value)
    ? `${value.replace(" ", "T")}Z`
    : value;
  const date = new Date(normalized);
  return Number.isFinite(date.getTime())
    ? new Intl.DateTimeFormat("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        timeZoneName: "short",
      }).format(date)
    : value;
}

export function humanize(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
