import { describe, expect, it } from "vitest";

import { ago, deltaClass, exactTime, liveness, money, price, utcClock } from "./format";

describe("dashboard formatting", () => {
  it("formats exact timestamp tooltips without invalid Intl options", () => {
    expect(exactTime("2026-07-21T12:00:00Z")).toContain("2026");
    expect(exactTime("not-a-date")).toBe("not-a-date");
  });

  it("formats missing and signed financial values", () => {
    expect(money(null)).toBe("n/a");
    expect(money(12.5, true)).toBe("+$12.50");
    expect(money(-12.5, true)).toBe("-$12.50");
    expect(money(-1.3e-15, true)).toBe("$0.00");
    expect(price(null)).toBe("—");
    expect(price(0.625)).toBe("0.625");
  });

  it("uses the cockpit age and liveness thresholds", () => {
    expect(ago(null)).toBe("—");
    expect(ago(59.9)).toBe("59s ago");
    expect(ago(120)).toBe("2m ago");
    expect(liveness(null)).toBe("unknown");
    expect(liveness(89)).toBe("fresh");
    expect(liveness(90)).toBe("stale");
    expect(liveness(600)).toBe("dead");
  });

  it("extracts the UTC clock from an API timestamp", () => {
    expect(utcClock("2026-07-19T20:15:30.000Z")).toBe("20:15:30 UTC");
  });
});

describe("delta styling", () => {
  it("keeps unavailable and unchanged values neutral", () => {
    expect(deltaClass(null)).toBe("");
    expect(deltaClass(0)).toBe("");
  });

  it("colors positive and negative values directionally", () => {
    expect(deltaClass(0.1)).toBe("up");
    expect(deltaClass(-0.1)).toBe("down");
  });
});
