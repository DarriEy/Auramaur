"""The observational source-presence split must be arithmetic, not vibes:
Brier buckets keyed strictly on ranked-evidence presence, sources never
credited on forecasts they didn't feed, and the confounded nature is the
caller's problem to caveat — this function just has to count honestly."""

from auramaur.cli.reporting import source_presence_stats


def _snap(p, outcome, runs):
    return {"calibrated_probability": p, "actual_outcome": outcome,
            "evidence_run_ids": runs}


def test_presence_split_and_delta_direction():
    # bluesky present on the two GOOD calls, absent on the two bad ones.
    snapshots = [
        _snap(0.9, 1, ["r1"]),   # brier 0.01, bluesky present
        _snap(0.1, 0, ["r2"]),   # brier 0.01, bluesky present
        _snap(0.9, 0, ["r3"]),   # brier 0.81, absent
        _snap(0.1, 1, ["r4"]),   # brier 0.81, absent
    ]
    sources_by_run = {"r1": {"bluesky", "rss"}, "r2": {"bluesky", "rss"},
                      "r3": {"rss"}, "r4": {"rss"}}
    stats = {row[0]: row for row in
             source_presence_stats(snapshots, sources_by_run)}
    src, nw, bw, nwo, bwo, delta = stats["bluesky"]
    assert (nw, nwo) == (2, 2)
    assert abs(bw - 0.01) < 1e-9 and abs(bwo - 0.81) < 1e-9
    assert delta < -0.7  # much better with
    # rss was present everywhere -> no without-bucket, no delta claimable.
    assert stats["rss"][3] == 0 and stats["rss"][5] is None


def test_source_absent_from_all_forecast_runs_is_not_invented():
    snapshots = [_snap(0.5, 1, ["r1"])]
    stats = source_presence_stats(snapshots, {"r1": {"rss"},
                                              "r9": {"twitter"}})
    names = [row[0] for row in stats]
    assert "twitter" not in names  # never present on any forecast: no row


def test_empty_inputs_yield_empty():
    assert source_presence_stats([], {}) == []
