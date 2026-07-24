import asyncio

import pytest

from auramaur.evaluation.runner import (
    AdapterResponse, ArmSpec, Episode, ExplorationPolicy,
    IntelligenceEvalRunner, TreatmentSpec,
)


class Adapter:
    def __init__(self, outputs):
        self.outputs = outputs
        self.requests = []
        self.active = self.peak = 0

    async def generate(self, request):
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        output = self.outputs(request) if callable(self.outputs) else self.outputs[request.sample_index]
        return AdapterResponse(output, {"tokens": 7, "stage": request.stage})


def spec(adapter, policy=ExplorationPolicy.SINGLE, count=1, name="t"):
    return TreatmentSpec(name, ArmSpec(name + "-arm", "fake", adapter), policy, count, 42)


@pytest.mark.asyncio
async def test_pairing_uses_identical_frozen_payload():
    episode = Episode("e", {"question": "q", "items": [1]})
    a = Adapter([{"prob_yes": .4, "action": "NO"}])
    b = Adapter([{"prob_yes": .6, "action": "YES"}])
    results = await IntelligenceEvalRunner().run(episode, [spec(a, name="a"), spec(b, name="b")])
    assert len(results) == 2
    assert a.requests[0].episode_payload is episode.payload is b.requests[0].episode_payload
    assert a.requests[0].episode_hash == b.requests[0].episode_hash
    with pytest.raises(TypeError):
        episode.payload["question"] = "changed"


@pytest.mark.asyncio
async def test_stable_seeds_aggregation_and_concurrency():
    outputs = [
        {"prob_yes": .2, "action": "NO"},
        {"prob_yes": .5, "action": "ABSTAIN"},
        {"prob_yes": .8, "action": "YES"},
    ]
    runner = IntelligenceEvalRunner(max_concurrency=2)
    a = Adapter(outputs)
    first = (await runner.run(Episode("e", {"x": 1}), [spec(a, ExplorationPolicy.SAMPLES, 3)]))[0]
    b = Adapter(outputs)
    replay = (await runner.run(Episode("e", {"x": 1}), [spec(b, ExplorationPolicy.SAMPLES, 3)]))[0]
    seeds = [x.seed for x in first.attempts]
    assert len(set(seeds)) == 3
    assert seeds == [x.seed for x in replay.attempts]
    assert first.final_forecast.prob_yes == pytest.approx(.5)
    assert first.final_forecast.action == "ABSTAIN"
    assert a.peak <= 2
    assert all(item.telemetry["queue_ms"] >= 0 for item in first.attempts)


@pytest.mark.asyncio
async def test_critic_receives_candidates_and_supplies_final():
    def output(request):
        if request.stage == "critic":
            assert [x.prob_yes for x in request.candidate_forecasts] == [.3, .7]
            return {"prob_yes": .62, "action": "YES"}
        return {"prob_yes": (.3, .7)[request.sample_index], "action": "NO"}

    result = (await IntelligenceEvalRunner().run(
        Episode("e", {}), [spec(Adapter(output), ExplorationPolicy.SAMPLES_CRITIC, 2)]
    ))[0]
    assert [x.stage for x in result.attempts] == ["sample", "sample", "critic"]
    assert result.aggregate_forecast.prob_yes == pytest.approx(.5)
    assert result.final_forecast.prob_yes == pytest.approx(.62)
    assert result.attempts[-1].telemetry["stage"] == "critic"


@pytest.mark.asyncio
async def test_partial_and_parse_failures_are_recorded():
    class Partial:
        async def generate(self, request):
            if request.sample_index == 0:
                return AdapterResponse("not json", {"tokens": 2})
            if request.sample_index == 1:
                raise RuntimeError("provider down")
            return AdapterResponse({"prob_yes": .75, "action": "YES"})

    result = (await IntelligenceEvalRunner().run(
        Episode("e", {}), [spec(Partial(), ExplorationPolicy.SAMPLES, 3)]
    ))[0]
    assert [x.succeeded for x in result.attempts] == [False, False, True]
    assert "invalid JSON" in result.attempts[0].error
    assert "provider down" in result.attempts[1].error
    assert result.attempts[1].telemetry["queue_ms"] >= 0
    assert result.final_forecast.prob_yes == .75


@pytest.mark.asyncio
async def test_strict_schema_all_failures_yield_no_forecast():
    adapter = Adapter([{"prob_yes": 1.2, "action": "YES", "extra": True}])
    result = (await IntelligenceEvalRunner().run(Episode("e", {}), [spec(adapter)]))[0]
    assert not result.succeeded
    assert result.final_forecast is None
    assert "unknown fields" in result.attempts[0].error


@pytest.mark.asyncio
async def test_extra_payload_enriches_requests_without_breaking_pairing():
    """A claims_evidence arm sees episode payload + evidence in its requests,
    while episode identity (hash) stays shared with the bare arm."""
    episode = Episode("e", {"question": "q"})
    bare = Adapter([{"prob_yes": .4, "action": "NO"}])
    rich = Adapter([{"prob_yes": .6, "action": "YES"}])
    claims = {"distilled_evidence": [{"claim": "c1", "source": "s"}]}
    enriched = TreatmentSpec(
        "claims", ArmSpec("claims-arm", "fake", rich),
        ExplorationPolicy.SINGLE, 1, 42, extra_payload=claims)
    results = await IntelligenceEvalRunner().run(
        episode, [spec(bare, name="bare"), enriched])
    assert len(results) == 2
    assert "distilled_evidence" not in bare.requests[0].episode_payload
    assert rich.requests[0].episode_payload["distilled_evidence"] == claims["distilled_evidence"]
    assert rich.requests[0].episode_payload["question"] == "q"
    # Pairing: both arms reference the same episode hash.
    assert bare.requests[0].episode_hash == rich.requests[0].episode_hash


@pytest.mark.asyncio
async def test_extra_payload_reaches_critic_stage():
    outputs = lambda request: {"prob_yes": .7, "action": "YES"}
    rich = Adapter(outputs)
    enriched = TreatmentSpec(
        "claims_critic", ArmSpec("arm", "fake", rich),
        ExplorationPolicy.SAMPLES_CRITIC, 2, 7,
        extra_payload={"distilled_evidence": [{"claim": "c"}]})
    await IntelligenceEvalRunner().run(Episode("e", {"q": 1}), [enriched])
    stages = {r.stage for r in rich.requests}
    assert stages == {"sample", "critic"}
    assert all("distilled_evidence" in r.episode_payload for r in rich.requests)
