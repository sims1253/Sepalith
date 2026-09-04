#!/usr/bin/env python3
"""Unit tests for the E1 EL-scheduler retrofit on rl_smoke.py (CPU-only,
no torch/trl/unsloth imports, no GPU, no CUDA context).

Covers: tier construction + ordering, admission gating (fire / hold /
defer / window reset), quota interplay (renormalized proportions over the
active set, background families always drawable, pool exhaustion
reshuffle), the sampler-stream contract (geometry identical to trl's
RepeatSampler), the group readout math (psg/full rates, legacy stash
tuples), the first-50 readout accumulator, the reward path regression
(rewards byte-identical; stash carries the prompt), and quota-resolution
parity with the pre-E1 inline logic (default-path byte-compat guard).

Run:  uv run --with pytest python -m pytest test_rl_smoke_el.py -q
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rl_smoke  # noqa: E402  (module import is CPU-light by design)


# ---------------------------------------------------------------------------
# helpers: pools + a mocked reward stream driving observe()/maybe_admit()
# exactly as the MetricsCb flush does in rl_smoke.main()
# ---------------------------------------------------------------------------

def make_pools(sizes=None):
    """family -> [dataset_idx, ...] pools with the default-quota shape."""
    sizes = sizes or {"pipe_rewrite": 150, "rename_propagation": 1400,
                      "format_propagation": 1400, "no_op": 350}
    pools, i = {}, 0
    for fam, n in sizes.items():
        pools[fam] = list(range(i, i + n))
        i += n
    return pools


def fam_of_idx(sched, idx):
    for fam, idxs in sched.pools.items():
        if idx in idxs:
            return fam
    raise KeyError(idx)


def fams_of_batch(sched, batch):
    return Counter(fam_of_idx(sched, i) for i in batch)


def mock_run_steps(sched, pass_rate, steps, groups_per_step=8, seed=11):
    """One optimizer step == one next_batch + one observe/maybe_admit flush
    (the MetricsCb loop). pass_rate: {family: probability a group of that
    family is FULLY solved}. Returns (draws, admissions)."""
    import random as _r
    rng = _r.Random(seed)
    tier_fams = {f for t in sched.tiers for f in t}
    draws, admissions = [], []
    for _ in range(steps):
        batch = sched.next_batch(groups_per_step)
        draws.append(batch)
        per_family = {}
        for idx in batch:
            fam = fam_of_idx(sched, idx)
            if fam not in tier_fams:
                continue
            full = rng.random() < pass_rate.get(fam, 0.0)
            g = per_family.setdefault(fam, [0, 0])
            g[0] += 1
            g[1] += int(full)
        sched.observe(per_family)
        if sched.maybe_admit():
            admissions.append(list(sched.tiers[sched.admitted - 1]))
    return draws, admissions


# ---------------------------------------------------------------------------
# tier construction + ordering
# ---------------------------------------------------------------------------

def test_tier_order_and_background():
    sched = rl_smoke.OrderedScheduler(make_pools())
    assert sched.tiers == [["pipe_rewrite"], ["rename_propagation"],
                           ["format_propagation"]]
    assert sched.background == ["no_op"]
    assert sched.admitted == 1


def test_compound_appended_as_final_tier_when_present():
    pools = make_pools({"pipe_rewrite": 10, "rename_propagation": 10,
                        "format_propagation": 10, "compound_edit": 10,
                        "compound_chain": 10, "no_op": 10})
    sched = rl_smoke.OrderedScheduler(pools)
    assert sched.tiers == [["pipe_rewrite"], ["rename_propagation"],
                           ["format_propagation"],
                           ["compound_chain", "compound_edit"]]
    assert sched.background == ["no_op"]


def test_absent_tier_families_skipped():
    sched = rl_smoke.OrderedScheduler(
        make_pools({"rename_propagation": 10, "format_propagation": 10}))
    assert sched.tiers == [["rename_propagation"], ["format_propagation"]]


# ---------------------------------------------------------------------------
# ordering: gating of draws
# ---------------------------------------------------------------------------

def test_initial_draws_gated_to_tier1_plus_background():
    sched = rl_smoke.OrderedScheduler(make_pools())
    draws, admissions = mock_run_steps(sched, pass_rate={}, steps=20)
    assert admissions == []
    for batch in draws:
        fams = fams_of_batch(sched, batch)
        assert set(fams) <= {"pipe_rewrite", "no_op"}, \
            f"pre-admission draw leaked a gated tier: {dict(fams)}"


def test_admitted_tier_appears_in_draws_after_gate():
    sched = rl_smoke.OrderedScheduler(make_pools())
    # pipe fully solved 100% of groups -> gate clears within K steps
    draws, admissions = mock_run_steps(
        sched, pass_rate={"pipe_rewrite": 1.0}, steps=10)
    assert admissions and admissions[0] == ["rename_propagation"]
    seen = Counter()
    for batch in draws[len(draws) // 2:]:
        seen.update(fams_of_batch(sched, batch))
    assert seen["rename_propagation"] > 0
    assert "format_propagation" not in seen   # still gated


def test_format_never_admitted_while_rename_below_threshold():
    sched = rl_smoke.OrderedScheduler(make_pools())
    draws, admissions = mock_run_steps(
        sched, pass_rate={"pipe_rewrite": 1.0, "rename_propagation": 0.3},
        steps=60)
    assert admissions == [["rename_propagation"]]
    for batch in draws:
        assert "format_propagation" not in fams_of_batch(sched, batch)


# ---------------------------------------------------------------------------
# admission rule: threshold, min-count guard, window reset
# ---------------------------------------------------------------------------

def test_no_admission_well_below_threshold():
    sched = rl_smoke.OrderedScheduler(make_pools())
    _, admissions = mock_run_steps(
        sched, pass_rate={"pipe_rewrite": 0.25}, steps=40)
    assert admissions == []


def test_admission_requires_min_groups_in_window():
    # frontier share so small that K=5 steps pool < 8 groups -> defer,
    # never admit, even at a 1.0 full-solve rate
    pools = make_pools({"pipe_rewrite": 10, "rename_propagation": 990,
                        "format_propagation": 990, "no_op": 5000})
    sched = rl_smoke.OrderedScheduler(pools)
    _, admissions = mock_run_steps(
        sched, pass_rate={"pipe_rewrite": 1.0}, steps=60)
    assert admissions == []
    assert sched.state()["el_window_groups"] < sched.min_groups


def test_window_resets_after_admission():
    sched = rl_smoke.OrderedScheduler(make_pools())
    admissions = []
    for step in range(30):
        sched.next_batch(8)
        # pipe era at 100% full for the first steps, then collapse; rename
        # (the new frontier) is never fed -> exactly one admission
        sched.observe({"pipe_rewrite": [8, 8 if step < 6 else 0]})
        if sched.maybe_admit():
            admissions.append(step)
    assert len(admissions) == 1
    assert sched.admitted == 2
    # frontier is now rename; its window is empty (fresh evaluation)
    assert sched.state()["el_window_groups"] == 0
    assert sched.state()["el_frontier"] == "rename_propagation"


def test_admission_boundary_exactly_at_threshold():
    # 0.75 exactly clears the >= gate: 8 pooled groups, 6 full
    sched = rl_smoke.OrderedScheduler(make_pools())
    sched.observe({"pipe_rewrite": [4, 3]})
    sched.observe({"pipe_rewrite": [4, 3]})
    assert sched.maybe_admit() == ["rename_propagation"]


def test_admission_holds_at_one_group_below_threshold():
    sched = rl_smoke.OrderedScheduler(make_pools())
    sched.observe({"pipe_rewrite": [4, 3]})
    sched.observe({"pipe_rewrite": [4, 2]})    # 5/8 = 0.625
    assert sched.maybe_admit() is None


def test_state_telemetry_shape():
    sched = rl_smoke.OrderedScheduler(make_pools())
    sched.observe({"pipe_rewrite": [4, 3], "no_op": [4, 4]})
    st = sched.state()
    assert st["el_tiers_admitted"] == 1
    assert st["el_frontier"] == "pipe_rewrite"
    assert st["el_window_groups"] == 4          # background excluded
    assert st["el_window_full_rate"] == 0.75


def test_all_tiers_admitted_reports_done():
    sched = rl_smoke.OrderedScheduler(make_pools())
    sched.admitted = 3
    assert sched.maybe_admit() is None
    assert sched.state()["el_frontier"] == "done"


# ---------------------------------------------------------------------------
# quota interplay
# ---------------------------------------------------------------------------

def test_quota_proportions_over_active_set():
    # all tiers admitted -> family shares match renormalized pool quotas
    pools = make_pools()
    sched = rl_smoke.OrderedScheduler(pools)
    sched.admitted = len(sched.tiers)
    counts = Counter()
    for _ in range(1500):
        counts.update(fams_of_batch(sched, sched.next_batch(8)))
    total = sum(counts.values())
    active = dict(pipe_rewrite=150, rename_propagation=1400,
                  format_propagation=1400, no_op=350)
    denom = sum(active.values())
    for fam, quota in active.items():
        expected = quota / denom
        got = counts[fam] / total
        assert abs(got - expected) < 0.03, (fam, got, expected)


def test_background_always_drawable_from_step_one():
    sched = rl_smoke.OrderedScheduler(make_pools())
    seen_no_op = 0
    for _ in range(200):
        seen_no_op += fams_of_batch(sched, sched.next_batch(8))["no_op"]
    assert seen_no_op > 0


def test_pool_exhaustion_reshuffles_without_dupes_in_batch():
    pools = make_pools({"pipe_rewrite": 3, "no_op": 3})
    sched = rl_smoke.OrderedScheduler(pools)
    for _ in range(50):
        batch = sched.next_batch(4)
        assert len(batch) == len(set(batch)), "duplicate prompt in one batch"


def test_small_pool_returns_short_batch():
    pools = make_pools({"pipe_rewrite": 2, "no_op": 2})
    sched = rl_smoke.OrderedScheduler(pools)
    assert len(sched.next_batch(8)) == 4        # all 4 unique prompts


# ---------------------------------------------------------------------------
# sampler-stream contract (geometry identical to trl RepeatSampler)
# ---------------------------------------------------------------------------

def test_sampler_stream_geometry_matches_repeat_sampler():
    N, B, G, R = 100, 8, 4, 4
    pools = {"pipe_rewrite": list(range(40)), "no_op": list(range(40, 100))}
    sched = rl_smoke.OrderedScheduler(pools)
    stream = rl_smoke.ELSamplerStream(sched, num_samples=N,
                                      num_generations=G,
                                      unique_per_batch=B, repeat_count=R)
    idxs = list(stream)
    # stock RepeatSampler length formula: (N//B)*B*G*R
    assert len(stream) == (N // B) * B * G * R
    assert len(idxs) == len(stream)
    # chunk structure: B unique prompts, each xG consecutive, block xR
    for c in range(0, len(idxs), B * G * R):
        chunk = idxs[c:c + B * G * R]
        base = chunk[: B * G]
        uniq = base[::G]
        assert len(uniq) == B and len(set(uniq)) == B
        for r in range(R):
            seg = chunk[r * B * G:(r + 1) * B * G]
            assert seg == base                  # R consecutive repeats
        for i in range(B):
            assert base[i * G:(i + 1) * G] == [uniq[i]] * G


def test_sampler_stream_deterministic_given_seed():
    def run():
        pools = make_pools()
        s = rl_smoke.OrderedScheduler(pools, seed=3407)
        st = rl_smoke.ELSamplerStream(s, num_samples=400, num_generations=4,
                                      unique_per_batch=8, repeat_count=4)
        return [x for i, x in enumerate(st) if i % 32 == 0]
    assert run() == run()


# ---------------------------------------------------------------------------
# group readout math + first-50 accumulator
# ---------------------------------------------------------------------------

def grp(prompt, fam, solved, size=4):
    """STASH records for one GRPO group with `solved` exact completions."""
    return [(fam, int(k < solved), 1.2 if k < solved else 0.0, prompt)
            for k in range(size)]


def test_group_stats_rates():
    recs = (grp("p0", "pipe_rewrite", 4) + grp("p1", "pipe_rewrite", 2)
            + grp("p2", "format_propagation", 0)
            + grp("p3", "format_propagation", 3))
    g = rl_smoke.group_stats(recs)
    assert g["n_groups"] == 4
    assert g["full_group_rate"] == 0.25          # p0
    assert g["psg_rate"] == 0.5                  # p1, p3
    assert g["per_family"]["pipe_rewrite"] == [2, 1, 1]
    assert g["per_family"]["format_propagation"] == [2, 0, 1]


def test_group_stats_legacy_3tuples_tolerated():
    # pre-E1 stash shape: no prompt -> one group per completion
    g = rl_smoke.group_stats([("no_op", 1, 1.2), ("no_op", 0, 0.0)])
    assert g["n_groups"] == 2
    assert g["full_group_rate"] == 0.5
    assert g["psg_rate"] == 0.0


def test_group_stats_empty():
    assert rl_smoke.group_stats([]) is None


def test_first50_readout_pools_steps_and_families():
    acc = rl_smoke.First50Readout(max_step=50)
    for step in range(1, 51):
        recs = (grp(f"a{step}", "pipe_rewrite", 4)
                + grp(f"b{step}", "format_propagation", 2))
        acc.add(step, recs, rl_smoke.group_stats(recs))
    recs = grp("z", "format_propagation", 0)     # beyond the window: ignored
    acc.add(51, recs, rl_smoke.group_stats(recs))
    out = acc.emit()
    assert out["first50_n_groups"] == 100
    assert out["first50_psg_rate"] == 0.5
    assert out["first50_psg_pipe_rewrite"] == 0.0
    assert out["first50_psg_format_propagation"] == 1.0
    # mean reward: 6 of 8 completions per step at 1.2, rest 0.0 -> 0.9
    assert out["first50_reward"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# reward path regression (byte-compat of the reward values)
# ---------------------------------------------------------------------------

def test_scenario_reward_values_unchanged_and_stash_carries_prompt():
    rl_smoke.STASH.clear()
    prompts = ["<bos>P1", "<bos>P2"]
    completions = ["alpha <- 1\n>>>>>>> UPDATED",
                   "beta <- 99\ngamma <- 3\n>>>>>>> UPDATED"]
    targets = ["alpha <- 1\n>>>>>>> UPDATED",
               "beta <- 99\n>>>>>>> UPDATED"]
    fams = ["pipe_rewrite", "rename_propagation"]
    out = rl_smoke.scenario_reward(
        prompts=prompts, completions=completions, target=targets,
        family=fams)
    assert out[0] == pytest.approx(1.2)          # exact + 0.2 * 1.0
    # second: no exact match, 1 of 2 lines match -> 0.2 * line_f1(0.5, 1.0)
    assert out[1] == pytest.approx(0.2 * (2 * 0.5 * 1.0 / 1.5))
    recs = list(rl_smoke.STASH)
    rl_smoke.STASH.clear()
    assert len(recs) == 2
    assert recs[0][0] == "pipe_rewrite" and recs[0][1] == 1
    assert recs[0][2] == pytest.approx(1.2)
    assert recs[0][3] == "<bos>P1"               # group key stashed (E1)
    assert recs[1][3] == "<bos>P2"


# ---------------------------------------------------------------------------
# quota resolution parity with the pre-E1 inline logic (default-path guard)
# ---------------------------------------------------------------------------

def test_pick_quotas_matches_pre_e1_inline_logic():
    for run2 in (False, True):
        for no_op_n in (None, 800):
            for smoke in (False, True):
                base = (rl_smoke.FAMILY_QUOTA_RUN2 if run2
                        else rl_smoke.FAMILY_QUOTA)
                expect = dict(base)              # all families selected
                if no_op_n is not None and "no_op" in expect:
                    expect["no_op"] = no_op_n
                if smoke:
                    expect = {f: min(n, 8) for f, n in expect.items()}
                got = rl_smoke.pick_quotas(
                    ",".join(base), run2=run2, no_op_n=no_op_n, smoke=smoke)
                assert got == expect


def test_defaults_unchanged():
    # the frozen constants the header pre-registers
    assert rl_smoke.EL_THRESHOLD == 0.75
    assert rl_smoke.EL_WINDOW_K == 5
    assert rl_smoke.EL_MIN_GROUPS == 8
    assert rl_smoke.EL_TIER_ORDER == (
        "pipe_rewrite", "rename_propagation", "format_propagation")
    assert rl_smoke.SHAPING == 0.2
