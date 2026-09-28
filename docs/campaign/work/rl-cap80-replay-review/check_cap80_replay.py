"""CPU-only independent comparison of cap75 d and cap80 e replay prefixes."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

D = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-d")
E = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-e")
E105 = E / "archive/full/checkpoint-105"
SCHEDULE = "2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132"
IDENTITY = "ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818"

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]

def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def event_map(run: Path) -> dict[int, dict]:
    result = {}
    for item in read_jsonl(run / "telemetry.jsonl"):
        if item.get("event") == "trainer_metrics":
            result[int(item["step"])] = item["metrics"]
    return result

def main() -> None:
    dg = {run: read_jsonl(run / "output/generation-records.jsonl") for run in (D, E)}
    dr = {run: read_jsonl(run / "output/reward-records.jsonl") for run in (D, E)}
    gd = {run: read_jsonl(run / "output/gradient-records.jsonl") for run in (D, E)}
    comparisons = {}
    for update in (100, 101, 102, 103):
        dgen = [x for x in dg[D] if x["global_step_before_update"] == update]
        egen = [x for x in dg[E] if x["global_step_before_update"] == update]
        drow = dr[D][(update - 100) * 32:(update - 99) * 32]
        erow = dr[E][(update - 100) * 32:(update - 99) * 32]
        assert len(dgen) == len(egen) == len(drow) == len(erow) == 32
        dnorm = [{k: v for k, v in x.items() if k != "elapsed_sec"} for x in dgen]
        enorm = [{k: v for k, v in x.items() if k != "elapsed_sec"} for x in egen]
        dgrad = [x for x in gd[D] if x.get("global_step") == update]
        egrad = [x for x in gd[E] if x.get("global_step") == update]
        comparisons[str(update)] = {
            "rows": 32,
            "generation_without_timing_equal": dnorm == enorm,
            "source_schedule_equal": all(x["source_schedule_sha256"] == SCHEDULE for x in dgen + egen),
            "prompt_hash_sequence_sha256": digest([x["prompt_ids_sha256"] for x in dgen]),
            "generated_id_hash_sequence_sha256": digest([x["generated_ids_sha256"] for x in dgen]),
            "row_id_sequence_sha256": digest([x["id"] for x in drow]),
            "reward_vector_sha256": digest([x["reward"] for x in drow]),
            "reward_records_equal": drow == erow,
            "gradient_records_d_and_e": {
                "d_count": len(dgrad), "e_count": len(egrad), "equal": dgrad == egrad,
            },
        }
        assert comparisons[str(update)]["generation_without_timing_equal"]
        assert comparisons[str(update)]["source_schedule_equal"]
        assert comparisons[str(update)]["reward_records_equal"]
        if update <= 102:
            assert dgrad == egrad and len(dgrad) == 1
        else:
            assert len(dgrad) == 0 and len(egrad) == 1

    dm, em = event_map(D), event_map(E)
    metric_keys = ("grad_norm", "loss", "reward", "reward_std", "completion_length", "num_tokens")
    for step in (101, 102, 103):
        assert all(dm[step].get(key) == em[step].get(key) for key in metric_keys)
    assert 104 in em and 105 in em
    assert not (E / "output/failure.json").exists()

    load_audit = json.loads((E / "output/load-audit.json").read_text(encoding="utf-8"))
    allocator = load_audit["cuda_allocator"]
    assert allocator["cuda_allocator_fraction"] == 0.8
    assert allocator["cuda_allocator_total_bytes"] == 34190458880
    assert allocator["cuda_allocator_cap_bytes"] == 27352367104

    with (E105 / "campaign-manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["full"] is True and manifest["step"] == 105
    assert digest(manifest["identity"]) == IDENTITY
    sys.path.insert(0, "/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source/experiments/training")
    from campaign_checkpoint import verify_checkpoint
    assert verify_checkpoint(E105, manifest["identity"], require_full=True)["step"] == 105

    for update, result in comparisons.items():
        print(update, json.dumps(result, sort_keys=True))
    print("e_step104", json.dumps({key: em[104].get(key) for key in metric_keys}, sort_keys=True))
    print("e_step105", json.dumps({key: em[105].get(key) for key in metric_keys}, sort_keys=True))
    print("cap80_allocator", json.dumps(allocator, sort_keys=True))
    print("full105_strict_verify=PASS")
    print("INDEPENDENT_CAP80_REPLAY_PASS")

if __name__ == "__main__":
    main()
