"""Fail-closed metadata binding for a local-to-DDP8 execution transition."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

def require(v, m):
    if not v: raise ValueError(m)
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(recipe, *, verify_large=False):
    require(recipe.get("schema") == "sepalith.sft11.a100-fullweight-continuation-template.v1", "schema differs")
    require(recipe.get("launch_authorized") is False, "preparation must not authorize launch")
    d=recipe["distributed"]
    require(d == {"backend":"nccl","world_size":8,"nodes":1,"gpus_per_node":8,
                  "rows_per_rank":2,"gradient_accumulation":1,"effective_batch":16,
                  "average_tokens_across_devices":True,"split_batches":False,
                  "dispatch_batches":False}, "DDP geometry differs")
    source=recipe["source_checkpoint"]; cp=Path(source["path"])
    mp=cp/"campaign-manifest.json"; sp=cp/"campaign-state.json"
    require(mp.is_file() and sha256(mp)==source["manifest_sha256"], "source manifest differs")
    manifest=json.loads(mp.read_text()); state=json.loads(sp.read_text())
    for name in ("campaign-state.json", "trainer_state.json", "scheduler.pt", "rng_state.pth"):
        record=manifest.get("files",{}).get(name,{}); target=cp/name
        require(target.is_file() and target.stat().st_size==record.get("bytes") and sha256(target)==record.get("sha256"), f"source metadata differs:{name}")
    source_recipe=source.get("source_recipe",{}); source_recipe_path=Path(source_recipe.get("path",""))
    require(source_recipe_path.is_file() and sha256(source_recipe_path)==source_recipe.get("sha256"), "source recipe differs")
    review=source.get("root_review",{}); review_path=Path(review.get("path",""))
    require(review_path.is_file() and sha256(review_path)==review.get("sha256"), "source root review differs")
    require(manifest.get("full") is True and manifest.get("checkpoint_kind")=="full_weights", "source is not full state")
    require(manifest.get("step")==state.get("step")==source["step"]==90, "source step differs")
    sampler=state.get("sampler",{})
    require(sampler.get("cursor")==source["cursor"]==384 and sampler.get("global_optimizer_step_offset")==source["offset"]==66, "source cursor/offset differs")
    require(sampler.get("draw_schedule_sha256")==recipe["data"]["draw_schedule_sha256"], "schedule differs")
    require(recipe["runtime"]["optimizer"]==manifest["identity"]["policy"]["optimizer"], "optimizer differs")
    require(recipe["runtime"]["learning_rate"]==manifest["identity"]["schedule"]["learning_rate"]==3e-6, "learning rate differs")
    require(recipe["runtime"]["warmup_steps"]==manifest["identity"]["schedule"]["warmup_steps"]==2, "warmup differs")
    require(recipe["runtime"]["scheduler"]==manifest["identity"]["schedule"]["scheduler"]=="constant_with_warmup", "scheduler differs")
    require(recipe["data"]["rows_sha256"]==manifest["identity"]["data"]["rows_sha256"], "row identity differs")
    require(recipe["data"]["cache_manifest_sha256"]==manifest["identity"]["data"]["streaming_cache_manifest_sha256"], "cache identity differs")
    require(recipe["runtime"]["initial_cursor"]==384 and recipe["runtime"]["global_step"]==90, "continuation origin differs")
    require(recipe.get("root_admission") is None, "root admission placeholder differs")
    if verify_large:
        for name, record in manifest["files"].items():
            p=cp/name
            require(p.is_file() and p.stat().st_size==record["bytes"] and sha256(p)==record["sha256"], f"source payload differs:{name}")
    return {"status":"prepared_not_admitted", "source_step":90,"cursor":384,"world_size":8,
            "large_payload_verified":verify_large}
