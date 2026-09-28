#!/usr/bin/env python3
"""CPU-only source gate for the isolated RUN-06 DSpark candidate.

This checks identity, patch shape, and source contracts.  It deliberately does
not instantiate a llama model or claim numerical/model equivalence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
TARGET = PLAN / "docs/campaign/work/dspark-opus12-validation"
SRC = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453")
PIN = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, cond: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(cond), "detail": detail})
    if not cond:
        raise AssertionError(f"{name}: {detail}")


checks: list[dict[str, object]] = []

commit = subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "HEAD"], text=True).strip()
check("pinned_commit", commit == PIN, f"got {commit}")

key_source_hashes = {
    "common/speculative.cpp": "81248dbf9b755f02f200a92ee613b0c32f999c27bd192b5d3bd9b997030818d3",
    "common/speculative.h": "17df3de19b6e706e4b5705be38677ff6d78883c5188d43f80b915be0ec1cd95c",
    "src/llama-context.cpp": "a061f57bc0aaf55697e41952569835a5d4e65cc8d7a97c25410d940b47ee3044",
    "src/llama-context.h": "7cd1b39819442f1f3c556bec487fdd8d1501967cd1fbe33feea59d5bbc556fc8",
    "src/llama-model.cpp": "518506c4aaf12a8cbb44f9f12ce3d75e876753f8131d94d6507515cd3a57ff5a",
    "src/llama-kv-cache-iswa.cpp": "567fe836543e1dd9cdd5b7cb3a91afd049c17c7010f0bddf9fe149b02b5243cb",
    "src/llama-kv-cache-dsv4.cpp": "eaff4b9a1ca7b9e5c311012492cdc8f58777a157fa4f0700ddb81207fc84d9e9",
    "src/models/dflash.cpp": "58d5feeb6f5a5c459e438b0bc531e8d53da33d13962f4bf589071eb649a82251",
    "tools/server/server-context.cpp": "26f130b76c27be72e4674943754575cf5efa14b6a6325591be07df57f651e681",
    "tools/server/server-task.cpp": "20b37328cacecbd0e8a91b5e5f34e70a191a2537ff998dd6594e44a96299614c",
    "tools/server/server-common.cpp": "067f3f5db9a0bb72c5c834938eb9469d3d81085d9358b85e3be914f39d42f584",
    "include/llama.h": "1fbcba4003cfc089fa9681973acb3d9e24e465f32c3506ceb0ec7fa29952bdae",
}
for rel, expected in key_source_hashes.items():
    got = sha256(SRC / rel)
    check(f"source_hash:{rel}", got == expected, f"got {got}")

raw = (TARGET / "raw-candidate.patch").read_text()
normalized = (TARGET / "candidate.patch").read_text()
check("raw_patch_preserved", "@@ (top of file, next to the existing includes) @@" in raw)
check("normalized_patch_has_real_headers", "@@ (top of file" not in normalized and normalized.startswith("--- a/common/speculative.cpp\n+++ b/common/speculative.cpp\n"))
check("normalized_patch_complete_hunks", normalized.count("@@ ") == 10, f"hunks={normalized.count('@@ ')}")
check("candidate_matches_normalized_apply", (TARGET / "audit/normalized-apply-diff.txt").read_text() == "")

header = (TARGET / "new-header/speculative-dspark.h").read_text()
candidate = (TARGET / "candidate-source/common/speculative.cpp").read_text()
baseline = (TARGET / "baseline-source/common/speculative.cpp").read_text()
check("header_plan_full_block", "return { block_size, n_max };" in header)
check("header_plan_default_dspark", "const int32_t n_block = n_max + (is_dspark ? 0 : 1);" in header)
check("header_exact_env", "v[0] == '1' && v[1] == '\\0'" in header)
check("candidate_include", '#include "speculative-dspark.h"' in candidate)
check("candidate_mode_guard", "n_seq != 1" in candidate and "block_size > (int32_t) llama_n_ubatch(ctx_dft)" in candidate)
check("candidate_only_leading_outputs", "i < plan.n_outputs" in candidate)
check("candidate_failure_cleanup", "exp_drop_blocks(pos0);" in candidate)
check("candidate_success_cleanup", candidate.count("exp_drop_blocks(pos0);") >= 2)
check("candidate_default_formula_present", "dspark_plan_block(is_dspark, exp_full_block, params.n_max, block_size)" in candidate)
check("baseline_has_original_all_output", "common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);" in baseline)
check("baseline_has_original_plan", "const int32_t n_block_tokens = n_draft + (is_dspark ? 0 : 1);" in baseline)
check("candidate_has_rm_return_guard", "if (!llama_memory_seq_rm(mem, seq_id, p0, -1))" in candidate)

# These are source contracts used by the review, rather than candidate claims.
ctx = (SRC / "src/llama-context.cpp").read_text()
api = (SRC / "include/llama.h").read_text()
model = (SRC / "src/llama-model.cpp").read_text()
iswa = (SRC / "src/llama-kv-cache-iswa.cpp").read_text()
dsv4 = (SRC / "src/llama-kv-cache-dsv4.cpp").read_text()
spec = (SRC / "common/speculative.cpp").read_text()
server = (SRC / "tools/server/server-context.cpp").read_text()
dflash = (SRC / "src/models/dflash.cpp").read_text()

check("api_seq_rm_is_bool", "LLAMA_API bool llama_memory_seq_rm" in api)
check("api_pos_max_dense_contract", "all positions in the range [pos_min, pos_max] are guaranteed" in api)
check("api_decode_abort_contract", "processed ubatches will remain in the context's memory" in api)
check("context_seq_rm_forwards_memory", "return mem->seq_rm(seq_id, p0, p1);" in ctx)
check("context_output_counts_flags", "if (!output_all && batch_inp.logits[i] == 0)" in ctx)
check("context_masked_nextn_uses_outputs", "const int64_t n_rows = masked ? n_outputs" in ctx)
check("model_dflash_selects_iswa", "case LLM_ARCH_DFLASH:" in model and "res = new llama_kv_cache_iswa" in model)
check("iswa_removes_base_and_swa", "kv_base->seq_rm(seq_id, p0, p1)" in iswa and "kv_swa ->seq_rm(seq_id, p0, p1)" in iswa)
check("dsv4_rejects_some_partial_removals", "if (n_rs_seq == 0)" in dsv4 and "return false;" in dsv4)
check("reservation_remains_nmax_plus_one", "const int32_t per_seq = std::max(1, params_spec.n_max + 1);" in spec)
check("server_output_limit_uses_nmax", "common_speculative_get_output_limits" in server)
check("server_calls_draft", "common_speculative_draft(spec.get());" in server)
check("server_cleans_draft_context", "llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)" in server)
check("server_cancel_releases_slot", "case SERVER_TASK_TYPE_CANCEL:" in server and "slot.release();" in server)
check("dflash_noncausal_graph", "graph_dsv4" in dflash and "DSpark" in dflash)

result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(SRC),
    "pinned_commit": commit,
    "checks": checks,
    "artifacts": {
        "raw_patch_sha256": sha256(TARGET / "raw-candidate.patch"),
        "normalized_patch_sha256": sha256(TARGET / "candidate.patch"),
        "header_sha256": sha256(TARGET / "new-header/speculative-dspark.h"),
        "test_sha256": sha256(TARGET / "tests/test-dspark-prefix.cpp"),
        "test_binary_sha256": sha256(TARGET / "audit/bin/test-dspark-prefix"),
        "candidate_source_sha256": sha256(TARGET / "candidate-source/common/speculative.cpp"),
    },
    "model_scope": "source gate only; KV/test state machine remains a model and no model/server was run",
}
(TARGET / "audit/source-gate-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": len(checks), "timestamp_utc": result["timestamp_utc"], "artifacts": result["artifacts"]}, indent=2))
