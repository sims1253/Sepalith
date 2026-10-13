#!/usr/bin/env python3
"""Make a redacted scalar aggregate from one native CUDA screen.

The input probe contains stored prompt/token identities and returned text.  This
report intentionally reads only the scalar timing/count fields and never copies
those fields into the output.  It is suitable for a review packet; it is not a
replacement for the full root-owned run record.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }


def _scalar_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    return _stats(row[key] for row in rows if _finite(row.get(key)))


def _timing_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    return _stats(
        row["timings"][key]
        for row in rows
        if isinstance(row.get("timings"), dict) and _finite(row["timings"].get(key))
    )


def _row_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    """Aggregate a scalar whose name is kept outside the nested timings map."""
    return _stats(row[key] for row in rows if _finite(row.get(key)))


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, (str, int, float, bool)):
            label = str(value).lower() if isinstance(value, bool) else str(value)
            out[label] = out.get(label, 0) + 1
    return dict(sorted(out.items()))


def _parse_server_log(path: Path) -> dict[str, Any]:
    """Extract only fixed scalar observations from a server log."""
    text = path.read_text(errors="replace")
    out: dict[str, Any] = {}

    m = re.search(r"CUDA0\s+:\s+([^(:]+)\s+\((\d+) MiB,\s+(\d+) MiB free\)", text)
    if m:
        out["device_class"] = m.group(1).strip()
        out["device_memory_mib"] = int(m.group(2))
        out["device_free_at_start_mib"] = int(m.group(3))

    m = re.search(r"system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\).*?USE_GRAPHS = (\d+)", text)
    if m:
        out["threads"] = int(m.group(1))
        out["batch_threads"] = int(m.group(2))
        out["use_graphs"] = bool(int(m.group(3)))

    m = re.search(r"loaded meta data with (\d+) key-value pairs and (\d+) tensors", text)
    if m:
        out["model_metadata_kv_pairs"] = int(m.group(1))
        out["model_tensor_count"] = int(m.group(2))
    tensor_types = {}
    for kind, count in re.findall(r"- type\s+([a-z0-9_]+):\s+(\d+) tensors", text):
        tensor_types[kind] = int(count)
    if tensor_types:
        out["model_tensor_types"] = dict(sorted(tensor_types.items()))
    out["flash_attention_enabled"] = "Flash Attention enabled" in text
    out["fused_paths_reported"] = sorted(set(re.findall(r"fused ([A-Za-z0-9 ]+) enabled", text)))

    patterns: list[tuple[str, str, type]] = [
        ("offloaded_layers", r"offloaded (\d+)/(\d+) layers", tuple),
        ("model_buffer_mib", r"CUDA0 model buffer size =\s+([0-9.]+) MiB", float),
        ("kv_buffer_mib", r"CUDA0 KV buffer size =\s+([0-9.]+) MiB", float),
        ("compute_buffer_mib", r"CUDA0 compute buffer size =\s+([0-9.]+) MiB", float),
        ("graph_nodes", r"graph nodes\s+=\s+(\d+)", int),
        ("graph_splits", r"graph splits\s+=\s+(\d+)", int),
        ("prompt_cache_limit_mib", r"prompt cache is enabled, size limit:\s+(\d+) MiB", int),
    ]
    for key, pattern, converter in patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        if converter is tuple:
            out[key] = {"used": int(m.group(1)), "total": int(m.group(2))}
        else:
            out[key] = converter(m.group(1))

    memory_matches = list(re.finditer(
        r"CUDA0 \([^)]*\)\s+\|\s+(\d+) = (\d+) \+ \((\d+) =\s+(\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)\) \+\s+(-?\d+)",
        text,
    ))
    if memory_matches:
        m = memory_matches[-1]
        out["final_cuda_memory_breakdown_mib"] = {
            "total": int(m.group(1)),
            "free": int(m.group(2)),
            "reserved": int(m.group(3)),
            "model": int(m.group(4)),
            "context": int(m.group(5)),
            "compute": int(m.group(6)),
            "unaccounted": int(m.group(7)),
        }

    reused = [int(v) for v in re.findall(r"graphs reused =\s+(\d+)", text)]
    if reused:
        out["graphs_reused_sequence_summary"] = _stats(reused)

    return out


def build_report(probe: Path, launch: Path, terminal: Path, server_log: Path) -> dict[str, Any]:
    probe_data = json.loads(probe.read_text())
    launch_data = json.loads(launch.read_text())
    terminal_data = json.loads(terminal.read_text())
    rows = probe_data.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("probe has no requests")
    if probe_data.get("status") != "completed":
        raise ValueError("probe is not completed")

    phase_stats: dict[str, Any] = {}
    for phase in ("cold", "warm"):
        phase_rows = [row for row in rows if row.get("phase") == phase]
        phase_stats[phase] = {
            "requests": len(phase_rows),
            "prompt_tokens": _scalar_stats(phase_rows, "prompt_token_count"),
            "combined_case_wall_ms": _scalar_stats(phase_rows, "combined_case_wall_ms"),
            "completion_wall_ms": _scalar_stats(phase_rows, "completion_wall_ms"),
            "ttft_ms": _scalar_stats(phase_rows, "ttft_ms"),
            "tokenize_wall_ms": _scalar_stats(phase_rows, "tokenize_wall_ms"),
            "prompt_eval_ms": _timing_stats(phase_rows, "prompt_ms"),
            "predicted_eval_ms": _timing_stats(phase_rows, "predicted_ms"),
            "prompt_tokens_per_second": _timing_stats(phase_rows, "prompt_per_second"),
            "predicted_tokens_per_second": _timing_stats(phase_rows, "predicted_per_second"),
            "prompt_work_tokens": _timing_stats(phase_rows, "prompt_n"),
            "predicted_work_tokens": _timing_stats(phase_rows, "predicted_n"),
            "prompt_cached_work_tokens": _row_stats(phase_rows, "timing_prompt_cached"),
            "protocol_status": _counts(phase_rows, "protocol_status"),
            "cap_status": _counts(phase_rows, "cap_status"),
            "truncated": _counts(phase_rows, "truncated"),
        }

    summary = probe_data.get("summary", {})
    contract = probe_data.get("native_contract", {})
    profile = probe_data.get("server_profile_expected", {})
    return {
        "schema_version": 1,
        "status": "completed_redacted_scalar_aggregate",
        "measurement_scope": "One CUDA graph-opt=0 arm; four stored TRAIN mechanical rows, one cold and one identical cached replay per row.",
        "provenance": {
            "logical_run": "RUN-05-primary500-cuda-graph0-a",
            "source": "llama.cpp b10453",
            "source_commit": "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
            "graph_opt": launch_data.get("graph_opt"),
            "backend": launch_data.get("backend"),
            "model_sha256": launch_data.get("model_sha256"),
            "cuda_backend_sha256": launch_data.get("cuda_backend_sha256"),
            "probe_sha256": launch_data.get("probe_sha256"),
        },
        "contract": {
            "fixture_rows": summary.get("fixture_rows"),
            "requests": len(rows),
            "cold_requests": sum(row.get("phase") == "cold" for row in rows),
            "warm_requests": sum(row.get("phase") == "warm" for row in rows),
            "accepted_requests": sum(row.get("protocol_status") == "accepted" for row in rows),
            "prompt_tokens_including_manual_bos": contract.get("tokens_evaluated"),
            "completion_cap": contract.get("cap"),
            "context_size": contract.get("vocab_size") and profile.get("ctx"),
            "user_deadline_ms": summary.get("user_deadline_ms"),
            "cap_includes_terminal_eos": summary.get("cap_includes_terminal_eos"),
            "no_retry": summary.get("no_retry_policy"),
        },
        "phases": phase_stats,
        "server_observations": _parse_server_log(server_log),
        "process": {
            "server_exit_code": terminal_data.get("server_exit_code"),
            "wall_seconds": terminal_data.get("seconds"),
            "peak_server_rss_kib": terminal_data.get("peak_server_rss_kib"),
        },
        "redactions": [
            "prompt text and prompt hashes",
            "row identifiers and token ID arrays/hashes",
            "returned text and output hashes",
            "absolute filesystem paths, credentials, and user context",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.probe, args.launch, args.terminal, args.server_log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
