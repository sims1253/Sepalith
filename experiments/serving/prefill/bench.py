#!/usr/bin/env python3
"""Time-to-suggestion bench for PRM03 edit sessions on llama-server (CPU).

Runs the frozen session set (``sessions.jsonl`` + ``manifest.json`` from
``build_sessions.py``) against a config-driven llama-server: spawn args,
warm-up policy and request flow all come from a JSON config, so candidate
configurations are data, not code.  Measures, per simulated event:

* client TTFT (first streamed token, ms) and full-response wall time;
* server timings (prompt_ms, cached vs processed prompt tokens, decode);
* generated token ids (for token-identical parity against a reference run);
* warm-up request overhead (count, wall seconds, tokens prefilled).

Aggregates by event type with session-cluster bootstrap CIs and emits
``METRIC name=value`` lines for the autoresearch loop.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "packages" / "sepalith" / "src"))

from sepalith.campaign_protocol import (  # noqa: E402
    TASK_CONTRACT,
    PromptContext,
    render_prompt,
)

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path("/tmp/prefill-models")
SERVER_BIN = Path(
    "/home/m0hawk/.local/opt/llama.cpp-b10453/build-cpu/bin/llama-server"
)
DEFAULT_TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
    "full/checkpoint-536/tokenizer.json"
)
ARTIFACTS = Path(
    "/mnt/e/sepalith/campaign-20260915/serving-work/prefill-20261017"
)
EDIT_TYPES = frozenset({"type", "accept", "diag", "cursor_move"})
WARMUP_SECTION_MARKERS = (
    "<filename>selected_references",
    "<filename>edit_history",
    "<filename>diagnostics",
    "<filename>retrieval",
)


@dataclass(frozen=True)
class Config:
    name: str
    model: str
    server_args: tuple[str, ...]
    warmup_policy: str
    warmup_events: tuple[str, ...]
    n_predict: int
    seed: int

    @classmethod
    def load(cls, path: Path) -> Config:
        data = json.loads(path.read_text(encoding="utf-8"))
        warmup = data.get("warmup", {})
        request = data.get("request", {})
        return cls(
            name=str(data.get("name", path.stem)),
            model=str(data["model"]),
            server_args=tuple(data.get("server_args", [])),
            warmup_policy=str(warmup.get("policy", "none")),
            warmup_events=tuple(warmup.get("events", ("open", "cursor_move", "accept"))),
            n_predict=int(request.get("n_predict", 64)),
            seed=int(request.get("seed", 3407)),
        )


def completion_request(
    ids: Sequence[int], n_predict: int, seed: int, stream: bool, cache_prompt: bool = True
) -> dict[str, Any]:
    return {
        "prompt": list(ids),
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "xtc_probability": 0.0,
        "mirostat": 0,
        "ignore_eos": False,
        "stop": [],
        "seed": seed,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
        "stream": stream,
        "n_probs": 0,
        "samplers": ["top_k"],
    }


def stream_completion(
    port: int,
    ids: Sequence[int],
    n_predict: int,
    seed: int,
    timeout: float = 600.0,
    cache_prompt: bool = True,
) -> dict[str, Any]:
    """POST /completion with stream=True; return TTFT, wall, tokens, timings."""
    body = json.dumps(
        completion_request(ids, n_predict, seed, stream=True, cache_prompt=cache_prompt)
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    began = time.monotonic()
    ttft_ms: float | None = None
    tokens: list[int] = []
    timings: dict[str, Any] = {}
    stop_type: str | None = None
    try:
        conn.request(
            "POST", "/completion", body=body, headers={"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"/completion returned {response.status}: {response.read()[:400]!r}")
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunk_tokens = event.get("tokens")
            if isinstance(chunk_tokens, list):
                tokens.extend(t for t in chunk_tokens if type(t) is int)
            has_token = bool(event.get("content")) or (
                isinstance(chunk_tokens, list) and bool(chunk_tokens)
            )
            if ttft_ms is None and has_token:
                ttft_ms = (time.monotonic() - began) * 1000.0
            if isinstance(event.get("timings"), dict):
                timings = event["timings"]
            if isinstance(event.get("stop_type"), str):
                stop_type = event["stop_type"]
    finally:
        conn.close()
    wall_ms = (time.monotonic() - began) * 1000.0
    if ttft_ms is None:
        ttft_ms = wall_ms
    return {
        "ttft_ms": round(ttft_ms, 1),
        "wall_ms": round(wall_ms, 1),
        "tokens": tokens,
        "stop_type": stop_type,
        "timings": timings,
    }


def warmup_completion(port: int, ids: Sequence[int], timeout: float = 600.0) -> dict[str, Any]:
    """Prefill-only request (n_predict=0): warms the slot cache, no decode."""
    body = completion_request(ids, 0, 3407, stream=False)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    began = time.monotonic()
    try:
        conn.request(
            "POST",
            "/completion",
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"warmup /completion returned {response.status}")
        payload = json.loads(response.read())
    finally:
        conn.close()
    wall_ms = (time.monotonic() - began) * 1000.0
    return {"wall_ms": round(wall_ms, 1), "timings": payload.get("timings", {})}


class Server:
    """Config-driven llama-server child with tracked-PID lifecycle."""

    def __init__(self, config: Config, port: int, log_path: Path) -> None:
        self.port = port
        self.log_path = log_path
        model = MODELS_DIR / config.model
        if not model.exists():
            raise FileNotFoundError(f"model not staged: {model}")
        argv = [
            "nice",
            "-n",
            "10",
            str(SERVER_BIN),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *config.server_args,
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("wb")
        self.process = subprocess.Popen(
            argv,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        self.started = time.monotonic()

    def wait_ready(self, timeout: float = 900.0) -> float:
        while time.monotonic() - self.started < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited {self.process.returncode}; see {self.log_path}"
                )
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.request("GET", "/health")
                response = conn.getresponse()
                response.read()
                conn.close()
                if response.status == 200:
                    return time.monotonic() - self.started
            except OSError:
                pass
            time.sleep(1.0)
        raise RuntimeError("llama-server did not become healthy")

    def stop(self) -> None:
        if self.process.poll() is None:
            os.kill(self.process.pid, signal.SIGTERM)
            for _ in range(100):
                if self.process.poll() is not None:
                    break
                time.sleep(0.1)
            if self.process.poll() is None:
                os.kill(self.process.pid, signal.SIGKILL)
        self.process.wait()
        self.log.close()


def head_ids_for_policy(
    context: Mapping[str, Any], full_ids: Sequence[int], tokenizer, policy: str  # noqa: ANN001
) -> list[int]:
    """Warm-up prompt ids under a policy: the longest true prefix of full_ids
    covered by sections that survive the next edit (contract, path, prefix,
    section markers; plus accumulated history for ``stable_history``), or the
    entire prompt for ``full`` (prefill-as-you-settle)."""
    if policy == "full":
        return list(full_ids)
    parts: list[str] = [TASK_CONTRACT, f"<filename>{context['path']}", *context["prefix"]]
    if policy == "stable_history":
        parts.extend(WARMUP_SECTION_MARKERS[:1])
        parts.extend(event["event_diff"] for event in context["history"])
        parts.extend(WARMUP_SECTION_MARKERS[1:])
    else:
        parts.extend(WARMUP_SECTION_MARKERS)
    head_text = "\n".join(parts) + "\n"
    head = tokenizer.encode(head_text, add_special_tokens=False).ids
    body = list(full_ids[1:])  # full_ids[0] is the manual BOS
    k = 0
    for a, b in zip(head, body):
        if a != b:
            break
        k += 1
    return [0, *body[:k]]


def load_sessions(
    sessions_path: Path, manifest_path: Path, tokenizer  # noqa: ANN001
) -> list[dict[str, Any]]:
    """Verify the frozen set against the manifest and pre-tokenize prompts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (event["session_id"], event["index"]): event for event in manifest["events"]
    }
    sessions: list[dict[str, Any]] = []
    for line in sessions_path.read_text(encoding="utf-8").splitlines():
        session = json.loads(line)
        prepared = []
        for index, event in enumerate(session["events"]):
            pin = expected[(session["session_id"], index)]
            context = PromptContext.from_mapping(event["context"])
            text = render_prompt(context)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest != pin["prompt_sha256"]:
                raise RuntimeError(
                    f"session drift at {session['session_id']}[{index}]: sha mismatch"
                )
            ids = [0, *tokenizer.encode(text, add_special_tokens=False).ids]
            if len(ids) - 1 != pin["prompt_tokens"]:
                raise RuntimeError(
                    f"session drift at {session['session_id']}[{index}]: token count"
                )
            prepared.append({"type": event["type"], "context": event["context"], "ids": ids})
        sessions.append({"session_id": session["session_id"], "events": prepared})
    return sessions


def run_pass(
    config: Config,
    sessions: Sequence[dict[str, Any]],
    port: int,
    pass_index: int,
    log_dir: Path,
    cold: bool = False,
) -> list[dict[str, Any]]:
    server = Server(config, port, log_dir / f"server-pass{pass_index}.log")
    records: list[dict[str, Any]] = []
    warmup_total_s = 0.0
    warmup_tokens = 0
    warmup_count = 0
    startup_s: float | None = None
    try:
        startup_s = server.wait_ready()
        for session in sessions:
            for event_index, event in enumerate(session["events"]):
                if config.warmup_policy != "none" and event["type"] in config.warmup_events:
                    head = head_ids_for_policy(
                        event["context"], event["ids"], TOKENIZER, config.warmup_policy
                    )
                    if head:
                        result = warmup_completion(port, head)
                        warmup_total_s += result["wall_ms"] / 1000.0
                        warmup_count += 1
                        warmup_tokens += len(head)
                result = stream_completion(
                    port, event["ids"], config.n_predict, config.seed, cache_prompt=not cold
                )
                timings = result["timings"]
                records.append(
                    {
                        "pass": pass_index,
                        "session_id": session["session_id"],
                        "index": event_index,
                        "type": event["type"],
                        "prompt_tokens": len(event["ids"]),
                        "ttft_ms": result["ttft_ms"],
                        "wall_ms": result["wall_ms"],
                        "cache_n": timings.get("cache_n"),
                        "processed_n": timings.get("prompt_n"),
                        "prompt_ms": timings.get("prompt_ms"),
                        "predicted_n": timings.get("predicted_n"),
                        "predicted_ms": timings.get("predicted_ms"),
                        "output_ids": result["tokens"],
                        "stop_type": result["stop_type"],
                    }
                )
                print(
                    json.dumps(
                        {
                            "pass": pass_index,
                            "session": session["session_id"][:8],
                            "type": event["type"],
                            "ttft_ms": result["ttft_ms"],
                            "cache_n": timings.get("cache_n"),
                            "processed_n": timings.get("prompt_n"),
                        }
                    ),
                    flush=True,
                )
    finally:
        server.stop()
    for record in records:
        record.update(
            {
                "startup_s": round(startup_s, 1) if startup_s else None,
                "warmup_total_s": round(warmup_total_s, 2),
                "warmup_count": warmup_count,
                "warmup_tokens": warmup_tokens,
            }
        )
    return records


def cluster_bootstrap_p50(
    values_by_session: Mapping[str, list[float]], samples: int = 2000, seed: int = 7
) -> list[float]:
    """Session-cluster bootstrap of the pooled median (correlated events)."""
    import random

    rng = random.Random(seed)
    keys = list(values_by_session)
    if not keys:
        return []
    pooled = [v for vs in values_by_session.values() for v in vs]
    point = statistics.median(pooled)
    out = [point]
    if len(keys) < 2:
        return out
    stats: list[float] = []
    for _ in range(samples):
        sample: list[float] = []
        for _ in range(len(keys)):
            sample.extend(values_by_session[rng.choice(keys)])
        stats.append(statistics.median(sample))
    stats.sort()
    out = [point, stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]]
    return [round(x, 1) for x in out]


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return round(ordered[index], 1)


def summarize(records: Sequence[dict[str, Any]], reference: Mapping[str, Sequence[int]] | None = None) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record["type"], []).append(record)
    summary: dict[str, Any] = {"by_type": {}, "config_records": len(records)}
    for event_type, rows in sorted(by_type.items()):
        ttfts = [r["ttft_ms"] for r in rows]
        walls = [r["wall_ms"] for r in rows]
        cached = [
            r["cache_n"] / r["prompt_tokens"]
            for r in rows
            if r["cache_n"] is not None and r["prompt_tokens"]
        ]
        by_session: dict[str, list[float]] = {}
        for row in rows:
            by_session.setdefault(row["session_id"], []).append(row["ttft_ms"])
        summary["by_type"][event_type] = {
            "n": len(rows),
            "ttft_p50_ms": percentile(ttfts, 0.50),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "ttft_mean_ms": round(statistics.mean(ttfts), 1),
            "ttft_p50_ci": cluster_bootstrap_p50(by_session),
            "wall_p50_ms": percentile(walls, 0.50),
            "cached_frac_p50": round(statistics.median(cached), 3) if cached else None,
            "processed_n_p50": percentile(
                [r["processed_n"] or 0 for r in rows], 0.50
            ),
            "prompt_tokens_p50": percentile([r["prompt_tokens"] for r in rows], 0.50),
        }
    edit_rows = [r for r in records if r["type"] in EDIT_TYPES]
    edit_by_session: dict[str, list[float]] = {}
    for row in edit_rows:
        edit_by_session.setdefault(row["session_id"], []).append(row["ttft_ms"])
    slow_rows = [r for r in records if r["type"] in ("accept", "cursor_move", "diag")]
    slow_by_session: dict[str, list[float]] = {}
    for row in slow_rows:
        slow_by_session.setdefault(row["session_id"], []).append(row["ttft_ms"])
    parity_total = parity_match = 0
    if reference:
        for row in records:
            key = f'{row["session_id"]}:{row["index"]}:{row["pass"]}'
            expected = reference.get(key)
            if expected is not None:
                parity_total += 1
                parity_match += int(list(row["output_ids"]) == list(expected))
    edit_ttfts = [r["ttft_ms"] for r in edit_rows]
    summary["edit"] = {
        "n": len(edit_rows),
        "ttft_p50_ms": percentile(edit_ttfts, 0.50) if edit_ttfts else None,
        "ttft_p95_ms": percentile(edit_ttfts, 0.95) if edit_ttfts else None,
        "ttft_p50_ci": cluster_bootstrap_p50(edit_by_session),
        "slow_ttft_p50_ms": percentile([r["ttft_ms"] for r in slow_rows], 0.50)
        if slow_rows
        else None,
        "slow_ttft_p50_ci": cluster_bootstrap_p50(slow_by_session) if slow_rows else None,
        "warmup_total_s": records[0]["warmup_total_s"] if records else 0.0,
        "warmup_count": records[0]["warmup_count"] if records else 0,
        "parity_rate": round(parity_match / parity_total, 4) if parity_total else None,
        "parity_n": parity_total,
    }
    return summary


TOKENIZER = None  # set in main


def main() -> int:
    global TOKENIZER  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, default=HERE / "sessions.jsonl")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--port", type=int, default=18071)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--reference", type=Path, default=HERE / "reference_outputs.json")
    parser.add_argument("--write-reference", action="store_true", help="freeze this run as the parity reference")
    parser.add_argument("--cold", action="store_true", help="cold mode: cache_prompt=false per request (per-event cold numerics; also the parity reference semantics)")
    args = parser.parse_args()

    import tokenizers

    TOKENIZER = tokenizers.Tokenizer.from_file(str(args.tokenizer))

    config = Config.load(args.config)
    sessions = load_sessions(args.sessions, args.manifest, TOKENIZER)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    label = args.label or config.name
    out_path = args.out or (
        ARTIFACTS / f"run-{label}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )

    records: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as out:
        for pass_index in range(args.passes):
            pass_records = run_pass(
                config, sessions, args.port, pass_index, ARTIFACTS, cold=args.cold
            )
            records.extend(pass_records)
            out.flush()
        for record in records:
            out.write(json.dumps(record) + "\n")
    if args.write_reference:
        reference = {
            f'{r["session_id"]}:{r["index"]}:{r["pass"]}': r["output_ids"]
            for r in records
        }
        args.reference.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    reference = None
    if args.reference.exists():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
    summary = summarize(records, reference)
    summary["cold"] = args.cold
    summary.update(
        {
            "label": label,
            "config_name": config.name,
            "model": config.model,
            "passes": args.passes,
            "server_args": list(config.server_args),
            "warmup_policy": config.warmup_policy,
            "records_path": str(out_path),
            "host_load1": float(open("/proc/loadavg").read().split()[0]),
        }
    )
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    edit = summary["edit"]
    by_type = summary["by_type"]
    print(f"METRIC edit_ttft_p50_ms={edit['ttft_p50_ms']}")
    print(f"METRIC edit_ttft_p95_ms={edit['ttft_p95_ms']}")
    print(f"METRIC slow_edit_ttft_p50_ms={edit['slow_ttft_p50_ms']}")
    if edit.get("parity_rate") is not None:
        print(f"METRIC parity_rate={edit['parity_rate']}")
    print(f"METRIC cold_ttft_p50_ms={by_type['open']['ttft_p50_ms']}")
    print(f"METRIC cold_ttft_p95_ms={by_type['open']['ttft_p95_ms']}")
    print(f"METRIC cold_wall_p50_ms={by_type['open']['wall_p50_ms']}")
    for name in ("switch_back", "type", "accept", "cursor_move"):
        if name in by_type:
            print(f"METRIC {name}_ttft_p50_ms={by_type[name]['ttft_p50_ms']}")
    print(f"METRIC edit_cached_frac={by_type['open']['cached_frac_p50']}")
    print(f"METRIC warmup_total_s={edit['warmup_total_s']}")
    print(f"METRIC host_load1={summary['host_load1']}")
    print(f"SUMMARY {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
