#!/usr/bin/env python3
"""Run the existing native probe while retaining a DSpark arm label/counters.

This is a client adapter only. It delegates HTTP, token, EOS, cap, timeout,
and parser behavior to the shared TRAIN-only runtime_native_probe.py. It never
loads a model or generates a token locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DRAFT_FIELDS = (
    "draft_n",
    "draft_n_accepted",
    "draft_n_verif_steps",
    "draft_n_tokens",
    "draft_n_accepted_tokens",
)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("sepalith_runtime_native_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import probe source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True, help="shared runtime_native_probe.py")
    p.add_argument("--fixture", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--url", required=True)
    p.add_argument(
        "--arm",
        choices=("model-free-ngram", "trained-dspark", "released-dspark"),
        required=True,
    )
    p.add_argument("--cap", type=int, default=192)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--user-deadline-ms", type=int, default=5000)
    p.add_argument("--diagnostic-timeout-ms", type=int, default=60000)
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    probe = load_module(Path(args.probe).resolve())
    captured: list[dict[str, object] | None] = []
    original_stream = probe.stream_completion

    def stream_with_final(*stream_args: Any, **stream_kwargs: Any) -> Any:
        result = original_stream(*stream_args, **stream_kwargs)
        final = result.get("final") if isinstance(result, dict) else None
        captured.append(final if isinstance(final, dict) else None)
        return result

    probe.stream_completion = stream_with_final
    original_case = probe.run_case

    def case_with_counters(*case_args: Any, **case_kwargs: Any) -> dict[str, object]:
        before = len(captured)
        record = original_case(*case_args, **case_kwargs)
        if len(captured) > before and captured[before] is not None:
            final = captured[before]
            for key in DRAFT_FIELDS:
                if key in final:
                    record[key] = final[key]
            record["draft_counters_status"] = (
                "present" if "draft_n" in final and "draft_n_accepted" in final else "missing"
            )
            record["draft_counters_source"] = "completion_final"
        else:
            record["draft_counters_status"] = "missing"
            record["draft_counters_source"] = "completion_not_completed"
        return record

    # run_probe resolves run_case and stream_completion through module globals.
    probe.run_case = case_with_counters
    namespace = SimpleNamespace(
        fixture=args.fixture,
        manifest=args.manifest,
        url=args.url,
        arm=args.arm,
        cap=args.cap,
        context=args.context,
        reps=args.reps,
        user_deadline_ms=args.user_deadline_ms,
        diagnostic_timeout_ms=args.diagnostic_timeout_ms,
    )
    result = probe.run_probe(namespace)
    result["adapter"] = {
        "schema_version": "sepalith.r2.dspark-probe-adapter.v1",
        "shared_probe": str(Path(args.probe).resolve()),
        "arm_label_only": True,
        "draft_counter_fields": list(DRAFT_FIELDS),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "out": str(out)}, sort_keys=True))
    return 0 if result.get("status") in {"completed", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
