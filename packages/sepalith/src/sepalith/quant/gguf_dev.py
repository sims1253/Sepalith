"""DEV generation for a GGUF through llama-server, scored like the PC evaluator.

The PC evaluator (``sepalith.training.eval.campaign_eval``) renders every DEV
case with ``build_training_row`` and the HF tokenizer, generates greedily from
``input_ids[:target_start]`` (cap 1,024, stop ids 1 and 130073), decodes
the generated ids without the stop token, and scores with ``classify`` and
``generation_termination``.  This module does the same, except that
generation runs in llama-server:

- the prompt is sent as the same token-id list (no text, no added BOS);
- one fresh request per case: ``temperature`` 0, ``top_k`` 1,
  ``cache_prompt`` false, no text stop strings, ``n_predict`` = cap;
- llama.cpp stops on its end-of-generation ids, which for this vocabulary
  are exactly 1 and 130073 (checked per case: any other stop is an error);
- ``return_tokens`` gives the generated ids, including the stop token.

Per-case records and the summary use the PC keys.  ``loss`` is null: there is
no teacher-forced pass, so the summary's ``prompt_nll``/``target_nll`` are
null too.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from sepalith.campaign_protocol import NATIVE_EOG_IDS, PromptContext, RENDERER_ID, build_training_row
from sepalith.training.eval.campaign_eval import classify, generation_termination

SCHEMA = "sepalith.c09.gguf-dev-generation.v1"
GATE_CAP = 1024
PANELS = {
    "dev75": ("/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl",
              "7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035", 75),
    "dev250": ("/mnt/e/sepalith/campaign-20260915/data-work/C05-dev250-v1r3/dev250-v1.jsonl",
               "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22", 250),
}


class DevTokenizer(Protocol):
    """The HF tokenizer surface the PC evaluator uses: ``encode`` for rendering, ``decode`` for scoring."""

    def encode(self, text: str, *, add_special_tokens: bool = False,
               split_special_tokens: bool = True) -> Sequence[int]: ...

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = False) -> str: ...


class GenerationError(RuntimeError):
    pass


def load_panel(path: Path) -> list[dict[str, Any]]:
    cases, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case["split"] != "dev":
            raise ValueError("only development records may enter the GGUF DEV evaluator")
        if not case["id"] or case["id"] in seen or not case.get("package_id") or not case.get("family"):
            raise ValueError("development cases need unique ids and package/family provenance")
        seen.add(case["id"])
        cases.append(case)
    return cases


def prepare(cases: Sequence[Mapping[str, Any]], tokenizer: DevTokenizer) -> list[tuple[Mapping[str, Any], PromptContext, dict]]:
    """Render each case exactly as ``development_evaluator`` does."""
    prepared = []
    for case in cases:
        context = PromptContext.from_mapping(case["context"])
        row = build_training_row(context, operation=case["operation"], region_new=case["region_new"],
                                 tokenizer=tokenizer, row_id=case["id"], family=case["family"],
                                 package_id=case["package_id"], split="dev")
        if row["renderer_id"] != RENDERER_ID:
            raise ValueError("renderer differs")
        prepared.append((case, context, row))
    return prepared


def select(prepared: list, limit: int | None, order: str) -> list:
    if order == "shortest":
        prepared = sorted(prepared, key=lambda item: (item[2]["target_start"], item[0]["id"]))
    elif order != "panel":
        raise ValueError(f"unknown case order {order!r}")
    return prepared[:limit] if limit else prepared


def completion_request(prompt_ids: Sequence[int], cap: int, seed: int = 3407) -> dict[str, Any]:
    return {"prompt": list(prompt_ids), "n_predict": cap, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0, "mirostat": 0,
            "ignore_eos": False, "stop": [], "seed": seed, "cache_prompt": False, "return_tokens": True,
            "stream": False, "n_probs": 0, "samplers": ["top_k"]}


def generated_ids(response: Mapping[str, Any], prompt_len: int, cap: int, stop_ids: Sequence[int] = NATIVE_EOG_IDS) -> list[int]:
    """Validate one llama-server /completion response and return its generated ids."""
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
        raise GenerationError("response has no generated token ids")
    if response.get("tokens_evaluated") != prompt_len:
        raise GenerationError(f"server evaluated {response.get('tokens_evaluated')} prompt tokens, sent {prompt_len}")
    if not 0 < len(tokens) <= cap:
        raise GenerationError(f"generated {len(tokens)} tokens with cap {cap}")
    stop_type = response.get("stop_type")
    if stop_type == "eos":
        if tokens[-1] not in stop_ids:
            raise GenerationError(f"stopped on id {tokens[-1]}, not one of {list(stop_ids)}")
    elif stop_type == "limit":
        if len(tokens) != cap or tokens[-1] in stop_ids:
            raise GenerationError("limit stop without a full cap")
    else:
        raise GenerationError(f"unexpected stop_type {stop_type!r}")
    if any(t in stop_ids for t in tokens[:-1]):
        raise GenerationError("generation continued past a stop id")
    return tokens


def case_record(case: Mapping[str, Any], context: PromptContext, row: Mapping[str, Any], generated: list[int],
                seconds: float, tokenizer: DevTokenizer, cap: int) -> dict[str, Any]:
    """The PC evaluator's per-case record (``campaign_eval.development_evaluator``), loss null."""
    termination = generation_termination(generated, cap, NATIVE_EOG_IDS)
    raw_ids = generated[:-1] if termination["response_complete"] else generated
    raw = tokenizer.decode(raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    expected_noop = list(context.region_old) == case["region_new"]
    outcome = classify(raw, context, case["region_new"], generated)
    return {
        "id": case["id"], "package_id": case["package_id"], "family": case["family"],
        "expected_noop": expected_noop, "strata": case.get("strata", {}),
        "prompt_tokens": row["target_start"], "generated_tokens": len(generated),
        "reference_tokens": len(row["input_ids"]) - row["target_start"],
        "reference_exceeds_generation_cap": len(row["input_ids"]) - row["target_start"] > cap,
        **termination,
        "generation_seconds": seconds, "raw_output": raw, "generated_ids": generated,
        "loss": None, **outcome,
    }


def summarize(results: Sequence[Mapping[str, Any]], cap: int, panel_sha256: str, cases_path: str) -> dict[str, Any]:
    """The PC evaluator's summary keys; NLL fields are null (no teacher-forced pass)."""
    denominators = {"cases": len(results), "complete_cases": sum(r["response_complete"] for r in results),
                    "incomplete_cases": sum(not r["response_complete"] for r in results),
                    "packages": len({r["package_id"] for r in results}),
                    "strict_noop": sum(r["expected_noop"] for r in results),
                    "edits": sum(not r["expected_noop"] for r in results),
                    "prompt_loss_tokens": None, "target_loss_tokens": None}
    counts = {name: sum(r[name] for r in results)
              for name in ("protocol_valid", "exact_region", "predicted_noop", "suggestion", "cap_hit")}
    counts.update(strict_noop_correct=sum(r["expected_noop"] and r["predicted_noop"] for r in results),
                  strict_noop_false_suggestions=sum(r["expected_noop"] and r["suggestion"] for r in results),
                  edit_exact=sum(not r["expected_noop"] and r["exact_region"] for r in results))
    complete = [r for r in results if r["response_complete"]]
    complete_case_counts = {"protocol_valid": sum(r["protocol_valid"] for r in complete),
                            "exact_region": sum(r["exact_region"] for r in complete),
                            "edit_exact": sum(not r["expected_noop"] and r["exact_region"] for r in complete),
                            "strict_noop_correct": sum(r["expected_noop"] and r["predicted_noop"] for r in complete),
                            "strict_noop_false_suggestions": sum(r["expected_noop"] and r["suggestion"] for r in complete)}
    return {"case_ids": [r["id"] for r in results], "denominators": denominators, "counts": counts,
            "complete_case_counts": complete_case_counts, "generation_max_new_tokens": cap,
            "prompt_nll": None, "target_nll": None, "panel_sha256": panel_sha256, "cases_path": cases_path,
            "interpretation": "Development exact/protocol/no-op diagnostic from a GGUF in llama.cpp; "
                              "no R semantic-validity or final-release claim."}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(url: str, body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


@contextlib.contextmanager
def llama_server(argv: list[str], port: int, env: Mapping[str, str], log_path: Path,
                 startup_timeout: float = 900.0) -> Iterator[str]:
    """Run llama-server until the block ends; wait for /health first."""
    with log_path.open("w") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=dict(env))
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    raise GenerationError(f"llama-server exited with {process.returncode}; see {log_path}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (URLError, ConnectionError, TimeoutError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise GenerationError("llama-server did not become healthy")
                time.sleep(2)
            yield f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
