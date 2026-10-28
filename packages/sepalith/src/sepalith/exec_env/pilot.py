"""Exec-env pilot: score gold, rewritten-correct, broken and arm A completions with the execution reward.

    # arm A samples (llama.cpp CPU, Q8_0, 8 threads, nice 10): greedy + k at temperature 1.0
    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.exec_env.pilot sample \\
        --tasks OUT/tasks.jsonl --out OUT/arm-a-samples.jsonl [--per-function 2 --k 4]
    # score everything, write the receipt
    ... pilot score --tasks OUT/tasks.jsonl --samples OUT/arm-a-samples.jsonl --root AUTHORS --receipt R.json

Completion kinds:

- ``gold``: the reference remainder (must score 1.0);
- ``alt_*``: deterministic behaviour-preserving rewrites of gold that are not
  textually equal (``alt_assign``: ``<-`` -> ``=`` on a statement;
  ``alt_rename``: a variable introduced in the remainder renamed consistently;
  ``alt_comment``: an added comment line; ``alt_return``: the final value
  wrapped in ``return()``); they measure whether tests accept non-exact
  correct code;
- ``broken_*``: truncated (closers only), cut-local mutants, a syntax error,
  ``quit()``, an escape from the function, a no-op and protocol-invalid text;
- ``arm_a_greedy`` / ``arm_a_t1``: model samples, also scored with reward v2.1
  so exec acceptance can be compared with canonical exact match.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import statistics
import tempfile
import time

from sepalith.campaign_protocol import serialize_target
from sepalith.data.synth.common import split_comment
from sepalith.exec_env import mutants as mutants_module
from sepalith.exec_env.harness import ExecHarness
from sepalith.exec_env.reward import ExecReward, load_tasks

GGUF = Path("/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-Q8_0.gguf")
PARENT_TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json")
CAP = 1024
SEED = 20261025
CLOSER = re.compile(r"^[\s})\],]*$")


def body_lines(task: dict) -> list[str]:
    return task["target_body_text"].split("\n")


def alt_variants(task: dict) -> dict[str, list[str]]:
    target = body_lines(task)
    out: dict[str, list[str]] = {}
    prefix = "\n".join(task["context"]["prefix"]) + "\n" + "\n".join(task["context"]["region_old"])
    after = task["exec"]["after"]
    for index, line in enumerate(target):
        match = re.match(r"^(\s+)([A-Za-z.][A-Za-z0-9._]*) <- (.*)$", line)
        if match and "function(" not in line:
            alt = list(target)
            alt[index] = f"{match.group(1)}{match.group(2)} = {match.group(3)}"
            out["alt_assign"] = alt
            break
    for line in target:
        match = re.match(r"^\s+([A-Za-z.][A-Za-z0-9._]*)\s*(<-|=)(?!=)", split_comment(line)[0])
        if not match:
            continue
        name = match.group(1)
        word = re.compile(r"(?<![A-Za-z0-9._$@])" + re.escape(name) + r"(?![A-Za-z0-9._(])(?!\s*=(?!=))")
        if word.search(prefix) or word.search(after) or len(name) < 2:
            continue
        new = name + "_v"
        alt = [word.sub(new, ln) if '"' not in ln and "'" not in ln else None for ln in target]
        if None in alt or alt == target:
            continue
        out["alt_rename"] = alt  # type: ignore[assignment]
        break
    lasts = [i for i, ln in enumerate(target) if not CLOSER.match(ln)]
    if lasts:
        i = lasts[-1]
        indent = re.match(r"\s*", target[i]).group(0)  # type: ignore[union-attr]
        out["alt_comment"] = [*target[:i], f"{indent}# result", *target[i:]]
        last = target[i]
        code = split_comment(last)[0].strip()
        is_final = i + 1 < len(target) and target[i + 1] == "}" and all(CLOSER.match(ln) for ln in target[i + 1:])
        if (is_final and re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*(\[[^\]]*\])?|[A-Za-z.][\w.]*\([^()]*\)", code)
                and not code.startswith(("return(", "stop(", "invisible(", "warning(", "message("))):
            out["alt_return"] = [*target[:i], f"{indent}return({code})", *target[i + 1:]]
    return out


def broken_variants(task: dict, unit_cache: dict[str, list]) -> dict[str, list[str] | str]:
    target = body_lines(task)
    out: dict[str, list[str] | str] = {}
    closers = [ln for ln in target if CLOSER.match(ln) and ln.strip()]
    if closers and task["family"] == "finish_block":
        out["broken_truncated"] = closers
    gold = task["exec"]["gold_text"]
    before = task["exec"]["before"].split("\n") if task["exec"]["before"] else []
    after = task["exec"]["after"].split("\n") if task["exec"]["after"] else []
    start = len(before)
    if task["family"] == "finish_block":
        if gold not in unit_cache:
            tokens = mutants_module.parse_tokens([gold + "\n"])[0]
            unit_cache[gold] = mutants_module.generate(gold + "\n", tokens) if tokens else []
        local = [m for m in unit_cache[gold] if m.line1 - 1 >= start + (1 if task["curriculum"].get("signature_region") else 0)]
        rng = random.Random(f"pilot:{task['id']}")
        for n, mutant in enumerate(rng.sample(local, min(3, len(local)))):
            lines = mutant.text.rstrip("\n").split("\n")
            out[f"broken_mutant_{n}"] = lines[start:len(lines) - len(after)]
    for i, line in enumerate(target):
        if "(" in line and ")" in line:
            out["broken_syntax"] = [*target[:i], line.replace(")", "", 1), *target[i + 1:]]
            break
    out["broken_quit"] = ['  quit("no")', *closers] if task["family"] == "finish_block" else ['  quit("no")']
    if task["family"] == "finish_block":
        out["broken_escape"] = [*target, "identical <- function(...) TRUE"]
    out["broken_noop"] = "NOOP"
    out["broken_protocol"] = "INVALID"
    return out


def text_for(lines: list[str] | str, task: dict) -> str:
    if lines == "NOOP":
        return serialize_target("no_op", [])
    if lines == "INVALID":
        return task["target_body_text"] + "\n"
    return serialize_target("replace", lines)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- gate statistics
def gate_stats(root: Path) -> dict:
    """Aggregate gate.json files: verdicts, assertion counts, kill rates with and without declared equivalents."""
    verdicts: Counter = Counter()
    skips: Counter = Counter()
    checks, rates, raw_rates, stubs = [], [], [], []
    by_op: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    totals = Counter()
    for directory in sorted(p for p in root.glob("*/*") if (p / "meta.json").is_file()):
        if (directory / "SKIP").exists():
            verdicts["skip"] += 1
            skips[(directory / "SKIP").read_text().strip().split(":")[0].split()[0].lower()[:40]] += 1
            continue
        gate_path = directory / "gate.json"
        if not gate_path.is_file():
            verdicts["no_test"] += 1
            continue
        gate = json.loads(gate_path.read_text())
        verdicts[gate["verdict"]] += 1
        if gate["verdict"] == "skip":
            skips[";".join(gate["problems"])] += 1
        if gate["verdict"] != "pass":
            continue
        m = gate["mutants"]
        checks.append(gate["checks"])
        stubs.append(gate["stub_pass_share"])
        rates.append(m["kill_rate"])
        raw_rates.append(m["killed"] / (m["valid"] + m["declared_equivalent"]))
        for key in ("generated", "sampled", "invalid", "declared_equivalent", "valid", "killed"):
            totals[key] += m[key]
        for op, cell in m["by_op"].items():
            by_op[op][0] += cell["killed"]
            by_op[op][1] += cell["valid"]
    return {"verdicts": dict(verdicts), "skip_reasons": dict(skips),
            "assertions_median": statistics.median(checks) if checks else None,
            "assertions_min": min(checks, default=None),
            "kill_rate_median": statistics.median(rates) if rates else None, "kill_rate_min": min(rates, default=None),
            "kill_rate_with_equivalents_as_survivors_median": statistics.median(raw_rates) if raw_rates else None,
            "kill_rate_with_equivalents_as_survivors_min": round(min(raw_rates), 3) if raw_rates else None,
            "stub_pass_share_max": max(stubs, default=None), "mutant_totals": dict(totals),
            "pooled_kill_rate": round(totals["killed"] / totals["valid"], 4) if totals["valid"] else None,
            "kill_by_op": {op: {"killed": k, "valid": v, "rate": round(k / v, 3)} for op, (k, v) in sorted(by_op.items())}}


# --------------------------------------------------------------------------- pilot selection
def command_select(args: argparse.Namespace) -> int:
    """Pick pilot functions from *train* slices: one per package, varied shapes, plain inputs."""
    from sepalith.exec_env.slices import materialize

    manifest = json.loads(args.slices.read_text())
    rows = {r["id"]: r for r in map(json.loads, args.pool.open())}
    train = [s for s in manifest["slices"] if s["role"] == "train"]
    rng = random.Random(SEED)
    candidates = []
    for entry in train:
        for fid in entry["function_ids"]:
            row = rows[fid]
            shape = row["shape"]
            code = row["unit"]
            # plain-input heuristics: few arguments, few helpers, no S4/R6/environment plumbing
            if len(row["formals"]) > 4 or len(row["helpers"]) > 2 or not 10 <= row["body_nonblank"] <= 45:
                continue
            if re.search(r"setClass|setMethod|R6Class|new\(|\.self|self\$|private\$|UseMethod|NextMethod|"
                         r"\bmodel\.frame|\bterms\(|\boptim\(|\buniroot\(|\bintegrate\(", code):
                continue
            candidates.append((entry["slice"], row["package"], fid, shape["body_bucket"]))
    rng.shuffle(candidates)
    seen_packages: set[str] = set()
    per_bucket: Counter = Counter()
    picked: dict[str, list[str]] = defaultdict(list)
    quota = args.n // 4 + 1
    for slice_id, package, fid, bucket in candidates:
        if package in seen_packages or per_bucket[bucket] >= quota:
            continue
        seen_packages.add(package)
        per_bucket[bucket] += 1
        picked[slice_id].append(fid)
        if sum(per_bucket.values()) >= args.n:
            break
    sub = {"schema": manifest["schema"], "slices": [
        {"slice": f"P{index:02d}", "role": "pilot", "packages": [], "functions": len(chunk), "function_ids": chunk}
        for index, chunk in enumerate(_chunks([f for ids in picked.values() for f in ids], args.per_author))]}
    written = materialize(sub, list(rows.values()), args.root)
    (args.root / "pilot-selection.json").write_text(json.dumps({"buckets": dict(per_bucket), "slices": sub["slices"],
                                                                 "source_slices": {k: v for k, v in picked.items()}},
                                                                indent=1) + "\n")
    print(json.dumps({"picked": written, "buckets": dict(per_bucket), "author_slices": len(sub["slices"])}))
    return 0


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# --------------------------------------------------------------------------- sampling
def command_sample(args: argparse.Namespace) -> int:
    from sepalith.quant import gguf_dev as gd
    from sepalith.quant.llama_tools import gpu_layers, llama_bin, tool_env

    if args.threads > 8:
        raise SystemExit("at most 8 threads")
    tasks = [json.loads(line) for line in args.tasks.open()]
    by_function: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        if task["family"] == "finish_block":
            by_function[task["function_id"]].append(task)
    chosen = []
    for fid in sorted(by_function):
        group = sorted(by_function[fid], key=lambda t: t["id"])
        random.Random(f"{SEED}:{fid}").shuffle(group)
        chosen += group[:args.per_function]
    chosen = chosen[:args.max_tasks] if args.max_tasks else chosen
    done: Counter = Counter()
    if args.out.exists():  # resume: keep complete tasks, regenerate partial ones
        kept = [line for line in args.out.read_text().splitlines() if line.strip()]
        for line in kept:
            done[json.loads(line)["task_id"]] += 1
        complete = {t for t, n in done.items() if n == 1 + args.k}
        args.out.write_text("".join(line + "\n" for line in kept if json.loads(line)["task_id"] in complete))
        chosen = [t for t in chosen if t["id"] not in complete]
    if not chosen:
        print(json.dumps({"tasks": 0, "note": "all chosen tasks already sampled"}))
        return 0
    prompt_max = max(len(t["prompt_ids"]) for t in chosen)
    ctx = -(-(prompt_max + CAP) // 256) * 256
    port = gd.free_port()
    argv = ["nice", "-n", "10", str(llama_bin("llama-server", "cpu")), "-m", str(GGUF), "-c", str(ctx), "-np", "1",
            "-t", str(args.threads), "-tb", str(args.threads), *gpu_layers("cpu"), "--host", "127.0.0.1",
            "--port", str(port), "--no-webui", "--seed", str(SEED), "--no-context-shift", "--no-cache-prompt"]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exec-pilot-") as tmp, args.out.open("a") as stream:
        with gd.llama_server(argv, port, tool_env("cpu"), Path(tmp) / "server.log") as url:
            for index, task in enumerate(chosen):
                prompt = task["prompt_ids"]
                requests = [("arm_a_greedy", gd.completion_request(prompt, CAP, SEED))]
                for k in range(args.k):
                    body = gd.completion_request(prompt, CAP, SEED + 1 + k)
                    body.update(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                                samplers=["top_k", "top_p", "min_p", "temperature"])
                    requests.append((f"arm_a_t1_{k}", body))
                for kind, body in requests:
                    response = gd.post_json(f"{url}/completion", body, 900)
                    ids = gd.generated_ids(response, len(prompt), CAP)
                    stream.write(json.dumps({"task_id": task["id"], "kind": kind, "completion_ids": ids}) + "\n")
                stream.flush()
                if (index + 1) % 10 == 0:
                    print(f"[{index + 1}/{len(chosen)}] {time.monotonic() - started:.0f}s", flush=True)
    print(json.dumps({"tasks": len(chosen), "samples_per_task": 1 + args.k,
                      "wall_seconds": round(time.monotonic() - started, 1)}))
    return 0


# --------------------------------------------------------------------------- scoring
def v21_exact(task_rows: dict[str, dict], samples: list[dict]) -> dict[int, dict]:
    from sepalith.campaign_protocol import PromptContext
    from sepalith.data.synth.build import make_scorer
    from sepalith.training.rl.reward_v21 import Evidence, ReferenceSpec, Sample

    scorer, _formatter, decoder = make_scorer(8)
    ids = sorted({s["task_id"] for s in samples})
    specs = [ReferenceSpec(i, task_rows[i]["family"], PromptContext.from_dict(task_rows[i]["context"]), "replace",
                           task_rows[i]["target_body_text"], Evidence("unverified")) for i in ids]
    references = scorer.build_references(specs)
    out = {}
    batch = [Sample(s["task_id"], completion_ids=tuple(s["completion_ids"]), sample_id=str(n)) for n, s in enumerate(samples)]
    for record in scorer.score(references, batch, decoder):
        out[int(str(record["sample_id"]))] = {"exact": bool(record["exact"]), "reward": float(str(record["reward"])),
                                               "outcome": record["outcome"]}
    return out


def command_score(args: argparse.Namespace) -> int:
    from sepalith.training.rl.reward_v21 import tokenizer_decoder

    rows = {t["id"]: t for t in map(json.loads, args.tasks.open())}
    tasks = load_tasks(args.tasks)
    harness = ExecHarness(workers=min(args.workers, 8))
    reward = ExecReward(tasks, harness, tokenizer_decoder(PARENT_TOKENIZER))
    unit_cache: dict[str, list] = {}
    text_items: list[tuple[str, str, str]] = []  # (task, kind, text)
    for tid, task in sorted(rows.items()):
        text_items.append((tid, "gold", text_for(body_lines(task), task)))
        for kind, lines in alt_variants(task).items():
            text_items.append((tid, kind, text_for(lines, task)))
        for kind, lines in broken_variants(task, unit_cache).items():
            text_items.append((tid, kind, text_for(lines, task)))
    started = time.perf_counter()
    text_records = reward.score_texts([t for t, _, _ in text_items], [x for _, _, x in text_items])
    text_seconds = time.perf_counter() - started
    results = [{"task_id": t, "kind": k, **r} for (t, k, _), r in zip(text_items, text_records, strict=True)]
    gold_ids = [(tid, tuple(rows[tid]["gold_completion_ids"])) for tid in sorted(rows)]
    gold_token = reward([t for t, _ in gold_ids], [c for _, c in gold_ids])
    samples = [json.loads(line) for line in args.samples.open()] if args.samples and args.samples.exists() else []
    samples = [sample for sample in samples if sample["task_id"] in tasks]
    if samples:
        model_records = reward([s["task_id"] for s in samples], [tuple(s["completion_ids"]) for s in samples])
        exact21 = v21_exact(rows, samples)
        for n, (sample, record) in enumerate(zip(samples, model_records, strict=True)):
            kind = "arm_a_greedy" if sample["kind"] == "arm_a_greedy" else "arm_a_t1"
            results.append({"task_id": sample["task_id"], "kind": kind, **record, "v21": exact21.get(n)})
    # throughput: a GRPO-sized batch (32 rewards) of model-like completions, repeated
    batch_items = [(t, x) for t, k, x in text_items if k in ("gold", "alt_comment", "broken_mutant_0")]
    latencies = []
    rng = random.Random(SEED)
    for _ in range(args.batches):
        picked = rng.sample(batch_items, min(32, len(batch_items)))
        t0 = time.perf_counter()
        reward.score_texts([t for t, _ in picked], [x for _, x in picked])
        latencies.append(time.perf_counter() - t0)
    job_walls = sorted(r.get("execution", {}).get("elapsed", 0.0) for r in results if r.get("execution"))
    harness.close()
    receipt = summarize(rows, results, gold_token, latencies, job_walls, text_seconds, len(text_items), harness)
    if args.root:
        receipt["gate"] = gate_stats(args.root)
    args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    if args.records:
        with args.records.open("w") as stream:
            for record in results:
                stream.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "by_function"}, indent=1))
    return 0


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(rows: dict, results: list[dict], gold_token: list[dict], latencies: list[float], job_walls: list[float],
              text_seconds: float, n_text: int, harness: ExecHarness) -> dict:
    by_kind: dict[str, Counter] = defaultdict(Counter)
    rewards: dict[str, list[float]] = defaultdict(list)
    for r in results:
        family = "alt" if r["kind"].startswith("alt_") else "broken" if r["kind"].startswith("broken_") else r["kind"]
        for key in sorted({r["kind"], family}):
            by_kind[key]["n"] += 1
            by_kind[key]["outcome:" + str(r["outcome"])] += 1
            by_kind[key]["n_exact_match"] += int(bool(r.get("exact_match")))
            by_kind[key]["n_exec_all_pass"] += int(bool(r.get("exact")))
            rewards[key].append(float(r["reward"]))
    kinds = {k: {**dict(c), "mean_reward": round(statistics.fmean(rewards[k]), 4),
                 "reward_1.0": sum(1 for x in rewards[k] if x >= 1.0)} for k, c in sorted(by_kind.items())}
    model = [r for r in results if r["kind"].startswith("arm_a")]
    nonexact = [r for r in model if r.get("protocol_valid") and r.get("operation") != "no_op"
                and not (r.get("v21") or {}).get("exact")]
    accepted = [r for r in nonexact if r.get("exact")]
    exact_v21 = [r for r in model if (r.get("v21") or {}).get("exact")]
    alt = [r for r in results if r["kind"].startswith("alt_")]
    return {
        "schema": "sepalith.exec-env.pilot.v1",
        "tasks": len(rows), "functions": len({t["function_id"] for t in rows.values()}),
        "packages": len({t["package"] for t in rows.values()}),
        "families": dict(Counter(t["family"] for t in rows.values())),
        "completions_scored": len(results),
        "gold_token_path_reward_1": sum(1 for r in gold_token if r["reward"] == 1.0), "gold_token_path_n": len(gold_token),
        "by_kind": kinds,
        "key_numbers": {
            "alt_correct_nonexact_accepted": [sum(1 for r in alt if r.get("exact")), len(alt)],
            "arm_a_samples": len(model),
            "arm_a_v21_exact": len(exact_v21),
            "arm_a_exec_all_pass": sum(1 for r in model if r.get("exact")),
            "arm_a_nonexact_valid_edits": len(nonexact),
            "arm_a_nonexact_accepted_by_exec": len(accepted),
            "arm_a_v21_exact_but_exec_fail": sum(1 for r in exact_v21 if not r.get("exact")),
        },
        "throughput": {"workers": harness.workers, "sandbox": harness.sandbox_mode,
                       "text_completions": n_text, "text_seconds": round(text_seconds, 2),
                       "rewards_per_second": round(n_text / text_seconds, 1) if text_seconds else None,
                       "batch32_seconds_p50": round(pct(latencies, 0.5), 3),
                       "batch32_seconds_p95": round(pct(latencies, 0.95), 3),
                       "job_seconds_p50": round(pct(job_walls, 0.5), 4), "job_seconds_p95": round(pct(job_walls, 0.95), 4)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--tasks", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--per-function", type=int, default=2)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--max-tasks", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("--tasks", type=Path, required=True)
    c.add_argument("--samples", type=Path)
    c.add_argument("--receipt", type=Path, required=True)
    c.add_argument("--records", type=Path)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--batches", type=int, default=30)
    c.add_argument("--root", type=Path, help="author root whose gate.json files are summarised")
    p = sub.add_parser("select")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--slices", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--n", type=int, default=56)
    p.add_argument("--per-author", type=int, default=14)
    args = parser.parse_args(argv)
    return {"sample": command_sample, "score": command_score, "select": command_select}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
