#!/usr/bin/env python3
"""rewrite_author_spark.py — author-under-constraint loop on the muse-spark /
openrouter free backends (overnight data-ops driver).

Same proven machinery as rewrite_author_zai.py (spec mining, prompt
construction, the FULL gate stack, resume-safe done-keys, drvfs writers,
429 breaker) — IMPORTED, not duplicated, so gate semantics stay identical.
Differences:

  * authors on opencode-spark (muse-spark-1.2-contributor, GO quota,
    Responses API), opencode-spark-free, openrouter dots-3-note, or
    openrouter stealth/ox-alpha (evaluation-only until judged trusted);
  * distinct outputs (never touches the zai driver's files):
      rewrite_lint_fix_spark.jsonl   (mine+inject, spark/dots authors)
      rewrite_fixissue_spark.jsonl   (buinject arm)
      rewrite_lint_fix_oxalpha.jsonl (ox-alpha lint-fix wave, purgeable)
  * own spec pool results/rewrite_author_spark/spec_pool.jsonl (fresh
    seed 137) and skips spec ids the zai driver already banked;
  * `judge` subcommand: single-attempt ox-alpha vs muse-spark on the same
    constraint-fix specs — objective gate pass rates + blind glm-5.3
    pairwise scoring — producing the pre-volume trust verdict.

Usage (system python3 from experiments/synthetic-data):
  python3 rewrite_author_spark.py mine
  python3 rewrite_author_spark.py author --backend opencode-spark [--arms mine,inject]
  python3 rewrite_author_spark.py judge --n 40 --mix mine:30,inject:10
  python3 rewrite_author_spark.py stats

Keys come from the environment (OPENCODE_API_KEY / OPENROUTER_API_KEY /
ZAI_API_KEY — judge only; the bulk zai quota belongs to rewrite_author_zai).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scenarios as S                                   # noqa: E402
import cases.corpus as C                                # noqa: E402
import cases.validators as V                            # noqa: E402
from cases.compound import BaseSample                   # noqa: E402
from cases.backends import (AgyBackend, BackendError, OpencodeBackend,
                            OpencodeSparkBackend, OpenrouterBackend,
                            ZaiBackend)  # noqa: E402
import rewrite_maint_proto as RW                        # noqa: E402
import rewrite_author_zai as ZA                         # noqa: E402

OUT_DIR = HERE / "results" / "rewrite_author_spark"
POOL_PATH = OUT_DIR / "spec_pool.jsonl"

DATASETS = Path("/mnt/h/sepalith/datasets/cases_v1")
LINTFIX_SPARK = DATASETS / "rewrite_lint_fix_spark.jsonl"
LINTFIX_SPARKFREE = DATASETS / "rewrite_lint_fix_sparkfree.jsonl"
FIXISSUE_SPARK = DATASETS / "rewrite_fixissue_spark.jsonl"
LINTFIX_OX = DATASETS / "rewrite_lint_fix_oxalpha.jsonl"
# zai takeover (only after the sibling rewrite_author_zai process exits)
LINTFIX_ZAI = DATASETS / "rewrite_lint_fix_zai.jsonl"
FIXISSUE_ZAI = DATASETS / "rewrite_fixissue_zai.jsonl"
# openrouter free-model variety wave (per-row model tags attribute/purge)
LINTFIX_ORFREE = DATASETS / "rewrite_lint_fix_orfree.jsonl"
FIXISSUE_ORFREE = DATASETS / "rewrite_fixissue_orfree.jsonl"
ZAI_DRIVER_DONE = DATASETS / "rewrite_lint_fix.jsonl.done.jsonl"   # read-only

ARMS = ("mine", "inject", "buinject")


# ---------------------------------------------------------------------------
# authoring backends
# ---------------------------------------------------------------------------

class SparkAuthorBackend(OpencodeSparkBackend):
    """GO-tier muse-spark authoring via the Responses API: generous output
    budget (reasoning burns first), JSON mode, low effort for minimal
    diffs."""
    name = "opencode-spark"
    timeout_s = 240.0

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt}]}],
            "max_output_tokens": 6000,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_object"}},
        }


class SparkFreeAuthorBackend(SparkAuthorBackend):
    """Same authoring payload on the free tier: harder rate limits."""
    name = "opencode-spark-free"
    url = "https://opencode.ai/zen/v1/responses"
    model = "muse-spark-1.2-contributor-free"
    pace_gap_s = 8.0
    cooldown_s = 900.0


class DotsAuthorBackend(OpenrouterBackend):
    """dots-3-note fallback author (shared free cap — polite pacing)."""
    name = "openrouter-dots"
    timeout_s = 120.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 8000,
                "reasoning": {"effort": "low"},
                "messages": [{"role": "user", "content": prompt}]}


class OxCalphaBackend(OpenrouterBackend):
    """openrouter stealth model — EVALUATION ONLY until the trust verdict;
    never pointed at buinject. max_tokens 8000: measured ~3k hidden
    reasoning tokens count against the completion budget, so 4k truncates
    the JSON on longer functions."""
    name = "oxalpha"
    model = "stealth/ox-alpha"
    timeout_s = 150.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}]}


class ZaiJudgeBackend(ZaiBackend):
    """glm-5.3 as a blind pairwise judge (judge_drafting.py pattern)."""
    name = "zai"
    model = "glm-5.3"
    pace_gap_s = 2.0

    def _payload(self, prompt: str) -> dict:
        return {"model": "glm-5.3", "thinking": {"type": "enabled"},
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": 900, "temperature": 0}


class OxCalpha12kBackend(OxCalphaBackend):
    """ox-alpha RE-CHECK config (user flag): 12k output budget, no format
    pressure, lenient extraction — tests whether the first eval's failure
    was formatting rather than capability. EVALUATION ONLY."""
    name = "oxalpha-12k"
    timeout_s = 240.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 12000,
                "messages": [{"role": "user", "content": prompt}]}


class XpreviewFreeAuthorBackend(OpencodeBackend):
    """x-preview-f-free on opencode zen CHAT-COMPLETIONS (NOT the
    responses base). REVERSED VERDICT (coordinator): 11/12 accepted (92%)
    through the full author gates via opencode — the openrouter "ox-alpha"
    rejections were a provider artifact, not the model. Top-tier volume
    backend; free tier on zen so pace it (3s gaps, 429 cooldown inherited
    from OpencodeBackend)."""
    name = "xpreview-free"
    model = "x-preview-f-free"
    url = "https://opencode.ai/zen/v1/chat/completions"
    timeout_s = 180.0
    pace_gap_s = 3.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 4000,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}



class AgyAuthorBackend(AgyBackend):
    """gemini-3.7-flash-low via the agy CLI as an author (Google family,
    separate agy quota from opus-4.6; user-designated generator+judge)."""
    name = "agy"
    timeout_s = 150.0
    pace_gap_s = 1.0



class OxAlphaGoBackend(OpencodeBackend):
    """ox-alpha on the GO subscription tier (user tip): chat-completions
    works cleanly here (json mode + separate reasoning, ~14 tok overhead),
    subscription rate limits instead of the starved free tier."""
    name = "oxalpha-go"
    model = "ox-alpha-free"
    url = "https://opencode.ai/zen/go/v1/chat/completions"
    timeout_s = 180.0
    pace_gap_s = 1.5

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 4000,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}



class OxAlphaNousBackend(OpencodeBackend):
    """stealth/ox-alpha on Nous Research's inference API (user key,
    NOUS_PORTAL_API_KEY) — third independent provider for the model."""
    name = "oxalpha-nous"
    model = "stealth/ox-alpha"
    url = "https://inference-api.nousresearch.com/v1/chat/completions"
    env_key = "NOUS_PORTAL_API_KEY"
    timeout_s = 180.0
    pace_gap_s = 2.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 4000,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}


AUTHOR_BACKENDS = {
    "opencode-spark": SparkAuthorBackend,
    "opencode-spark-free": SparkFreeAuthorBackend,
    "openrouter-dots": DotsAuthorBackend,
    "oxalpha": OxCalphaBackend,
    "oxalpha-12k": OxCalpha12kBackend,
    "xpreview-free": XpreviewFreeAuthorBackend,   # reversed verdict 92%
    "zai": ZA.ZaiAuthorBackend,          # takeover after the sibling exits
    "agy": AgyAuthorBackend,            # gemini-3.7 author (parallel quota)
    "oxalpha-go": OxAlphaGoBackend,     # ox on the GO subscription (user tip)
    "oxalpha-nous": OxAlphaNousBackend,  # ox via Nous portal (third provider)
}


class OpenrouterFreeAuthorBackend(OpenrouterBackend):
    """Any openrouter :free model as an author ("openrouter-free:<id>") —
    always spot-checked (gates + glm-5.3 blind pairwise vs muse-spark)
    before a volume wave. No reasoning param: support varies per model.
    The instance name carries the model id so every model gets PER-MODEL
    output files (own done-keys): that is what makes the multi-model
    rotation possible and each model's rows purgeable."""
    timeout_s = 150.0

    def __init__(self, model_id: str):
        self.model = model_id
        self.name = f"openrouter-free:{model_id}"
        super().__init__()

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}]}


class ZenFreeAuthorBackend(OpencodeBackend):
    """Any opencode-zen free model as an author ("zen-free:<id>") via
    CHAT-COMPLETIONS (not the responses base). Motivated by the ox-alpha
    reversal: openrouter serving can fail while the model is fine — zen
    chat-completions serves plain content with reasoning_content properly
    separated. JSON mode + free-tier pacing (3s gaps, inherited 429
    cooldown). Per-model output files (own done-keys) via arm_out."""
    timeout_s = 180.0
    pace_gap_s = 3.0

    def __init__(self, model_id: str):
        self.model = model_id
        self.name = f"zen-free:{model_id}"
        super().__init__()

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 6000,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}


def resolve_backend(name: str):
    if name.startswith("openrouter-free:"):
        return OpenrouterFreeAuthorBackend(name.split(":", 1)[1])
    if name.startswith("zen-free:"):
        return ZenFreeAuthorBackend(name.split(":", 1)[1])
    if name not in AUTHOR_BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; known: "
                         f"{sorted(AUTHOR_BACKENDS)} + openrouter-free:<id> "
                         f"+ zen-free:<id>")
    return AUTHOR_BACKENDS[name]()


def arm_out(backend_name: str, arm: str) -> tuple[str, Path]:
    """(family, output file) per arm; ox-alpha is lint-fix only."""
    if backend_name == "oxalpha" and arm == "buinject":
        raise SystemExit("oxalpha is lint-fix only (stealth, purgeable wave)")
    if backend_name.startswith("openrouter-free:"):
        slug = backend_name.split(":", 1)[1].replace("/", "-") \
            .replace(":free", "")
        if arm == "buinject":
            return "fix_issue_inject", DATASETS / \
                f"rewrite_fixissue_orfree_{slug}.jsonl"
        return "rewrite_lint_fix", DATASETS / \
                f"rewrite_lint_fix_orfree_{slug}.jsonl"
    if backend_name == "xpreview-free":   # per-model files, own done-keys
        if arm == "buinject":
            return "fix_issue_inject", DATASETS / \
                "rewrite_fixissue_xpreview-free.jsonl"
        return "rewrite_lint_fix", DATASETS / \
            "rewrite_lint_fix_xpreview-free.jsonl"
    if backend_name.startswith("zen-free:"):
        slug = backend_name.split(":", 1)[1].replace("/", "-") \
            .replace(":free", "")
        if arm == "buinject":
            return "fix_issue_inject", DATASETS / \
                f"rewrite_fixissue_zenfree_{slug}.jsonl"
        return "rewrite_lint_fix", DATASETS / \
            f"rewrite_lint_fix_zenfree_{slug}.jsonl"
    if arm == "buinject":
        return "fix_issue_inject", {"zai": FIXISSUE_ZAI}.get(backend_name,
                                                            FIXISSUE_SPARK)
    return "rewrite_lint_fix", {"oxalpha": LINTFIX_OX,
                                "oxalpha-12k": LINTFIX_OX,
                                "zai": LINTFIX_ZAI,
                                "opencode-spark-free": LINTFIX_SPARKFREE
                                }.get(backend_name, LINTFIX_SPARK)


# ---------------------------------------------------------------------------
# PHASE 1: mine my own spec pool (deterministic, zero quota)
# ---------------------------------------------------------------------------

def cmd_mine(args) -> int:
    rng = random.Random(args.seed)
    pool_path = Path(args.out_pool) if args.out_pool else POOL_PATH
    existing: set[str] = set()
    if pool_path.exists() and not args.fresh:
        for line in pool_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
    pool_pkgs = RW.sample_packages(rng, args.tidy_packages, args.random_packages)
    print(f"[mine] pool: {len(pool_pkgs)} packages "
          f"(tidy={args.tidy_packages} rest={args.random_packages}, "
          f"seed={args.seed}); {len(existing)} specs already in pool",
          flush=True)

    quota = dict(mine=args.mine_specs, inject=args.inject_specs,
                 buinject=args.buinject_specs)
    stats = dict(functions_scanned=0, specs_emitted=0, per_rule=Counter(),
                 per_arm=Counter(), findings_total=0, packages_seen=set())
    site = len(existing)
    specs_buf: list[dict] = []
    t0 = time.time()

    def emit(bs: BaseSample, findings: list[dict], arm: str) -> None:
        nonlocal site
        b = bs.b
        bs_base_id = RW.base_sample_id(bs)
        start = bs.fn.parent.start_byte \
            if (bs.fn.parent is not None
                and bs.fn.parent.type == "binary_operator") \
            else bs.fn.start_byte
        suffix = f":{findings[0]['sb']}" if arm == "buinject" else ""
        spec = dict(
            id=f"rw:{arm}:{bs_base_id}{suffix}", arm=arm,
            base_sample_id=bs_base_id, package=b.package, path=b.rel,
            fn_head=b.line_str(bs.head_row), rows=[bs.top_row, bs.r1],
            start=start, end=bs.fn.end_byte,
            top_row=bs.top_row, head_row=bs.head_row,
            nlines=b.nlines(),
            behavior_call=RW.fn_signature_callable(bs),
            findings=[dict(rule=f["rule"], row=f["row"],
                           erow=f.get("erow", f["row"]), sb=f["sb"], eb=f["eb"],
                           old=f["old"], new=f["new"], fix=f.get("fix"),
                           col=b.rowcol(f["sb"])[1], snippet=f["old"],
                           rationale=f["rationale"])
                      for f in findings],
            src_b64=ZA.base64.b64encode(b.src).decode("ascii"),
            generated_at=ZA._now())
        specs_buf.append(spec)
        site += 1
        stats["specs_emitted"] += 1
        stats["per_arm"][arm] += 1
        stats["findings_total"] += len(findings)
        for f in findings:
            stats["per_rule"][f["rule"]] += 1

    try:
        for b in C.iter_bundles_highest(pool_pkgs, rng):
            if all(v <= 0 for v in quota.values()) or \
                    time.time() - t0 > args.time_budget:
                break
            stats["packages_seen"].add(b.package)
            for fn in (n for n in V._walk(b.tree.root_node)
                       if n.type == "function_definition"):
                if all(v <= 0 for v in quota.values()) or \
                        time.time() - t0 > args.time_budget:
                    break
                geom = C._fn_body(b, fn)
                if geom is None:
                    continue
                _body, _head, _r0, _r1, nb = geom
                if not 6 <= len(nb) <= 60:
                    continue
                stats["functions_scanned"] += 1
                try:
                    bs = BaseSample(b, fn, site)
                except ValueError:
                    continue

                def ok_span(f: dict) -> bool:
                    return b.line_str(f["row"]).strip() and \
                        f.get("erow", f["row"]) - f["row"] < 8

                if quota["mine"] > 0:
                    findings = RW.drop_overlaps(
                        [f for f in RW.detect_findings(bs) if ok_span(f)])
                    if findings:
                        emit(bs, findings[:6], "mine")
                        quota["mine"] -= 1
                        continue
                if quota["inject"] > 0:
                    findings = RW.drop_overlaps(
                        [f for f in RW.detect_injectable(bs) if ok_span(f)])
                    if findings:
                        emit(bs, findings[:4], "inject")
                        quota["inject"] -= 1
                        continue
                if quota["buinject"] > 0:
                    one = RW.drop_overlaps(
                        [f for f in RW.detect_bug_injectable(bs) if ok_span(f)])
                    for f in one[:args.bug_per_fn]:
                        if quota["buinject"] <= 0:
                            break
                        emit(bs, [f], "buinject")
                        quota["buinject"] -= 1
    except KeyboardInterrupt:
        print("[mine] interrupted; keeping buffered specs", flush=True)

    new = [s for s in specs_buf if s["id"] not in existing]
    dups = len(specs_buf) - len(new)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "a") as fh:                     # local fs, no drvfs
        for s in new:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    total = len(existing) + len(new)
    report = dict(ts=ZA._now(), seed=args.seed,
                  elapsed_s=round(time.time() - t0, 1),
                  functions_scanned=stats["functions_scanned"],
                  packages_seen=len(stats["packages_seen"]),
                  new_specs=len(new), duplicate_ids_skipped=dups,
                  pool_total=total, quota_left=dict(quota),
                  per_arm=dict(stats["per_arm"]),
                  per_rule=dict(stats["per_rule"].most_common()))
    ZA._write_json(OUT_DIR / "mine_report.json", report)
    print(f"[mine] +{len(new)} specs ({dups} dups skipped) in "
          f"{report['elapsed_s']}s; pool now {total}; "
          f"per_arm={report['per_arm']} quota_left={quota}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# PHASE 2: the author loop (resume-safe, <=4 in flight, 429 breaker)
# ---------------------------------------------------------------------------

def load_pool(arms: list[str], pool_path: Path | None = None) -> list[dict]:
    path = pool_path or POOL_PATH
    if not path.exists():
        sys.exit(f"no spec pool at {path}; run `mine` first")
    specs = []
    for line in path.read_text().splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue
        if s.get("arm") in arms:
            specs.append(s)
    specs.sort(key=lambda s: (s["arm"] != "mine", s["arm"] != "inject",
                              s["id"]))
    return specs


def external_done_keys() -> set[str]:
    """Spec ids the sibling zai driver already banked (read-only skip):
    both its done files, so the takeover never re-burns its territory."""
    keys: set[str] = set()
    for done in (ZAI_DRIVER_DONE,
                 DATASETS / "fix_issue_inject.jsonl.done.jsonl"):
        if done.exists():
            for line in done.read_text().splitlines():
                try:
                    keys.add(json.loads(line)["key"])
                except (ValueError, KeyError):
                    pass
    return keys


def cmd_author(args) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; known: {ARMS}")
    backend = resolve_backend(args.backend)
    specs = load_pool(arms, Path(args.pool) if args.pool else None)
    skip = external_done_keys() if args.skip_zai_done else set()

    outs: dict[Path, dict] = {}
    for arm in arms:
        _family, out = arm_out(args.backend, arm)
        done_path = Path(str(out) + ".done.jsonl")
        if out not in outs:
            outs[out] = dict(out=out, done_path=done_path,
                             done=ZA.load_done(done_path),
                             hashes=ZA.load_hashes(out), rows=0,
                             stats=dict(attempted=0, accepted=0, retried=0,
                                        backend_error=0, layer1_json=0,
                                        extract_rawslice=0,
                                        gate_rejects=Counter(),
                                        behavior_failed=0, dups=0,
                                        per_arm=Counter()))
    pending = [s for s in specs
               if s["id"] not in outs[arm_out(args.backend, s["arm"])[1]]["done"]
               and s["id"] not in skip]
    print(f"[author] backend={backend.name} model={backend.model} "
          f"arms={arms} pool={len(specs)} external_skip={len(skip)} "
          f"pending={len(pending)} (done keys: " +
          ", ".join(f"{Path(k).name}={len(v['done'])}"
                    for k, v in outs.items()) + ")", flush=True)
    if args.max > 0:
        pending = pending[:args.max]

    breaker = ZA.Breaker(max_rounds=args.breaker_rounds)
    stats = dict(started=ZA._now(), arms=arms, backend=backend.name,
                 model=backend.model, workers=args.workers,
                 pool=len(specs), pending=len(pending))
    site_lock, site_ctr = threading.Lock(), [0]
    stats_lock = threading.Lock()

    def bump(entry: dict, key: str, n: int = 1):
        with stats_lock:
            entry["stats"][key] += n

    t0 = time.time()
    last_log = 0.0
    stop_hard = threading.Event()
    be_fail_lock, be_fail = threading.Lock(), [0]

    def process(spec: dict) -> dict:
        ctx = ZA.build_ctx(spec)
        with site_lock:
            site_ctr[0] += 1
            site = site_ctr[0]
        outs_entry = outs[arm_out(backend.name, spec["arm"])[1]]
        bump(outs_entry, "attempted")
        feedback = None
        last_raw_kind = "?"
        first_prompt = ""
        for attempt in (1, 2):
            if not breaker.wait_turn() or stop_hard.is_set():
                return dict(kind="aborted", spec=spec)
            prompt = ZA.build_prompt(ctx, feedback)
            if attempt == 1:
                first_prompt = prompt
            try:
                raw = backend.complete(prompt)
                breaker.report(ok=True, rate_error=False)
            except BackendError as e:
                breaker.report(ok=False, rate_error=(e.kind == "rate"))
                bump(outs_entry, "backend_error")
                with be_fail_lock:
                    be_fail[0] += 1
                    hard = be_fail[0] >= 15
                if hard:
                    stop_hard.set()
                return dict(kind="backend_error", spec=spec, err=str(e)[:160])
            with be_fail_lock:
                be_fail[0] = 0
            fn_text, kind = ZA.extract_fn_text(raw)
            last_raw_kind = kind
            if fn_text is None:
                bump(outs_entry, "layer1_json")
                feedback = "the response was not parseable (return ONLY the JSON object)."
                continue
            fn_text = "\n".join(ZA._norm_trim(fn_text))
            if kind == "rawslice":
                bump(outs_entry, "extract_rawslice")
            gates = ZA.run_gates(ctx, fn_text, site)
            beh, beh_detail = ZA.behavior_gate(ctx, fn_text)
            gates["behavior"] = beh
            rows = ZA.build_rows(ctx, fn_text, gates["_spliced"],
                                 first_prompt, kind)
            for r in rows:                    # our provenance, not zai's
                r["case"] = "rewrite_author_spark"
                r["backend"] = backend.name
                r["model"] = backend.model
            gates["fix_applied"] = all(
                ZA.fix_applied(f, ctx["arm"], "\n".join(r["region_old"]),
                               "\n".join(r["region_new"]))
                for f, r in zip(ctx["findings"], rows))
            gates["row_check"] = all(RW.check_rewrite_row(r)[0] for r in rows)
            summary = ZA.gate_summary(gates, beh)
            if not summary:
                gate_snap = {k: v for k, v in gates.items()
                             if not k.startswith("_") and k != "loc_delta"}
                for r in rows:
                    r["gates"] = gate_snap
                return dict(kind="accepted", spec=spec, ctx=ctx, rows=rows,
                            gates=gates, attempts=attempt)
            feedback = summary
            if attempt == 2:
                for k in ZA.GATE_ORDER:
                    if gates.get(k) is False:
                        with stats_lock:
                            outs_entry["stats"]["gate_rejects"][k] += 1
                if beh == "failed":
                    bump(outs_entry, "behavior_failed")
                return dict(kind="rejected", spec=spec, gates=gates,
                            reason=summary, attempts=2, kind2=last_raw_kind)
            bump(outs_entry, "retried")
        return dict(kind="rejected", spec=spec, gates={}, reason="unparseable",
                    attempts=2, kind2=last_raw_kind)

    def flush_stats(final: bool = False):
        for out, entry in outs.items():
            if not entry["stats"]["attempted"] and not final:
                continue
            rep = dict(stats)
            rep.update(dict(ts=ZA._now(), out=str(out), final=final,
                            rows_total=entry["rows"],
                            elapsed_s=round(time.time() - t0, 1),
                            backend_stats=backend.stats_summary(),
                            breaker=dict(rate_errors=breaker.rate_errors,
                                         rounds=breaker.rounds,
                                         stopped=breaker.stop.is_set()),
                            counts={k: (dict(v) if isinstance(v, Counter) else v)
                                    for k, v in entry["stats"].items()},
                            done_keys=len(entry["done"])))
            ZA._write_json(Path(str(out) + ".stats.json"), rep)

    ex = ThreadPoolExecutor(max_workers=args.workers)
    outstanding: set[Future] = set()
    it = iter(pending)
    n_done = 0
    try:
        while True:
            if time.time() - t0 > args.time_budget or breaker.stop.is_set() \
                    or stop_hard.is_set():
                break
            while len(outstanding) < args.workers * 2:
                try:
                    s = next(it)
                except StopIteration:
                    break
                outstanding.add(ex.submit(process, s))
            if not outstanding:
                break
            done_set, _ = wait(outstanding, timeout=30,
                               return_when=FIRST_EXCEPTION)
            for fut in done_set:
                outstanding.discard(fut)
                try:
                    res = fut.result()
                except Exception as e:                       # driver bug guard
                    print(f"  [worker-exception] {e!r}", flush=True)
                    continue
                n_done += 1
                spec, arm = res["spec"], res["spec"]["arm"]
                entry = outs[arm_out(backend.name, arm)[1]]
                if res["kind"] == "accepted":
                    new_rows = []
                    for row in res["rows"]:
                        if row["content_hash"] in entry["hashes"]:
                            entry["stats"]["dups"] += 1
                            continue
                        entry["hashes"].add(row["content_hash"])
                        new_rows.append(row)
                    for row in new_rows:
                        ZA._append_line(entry["out"], row)
                        entry["rows"] += 1
                        entry["stats"]["accepted"] += 1
                        entry["stats"]["per_arm"][arm] += 1
                    rec = dict(key=spec["id"], ok=True,
                               rows=len(new_rows), attempts=res["attempts"],
                               rules=[f["rule"] for f in spec["findings"]],
                               ts=ZA._now())
                elif res["kind"] in ("rejected",):
                    rec = dict(key=spec["id"], ok=False,
                               reason=res["reason"][:200],
                               gates={k: v for k, v in res["gates"].items()
                                      if v is False},
                               rules=[f["rule"] for f in spec["findings"]],
                               ts=ZA._now())
                else:                       # backend_error / aborted: retryable
                    rec = None
                if rec is not None:
                    ZA._append_line(entry["done_path"], rec)
                    entry["done"][spec["id"]] = rec
            if time.time() - last_log > 60:
                last_log = time.time()
                acc = sum(e["stats"]["accepted"] for e in outs.values())
                bs = backend.stats_summary()
                print(f"  [progress] backend={backend.name} "
                      f"done={n_done}/{len(pending)} accepted={acc} "
                      f"rejects=" +
                      "/".join(f"{k}:{v}" for e in outs.values()
                               for k, v in e["stats"]["gate_rejects"].items()
                               if v) +
                      f" backend_ok={bs['ok']} 429={bs['err_429']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                flush_stats()
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        flush_stats(final=True)
    acc = sum(e["stats"]["accepted"] for e in outs.values())
    why = ("breaker" if breaker.stop.is_set() else
           "backend-hard-fail" if stop_hard.is_set() else
           "drained" if n_done >= len(pending) else "time-budget")
    print(f"[author] FINISHED backend={backend.name} done={n_done}/"
          f"{len(pending)} accepted={acc} rows in {time.time()-t0:.0f}s; "
          f"stopped={why}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# PHASE 3: ox-alpha trust evaluation (single attempt, blind glm-5.3 judge)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading two candidate fixes of an R function. Two different tools rewrote the SAME function to fix the listed issues; grade each candidate independently, then pick the better one.

Issues the fix must address (line numbers count from the function's first line; fix EXACTLY these and nothing else):
{issues}

Original (dirty) function:
```r
{dirty}
```

Candidate A:
```r
{a}
```

Candidate B:
```r
{b}
```

Grade each candidate on:
- "correct": 2 = every listed issue correctly fixed with no behavior change; 1 = partially fixed, or a fix applied wrongly; 0 = issues unfixed or the code is broken
- "minimal": 2 = only the flagged lines touched; 1 = a couple of stray edits; 0 = broad rewrite or reformatting

Respond ONLY with JSON:
{{"A": {{"correct": <0|1|2>, "minimal": <0|1|2>}}, "B": {{"correct": <0|1|2>, "minimal": <0|1|2>}}, "better": "A" or "B" or "tie", "note": "<one short sentence>"}}"""


def single_attempt(ctx: dict, backend, site: int) -> dict:
    """One prompt -> one completion -> full gates. No feedback retry: the
    eval measures raw single-shot quality, identically for both models."""
    prompt = ZA.build_prompt(ctx)
    try:
        raw = backend.complete(prompt)
    except BackendError as e:
        return dict(error=str(e)[:200], prompt=prompt)
    fn_text, kind = ZA.extract_fn_text(raw)
    rec = dict(extract=kind, prompt=prompt)
    if fn_text is None:
        rec.update(ok=False, gates={"extract": False})
        return rec
    fn_text = "\n".join(ZA._norm_trim(fn_text))
    gates = ZA.run_gates(ctx, fn_text, site)
    beh, _beh_detail = ZA.behavior_gate(ctx, fn_text)
    gates["behavior"] = beh
    rows = ZA.build_rows(ctx, fn_text, gates["_spliced"], prompt, kind)
    gates["fix_applied"] = all(
        ZA.fix_applied(f, ctx["arm"], "\n".join(r["region_old"]),
                       "\n".join(r["region_new"]))
        for f, r in zip(ctx["findings"], rows))
    gates["row_check"] = all(RW.check_rewrite_row(r)[0] for r in rows)
    summary = ZA.gate_summary(gates, beh)
    rec.update(fn_text=fn_text,
               gates={k: v for k, v in gates.items()
                      if not k.startswith("_") and k != "loc_delta"},
               ok=not summary, fail=summary)
    return rec


def cmd_judge(args) -> int:
    mix = _parse_mix(args.mix)
    rng = random.Random(args.seed)
    skip = external_done_keys()
    by_arm: dict[str, list[dict]] = {}
    for spec in load_pool([a for a in mix if mix[a] > 0]):
        if spec["id"] in skip:
            continue
        by_arm.setdefault(spec["arm"], []).append(spec)
    chosen: list[dict] = []
    for arm, n in mix.items():
        pool_arm = by_arm.get(arm, [])
        rng.shuffle(pool_arm)
        chosen += pool_arm[:n]
    print(f"[judge] sample: " +
          ", ".join(f"{a}={len([s for s in chosen if s['arm'] == a])}"
                    for a in mix) + f" (skip_external={len(skip)})", flush=True)

    spark = AUTHOR_BACKENDS["opencode-spark"]()
    ox = OxCalphaBackend()
    judge_be = ZaiJudgeBackend()
    out_path = OUT_DIR / "oxalpha_eval.jsonl"
    site = 0
    gate_pass: dict[str, Counter] = {"oxalpha": Counter(),
                                     "opencode-spark": Counter()}
    gate_tot: dict[str, Counter] = {"oxalpha": Counter(),
                                    "opencode-spark": Counter()}
    extract: Counter = Counter()
    t0 = time.time()

    def run_one(spec: dict) -> dict:
        nonlocal site
        site += 1
        ctx = ZA.build_ctx(spec)
        rec_ox = single_attempt(ctx, ox, site)
        rec_sp = single_attempt(ctx, spark, site)
        swap = bool(rng.getrandbits(1))
        a_rec, b_rec = (rec_sp, rec_ox) if not swap else (rec_ox, rec_sp)
        jd = _judge_pair_generic(ctx, a_rec, b_rec, swap, judge_be)
        for name, rec in (("oxalpha", rec_ox), ("opencode-spark", rec_sp)):
            extract[f"{name}:{rec.get('extract', 'error')}"] += 1
            if rec.get("error"):
                gate_tot[name]["backend_error"] += 1
                continue
            for k in ZA.GATE_ORDER:
                if k == "gt_corpus_exact" and spec["arm"] == "mine":
                    continue                       # not applicable
                gate_tot[name][k] += 1
                if rec["gates"].get(k) is not False:
                    gate_pass[name][k] += 1
            gate_tot[name]["accepted_equiv"] += 1
            if rec.get("ok"):
                gate_pass[name]["accepted_equiv"] += 1
        out = dict(ts=ZA._now(), spec_id=spec["id"], arm=spec["arm"],
                   rules=[f["rule"] for f in spec["findings"]],
                   oxalpha={k: v for k, v in rec_ox.items()
                            if k not in ("prompt", "fn_text")},
                   spark={k: v for k, v in rec_sp.items()
                          if k not in ("prompt", "fn_text")},
                   judge=jd)
        ZA._append_line(out_path, out)
        return out

    ex = ThreadPoolExecutor(max_workers=args.workers)
    futs = [ex.submit(run_one, s) for s in chosen]
    n = 0
    for fut in wait(futs, timeout=None, return_when=FIRST_EXCEPTION)[0]:
        try:
            fut.result()
        except Exception as e:
            print(f"  [judge-worker-exception] {e!r}", flush=True)
            continue
        n += 1
        if n % 10 == 0:
            print(f"  [judge] {n}/{len(chosen)} elapsed="
                  f"{time.time()-t0:.0f}s ox_err={ox.stats_summary()['err_429']}"
                  f" spark_ok={spark.stats_summary()['ok']}", flush=True)
    ex.shutdown(wait=True)

    # aggregate + verdict
    def rates(name: str) -> dict:
        tot = gate_tot[name]
        return {k: (round(gate_pass[name][k] / tot[k], 3) if tot.get(k) else None)
                for k in list(ZA.GATE_ORDER) + ["accepted_equiv",
                                                "backend_error"]}

    summary = dict(
        ts=ZA._now(), n=len(chosen), elapsed_s=round(time.time() - t0, 1),
        extract=dict(extract),
        gate_rates={n_: rates(n_) for n_ in ("oxalpha", "opencode-spark")},
        gate_n={n_: dict(gate_tot[n_]) for n_ in ("oxalpha", "opencode-spark")},
        backend_stats=dict(oxalpha=ox.stats_summary(),
                           spark=spark.stats_summary(),
                           judge=judge_be.stats_summary()))

    # remap blind judge sides -> models (swap-aware), straight from the file
    jm = {"oxalpha": {"correct": [], "minimal": []},
          "opencode-spark": {"correct": [], "minimal": []}}
    better = Counter()
    judged_n = 0
    for line in out_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        jd = rec.get("judge") or {}
        sc = jd.get("scores") or {}
        if not sc or jd.get("error"):
            continue
        judged_n += 1
        sides = {"A": "opencode-spark", "B": "oxalpha"} \
            if not jd.get("swap") else {"A": "oxalpha", "B": "opencode-spark"}
        for side, model in sides.items():
            for k in ("correct", "minimal"):
                v = (sc.get(side) or {}).get(k)
                if isinstance(v, (int, float)):
                    jm[model][k].append(int(v))
        b = sc.get("better")
        if b in ("A", "B"):
            better[sides[b]] += 1
        elif b == "tie":
            better["tie"] += 1
    for model in jm:
        for k in ("correct", "minimal"):
            vv = jm[model][k]
            jm[model][k] = dict(n=len(vv),
                                mean=round(sum(vv) / len(vv), 2) if vv else None)
    summary["judge"] = dict(judged_n=judged_n, per_model=jm,
                            better=dict(better))
    acc_o = summary["gate_rates"]["oxalpha"]["accepted_equiv"]
    acc_s = summary["gate_rates"]["opencode-spark"]["accepted_equiv"]
    json_o = extract.get("oxalpha:json", 0) / max(1, len(chosen))
    cor_o = jm["oxalpha"]["correct"]["mean"]
    cor_s = jm["opencode-spark"]["correct"]["mean"]
    verdict = dict(
        accepted_equiv=dict(oxalpha=acc_o, spark=acc_s),
        json_compliance_ox=round(json_o, 3),
        judge_correct=dict(oxalpha=cor_o, spark=cor_s),
        trust=(acc_o is not None and acc_s is not None and acc_o >= 0.75 * acc_s
               and json_o >= 0.8 and cor_o is not None and cor_s is not None
               and cor_o >= cor_s - 0.5))
    summary["verdict"] = verdict
    ZA._write_json(OUT_DIR / "oxalpha_verdict.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    print(f"[judge] TRUST={verdict['trust']}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# spot-check: candidate free models vs the muse-spark anchor (blind judge)
# ---------------------------------------------------------------------------

def _judge_pair_generic(ctx: dict, a_rec: dict, b_rec: dict, swap: bool,
                        judge_be) -> dict:
    """Blind glm-5.3 pairwise scoring of two candidate fixes."""
    issues = []
    for f in ctx["findings"]:
        why = ZA.PROMPT_RATIONALE.get((ctx["arm"], f["rule"]), f["rationale"])
        cur = f["new"] if ctx["arm"] in ("inject", "buinject") else f["old"]
        line = f["row"] - ctx["top_row"] + 1
        issues.append(f"- line {line} [{f['rule']}] `{cur}`: {why}")
    txt_a = a_rec.get("fn_text") or "(no parseable function returned)"
    txt_b = b_rec.get("fn_text") or "(no parseable function returned)"
    prompt = JUDGE_PROMPT.format(
        issues="\n".join(issues),
        dirty=ctx["dirty_fn"][:2500], a=txt_a[:2500], b=txt_b[:2500])
    try:
        raw = judge_be.complete(prompt)
        obj = ZA.extract_json_object(ZA.strip_fences(raw)) or {}
        return dict(scores=obj, swap=swap,
                    note=str(obj.get("note", ""))[:200])
    except BackendError as e:
        return dict(error=str(e)[:160], swap=swap)


def _parse_mix(mix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in mix.split(","):
        arm, _, n = part.partition(":")
        out[arm.strip()] = int(n or 0)
    return out


def _choose_specs(mix: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    skip = external_done_keys()
    by_arm: dict[str, list[dict]] = {}
    for spec in load_pool([a for a in mix if mix[a] > 0]):
        if spec["id"] in skip:
            continue
        by_arm.setdefault(spec["arm"], []).append(spec)
    chosen: list[dict] = []
    for arm, n in mix.items():
        pool_arm = by_arm.get(arm, [])
        rng.shuffle(pool_arm)
        chosen += pool_arm[:n]
    return chosen


def _judge_file_aggregate(out_path: Path, cand_names: list[str]) -> dict:
    """Swap-aware aggregation of judge records (one per spec x candidate)."""
    jm = {c: {"correct": [], "minimal": []} for c in cand_names}
    jm["anchor"] = {"correct": [], "minimal": []}
    better = Counter()
    judged = 0
    for line in out_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        jd = rec.get("judge") or {}
        sc = jd.get("scores") or {}
        if not sc or jd.get("error"):
            continue
        judged += 1
        # side A = candidate, side B = anchor, unless swapped
        if jd.get("swap"):
            cand_side, anchor_side = "B", "A"
        else:
            cand_side, anchor_side = "A", "B"
        for side, bucket in ((cand_side, jm[rec["candidate"]]),
                             (anchor_side, jm["anchor"])):
            for k in ("correct", "minimal"):
                v = (sc.get(side) or {}).get(k)
                if isinstance(v, (int, float)):
                    bucket[k].append(int(v))
        b = sc.get("better")
        if b in ("A", "B"):
            winner = rec["candidate"] if b == cand_side else "anchor"
            better[winner] += 1
        elif b == "tie":
            better["tie"] += 1
    for bucket in list(jm.values()) + []:
        for k in ("correct", "minimal"):
            vv = bucket[k]
            bucket[k] = dict(n=len(vv),
                             mean=round(sum(vv) / len(vv), 2) if vv else None)
    return dict(judged_n=judged, per_model=jm, better=dict(better))


def cmd_spotcheck(args) -> int:
    cands = [c.strip() for c in args.models.split(",") if c.strip()]
    mix = _parse_mix(args.mix)
    chosen = _choose_specs(mix, args.seed)
    print(f"[spotcheck] models={cands} sample=" +
          ", ".join(f"{a}:{sum(1 for s in chosen if s['arm'] == a)}"
                    for a in mix) + f" workers={args.workers}", flush=True)

    spark = AUTHOR_BACKENDS["opencode-spark"]()
    judge_be = ZaiJudgeBackend()
    cand_backends = [resolve_backend(c if (":" in c and not c.endswith(":free"))
                                     or c in AUTHOR_BACKENDS
                                     else f"openrouter-free:{c}")
                     for c in cands]
    tag = args.tag or "-".join(c.split("/")[-1].split(":")[0] for c in cands)
    out_path = OUT_DIR / f"spotcheck_{tag}.jsonl"
    if args.resume and out_path.exists():
        done_specs = set()
        for line in out_path.read_text().splitlines():
            try:
                done_specs.add(json.loads(line)["spec_id"])
            except (ValueError, KeyError):
                pass
        print(f"[spotcheck] resume: {len(done_specs)} specs already done",
              flush=True)
        chosen = [s for s in chosen if s["id"] not in done_specs]
    site = 0
    lock = threading.Lock()
    rng = random.Random(args.seed + 1)

    def run_one(spec: dict) -> None:
        nonlocal site
        with lock:
            site += 1
            my_site = site
        ctx = ZA.build_ctx(spec)
        anchor = single_attempt(ctx, spark, my_site)   # once per spec
        for cand, cand_be in zip(cands, cand_backends):
            rec_c = single_attempt(ctx, cand_be, my_site)
            swap = bool(rng.getrandbits(1))
            a_rec, b_rec = ((rec_c, anchor) if not swap
                            else (anchor, rec_c))
            jd = _judge_pair_generic(ctx, a_rec, b_rec, swap, judge_be)
            ZA._append_line(out_path, dict(
                ts=ZA._now(), spec_id=spec["id"], arm=spec["arm"],
                candidate=cand, rules=[f["rule"] for f in spec["findings"]],
                cand={k: v for k, v in rec_c.items()
                      if k not in ("prompt", "fn_text")},
                anchor={k: v for k, v in anchor.items()
                        if k not in ("prompt", "fn_text")},
                judge=jd))

    ex = ThreadPoolExecutor(max_workers=args.workers)
    futs = [ex.submit(run_one, s) for s in chosen]
    n = 0
    for fut in wait(futs, timeout=None, return_when=FIRST_EXCEPTION)[0]:
        try:
            fut.result()
        except Exception as e:
            print(f"  [spotcheck-worker-exception] {e!r}", flush=True)
            continue
        n += 1
        if n % 10 == 0:
            print(f"  [spotcheck] {n}/{len(chosen)} specs done", flush=True)
    ex.shutdown(wait=True)

    # aggregate per candidate
    verdicts = {}
    for cand in cands:
        gate_pass, gate_tot, extract = Counter(), Counter(), Counter()
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("candidate") != cand:
                continue
            r = rec["cand"]
            extract[r.get("extract", "error")] += 1
            if r.get("error"):
                gate_tot["backend_error"] += 1
                continue
            for k in ZA.GATE_ORDER:
                if k == "gt_corpus_exact" and rec["arm"] == "mine":
                    continue
                gate_tot[k] += 1
                if r["gates"].get(k) is not False:
                    gate_pass[k] += 1
            gate_tot["accepted_equiv"] += 1
            if r.get("ok"):
                gate_pass["accepted_equiv"] += 1
        n_c = sum(extract.values())
        json_rate = (extract.get("json", 0) / n_c) if n_c else 0.0
        acc = (gate_pass["accepted_equiv"] / gate_tot["accepted_equiv"]
               if gate_tot.get("accepted_equiv") else None)
        verdicts[cand] = dict(n=n_c, extract=dict(extract),
                              json_rate=round(json_rate, 3),
                              gate_pass=dict(gate_pass), gate_n=dict(gate_tot),
                              accepted_equiv=round(acc, 3) if acc else acc,
                              trust=bool(acc and acc >= 0.5 and
                                         json_rate >= 0.7))
    agg = _judge_file_aggregate(out_path, cands)
    summary = dict(ts=ZA._now(), models=cands, n_specs=len(chosen),
                   gate_verdicts=verdicts, judge=agg,
                   backend_stats=dict(spark=spark.stats_summary(),
                                      judge=judge_be.stats_summary()))
    ZA._write_json(OUT_DIR / f"spotcheck_{tag}_verdict.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    for c, v in verdicts.items():
        print(f"[spotcheck] {c}: accepted_equiv={v['accepted_equiv']} "
              f"json={v['json_rate']} trust={v['trust']}", flush=True)
    return 0

# ---------------------------------------------------------------------------
# effort A/B: does higher reasoning effort beat low on accepted-rows-per-
# 1000-output-tokens? Same seed-matched spec set per arm, production
# 2-attempt feedback protocol, usage-field token accounting.
# ---------------------------------------------------------------------------

class SparkEffortBackend(SparkAuthorBackend):
    """Instrumented muse-spark author at a given reasoning effort: records
    per-call output/reasoning tokens (usage field) and latency."""

    def __init__(self, effort: str):
        self.effort = effort
        super().__init__()
        self.calls: list[dict] = []
        self._ilock = threading.Lock()

    def _payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt}]}],
            "max_output_tokens": 6000,
            "reasoning": {"effort": self.effort},
            "text": {"format": {"type": "json_object"}},
        }

    def _extract(self, payload: dict) -> str:
        u = payload.get("usage") or {}
        with self._ilock:
            self.calls.append(dict(
                dt=None,
                out_tok=u.get("output_tokens") or 0,
                reason_tok=(u.get("output_tokens_details") or {})
                .get("reasoning_tokens") or 0))
        return super()._extract(payload)

    def _complete_once(self, prompt: str) -> str:
        t0 = time.time()
        res = super()._complete_once(prompt)
        with self._ilock:
            for c in reversed(self.calls):
                if c["dt"] is None:
                    c["dt"] = round(time.time() - t0, 2)
                    break
        return res


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    import math
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def cmd_effort_ab(args) -> int:
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    specs = _choose_specs(_parse_mix(args.mix), args.seed)
    print(f"[effort_ab] efforts={efforts} specs={len(specs)} "
          f"(mine={sum(1 for s in specs if s['arm'] == 'mine')}, "
          f"inject={sum(1 for s in specs if s['arm'] == 'inject')}) "
          f"[sequential, 1 in flight]", flush=True)
    out_rows: list[dict] = []
    per_arm: dict[str, dict] = {}

    def run_spec(ctx, be, site) -> dict:
        """Production protocol: up to 2 attempts with verifier feedback."""
        feedback = None
        for attempt in (1, 2):
            prompt = ZA.build_prompt(ctx, feedback)
            raw = be.complete(prompt)
            fn_text, kind = ZA.extract_fn_text(raw)
            if fn_text is None:
                feedback = ("the response was not parseable "
                            "(return ONLY the JSON object).")
                continue
            fn_text = "\n".join(ZA._norm_trim(fn_text))
            gates = ZA.run_gates(ctx, fn_text, site)
            beh, _d = ZA.behavior_gate(ctx, fn_text)
            gates["behavior"] = beh
            rows = ZA.build_rows(ctx, fn_text, gates["_spliced"], prompt,
                                 kind)
            gates["fix_applied"] = all(
                ZA.fix_applied(f, ctx["arm"], "\n".join(r["region_old"]),
                               "\n".join(r["region_new"]))
                for f, r in zip(ctx["findings"], rows))
            gates["row_check"] = all(RW.check_rewrite_row(r)[0]
                                     for r in rows)
            summary = ZA.gate_summary(gates, beh)
            if not summary:
                return dict(accepted=True, attempts=attempt, rows=len(rows),
                            fail="", fn_text=fn_text, kind=kind)
            feedback = summary
            if attempt == 2:
                return dict(accepted=False, attempts=2, rows=0, fail=summary,
                            fn_text=fn_text, kind=kind)
        return dict(accepted=False, attempts=2, rows=0, fail="unparseable",
                    fn_text="", kind="none")

    site = 0
    bes: dict[str, SparkEffortBackend] = {}
    for effort in efforts:
        be = SparkEffortBackend(effort)
        bes[effort] = be
        breaker = ZA.Breaker()
        arm_stats = dict(effort=effort, attempted=0, accepted=0, rows=0,
                         attempts_sum=0, out_tok=0, reason_tok=0,
                         latencies=[], fails=Counter(), samples=[])
        per_arm[effort] = arm_stats
        t0 = time.time()
        for spec in specs:
            if not breaker.wait_turn():
                print(f"  [effort_ab] {effort}: breaker stop at "
                      f"{arm_stats['attempted']} specs", flush=True)
                break
            site += 1
            ctx = ZA.build_ctx(spec)
            i0 = len(be.calls)
            try:
                res = run_spec(ctx, be, site)
                breaker.report(ok=True, rate_error=False)
            except BackendError as e:
                breaker.report(ok=False, rate_error=(e.kind == "rate"))
                arm_stats["fails"][f"backend:{e.kind}"] += 1
                continue
            spec_calls = be.calls[i0:]
            out_tok = sum(c["out_tok"] for c in spec_calls)
            reason_tok = sum(c["reason_tok"] for c in spec_calls)
            arm_stats["attempted"] += 1
            arm_stats["attempts_sum"] += res["attempts"]
            arm_stats["out_tok"] += out_tok
            arm_stats["reason_tok"] += reason_tok
            arm_stats["latencies"] += [c["dt"] for c in spec_calls
                                       if c["dt"] is not None]
            if res["accepted"]:
                arm_stats["accepted"] += 1
                arm_stats["rows"] += res["rows"]
            else:
                arm_stats["fails"][res["fail"].split(";")[0]] += 1
            if len(arm_stats["samples"]) < 3:
                arm_stats["samples"].append(dict(
                    spec_id=spec["id"], arm=spec["arm"],
                    attempts=res["attempts"], accepted=res["accepted"],
                    fail=res["fail"][:120], out_tok=out_tok,
                    reason_tok=reason_tok,
                    fn_head=(res["fn_text"] or "")[:200]))
            out_rows.append(dict(
                effort=effort, spec_id=spec["id"], arm=spec["arm"],
                accepted=res["accepted"], attempts=res["attempts"],
                rows=res["rows"], out_tok=out_tok, reason_tok=reason_tok,
                fail=res["fail"][:160]))
            if arm_stats["attempted"] % 10 == 0:
                print(f"  [effort_ab] {effort}: {arm_stats['attempted']}/"
                      f"{len(specs)} acc={arm_stats['accepted']} "
                      f"out_tok={arm_stats['out_tok']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        lats = sorted(arm_stats["latencies"])
        n_att = arm_stats["attempted"]
        acc_rate = arm_stats["accepted"] / n_att if n_att else 0.0
        lo, hi = _wilson(acc_rate, n_att)
        rows_per_1k = (arm_stats["rows"] / (arm_stats["out_tok"] / 1000.0)
                       if arm_stats["out_tok"] else None)
        arm_stats["summary"] = dict(
            n=n_att, acceptance=round(acc_rate, 3),
            ci95=[round(lo, 3), round(hi, 3)],
            mean_attempts=round(arm_stats["attempts_sum"] / n_att, 2)
            if n_att else None,
            out_tok_total=arm_stats["out_tok"],
            reason_tok_total=arm_stats["reason_tok"],
            rows=arm_stats["rows"],
            rows_per_1k_out_tok=round(rows_per_1k, 3)
            if rows_per_1k is not None else None,
            p50_latency_s=lats[len(lats) // 2] if lats else None,
            p95_latency_s=lats[int(len(lats) * 0.95)] if lats else None,
            fails=dict(arm_stats["fails"].most_common(6)))
        print(f"[effort_ab] {effort}: {json.dumps(arm_stats['summary'])}",
              flush=True)

    # verdict: max rows_per_1k_out_tok; overlapping CI with low -> stay low
    lows = per_arm.get("low", {}).get("summary", {})
    def rpk(e):
        return per_arm[e]["summary"].get("rows_per_1k_out_tok") or 0.0
    best = max(per_arm, key=rpk)
    ci_best = per_arm[best]["summary"]["ci95"]
    ci_low = lows.get("ci95", [0.0, 1.0])
    overlap = not (ci_best[1] < ci_low[0] or ci_low[1] < ci_best[0])
    verdict = dict(best_on_rows_per_1k=best, ci_overlap_with_low=overlap,
                   recommendation=(("low" if overlap else best)
                                   if lows else best),
                   note="overlapping acceptance CIs with low at this n -> "
                        "stay low (cheaper calls, same economics)"
                        if overlap else
                        f"{best} wins on accepted-rows-per-1000-tokens")
    report = dict(ts=ZA._now(), efforts=efforts, n_specs=len(specs),
                  per_arm={e: per_arm[e]["summary"] for e in efforts},
                  samples={e: per_arm[e]["samples"] for e in efforts},
                  verdict=verdict,
                  backend_stats={e: bes[e].stats_summary() for e in efforts})
    ZA._write_json(OUT_DIR / "effort_ab_verdict.json", report)
    with open(OUT_DIR / "effort_ab_rows.jsonl", "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(verdict, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def cmd_stats(_args) -> int:
    for out in (LINTFIX_SPARK, FIXISSUE_SPARK, LINTFIX_OX, LINTFIX_ZAI,
                FIXISSUE_ZAI, LINTFIX_ORFREE, FIXISSUE_ORFREE):
        sp = Path(str(out) + ".stats.json")
        print(f"== {out.name}")
        if sp.exists():
            print(sp.read_text())
        else:
            print("  (no stats yet)")
    return 0


def main(argv=None) -> int:
    if argv is None:
        # supervisor scripts pass `${pool:+--pool "$pool"}` which zsh fuses
        # into ONE token ("--pool <path>"); split such fused flag tokens
        argv = []
        for a in sys.argv[1:]:
            if a.startswith("--") and " " in a:
                argv += a.split(" ", 1)
            else:
                argv.append(a)
    ap = argparse.ArgumentParser(prog="python3 rewrite_author_spark.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mine")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--tidy-packages", type=int, default=250)
    p.add_argument("--random-packages", type=int, default=2200)
    p.add_argument("--mine-specs", type=int, default=2200)
    p.add_argument("--inject-specs", type=int, default=2800)
    p.add_argument("--buinject-specs", type=int, default=3400)
    p.add_argument("--bug-per-fn", type=int, default=2)
    p.add_argument("--time-budget", type=float, default=2400)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out-pool", default="",
                   help="alternate pool jsonl to append specs to")
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("author")
    p.add_argument("--backend", default="opencode-spark",
                   help="registered name or openrouter-free:<model-id> "
                        "(resolve_backend validates)")
    p.add_argument("--arms", default="mine,inject")
    p.add_argument("--pool", default="",
                   help="alternate spec-pool jsonl (default: own pool)")
    p.add_argument("--max", type=int, default=0, help="0 = all pending")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--time-budget", type=float, default=21600)
    p.add_argument("--breaker-rounds", type=int, default=4)
    p.add_argument("--skip-zai-done", type=int, default=1)
    p.set_defaults(fn=cmd_author)

    p = sub.add_parser("judge")
    p.add_argument("--n", type=int, default=40, help="unused; --mix governs")
    p.add_argument("--mix", default="mine:30,inject:10")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=99)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("effort_ab")
    p.add_argument("--efforts", default="low,medium,high")
    p.add_argument("--mix", default="mine:35,inject:15")
    p.add_argument("--seed", type=int, default=31337)
    p.set_defaults(fn=cmd_effort_ab)

    p = sub.add_parser("spotcheck")
    p.add_argument("--models", required=True,
                   help="comma list; bare ids are prefixed openrouter-free:")
    p.add_argument("--mix", default="mine:8,inject:4")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--tag", default="")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_spotcheck)

    p = sub.add_parser("stats")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
