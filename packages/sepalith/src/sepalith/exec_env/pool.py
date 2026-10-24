"""Function pool for execution-grounded tasks: real R functions from TRAIN-split packages.

Source: the CPT corpus actually trained (``CPT-prefix-extension-v1``,
sha256-pinned), which holds complete package ``R/`` files as token IDs with
per-document package and identity-group provenance. Documents are admitted by
``guard.SplitGuard`` (TRAIN identity group and TRAIN-only package name); every
other document is skipped before it is decoded. Every CPT-derived function was
therefore seen in CPT (``cpt_exposed``); overlap with the 15,006 SFT TRAIN rows
and the DEV panels is measured per function (token 12-gram containment, the
synth fingerprints).

    PYTHONPATH=packages/sepalith/src nice -n 10 .venv-sft/bin/python -m sepalith.exec_env.pool extract --scratch S
    ... pool analyze --scratch S          # r_pool.R over every decoded file (8 processes)
    ... pool select --scratch S --out OUT # selection, helpers, air, dedup, exposure -> OUT/pool.jsonl

Selection (all must hold):

- a top-level ``name <- function(...) { ... }`` with a syntactic name, defined
  once in its file, 8-80 non-blank body lines and at least 3 body statements;
- every global it uses resolves to the default search path (base, stats,
  utils, methods, datasets; graphics only through ``GRDEVICES_PURE``), to a
  ``pkg::fn`` export of a base or recommended package, or to another top-level
  function of the same file (an in-file helper, included transitively, which
  must satisfy the same rules);
- deterministic and side-effect free: no call in ``FORBIDDEN_CALLS`` or with a
  forbidden prefix (RNG, time, environment, system, file or network I/O,
  dynamic lookup/eval, native code, package loading, graphics devices,
  parallelism) anywhere in the function or its helpers;
- ASCII, air formats the unit file (helpers, roxygen, function) idempotently,
  and it is not an exact or near duplicate (Jaccard >= 0.8 on 5-line shingles)
  of an earlier pool function.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from sepalith.exec_env.guard import CPT_ROWS, CPT_ROWS_SHA256, SplitGuard, guard_path, sha256_file

HERE = Path(__file__).resolve().parent
R_POOL = HERE / "r_pool.R"
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")
SCHEMA = "sepalith.exec-env.pool.v1"
BODY_MIN, BODY_MAX, STMTS_MIN, UNIT_MAX_LINES = 8, 80, 3, 260

FORBIDDEN_CALLS = frozenset("""
sample sample.int runif rnorm rbinom rpois rexp rgamma rbeta rt rchisq rmultinom rgeom rhyper rlogis rweibull
rcauchy rlnorm rnbinom rsignrank rwilcox rf set.seed RNGkind jitter r2dtable simulate
date proc.time system.time timestamp
system system2 shell shell.exec setwd getwd R.home .libPaths installed.packages find.package system.file
packageVersion packageDescription interactive readline menu browser debug debugonce undebug trace untrace
quit q commandArgs readRenviron
file url gzfile bzfile xzfile unz pipe fifo socketConnection open close readLines writeLines readRDS saveRDS
load save save.image source sys.source sink read.table read.csv read.csv2 read.delim read.delim2 read.fwf
write.table write.csv write.csv2 write scan unlink tempfile tempdir normalizePath download.file browseURL
readBin writeBin readChar writeChar dget dump Rprof data list.files list.dirs dir
library require requireNamespace loadNamespace attachNamespace attach detach getNamespace asNamespace
get get0 mget exists assign eval evalq eval.parent parse sys.function sys.frames sys.frame parent.env
environment<- body<- formals<- lockBinding unlockBinding makeActiveBinding reg.finalizer
.Call .External .External2 .C .Fortran .Internal .Primitive dyn.load dyn.unload library.dynam
setTimeLimit setSessionTimeLimit gc.time Sys.setenv Sys.setlocale
mclapply mcmapply mcparallel makeCluster parLapply parSapply clusterApply
dev.new dev.off pdf png jpeg svg x11 X11 windows quartz
""".split())
FORBIDDEN_PREFIXES = ("Sys.", "file.", "dir.", "dev.", "socket", "url", "rand")
FORBIDDEN_NAMESPACES = frozenset({"parallel", "tcltk", "graphics", "grDevices", "grid", "lattice", "compiler"})
GRDEVICES_PURE = frozenset("""rgb col2rgb hsv hcl gray grey colorRampPalette colorRamp adjustcolor rgb2hsv
convertColor extendrange xy.coords xyz.coords n2mfrow chull contourLines boxplot.stats nclass.Sturges
nclass.scott nclass.FD colors colours rainbow heat.colors terrain.colors topo.colors cm.colors gray.colors
grey.colors hcl.colors palette.colors""".split())
ALLOWED_SEARCH = frozenset({"pkg:base", "pkg:stats", "pkg:utils", "pkg:methods", "pkg:datasets"})


# --------------------------------------------------------------------------- extract
_STATE: dict = {}


def _init() -> None:
    from tokenizers import Tokenizer
    os.nice(max(0, 10 - os.nice(0)))
    _STATE.update(tok=Tokenizer.from_file(str(TOKENIZER)), guard=SplitGuard(verify_sha=False))


def _scan(job: tuple[int, int, str]) -> tuple[Counter, list[dict]]:
    start, end, scratch = job
    tok, guard = _STATE["tok"], _STATE["guard"]
    counts: Counter = Counter()
    docs: list[dict] = []
    with CPT_ROWS.open("rb") as stream:
        stream.seek(start)
        if start:
            stream.readline()
        while stream.tell() <= end:
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            counts["rows"] += 1
            verdict = guard.verdict(str(row["group_id"]), str(row["package"]))
            if verdict != "train":
                counts["skip_" + verdict] += 1
                continue
            if str(row["chunk_index"]) != "0" or str(row["is_document_end"]) not in ("True", "true", "1"):
                counts["skip_multi_row_document"] += 1
                continue
            source = str(row["source_path"])
            if "/R/" not in source or not source.lower().endswith(".r"):
                counts["skip_not_package_R_file"] += 1
                continue
            ids = [int(t) for t in row["input_ids"]]
            text = tok.decode([t for t in ids if t not in (0, 1)], skip_special_tokens=True)
            sha_ok = hashlib.sha256(text.encode("utf-8")).hexdigest() == row["source_sha256"]
            counts["sha_match" if sha_ok else "sha_mismatch"] += 1
            package = str(row["package"])
            relative = "R/" + source.split("/R/", 1)[1]
            key = hashlib.sha256(f"{package}\0{relative}\0{row['document_id']}".encode()).hexdigest()[:10]
            path = Path(scratch) / "sources" / package / f"{key}-{Path(relative).name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            counts["files"] += 1
            docs.append({"path": str(path), "package": package, "relative": relative, "group_id": row["group_id"],
                         "document_id": row["document_id"], "source_sha256": row["source_sha256"], "sha_match": sha_ok})
    return counts, docs


def command_extract(args: argparse.Namespace) -> int:
    path = Path(guard_path(CPT_ROWS))
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    size = path.stat().st_size
    spans = [(size * i // 96, size * (i + 1) // 96, str(scratch)) for i in range(96)]
    total: Counter = Counter()
    docs: list[dict] = []
    with ThreadPoolExecutor(max_workers=1) as hasher:
        corpus_sha = hasher.submit(sha256_file, path) if not args.skip_sha else None
        with ProcessPoolExecutor(max_workers=min(args.workers, 8), initializer=_init) as pool:
            for counts, found in pool.map(_scan, spans):
                total.update(counts)
                docs += found
        if corpus_sha is not None and corpus_sha.result() != CPT_ROWS_SHA256:
            raise SystemExit("CPT corpus differs from its pinned sha256")
    SplitGuard()  # verifies the split file's pinned sha256
    with (scratch / "documents.jsonl").open("w") as stream:
        for doc in sorted(docs, key=lambda d: (d["package"], d["relative"], d["document_id"])):
            stream.write(json.dumps(doc) + "\n")
    summary = {"rows": total["rows"], "counts": dict(total), "packages": len({d["package"] for d in docs}),
               "cpt_rows_sha256_verified": not args.skip_sha}
    (scratch / "extract-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


# --------------------------------------------------------------------------- analyze
def _analyze(job: tuple[int, list[str], str]) -> str:
    index, paths, scratch = job
    out = Path(scratch) / "analysis" / f"part-{index:04d}.jsonl"
    if out.exists():
        return str(out)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        listing.write("\n".join(paths) + "\n")
    tmp = out.with_suffix(".tmp")
    try:
        subprocess.run(["nice", "-n", "10", "Rscript", "--vanilla", str(R_POOL), listing.name, str(tmp)], check=True,
                       capture_output=True, timeout=3600)
        tmp.rename(out)
    finally:
        os.unlink(listing.name)
    return str(out)


def command_analyze(args: argparse.Namespace) -> int:
    scratch = Path(args.scratch)
    (scratch / "analysis").mkdir(exist_ok=True)
    docs = [json.loads(line) for line in (scratch / "documents.jsonl").open()]
    paths = [d["path"] for d in docs if "function" in Path(d["path"]).read_text(encoding="utf-8", errors="replace")]
    chunks = [(i // 400, paths[i:i + 400], str(scratch)) for i in range(0, len(paths), 400)]
    with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        for n, _ in enumerate(pool.map(_analyze, chunks), 1):
            if n % 20 == 0:
                print(f"analyzed {n}/{len(chunks)} chunks", file=sys.stderr, flush=True)
    print(json.dumps({"files": len(paths), "chunks": len(chunks)}))
    return 0


# --------------------------------------------------------------------------- select
def forbidden_reason(record: dict) -> str | None:
    for call in record.get("calls", []):
        if call in FORBIDDEN_CALLS or call.startswith(FORBIDDEN_PREFIXES):
            return f"forbidden_call:{call}"
    for ref in record.get("ns", []):
        if ref.endswith("!") or ":::" in ref:
            return "namespace_unavailable"
        pkg, fn = ref.split("::", 1)
        if pkg in FORBIDDEN_NAMESPACES and not (pkg == "grDevices" and fn in GRDEVICES_PURE):
            return f"forbidden_namespace:{pkg}"
        if fn in FORBIDDEN_CALLS or fn.startswith(FORBIDDEN_PREFIXES):
            return f"forbidden_call:{fn}"
    for kind in ("fun_globals", "var_globals"):
        for name, where in (record.get(kind) or {}).items():
            if where in ("file_fun",) or where in ALLOWED_SEARCH:
                continue
            if where == "pkg:grDevices" and name in GRDEVICES_PURE:
                continue
            if name in ("::", ":::"):
                continue
            if where == "file_var":
                return "uses_file_level_variable"
            if where == "unresolved":
                return "unresolved_global"
            return f"global_from_{where}"
    return None


def resolve_helpers(record: dict, by_name: dict[str, dict]) -> tuple[list[str] | None, str | None]:
    """Transitive in-file helpers of a function (file order), or a rejection reason."""
    seen: list[str] = []
    stack = [record]
    while stack:
        current = stack.pop()
        names = [n for kind in ("fun_globals", "var_globals")
                 for n, where in (current.get(kind) or {}).items() if where == "file_fun"]
        for name in names:
            if name == record["name"] or name in seen:
                continue
            helper = by_name.get(name)
            if helper is None or "error" in helper:
                return None, "helper_unavailable"
            reason = forbidden_reason(helper)
            if reason:
                return None, "helper_" + reason
            seen.append(name)
            stack.append(helper)
    return sorted(seen, key=lambda n: by_name[n]["start"]), None


def roxygen_start(lines: list[str], start: int) -> int:
    """1-based first line of the contiguous #' block directly above line ``start``."""
    line = start - 1
    while line >= 1 and lines[line - 1].lstrip().startswith("#'"):
        line -= 1
    return line + 1


def code_key(text: str) -> str:
    from sepalith.data.synth.common import code_tokens
    return hashlib.sha256(" ".join(code_tokens(text.split("\n"))).encode()).hexdigest()


def shape(text: str, record: dict) -> dict:
    from sepalith.data.synth.common import strip_comment
    code = "\n".join(strip_comment(line) for line in text.split("\n"))
    assigned = re.findall(r"^\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=)(?!=)", code, flags=re.M)
    formals = set(record.get("formals") or [])
    rename = sorted({n for n in assigned if n not in formals
                     and len(re.findall(r"(?<![A-Za-z0-9._$@])" + re.escape(n) + r"(?![A-Za-z0-9._(])", code)) >= 3})
    na_calls = re.findall(r"\b(sum|mean|max|min|median|sd|var|range|prod|quantile)\(", code)
    nested = len(re.findall(r"[A-Za-z_.][\w.]*\(\s*[A-Za-z_.][\w.]*\(\s*[A-Za-z_.][\w.]*\(", code))
    body = record["nonblank"]
    return {"body_bucket": "8-15" if body <= 15 else "16-30" if body <= 30 else "31-50" if body <= 50 else "51-80",
            "loops": record["loops"], "ifs": record["ifs"], "returns": record["returns"],
            "nested_funs": record["nested_funs"], "stmts": record["stmts"], "strings": record["strings"],
            "rename_candidates": len(rename), "na_rm_calls": len(na_calls),
            "na_rm_present": "na.rm" in code, "pipe_native": code.count("|>"), "nested_call_chains": nested}


def command_select(args: argparse.Namespace) -> int:
    from sepalith.data.synth import dedup
    from sepalith.data.synth.normalize import format_text
    from sepalith.training.rl.air_canonical import AirFormatter

    scratch, out = Path(args.scratch), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = {d["path"]: d for d in map(json.loads, (scratch / "documents.jsonl").open())}
    by_file: dict[str, list[dict]] = defaultdict(list)
    for part in sorted((scratch / "analysis").glob("part-*.jsonl")):
        for line in part.open(encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_file[record["file"]].append(record)
    reasons: Counter = Counter()
    candidates: list[dict] = []
    for path in sorted(by_file, key=lambda p: (docs[p]["package"], docs[p]["relative"])):
        records = by_file[path]
        if any("file_error" in r for r in records):
            reasons["file_does_not_parse"] += 1
            continue
        names = Counter(r["name"] for r in records)
        by_name = {r["name"]: r for r in records if names[r["name"]] == 1}
        lines = Path(path).read_text(encoding="utf-8").split("\n")
        for record in records:
            reasons["functions_seen"] += 1
            if "error" in record:
                reasons["analysis_error"] += 1
                continue
            if names[record["name"]] != 1 or not NAME_RE.match(record["name"]):
                reasons["name_redefined_or_nonsyntactic"] += 1
                continue
            if not record["braced"] or not BODY_MIN <= record["nonblank"] <= BODY_MAX:
                reasons["body_length_outside_8_80"] += 1
                continue
            if record["stmts"] < STMTS_MIN:
                reasons["fewer_than_3_statements"] += 1
                continue
            reason = forbidden_reason(record)
            if reason:
                reasons[reason.split(":")[0]] += 1
                continue
            helpers, reason = resolve_helpers(record, by_name)
            if helpers is None:
                reasons[str(reason).split(":")[0]] += 1
                continue
            blocks = []
            for name in helpers:
                h = by_name[name]
                blocks.append("\n".join(lines[h["start"] - 1:h["end"]]))
            first = roxygen_start(lines, record["start"])
            target = "\n".join(lines[first - 1:record["end"]])
            unit = "\n\n".join([*blocks, target]) + "\n"
            if sum(1 for _ in unit.split("\n")) > UNIT_MAX_LINES:
                reasons["unit_too_long"] += 1
                continue
            if any(ord(ch) > 127 for ch in unit) or "\t" in unit or "\r" in unit:
                reasons["non_ascii_or_tab"] += 1
                continue
            doc = docs[path]
            candidates.append({"package": doc["package"], "relative": doc["relative"], "group_id": doc["group_id"],
                               "document_id": doc["document_id"], "name": record["name"], "helpers": helpers,
                               "formals": record.get("formals") or [], "raw_unit": unit,
                               "function_text": "\n".join(lines[record["start"] - 1:record["end"]]),
                               "record": record})
    reasons["candidates_before_format"] = len(candidates)
    formatter = AirFormatter(("air",))

    def fmt(item: dict) -> str | None:
        formatted = format_text(formatter, item["raw_unit"])
        if formatted is None or format_text(formatter, formatted) != formatted:
            return None
        return formatted

    with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        formatted = list(pool.map(fmt, candidates))
    # exact and near duplicates (target function code, formatting-invariant), first by package order wins
    kept: list[dict] = []
    keys_seen: set[str] = set()
    for item, text in zip(candidates, formatted, strict=True):
        if text is None:
            reasons["air_failed_or_not_idempotent"] += 1
            continue
        key = code_key(item["function_text"])
        if key in keys_seen:
            reasons["exact_duplicate"] += 1
            continue
        keys_seen.add(key)
        item["unit"] = text
        kept.append(item)
    near = dedup.near_duplicates([dedup.shingles(i["function_text"]) for i in kept], 0.8)
    pool_items = []
    for item, match in zip(kept, near, strict=True):
        if match is not None:
            reasons["near_duplicate"] += 1
            continue
        pool_items.append(item)
    exposure = exposure_checks(pool_items) if not args.skip_exposure else {}
    rows = []
    for item in pool_items:
        record = item["record"]
        fid = hashlib.sha256(f"{item['package']}\0{item['relative']}\0{item['name']}".encode()).hexdigest()[:16]
        row = {"schema": SCHEMA, "id": fid, "package": item["package"], "path": item["relative"],
               "group_id": item["group_id"], "document_id": item["document_id"], "name": item["name"],
               "helpers": item["helpers"], "formals": item["formals"], "body_nonblank": record["nonblank"],
               "body_code": record["code"], "unit": item["unit"], "unit_sha256": hashlib.sha256(item["unit"].encode()).hexdigest(),
               "shape": shape(item["function_text"], record), "exposure": {"cpt": True, **exposure.get(id(item), {})},
               "split": "train"}
        if row["exposure"].get("dev_containment", 0.0) > 0.3:
            reasons["dev_overlap_over_0.3"] += 1
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r["package"], r["path"], r["name"]))
    with (out / "pool.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    summary = summarize(rows)
    summary["selection"] = dict(reasons)
    summary["inputs"] = {"cpt_rows": str(CPT_ROWS), "cpt_rows_sha256": CPT_ROWS_SHA256,
                         "extract": json.loads((scratch / "extract-summary.json").read_text())}
    (out / "pool-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_package"}, indent=1))
    return 0


def exposure_checks(items: list[dict]) -> dict[int, dict]:
    """SFT TRAIN-row and DEV containment of each function (12-gram token fingerprints)."""
    from sepalith.data.synth import build, dedup

    _lines4, train_exact, _masked = build.train_indexes()
    _dlines4, dev_exact, _dmasked, _info = build.dev_indexes()
    out = {}
    for item in items:
        hashes = dedup.kgram_hashes(dedup.token_ids(dedup.text_tokens(item["function_text"].split("\n"))))
        sft, sft_run = dedup.overlap(train_exact.hits(hashes))
        dev, dev_run = dedup.overlap(dev_exact.hits(hashes))
        out[id(item)] = {"sft_train_containment": round(sft, 3), "sft_train_longest_run": sft_run,
                         "dev_containment": round(dev, 3), "dev_longest_run": dev_run}
    return out


def summarize(rows: list[dict]) -> dict:
    by_package = Counter(r["package"] for r in rows)
    buckets = Counter(r["shape"]["body_bucket"] for r in rows)
    sft = [r["exposure"].get("sft_train_containment", 0.0) for r in rows]
    return {"functions": len(rows), "packages": len(by_package),
            "functions_per_package": {"max": max(by_package.values(), default=0),
                                      "median": sorted(by_package.values())[len(by_package) // 2] if by_package else 0},
            "body_buckets": dict(sorted(buckets.items())),
            "with_helpers": sum(1 for r in rows if r["helpers"]),
            "with_loops": sum(1 for r in rows if r["shape"]["loops"]),
            "with_if": sum(1 for r in rows if r["shape"]["ifs"]),
            "rename_candidates_any": sum(1 for r in rows if r["shape"]["rename_candidates"]),
            "na_rm_calls_any": sum(1 for r in rows if r["shape"]["na_rm_calls"]),
            "nested_call_chains_any": sum(1 for r in rows if r["shape"]["nested_call_chains"]),
            "sft_train_containment_over_0.5": sum(1 for v in sft if v > 0.5),
            "by_package": dict(sorted(by_package.items()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "analyze", "select"):
        p = sub.add_parser(name)
        p.add_argument("--scratch", required=True)
        p.add_argument("--workers", type=int, default=8)
        if name == "extract":
            p.add_argument("--skip-sha", action="store_true")
        if name == "select":
            p.add_argument("--out", required=True)
            p.add_argument("--skip-exposure", action="store_true")
    args = parser.parse_args(argv)
    return {"extract": command_extract, "analyze": command_analyze, "select": command_select}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
