#!/usr/bin/env python3
"""rewrite_verify_proto.py — verification-first "rewrite this block" prototype.

Execution/verification-side exploration of the rewrite-as-a-product idea:
"better" is EXECUTION-VERIFIED (digest of canonicalized outputs at multiple
input sizes), not judged. Companion of cases/compound.py (which stays on the
static side); this file owns the dynamic side.

Stages (all system python3, zero LLM calls, no GPU; every R run is a
sandboxed throwaway Rscript: unshare -rn netns + rlimits + temp HOME/cwd):

  probe        N corpus functions -> runnable fraction + failure taxonomy
               (tier-0 default call, tier-1 name heuristics, tier-2 ladder).
               Retry once with the WHOLE package R/ dir sourced on missing
               deps (attempt B).
  rename       benign-rewrite behavior check: rename one local variable in
               each runnable fn, digest-compare orig vs renamed in SEPARATE
               processes with identical input streams. Measures the
               behavior-preservation gate's false-positive rate on real code.
  scan         loop-shape prevalence: c()-accumulator / [[i]]<- accumulation
               / apply(X, 1|2,) / sapply(1:n) hit rates over a package pool,
               with runnable-looking candidate sites emitted for curation.
  equivalence  CURATED real corpus loop sites -> hand-authored vectorized /
               apply rewrites, digest-proven at 10..1e5, then benched
               (interleaved ABAB, adaptive batch) -> speedup distribution.
               Includes REVERSE cases (vectorized-but-not-better) as
               evidence for the maintainability prior.
  gates        the four-gate verifier run over the equivalence set:
               G0 parse/splice (tree-sitter) -> G1 lint-delta (ry + jarl
               findings) -> G2 diff-minimality (diff lines, LOC delta) ->
               G3 behavior (execution digest) -> G4 perf (>=2x at largest
               measurable size, hot-loop trigger). Per-gate cost recorded to
               justify the cheap-static-first ordering.

Artifacts under results/rewrite_verify_proto/ (gitignored). Usage:

  python3 rewrite_verify_proto.py --probe 50 --scan 250 --equivalence --gates
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # synthetic-data/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "cases"))

import scenarios as S                            # noqa: E402
from scenarios import Bundle, node_text          # noqa: E402
import cases.corpus as C                         # noqa: E402
import cases.validators as V                     # noqa: E402
import build_astfim as ASTFIM                    # noqa: E402

RESULTS = HERE / "results" / "rewrite_verify_proto"
HARNESS = HERE / "rw_harness.R"
ROOT = Path("/mnt/h/sepalith/normalized")
GENERATED_AT = "2026-08-20T00:00:00"             # deterministic output stamp
VERIFIER_VERSION = "rvp/1"                       # gate spec + harness version

# parent-link convention (registry subagent owns the general interface):
# every derived row carries base_sample_id (stable content hash of the
# normalized base = corpus file content + function identity) + rule id +
# rule version, and eval holdout happens at the base-sample level.
_RULE_VERSION = {"vectorize_c_acc": "1", "vectorize_subset2_acc": "1",
                 "loop_to_vapply": "1", "apply_m1_reverse": "1",
                 "manual": "1"}


def base_sample_id(file: str, fn: str, span: tuple[int, int] | None = None):
    h = hashlib.sha1()
    try:
        h.update(Path(file).read_bytes())
    except OSError:
        h.update(file.encode())
    h.update(f"\x00{fn}\x00{span[0] if span else -1}:"
             f"{span[1] if span else -1}".encode())
    return "bs:" + h.hexdigest()[:16]


def rule_id(cls: str) -> str:
    return f"{cls}@{_RULE_VERSION.get(cls, '1')}+{VERIFIER_VERSION}"

# real library paths (jsonlite/digest live in the USER library, which the
# sandbox HOME no longer owns); read-only use, discovered once per run
_REAL_LIBS = None


def real_libs() -> str:
    global _REAL_LIBS
    if _REAL_LIBS is None:
        try:
            p = subprocess.run(["Rscript", "--vanilla", "-e",
                                "cat(paste(.libPaths(), collapse=':'))"],
                               capture_output=True, text=True, timeout=60)
            _REAL_LIBS = p.stdout.strip()
        except Exception:
            _REAL_LIBS = ""
    return _REAL_LIBS

IO_CALLEES = {"readLines", "read.csv", "read.csv2", "read.table", "read.delim",
              "read.delim2", "scan", "download.file", "url", "file",
              "install.packages", "source", "readRDS", "readBin", "write.csv",
              "write.table", "saveRDS", "writeLines", "png", "pdf", "jpeg",
              "svg", "bmp", "dev.new"}
NET_CALLEES = {"download.file", "url", "curl", "GET", "POST", "install.packages"}
NATIVE_RE = re.compile(r"\.Call\s*\(|\.C\s*\(|\.Fortran\s*\(|\.External\s*\(|"
                       r"Rcpp::|Rcpp\.module|loadModule|\.Call")

RLIMIT_AS = 2 * 1024 ** 3
RLIMIT_CPU_S = 25
RLIMIT_FSIZE = 64 * 1024 ** 2


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# corpus access
# ---------------------------------------------------------------------------

def fn_name(b: Bundle, fn) -> str | None:
    p = fn.parent
    if p is not None and p.type == "binary_operator" and p.children:
        lhs = p.children[0]
        if lhs.type == "identifier":
            return node_text(b.src, lhs).decode("utf-8", "replace")
    return None


def fn_span(b: Bundle, fn) -> tuple[int, int]:
    """Byte span of the whole defining statement (name <- function ...)."""
    p = fn.parent
    if p is not None and p.type == "binary_operator":
        return p.start_byte, p.end_byte
    return fn.start_byte, fn.end_byte


def abs_file(pkg: str, rel: str) -> Path | None:
    """Absolute path of a bundle's file (highest version, same rule as
    iter_bundles_highest)."""
    versions = C._resolve_pkg_versions([pkg])
    vd = versions.get(pkg)
    if not vd:
        return None
    rdir = ASTFIM.src_root_for(Path(vd), pkg)
    if rdir is None:
        return None
    return rdir / Path(rel).name


def package_r_files(pkg: str, cap: int = 60) -> list[Path]:
    versions = C._resolve_pkg_versions([pkg])
    vd = versions.get(pkg)
    if not vd:
        return []
    rdir = ASTFIM.src_root_for(Path(vd), pkg)
    if rdir is None:
        return []
    try:
        return sorted(list(rdir.glob("*.R")) + list(rdir.glob("*.r")))[:cap]
    except OSError:
        return []


def make_pool(rng: random.Random, n_tidy: int = 40, n_rest: int = 80):
    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in S.list_packages() if p not in set(tidy)]
    rng.shuffle(rest)
    return tidy[:n_tidy] + rest[:n_rest]


def select_functions(rng: random.Random, want: int, params: dict):
    """Corpus functions (3..40 body lines, per-package cap) with statics:
    callee set, io/native flags, loop shapes. Returns (list, funnel)."""
    min_body = int(params.get("min_body_lines", 3))
    max_body = int(params.get("max_body_lines", 40))
    per_pkg = int(params.get("per_package", 2))
    budget = float(params.get("time_budget_s", 600))
    pool = params.get("pool") or make_pool(rng)
    rng.shuffle(pool)
    out, funnel = [], dict(functions_seen=0, size_ok=0, named=0,
                           packages=set(), files=0)
    counts: dict[str, int] = {}
    t0 = time.time()
    for b in C.iter_bundles_highest(pool, rng):
        funnel["files"] += 1
        funnel["packages"].add(b.package)
        if time.time() - t0 > budget or len(out) >= want:
            break
        if counts.get(b.package, 0) >= per_pkg:
            continue
        af = abs_file(b.package, b.rel)
        if af is None:
            continue
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            if len(out) >= want or time.time() - t0 > budget:
                break
            funnel["functions_seen"] += 1
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            _body, _head, _r0, _r1, nb = geom
            if not (min_body <= len(nb) <= max_body):
                continue
            funnel["size_ok"] += 1
            name = fn_name(b, fn)
            if not name or not re.fullmatch(r"[A-Za-z.][\w.]*", name):
                continue
            funnel["named"] += 1
            sb, eb = fn_span(b, fn)
            body_src = b.src[geom[2] if geom[2] > sb else sb:eb]
            callees = {S.callee_name(b.src, n) for n in V._walk(fn)
                       if n.type == "call"}
            has_native = bool(NATIVE_RE.search(body_src.decode(
                "utf-8", "replace")))
            out.append(dict(
                package=b.package, rel=b.rel, name=name,
                file=str(af), span=[sb, eb],
                body_lines=len(nb),
                io_calls=sorted(callees & IO_CALLEES),
                net_calls=sorted(callees & NET_CALLEES),
                has_native=has_native,
                for_loops=sum(1 for n in V._walk(fn)
                              if n.type == "for_statement"),
                callees=sorted(callees)[:40]))
            counts[b.package] = counts.get(b.package, 0) + 1
    funnel["packages"] = len(funnel["packages"])
    return out, funnel


# ---------------------------------------------------------------------------
# sandboxed R execution
# ---------------------------------------------------------------------------

_CPU_LIMIT = [RLIMIT_CPU_S]          # per-run override (single-threaded driver)


def _rlimits():
    resource.setrlimit(resource.RLIMIT_AS, (RLIMIT_AS, RLIMIT_AS))
    cpu = _CPU_LIMIT[0]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (RLIMIT_FSIZE, RLIMIT_FSIZE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_r(payload: dict, timeout_s: float = 30.0, tmp_root: Path | None = None,
          cpu_s: int | None = None):
    """Run the harness in a throwaway sandbox. Returns (result_dict, meta)."""
    if cpu_s is not None:
        _CPU_LIMIT[0] = cpu_s
    tmp = Path(tempfile.mkdtemp(prefix="rwsb_", dir=str(tmp_root)
                                if tmp_root else None))
    try:
        pf, of = tmp / "payload.json", tmp / "out.json"
        pf.write_text(json.dumps(payload))
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp), "TMPDIR": str(tmp),
            "R_LIBS_USER": str(tmp / "rlibs"), "R_HISTFILE": str(tmp / ".Rh"),
            "LANG": "C",
        }
        (tmp / "rlibs").mkdir(exist_ok=True)
        cmd = ["unshare", "-rn", "Rscript", "--vanilla",
               str(HARNESS), str(pf), str(of), real_libs()]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=str(tmp), env=env, preexec_fn=_rlimits,
                start_new_session=True, timeout=timeout_s,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.TimeoutExpired:
            return None, dict(kind="timeout", seconds=round(time.time() - t0, 1),
                              tmp=str(tmp))
        meta = dict(kind="done", rc=proc.returncode,
                    seconds=round(time.time() - t0, 2))
        if proc.returncode != 0:
            meta["stderr"] = proc.stderr[-400:]
            return None, meta
        if not of.exists():
            meta["stderr"] = proc.stderr[-400:] or "no output file"
            return None, meta
        try:
            return json.loads(of.read_text()), meta
        except ValueError as e:
            meta["stderr"] = f"bad json: {e}"
            return None, meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------

INPUT_GEN_HINTS = ("at least", "must be", "missing", "requires", "provide",
                   "invalid", "length", "dimensions", "one of", "either",
                   "cannot be", "should be", "expected", "supply", "empty")


def classify_probe(rec: dict, res, meta) -> str:
    if meta.get("kind") == "timeout":
        return "timeout"
    if res is None:
        return "sandbox_error"
    st = res.get("status")
    if st == "target_not_found":
        return "target_not_defined_after_source"
    if st == "ok":
        return "ok"
    errs = res.get("errors_seen") or []
    last = (errs[-1]["message"] if errs else "").lower()
    if "could not find function" in last:
        return "deps_missing"
    if "not found" in last:
        return "deps_missing"
    if rec.get("io_calls") and any(h in last for h in
                                   ("file", "connection", "url")):
        return "io_dependent"
    if any(h in last for h in INPUT_GEN_HINTS):
        return "input_gen_exhausted"
    return "runtime_error"


# ---------------------------------------------------------------------------
# stage: probe
# ---------------------------------------------------------------------------

def stage_probe(rng: random.Random, n: int, params: dict):
    fns, funnel = select_functions(rng, n, params)
    log(f"[probe] selected {len(fns)} functions "
        f"(funnel {json.dumps(funnel)})")
    rows = []
    for i, f in enumerate(fns):
        rec = dict(**f)
        payload = dict(mode="probe", files=[f["file"]], target=f["name"],
                       sizes=[10, 100, 1000], seed=int(params.get("seed", 13)))
        res, meta = run_r(payload, timeout_s=float(params.get("timeout_s", 30)))
        attempt = "file_only"
        if res is None or res.get("status") in ("target_not_found",
                                                "deps_missing"):
            pkg_files = [str(p) for p in package_r_files(f["package"])]
            if len(pkg_files) > 1:
                attempt = "package_dir"
                payload = dict(payload, files=pkg_files)
                res, meta = run_r(payload, timeout_s=float(
                    params.get("timeout_s", 30)))
        rec.update(attempt=attempt, taxonomy=classify_probe(rec, res, meta),
                   result=res, meta=meta)
        if rec["taxonomy"] == "ok":
            sizes_ok = [s for s in (res.get("sizes") or []) if s.get("ok")]
            rec["digest_sizes"] = [s["size"] for s in sizes_ok]
            rec["env_bearing"] = any(s.get("env_count", 0) > 0 for s in sizes_ok)
        rec.pop("callees", None)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            log(f"  [probe] {i + 1}/{len(fns)} done "
                f"({sum(1 for r in rows if r['taxonomy'] == 'ok')} ok)")
    out = RESULTS / "probe.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tax = {}
    for r in rows:
        tax[r["taxonomy"]] = tax.get(r["taxonomy"], 0) + 1
    stats = dict(n=len(rows), funnel=funnel, taxonomy=tax,
                 runnable=sum(1 for r in rows if r["taxonomy"] == "ok"),
                 attempt_b=sum(1 for r in rows if r["attempt"] == "package_dir"),
                 io_flagged=sum(1 for r in rows if r["io_calls"]),
                 native=sum(1 for r in rows if r["has_native"]))
    log(f"[probe] taxonomy: {json.dumps(tax)}")
    (RESULTS / "probe_stats.json").write_text(json.dumps(stats, indent=1))
    return rows, stats


# ---------------------------------------------------------------------------
# masked (string/comment-aware) replacement inside a byte span
# ---------------------------------------------------------------------------

def editable_mask(src: str) -> list[bool]:
    """True where a byte of src is 'code' (not inside string/comment)."""
    mask = [True] * len(src)
    i, q = 0, None
    while i < len(src):
        c = src[i]
        if q is None and c == "#":
            j = src.find("\n", i)
            j = len(src) if j < 0 else j
            for k in range(i, j):
                mask[k] = False
            i = j
            continue
        if q is None and c in "\"'":
            q = c
            mask[i] = False
        elif q is not None:
            mask[i] = False
            if c == "\\" and i + 1 < len(src):
                mask[i + 1] = False
                i += 2
                continue
            if c == q:
                q = None
        i += 1
    return mask


def replace_in_span(text: str, span: tuple[int, int], pattern: re.Pattern,
                    repl: str) -> tuple[str, int]:
    """Replace masked matches of pattern inside [span0, span1)."""
    mask = editable_mask(text)
    out, n = [], 0
    last = 0
    for m in pattern.finditer(text, span[0], span[1]):
        if not all(mask[m.start():m.end()]):
            continue
        out.append(text[last:m.start()])
        out.append(repl)
        last = m.end()
        n += 1
    out.append(text[last:])
    return "".join(out), n


def build_rename_variant(f: dict, rng: random.Random):
    """Rename one declared local (>= 2 occurrences incl. the assignment)
    inside the function span. Returns (new_text, old, new, n_occ) or None."""
    src = Path(f["file"]).read_bytes().decode("utf-8", "replace")
    sb, eb = f["span"]
    body = src[sb:eb]
    try:
        tree = V.parse_fragment(body)
    except Exception:
        return None
    # candidate locals: LHS identifiers of `<-` inside the function
    assigned: set[str] = set()
    for n in V._walk(tree.root_node):
        if n.type == "binary_operator" and len(n.children) >= 3 \
                and node_text(body.encode(), n.children[1]) == b"<-" \
                and n.children[0].type == "identifier":
            nm = node_text(body.encode(), n.children[0]).decode(
                "utf-8", "replace")
            if re.fullmatch(r"[A-Za-z.][\w.]*", nm):
                assigned.add(nm)
    file_ids = set(re.findall(r"[A-Za-z.][\w.]*", src))
    for nm in sorted(assigned):
        new = nm + "2"
        if new in file_ids or new in ("TRUE", "FALSE", "NULL", "NA"):
            continue
        pat = re.compile(r"(?<![\w.])" + re.escape(nm) + r"(?![\w.])")
        n_occ = sum(1 for m in pat.finditer(body))
        if n_occ < 2:
            continue
        new_src, n_repl = replace_in_span(src, (sb, eb), pat, new)
        if n_repl >= 2:
            return new_src, nm, new, n_repl
    return None


def stage_rename(rows: list[dict], params: dict):
    rng = random.Random(int(params.get("seed", 13)) + 7)
    ok_rows = [r for r in rows if r["taxonomy"] == "ok"]
    out_rows = []
    for r in ok_rows:
        got = build_rename_variant(r, rng)
        entry = dict(package=r["package"], rel=r["rel"], name=r["name"])
        if got is None:
            entry.update(status="no_rename_candidate")
            out_rows.append(entry)
            continue
        new_src, old_nm, new_nm, n_repl = got
        tmp = Path(tempfile.mkdtemp(prefix="rwrn_"))
        try:
            varfile = tmp / "variant.R"
            varfile.write_text(new_src)
            # same file set that made the probe succeed; in the package-dir
            # case the variant REPLACES the original file inside the set
            if r["attempt"] == "package_dir":
                files_o = [str(p) for p in package_r_files(r["package"])]
                files_n = [str(varfile) if p == r["file"] else p
                           for p in files_o]
            else:
                files_o = [r["file"]]
                files_n = [str(varfile)]
            base = dict(sizes=[10, 100, 1000],
                        seed=int(params.get("seed", 13)),
                        target=r["name"])
            res_o, meta_o = run_r(dict(base, mode="compare", files=files_o),
                                  timeout_s=float(params.get("timeout_s", 30)))
            res_n, meta_n = run_r(dict(base, mode="compare", files=files_n),
                                  timeout_s=float(params.get("timeout_s", 30)))
            do = [s.get("digest") for s in (res_o or {}).get("sizes", [])
                  if s.get("ok")]
            dn = [s.get("digest") for s in (res_n or {}).get("sizes", [])
                  if s.get("ok")]
            equal = (len(do) == 3 and do == dn)
            entry.update(status="ok" if equal else "mismatch",
                         old=old_nm, new=new_nm, replacements=n_repl,
                         digest_sizes=len(do),
                         env_bearing=any(s.get("env_count", 0) > 0
                                         for s in (res_o or {}).get("sizes", [])),
                         err_orig=(res_o or {}).get("status"),
                         err_new=(res_n or {}).get("status"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out_rows.append(entry)
    with open(RESULTS / "rename_compare.jsonl", "w") as fh:
        for e in out_rows:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_ok = sum(1 for e in out_rows if e["status"] == "ok")
    n_mm = sum(1 for e in out_rows if e["status"] == "mismatch")
    stats = dict(n=len(out_rows), digest_equal=n_ok, mismatch=n_mm,
                 no_candidate=sum(1 for e in out_rows
                                  if e["status"] == "no_rename_candidate"),
                 mismatch_details=[e for e in out_rows
                                   if e["status"] == "mismatch"][:20])
    log(f"[rename] digest_equal {n_ok}/{len(out_rows)} "
        f"(mismatch {n_mm})")
    (RESULTS / "rename_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------------------
# stage: loop-shape scan
# ---------------------------------------------------------------------------

C_ACC_RE = re.compile(rb"([\w.]+)\s*<-\s*c\s*\(\s*\1\s*[,)]")
SUBSET2_ACC_RE = re.compile(rb"([\w.]+)\[\[\s*\w+\s*\]\]\s*<-")
GROW_LEN_RE = re.compile(rb"([\w.]+)\[\[\s*length\s*\(\s*\1\s*\)\s*\+\s*1\s*\]\]\s*<-")
ROWFILL_RE = re.compile(rb"([\w.]+)\s*\[\s*[\w.]+\s*,\s*\]\s*<-")
VECFILL_RE = re.compile(rb"([\w.]+)\s*\[\s*[\w.]+\s*\]\s*<-")
APPLY_M1_RE = re.compile(rb"apply\s*\([^,]*,\s*1\s*[,)]")
SAPPLY_SEQ_RE = re.compile(rb"sapply\s*\(\s*(1\s*[:n]|seq_)")


def classify_loop_fn(b: Bundle, fn) -> dict:
    src_stripped = S.strip_strings(b.src)
    sb, eb = fn_span(b, fn)
    blob = src_stripped[sb:eb]
    return dict(
        has_for=blob.find(rb"for (") >= 0,
        c_acc=bool(C_ACC_RE.search(blob)),
        subset2_acc=bool(SUBSET2_ACC_RE.search(blob)) and
                    blob.find(rb"for (") >= 0,
        grow_len=bool(GROW_LEN_RE.search(blob)),
        apply_m1=bool(APPLY_M1_RE.search(blob)),
        sapply_seq=bool(SAPPLY_SEQ_RE.search(blob)),
    )


def stage_scan(rng: random.Random, params: dict):
    n_pkgs = int(params.get("scan_packages", 250))
    budget = float(params.get("scan_time_budget_s", 900))
    pool = make_pool(rng, n_tidy=n_pkgs // 5, n_rest=n_pkgs - n_pkgs // 5)
    st = dict(files=0, functions=0, packages=0, has_for=0, c_acc=0,
              subset2_acc=0, grow_len=0, apply_m1=0, sapply_seq=0,
              c_acc_runnable=0, candidates=[], elapsed_s=0.0)
    seen = set()
    t0 = time.time()
    for b in C.iter_bundles_highest(pool, rng):
        if time.time() - t0 > budget:
            break
        st["files"] += 1
        if b.package not in seen:
            seen.add(b.package)
            st["packages"] += 1
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            st["functions"] += 1
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            _b, _h, _r0, _r1, nb = geom
            if not (3 <= len(nb) <= 60):
                continue
            cls = classify_loop_fn(b, fn)
            for k in ("has_for", "c_acc", "subset2_acc", "grow_len",
                      "apply_m1", "sapply_seq"):
                if cls[k]:
                    st[k] += 1
            if cls["c_acc"] or cls["apply_m1"] or cls["grow_len"]:
                name = fn_name(b, fn)
                if not name:
                    continue
                callees = {S.callee_name(b.src, n) for n in V._walk(fn)
                           if n.type == "call"}
                af = abs_file(b.package, b.rel)
                entry = dict(package=b.package, rel=b.rel, name=name,
                             file=str(af) if af else None,
                             shapes=sorted(k for k in cls if cls[k]),
                             body_lines=len(nb),
                             io=bool(callees & IO_CALLEES),
                             native=bool(NATIVE_RE.search(
                                 b.src[geom[2]:geom[3]].decode(
                                     "utf-8", "replace"))),
                             n_callees=len(callees))
                st["candidates"].append(entry)
    st["elapsed_s"] = round(time.time() - t0, 1)
    (RESULTS / "loop_scan.json").write_text(json.dumps(st, indent=1))
    log(f"[scan] {st['packages']} pkgs / {st['functions']} fns / "
        f"{st['elapsed_s']}s: for {st['has_for']} c_acc {st['c_acc']} "
        f"subset2_acc {st['subset2_acc']} grow_len {st['grow_len']} "
        f"apply_m1 {st['apply_m1']} sapply_seq {st['sapply_seq']}")
    return st


# ---------------------------------------------------------------------------
# stage: deterministic bug injection (FIX-ISSUE direction, wave 2)
# Single-defect mutations on runnable corpus functions; the SAME digest
# harness measures the behavior gate's false-NEGATIVE rate (a one-defect
# diff should change the digest or error) — the counterpart of the rename
# stage's false-POSITIVE rate. Injected diffs are also the trivial-validator
# RL env (target = the original code) and judge-calibration ground truth.
# ---------------------------------------------------------------------------

BOUNDARY_RE = re.compile(r"(?<!<)(<=|>=)(?!>)")


def inject_defects(f: dict) -> list[tuple[str, str, str]]:
    """[(kind, old_text, new_text)] deterministic single-defect mutations
    inside the function span (masked: never inside strings/comments)."""
    src = Path(f["file"]).read_text(errors="replace")
    sb, eb = f["span"]
    out = []
    # (a) boundary-operator flip: first <= or >= in code
    mask = editable_mask(src)
    for m in BOUNDARY_RE.finditer(src, sb, eb):
        if all(mask[m.start():m.end()]):
            new_op = "<" if m.group(1) == "<=" else ">"
            out.append(("boundary_flip", m.group(1), new_op))
            break
    # (b) identifier swap: first local assigned >= 2x swapped with another
    # local assigned in the same function (use-before-def style defect)
    body = src[sb:eb]
    ids = sorted(set(re.findall(r"(?<![\w.])([A-Za-z.][\w.]*)", body)))
    assigned = []
    for nm in ids:
        if re.search(r"(?<![\w.])" + re.escape(nm) + r"(?![\w.])\s*<-", body):
            assigned.append(nm)
    if len(assigned) >= 2:
        a, b = assigned[0], assigned[1]
        pa = re.compile(r"(?<![\w.])" + re.escape(a) + r"(?![\w.])")
        # swap: a -> b at its LAST read occurrence (mask-aware)
        last = None
        for m in pa.finditer(body):
            if all(mask[sb + m.start():sb + m.end()]):
                last = m
        if last is not None:
            out.append(("identifier_swap", a, b))
    return out


def apply_first_defect(src: str, span: tuple[int, int], kind: str,
                       old: str, new: str) -> str:
    mask = editable_mask(src)
    if kind == "boundary_flip":
        pat = re.compile(r"(?<!<)" + re.escape(old) + r"(?!>)")
        for m in pat.finditer(src, span[0], span[1]):
            if all(mask[m.start():m.end()]):
                return src[:m.start()] + new + src[m.end():]
    if kind == "identifier_swap":
        pat = re.compile(r"(?<![\w.])" + re.escape(old) + r"(?![\w.])")
        last = None
        for m in pat.finditer(src, span[0], span[1]):
            if all(mask[m.start():m.end()]):
                last = m
        if last is not None:
            return src[:last.start()] + new + src[last.end():]
    return src


def stage_inject(rows: list[dict], params: dict):
    ok_rows = [r for r in rows if r["taxonomy"] == "ok"]
    out_rows = []
    for r in ok_rows:
        defects = inject_defects(r)
        entry = dict(package=r["package"], rel=r["rel"], name=r["name"],
                     n_defects=len(defects))
        if not defects:
            entry.update(status="no_injectable_site")
            out_rows.append(entry)
            continue
        kind, old, new = defects[0]
        tmp = Path(tempfile.mkdtemp(prefix="rwij_"))
        try:
            src = Path(r["file"]).read_text(errors="replace")
            mutated = apply_first_defect(src, tuple(r["span"]), kind, old, new)
            (tmp / "variant.R").write_text(mutated)
            (tmp / "orig.R").write_text(src)
            base = dict(sizes=[10, 100, 1000],
                        seed=int(params.get("seed", 13)),
                        target=r["name"])
            res_o, _ = run_r(dict(base, mode="compare",
                                  files=[str(tmp / "orig.R")]),
                             timeout_s=float(params.get("timeout_s", 30)))
            res_n, _ = run_r(dict(base, mode="compare",
                                  files=[str(tmp / "variant.R")]),
                             timeout_s=float(params.get("timeout_s", 30)))
            so = [s.get("digest") for s in (res_o or {}).get("sizes", [])
                  if s.get("ok")]
            sn_ok = [s.get("ok") for s in (res_n or {}).get("sizes", [])]
            sn = [s.get("digest") for s in (res_n or {}).get("sizes", [])
                  if s.get("ok")]
            detected = "no" if (len(so) == 3 and so == sn and
                                len(sn) == 3) else "yes"
            if len(sn) == 3 and so != sn:
                how = "digest_changed"
            elif not any(sn_ok or []):
                how = "errors_now"
            elif any(sn_ok or []):
                how = "partial"
            else:
                how = "unknown"
            entry.update(status="ok", kind=kind, mutated=old, into=new,
                         detected=detected, detection_mode=how)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out_rows.append(entry)
    with open(RESULTS / "inject.jsonl", "w") as fh:
        for e in out_rows:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    n = [e for e in out_rows if e["status"] == "ok"]
    stats = dict(n=len(out_rows), injected=len(n),
                 no_site=sum(1 for e in out_rows
                             if e["status"] == "no_injectable_site"),
                 detected=sum(1 for e in n if e["detected"] == "yes"),
                 missed=sum(1 for e in n if e["detected"] == "no"),
                 by_mode={k: sum(1 for e in n if e.get("detection_mode") == k)
                          for k in ("digest_changed", "errors_now",
                                    "partial", "unknown")})
    log(f"[inject] single-defect detection: {stats['detected']}/{len(n)} "
        f"detected (missed {stats['missed']}); modes {stats['by_mode']}")
    (RESULTS / "inject_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------------------
# curated equivalence set: REAL corpus sites, hand-authored rewrites
# (found via stage_scan; verified by execution below). expect:
#   "gain"    hot loop -> vectorized/apply, speedup expected
#   "no_gain" reverse case: vectorized/apply form that is NOT better
# ---------------------------------------------------------------------------

CURATED_FILE = HERE / "rewrite_curated.json"


def load_curated():
    if CURATED_FILE.exists():
        return json.loads(CURATED_FILE.read_text())
    return []


def stage_equivalence(params: dict):
    curated = load_curated()
    if not curated:
        log("[equiv] no curated entries (run --scan first, then author "
            "rewrite_curated.json)")
        return []
    rows = []
    for e in curated:
        src_path = Path(e["file"])
        if not src_path.exists():
            log(f"[equiv] MISSING FILE {e['file']}")
            continue
        src = src_path.read_text(errors="replace")   # universal newlines:
        # CRLF corpus files must match the curated blocks (authored \n)
        old, new = e["old"], e["new"]
        if src.count(old) != 1:
            log(f"[equiv] {e['id']}: old block not found uniquely "
                f"({src.count(old)} matches) — skipped")
            continue
        tmp = Path(tempfile.mkdtemp(prefix="rweq_"))
        try:
            varfile = tmp / "variant.R"
            varfile.write_text(src.replace(old, new, 1))
            orig_file = tmp / "orig.R"
            orig_file.write_text(src)
            sizes_digest = e.get("sizes_digest") or [10, 100, 1000, 10000,
                                                     100000]
            sizes_bench = e.get("sizes_bench") or [1000, 10000, 100000]
            base = dict(seed=int(params.get("seed", 13)),
                        target=e["fn"], max_attempts=12,
                        fixed_args=e.get("fixed_args"),
                        attach=e.get("attach"))
            # behavior: digests in separate processes, identical input streams
            t_exec = time.time()
            res_o, _ = run_r(dict(base, mode="compare",
                                  files=[str(orig_file)],
                                  sizes=sizes_digest),
                             timeout_s=120, cpu_s=90)
            res_n, _ = run_r(dict(base, mode="compare",
                                  files=[str(varfile)],
                                  sizes=sizes_digest),
                             timeout_s=120, cpu_s=90)
            t_behavior = round(time.time() - t_exec, 2)
            so = [s for s in (res_o or {}).get("sizes", [])]
            sn = [s for s in (res_n or {}).get("sizes", [])]
            per_size = []
            for a, b2 in zip(so, sn):
                per_size.append(dict(
                    size=a.get("size"), ok_old=a.get("ok"), ok_new=b2.get("ok"),
                    equal=bool(a.get("ok") and b2.get("ok")
                               and a.get("digest") == b2.get("digest")),
                    shape_old=(a.get("shape") or {}).get("class"),
                    shape_new=(b2.get("shape") or {}).get("class"),
                    env_bearing=bool((a.get("env_count") or 0) > 0)))
            # bench: both variants interleaved in ONE process
            files = e.get("files") or [str(orig_file)]
            t_exec = time.time()
            res_b, _ = run_r(dict(base, mode="bench", files=files,
                                  files_new=[str(varfile)],
                                  sizes=sizes_bench),
                             timeout_s=400, cpu_s=350)
            t_bench = round(time.time() - t_exec, 2)
            bench = (res_b or {}).get("per_size") or []
            row = dict(**{k: e[k] for k in
                          ("id", "package", "rel", "fn", "cls", "expect",
                           "note")})
            row.update(
                base_sample_id=base_sample_id(e["file"], e["fn"]),
                rule=rule_id(e.get("cls", "manual")),
                status_probe_old=(res_o or {}).get("status"),
                status_probe_new=(res_n or {}).get("status"),
                status_bench=(res_b or {}).get("status"),
                exec_s_behavior=t_behavior, exec_s_bench=t_bench,
                behavior=per_size, bench=bench)
            rows.append(row)
            eq_sizes = [p["size"] for p in per_size if p.get("equal")]
            top_speedup = max([b2.get("speedup", 0) for b2 in bench] or [0])
            log(f"[equiv] {e['id']} {e['fn']} ({e['package']}): "
                f"equal at {len(eq_sizes)}/{len(per_size)} sizes; "
                f"max speedup {top_speedup}x")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    with open(RESULTS / "equivalence.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


# ---------------------------------------------------------------------------
# four-gate verifier (cheap static first, execution last)
# ---------------------------------------------------------------------------

RY = str(Path.home() / ".local" / "bin" / "ry")
JARL = str(Path.home() / ".local" / "bin" / "jarl")


def lint_counts(path: Path) -> tuple[int, int] | None:
    """(ry_findings, jarl_findings) for one file."""
    try:
        r = subprocess.run([RY, "check", "--output-format", "json",
                            "--exit-zero", str(path)],
                           capture_output=True, text=True, timeout=60)
        ry_n = len(json.loads(r.stdout or "[]"))
    except Exception:
        ry_n = None
    try:
        r = subprocess.run([JARL, "check", "--output-format", "json",
                            "--no-color", str(path)],
                           capture_output=True, text=True, timeout=60)
        obj = json.loads(r.stdout or "{}")
        if isinstance(obj, dict):                 # {"diagnostics": [...]}
            jarl_n = len(obj.get("diagnostics") or [])
        else:
            jarl_n = len(obj)
    except Exception:
        jarl_n = None
    if ry_n is None and jarl_n is None:
        return None
    return (ry_n or 0, jarl_n or 0)


def hot_loop_trigger(old_text: str) -> bool:
    """Static hot-loop: a for whose sequence is data-scaled and whose body
    accumulates ([[i]]<- / v[i] <- / m[i, ] <- / c(append)). Informational;
    the MEASURED scaling (bench) is what arms the perf gate."""
    blob = S.strip_strings(old_text.encode())
    for m in re.finditer(rb"for\s*\(\s*[\w.]+\s+in\s+([^\)]{0,60})\)", blob):
        seq = m.group(1)
        if re.match(rb"^(seq_along|seq_len)\s*\(", seq) or \
                re.match(rb"^[\w.]+$", seq) or \
                re.match(rb"^1\s*:\s*[\w.]+$", seq):
            rest = blob[m.end():m.end() + 400]
            if any(rx.search(rest) for rx in (C_ACC_RE, SUBSET2_ACC_RE,
                                              GROW_LEN_RE, ROWFILL_RE,
                                              VECFILL_RE)):
                return True
    return False


def gate_verdicts(row: dict, params: dict) -> dict:
    """Run gates G0..G5 for one equivalence row, cheap-first, short-circuit.
    Safety gates G0-G4; VALUE gate G5 (the no_op rationale). Returns per-gate
    pass/fail + milliseconds.

    G0 parse/splice (tree-sitter, static)         ~0 ms
    G1 lint-delta (ry + jarl findings, CLI)       ~0.3 s
    G2 diff-minimality (diff lines, LOC delta)    ~0 ms
    G3 behavior (cross-process digests, measured in the equivalence stage)
    G4 perf: ONLY if hot_effective = static hot-loop AND measured N-scaling
       (orig_ms largest >= 2x smallest bench size) -> >= 2x at largest size
    G5 value: at least ONE axis must improve (LOC down, lint findings down,
       or >= 2x speedup); otherwise the rewrite is churn -> no_op wins.
    """
    e = {k: row[k] for k in ("id", "package", "rel", "fn", "cls", "expect")}
    curated = {c["id"]: c for c in load_curated()}
    c = curated.get(row["id"], {})
    old_text, new_text = c.get("old", ""), c.get("new", "")
    g = dict(id=row["id"], cls=row.get("cls"), expect=row.get("expect"))
    if row.get("base_sample_id"):
        g["base_sample_id"] = row["base_sample_id"]
    if row.get("rule"):
        g["rule"] = row["rule"]
    t0 = time.time()
    # G0 parse/splice (static, tree-sitter)
    ok0 = V.fragment_clean(new_text) if new_text else False
    g["G0_parse"] = ok0
    g["ms_G0"] = round((time.time() - t0) * 1000, 2)
    if not ok0:
        g["verdict"] = "reject_G0"
        return g
    # G1 lint-delta (ry + jarl findings counts) on the FULL FILES: fragment
    # lint inflates undefined-symbol findings (the block has no context);
    # on matched files the context noise cancels in the delta
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="rwlg_"))
    try:
        fo, fn = tmp / "o.R", tmp / "n.R"
        try:
            full_src = Path(c.get("file", "")).read_text(errors="replace")
        except OSError:
            full_src = old_text
        fo.write_text(full_src)
        fn.write_text(full_src.replace(old_text, new_text, 1)
                      if full_src.count(old_text) == 1 else new_text)
        lo, ln = lint_counts(fo), lint_counts(fn)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    g["lint_old"], g["lint_new"] = lo, ln
    if lo is None or ln is None:
        g["G1_lint"] = None
    else:
        g["G1_lint"] = (ln[0] + ln[1]) <= (lo[0] + lo[1])
    g["ms_G1"] = round((time.time() - t0) * 1000, 2)
    # G2 diff-minimality (rules depend on hotness, resolved after bench look)
    t0 = time.time()
    diff_lines = [l for l in difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(), lineterm="")][2:]
    loc_old = sum(1 for l in old_text.splitlines() if l.strip())
    loc_new = sum(1 for l in new_text.splitlines() if l.strip())
    hot_static = hot_loop_trigger(old_text)
    bench = [b for b in row.get("bench", []) if b.get("speedup")]
    # hot_effective: MEASURED data-scaling (orig runtime at the largest bench
    # size >= 2x the smallest); falls back to the static trigger when there
    # is no bench. This is what arms the perf path — a capped-N loop (e.g.
    # bayesplot's length-6 scheme) stays on the maintainability track.
    hot_effective = False
    g["scaling_ratio"] = None
    if len(bench) >= 2:
        by_size = sorted(bench, key=lambda b: b["size"])
        ratio = by_size[-1]["orig_ms"] / max(by_size[0]["orig_ms"], 1e-9)
        g["scaling_ratio"] = round(ratio, 2)
        hot_effective = ratio >= 2.0
    elif hot_static:
        hot_effective = True
    g["diff_lines"] = len(diff_lines)
    g["loc_delta"] = loc_new - loc_old
    g["hot_static"], g["hot_effective"] = hot_static, hot_effective
    if hot_effective:
        g["G2_diff"] = len(diff_lines) <= 30 and (loc_new - loc_old) <= 4
    else:
        g["G2_diff"] = len(diff_lines) <= 30 and (loc_new - loc_old) <= 0
    g["ms_G2"] = round((time.time() - t0) * 1000, 2)
    # G3 behavior (execution) — from the equivalence row; cost = the measured
    # cross-process digest run (exec_s_behavior), not re-measured here
    sizes_ok = [p for p in row.get("behavior", []) if p.get("equal")]
    env_bearing = any(p.get("env_bearing", False) for p in row.get(
        "behavior", []))
    g["G3_behavior"] = len(sizes_ok) >= 3 and not env_bearing
    g["behavior_sizes"] = [p["size"] for p in sizes_ok]
    g["env_bearing"] = env_bearing
    g["ms_G3"] = row.get("exec_s_behavior", 0) * 1000
    # G4 perf (execution) — hot_effective AND >= 2x at largest measurable
    # size, with a timing-stability guard (IQR < 30% of median on orig arm)
    gate4 = None
    speedup_largest = None
    if hot_effective and bench:
        largest = max(bench, key=lambda b: b["size"])
        speedup_largest = largest["speedup"]
        stable = (largest.get("orig_iqr_ms", 1) < 0.3 * max(
            largest.get("orig_ms", 1), 1e-9))
        gate4 = bool(largest["speedup"] >= 2.0 and stable)
    g["G4_perf"] = gate4
    g["speedup_largest"] = speedup_largest
    g["ms_G4"] = row.get("exec_s_bench", 0) * 1000
    # G5 value — no_op is the right action unless something measurably improves
    lint_improved = bool(
        lo and ln and (ln[0] + ln[1]) < (lo[0] + lo[1]))
    perf_improved = bool(bench and max(b["speedup"] for b in bench) >= 2.0)
    g["value_axes"] = dict(loc=(loc_new - loc_old) < 0, lint=lint_improved,
                           perf=perf_improved)
    g["G5_value"] = ((loc_new - loc_old) < 0 or lint_improved or perf_improved)
    # overall
    core = [g["G0_parse"], g.get("G1_lint"), g["G2_diff"], g["G3_behavior"]]
    core_ok = all(v is not False and v is not None for v in core)
    if not core_ok:
        g["verdict"] = "reject"
    elif hot_effective and g["G4_perf"] is False:
        g["verdict"] = "reject_no_perf_gain"
    elif not g["G5_value"]:
        g["verdict"] = "reject_no_value_no_op_wins"
    else:
        g["verdict"] = "accept"
    return g


def stage_gates(equiv_rows: list[dict]):
    rows = [gate_verdicts(r, {}) for r in equiv_rows]
    with open(RESULTS / "gates.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    def frac(k):
        vals = [r.get(k) for r in rows]
        return dict(true=sum(1 for v in vals if v is True),
                    false=sum(1 for v in vals if v is False),
                    none=sum(1 for v in vals if v is None))
    ms = {f"ms_{k}": round(sum(r.get(f"ms_{k}", 0) for r in rows) / max(n, 1), 1)
          for k in ("G0", "G1", "G2", "G3", "G4")}
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    stats = dict(n=n, gates={k: frac(k) for k in
                             ("G0_parse", "G1_lint", "G2_diff", "G3_behavior",
                              "G4_perf", "G5_value")},
                 mean_gate_ms=ms, verdicts=verdicts)
    (RESULTS / "gates_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"[gates] verdicts: {json.dumps(verdicts)}; mean ms {json.dumps(ms)}")
    return stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--rename", action="store_true")
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    args = ap.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    params = dict(seed=args.seed, timeout_s=args.timeout_s)
    rng = random.Random(args.seed)
    t0 = time.time()

    probe_rows = []
    if args.probe:
        probe_rows, _ = stage_probe(rng, args.probe, params)
    if args.rename:
        if not probe_rows and (RESULTS / "probe.jsonl").exists():
            probe_rows = [json.loads(l) for l in
                          (RESULTS / "probe.jsonl").read_text().splitlines()]
        stage_rename(probe_rows, params)
    if args.inject:
        if not probe_rows and (RESULTS / "probe.jsonl").exists():
            probe_rows = [json.loads(l) for l in
                          (RESULTS / "probe.jsonl").read_text().splitlines()]
        stage_inject(probe_rows, params)
    if args.scan:
        stage_scan(rng, dict(params, scan_packages=args.scan,
                             scan_time_budget_s=1200))
    equiv_rows = []
    if args.equivalence:
        equiv_rows = stage_equivalence(params)
        if not equiv_rows and (RESULTS / "equivalence.jsonl").exists():
            equiv_rows = [json.loads(l) for l in
                          (RESULTS / "equivalence.jsonl").read_text().splitlines()]
    if args.gates:
        if not equiv_rows and (RESULTS / "equivalence.jsonl").exists():
            equiv_rows = [json.loads(l) for l in
                          (RESULTS / "equivalence.jsonl").read_text().splitlines()]
        stage_gates(equiv_rows)
    log(f"[done] {round(time.time() - t0, 1)}s total; artifacts in {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
