#!/usr/bin/env python3
"""Normalize embedded R code in external-code families with the project
toolchain: `air format` + `jarl check --fix --allow-no-vcs` (both CPU-only,
run under nice).

Targets and code locations
  hidden_r_instruction_v1/ling_coder_r.jsonl    OUTPUT = messages[-1].content
  hidden_r_instruction_v1/codex_r_strict.jsonl  OUTPUT field
  synthetic_analyst_v1/analyst_scripts.jsonl    code field
  paper_to_r_pilot/examples.jsonl               implementation field

Code extraction policy
  - fenced blocks tagged R (```r / ```R / ```{r,...} knitr chunks) are
    extracted, normalized, and re-embedded with their fences preserved;
    a block that still fails to parse after normalization (air format AND
    jarl check both reject it) drops the whole row -- the record claims to
    contain R and does not;
  - untagged fences are normalized best-effort: if the block does not parse
    it is kept verbatim (likely console output / prose, not code);
  - fences tagged with another language (yaml, python, ...) are untouched;
  - a fenceless OUTPUT field (assistant answer) is normalized only when it
    parses as R (bare-code answers); prose / unparseable text passes through
    verbatim and is never dropped;
  - a field with no fences (pure-code fields: analyst `code`, paper
    `implementation`) is normalized as a single block; unparseable -> row
    dropped in legacy mode, retained with dropped_reason in dual mode;
  - rows with no R content at all pass through unchanged.

Incremental + atomic: progress (processed row index + kept-line count) is
checkpointed to <stem>.normalize_state.json every batch; interrupted runs
resume from the checkpoint. The fully processed output is written to
<stem>.normalized_tmp.jsonl, then os.replace()d over the original.
Before replacing, the original's (size, mtime_ns) is re-checked against the
value seen at read time: if a live appender touched it in between (the
detached generate_analyst.py runner), the replace is ABORTED and the result
kept as <name>.normalized.jsonl instead (enrich_provenance.py convention).

Dual mode (default; --no-dual for the legacy replace-only behavior) keeps
BOTH versions instead of rewriting in place, for maximum retention:
  code_original : verbatim pre-normalization value of the code field
                  (null when the original is not recoverable)
  normalized    : true iff R content in the record was processed by
                  air/jarl (formatted or verified already-clean); false for
                  pass-through prose answers and never-processed rows
  dropped_reason: present only on rows whose declared-R code failed to
                  parse after normalization ("unparseable_r_block(N)") --
                  legacy mode deleted these rows; dual mode RETAINS them
                  verbatim (normalized=false) instead of dropping.

Usage
  nice -n 19 uv run python experiments/post-processing/normalize_external.py \
      [--workers 6] [--only ling,codex,analyst,paper] [--no-dual]
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAS = Path("/mnt/h/sepalith/datasets")
DIFF_SAMPLES = 3

# (path, mode) -- mode names the getter/setter for the code-bearing field
TARGETS = [
    (NAS / "hidden_r_instruction_v1/ling_coder_r.jsonl", "last_message"),
    (NAS / "hidden_r_instruction_v1/codex_r_strict.jsonl", "output"),
    (NAS / "synthetic_analyst_v1/analyst_scripts.jsonl", "code"),
    (NAS / "paper_to_r_pilot/examples.jsonl", "implementation"),
]

FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)

# output-like modes carry markdown ANSWER TEXT (code lives in fences; a
# fenceless answer is prose and must pass through untouched). The pure-code
# modes (analyst `code`, paper `implementation`) hold R source as the whole
# value, so the entire field is one normalizable block.
OUTPUT_MODES = {"output", "last_message"}


def field_get(rec: dict, mode: str) -> str:
    if mode == "last_message":
        return rec["messages"][-1]["content"]
    return rec[mode]


def field_set(rec: dict, mode: str, value: str) -> None:
    if mode == "last_message":
        rec["messages"][-1]["content"] = value
    else:
        rec[mode] = value


def classify_tag(tag: str) -> str:
    """'r' | 'untagged' | 'other' for a fence opening tag."""
    t = tag.strip().lower()
    if t == "r" or t.startswith("r ") or t.startswith("r,") or t.startswith("r;") \
            or t.startswith("r'") or t.startswith("{r"):
        return "r"
    if t == "":
        return "untagged"
    return "other"


# ---------------------------------------------------------------------------
# air + jarl (one temp .R file per worker thread)
# ---------------------------------------------------------------------------

_tls = threading.local()
_cache: dict[str, tuple[str, str]] = {}     # sha1(code) -> (status, text)
_cache_lock = threading.Lock()


def _tmp_r() -> Path:
    p = getattr(_tls, "tmp_r", None)
    if p is None:
        _tls.tmp_dir = tempfile.TemporaryDirectory(prefix="normalize_ext_")
        p = Path(_tls.tmp_dir.name) / "block.R"
        _tls.tmp_r = p
    return p


def _run(cmd: list[str]) -> int:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60).returncode
    except (subprocess.TimeoutExpired, OSError):
        return 255


def normalize_code(code: str) -> tuple[str, str]:
    """-> (status, text). status: 'ok' (normalized text), 'same' (normalized
    == original), 'unparseable' (air AND jarl reject), 'air_unformattable'
    (parses per jarl but air refuses -> original kept)."""
    key = hashlib.sha1(code.encode("utf-8", "surrogateescape")).hexdigest()
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    p = _tmp_r()
    try:
        p.write_text(code, encoding="utf-8", errors="strict")
    except (UnicodeEncodeError, OSError):
        return ("air_unformattable", code)   # non-utf8 payload: keep verbatim
    air_rc = _run(["air", "format", str(p)])
    if air_rc == 0:
        jarl_rc = _run(["jarl", "check", "--fix", "--allow-no-vcs", str(p)])
        if jarl_rc in (0, 1):                # 1 = leftover unfixable lints
            try:
                out = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                out = code
            res = ("ok", out) if out != code else ("same", code)
        else:                                # jarl unhappy with air output
            res = ("air_unformattable", code)
    else:
        # second opinion: jarl is the authoritative parse gate
        jcheck = _run(["jarl", "check", "--allow-no-vcs", str(p)])
        res = ("unparseable", code) if jcheck == 255 else \
            ("air_unformattable", code)
    with _cache_lock:
        if len(_cache) < 200_000:
            _cache[key] = res
    return res


# ---------------------------------------------------------------------------
# per-row processing (thread-safe: returns its own counters)
# ---------------------------------------------------------------------------

def process_field(text: str, mode: str) -> tuple[str, bool, dict, str, str]:
    """-> (new_text, drop_row, counters, old_block, new_block). old/new_block
    are the first changed block (for diff samples), '' when unchanged."""
    c = dict(blocks=0, blocks_normalized=0, blocks_already_clean=0,
             blocks_unparseable_r=0, blocks_unparseable_untagged_kept=0,
             blocks_air_unformattable_kept=0, blocks_other_lang_skipped=0,
             pure_field=0, output_no_fences=0)
    sample = ["", ""]

    def do_block(body: str, kind: str) -> tuple[str, bool]:
        c["blocks"] += 1
        if not body.strip():
            return body, False
        status, out = normalize_code(body)
        if status == "ok":
            c["blocks_normalized"] += 1
            if not sample[0]:
                sample[:] = [body, out]
            return out, False
        if status == "same":
            c["blocks_already_clean"] += 1
            return body, False
        if status == "unparseable":
            if kind == "r":
                c["blocks_unparseable_r"] += 1
                return body, True           # declared R, does not parse -> drop
            c["blocks_unparseable_untagged_kept"] += 1
            return body, False              # probably not code: keep verbatim
        c["blocks_air_unformattable_kept"] += 1
        return body, False

    if "```" in text:
        out, pos, drop = [], 0, False
        for m in FENCE_RE.finditer(text):
            out.append(text[pos:m.start()])
            tag, kind = m.group(1), classify_tag(m.group(1))
            if kind == "other":
                c["blocks_other_lang_skipped"] += 1
                out.append(m.group(0))
            else:
                body, d = do_block(m.group(2), kind)
                drop |= d
                out.append(f"```{tag}\n{body}```")
            pos = m.end()
        out.append(text[pos:])
        return "".join(out), drop, c, sample[0], sample[1]

    # fenceless field: pure-code modes normalize the whole value; output-like
    # modes hold answers that MAY be bare code or prose -- try to normalize,
    # and pass through verbatim when it does not parse (prose is never
    # rewriteable, and unparseable fenceless text is never dropped)
    if mode in OUTPUT_MODES:
        c["output_no_fences"] += 1
        if text.strip():
            status, out = normalize_code(text)
            if status == "ok":
                c["blocks"] += 1
                c["blocks_normalized"] += 1
                return out, False, c, text, out
            if status == "same":
                c["blocks"] += 1
                c["blocks_already_clean"] += 1
        return text, False, c, "", ""
    c["pure_field"] += 1
    body, drop = do_block(text, "r")
    if body != text and not sample[0]:
        sample[:] = [text, body]
    return body, drop, c, sample[0], sample[1]


def diff_sample(old: str, new: str, n_lines: int = 14) -> str:
    ob, nb = None, None
    for a, b in zip(old.split("```"), new.split("```")):
        if a != b:
            ob, nb = a, b
            break
    if ob is None:
        ob, nb = old, new
    d = [l for l in difflib.unified_diff(ob.splitlines(), nb.splitlines(),
                                         "before", "after", lineterm="", n=1)]
    return "\n".join(d[:n_lines]) + (" ..." if len(d) > n_lines else "")


# ---------------------------------------------------------------------------
# incremental + atomic file driver
# ---------------------------------------------------------------------------

def stat_sig(p: Path) -> tuple[int, int]:
    s = p.stat()
    return s.st_size, s.st_mtime_ns


BLOCK_COUNTERS = ("blocks", "blocks_normalized", "blocks_already_clean",
                  "blocks_unparseable_r", "blocks_unparseable_untagged_kept",
                  "blocks_air_unformattable_kept", "blocks_other_lang_skipped",
                  "pure_field", "output_no_fences")
BATCH = 256


def normalize_file(path: Path, mode: str, workers: int,
                   dual: bool = True) -> dict:
    stem = path.with_suffix("")            # <dir>/<name> w/o .jsonl
    state_p = Path(str(stem) + ".normalize_state.json")
    tmp_p = Path(str(stem) + ".normalized_tmp.jsonl")
    st = dict(rows_total=0, rows_read=0, rows_normalized=0,
              rows_unchanged=0, rows_dropped_unparseable=0,
              replaced=None, **{k: 0 for k in BLOCK_COUNTERS}, diffs=[])

    sig = stat_sig(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    st["rows_total"] = len(lines)

    done, out_lines = 0, 0
    if state_p.exists():
        try:
            state = json.loads(state_p.read_text())
            if (state["size"], state["mtime_ns"]) == sig and \
                    tmp_p.exists() and \
                    sum(1 for _ in tmp_p.open("rb")) == state["out_lines"]:
                done, out_lines = state["done"], state["out_lines"]
                for k, v in state.get("stats", {}).items():
                    if k == "diffs":
                        st["diffs"] = v
                    elif k in st and isinstance(v, (int, float)):
                        st[k] = v
                print(f"  resuming at row {done}/{len(lines)} "
                      f"({out_lines} kept lines)")
        except (ValueError, KeyError, OSError):
            pass
    if done == 0:
        tmp_p.write_text("")

    def save_state():
        s = dict(size=sig[0], mtime_ns=sig[1], done=done, out_lines=out_lines,
                 stats=st)
        t = Path(str(state_p) + ".tmp")
        t.write_text(json.dumps(s))
        os.replace(t, state_p)

    def one(i: int):
        rec = json.loads(lines[i])
        old = field_get(rec, mode)
        new, drop, c, sob, snb = process_field(old, mode)
        if dual:
            rec["code_original"] = old
            # the flag describes the STORED content: retained-unparseable
            # rows keep the verbatim original -> normalized=false
            rec["normalized"] = (not drop) and (
                c["blocks_normalized"] + c["blocks_already_clean"]) > 0
            if drop:      # dual mode RETAINS unparseable rows verbatim
                rec["dropped_reason"] = \
                    f"unparseable_r_block({c['blocks_unparseable_r']})"
            else:
                field_set(rec, mode, new)
            return (i, drop, (not drop and new != old), c, sob, snb,
                    json.dumps(rec, ensure_ascii=False))
        if not drop and new != old:
            field_set(rec, mode, new)
        return (i, drop, new != old, c, sob, snb,
                None if drop else json.dumps(rec, ensure_ascii=False))

    pool = ThreadPoolExecutor(max_workers=workers)
    with tmp_p.open("a", encoding="utf-8") as out_f:
        for base in range(done, len(lines), BATCH):
            res = list(pool.map(one, range(base, min(base + BATCH, len(lines)))))
            for i, drop, changed, c, sob, snb, line in res:
                st["rows_read"] += 1
                for k in BLOCK_COUNTERS:
                    st[k] += c[k]
                if drop:
                    st["rows_dropped_unparseable"] += 1
                    if not dual:      # legacy mode deletes the row
                        continue
                if changed:
                    st["rows_normalized"] += 1
                    if len(st["diffs"]) < DIFF_SAMPLES:
                        st["diffs"].append(
                            dict(row=i, diff=diff_sample(sob, snb)))
                else:
                    st["rows_unchanged"] += 1
                out_f.write(line + "\n")
                out_lines += 1
            out_f.flush()
            os.fsync(out_f.fileno())
            done = min(base + BATCH, len(lines))
            save_state()
    pool.shutdown()

    # atomic replace, guarded against a live appender (generate_analyst.py)
    if stat_sig(path) != sig:
        side = Path(str(stem) + ".normalized.jsonl")
        os.replace(tmp_p, side)
        state_p.unlink(missing_ok=True)
        st["replaced"] = False
        st["snapshot_sidecar"] = str(side)
        print(f"  ABORTED replace ({path.name} changed mid-run by a live "
              f"writer); normalized snapshot kept at {side.name}")
    else:
        os.replace(tmp_p, path)
        state_p.unlink(missing_ok=True)
        st["replaced"] = True
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel air/jarl subprocess calls (CPU-polite)")
    ap.add_argument("--only", default="",
                    help="comma substrings: ling,codex,analyst,paper")
    ap.add_argument("--no-dual", dest="dual", action="store_false",
                    default=True,
                    help="legacy behavior: rewrite the code field only and "
                         "DELETE rows with unparseable R code (default: dual "
                         "mode keeps code_original + normalized flag and "
                         "retains unparseable rows with dropped_reason)")
    args = ap.parse_args()
    sel = [s for s in args.only.split(",") if s]

    report = {}
    for path, mode in TARGETS:
        if sel and not any(s in path.name or s in str(path.parent) for s in sel):
            continue
        print(f"normalizing {path} [{mode}] "
              f"{'dual' if args.dual else 'legacy'} ...", flush=True)
        st = normalize_file(path, mode, args.workers, dual=args.dual)
        report[str(path)] = st
        print(json.dumps({k: v for k, v in st.items() if k != "diffs"},
                         indent=1))
        for d in st.get("diffs", []):
            print(f"--- diff sample row {d['row']} ---\n{d['diff']}")
    print("\n===== normalization summary =====")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "diffs"}
                      for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
