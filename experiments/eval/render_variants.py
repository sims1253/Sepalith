#!/usr/bin/env python3
"""H3-S0 render variants: the surviving candidate renders + zero-shot
format-fail scorer (docs/research/2026-09-09-h3s0-candidate-matrix.md).

Analysis-only module: pure string rendering/parsing plus byte-level prefix
stability measurement. No serving, no GPU, no queue side effects.

Variants (matrix §1.1; axis definitions §1):
  v10_psmtail_merge     PSM-T + MV1 + HP3 + CE2   primary candidate
  v11_psmtail_merge_empty  PSM-T + MV1 + HP3 + CE1   V10 twin (empty region)
  v05_psm_merge         PSM   + MV1 + HP2 + CE2   primary candidate
  v06_psm_merge_empty   PSM   + MV1 + HP2 + CE1   V05 twin
  v14_psm_merge_nohist  PSM   + MV1 + HP0 + CE2   history-slot isolation arm
  v13_zeta1_alpaca      history-first alpaca, zeta1 tags   must-FAIL control
  zeta2                 SPM   + MV1 + HP1 + CE2   incumbent reference (imported
                                                byte-exact from run_eval)

Layouts (MV1 marker multiset byte-identical to zeta2, order swapped; the
matrix §1.2c "delta 0 structural bytes" row pinned these choices):

  zeta2 (SPM):  <[fim-suffix]> suffix / <[fim-prefix]><filename>edit_history
                hist "" / <filename>path prefix / CURRENT region ======= /
                <[fim-middle]>
  v05  (PSM):   <filename>path prefix / hist block (HP2) / CURRENT region
                ======= / <[fim-suffix]> suffix / <[fim-middle]>
  v10  (PSM-T): <filename>path prefix / <[fim-suffix]> suffix / hist block
                (HP3: suffix -> history -> region) / CURRENT region ======= /
                <[fim-middle]>
  v14  (PSM):   v05 with the history block dropped entirely (HP0)
  CE1 (v06/v11): the region is the bare cursor marker line <|user_cursor|>
                (assemble_sft_v2.comment_to_code_row convention); region_old
                and cursor_idx are not encoded (the encoding is
                keystroke-invariant by construction).
  v13:          run_eval.render_zeta1, imported (registry ancestor).

Conventions inherited byte-exact from run_eval/assemble_sft_v2: history
normalization (strip ```diff fences, drop the "User edited" header, pop
leading blanks), history header ALWAYS emitted when the slot exists (empty
history = header with no body, as render_zeta2 does), history body followed
by one blank line, "\n".join of all parts, target =
"\n".join(region_new).rstrip() + "\n>>>>>>> UPDATED".

Parsers: parse_merge mirrors run_eval.parse_pred's zeta2 branch exactly
(cut at >>>>>>> UPDATED, strip a leading ``` fence, strip cursor markers,
norm) so K1 numbers stay comparable with the incumbent battery; parse_zeta1
mirrors the zeta1 branch (None = format failure when the start tag is
missing). parse_prompt is the exact prompt-level inverse: it recovers the
canonical example from a rendered prompt (up to the information the cursor
encoding actually carries; the single deliberate ambiguity is that a CE2
region of exactly one empty cursor line is byte-identical to CE1 and is
parsed as CE1).

format_fail(text, name=None, stopped=False) is the zero-shot compliance
check: fails on empty output, no marker evidence (stop-free text), an
unparseable structured output, or an empty parsed region. name=None uses
the shared merge completion contract (all MV1 variants and zeta2 have the
identical completion contract); stopped=True marks text returned by a
server whose stop string already consumed the terminator, in which case
the no-markers rule is skipped (a clean region needs no in-text markers).

Prefix stability (the K0 granularity floor, ~512-tok n_batch reuse
granules): keystroke_pair() builds two successive states of one in-progress
edit; prefix_stability() measures the common byte prefix of the two prompts
and estimates tokens as bytes/4.0 (the BPE mean where both cache_bench house
calibrations meet: 37 B/line, 9.3 tok/line). Event kinds: "typing" (insert before cursor; the
post-cursor text is unchanged -- matrix arm a), "cursor_advance" (cursor
moves right over a character: the partial grows AND the suffix head loses
its first byte -- the m2/backspace class where the incumbent pays 100%),
"edit_event" (a debounced edit near the cursor: the history ring rotates
and a prefix line changes -- the K2 ordering discriminator).

Usage:
  python3 render_variants.py --smoke          # 3 fixtures x all variants
  python3 render_variants.py --smoke --json   # machine-readable lines only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_eval import norm, render_zeta1, render_zeta2  # noqa: E402

# --- MV1 marker vocabulary (byte-exact zeta2 set, matrix §0.1) ---------------
FIM_SUFFIX = "<[fim-suffix]>"
FIM_MIDDLE = "<[fim-middle]>"
HIST_HEADER = "<[fim-prefix]><filename>edit_history"
REGION_OPEN = "<<<<<<< CURRENT"
REGION_CLOSE = "======="
CURSOR2 = "<|user_cursor|>"
TERMINATOR = ">>>>>>> UPDATED"
FILENAME_TAG = "<filename>"          # prefix header: "<filename>{path}"
# zeta1 vocabulary (V13 control only)
START1 = "<|editable_region_start|>"
END1 = "<|editable_region_end|>"
CURSOR1 = "<|user_cursor_is_here|>"

# --- K0 measurement constants (cache_bench house means) ----------------------
K0_FLOOR_TOK = 512                   # reuse granularity = n_batch (~512 tok)
TOK_PER_LINE = 9.3                   # cache_bench house mean
BYTES_PER_TOK = 4.0                  # conservative BPE mean for R code
CTX_BUDGET_TOK = 2048                # the canonical 2K-ctx request


# --- shared pieces ------------------------------------------------------------
def _with_cursor(lines, idx, marker):
    """run_eval.with_cursor, guarded against missing/invalid cursor_idx."""
    out = list(lines)
    if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(out):
        out[idx] = out[idx] + marker
    return out


def hist_lines(ex):
    """Exact render_zeta2 history normalization (fences/header stripped)."""
    ev = ex.get("event_diff") or ""
    for tag in ("```diff\n", "```"):
        ev = ev.replace(tag, "")
    ev_lines = list(ev.splitlines())
    if ev_lines and ev_lines[0].startswith("User edited"):
        ev_lines = ev_lines[1:]
    while ev_lines and not ev_lines[0].strip():
        ev_lines.pop(0)
    return ev_lines


def _history_block(ex):
    """HP1/HP2/HP3 block: header always, body + one blank line when present."""
    ev = hist_lines(ex)
    return [HIST_HEADER] + ev + [""] if ev else [HIST_HEADER]


def _region(ex, cursor_enc):
    if cursor_enc == "CE1":
        return [CURSOR2]
    return _with_cursor(ex.get("region_old") or [], ex.get("cursor_idx"), CURSOR2)


def _render_merge(ex, suffix_after_region, cursor_enc, history_slot):
    """PSM / PSM-T core. suffix_after_region: PSM puts the suffix block after
    the closed region (V05/V06/V14); PSM-T keeps the incumbent's region-last
    tail and moves the suffix to just after the prefix (V10/V11)."""
    prefix = list(ex.get("prefix") or [])
    suffix = list(ex.get("suffix") or [])
    region = _region(ex, cursor_enc)
    hist = [] if history_slot == "HP0" else _history_block(ex)
    head = [FILENAME_TAG + str(ex.get("path", ""))] + prefix
    if suffix_after_region:
        parts = head + hist + [REGION_OPEN] + region
        parts += [REGION_CLOSE, FIM_SUFFIX] + suffix + [FIM_MIDDLE]
    else:
        head = head + [FIM_SUFFIX] + suffix
        parts = head + hist + [REGION_OPEN] + region + [REGION_CLOSE, FIM_MIDDLE]
    return "\n".join(parts)


def render_v05(ex):
    """PSM + MV1 + HP2 + CE2 (the document's literal prefix-first shape)."""
    return _render_merge(ex, suffix_after_region=True, cursor_enc="CE2",
                         history_slot="HP2")


def render_v06(ex):
    """PSM + MV1 + HP2 + CE1 (V05 empty-region twin)."""
    return _render_merge(ex, suffix_after_region=True, cursor_enc="CE1",
                         history_slot="HP2")


def render_v10(ex):
    """PSM-T + MV1 + HP3 + CE2: prefix head, suffix mid, region last."""
    return _render_merge(ex, suffix_after_region=False, cursor_enc="CE2",
                         history_slot="HP3")


def render_v11(ex):
    """PSM-T + MV1 + HP3 + CE1 (V10 empty-region twin)."""
    return _render_merge(ex, suffix_after_region=False, cursor_enc="CE1",
                         history_slot="HP3")


def render_v14(ex):
    """PSM + MV1 + HP0 + CE2: history-slot isolation arm (no history block)."""
    return _render_merge(ex, suffix_after_region=True, cursor_enc="CE2",
                         history_slot="HP0")


# --- completion parsers (the parse_pred companions) ---------------------------
def parse_merge(text):
    """run_eval.parse_pred zeta2 branch, byte-exact conventions. All MV1
    variants share the completion contract region_new + '\\n>>>>>>> UPDATED'."""
    if text is None:
        return None
    t = text
    if ">>>>>>>" in t:
        t = t.split(">>>>>>> UPDATED", 1)[0].split(">>>>>>>", 1)[0]
    if t.startswith("```"):
        t = t[3:]
    t = t.replace(CURSOR2, "").replace(CURSOR1, "")
    return norm(t.splitlines())


def parse_zeta1(text):
    """run_eval.parse_pred zeta1 branch: None when the start tag is missing
    (the format failure the must-FAIL control is expected to produce)."""
    if text is None:
        return None
    t = text
    if START1 in t:
        t = t.split(START1, 1)[1]
    else:
        return None
    if END1 in t:
        t = t.split(END1, 1)[0]
    t = t.replace(CURSOR1, "").replace(CURSOR2, "")
    return norm(t.splitlines())


def _compat(render):
    """assemble_sft_v2 row defaulting (scenario rows carry no suffix key):
    byte-exact on rows that carry the keys, robust on rows that do not."""
    def wrapped(ex):
        ex = dict(ex)
        for k in ("suffix", "prefix", "region_old"):
            ex.setdefault(k, [])
        return render(ex)
    return wrapped


VARIANTS = {
    "v10_psmtail_merge": (render_v10, parse_merge),
    "v11_psmtail_merge_empty": (render_v11, parse_merge),
    "v05_psm_merge": (render_v05, parse_merge),
    "v06_psm_merge_empty": (render_v06, parse_merge),
    "v14_psm_merge_nohist": (render_v14, parse_merge),
    "v13_zeta1_alpaca": (_compat(render_zeta1), parse_zeta1),
    "zeta2": (_compat(render_zeta2), parse_merge),   # incumbent reference
}

# K4 stop strings per variant (the deterministic terminator contract).
STOP_STRINGS = {n: ([END1] if n == "v13_zeta1_alpaca" else [TERMINATOR])
                for n in VARIANTS}

# marker evidence that proves format engagement in stop-free raw text
REQUIRED_MARKERS = {n: ((END1,) if n == "v13_zeta1_alpaca" else (">>>>>>>",))
                    for n in VARIANTS}

VARIANT_INFO = {  # axes per matrix §1.1 + analysis annotations
    "v10_psmtail_merge": dict(ordering="PSM-T", vocab="MV1", history="HP3",
                              cursor="CE2", prefix_dominant=True,
                              note="primary candidate; generation tail "
                                   "byte-identical to zeta2"),
    "v11_psmtail_merge_empty": dict(ordering="PSM-T", vocab="MV1", history="HP3",
                                    cursor="CE1", prefix_dominant=True,
                                    note="V10 twin; noopFP hazard (S1 re-check)"),
    "v05_psm_merge": dict(ordering="PSM", vocab="MV1", history="HP2",
                          cursor="CE2", prefix_dominant=True,
                          note="primary candidate; fim-suffix below ======= "
                               "(stronger role-swap OOD)"),
    "v06_psm_merge_empty": dict(ordering="PSM", vocab="MV1", history="HP2",
                                cursor="CE1", prefix_dominant=True,
                                note="V05 twin"),
    "v14_psm_merge_nohist": dict(ordering="PSM", vocab="MV1", history="HP0",
                                 cursor="CE2", prefix_dominant=True,
                                 note="history-slot isolation arm"),
    "v13_zeta1_alpaca": dict(ordering="alpaca-history-first", vocab="zeta1",
                             history="header-first", cursor="region-tags",
                             prefix_dominant=False,
                             note="must-FAIL control (K1 victim; K2(i)/K3 fire)"),
    "zeta2": dict(ordering="SPM", vocab="MV1", history="HP1", cursor="CE2",
                  prefix_dominant=False,
                  note="incumbent reference, imported byte-exact"),
}

# declared section order (for the K2 paper rule; cursor zone last = good)
SECTION_ORDER = {
    "zeta2": ("suffix", "history", "prefix", "region"),
    "v05_psm_merge": ("prefix", "history", "region", "suffix"),
    "v06_psm_merge_empty": ("prefix", "history", "region", "suffix"),
    "v14_psm_merge_nohist": ("prefix", "region", "suffix"),
    "v10_psmtail_merge": ("prefix", "suffix", "history", "region"),
    "v11_psmtail_merge_empty": ("prefix", "suffix", "history", "region"),
    "v13_zeta1_alpaca": ("history", "prefix", "region", "suffix"),
}


def k2_verdict(name):
    """Matrix §3 K2 (pure paper rule) on the declared section order.
    (i) history must not precede the file prefix; (ii) the cursor region must
    not precede the history; (iii) the suffix must not sit at the prompt head.
    Returns (pass, [failed sub-rules])."""
    order = SECTION_ORDER[name]
    fails = []
    if "history" in order and order.index("history") < order.index("prefix"):
        fails.append("i:history_precedes_prefix")
    if "history" in order and order.index("region") < order.index("history"):
        fails.append("ii:region_precedes_history")
    if order[0] == "suffix":
        fails.append("iii:suffix_at_head")
    return (not fails), fails


# --- prompt-level exact inverse ------------------------------------------------
def _cursor_from_region(region_lines, marker):
    """(lines_without_marker, cursor_idx or None, encoding) for a region."""
    if region_lines == [CURSOR2]:
        return [], 0, "CE1"     # bare cursor marker line == the CE1 signature
    idx = None
    out = []
    for k, line in enumerate(region_lines):
        if idx is None and line.endswith(marker):
            idx = k
            out.append(line[: -len(marker)])
        else:
            out.append(line)
    return out, idx, "CE2"


def parse_prompt(prompt, name):
    """Exact inverse of the variant's render: recover the canonical example
    fields from the prompt string. Sections are recovered verbatim (no norm);
    history comes back in the already-normalized render form."""
    if name == "v13_zeta1_alpaca":
        _, rest = prompt.split("### User Edits:\n\n", 1)
        events, rest2 = rest.split("\n\n### User Excerpt:\n\n", 1)
        inp, _ = rest2.split("\n\n### Response:\n\n", 1)
        lines = inp.split("\n")
        path = lines[0][3:] if lines[0].startswith("```") else lines[0]
        i_s, i_e = lines.index(START1), lines.index(END1)
        prefix, region, suffix = lines[1:i_s], lines[i_s + 1:i_e], lines[i_e + 1:-1]
        region, cur, enc = _cursor_from_region(region, CURSOR1)
        if enc == "CE1":
            region, cur = list(region), cur  # zeta1 has no CE1; keep parsed form
        return dict(path=path, prefix=prefix, history=events.splitlines(),
                    suffix=suffix, region_old=region, cursor_idx=cur,
                    cursor_encoding="CE2", ordering="alpaca-history-first",
                    history_slot="header-first", vocab="zeta1")
    if name == "zeta2":
        lines = prompt.split("\n")
        io = lines.index(REGION_OPEN)
        ic = lines.index(REGION_CLOSE, io + 1)
        ih = lines.index(HIST_HEADER)
        suffix = lines[1:ih]
        rest = lines[ih + 1:io]
        k = next(i for i, l in enumerate(rest) if l.startswith(FILENAME_TAG))
        hist = rest[:k]
        if hist and hist[-1] == "":
            hist = hist[:-1]
        path, prefix = rest[k][len(FILENAME_TAG):], rest[k + 1:]
        region, cur, enc = _cursor_from_region(lines[io + 1:ic], CURSOR2)
        return dict(path=path, prefix=prefix, history=hist, suffix=suffix,
                    region_old=region, cursor_idx=cur, cursor_encoding=enc,
                    ordering="SPM", history_slot="HP1", vocab="MV1")
    # merge variants (v05/v06/v10/v11/v14)
    lines = prompt.split("\n")
    io, ic = lines.index(REGION_OPEN), lines.index(REGION_CLOSE)
    region, cur, enc = _cursor_from_region(lines[io + 1:ic], CURSOR2)
    info = VARIANT_INFO[name]
    suffix = []
    if info["ordering"] == "PSM":            # suffix block after the region
        tail = lines[ic + 1:]
        if not (tail and tail[0] == FIM_SUFFIX and tail[-1] == FIM_MIDDLE):
            raise ValueError(f"{name}: malformed tail {tail[:2]}...")
        suffix = tail[1:-1]
        head = lines[:io]
    else:                                    # PSM-T: suffix mid, bare tail
        if not (len(lines) > ic + 1 and lines[ic + 1] == FIM_MIDDLE and
                lines[-1] == FIM_MIDDLE):
            raise ValueError(f"{name}: malformed tail")
        head = lines[:io]
    path = head[0][len(FILENAME_TAG):] if head[0].startswith(FILENAME_TAG) else ""
    rest = head[1:]
    hist = []
    if info["history"] != "HP0":
        if HIST_HEADER not in rest:
            raise ValueError(f"{name}: missing history header")
        h = rest.index(HIST_HEADER)
        prefix, after = rest[:h], rest[h + 1:]
        if info["ordering"] == "PSM-T":
            f = prefix.index(FIM_SUFFIX)
            suffix = prefix[f + 1:]
            prefix = prefix[:f]
        if after and after[-1] == "":
            after = after[:-1]
        hist = after
    else:
        prefix = rest
    return dict(path=path, prefix=prefix, history=hist, suffix=suffix,
                region_old=region, cursor_idx=cur, cursor_encoding=enc,
                ordering=info["ordering"], history_slot=info["history"],
                vocab="MV1")


def canonical_example(ex, name):
    """The example dict parse_prompt(render(ex)) must return: the render's own
    view of the example (history normalized, CE1 collapses the region,
    invalid cursor_idx -> None)."""
    info = VARIANT_INFO[name]
    if name == "v13_zeta1_alpaca":
        region, cur, _ = _cursor_from_region(
            _with_cursor(ex.get("region_old") or [], ex.get("cursor_idx"),
                         CURSOR1), CURSOR1)
        return dict(path=ex.get("path", ""), prefix=list(ex.get("prefix") or []),
                    history=(ex.get("event_diff") or "").splitlines(),
                    suffix=list(ex.get("suffix") or []), region_old=region,
                    cursor_idx=cur, cursor_encoding="CE2",
                    ordering="alpaca-history-first", history_slot="header-first",
                    vocab="zeta1")
    enc = info["cursor"]
    region, cur, parsed_enc = _cursor_from_region(_region(ex, enc), CURSOR2)
    # a CE2 single empty cursor line renders byte-identically to CE1
    enc = parsed_enc if enc == "CE2" else enc
    return dict(path=ex.get("path", ""), prefix=list(ex.get("prefix") or []),
                history=([] if info["history"] == "HP0" else hist_lines(ex)),
                suffix=list(ex.get("suffix") or []), region_old=region,
                cursor_idx=cur, cursor_encoding=enc, ordering=info["ordering"],
                history_slot=info["history"], vocab="MV1")


def completion(ex, name):
    """The canonical compliant completion (what a perfect model emits)."""
    body = "\n".join(ex["region_new"]).rstrip()
    if name == "v13_zeta1_alpaca":
        return START1 + body + END1   # tags glue to the body: parse_pred's
        # zeta1 branch keeps leading blank lines, so newlines here would
        # survive norm() and break the round-trip identity
    return body + "\n" + TERMINATOR


# --- zero-shot format-fail scorer ----------------------------------------------
def format_fail(text, name=None, stopped=False):
    """(failed: bool, reason: str) for a raw zero-shot completion.

    Fails on: "empty" (no output at all), "no_markers" (stop-free text shows
    none of the variant's required marker vocabulary -- the gemma-class
    never-emits-the-terminator failure), "unparseable" (the variant parser
    returns None), "empty_region" (parses to zero lines).

    name=None applies the shared merge completion contract (v05/v06/v10/v11/
    v14 and zeta2 are identical on the completion side). stopped=True marks
    server text whose stop string already consumed the terminator: the
    no-markers rule is skipped because a clean stop-terminated region carries
    no in-text markers by construction.
    """
    if name is not None and name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; have {sorted(VARIANTS)}")
    if text is None or not str(text).strip():
        return True, "empty"
    parse = VARIANTS[name][1] if name is not None else parse_merge
    need = REQUIRED_MARKERS[name] if name is not None else (">>>>>>>",)
    if not stopped and not any(m in text for m in need):
        return True, "no_markers"
    pred = parse(text)
    if pred is None:
        return True, "unparseable"
    if not pred:
        return True, "empty_region"
    return False, "ok"


# --- prefix-stability measurement (the cache property / K0 floor) --------------
def common_prefix_len(a, b):
    """Length of the shared leading bytes of two prompts (what a byte-prefix
    KV cache can reuse, modulo the ~512-token n_batch granularity)."""
    n = min(len(a), len(b))
    i, chunk = 0, 4096
    while i < n:
        j = min(i + chunk, n)
        if a[i:j] != b[i:j]:
            for k in range(i, j):
                if a[k] != b[k]:
                    return k
        i = j
    return n


def est_tokens(text):
    """Token estimate: bytes / 4.0 (conservative Qwen-class BPE mean for R
    code). Cross-check: the cache_bench house means agree at the corpus
    average line -- 37 B/line over 9.3 tok/line is 4.0 B/tok -- so the two
    house calibrations meet here; byte-based is kept because the
    lines*9.3 model breaks on marker-heavy single-line text."""
    if not text:
        return 0.0
    return len(text) / BYTES_PER_TOK


def cursor_zone_offset(prompt, name):
    """Byte offset where the cursor/region zone begins (the REGION_OPEN line
    for merge variants and zeta2, the START1 tag for the alpaca control)."""
    return prompt.find(START1 if name == "v13_zeta1_alpaca" else REGION_OPEN)


def keystroke_pair(ex, kind="typing", ch="e"):
    """Two successive states of one in-progress edit (mid-line cursor).

    Requires the midtyping geometry: region_old ends with the typed partial
    and suffix[0] is the post-cursor remainder of that line.
      typing: insert before the cursor -- partial grows, post-cursor text
        unchanged (matrix arm a; every ordering keeps its head here).
      cursor_advance: the cursor moves right over `ch` -- the partial grows
        AND the suffix head loses its first byte (the m2/backspace class
        where the incumbent's suffix-first head pays 100% re-prefill).
      edit_event: a debounced edit near the cursor -- the history ring
        rotates (first hunk dropped, a new one appended) and the prefix line
        `near_lines` above the region changes (the K2 discriminator).
    """
    ro = list(ex.get("region_old") or [])
    suf = list(ex.get("suffix") or [])
    if not ro or not suf or not suf[0]:
        raise ValueError("keystroke_pair needs a mid-line cursor geometry: "
                         "non-empty region_old and a non-empty suffix head")
    b = dict(ex)
    if kind in ("typing", "cursor_advance"):
        b["region_old"] = ro[:-1] + [ro[-1] + ch]
        if isinstance(ex.get("cursor_idx"), int):
            b["cursor_idx"] = len(b["region_old"]) - 1
        if kind == "cursor_advance":
            b["suffix"] = [suf[0][1:]] + list(suf[1:])
    elif kind == "edit_event":
        near = min(ex.get("_near_lines", 30), len(ex.get("prefix") or []) - 1)
        prefix = list(ex.get("prefix") or [])
        prefix[-(near + 1)] = prefix[-(near + 1)] + "  # touched"
        b["prefix"] = prefix
        b["event_diff"] = _rotate_history(ex.get("event_diff") or "")
    else:
        raise ValueError(f"unknown pair kind {kind!r}")
    return dict(ex), b


def _rotate_history(ev):
    """Ring-buffer rotation: drop the oldest hunk, append a fresh one."""
    lines = ev.splitlines()
    hunks, cur = [], []
    for line in lines:
        if line.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        hunks.append(cur)
    kept = hunks[1:] if len(hunks) > 1 else hunks
    return "\n".join("\n".join(h) for h in kept) + (
        "\n@@ -%d,3 +%d,4 @@\n-  old_line <- 1\n+  old_line <- 2\n" % (41, 41))


def prefix_stability(ex, name, kind="typing", ch="e"):
    """Common-prefix measurement across one event, per variant.

    Returns byte length, conservative token estimate, whether the estimate
    clears the 512-token K0 reuse floor, whether everything before the cursor
    zone is byte-identical (the strict cache property), and the estimated
    reusable tokens in whole 512-token granules (the K0 floor quantizes
    reuse; a 448-token stable head reuses nothing).
    """
    a, b = keystroke_pair(ex, kind=kind, ch=ch)
    render = VARIANTS[name][0]
    pa, pb = render(a), render(b)
    n = common_prefix_len(pa, pb)
    tok = est_tokens(pa[:n])
    zone = cursor_zone_offset(pa, name)
    return dict(variant=name, kind=kind, stable_bytes=n,
                stable_tok_est=round(tok, 1),
                reusable_tok_est=int(tok // K0_FLOOR_TOK) * K0_FLOOR_TOK,
                over_k0_floor=bool(tok > K0_FLOOR_TOK),
                prezone_identical=bool(pa[:zone] == pb[:zone]),
                prezone_bytes=zone)


def structural_bytes(prompt, ex, name):
    """Non-content prompt bytes: prompt size minus the content the render
    actually carries (matrix §4.3). Sections are taken from the variant's
    canonical view, so CE1 (region collapsed to the marker) and HP0 (history
    dropped) do not count unrendered example content; marker vocabulary,
    section headers and join newlines count as structural."""
    c = canonical_example(ex, name)
    parts = [c["prefix"], c["suffix"], c["region_old"], c["history"]]
    content = "\n".join("\n".join(p) for p in parts if p)
    return len(prompt) - len(content)


# --- fixtures (deterministic, synthetic, no NAS) --------------------------------
_R_BODY = [
    "  x <- x[order(group, x)]",
    "  if (anyNA(x)) x <- safena(x, verbose = FALSE)",
    "  out <- list(a = x, b = group, call = match.call())",
    "  attr(out, \"sorted\") <- TRUE",
    "  class(out) <- c(\"dt_result\", \"list\")",
    "  for (i in seq_along(x)) out[[i]] <- x[[i]] * scale_k",
    "  ans <- forderv(x, by = by, retGrp = TRUE, sort = FALSE)",
    "  if (isTRUE(check)) ans <- check_int(ans)",
]


def _r_lines(n, offset=0):
    return ["# line %d" % (offset + i) if i % 9 == 0 else _R_BODY[(offset + i) % len(_R_BODY)]
            for i in range(n)]


_HISTORY_2HUNK = """```diff
User edited R/models.R:
@@ -12,3 +12,3 @@
-  fit <- glm(y ~ x, data = d)
+  fit <- glm(y ~ x + z, data = d)
   se <- sqrt(diag(vcov(fit)))
@@ -40,4 +40,4 @@
-  pred <- predict(fit, newdata = new)
+  pred <- predict(fit, newdata = new, type = "response")
   upr <- pred + 1.96 * se
```"""


def fixture_2k():
    """The canonical 2K-ctx request (matrix §2 geometry): 156-line prefix,
    2-hunk history, 6-line region, 48-line suffix, mid-line cursor."""
    partial, rest = "  ans <- m", "ean(v, na.rm = TRUE)"
    return dict(
        lang="r", repo="fixture/pkg", path="R/models.R",
        sha="f" * 40, is_test=True,
        prefix=_r_lines(156),
        region_old=_r_lines(5, 100) + [partial],
        region_new=[rest, "  ans <- check(ans)", "  out <- c(out, ans)"],
        cursor_idx=5,
        suffix=[rest] + _r_lines(47, 200),
        event_diff=_HISTORY_2HUNK,
    )


def fixture_small():
    """Tiny example with history, for exhaustive round-trips."""
    return dict(
        lang="r", repo="fixture/pkg", path="R/util.R",
        sha="a" * 40, is_test=True,
        prefix=["f <- function(x) {", "  # TODO: handle NA"],
        region_old=["  y <- mean(x)"], region_new=["  y <- mean(x, na.rm = TRUE)"],
        cursor_idx=0,
        suffix=["  y", "}", "", "g <- 2"],
        event_diff="```diff\nUser edited R/util.R:\n@@ -3,2 +3,2 @@\n-  y <- mean(x)\n+  y <- mean(x, na.rm = TRUE)\n```",
    )


def fixture_scenario_row():
    """Scenario-row shape: no suffix key, no event_diff (assemble_sft_v2
    setdefaults suffix to []; history-absence is in-distribution)."""
    return dict(
        lang="r", repo="fixture/pkg", path="R/prop.R",
        sha="b" * 40, is_test=True,
        prefix=_r_lines(8),
        region_old=["  res <- prop(x)"], region_new=["  res <- prop(x, tidy = TRUE)"],
        cursor_idx=0,
    )


def midtyping_state(ex, frac=0.5):
    """Deterministic make_midtyping-style transform (run_eval.make_midtyping
    uses a process-salted hash seed; this one is stable across runs): the
    region ends with a typed partial of the first changed line."""
    ro, rn = ex["region_old"], ex["region_new"]
    i = 0
    while i < min(len(ro), len(rn)) and ro[i] == rn[i]:
        i += 1
    first = rn[i] if i < len(rn) else ""
    cut = max(1, int(len(first) * frac)) if first else 0
    new = dict(ex)
    new["region_old"] = ro[:i] + [first[:cut]]
    new["cursor_idx"] = len(new["region_old"]) - 1
    new["region_new"] = [first[cut:]] + rn[i + 1:]
    return new


SMOKE_FIXTURES = (("2k-canonical", fixture_2k),
                  ("small", fixture_small),
                  ("midtyping", lambda: midtyping_state(fixture_small())))


# --- smoke ----------------------------------------------------------------------
def run_smoke(as_json=False):
    out = []
    for fixname, fix in SMOKE_FIXTURES:
        ex = fix()
        for name in VARIANTS:
            render, _ = VARIANTS[name]
            prompt = render(ex)
            sb = structural_bytes(prompt, ex, name)
            rec = dict(fixture=fixname, variant=name, chars=len(prompt),
                       tok_est=round(est_tokens(prompt), 1),
                       structural_bytes=sb,
                       structural_tok_est=round(sb / BYTES_PER_TOK, 1),
                       structural_pct_of_2k=round(
                           100.0 * (sb / BYTES_PER_TOK) / CTX_BUDGET_TOK, 2),
                       k2_pass=k2_verdict(name)[0],
                       stop=STOP_STRINGS[name])
            out.append(rec)
            if not as_json:
                print("%-14s %-24s %6d ch %7.1f tok  struct %3d B (%4.1f tok, "
                      "%4.1f%% of 2K)  k2=%s" %
                      (fixname, name, rec["chars"], rec["tok_est"], sb,
                       rec["structural_tok_est"], rec["structural_pct_of_2k"],
                       rec["k2_pass"]))
    ex2k = fixture_2k()
    stability = []
    for name in VARIANTS:
        row = dict(variant=name, prefix_dominant=VARIANT_INFO[name]["prefix_dominant"])
        for kind in ("typing", "cursor_advance", "edit_event"):
            m = prefix_stability(ex2k, name, kind=kind)
            row[kind] = dict(stable_tok=m["stable_tok_est"],
                             over_k0_floor=m["over_k0_floor"],
                             reusable_tok=m["reusable_tok_est"],
                             prezone_identical=m["prezone_identical"])
        stability.append(row)
        if not as_json:
            print("%-24s typing %6.1f tok (%s) | cursor_adv %6.1f tok (%s) | "
                  "edit_event %6.1f tok (%s) | prezone identical: typing=%s "
                  "cursor_adv=%s" %
                  (name,
                   row["typing"]["stable_tok"], row["typing"]["over_k0_floor"],
                   row["cursor_advance"]["stable_tok"],
                   row["cursor_advance"]["over_k0_floor"],
                   row["edit_event"]["stable_tok"],
                   row["edit_event"]["over_k0_floor"],
                   row["typing"]["prezone_identical"],
                   row["cursor_advance"]["prezone_identical"]))
    print(json.dumps(dict(smoke="h3s0-render-variants", rows=out,
                          prefix_stability_2k=stability,
                          k0_floor_tok=K0_FLOOR_TOK), indent=None))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="render the 3 fixtures in every variant; print "
                         "lengths + prefix-stability booleans (no server)")
    ap.add_argument("--json", action="store_true", help="JSON lines only")
    args = ap.parse_args(argv)
    if not args.smoke:
        ap.error("nothing to do: pass --smoke (this module is analysis-only; "
                 "serving lives in the S0 eval runners)")
    return run_smoke(as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
