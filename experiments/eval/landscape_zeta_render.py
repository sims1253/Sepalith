"""Faithful Python port of Zed's V0318 "Zeta 2.1" prompt format.

Ported line-by-line from the PRODUCTION Rust source (the ground truth the
user pointed at), NOT from our legacy run_eval.render_zeta2_1:

  crates/zeta_prompt/src/zeta_prompt.rs      (assembly, suffix/edit-history
                                              sections, seed_coder constants)
  crates/zeta_prompt/src/multi_region.rs     (marker offsets, marker writer,
                                              output span applier)

ZetaFormat::V0318SeedMultiRegions carries the serde alias "Zeta2.1" — it is
the format the zeta-2.1 model was trained on. Key facts from the source:

  prompt  = suffix_section + FIM_PREFIX + edit_history_section(+\n)
            + cursor_prefix + FIM_MIDDLE
  suffix_section = FIM_SUFFIX + context[editable_end:] + (\n if needed)
  edit_history   = "<filename>edit_history\n" + events, each
                   "--- a{path}\n+++ b{path}\n" + diff  (write_event; the
                   prose "User edited" line and ```diff fences are OUR
                   zeta2 additions and do NOT exist in the Rust source)
  cursor_prefix  = "<filename>{path}\n" + context[..editable_start]
                   + write_editable_with_markers_v0318(...) + (\n if needed)
  the editable text is partitioned into blocks of 6..16 lines
  (compute_marker_offsets_v0318; blank-line-preferred boundaries, nudged up
  to 5 lines to a "good start" — a non-blank line that is not a structural
  tail like '}', ')', ']', 'return;', 'end'), and every block is wrapped in
  numbered markers: <|marker_1|>block<|marker_2|>block...<|marker_N|>.
  <|user_cursor|> is inserted at the cursor byte offset inside its block.

  OUTPUT contract (apply_marker_span_v0318): the model emits
  <|marker_i|>content<|marker_j|> (intermediate tags allowed and stripped);
  the span old[start_byte..end_byte] is replaced by the concatenated
  content. A repeated marker (<|marker_k|><|marker_k|>) means NO EDIT.
  Stop token / EOS: "<[end▁of▁sentence]>" (V0318_END_MARKER).

Deltas vs our legacy run_eval.render_zeta2_1 (documented in the landscape
report): the legacy render wrapped ONLY region_old in markers, kept the
merge-marker-era stripped edit history (no --- a/+++ b headers), and had no
block partitioning — a single-region simplification, not the training
format. This module is the faithful port used for the zeta-2.1 benchmark
leg; the mapping from our scenario rows is:

  context       = prefix + region_old lines (scenario rows carry no suffix;
                  edit_row() setdefaults suffix=[] for the same reason)
  editable      = the whole context (rows are far below the 350-editable-
                  token window, so the entire excerpt is editable — the
                  same situation as a small file at the cursor in Zed)
  cursor offset = end of region_old[cursor_idx] when the index is in range
                  (mirroring run_eval.with_cursor appending the marker to
                  that line), else the end of the last region_old line
  edit history  = the one BufferChange event our rows carry: the hunk body
                  of event_diff with the Rust write_event headers
"""
from __future__ import annotations

import difflib
import re

# --- seed_coder dialect constants (zeta_prompt.rs:4150-4153) -----------------
FIM_SUFFIX = "<[fim-suffix]>"
FIM_PREFIX = "<[fim-prefix]>"
FIM_MIDDLE = "<[fim-middle]>"
FILE_MARKER = "<filename>"

# --- multi_region.rs constants ----------------------------------------------
MARKER_TAG_PREFIX = "<|marker_"
MARKER_TAG_SUFFIX = "|>"
V0318_MIN_BLOCK_LINES = 6
V0318_MAX_BLOCK_LINES = 16
MAX_NUDGE_LINES = 5
V0318_END_MARKER = "<[end▁of▁sentence]>"
CURSOR_MARKER = "<|user_cursor|>"

_TAG_RE = re.compile(r"<\|marker_([0-9]+)\|>")


def marker_tag(number: int) -> str:
    return f"{MARKER_TAG_PREFIX}{number}{MARKER_TAG_SUFFIX}"


# --- block boundaries (multi_region.rs collect_line_info / good starts) ------
def _is_structural_tail(trimmed: str) -> bool:
    if trimmed.startswith(("}", "]", ")")):
        return True
    return trimmed.rstrip(";") in ("break", "continue", "return", "throw", "end")


def _collect_line_info(text: str):
    """[(start_byte, is_blank, is_good_start)]; phantom trailing element of a
    text ending in \\n is dropped (Rust collect_line_info)."""
    lines, offset = [], 0
    for line in text.split("\n"):
        trimmed = line.strip()
        lines.append((offset, not trimmed,
                      bool(trimmed) and not _is_structural_tail(trimmed)))
        offset += len(line) + 1
    if text.endswith("\n") and len(lines) > 1:
        lines.pop()
    return lines


def _skip_to_good_start(lines, start: int):
    end = min(len(lines), start + MAX_NUDGE_LINES)
    for i in range(start, end):
        if lines[i][2]:
            return i
    return None


def compute_marker_offsets_v0318(editable_text: str) -> list[int]:
    """compute_marker_offsets_with_limits(text, 6, 16), byte offsets, starts
    at 0 and ends at len; interior offsets sit right after a newline."""
    if not editable_text:
        return [0, 0]
    lines = _collect_line_info(editable_text)
    offsets = [0]
    last_boundary_line = 0
    i = 0
    while i < len(lines):
        gap = i - last_boundary_line
        # blank-line split: non-blank line after blank line(s), enough lines
        if gap >= V0318_MIN_BLOCK_LINES and not lines[i][1] and i > 0 \
                and lines[i - 1][1]:
            nudged = _skip_to_good_start(lines, i)
            target = i if lines[i][2] else (nudged if nudged is not None else i)
            if len(lines) - target >= V0318_MIN_BLOCK_LINES \
                    and lines[target][0] > offsets[-1]:
                offsets.append(lines[target][0])
                last_boundary_line = target
                i = target + 1
                continue
        # hard cap: too many lines without a split
        if gap >= V0318_MAX_BLOCK_LINES:
            t = _skip_to_good_start(lines, i)
            target = t if t is not None else i
            if lines[target][0] > offsets[-1]:
                offsets.append(lines[target][0])
                last_boundary_line = target
                i = target + 1
                continue
        i += 1
    end = len(editable_text)
    if offsets[-1] != end:
        offsets.append(end)
    return offsets


def write_editable_with_markers_v0318(editable_text: str,
                                      cursor_offset_in_editable: int) -> str:
    """write_editable_with_markers_impl + marker_tag(i+1): tag, then block,
    no separator (blocks end in \\n at interior boundaries, so markers land
    at line starts; the cursor marker is spliced at its byte offset)."""
    marker_offsets = compute_marker_offsets_v0318(editable_text)
    out, cursor_placed = [], False
    for i, offset in enumerate(marker_offsets):
        out.append(marker_tag(i + 1))
        if i + 1 < len(marker_offsets):
            next_offset = marker_offsets[i + 1]
            block = editable_text[offset:next_offset]
            if not cursor_placed and offset <= cursor_offset_in_editable \
                    <= next_offset:
                cursor_placed = True
                c = cursor_offset_in_editable - offset
                out.append(block[:c] + CURSOR_MARKER + block[c:])
            else:
                out.append(block)
    return "".join(out)


# --- prompt sections (zeta_prompt.rs seed_coder + build_v0318_cursor_prefix) -
def build_suffix_section(context: str, editable_range) -> str:
    section = FIM_SUFFIX + context[editable_range[1]:]
    return section if section.endswith("\n") else section + "\n"


def unix_path(path: str) -> str:
    """write_path_as_unix_str over our relative paths -> '/R/check.R' style
    (Zed feeds absolute paths; this is the closest faithful mapping)."""
    return "/" + path.lstrip("/")


def diff_body_from_event_diff(event_diff: str) -> str:
    """Our scenario rows' event_diff carries a 'User edited "path":' prose
    line and ```diff fences (zeta1 legacy). The Rust write_event uses neither
    — strip both and keep the hunk body."""
    lines = (event_diff or "").splitlines()
    if lines and lines[0].startswith("User edited"):
        lines = lines[1:]
    lines = [l for l in lines if l.strip() not in ("```diff", "```")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def format_edit_history(path: str, diff_body: str) -> str:
    """format_edit_history_within_budget + write_event for ONE BufferChange:
    header + '--- a{p}\\n+++ b{p}\\n' + diff. diff_body is the hunk body of
    our event_diff (prose line and fences already stripped)."""
    if not diff_body.strip():
        return ""
    header = f"{FILE_MARKER}edit_history\n"
    return header + f"--- a{unix_path(path)}\n+++ b{unix_path(path)}\n{diff_body}"


def cursor_offset_for(prefix_lines, region_old, cursor_idx) -> int:
    """Byte offset of the cursor: end of region_old[cursor_idx] when the
    index is in range (run_eval.with_cursor semantics — the marker is
    appended to that line), else the end of the last region_old line."""
    base = sum(len(l) + 1 for l in prefix_lines)
    if 0 <= cursor_idx < len(region_old):
        return base + sum(len(l) + 1 for l in region_old[:cursor_idx]) \
            + len(region_old[cursor_idx])
    if not region_old:
        return base
    return base + sum(len(l) + 1 for l in region_old) - 1


def build_v0318_prompt(prefix_lines, region_old, cursor_idx, path, diff_body):
    """Full V0318 prompt for one scenario row. Returns (prompt, context,
    editable_text, cursor_offset) — the last three feed the output parser.
    The context carries a trailing newline: Zed excerpts end with one (the
    official sample.prompt's final marker sits on its own line)."""
    context = "\n".join(list(prefix_lines) + list(region_old)) + "\n"
    editable_range = (0, len(context))   # whole small excerpt is editable
    cursor = cursor_offset_for(prefix_lines, region_old, cursor_idx)

    suffix_section = build_suffix_section(context, editable_range)
    history = format_edit_history(path, diff_body)

    cursor_prefix = f"{FILE_MARKER}{unix_path(path)}\n" + context[:editable_range[0]]
    cursor_prefix += write_editable_with_markers_v0318(
        context[editable_range[0]:editable_range[1]],
        cursor - editable_range[0])
    if not cursor_prefix.endswith("\n"):
        cursor_prefix += "\n"

    prompt = suffix_section + FIM_PREFIX + history
    if history:
        prompt += "\n"
    prompt += cursor_prefix + FIM_MIDDLE
    return prompt, context, context[editable_range[0]:editable_range[1]], cursor


# --- output contract (apply_marker_span_v0318) --------------------------------
def apply_marker_span_v0318(old_editable: str, output: str):
    """Returns (ok, new_editable_or_error). Faithful port: strip the end
    marker, collect tags, repeated tag = no edit, span content excludes all
    intermediate tags, 1-indexed marker numbers resolved against the block
    offsets of the OLD editable text."""
    if output.endswith(V0318_END_MARKER):
        output = output[: -len(V0318_END_MARKER)]
    tags = [(m.start(), m.end(), int(m.group(1)))
            for m in _TAG_RE.finditer(output)]
    if not tags:
        return False, "no marker tags found in output"
    if len(tags) == 1:
        return False, "only one marker tag found in output, expected at least two"
    start_val, end_val = tags[0][2], tags[-1][2]
    if start_val == end_val:
        return True, old_editable          # no-edit sentinel
    offsets = compute_marker_offsets_v0318(old_editable)
    if not (1 <= start_val <= len(offsets)) or not (1 <= end_val <= len(offsets)):
        return False, f"marker number out of range ({start_val}..{end_val})"
    start_byte, end_byte = offsets[start_val - 1], offsets[end_val - 1]
    if start_byte > end_byte:
        return False, "start marker must come before end marker"
    content = "".join(output[tags[i][1]:tags[i + 1][0]]
                      for i in range(len(tags) - 1))
    return True, old_editable[:start_byte] + content + old_editable[end_byte:]


def extract_region_lines(old_editable: str, new_editable: str,
                         region_old_lines: list[str], prefix_len: int):
    """Predicted region_new: line-diff the rewritten editable against the old
    one and pull out what happened to region_old's line range. The editable
    text is prefix + region_old, so the region is the line run at
    [prefix_len, prefix_len + len(region_old)). None = unmappable (error)."""
    old_lines = old_editable.split("\n")
    new_lines = new_editable.split("\n")
    i0, i1 = prefix_len, prefix_len + len(region_old_lines)
    if i1 > len(old_lines) or old_lines[i0:i1] != list(region_old_lines):
        return None
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    out = []
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if a2 <= i0 or a1 >= i1:
            continue                       # opcode outside the region
        if tag == "equal":
            out.extend(old_lines[max(a1, i0):min(a2, i1)])
        else:                              # replace/delete/insert overlap
            out.extend(new_lines[b1:b2])
    return out
