// Editor state -> PRM03 context, and the recent-edit history provider. Pure
// (no vscode import) so scripts/check-prm03-context.ts runs it under node.
//
// The rules follow arm A's TRAIN distribution (DAT10 v4, 15,006 rows), where
// every context has scope_mode "off" and no diagnostics, retrieval or
// selected references, and the cursor always sits on the region's first line:
//
// * Region = the cursor line. A non-blank cursor line is a one-line region
//   with the cursor marker at the cursor column (TRAIN: 6,167 one-line
//   regions). A blank cursor line is an insertion point: an empty region with
//   no cursor marker and a zero-width range at the line start (TRAIN: 7,940
//   rows; there the insertion line is always truly empty and the source window
//   prices it as an empty line). An indented blank line (editor auto-indent)
//   renders the same way; applying the edit also replaces its whitespace.
// * Prefix and suffix come from the prm05-source-balanced-v1 window with the
//   6,000 UTF-16-unit budget the TRAIN rows were selected with.
// * History = the most recent committed line-level edit in the same file,
//   in the TRAIN `event_diff` spelling (`User edited "<path>":` + a diff hunk
//   without context lines). TRAIN rows carry 0 or 1 event (5 no-op rows carry
//   2), always in the same file and on other lines than the region.

import { createHash } from "node:crypto";
import { ProtocolError, SCHEMA_VERSION, mapWireToDocumentEol, renderPrompt, utf16ToCodepointColumn } from "./prm03.ts";
import type { HistoryEvent, OutputResult, PromptContext } from "./prm03.ts";
import { DEFAULT_SELECTION_UTF16_UNITS, selectSourceWindow } from "./prm03_select.ts";

/** Most recent edits rendered into the prompt (TRAIN carries at most one per edit row). */
export const MAX_HISTORY_EVENTS = 1;
/** Larger edits (pastes, reformat-all) are not rendered; TRAIN events have at most 28 lines and 1,167 characters. */
export const MAX_EVENT_LINES = 30;
export const MAX_EVENT_DIFF_CHARS = 1500;

export function sha256Text(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** LF/CRLF text -> logical lines (campaign_selection.logical_lines, without the lone-CR rejection). */
export function logicalLines(text: string): string[] {
  return text.split("\n").map(line => (line.endsWith("\r") ? line.slice(0, -1) : line));
}

/** A model-visible relative POSIX path (`R/foo.R`). */
export function promptPath(relativePath: string): string {
  const path = relativePath.split("\\").join("/").replace(/^\/+/, "").split("\x00").join("");
  return path || "untitled.R";
}

export interface EditorSnapshot {
  lines: string[]; // logical lines of the whole document
  eol: "lf" | "crlf";
  cursorLine: number;
  cursorUtf16: number;
  path: string; // model-visible relative POSIX path
  uri: string;
  version: number;
  contentSha256: string; // identity of the document text the context was captured from
  history: HistoryEvent[];
}

export interface EditorRequest {
  context: PromptContext;
  prompt: string;
  line: number; // document line the edit replaces
  lineText: string; // its text when captured
  cursorUtf16: number;
  /** Prefix lines dropped by the source budget (logging only). */
  omittedLines: number;
}

const BLANK = /^[ \t]*$/;

export function buildEditorRequest(snapshot: EditorSnapshot, budget = DEFAULT_SELECTION_UTF16_UNITS): EditorRequest {
  const { lines, cursorLine } = snapshot;
  if (cursorLine < 0 || cursorLine >= lines.length) throw new ProtocolError("cursor line is outside the document");
  const lineText = lines[cursorLine];
  const insertion = BLANK.test(lineText);
  const source = insertion ? lines.map((line, i) => (i === cursorLine ? "" : line)) : lines;
  const window = selectSourceWindow(source, cursorLine, cursorLine, budget);
  const range = {
    uri: snapshot.uri,
    document_version: snapshot.version,
    content_sha256: snapshot.contentSha256,
    start: { line: cursorLine, character: 0 },
    end: { line: cursorLine, character: insertion ? 0 : lineText.length },
  };
  const cursorUtf16 = Math.min(snapshot.cursorUtf16, lineText.length);
  const context: PromptContext = {
    schema_version: SCHEMA_VERSION,
    path: snapshot.path,
    prefix: window.prefix,
    selected_references: [],
    history: snapshot.history,
    diagnostics: [],
    retrieval: [],
    scope_mode: "off",
    scope_lines: [],
    suffix_lines: window.suffix,
    region_old: insertion ? [] : [lineText],
    cursor: insertion
      ? { region_line_index: -1, code_point_column: null, utf16_column: null }
      : { region_line_index: 0, code_point_column: utf16ToCodepointColumn(lineText, cursorUtf16), utf16_column: cursorUtf16 },
    replacement_range: range,
    document_eol: snapshot.eol,
  };
  return {
    context,
    prompt: renderPrompt(context),
    line: cursorLine,
    lineText,
    cursorUtf16,
    omittedLines: window.omissions.reduce((n, o) => n + o.end_line - o.start_line + 1, 0),
  };
}

// ---------------------------------------------------------------------------
// Presentation of an accepted edit
// ---------------------------------------------------------------------------

export type Presentation =
  | { kind: "none" }
  // pure insertion at the cursor: VS Code ghost text
  | { kind: "ghost"; line: number; character: number; text: string }
  // anything else replaces the whole cursor line (characters [0, endCharacter)); shown as a diff preview
  | { kind: "replace"; line: number; endCharacter: number; text: string; oldText: string };

/**
 * Map a parsed output to an editor change of the captured cursor line. The
 * protocol body replaces the region exactly (no whitespace trimming). For an
 * indented blank line the line's whitespace is replaced as well.
 */
export function presentEdit(request: EditorRequest, result: OutputResult): Presentation {
  if (result.status !== "accepted" || result.operation === null || result.operation === "no_op") return { kind: "none" };
  const eol = request.context.document_eol;
  const text = result.body_text;
  const { lineText, cursorUtf16, line } = request;
  if (text === lineText) return { kind: "none" };
  const before = lineText.slice(0, cursorUtf16);
  const after = lineText.slice(cursorUtf16);
  if (text.length > lineText.length && text.startsWith(before) && text.endsWith(after)) {
    return { kind: "ghost", line, character: cursorUtf16, text: mapWireToDocumentEol(text.slice(before.length, text.length - after.length), eol) };
  }
  return { kind: "replace", line, endCharacter: lineText.length, text: mapWireToDocumentEol(text, eol), oldText: lineText };
}

// ---------------------------------------------------------------------------
// Recent-edit history
// ---------------------------------------------------------------------------

interface Block { start: number; oldEnd: number; newEnd: number }

/** Changed line block between two documents: common prefix and suffix lines removed. */
export function changedBlock(before: readonly string[], after: readonly string[]): Block | null {
  let start = 0;
  const limit = Math.min(before.length, after.length);
  while (start < limit && before[start] === after[start]) start++;
  if (start === before.length && start === after.length) return null;
  let suffix = 0;
  while (suffix < limit - start && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) suffix++;
  return { start, oldEnd: before.length - suffix, newEnd: after.length - suffix };
}

/** The TRAIN `event_diff` spelling (sepalith.data.synth.cases.Event.event_diff). */
export function eventDiff(path: string, start: number, oldLines: readonly string[], newLines: readonly string[]): string {
  const line = start + 1;
  const header = oldLines.length === 1 && newLines.length === 1
    ? `@@ -${line} +${line} @@`
    : `@@ -${line},${oldLines.length} +${line},${newLines.length} @@`;
  const body = oldLines.map(l => `-${l}\n`).join("") + newLines.map(l => `+${l}\n`).join("");
  return `User edited "${path}":\n\n\`\`\`diff\n${header}\n${body}\`\`\``;
}

interface Pending { base: string[]; baseVersion: number; start: number; newEnd: number }

/**
 * Per-document edit history. Consecutive changes on touching lines coalesce
 * into one pending burst; the burst is committed as a history event when the
 * cursor leaves its lines or an edit lands elsewhere. Only committed events
 * enter the prompt, so the line being typed (the region) is never also shown
 * as history.
 */
export class EditHistory {
  private pending: Pending | null = null;
  private committed: HistoryEvent[] = [];
  private readonly uri: string;
  private readonly path: () => string;

  constructor(uri: string, path: () => string) {
    this.uri = uri;
    this.path = path;
  }

  /** One document change: `before` and `after` are the full logical lines around it. */
  observe(before: string[], after: string[], beforeVersion: number): void {
    const block = changedBlock(before, after);
    if (block === null) return;
    const pending = this.pending;
    if (pending && block.start <= Math.max(pending.newEnd, pending.start + 1) && block.oldEnd >= pending.start) {
      const merged = changedBlock(pending.base, after);
      if (merged === null) this.pending = null; // the burst was undone
      else this.pending = { ...pending, start: merged.start, newEnd: merged.newEnd };
      return;
    }
    if (pending) this.commit(pending, before);
    this.pending = { base: before, baseVersion: beforeVersion, start: block.start, newEnd: block.newEnd };
  }

  /** The cursor is on `line` of `current`; commit the burst once the cursor has left it. */
  cursorAt(line: number, current: string[]): void {
    const pending = this.pending;
    if (pending && (line < pending.start || line >= Math.max(pending.newEnd, pending.start + 1))) this.commit(pending, current);
  }

  events(): HistoryEvent[] {
    return this.committed.slice(-MAX_HISTORY_EVENTS);
  }

  private commit(pending: Pending, state: string[]): void {
    this.pending = null;
    const block = changedBlock(pending.base, state);
    if (block === null) return;
    let { start, oldEnd, newEnd } = block;
    // TRAIN events always have both a removed and an added side: widen a pure
    // line insertion or deletion by one unchanged neighbour line.
    if (start === oldEnd || start === newEnd) {
      if (start > 0) start--;
      else if (oldEnd < pending.base.length && newEnd < state.length) { oldEnd++; newEnd++; }
      else return;
    }
    const oldLines = pending.base.slice(start, oldEnd);
    const newLines = state.slice(start, newEnd);
    if (oldLines.length + newLines.length > MAX_EVENT_LINES) return;
    const path = this.path();
    const diff = eventDiff(path, start, oldLines, newLines);
    if (diff.length > MAX_EVENT_DIFF_CHARS) return;
    const revision = sha256Text(pending.base.join("\n") + "\n");
    this.committed.push({
      event_id: sha256Text(revision + diff).slice(0, 32),
      kind: "observed_edit_diff",
      path,
      workspace_revision_before: revision,
      old_text: oldLines.join("\n"),
      new_text: newLines.join("\n"),
      range_utf16: {
        uri: this.uri,
        document_version: pending.baseVersion,
        content_sha256: revision,
        start: { line: start, character: 0 },
        end: { line: oldEnd - 1, character: oldLines[oldLines.length - 1].length },
      },
      event_diff: diff,
    });
    if (this.committed.length > 8) this.committed.splice(0, this.committed.length - 8);
  }
}
