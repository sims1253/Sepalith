// PRM03 prompt and output contract (renderer `zeta2-prm03-v1`), ported from
// packages/sepalith/src/sepalith/campaign_protocol.py. Pure: no vscode import,
// so the parity checks in scripts/ run it under plain node. Keep this file in
// lockstep with the Python module; scripts/check-prm03.ts compares both on
// TRAIN-derived golden fixtures.
//
// Strings: Python indexes code points, JavaScript UTF-16 code units. Every
// column below is converted explicitly; rendered text is identical because the
// cursor marker is inserted at the same character boundary.

export const SCHEMA_VERSION = "sepalith.prompt.prm03.v1";
export const RENDERER_ID = "zeta2-prm03-v1";
export const TASK_CONTRACT =
  "Edit the current region. Output the complete replacement region, " +
  "preserving whitespace. For no change, output the ordinary line " +
  "[NO_EDIT] followed by the exact terminal line >>>>>>> UPDATED; an " +
  "empty body is reserved for deletion of a nonempty range.";
export const TERMINAL = ">>>>>>> UPDATED";
export const NO_EDIT = "[NO_EDIT]";
export const CURSOR_MARKER = "<|user_cursor|>";
export const GENERATION_BOUNDARY = "<[fim-middle]>";
export const BOS_ID = 0;
export const EOS_ID = 1;
export const NATIVE_EOG_IDS: readonly number[] = [1, 130073];
/** First token of `[NO_EDIT]`; the C12 census shows it is the model's no-op decision. */
export const NO_EDIT_FIRST_TOKEN_ID = 80;
export const VOCAB_SIZE = 130560;

const RESERVED_BODY_LINES = new Set([
  "<<<<<<< CURRENT", "=======", ">>>>>>> UPDATED", "<[fim-middle]>", "<[fim-prefix]>",
  "<[fim-suffix]>", "<|user_cursor|>", "<|outline|>", NO_EDIT,
]);
const SCOPES = new Set(["pin+outline", "outline", "off"]);
const EOLS = new Set(["lf", "crlf", "mixed"]);
const SHA256 = /^[0-9a-f]{64}$/;

export class ProtocolError extends Error {}
export class StaleContextError extends ProtocolError {}

export interface Position { line: number; character: number }
export interface ReplacementRange {
  uri: string;
  document_version: number;
  content_sha256: string;
  start: Position;
  end: Position;
}
export interface Cursor {
  region_line_index: number;
  code_point_column: number | null;
  utf16_column: number | null;
}
export interface EvidenceRecord { content: string; path: string | null }
export interface HistoryEvent {
  event_id: string;
  kind: string;
  path: string;
  workspace_revision_before: string;
  old_text: string;
  new_text: string;
  range_utf16: ReplacementRange;
  event_diff: string;
}
export type DocumentEol = "lf" | "crlf" | "mixed";
export interface PromptContext {
  schema_version: string;
  path: string;
  prefix: string[];
  selected_references: EvidenceRecord[];
  history: HistoryEvent[];
  diagnostics: EvidenceRecord[];
  retrieval: EvidenceRecord[];
  scope_mode: string;
  scope_lines: string[];
  suffix_lines: string[];
  region_old: string[];
  cursor: Cursor;
  replacement_range: ReplacementRange;
  document_eol: DocumentEol;
}

// ---------------------------------------------------------------------------
// Code points and UTF-16
// ---------------------------------------------------------------------------

/** Number of code points (Python `len`). */
export function codePointLength(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) {
      const d = text.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) i++;
    }
    n++;
  }
  return n;
}

/** UTF-16 offset of a code-point column. */
export function codepointToUtf16Column(text: string, column: number): number {
  if (!Number.isSafeInteger(column) || column < 0) throw new ProtocolError("code-point column must be an integer >= 0");
  let units = 0;
  for (let seen = 0; seen < column; seen++) {
    if (units >= text.length) throw new ProtocolError("code-point column exceeds line length");
    const c = text.charCodeAt(units);
    units += c >= 0xd800 && c <= 0xdbff && units + 1 < text.length && isLowSurrogate(text.charCodeAt(units + 1)) ? 2 : 1;
  }
  return units;
}

/** Code-point column of a UTF-16 offset; rejects an offset inside a surrogate pair. */
export function utf16ToCodepointColumn(text: string, column: number): number {
  if (!Number.isSafeInteger(column) || column < 0) throw new ProtocolError("UTF-16 column must be an integer >= 0");
  if (column > text.length) throw new ProtocolError("UTF-16 column exceeds line length");
  if (column > 0 && column < text.length && isLowSurrogate(text.charCodeAt(column)) && isHighSurrogate(text.charCodeAt(column - 1)))
    throw new ProtocolError("UTF-16 column falls inside a surrogate pair");
  return codePointLength(text.slice(0, column));
}

function isHighSurrogate(c: number): boolean { return c >= 0xd800 && c <= 0xdbff; }
function isLowSurrogate(c: number): boolean { return c >= 0xdc00 && c <= 0xdfff; }

// Python str.isspace() characters, so `line.strip()` matches exactly.
const PY_WHITESPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

/** Python `str.strip()` with no arguments. */
export function pyStrip(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && PY_WHITESPACE.has(text.charCodeAt(start))) start++;
  while (end > start && PY_WHITESPACE.has(text.charCodeAt(end - 1))) end--;
  return text.slice(start, end);
}

// ---------------------------------------------------------------------------
// Validation (PromptContext.__post_init__)
// ---------------------------------------------------------------------------

function validateReservedSlots(text: string): void {
  // Added tokens that HF does not mark special but native GGUF treats as CONTROL.
  for (const match of text.matchAll(/<unused_token_(0|[1-9][0-9]{0,2})>/g)) {
    if (Number(match[1]) <= 477) throw new ProtocolError("reserved_slot_literal_has_inconsistent_tokenization");
  }
}

function positionKey(position: Position): number[] { return [position.line, position.character]; }
function precedes(a: Position, b: Position): boolean {
  const [al, ac] = positionKey(a);
  const [bl, bc] = positionKey(b);
  return al < bl || (al === bl && ac < bc);
}
export function isEmptyRange(range: ReplacementRange): boolean {
  return range.start.line === range.end.line && range.start.character === range.end.character;
}

function validateRange(range: ReplacementRange, name: string): void {
  if (!range.uri) throw new ProtocolError(`${name}.uri must be a non-empty string`);
  for (const value of [range.document_version, range.start.line, range.start.character, range.end.line, range.end.character]) {
    if (!Number.isSafeInteger(value) || value < 0) throw new ProtocolError(`${name} integers must be >= 0`);
  }
  if (!SHA256.test(range.content_sha256)) throw new ProtocolError(`${name}.content_sha256 must be a lowercase SHA256 hex digest`);
  if (precedes(range.end, range.start)) throw new ProtocolError(`${name}.end must not precede start`);
}

function validateCursor(cursor: Cursor, regionOld: readonly string[]): void {
  if (regionOld.length === 0) {
    if (cursor.region_line_index !== -1 || cursor.code_point_column !== null || cursor.utf16_column !== null)
      throw new ProtocolError("empty region requires cursor index -1 and null columns");
    return;
  }
  if (!(cursor.region_line_index >= 0 && cursor.region_line_index < regionOld.length))
    throw new ProtocolError("cursor.region_line_index must address region_old");
  if (cursor.code_point_column === null || cursor.utf16_column === null)
    throw new ProtocolError("nonempty region requires both cursor columns");
  const line = regionOld[cursor.region_line_index];
  if (cursor.code_point_column > codePointLength(line)) throw new ProtocolError("cursor code-point column exceeds line");
  if (cursor.utf16_column !== codepointToUtf16Column(line, cursor.code_point_column))
    throw new ProtocolError("cursor UTF-16 and code-point columns disagree");
  if (utf16ToCodepointColumn(line, cursor.utf16_column) !== cursor.code_point_column)
    throw new ProtocolError("cursor UTF-16 column is not a code-point boundary");
}

function validateRangeGeometry(range: ReplacementRange, regionOld: readonly string[]): void {
  if (regionOld.length === 0) {
    if (!isEmptyRange(range)) throw new ProtocolError("empty region requires a zero-width replacement range");
    return;
  }
  const lineSpan = range.end.line - range.start.line;
  if (lineSpan !== regionOld.length - 1) throw new ProtocolError("replacement range line span disagrees with region_old line count");
  if (lineSpan === 0) {
    if (range.end.character - range.start.character !== regionOld[0].length)
      throw new ProtocolError("same-line replacement range width disagrees with region_old UTF-16 length");
  } else if (range.end.character !== regionOld[regionOld.length - 1].length) {
    throw new ProtocolError("multiline replacement range end disagrees with final region_old UTF-16 length");
  }
}

/** Throw ProtocolError unless `context` satisfies the PRM03 context invariants. */
export function validateContext(context: PromptContext): void {
  if (context.schema_version !== SCHEMA_VERSION) throw new ProtocolError(`unsupported schema_version: ${context.schema_version}`);
  const path = context.path;
  if (!path) throw new ProtocolError("path must be a non-empty string");
  if (path.startsWith("/") || path.includes("\\") || path.includes("\x00")) throw new ProtocolError("path must be a relative POSIX path");
  if (context.region_old.length === 1 && context.region_old[0] === "")
    throw new ProtocolError("region_old empty text must use the canonical [] representation");
  for (const lines of [context.prefix, context.scope_lines, context.suffix_lines, context.region_old]) {
    if (lines.some(line => line.includes("\n") || line.includes("\r")))
      throw new ProtocolError("line arrays must contain one LF-free, CR-free wire line per entry");
  }
  if (!SCOPES.has(context.scope_mode)) throw new ProtocolError(`unsupported scope_mode: ${context.scope_mode}`);
  if (!EOLS.has(context.document_eol)) throw new ProtocolError(`unsupported document_eol: ${context.document_eol}`);
  validateRange(context.replacement_range, "replacement_range");
  for (const event of context.history) validateRange(event.range_utf16, "replacement_range");
  if ((context.region_old.length > 0) === isEmptyRange(context.replacement_range))
    throw new ProtocolError("region_old emptiness must agree with replacement range geometry");
  validateCursor(context.cursor, context.region_old);
  validateRangeGeometry(context.replacement_range, context.region_old);
}

// ---------------------------------------------------------------------------
// JSON boundary (PromptContext.from_dict), used by fixtures and tests
// ---------------------------------------------------------------------------

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
type JsonObject = { [key: string]: JsonValue };

function object(value: JsonValue | undefined, name: string, keys: string[]): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new ProtocolError(`${name} must be an object`);
  const actual = Object.keys(value);
  const missing = keys.filter(k => !actual.includes(k)).sort();
  const extra = actual.filter(k => !keys.includes(k)).sort();
  if (missing.length || extra.length) {
    const detail = [missing.length ? `missing=${JSON.stringify(missing)}` : "", extra.length ? `unexpected=${JSON.stringify(extra)}` : ""];
    throw new ProtocolError(`${name} fields mismatch (${detail.filter(Boolean).join(", ")})`);
  }
  return value;
}
function str(value: JsonValue | undefined, name: string, nonempty = true): string {
  if (typeof value !== "string" || (nonempty && !value)) throw new ProtocolError(`${name} must be a ${nonempty ? "non-empty " : ""}string`);
  return value;
}
function int(value: JsonValue | undefined, name: string, minimum = 0): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) throw new ProtocolError(`${name} must be an integer >= ${minimum}`);
  return value;
}
function lines(value: JsonValue | undefined, name: string): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== "string")) throw new ProtocolError(`${name} must be an array of strings`);
  return value.map(item => String(item));
}
function list(value: JsonValue | undefined, name: string): JsonValue[] {
  if (!Array.isArray(value)) throw new ProtocolError(`${name} must be an array of objects`);
  return value;
}
function nullableInt(value: JsonValue | undefined, name: string): number | null {
  return value === null ? null : int(value, name);
}
function position(value: JsonValue | undefined, name: string): Position {
  const data = object(value, name, ["line", "character"]);
  return { line: int(data.line, `${name}.line`), character: int(data.character, `${name}.character`) };
}
function range(value: JsonValue | undefined): ReplacementRange {
  const data = object(value, "replacement_range", ["uri", "document_version", "content_sha256", "start", "end"]);
  return {
    uri: str(data.uri, "replacement_range.uri"),
    document_version: int(data.document_version, "replacement_range.document_version"),
    content_sha256: str(data.content_sha256, "replacement_range.content_sha256"),
    start: position(data.start, "replacement_range.start"),
    end: position(data.end, "replacement_range.end"),
  };
}
function evidence(value: JsonValue, name: string): EvidenceRecord {
  const data = object(value, name, ["content", "path"]);
  return { content: str(data.content, `${name}.content`, false), path: data.path === null ? null : str(data.path, `${name}.path`) };
}
function historyEvent(value: JsonValue): HistoryEvent {
  const data = object(value, "history event", ["event_id", "kind", "path", "workspace_revision_before", "old_text", "new_text", "range_utf16", "event_diff"]);
  return {
    event_id: str(data.event_id, "history.event_id"),
    kind: str(data.kind, "history.kind"),
    path: str(data.path, "history.path"),
    workspace_revision_before: str(data.workspace_revision_before, "history.workspace_revision_before"),
    old_text: str(data.old_text, "history.old_text", false),
    new_text: str(data.new_text, "history.new_text", false),
    range_utf16: range(data.range_utf16),
    event_diff: str(data.event_diff, "history.event_diff", false),
  };
}

/** Parse and validate a JSON context (the DAT10 context-sidecar `context` object). */
// Fixture JSON is untrusted until every consumed field has been checked here.
// oxlint-disable-next-line anti-slop/no-unknown-parameters
export function promptContextFromJson(value: unknown): PromptContext {
  // SAFETY: JSON.parse output is a JsonValue; `object` rejects every non-object shape before use.
  const data = object(value as JsonValue, "context", [
    "schema_version", "path", "prefix", "selected_references", "history", "diagnostics", "retrieval",
    "scope_mode", "scope_lines", "suffix_lines", "region_old", "cursor", "replacement_range", "document_eol",
  ]);
  const cursor = object(data.cursor, "cursor", ["region_line_index", "code_point_column", "utf16_column"]);
  const eol = str(data.document_eol, "document_eol");
  if (eol !== "lf" && eol !== "crlf" && eol !== "mixed") throw new ProtocolError(`unsupported document_eol: ${eol}`);
  const context: PromptContext = {
    schema_version: str(data.schema_version, "schema_version"),
    path: str(data.path, "path"),
    prefix: lines(data.prefix, "prefix"),
    selected_references: list(data.selected_references, "selected_references").map(x => evidence(x, "selected_reference")),
    history: list(data.history, "history").map(historyEvent),
    diagnostics: list(data.diagnostics, "diagnostics").map(x => evidence(x, "diagnostic")),
    retrieval: list(data.retrieval, "retrieval").map(x => evidence(x, "retrieval")),
    scope_mode: str(data.scope_mode, "scope_mode"),
    scope_lines: lines(data.scope_lines, "scope_lines"),
    suffix_lines: lines(data.suffix_lines, "suffix_lines"),
    region_old: lines(data.region_old, "region_old"),
    cursor: {
      region_line_index: int(cursor.region_line_index, "cursor.region_line_index", -1),
      code_point_column: nullableInt(cursor.code_point_column, "cursor.code_point_column"),
      utf16_column: nullableInt(cursor.utf16_column, "cursor.utf16_column"),
    },
    replacement_range: range(data.replacement_range),
    document_eol: eol,
  };
  validateContext(context);
  return context;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function renderRecords(parts: string[], records: readonly EvidenceRecord[]): void {
  for (const record of records) {
    if (record.path !== null) parts.push(`<filename>${record.path}`);
    parts.push(record.content);
  }
}

/** Model-visible prompt text from pre-edit context only (render_prompt). Ends with the boundary LF. */
export function renderPrompt(context: PromptContext): string {
  validateContext(context);
  const parts: string[] = [TASK_CONTRACT, `<filename>${context.path}`, ...context.prefix];
  parts.push("<filename>selected_references");
  renderRecords(parts, context.selected_references);
  parts.push("<filename>edit_history");
  for (const event of context.history) parts.push(event.event_diff);
  parts.push("<filename>diagnostics");
  renderRecords(parts, context.diagnostics);
  parts.push("<filename>retrieval");
  renderRecords(parts, context.retrieval);
  parts.push("<[fim-suffix]>", ...context.scope_lines, ...context.suffix_lines, "<<<<<<< CURRENT");
  const region = [...context.region_old];
  if (context.cursor.region_line_index >= 0 && context.cursor.utf16_column !== null) {
    const index = context.cursor.region_line_index;
    const line = region[index];
    const column = context.cursor.utf16_column;
    region[index] = line.slice(0, column) + CURSOR_MARKER + line.slice(column);
  }
  parts.push(...region, "=======", GENERATION_BOUNDARY);
  // The final LF is part of the v1 prompt boundary (keeps BPE from merging the boundary with the response).
  const rendered = parts.join("\n") + "\n";
  validateReservedSlots(rendered);
  return rendered;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/** Split LF/CRLF transport without trimming source whitespace. */
export function splitWireLines(text: string): string[] {
  return text.split("\n").map(line => (line.endsWith("\r") ? line.slice(0, -1) : line));
}

export type Operation = "no_op" | "replace" | "delete";
export interface OutputResult {
  status: "accepted" | "invalid";
  operation: Operation | null;
  body: string[];
  body_text: string;
  reason: string | null;
}

function invalid(reason: string): OutputResult {
  return { status: "invalid", operation: null, body: [], body_text: "", reason };
}

function sameLines(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((line, i) => line === b[i]);
}

/** Exact-line output parse (parse_output): only an exact terminal line frames the response. */
export function parseOutput(rawText: string, context: PromptContext): OutputResult {
  try {
    validateReservedSlots(rawText);
  } catch (error) {
    return invalid(error instanceof Error ? error.message : String(error));
  }
  const all = splitWireLines(rawText);
  const terminals = all.flatMap((line, index) => (line === TERMINAL ? [index] : []));
  if (terminals.length === 0) return invalid("missing_exact_terminal");
  if (terminals.length !== 1) return invalid("duplicate_exact_terminal");
  const terminal = terminals[0];
  if (all.slice(terminal + 1).some(line => line !== "")) return invalid("content_after_terminal");
  let body = all.slice(0, terminal);
  // One empty line before the terminal is the canonical spelling of an empty body.
  if (body.length === 1 && body[0] === "") body = [];
  const isNoEdit = body.length === 1 && body[0] === NO_EDIT;
  if (!isNoEdit && body.some(line => RESERVED_BODY_LINES.has(pyStrip(line)))) return invalid("reserved_full_body_line");
  if (isNoEdit || sameLines(body, context.region_old)) {
    return { status: "accepted", operation: "no_op", body, body_text: body.join("\n"), reason: null };
  }
  if (body.length === 0) {
    const operation: Operation = isEmptyRange(context.replacement_range) ? "no_op" : "delete";
    return { status: "accepted", operation, body, body_text: "", reason: null };
  }
  return { status: "accepted", operation: "replace", body, body_text: body.join("\n"), reason: null };
}

/** Native CONTROL ranges of the pinned GGUF, including reserved slots. */
export function isNativeControlToken(token: number): boolean {
  return (token >= 0 && token <= 7) || (token >= 10 && token <= 21) || (token >= 130072 && token < VOCAB_SIZE);
}

/** One canonical terminal EOS and no hidden native control tokens (valid_generation_tokens). */
export function validGenerationTokens(tokens: readonly number[]): boolean {
  return tokens.length > 0 && tokens[tokens.length - 1] === EOS_ID
    && tokens.every(t => Number.isSafeInteger(t) && t >= 0 && t < VOCAB_SIZE)
    && tokens.slice(0, -1).every(t => !isNativeControlToken(t));
}

export function checkStale(range: ReplacementRange, activeUri: string, activeVersion: number, activeSha256: string): void {
  if (activeUri !== range.uri || activeVersion !== range.document_version || activeSha256 !== range.content_sha256)
    throw new StaleContextError("document URI, version or pre-edit content identity changed");
}

/** Map an LF wire payload to the document's separators without trimming. */
export function mapWireToDocumentEol(text: string, documentEol: DocumentEol): string {
  if (documentEol === "lf") return text;
  if (documentEol === "crlf") {
    if (text.includes("\r")) throw new ProtocolError("wire payload must use LF separators before CRLF mapping");
    return text.split("\n").join("\r\n");
  }
  throw new ProtocolError("mixed document EOL requires an explicit application policy");
}

export interface ApplicationPlan {
  operation: Operation;
  mode: "none" | "inline" | "workspace_edit_user_accept";
  replacement_range: ReplacementRange;
  replacement_text: string;
  user_accept_required: boolean;
}

/** make_application_plan: stale check, then the edit of the captured replacement range. */
export function makeApplicationPlan(result: OutputResult, context: PromptContext, activeUri: string,
                                    activeVersion: number, activeSha256: string): ApplicationPlan {
  if (result.status !== "accepted" || result.operation === null) throw new ProtocolError(result.reason ?? "invalid output");
  checkStale(context.replacement_range, activeUri, activeVersion, activeSha256);
  const range = context.replacement_range;
  if (result.operation === "no_op") return { operation: "no_op", mode: "none", replacement_range: range, replacement_text: "", user_accept_required: false };
  const text = mapWireToDocumentEol(result.body_text, context.document_eol);
  const sameLine = range.start.line === range.end.line;
  return { operation: result.operation, mode: sameLine ? "inline" : "workspace_edit_user_accept",
    replacement_range: range, replacement_text: text, user_accept_required: !sameLine };
}
