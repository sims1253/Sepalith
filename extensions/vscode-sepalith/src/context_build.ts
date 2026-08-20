// Scope-aware context (docs/prompt-format.md "Scope-aware context"): the
// enclosing-function pin and the file outline, plus the prompt renderer.
// Everything in this file is PURE — no vscode import — so the node script
// scripts/check-context.ts can exercise it directly; extension.ts supplies
// only the LSP symbol fetch, the config gate and the document text.
//
// Cache rules (docs/prompt-format.md "Why this order"): sections are
// ordered by churn — file prefix < outline < cursor zone. Typing more on
// the cursor line changes ONLY the two cursor-line-derived lines (the
// after-cursor suffix head and the region's before-cursor text); the
// outline and the pinned function are functions of the document up to the
// enclosing function's end and must not move while the user types plain
// statements.

// ---------------------------------------------------------------------------
// Prompt render (must match experiments/eval/run_eval.py render_zeta2).
// v0: the region is EMPTY — cursor marker alone ("signature" style).
// ---------------------------------------------------------------------------

export const MAX_PREFIX_SUFFIX_CHARS = 6000;

export function renderPrompt(
  prefixLines: string[],
  regionOld: string[],
  suffixLines: string[],
  relPath: string,
  outlineLines: string[] = [], // empty = plain v0.0.6 prompt, byte-identical
): string {
  return [
    "<[fim-suffix]>",
    ...suffixLines,
    `<[fim-prefix]><filename>${relPath}`,
    ...prefixLines,
    ...outlineLines, // docs/prompt-format.md: outline sits between the file prefix and the cursor zone
    "<<<<<<< CURRENT",
    ...regionOld,
    "=======",
    "<[fim-middle]>",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Scope model
// ---------------------------------------------------------------------------

export const MAX_PIN_CHARS = 4000; // pinned enclosing-function text cap
export const MAX_OUTLINE_ENTRIES = 60;
export const MAX_OUTLINE_CHARS = 1500; // the entry lines, excluding the <|outline|> marker

export interface OutlineEntry {
  line: number; // 1-based line of the function signature
  name: string;
}

export interface ScopePin {
  startLine: number; // 0-based, inclusive
  endLine: number; // 0-based, inclusive
  name: string | null; // null = anonymous function(...)
}

export interface ScopeInfo {
  mode: "pin+outline" | "outline" | "off";
  outline: string[]; // formatted, capped, ready to render under <|outline|>
  pin: ScopePin | null;
}

// ---------------------------------------------------------------------------
// LSP document symbols (plain JSON — the shapes
// vscode.executeDocumentSymbolProvider returns). SymbolKind: Function = 12,
// Method = 6.
// ---------------------------------------------------------------------------

const FUNCTION_KINDS = [12, 6];

export interface RawSymbol {
  name: string;
  kind: number;
  startLine: number;
  startChar: number;
  endLine: number;
  endChar: number;
  children: RawSymbol[];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

// accepts both DocumentSymbol ({ range, children }) and the flat
// SymbolInformation ({ location: { range } }) shapes
export function normalizeSymbols(raw: unknown): RawSymbol[] {
  if (!Array.isArray(raw)) return [];
  const out: RawSymbol[] = [];
  for (const item of raw) {
    if (!isRecord(item) || typeof item.name !== "string" || typeof item.kind !== "number") continue;
    const range = isRecord(item.range)
      ? item.range
      : isRecord(item.location) && isRecord(item.location.range)
        ? item.location.range
        : null;
    if (!range || !isRecord(range.start) || !isRecord(range.end)) continue;
    const { start, end } = range;
    if (
      typeof start.line !== "number" || typeof start.character !== "number" ||
      typeof end.line !== "number" || typeof end.character !== "number"
    ) continue;
    out.push({
      name: item.name,
      kind: item.kind,
      startLine: start.line,
      startChar: start.character,
      endLine: end.line,
      endChar: end.character,
      children: normalizeSymbols(item.children),
    });
  }
  return out;
}

function contains(s: RawSymbol, line: number, char: number): boolean {
  if (line < s.startLine || line > s.endLine) return false;
  if (line === s.startLine && char < s.startChar) return false;
  if (line === s.endLine && char > s.endChar) return false;
  return true;
}

// innermost Function/Method symbol whose range contains the cursor
// (smallest range wins, so flat SymbolInformation lists also work)
export function findEnclosingSymbol(symbols: RawSymbol[], cursorLine: number, cursorChar: number): RawSymbol | null {
  let best: RawSymbol | null = null;
  const visit = (list: RawSymbol[]): void => {
    for (const s of list) {
      if (!contains(s, cursorLine, cursorChar)) continue;
      visit(s.children);
      if (!FUNCTION_KINDS.includes(s.kind)) continue;
      if (!best || s.endLine - s.startLine < best.endLine - best.startLine) best = s;
    }
  };
  visit(symbols);
  return best;
}

// top-level (root) function symbols only — no nested definitions
export function outlineFromSymbols(symbols: RawSymbol[]): OutlineEntry[] {
  return symbols.filter((s) => FUNCTION_KINDS.includes(s.kind)).map((s) => ({ line: s.startLine + 1, name: s.name }));
}

// ---------------------------------------------------------------------------
// Brace-scan fallback (no R LSP answering): R functions are
// `name <- function(...)` / `name = function(...)` or an anonymous
// `function(...)` followed by a `{...}` block. Brace depth is tracked from
// the TOP of the SIGNATURE line, not from file start alone; strings and
// comments are blanked first. Approximate is acceptable for this path.
// ---------------------------------------------------------------------------

function cleanRLine(line: string): string {
  // blank string bodies first (a '#' inside a string is not a comment),
  // then drop the comment — braces inside either never count
  return line.replace(/'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/g, '""').replace(/#.*/, "");
}

const NAMED_SIG = /^\s*([A-Za-z._][\w.]*)\s*(?:<-|=)\s*function\s*\(/;
const ANON_SIG = /(?:^|[(,=\s])function\s*\(/;

interface SigMatch {
  name: string | null; // null = anonymous function(...)
}

function matchSignature(cleaned: string): SigMatch | null {
  const m = NAMED_SIG.exec(cleaned);
  if (m) return { name: m[1] ?? null };
  if (ANON_SIG.test(cleaned)) return { name: null };
  return null; // not a signature line
}

function netBraces(cleaned: string): number {
  let n = 0;
  for (const ch of cleaned) {
    if (ch === "{") n++;
    else if (ch === "}") n--;
  }
  return n;
}

export function findEnclosingFunctionByScan(lines: string[], cursorLine: number): ScopePin | null {
  const cleaned = lines.map(cleanRLine);
  for (let c = cursorLine; c >= 0; c--) {
    const sig = matchSignature(cleaned[c] ?? "");
    if (!sig) continue; // not a signature line
    // brace depth from the top of the signature line through the cursor
    let depth = 0;
    let sawOpen = false;
    let inside = false;
    for (let i = c; i <= cursorLine; i++) {
      const text = cleaned[i] ?? "";
      if (text.includes("{")) sawOpen = true;
      depth += netBraces(text);
      inside = sawOpen && depth > 0; // set BEFORE the breaks: a closed block is not enclosing
      if (depth < 0) break; // unbalanced above the block: wrong candidate
      if (sawOpen && depth <= 0) break; // block closed at/before the cursor
    }
    if (!inside) continue;
    // the block opened at c is still open at the cursor — innermost wins.
    // Find the end: scan on until the depth returns to 0.
    for (let i = cursorLine + 1; i < lines.length; i++) {
      depth += netBraces(cleaned[i] ?? "");
      if (depth <= 0) return { startLine: c, endLine: i, name: sig.name };
    }
    return { startLine: c, endLine: lines.length - 1, name: sig.name }; // unclosed — approximate
  }
  return null;
}

// top-level named signatures (brace depth 0) — the file's vocabulary
export function outlineFromScan(lines: string[]): OutlineEntry[] {
  const entries: OutlineEntry[] = [];
  let depth = 0;
  for (let i = 0; i < lines.length; i++) {
    const cleaned = cleanRLine(lines[i] ?? "");
    const sig = matchSignature(cleaned);
    if (depth === 0 && sig && sig.name !== null) entries.push({ line: i + 1, name: sig.name });
    depth = Math.max(0, depth + netBraces(cleaned)); // clamp: survive stray garbage
  }
  return entries;
}

// ---------------------------------------------------------------------------
// Caps + assembly (the ONE place outline/pin truncation happens)
// ---------------------------------------------------------------------------

// outline: <= 60 entries and <= ~1500 chars, then an ellipsis marker line
export function formatOutline(entries: OutlineEntry[]): string[] {
  const out: string[] = [];
  let used = 0;
  let dropped = 0;
  for (const e of entries) {
    const text = `${e.line} ${e.name}`;
    if (out.length >= MAX_OUTLINE_ENTRIES || used + text.length + 1 > MAX_OUTLINE_CHARS) {
      dropped = entries.length - out.length;
      break;
    }
    out.push(text);
    used += text.length + 1;
  }
  if (dropped > 0) out.push(`... (${dropped} more)`);
  return out;
}

function lineSpanChars(lines: string[], startLine: number, endLine: number): number {
  let n = 0;
  for (let i = startLine; i <= endLine && i < lines.length; i++) n += (lines[i] ?? "").length + 1;
  return n;
}

// shared by the LSP and the scan paths: pin cap, outline dedup, mode
export function buildScope(lines: string[], cursorLine: number, enclosing: ScopePin | null, entries: OutlineEntry[]): ScopeInfo {
  let pin: ScopePin | null = null;
  if (enclosing && enclosing.startLine <= cursorLine && enclosing.endLine >= cursorLine) {
    // if the pinned function would blow the cap, fall back to outline only
    if (lineSpanChars(lines, enclosing.startLine, enclosing.endLine) <= MAX_PIN_CHARS) {
      pin = enclosing;
    }
  }
  // doc rule 1: the enclosing function's entry is dropped from the outline
  // when the function is already fully present (pinned)
  const outlineEntries = pin ? entries.filter((e) => e.line - 1 !== pin.startLine) : entries;
  const outline = formatOutline(outlineEntries);
  const mode: ScopeInfo["mode"] = pin ? "pin+outline" : outline.length > 0 ? "outline" : "off";
  return { mode, outline, pin };
}

export function scopeFromSymbols(symbols: RawSymbol[], lines: string[], cursorLine: number, cursorChar: number): ScopeInfo {
  const enc = findEnclosingSymbol(symbols, cursorLine, cursorChar);
  return buildScope(
    lines,
    cursorLine,
    enc ? { startLine: enc.startLine, endLine: enc.endLine, name: enc.name } : null,
    outlineFromSymbols(symbols),
  );
}

export function scopeFromScan(lines: string[], cursorLine: number): ScopeInfo {
  return buildScope(lines, cursorLine, findEnclosingFunctionByScan(lines, cursorLine), outlineFromScan(lines));
}

// ---------------------------------------------------------------------------
// Prompt build — scope === null (or mode "off" with nothing to add) renders
// the byte-identical v0.0.6 prompt.
// ---------------------------------------------------------------------------

export interface PromptBuild {
  prompt: string;
  truncatedLines: number; // prefix lines cut from the START (as v0.0.6)
  truncatedSuffixLines: number; // suffix lines cut from the END (pin path only)
}

export function buildScopedPrompt(
  lines: string[],
  cursorLine: number,
  cursorChar: number,
  relPath: string,
  scope: ScopeInfo | null,
): PromptBuild {
  // the CURSOR LINE belongs to the prompt: text before the cursor is the
  // typed partial (training's midtyping convention: partial + cursor marker
  // in the region), text after the cursor leads the suffix. v0 dropped the
  // line entirely, so the model re-predicted it from scratch and accepting
  // glued the duplicate into the document.
  const line = lines[cursorLine] ?? "";
  const before = line.slice(0, cursorChar);
  const after = line.slice(cursorChar);
  const regionOld = [before === "" ? "<|user_cursor|>" : before + "<|user_cursor|>"];
  const pin = scope ? scope.pin : null;
  let suffix: string[];
  let truncatedSuffixLines = 0;
  if (pin && pin.endLine >= cursorLine) {
    // the pinned remainder (the function's lines directly below the cursor)
    // leads the suffix; the OLDER file-below content is truncated from its
    // END so the cut hits the deepest lines, never the enclosing function
    const pinned = lines.slice(cursorLine + 1, pin.endLine + 1);
    const older = lines.slice(pin.endLine + 1);
    let used = after.length + 1 + pinned.reduce((n, l) => n + l.length + 1, 0);
    let olderKeep = 0;
    for (; olderKeep < older.length; olderKeep++) {
      if (used + older[olderKeep].length + 1 > MAX_PREFIX_SUFFIX_CHARS) break;
      used += older[olderKeep].length + 1;
    }
    truncatedSuffixLines = older.length - olderKeep;
    suffix = [after, ...pinned, ...older.slice(0, olderKeep)];
  } else {
    suffix = [after, ...lines.slice(cursorLine + 1)];
  }
  const suffixChars = suffix.reduce((n, l) => n + l.length + 1, 0);
  const budget = Math.max(0, MAX_PREFIX_SUFFIX_CHARS - suffixChars);
  const prefix = lines.slice(0, cursorLine); // everything above the cursor's line
  // truncate the prefix from its START (keep the tail) so prefix+suffix stays under ~6000 chars
  let keep = 0;
  let used = 0;
  for (let i = prefix.length - 1; i >= 0; i--) {
    if (used + prefix[i].length + 1 > budget) break;
    used += prefix[i].length + 1;
    keep++;
  }
  const truncatedLines = prefix.length - keep;
  const outlineLines = scope && scope.outline.length > 0 ? ["<|outline|>", ...scope.outline] : [];
  return {
    prompt: renderPrompt(prefix.slice(truncatedLines), regionOld, suffix, relPath, outlineLines),
    truncatedLines,
    truncatedSuffixLines,
  };
}
