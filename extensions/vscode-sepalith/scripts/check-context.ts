// Unit tests for the pure scope-context helpers in src/context_build.ts:
// outline formatting + caps, the brace-scan enclosing-function fallback,
// LSP symbol normalization/selection, pin/suffix truncation, legacy
// (v0.0.6) prompt parity, and the stable-prefix property (typing more on
// the cursor line must not move the outline or the enclosing-function pin).
// No vscode import, plain node. Run: npm run check-context
import {
  buildScopedPrompt,
  findEnclosingFunctionByScan,
  findEnclosingSymbol,
  formatOutline,
  normalizeSymbols,
  outlineFromScan,
  scopeFromScan,
  scopeFromSymbols,
  MAX_OUTLINE_CHARS,
  type OutlineEntry,
  type RawSymbol,
  type ScopeInfo,
} from "../src/context_build.ts";

let checks = 0;
let failures = 0;
function check(name: string, cond: boolean): void {
  checks++;
  if (!cond) {
    failures++;
    console.error(`check-context: FAIL ${name}`);
  }
}
function json(x: unknown): string {
  return JSON.stringify(x);
}

// the v0.0.6 buildPrompt, copied verbatim as the parity target
const MAX = 6000;
function legacyPrompt(docLines: string[], cursorLine: number, cursorChar: number, relPath: string): string {
  const line = docLines[cursorLine] ?? "";
  const before = line.slice(0, cursorChar);
  const after = line.slice(cursorChar);
  const regionOld = [before === "" ? "<|user_cursor|>" : before + "<|user_cursor|>"];
  const suffix = [after, ...docLines.slice(cursorLine + 1)];
  const suffixChars = suffix.reduce((n, l) => n + l.length + 1, 0);
  const budget = Math.max(0, MAX - suffixChars);
  const prefix = docLines.slice(0, cursorLine);
  let keep = 0;
  let used = 0;
  for (let i = prefix.length - 1; i >= 0; i--) {
    if (used + prefix[i].length + 1 > budget) break;
    used += prefix[i].length + 1;
    keep++;
  }
  const truncatedLines = prefix.length - keep;
  return [
    "<[fim-suffix]>",
    ...suffix,
    `<[fim-prefix]><filename>${relPath}`,
    ...prefix.slice(truncatedLines),
    "<<<<<<< CURRENT",
    ...regionOld,
    "=======",
    "<[fim-middle]>",
  ].join("\n");
}

// replace the two cursor-line-derived prompt lines (the after-cursor suffix
// head and the region's before-cursor text) with placeholders — what is
// left must be byte-identical across typing states
function stabilize(prompt: string): string {
  const ls = prompt.split("\n");
  const cur = ls.indexOf("<<<<<<< CURRENT");
  ls[1] = "<VOLATILE-AFTER>";
  ls[cur + 1] = "<VOLATILE-REGION>";
  return ls.join("\n");
}

function commonPrefixLen(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

// --- fixtures ---------------------------------------------------------------

// 0  library(dplyr)
// 2  summarise_data <- function(...) { ... }        (ends line 6)
// 8  fit_model <- function(...) {                   (ends line 14)
// 9    inner <- function(x) { ... }                 (ends line 12)
// 16 top_level_code <- 42
const DOC = [
  "library(dplyr)",
  "",
  "summarise_data <- function(df, group) {",
  "  df %>%",
  "    group_by({{ group }}) %>%",
  "    summarise(mean = mean(x, na.rm = TRUE))  # braces } in comment",
  "}",
  "",
  "fit_model <- function(data, weights) {",
  "  inner <- function(x) {",
  '    s <- "if (x) { y }"',
  "    x + 1",
  "  }",
  "  inner(data)",
  "}",
  "",
  "top_level_code <- 42",
];

// --- brace-scan fallback ----------------------------------------------------

const e1 = findEnclosingFunctionByScan(DOC, 10);
check("scan: innermost function at line 10 is inner (9-12)", !!e1 && e1.startLine === 9 && e1.endLine === 12 && e1.name === "inner");
const e2 = findEnclosingFunctionByScan(DOC, 13);
check("scan: after inner closes, fit_model encloses (8-14)", !!e2 && e2.startLine === 8 && e2.endLine === 14 && e2.name === "fit_model");
check("scan: top level has no enclosing function", findEnclosingFunctionByScan(DOC, 0) === null);
check("scan: top-level statement line 16 has no enclosing function", findEnclosingFunctionByScan(DOC, 16) === null);
const e3 = findEnclosingFunctionByScan(DOC, 5);
check("scan: braces in strings/comments do not fool the scan (2-6)", !!e3 && e3.startLine === 2 && e3.endLine === 6 && e3.name === "summarise_data");
const e4 = findEnclosingFunctionByScan(["f <- function(x)", "{", "  x", "}"], 2);
check("scan: brace on the line after the signature", !!e4 && e4.startLine === 0 && e4.endLine === 3 && e4.name === "f");
check("scan: one-line function without a {...} block is not pinned", findEnclosingFunctionByScan(["f <- function(x) x + 1", "g <- 1"], 0) === null);

const entries = outlineFromScan(DOC);
check(
  "scan outline: top-level named functions only (1-based lines)",
  json(entries) === json([{ line: 3, name: "summarise_data" }, { line: 9, name: "fit_model" }]),
);

// --- outline formatting + caps (one place: formatOutline) --------------------

check("outline: one entry per line as `LINE name`", json(formatOutline([{ line: 42, name: "summarise_data" }])) === json(["42 summarise_data"]));
check("outline: empty file gives no outline", formatOutline([]).length === 0);
const many: OutlineEntry[] = [];
for (let i = 1; i <= 100; i++) many.push({ line: i, name: `f${i}` });
const cappedEntries = formatOutline(many);
check("outline: 60-entry cap + ellipsis marker line", cappedEntries.length === 61 && cappedEntries[60] === "... (40 more)");
const wide: OutlineEntry[] = [];
for (let i = 1; i <= 200; i++) wide.push({ line: i, name: "n".repeat(30) });
const cappedChars = formatOutline(wide);
check(
  "outline: ~1500-char cap + ellipsis marker line",
  cappedChars.length <= 61 &&
    cappedChars.slice(0, -1).join("\n").length <= MAX_OUTLINE_CHARS &&
    /^\.\.\. \(\d+ more\)$/.test(cappedChars[cappedChars.length - 1]),
);

// --- pin cap ------------------------------------------------------------------

function bigFunctionDocs(commentLines: number): string[] {
  const lines = ["big <- function(x) {"];
  for (let i = 0; i < commentLines; i++) lines.push("  # " + "a".repeat(80));
  lines.push("}");
  return lines;
}
const overCap = scopeFromScan(bigFunctionDocs(60), 30); // ~5.1k chars of function
check("pin cap: oversized function falls back to outline only", overCap.mode === "outline" && overCap.pin === null);
check("pin cap: oversized function keeps its outline entry (no dedup without pin)", overCap.outline.includes("1 big"));
const underCap = scopeFromScan(bigFunctionDocs(40), 20); // ~3.4k chars of function
check("pin cap: function under the cap is pinned", underCap.mode === "pin+outline" && underCap.pin !== null && underCap.pin.name === "big");

// --- LSP symbol path -----------------------------------------------------------

const r = (sl: number, sc: number, el: number, ec: number) => ({ start: { line: sl, character: sc }, end: { line: el, character: ec } });
const syms = normalizeSymbols([
  { name: "summarise_data", kind: 12, range: r(2, 0, 6, 1), children: [] },
  {
    name: "fit_model",
    kind: 12,
    range: r(8, 0, 14, 1),
    children: [{ name: "inner", kind: 12, range: r(9, 2, 12, 3), children: [] }],
  },
]);
check("symbols: normalize DocumentSymbol shape", syms.length === 2 && syms[1].children.length === 1);
check("symbols: garbage input normalizes to empty", normalizeSymbols(undefined).length === 0 && normalizeSymbols([{}, null, "x"]).length === 0);
const flat = normalizeSymbols([
  { name: "a", kind: 12, location: { range: r(0, 0, 3, 0) } },
  { name: "b", kind: 12, location: { range: r(1, 0, 2, 0) } },
]);
check("symbols: flat SymbolInformation shape normalizes", flat.length === 2);
check("symbols: smallest containing range wins", findEnclosingSymbol(flat, 1, 0)?.name === "b");
check("symbols: Method kind (6) counts as a function", findEnclosingSymbol(normalizeSymbols([{ name: "m", kind: 6, range: r(0, 0, 2, 0), children: [] }]), 1, 0)?.name === "m");
const vf = normalizeSymbols([
  { name: "v", kind: 13, range: r(0, 0, 5, 0), children: [{ name: "g", kind: 12, range: r(1, 0, 3, 0), children: [] }] },
]);
check("symbols: non-function kinds never pin", findEnclosingSymbol(vf, 4, 0) === null && findEnclosingSymbol(vf, 2, 0)?.name === "g");

const sLspInner = scopeFromSymbols(syms, DOC, 10, 3);
check("symbols: cursor in nested function pins the inner one", sLspInner.mode === "pin+outline" && !!sLspInner.pin && sLspInner.pin.startLine === 9 && sLspInner.pin.endLine === 12);
check("symbols: nested pin keeps the outer top-level outline entry", json(sLspInner.outline) === json(["3 summarise_data", "9 fit_model"]));
const sLspFit = scopeFromSymbols(syms, DOC, 13, 2);
check("symbols: pin dedups the pinned function's outline entry", sLspFit.mode === "pin+outline" && json(sLspFit.outline) === json(["3 summarise_data"]));
const sLspTop = scopeFromSymbols(syms, DOC, 0, 0);
check("symbols: top-level cursor = outline mode, no pin", sLspTop.mode === "outline" && sLspTop.pin === null && json(sLspTop.outline) === json(["3 summarise_data", "9 fit_model"]));

// --- scope modes (scan path) ----------------------------------------------------

check("scan scope: inside a function", scopeFromScan(DOC, 11).mode === "pin+outline");
check("scan scope: top level", scopeFromScan(DOC, 0).mode === "outline");
check("scan scope: file with no functions is plain (off)", scopeFromScan(["x <- 1"], 0).mode === "off");

// --- prompt build: legacy parity (v0.0.6 bytes) ---------------------------------

check("legacy: scope null renders the v0.0.6 prompt (line start)", buildScopedPrompt(DOC, 10, 0, "R/foo.R", null).prompt === legacyPrompt(DOC, 10, 0, "R/foo.R"));
check("legacy: scope null renders the v0.0.6 prompt (mid-line)", buildScopedPrompt(DOC, 11, 5, "R/foo.R", null).prompt === legacyPrompt(DOC, 11, 5, "R/foo.R"));
const LONG: string[] = [];
for (let i = 0; i < 300; i++) LONG.push(`line${i} <- ${"z".repeat(30)} # pad ${"p".repeat(20)}`);
const legacyLong = legacyPrompt(LONG, 280, 10, "R/big.R");
check("legacy: prefix truncation identical under scope null", buildScopedPrompt(LONG, 280, 10, "R/big.R", null).prompt === legacyLong);
check("legacy: prefix truncation actually engages on the long doc", buildScopedPrompt(LONG, 280, 10, "R/big.R", null).truncatedLines > 0);
check("legacy: enabled but nothing found = plain prompt", buildScopedPrompt(LONG, 280, 10, "R/big.R", scopeFromScan(LONG, 280)).prompt === legacyLong);
check("legacy: enabled on a no-function file = plain prompt", buildScopedPrompt(["x <- 1"], 0, 5, "R/a.R", scopeFromScan(["x <- 1"], 0)).prompt === legacyPrompt(["x <- 1"], 0, 5, "R/a.R"));

// --- prompt build: outline slot placement ----------------------------------------

const pOut = buildScopedPrompt(DOC, 0, 0, "R/foo.R", scopeFromScan(DOC, 0)).prompt;
check("outline: slot renders between the file prefix and the CURRENT marker", pOut.includes("<|outline|>\n3 summarise_data\n9 fit_model\n<<<<<<< CURRENT"));
check("outline: file prefix before the slot is byte-identical to the plain prompt", pOut.split("<|outline|>")[0] === legacyPrompt(DOC, 0, 0, "R/foo.R").split("<<<<<<< CURRENT")[0]);

// --- prompt build: pin protects the suffix, deepest lines truncate -----------------

const BIG: string[] = ["f <- function(x) {", "  keep_me_pinned <- 1", "  also_pinned <- 2", "}"];
for (let i = 0; i < 500; i++) BIG.push(`below${i} <- 1  # ${"b".repeat(30)}`);
const bigBuild = buildScopedPrompt(BIG, 1, 0, "R/big.R", scopeFromScan(BIG, 1)); // cursor at line start
check("suffix: over-budget suffix truncates (from the end)", bigBuild.truncatedSuffixLines > 0);
check(
  "suffix: the pinned function leads the suffix and survives truncation",
  bigBuild.prompt.includes("  keep_me_pinned <- 1\n  also_pinned <- 2\n}\nbelow0 <- 1"),
);
check("suffix: the cut hits the deepest lines, the shallowest stay", bigBuild.prompt.includes("below0 <-") && !bigBuild.prompt.includes("below400 <-"));
check("suffix: total prompt stays under ~8000 chars", bigBuild.prompt.length < 8000);

// --- stable-prefix property: typing more on the cursor line ------------------------

// cursor at END of the typed line: after-cursor text is empty, so the whole
// context (suffix + prefix + outline) must be shared bytes and the prompts
// can only diverge at the cursor zone
const typedDoc = (typed: string): string[] => ["helper <- function(x) {", "  y <- x * 2", typed, "  y", "}", "other <- function() {", "  1", "}"];
const typedStates = ["  total <- su", "  total <- sum", "  total <- sum(data)"];
const endScopes: ScopeInfo[] = [];
const endPrompts: string[] = [];
for (const t of typedStates) {
  const lines = typedDoc(t);
  const s = scopeFromScan(lines, 2);
  endScopes.push(s);
  endPrompts.push(buildScopedPrompt(lines, 2, t.length, "R/typed.R", s).prompt);
}
check("stable: typing on the cursor line does not move the pin", json(endScopes[0].pin) === json(endScopes[1].pin) && json(endScopes[1].pin) === json(endScopes[2].pin));
check("stable: typing on the cursor line does not move the outline", json(endScopes[0].outline) === json(endScopes[1].outline) && json(endScopes[1].outline) === json(endScopes[2].outline));
check(
  "stable: prompt prefix identical up to the cursor zone",
  commonPrefixLen(endPrompts[0], endPrompts[1]) >= endPrompts[0].indexOf("<<<<<<< CURRENT") &&
    commonPrefixLen(endPrompts[1], endPrompts[2]) >= endPrompts[1].indexOf("<<<<<<< CURRENT"),
);

// cursor MID-line (after-cursor text changes too): everything except the
// two cursor-line-derived lines must be byte-identical
const midLine = "  total <- sum(x, na.rm)";
const midScopes: ScopeInfo[] = [];
const midPrompts: string[] = [];
for (const ch of [12, 14, 17]) {
  const lines = typedDoc(midLine);
  const s = scopeFromScan(lines, 2);
  midScopes.push(s);
  midPrompts.push(buildScopedPrompt(lines, 2, ch, "R/typed.R", s).prompt);
}
check("stable (mid-line): pin unchanged", json(midScopes[0].pin) === json(midScopes[1].pin) && json(midScopes[1].pin) === json(midScopes[2].pin));
check("stable (mid-line): outline unchanged", json(midScopes[0].outline) === json(midScopes[1].outline) && json(midScopes[1].outline) === json(midScopes[2].outline));
check(
  "stable (mid-line): prompts identical except the two cursor-line lines",
  stabilize(midPrompts[0]) === stabilize(midPrompts[1]) && stabilize(midPrompts[1]) === stabilize(midPrompts[2]),
);

// --- summary ---------------------------------------------------------------------

if (failures > 0) {
  console.error(`check-context: ${failures}/${checks} checks FAILED`);
  process.exit(1);
}
console.log(`check-context: OK (${checks} checks)`);
