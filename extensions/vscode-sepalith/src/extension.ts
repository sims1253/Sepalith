import { parseCompletion } from "./completion";
import { loadManifest, provision, sharedCacheRoot } from "./runtime";
import * as vscode from "vscode";
import * as fs from "node:fs";
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { terminateChild } from "./process_lifecycle";
import { buildScopedPrompt, normalizeSymbols, scopeFromScan, scopeFromSymbols } from "./context_build";
import type { RawSymbol, ScopeInfo } from "./context_build";

// Sepalith v0 — see SPEC.md. This is END-TO-END PLUMBING, not suggestion
// quality: garbage suggestions from the current weak models are acceptable.

const DEFAULT_MODEL = "";
let runtimeStorage = "";
const DEFAULT_SERVER = "";

interface Config {
  manifestUrl: string;
  backend: "auto" | "cpu" | "vulkan" | "metal";
  gpuLayers: number;
  modelPath: string;
  serverPath: string;
  port: number;
  threads: number;
  contextSize: number;
  autoStart: boolean;
  debounceMs: number;
  postAcceptCooldown: boolean;
  debugMode: boolean;
  scopeContext: boolean;
}

function cfg(): Config {
  const c = vscode.workspace.getConfiguration("sepalith");
  return {
    manifestUrl: c.get("manifestUrl", ""),
    backend: c.get("backend", "auto"),
    gpuLayers: c.get("gpuLayers", 0),
    modelPath: c.get("modelPath", DEFAULT_MODEL),
    serverPath: c.get("serverPath", DEFAULT_SERVER),
    port: c.get("port", 18099),
    threads: c.get("threads", 8),
    contextSize: c.get("contextSize", 8192),
    autoStart: c.get("autoStart", true),
    debounceMs: c.get("debounceMs", 1500),
    postAcceptCooldown: c.get("postAcceptCooldown", true),
    debugMode: c.get("debugMode", false),
    scopeContext: c.get("scopeContext", true),
  };
}

// ---------------------------------------------------------------------------
// Prompt render (must match experiments/eval/run_eval.py render_zeta2) and
// the scope-aware context additions live in ./context_build — pure, so
// scripts/check-context.ts can exercise them under plain node. This wrapper
// supplies only the vscode bits (document text, relative path).
// scope === null renders the byte-identical v0.0.6 prompt.
// ---------------------------------------------------------------------------

function buildPrompt(document: vscode.TextDocument, position: vscode.Position, scope: ScopeInfo | null) {
  return buildScopedPrompt(
    document.getText().split("\n"),
    position.line,
    position.character,
    vscode.workspace.asRelativePath(document.uri),
    scope,
  );
}

// marker lines the model sometimes echoes from the prompt back into its
// completion (seen live: an empty-file proposal consisting mostly of
// "<<<<<<< CURRENT / ======= / <[fim-middle]" lines)
const MARKER_LINE = /^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|suffix)\]>|<\|user_cursor\|>|<\|outline\|>)\s*$/;

function parsePrediction(text: string): string[] {
  // everything before the first ">>>>>>>" is the predicted region
  if (text.includes(">>>>>>>")) text = text.split(">>>>>>>")[0];
  text = text.split("<|user_cursor|>").join("");
  const lines = text
    .split("\n")
    .map((l) => l.replace(/\r$/, ""))
    .filter((l) => !MARKER_LINE.test(l));
  // strip leading blank lines (and trailing, matching run_eval.py norm())
  while (lines.length && lines[0].trim() === "") lines.shift();
  while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
  // degenerate repetition (temp-0 small models: "  return(x^2)" x N) —
  // cut at the third identical consecutive line, keeping the first two
  for (let i = 2; i < lines.length; i++) {
    if (lines[i] === lines[i - 1] && lines[i] === lines[i - 2]) {
      lines.length = i;
      break;
    }
  }
  return lines;
}

// ---------------------------------------------------------------------------
// HTTP (llama-server on 127.0.0.1)
// ---------------------------------------------------------------------------

interface CompletionResult {
  text: string;
  completionTokens: number;
}

// generation stops: the UPDATED terminator plus every prompt marker the
// model has been seen echoing (marker echo = wasted seconds of generation)
const STOPS = [">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======", "<[fim-middle]>", "<[fim-suffix]>", "<[fim-prefix]>", "<|outline|>"];

async function postCompletion(port: number, prompt: string, maxTokens: number, stop: string[] | null, signal?: AbortSignal): Promise<CompletionResult> {
  const body = JSON.stringify({ prompt, max_tokens: maxTokens, temperature: 0, stop, stream: false });
  const res = await fetch(`http://127.0.0.1:${port}/v1/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = parseCompletion(await res.json());
  return data;
}

async function portAnswers(port: number): Promise<boolean> {
  try {
    await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1500) });
    return true; // any HTTP answer counts as "the port is answering"
  } catch {
    return false;
  }
}

function errText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Sidecar lifecycle (the four hard rules in SPEC.md)
// ---------------------------------------------------------------------------

type SidecarState = "starting" | "ready" | "external" | "stopped" | "error";

class Sidecar {
  private child: ChildProcess | null = null;
  private setup: AbortController | null = null;
  private state: SidecarState = "stopped";
  private detail = "";
  private outRing: string[] = [];
  private errRing: string[] = [];
  private pendingOut = "";
  private pendingErr = "";
  private stopping = false;

  get currentState(): SidecarState {
    return this.state;
  }

  get detailText(): string {
    return this.detail; // error message / model path, for the status-bar tooltip
  }

  private setState(s: SidecarState, detail = ""): void {
    this.state = s;
    this.detail = detail;
    renderStatusBar();
  }

  async start(forceCpu = false): Promise<void> {
    if (this.child || this.state === "starting" || this.state === "ready" || this.state === "external") return;
    const c = cfg();
    this.setState("starting");
    this.setup = new AbortController();
    const setup = this.setup;
    // rule 4: if the port is already answering, treat the server as external — use it, never spawn, never kill
    const occupied = await portAnswers(c.port);
    if (setup.signal.aborted) return;
    if (occupied) {
      try {
        await postCompletion(c.port, "x", 1, null, AbortSignal.timeout(5000));
        if (setup.signal.aborted) return;
        channel.appendLine(`sidecar: port ${c.port} passed completion probe — treating server as external`);
        this.setState("external");
      } catch (e) {
        if (!setup.signal.aborted) this.fail(`Port ${c.port} is occupied but failed the completion probe: ${errText(e)}`);
      }
      return;
    }
    if (setup.signal.aborted) return;
    if (!c.serverPath && c.manifestUrl) {
      try {
        Object.assign(c, await provision(c.manifestUrl, runtimeStorage, forceCpu ? "cpu" : c.backend,
          setup.signal, message => channel.appendLine(message), c.modelPath));
      } catch (e) {
        if (!setup.signal.aborted) this.fail(errText(e));
        return;
      }
    }
    if (setup.signal.aborted) return;
    for (const p of [c.serverPath, c.modelPath]) {
      if (!fs.existsSync(p)) {
        channel.appendLine(`sidecar: error: missing ${p}`);
        this.setState("error", `Set modelPath/serverPath or a release manifest URL (${p || "path unset"})`);
        return;
      }
    }
    this.setState("starting");
    channel.appendLine(`sidecar: spawning ${c.serverPath} --port ${c.port} -t ${c.threads} -c ${c.contextSize} -ngl ${c.gpuLayers}`);
    this.stopping = false;
    this.outRing = [];
    this.errRing = [];
    this.pendingOut = "";
    this.pendingErr = "";
    // rule 1: exact spawn args, array form, no shell — WSL/Remote-safe
    const child = spawn(
      c.serverPath,
      [
        "-m", c.modelPath,
        "--alias", "sepalith",
        "--temp", "0",
        "--port", String(c.port),
        "--host", "127.0.0.1",
        "-c", String(c.contextSize),
        "--parallel", "1",
        "-t", String(c.threads),
        "-ngl", String(c.gpuLayers),
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    this.child = child;
    child.stdout?.on("data", (d: Buffer) => this.pushRing("out", d));
    child.stderr?.on("data", (d: Buffer) => this.pushRing("err", d));
    const failed = (message: string) => {
      if (!setup.signal.aborted && !forceCpu && c.manifestUrl && !cfg().serverPath && c.gpuLayers > 0) {
        channel.appendLine(`${message}; retrying with CPU runtime`);
        this.setState("stopped");
        void this.start(true).catch(e => this.fail(errText(e)));
      } else if (!setup.signal.aborted) this.fail(message);
    };
    child.on("error", (err) => {
      if (this.child !== child) return;
      // An error can also mean signaling failed; it does not prove exit.
      if (child.pid && child.exitCode === null && child.signalCode === null) {
        this.fail(`owned server error: ${err.message}`);
        return;
      }
      this.child = null;
      failed(`spawn failed: ${err.message}`);
    });
    child.on("exit", (code, sig) => {
      if (this.child !== child) return;
      this.child = null;
      if (this.stopping) {
        this.setState("stopped");
        return;
      }
      failed(`server exited (code=${code} sig=${sig})`);
    });
    // rule 2: /health answers ok DURING load — poll completions until HTTP 200 (3 s interval, 180 s budget)
    const t0 = Date.now();
    for (;;) {
      if (setup.signal.aborted || this.child !== child) return; // This start was stopped or replaced.
      if (Date.now() - t0 > 180_000) {
        try {
          await this.stop();
          this.fail("readiness timeout after 180 s");
        } catch (error) { this.fail(errText(error)); }
        return;
      }
      try {
        await postCompletion(c.port, "x", 1, null, AbortSignal.timeout(5000));
        break;
      } catch (e) {
        channel.appendLine(`sidecar: not ready (${Math.round((Date.now() - t0) / 1000)} s): ${errText(e)}`);
      }
      await sleep(3000);
    }
    if (setup.signal.aborted || this.child !== child) return;
    channel.appendLine(`sidecar: ready after ${Math.round((Date.now() - t0) / 1000)} s`);
    this.setState("ready", c.modelPath);
  }

  async stop(): Promise<void> {
    this.setup?.abort();
    if (this.state === "external") return; // Never stop an unowned server.
    const child = this.child;
    this.stopping = true;
    if (child) {
      try {
        await terminateChild(child);
      } catch (error) {
        // Keep the handle: a failed stop must not permit a duplicate start.
        this.fail(errText(error));
        throw error;
      }
      if (this.child === child) this.child = null;
    }
    this.setState("stopped");
  }

  private pushRing(which: "out" | "err", d: Buffer): void {
    const chunk = (which === "out" ? (this.pendingOut += d.toString()) : (this.pendingErr += d.toString()));
    const parts = chunk.split("\n");
    const rest = parts.pop() ?? "";
    if (which === "out") this.pendingOut = rest;
    else this.pendingErr = rest;
    const ring = which === "out" ? this.outRing : this.errRing;
    ring.push(...parts);
    if (ring.length > 50) ring.splice(0, ring.length - 50); // ring buffer: last 50 lines
  }

  private fail(msg: string): void {
    this.setState("error", msg);
    channel.appendLine(`sidecar: error: ${msg}`);
    // dump the ring-buffered tails into the channel on error
    channel.appendLine("--- server stdout tail ---");
    channel.appendLine(this.outRing.join("\n"));
    channel.appendLine("--- server stderr tail ---");
    channel.appendLine(this.errRing.join("\n"));
  }

  async noteRequestError(msg: string): Promise<void> {
    channel.appendLine(`request error: ${msg}`);
    if (this.state === "external") return; // not ours to diagnose
    if (await portAnswers(cfg().port)) return; // server still alive — transient error, keep state
    this.fail(`request failed: ${msg}`);
  }
}

// ---------------------------------------------------------------------------
// Request log + debug tree view (sidebar: Explorer > Sepalith Requests)
// ---------------------------------------------------------------------------

interface ReqEntry {
  id: number;
  time: string;
  promptChars: number;
  latencyMs: number | null; // null = in flight / failed
  completionTokens: number | null;
  outcome: "ok" | "in-flight" | "aborted" | "error" | "skipped" | "cached";
  note: string; // skip reason / error text / "cache hit"
  prompt: string;
  raw: string;
  parsed: string[];
}

class RequestLog implements vscode.TreeDataProvider<ReqEntry | { parent: ReqEntry; kind: string }> {
  private readonly entries: ReqEntry[] = [];
  private nextId = 1;
  private readonly emitter = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.event();

  event() {
    return this.emitter.event;
  }

  begin(prompt: string, promptChars: number, outcome: ReqEntry["outcome"], note = ""): ReqEntry {
    const e: ReqEntry = {
      id: this.nextId++,
      time: new Date().toLocaleTimeString(),
      promptChars,
      latencyMs: null,
      completionTokens: null,
      outcome,
      note,
      prompt,
      raw: "",
      parsed: [],
    };
    this.entries.unshift(e);
    if (this.entries.length > 100) this.entries.length = 100;
    this.emitter.fire();
    return e;
  }

  finish(e: ReqEntry, latencyMs: number, completionTokens: number, raw: string, parsed: string[]): void {
    e.latencyMs = latencyMs;
    e.completionTokens = completionTokens;
    e.outcome = "ok";
    e.raw = raw;
    e.parsed = parsed;
    this.emitter.fire();
  }

  fail(e: ReqEntry, outcome: ReqEntry["outcome"], note: string): void {
    e.outcome = outcome;
    e.note = note;
    this.emitter.fire();
  }

  clear(): void {
    this.entries.length = 0;
    this.emitter.fire();
  }

  getTreeItem(node: ReqEntry | { parent: ReqEntry; kind: string }): vscode.TreeItem {
    if ("parent" in node) {
      const p = node.parent;
      const item = new vscode.TreeItem(node.kind);
      item.collapsibleState = vscode.TreeItemCollapsibleState.None;
      if (node.kind === "prompt") {
        item.description = `${p.promptChars} chars — click to copy`;
        item.tooltip = p.prompt.slice(0, 3000);
        item.command = {
          title: "Copy prompt", command: "sepalith.copyText",
          arguments: [p.prompt],
        };
      } else if (node.kind === "raw completion") {
        item.description = `${(p.raw || "(none)").split("\n").length} lines`;
        item.tooltip = p.raw.slice(0, 3000) || "(none)";
        item.command = { title: "Copy", command: "sepalith.copyText", arguments: [p.raw] };
      } else if (node.kind === "parsed prediction") {
        item.description = `${p.parsed.length} lines`;
        item.tooltip = p.parsed.join("\n").slice(0, 3000) || "(empty)";
        item.command = { title: "Copy", command: "sepalith.copyText", arguments: [p.parsed.join("\n")] };
      }
      return item;
    }
    const e = node;
    const stat = e.latencyMs !== null ? `${e.latencyMs} ms` : e.outcome;
    const item = new vscode.TreeItem(
      `#${e.id} ${e.time} · ${e.outcome} · ${stat}${e.completionTokens !== null ? ` · ${e.completionTokens} tok` : ""}`,
    );
    item.description = `${e.promptChars} prompt chars`;
    item.tooltip = e.note || e.outcome;
    item.collapsibleState = vscode.TreeItemCollapsibleState.Collapsed;
    item.contextValue = "request";
    return item;
  }

  getChildren(node?: ReqEntry | { parent: ReqEntry; kind: string }) {
    if (!node) return this.entries;
    if ("parent" in node) return [];
    return (["prompt", "raw completion", "parsed prediction"] as const).map((kind) => ({ parent: node, kind }));
  }
}

// ---------------------------------------------------------------------------
// Inline completion provider
// ---------------------------------------------------------------------------

export class SepalithProvider implements vscode.InlineCompletionItemProvider {
  private controller: AbortController | null = null;
  private generation = 0;
  // Local counters use stable APIs: offered results and full accept commands.
  // Partial acceptance and actual display events require proposed editor APIs.
  static offered = 0;
  static accepted = 0;
  // identical prompts: reuse the in-flight promise or the cached result —
  // VS Code re-invokes the provider on every cursor move, which previously
  // aborted and re-sent the SAME request dozens of times per keystroke
  private lastPrompt = "";

  invalidate(): void {
    this.generation++;
    this.controller?.abort();
    this.lastItems = null;
    this.inFlight = null;
  }
  private lastItems: vscode.InlineCompletionItem[] | null = null;
  private inFlight: { prompt: string; promise: Promise<vscode.InlineCompletionItem[]> } | null = null;
  private lastSkipLog = 0;
  // LSP document symbols, cached per document VERSION (a pure cursor move
  // reuses them; any edit invalidates) — the scope path never re-queries
  // the language server for the same document state
  private symCache = new Map<string, { version: number; symbols: RawSymbol[] | null }>();

  private async fetchSymbols(document: vscode.TextDocument): Promise<RawSymbol[] | null> {
    const key = document.uri.toString();
    const hit = this.symCache.get(key);
    if (hit && hit.version === document.version) return hit.symbols;
    let symbols: RawSymbol[] | null = null;
    try {
      // huge files can make the symbol provider slow: race it, never let it
      // stall the request path longer than 500 ms
      const raw = await Promise.race([
        vscode.commands.executeCommand("vscode.executeDocumentSymbolProvider", document.uri),
        sleep(500).then(() => null),
      ]);
      symbols = normalizeSymbols(raw);
      if (symbols.length === 0) symbols = null; // no R LSP answering — brace-scan fallback
    } catch {
      symbols = null;
    }
    if (this.symCache.size > 16) this.symCache.clear(); // a few open files is the real working set
    this.symCache.set(key, { version: document.version, symbols });
    return symbols;
  }

  // scope-aware context (docs/prompt-format.md): enclosing-function pin +
  // file outline, from the R language server's document symbols when it
  // answers, from the brace-scan heuristic when it does not. Computed here
  // — inside the debounced request path — so the symbol fetch never blocks
  // anything outside this request.
  private async buildScopeContext(document: vscode.TextDocument, position: vscode.Position): Promise<ScopeInfo> {
    const lines = document.getText().split("\n");
    const symbols = await this.fetchSymbols(document);
    return symbols
      ? scopeFromSymbols(symbols, lines, position.line, position.character)
      : scopeFromScan(lines, position.line);
  }

  private skipLogRateLimited(msg: string): void {
    const now = Date.now();
    if (now - this.lastSkipLog > 30_000) {
      this.lastSkipLog = now;
      channel.appendLine(msg);
    }
  }

  async provideInlineCompletionItems(document: vscode.TextDocument, position: vscode.Position, _context: vscode.InlineCompletionContext, token: vscode.CancellationToken): Promise<vscode.InlineCompletionItem[]> {
    const version = document.version;
    if (cfg().postAcceptCooldown && acceptedDocument?.uri === document.uri.toString() && acceptedDocument.version === version) return [];
    const generation = this.generation;
    const current = () => generation === this.generation && !token.isCancellationRequested && !document.isClosed && document.version === version;
    if (!current()) return [];
    if (sidecar.currentState !== "ready" && sidecar.currentState !== "external") {
      this.skipLogRateLimited(`request skipped: sidecar is ${sidecar.currentState}`);
      return [];
    }
    // an empty (or near-empty) document gives the model nothing to work
    // with and reliably produces roxygen hallucinations — propose nothing
    if (document.getText().trim().length < 30) {
      this.skipLogRateLimited("request skipped: document is empty");
      return [];
    }
    const c = cfg();
    const scope = c.scopeContext ? await this.buildScopeContext(document, position) : null;
    if (!current()) return [];
    logScopeMode(scope ? scope.mode : "off");
    const { prompt, truncatedLines, truncatedSuffixLines } = buildPrompt(document, position, scope);
    if (truncatedLines > 0) channel.appendLine(`prompt: truncated ${truncatedLines} lines from the start of the prefix`);
    if (truncatedSuffixLines > 0) {
      channel.appendLine(`prompt: truncated ${truncatedSuffixLines} lines from the end of the suffix (enclosing function protected)`);
    }

    lastPrompt = prompt;
    const key = JSON.stringify([document.uri.toString(), version, position.line, position.character, c, prompt]);
    if (this.lastItems && key === this.lastPrompt) {
      requestLog.begin(prompt, prompt.length, "cached", "identical prompt — reused cached result");
      this.skipLogRateLimited("request: identical prompt — cached result");
      return this.lastItems;
    }
    if (this.inFlight && !this.controller?.signal.aborted && key === this.inFlight.prompt) {
      this.skipLogRateLimited("request: identical prompt already in flight — joining it");
      const pending = this.inFlight;
      const controller = this.controller;
      const subscription = token.onCancellationRequested(() => controller?.abort());
      try { return await pending.promise; } finally { subscription.dispose(); }
    }

    const entry = requestLog.begin(prompt, prompt.length, "in-flight");
    channel.appendLine(`request: ${prompt.length} prompt chars`);
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const subscription = token.onCancellationRequested(() => controller.abort());
    let promise: Promise<vscode.InlineCompletionItem[]>;
    promise = (async () => {
      try {
        const t0 = Date.now();
        const r = await postCompletion(c.port, prompt, 320, STOPS, controller.signal);
        if (!current() || controller.signal.aborted) {
          requestLog.fail(entry, "aborted", "document changed or request cancelled");
          return [];
        }
        lastStats = `latency ${Date.now() - t0} ms, completion tokens ${r.completionTokens}`;
        channel.appendLine(`response: ${lastStats}`);
        if (c.debugMode) {
          channel.appendLine(`--- prompt (${prompt.length} chars) ---\n${prompt}\n--- raw completion ---\n${r.text}`);
        }
        renderStatusBar();
        const lines = parsePrediction(r.text);
        requestLog.finish(entry, Date.now() - t0, r.completionTokens, r.text, lines);
        if (lines.length === 0) {
          this.lastPrompt = key;
          this.lastItems = [];
          return [];
        }
        // cursor on a non-empty comment line + code-looking first prediction:
        // start the prediction on the NEXT line instead of gluing code into
        // the comment (seen live: roxygen title + "  if (is.list(x)) {")
        const currentLine = document.lineAt(position.line).text.trim();
        const codeFirst = /^[A-Za-z.][\w.$]*\s*(<-|=|\()/.test(lines[0].trim());
        if (currentLine.startsWith("#") && currentLine !== "#" && codeFirst) {
          lines.unshift("");
        }
        // the dominant finish-block training convention re-emits the WHOLE
        // region (typed partial included) as the target. If the prediction
        // starts with the typed partial, it is a full-region output: make
        // Tab replace the entire cursor line instead of appending after the
        // partial (which produced "sum(x) / length(x)  sum(x) / length(x)").
        const typedPartial = document.lineAt(position.line).text.slice(0, position.character).trim();
        let rangeStart = position;
        if (typedPartial.length >= 4 && lines[0].trim().startsWith(typedPartial)) {
          rangeStart = position.with({ character: 0 });
        }
        const eol = document.lineAt(position.line).range.end;
        const item = new vscode.InlineCompletionItem(lines.join("\n"), new vscode.Range(rangeStart, eol));
        item.command = { command: "sepalith.accepted", title: "Record accepted suggestion" };
        SepalithProvider.offered++;
        this.lastPrompt = key;
        this.lastItems = [item];
        return [item];
      } catch (e) {
        if (controller.signal.aborted) {
          requestLog.fail(entry, "aborted", "superseded by a new request");
          return [];
        }
        requestLog.fail(entry, "error", errText(e));
        void sidecar.noteRequestError(errText(e));
        return [];
      } finally {
        subscription.dispose();
        if (this.controller === controller) this.inFlight = null;
      }
    })();
    this.inFlight = { prompt: key, promise };
    this.lastItems = null; // a different prompt invalidates the cache
    return promise;
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

let channel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;
let requestLog: RequestLog;
let lastStats = "no requests yet";
let lastPrompt = "(no request yet)";
let debounceTimer: ReturnType<typeof setTimeout> | null = null;
let acceptedDocument: { uri: string; version: number } | null = null;
const sidecar = new Sidecar();

function modelName(): string {
  const m = cfg().modelPath;
  return m.slice(m.lastIndexOf("/") + 1); // basename is enough for the status bar
}

function renderStatusBar(): void {
  const s = sidecar.currentState;
  const label = s === "starting" ? "starting…" : s === "ready" ? `ready (${modelName()})` : s === "external" ? "external server" : s;
  statusBarItem.text = `Sepalith: ${label}`;
  statusBarItem.tooltip = `Sepalith sidecar\nstate: ${s}${sidecar.detailText ? `\n${sidecar.detailText}` : ""}\nlast request: ${lastStats}\nsuggestions offered ${SepalithProvider.offered} / accepted ${SepalithProvider.accepted}`;
}

// which context mode built the prompt (scope:pin+outline / scope:outline /
// scope:off) — one line per mode CHANGE, or every 50 requests, never per
// request
let lastScopeMode = "";
let scopeModeRequests = 0;
function logScopeMode(mode: string): void {
  scopeModeRequests++;
  if (mode !== lastScopeMode || scopeModeRequests % 50 === 0) {
    lastScopeMode = mode;
    channel.appendLine(`scope:${mode}`);
  }
}

function triggerInlineSuggestion(): void {
  void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
}

export function activate(context: vscode.ExtensionContext): void {
  const provider = new SepalithProvider();
  runtimeStorage = sharedCacheRoot();
  channel = vscode.window.createOutputChannel("Sepalith");
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.show();
  requestLog = new RequestLog();
  context.subscriptions.push(
    channel,
    statusBarItem,
    vscode.window.createTreeView("sepalith.requests", { treeDataProvider: requestLog }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("sepalith.startServer", () => { provider.invalidate(); return sidecar.start(); }),
    vscode.commands.registerCommand("sepalith.accepted", () => {
      SepalithProvider.accepted++;
      const document = vscode.window.activeTextEditor?.document;
      acceptedDocument = document ? { uri: document.uri.toString(), version: document.version } : null;
      provider.invalidate();
      if (cfg().postAcceptCooldown && debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      renderStatusBar();
    }),
    vscode.commands.registerCommand("sepalith.stopServer", () => { provider.invalidate(); return sidecar.stop(); }),
    vscode.commands.registerCommand("sepalith.refreshRuntime", async () => {
      const url = cfg().manifestUrl;
      if (!url) {
        channel.appendLine("Set sepalith.manifestUrl before refreshing a managed runtime.");
        channel.show(true);
        return;
      }
      try {
        await loadManifest(url, runtimeStorage, new AbortController().signal, { refreshManifest: true });
        channel.appendLine("Runtime manifest refreshed. Stop and start the server to apply it.");
      } catch (error) { channel.appendLine(`Runtime refresh failed: ${errText(error)}`); }
      channel.show(true);
    }),
    // manual suggest: bypasses the debounce entirely
    vscode.commands.registerCommand("sepalith.suggest", () => {
      acceptedDocument = null; // an explicit user request ends the cooldown
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      triggerInlineSuggestion();
    }),
    vscode.commands.registerCommand("sepalith.copyText", (text: string) => {
      void vscode.env.clipboard.writeText(text);
    }),
    vscode.commands.registerCommand("sepalith.copyPrompt", () => {
      void vscode.env.clipboard.writeText(lastPrompt).then(() =>
        vscode.window.showInformationMessage("Sepalith: last prompt copied to clipboard"));
    }),
    vscode.commands.registerCommand("sepalith.showLogs", () => channel.show()),
    vscode.commands.registerCommand("sepalith.dumpStats", () => {
      // local-only acceptance stats (the opt-in upload ships later, off
      // by default per the recorded telemetry decision)
      const line = JSON.stringify({
        time: new Date().toISOString(),
        offered: SepalithProvider.offered,
        accepted: SepalithProvider.accepted,
        lastStats,
      });
      channel.appendLine(`stats: ${line}`);
      vscode.window.showInformationMessage(
        `Sepalith: offered ${SepalithProvider.offered}, accepted ${SepalithProvider.accepted} (logged)`);
    }),
    vscode.commands.registerCommand("sepalith.clearRequests", () => requestLog.clear()),
    vscode.languages.registerInlineCompletionItemProvider({ language: "r" }, provider),
    { dispose: () => provider.invalidate() },
    vscode.window.onDidChangeActiveTextEditor(() => provider.invalidate()),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId !== "r") return;
      provider.invalidate();
      const ms = cfg().debounceMs;
      if (ms <= 0) return; // 0 disables auto-trigger (manual only)
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        triggerInlineSuggestion();
      }, ms);
    }),
  );

  renderStatusBar();
  if (cfg().autoStart) void sidecar.start();
}

export async function deactivate(): Promise<void> {
  await sidecar.stop(); // kills only a child we spawned; external servers are left alone
}
