import { completionBody, generateSuggestion, interpretCompletion, parseNativeCompletion, parseTokenize, promptIds, SUGGESTION_MAX_TOKENS, tokenizeBody, WARMUP_MAX_TOKENS } from "./completion";
import type { Interpretation, NativeCompletion } from "./completion";
import { ensureDefaultModel, loadManifest, provision, sharedCacheRoot } from "./runtime";
import * as vscode from "vscode";
import * as fs from "node:fs";
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { terminateChild } from "./process_lifecycle";
import { BOS_ID, makeApplicationPlan, ProtocolError } from "./prm03";
import { buildEditorRequest, EditHistory, logicalLines, presentEdit, promptPath, sha256Text } from "./prm03_context";
import type { EditorRequest, Presentation } from "./prm03_context";
import { DEFAULT_CONTEXT_SIZE, llamaServerArgs } from "./server_args";
import { WarmupScheduler } from "./warmup";

// Sepalith: PRM03 (`zeta2-prm03-v1`) next-edit suggestions for R from a local
// llama-server sidecar running the arm A GGUF. The prompt and output contract
// live in ./prm03 (a port of sepalith.campaign_protocol), the editor-to-context
// rules in ./prm03_context, the request bodies in ./completion.

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
  warmup: boolean;
  allowNoEdit: boolean;
  speculative: boolean;
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
    contextSize: c.get("contextSize", DEFAULT_CONTEXT_SIZE),
    autoStart: c.get("autoStart", true),
    debounceMs: c.get("debounceMs", 1500),
    postAcceptCooldown: c.get("postAcceptCooldown", true),
    debugMode: c.get("debugMode", false),
    warmup: c.get("warmup", true),
    allowNoEdit: c.get("allowNoEdit", true),
    speculative: c.get("speculative", false),
  };
}

// ---------------------------------------------------------------------------
// Recent-edit history per open R document (./prm03_context EditHistory)
// ---------------------------------------------------------------------------

interface Tracked { history: EditHistory; lines: string[]; version: number }
const tracked = new Map<string, Tracked>();

function modelPath(uri: vscode.Uri): string {
  const relative = vscode.workspace.asRelativePath(uri, false);
  // outside the workspace asRelativePath returns the absolute path: use the file name
  return promptPath(relative === uri.fsPath || relative.startsWith("/") ? uri.path.slice(uri.path.lastIndexOf("/") + 1) : relative);
}

function track(document: vscode.TextDocument): Tracked {
  const key = document.uri.toString();
  let entry = tracked.get(key);
  if (!entry) {
    entry = { history: new EditHistory(key, () => modelPath(document.uri)), lines: logicalLines(document.getText()), version: document.version };
    tracked.set(key, entry);
  }
  return entry;
}

function observeChange(document: vscode.TextDocument): void {
  const known = tracked.get(document.uri.toString());
  const after = logicalLines(document.getText());
  if (!known) { track(document); return; }
  known.history.observe(known.lines, after, known.version);
  known.lines = after;
  known.version = document.version;
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

function editorRequest(document: vscode.TextDocument, position: vscode.Position): EditorRequest {
  const text = document.getText();
  return buildEditorRequest({
    lines: logicalLines(text),
    eol: document.eol === vscode.EndOfLine.CRLF ? "crlf" : "lf",
    cursorLine: position.line,
    cursorUtf16: position.character,
    path: modelPath(document.uri),
    uri: document.uri.toString(),
    version: document.version,
    contentSha256: sha256Text(text),
    history: track(document).history.events(),
  });
}

// ---------------------------------------------------------------------------
// HTTP (llama-server on 127.0.0.1)
// ---------------------------------------------------------------------------

async function postJson(port: number, endpoint: string, body: string, signal?: AbortSignal): Promise<Response> {
  const res = await fetch(`http://127.0.0.1:${port}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res;
}

// rendered prompt -> prompt IDs; warm-ups and the real request render the same
// prompt, so the request usually reuses the warm-up's tokenization
const tokenCache = new Map<string, number[]>();

async function promptTokenIds(port: number, prompt: string, signal?: AbortSignal): Promise<number[]> {
  const hit = tokenCache.get(prompt);
  if (hit) return hit;
  const ids = promptIds(parseTokenize(await (await postJson(port, "/tokenize", tokenizeBody(prompt), signal)).json()));
  tokenCache.set(prompt, ids);
  if (tokenCache.size > 8) tokenCache.delete(tokenCache.keys().next().value ?? "");
  return ids;
}

async function postCompletion(port: number, body: string, signal?: AbortSignal): Promise<NativeCompletion> {
  return parseNativeCompletion(await (await postJson(port, "/completion", body, signal)).json());
}

const PROBE_BODY = completionBody([BOS_ID], 1);

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
    tokenCache.clear(); // a different server may use a different tokenizer
    this.setup = new AbortController();
    const setup = this.setup;
    // rule 4: if the port is already answering, treat the server as external — use it, never spawn, never kill
    const occupied = await portAnswers(c.port);
    if (setup.signal.aborted) return;
    if (occupied) {
      try {
        await postCompletion(c.port, PROBE_BODY, AbortSignal.timeout(5000));
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
    if (!c.modelPath && c.serverPath && fs.existsSync(c.serverPath)) {
      try {
        c.modelPath = await ensureDefaultModel(runtimeStorage, setup.signal, message => channel.appendLine(message));
      } catch (e) {
        if (!setup.signal.aborted) this.fail(`default model download failed: ${errText(e)}`);
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
    const args = llamaServerArgs({ modelPath: c.modelPath, port: c.port, threads: c.threads, contextSize: c.contextSize,
      gpuLayers: c.gpuLayers, speculative: c.speculative });
    channel.appendLine(`sidecar: spawning ${c.serverPath} ${args.join(" ")}`);
    this.stopping = false;
    this.outRing = [];
    this.errRing = [];
    this.pendingOut = "";
    this.pendingErr = "";
    tokenCache.clear();
    // array form, no shell — WSL/Remote-safe
    const child = spawn(
      c.serverPath,
      args,
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
        await postCompletion(c.port, PROBE_BODY, AbortSignal.timeout(5000));
        break;
      } catch (e) {
        channel.appendLine(`sidecar: not ready (${Math.round((Date.now() - t0) / 1000)} s): ${errText(e)}`);
      }
      await sleep(3000);
    }
    if (setup.signal.aborted || this.child !== child) return;
    channel.appendLine(`sidecar: ready after ${Math.round((Date.now() - t0) / 1000)} s`);
    this.setState("ready", c.modelPath);
    activeModel = c.modelPath;
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

  finish(e: ReqEntry, latencyMs: number, completionTokens: number, raw: string, parsed: string[], note: string): void {
    e.latencyMs = latencyMs;
    e.completionTokens = completionTokens;
    e.outcome = "ok";
    e.raw = raw;
    e.parsed = parsed;
    e.note = note;
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
    item.description = `${e.promptChars} prompt chars${e.note ? ` · ${e.note}` : ""}`;
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
// Edit preview: replacements that ghost text cannot show (renames, rewrites,
// deletions) appear as an after-line hint with the full change on hover;
// Alt+Enter applies it, Escape dismisses it.
// ---------------------------------------------------------------------------

interface PendingEdit {
  uri: string;
  version: number;
  sha256: string;
  edit: Extract<Presentation, { kind: "replace" }>;
}

class EditPreview {
  private pending: PendingEdit | null = null;
  private readonly decoration = vscode.window.createTextEditorDecorationType({
    after: { margin: "0 0 0 2em", color: new vscode.ThemeColor("editorGhostText.foreground"), fontStyle: "italic" },
    isWholeLine: true,
  });

  get line(): number | null {
    return this.pending ? this.pending.edit.line : null;
  }

  show(editor: vscode.TextEditor, edit: PendingEdit["edit"]): void {
    const document = editor.document;
    this.pending = { uri: document.uri.toString(), version: document.version, sha256: sha256Text(document.getText()), edit };
    const newLines = edit.text.split(/\r?\n/);
    const summary = edit.text === "" ? "delete line content" : `${newLines[0].trim()}${newLines.length > 1 ? ` (+${newLines.length - 1} lines)` : ""}`;
    const hover = new vscode.MarkdownString();
    hover.appendMarkdown("**Sepalith edit** (Alt+Enter to apply, Escape to dismiss)\n");
    hover.appendCodeblock([...edit.oldText.split("\n").map(l => `-${l}`), ...(edit.text === "" ? [] : newLines.map(l => `+${l}`))].join("\n"), "diff");
    editor.setDecorations(this.decoration, [{
      range: new vscode.Range(edit.line, 0, edit.line, 0),
      hoverMessage: hover,
      renderOptions: { after: { contentText: `\u21e2 ${summary.length > 80 ? summary.slice(0, 79) + "\u2026" : summary}` } },
    }]);
    void vscode.commands.executeCommand("setContext", "sepalith.editPending", true);
  }

  clear(): void {
    if (!this.pending) return;
    this.pending = null;
    for (const editor of vscode.window.visibleTextEditors) editor.setDecorations(this.decoration, []);
    void vscode.commands.executeCommand("setContext", "sepalith.editPending", false);
  }

  /** Apply the pending edit if the document is still exactly the one it was computed for. */
  async accept(): Promise<boolean> {
    const pending = this.pending;
    this.clear();
    const editor = vscode.window.activeTextEditor;
    if (!pending || !editor) return false;
    const document = editor.document;
    if (document.uri.toString() !== pending.uri || document.version !== pending.version || sha256Text(document.getText()) !== pending.sha256) {
      channel.appendLine("edit preview: document changed, edit discarded");
      return false;
    }
    const { line, endCharacter, text } = pending.edit;
    const edit = new vscode.WorkspaceEdit();
    edit.replace(document.uri, new vscode.Range(line, 0, line, endCharacter), text);
    return vscode.workspace.applyEdit(edit);
  }

  dispose(): void {
    this.clear();
    this.decoration.dispose();
  }
}

// ---------------------------------------------------------------------------
// Inline completion provider
// ---------------------------------------------------------------------------

// an empty (or near-empty) document gives the model nothing to work with
const MIN_DOCUMENT_CHARS = 30;

function describe(outcome: Interpretation): string {
  if (outcome.outcome === "invalid") return `no suggestion (${outcome.reason})`;
  if (outcome.outcome === "no_edit") return "no edit";
  return `${outcome.result.operation} (${outcome.result.body.length} lines)`;
}

export class SepalithProvider implements vscode.InlineCompletionItemProvider {
  private controller: AbortController | null = null;
  private generation = 0;
  // Local counters use stable APIs: offered results and full accept commands.
  // Partial acceptance and actual display events require proposed editor APIs.
  static offered = 0;
  static accepted = 0;
  // identical prompts: reuse the in-flight promise or the cached result —
  // VS Code re-invokes the provider on every cursor move
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

  // warm-up prompt for the active editor's cursor; null = nothing a real
  // request would send (no R editor, server not usable, empty document)
  async warmupPrompt(): Promise<string | null> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "r") return null;
    if (sidecar.currentState !== "ready" && sidecar.currentState !== "external") return null;
    const { document } = editor;
    if (document.getText().trim().length < MIN_DOCUMENT_CHARS) return null;
    try {
      return editorRequest(document, editor.selection.active).prompt;
    } catch {
      return null; // a context the protocol rejects is never requested either
    }
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
    // an empty (or near-empty) document gives the model nothing to work with
    if (document.getText().trim().length < MIN_DOCUMENT_CHARS) {
      this.skipLogRateLimited("request skipped: document is empty");
      return [];
    }
    const c = cfg();
    let request: EditorRequest;
    try {
      request = editorRequest(document, position);
    } catch (e) {
      this.skipLogRateLimited(`request skipped: ${errText(e)}`);
      return [];
    }
    const prompt = request.prompt;
    if (request.omittedLines > 0 && c.debugMode) channel.appendLine(`prompt: ${request.omittedLines} source lines outside the 6,000-unit window`);

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
        const ids = await promptTokenIds(c.port, prompt, controller.signal);
        if (ids.length + SUGGESTION_MAX_TOKENS > c.contextSize) {
          requestLog.fail(entry, "skipped", `prompt ${ids.length} tokens + cap ${SUGGESTION_MAX_TOKENS} exceeds context ${c.contextSize}`);
          return [];
        }
        const r = await generateSuggestion(body => postCompletion(c.port, body, controller.signal), ids, c.allowNoEdit);
        if (!current() || controller.signal.aborted) {
          requestLog.fail(entry, "aborted", "document changed or request cancelled");
          return [];
        }
        const outcome = interpretCompletion(request.context, r.content, r.tokens, SUGGESTION_MAX_TOKENS);
        lastStats = `latency ${Date.now() - t0} ms, prompt ${ids.length} tok, completion ${r.tokens.length} tok, ${describe(outcome)}`;
        channel.appendLine(`response: ${lastStats}`);
        if (c.debugMode) {
          channel.appendLine(`--- prompt (${prompt.length} chars, ${ids.length} tokens) ---\n${prompt}--- raw completion ---\n${r.content}`);
        }
        renderStatusBar();
        requestLog.finish(entry, Date.now() - t0, r.tokens.length, r.content, outcome.result.body, describe(outcome));
        this.lastPrompt = key;
        this.lastItems = [];
        if (outcome.outcome !== "edit") return [];
        // the protocol's stale check: the document is still the one the context captured
        makeApplicationPlan(outcome.result, request.context, document.uri.toString(), document.version, sha256Text(document.getText()));
        const shown = presentEdit(request, outcome.result);
        if (shown.kind === "ghost") {
          const at = new vscode.Position(shown.line, shown.character);
          const item = new vscode.InlineCompletionItem(shown.text, new vscode.Range(at, at));
          item.command = { command: "sepalith.accepted", title: "Record accepted suggestion" };
          SepalithProvider.offered++;
          this.lastItems = [item];
          return [item];
        }
        if (shown.kind === "replace") {
          const editor = vscode.window.activeTextEditor;
          if (editor && editor.document === document && editor.selection.active.line === shown.line) {
            preview.show(editor, shown);
            SepalithProvider.offered++;
          }
        }
        return [];
      } catch (e) {
        if (controller.signal.aborted) {
          requestLog.fail(entry, "aborted", "superseded by a new request");
          return [];
        }
        if (e instanceof ProtocolError) {
          requestLog.fail(entry, "skipped", errText(e));
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
let preview: EditPreview;

let activeModel = "";
function modelName(): string {
  const m = activeModel || cfg().modelPath;
  return m.slice(m.lastIndexOf("/") + 1); // basename is enough for the status bar
}

function renderStatusBar(): void {
  const s = sidecar.currentState;
  const label = s === "starting" ? "starting…" : s === "ready" ? `ready (${modelName()})` : s === "external" ? "external server" : s;
  statusBarItem.text = `Sepalith: ${label}`;
  statusBarItem.tooltip = `Sepalith sidecar\nstate: ${s}${sidecar.detailText ? `\n${sidecar.detailText}` : ""}\nlast request: ${lastStats}\nsuggestions offered ${SepalithProvider.offered} / accepted ${SepalithProvider.accepted}`;
}

// warm-ups are prefill only, but a cold 2.4k-token prefill on a laptop takes seconds
const WARMUP_TIMEOUT_MS = 120_000;

async function sendWarmup(prompt: string): Promise<void> {
  const t0 = Date.now();
  try {
    const signal = AbortSignal.timeout(WARMUP_TIMEOUT_MS);
    const ids = await promptTokenIds(cfg().port, prompt, signal);
    await postCompletion(cfg().port, completionBody(ids, WARMUP_MAX_TOKENS), signal);
  } catch (e) {
    channel.appendLine(`warmup error: ${errText(e)}`);
    throw e;
  }
  if (cfg().debugMode) channel.appendLine(`warmup: ${prompt.length} prompt chars prefilled in ${Date.now() - t0} ms`);
}

function triggerInlineSuggestion(): void {
  void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
}

export function activate(context: vscode.ExtensionContext): void {
  const provider = new SepalithProvider();
  const warmup = new WarmupScheduler(() => provider.warmupPrompt(), sendWarmup, () => cfg().warmup);
  // prompt-changing events in the active R editor (edit, cursor settle) warm the next prompt
  const warmActive = (document: vscode.TextDocument | undefined) => {
    if (document && document === vscode.window.activeTextEditor?.document && document.languageId === "r") warmup.schedule();
  };
  runtimeStorage = sharedCacheRoot();
  channel = vscode.window.createOutputChannel("Sepalith");
  preview = new EditPreview();
  for (const document of vscode.workspace.textDocuments) if (document.languageId === "r") track(document);
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.show();
  requestLog = new RequestLog();
  context.subscriptions.push(
    channel,
    statusBarItem,
    preview,
    vscode.window.createTreeView("sepalith.requests", { treeDataProvider: requestLog }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("sepalith.startServer", () => {
      provider.invalidate();
      warmup.reset(); // a new server starts with an empty prompt cache
      return sidecar.start().then(() => warmup.schedule());
    }),
    vscode.commands.registerCommand("sepalith.acceptEdit", async () => {
      if (await preview.accept()) void vscode.commands.executeCommand("sepalith.accepted");
    }),
    vscode.commands.registerCommand("sepalith.dismissEdit", () => preview.clear()),
    vscode.commands.registerCommand("sepalith.accepted", () => {
      SepalithProvider.accepted++;
      const document = vscode.window.activeTextEditor?.document;
      acceptedDocument = document ? { uri: document.uri.toString(), version: document.version } : null;
      provider.invalidate();
      if (cfg().postAcceptCooldown && debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      warmup.schedule(); // the accepted text changes the next prompt
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
    { dispose: () => { provider.invalidate(); warmup.dispose(); } },
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      provider.invalidate();
      preview.clear();
      if (editor?.document.languageId === "r") track(editor.document);
      warmActive(editor?.document); // file open / switch
    }),
    vscode.window.onDidChangeTextEditorSelection((e) => {
      const document = e.textEditor.document;
      if (document.languageId !== "r") return;
      const line = e.selections[0]?.active.line ?? 0;
      if (preview.line !== null && preview.line !== line) preview.clear();
      const entry = track(document);
      entry.history.cursorAt(line, entry.lines); // leaving an edited line commits it to the history
      warmActive(document);
    }),
    vscode.workspace.onDidOpenTextDocument((document) => { if (document.languageId === "r") track(document); }),
    vscode.workspace.onDidCloseTextDocument((document) => { tracked.delete(document.uri.toString()); }),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId !== "r" || e.contentChanges.length === 0) return;
      observeChange(e.document);
      provider.invalidate();
      preview.clear();
      warmActive(e.document);
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
  if (cfg().autoStart) void sidecar.start().then(() => warmup.schedule());
}

export async function deactivate(): Promise<void> {
  await sidecar.stop(); // kills only a child we spawned; external servers are left alone
}
