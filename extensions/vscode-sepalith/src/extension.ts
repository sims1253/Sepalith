import * as vscode from "vscode";
import * as fs from "node:fs";
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";

// Sepalith v0 — see SPEC.md. This is END-TO-END PLUMBING, not suggestion
// quality: garbage suggestions from the current weak models are acceptable.

const DEFAULT_MODEL = "/home/m0hawk/Documents/Sepalith/experiments/models/abl_dropout-Q8_0.gguf";
const DEFAULT_SERVER = "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server";

interface Config {
  modelPath: string;
  serverPath: string;
  port: number;
  threads: number;
  contextSize: number;
  autoStart: boolean;
  debounceMs: number;
  debugMode: boolean;
}

function cfg(): Config {
  const c = vscode.workspace.getConfiguration("sepalith");
  return {
    modelPath: c.get("modelPath", DEFAULT_MODEL),
    serverPath: c.get("serverPath", DEFAULT_SERVER),
    port: c.get("port", 18099),
    threads: c.get("threads", 8),
    contextSize: c.get("contextSize", 8192),
    autoStart: c.get("autoStart", true),
    debounceMs: c.get("debounceMs", 1500),
    debugMode: c.get("debugMode", false),
  };
}

// ---------------------------------------------------------------------------
// Prompt render (must match experiments/eval/run_eval.py render_zeta2).
// v0: the region is EMPTY — cursor marker alone ("signature" style).
// ---------------------------------------------------------------------------

const MAX_PREFIX_SUFFIX_CHARS = 6000;

function renderPrompt(prefixLines: string[], suffixLines: string[], relPath: string): string {
  return [
    "<[fim-suffix]>",
    ...suffixLines,
    `<[fim-prefix]><filename>${relPath}`,
    ...prefixLines,
    "<<<<<<< CURRENT",
    "<|user_cursor|>",
    "=======",
    "<[fim-middle]>",
  ].join("\n");
}

function buildPrompt(document: vscode.TextDocument, position: vscode.Position): { prompt: string; truncatedLines: number } {
  const lines = document.getText().split("\n");
  const suffix = lines.slice(position.line + 1); // everything below the cursor's line
  const suffixChars = suffix.reduce((n, l) => n + l.length + 1, 0);
  const budget = Math.max(0, MAX_PREFIX_SUFFIX_CHARS - suffixChars);
  const prefix = lines.slice(0, position.line); // everything above the cursor's line
  // truncate the prefix from its START (keep the tail) so prefix+suffix stays under ~6000 chars
  let keep = 0;
  let used = 0;
  for (let i = prefix.length - 1; i >= 0; i--) {
    if (used + prefix[i].length + 1 > budget) break;
    used += prefix[i].length + 1;
    keep++;
  }
  const truncatedLines = prefix.length - keep;
  const relPath = vscode.workspace.asRelativePath(document.uri);
  return { prompt: renderPrompt(prefix.slice(truncatedLines), suffix, relPath), truncatedLines };
}

// marker lines the model sometimes echoes from the prompt back into its
// completion (seen live: an empty-file proposal consisting mostly of
// "<<<<<<< CURRENT / ======= / <[fim-middle]" lines)
const MARKER_LINE = /^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|suffix)\]>|<\|user_cursor\|>)\s*$/;

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

async function postCompletion(port: number, prompt: string, maxTokens: number, stop: string[] | null, signal?: AbortSignal): Promise<CompletionResult> {
  const body = JSON.stringify({ prompt, max_tokens: maxTokens, temperature: 0, stop, stream: false });
  const res = await fetch(`http://127.0.0.1:${port}/v1/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { choices?: { text?: string }[]; usage?: { completion_tokens?: number } };
  return { text: data.choices?.[0]?.text ?? "", completionTokens: data.usage?.completion_tokens ?? 0 };
}

async function portAnswers(port: number): Promise<boolean> {
  try {
    await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1500) });
    return true; // any HTTP answer counts as "the port is answering"
  } catch {
    return false;
  }
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Sidecar lifecycle (the four hard rules in SPEC.md)
// ---------------------------------------------------------------------------

type SidecarState = "starting" | "ready" | "external" | "stopped" | "error";

class Sidecar {
  private child: ChildProcess | null = null;
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

  async start(): Promise<void> {
    if (this.child || this.state === "starting" || this.state === "ready" || this.state === "external") return;
    const c = cfg();
    // rule 4: if the port is already answering, treat the server as external — use it, never spawn, never kill
    if (await portAnswers(c.port)) {
      channel.appendLine(`sidecar: port ${c.port} already answering — treating server as external`);
      this.setState("external");
      return;
    }
    for (const p of [c.serverPath, c.modelPath]) {
      if (!fs.existsSync(p)) {
        channel.appendLine(`sidecar: error: missing ${p}`);
        this.setState("error", `missing ${p}`);
        return;
      }
    }
    this.setState("starting");
    channel.appendLine(`sidecar: spawning ${c.serverPath} --port ${c.port} -t ${c.threads} -c ${c.contextSize} -ngl 0 (cpu-only)`);
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
        "--port", String(c.port),
        "--host", "127.0.0.1",
        "-c", String(c.contextSize),
        "--parallel", "1",
        "-t", String(c.threads),
        "-ngl", "0",
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    this.child = child;
    child.stdout?.on("data", (d: Buffer) => this.pushRing("out", d));
    child.stderr?.on("data", (d: Buffer) => this.pushRing("err", d));
    child.on("error", (err) => {
      this.child = null;
      this.fail(`spawn failed: ${err.message}`);
    });
    child.on("exit", (code, sig) => {
      this.child = null;
      if (this.stopping) {
        this.setState("stopped");
        return;
      }
      this.fail(`server exited (code=${code} sig=${sig})`);
    });
    // rule 2: /health answers ok DURING load — poll completions until HTTP 200 (3 s interval, 180 s budget)
    const t0 = Date.now();
    for (;;) {
      if (!this.child) return; // exited/errored; handler already surfaced it
      if (Date.now() - t0 > 180_000) {
        this.killChild();
        this.fail("readiness timeout after 180 s");
        return;
      }
      try {
        await postCompletion(c.port, "x", 1, null);
        break;
      } catch (e) {
        channel.appendLine(`sidecar: not ready (${Math.round((Date.now() - t0) / 1000)} s): ${errText(e)}`);
      }
      await sleep(3000);
    }
    channel.appendLine(`sidecar: ready after ${Math.round((Date.now() - t0) / 1000)} s`);
    this.setState("ready", c.modelPath);
  }

  stop(): void {
    if (this.state === "external") return; // rule 4: never kill a server we did not spawn
    const child = this.child;
    if (!child || child.exitCode !== null) {
      this.setState("stopped");
      return;
    }
    // rule 3: kill the TRACKED child pid only — never pkill by name
    this.stopping = true;
    try {
      child.kill("SIGTERM");
    } catch {
      // already gone
    }
    const t = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {
        // already gone
      }
    }, 5000);
    child.once("exit", () => clearTimeout(t));
    this.child = null;
    this.setState("stopped");
  }

  private killChild(): void {
    const child = this.child;
    this.child = null;
    this.stopping = true;
    if (child && child.exitCode === null) {
      try {
        child.kill("SIGTERM");
      } catch {
        // already gone
      }
    }
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

class SepalithProvider implements vscode.InlineCompletionItemProvider {
  private controller: AbortController | null = null;
  // identical prompts: reuse the in-flight promise or the cached result —
  // VS Code re-invokes the provider on every cursor move, which previously
  // aborted and re-sent the SAME request dozens of times per keystroke
  private lastPrompt = "";
  private lastItems: vscode.InlineCompletionItem[] | null = null;
  private inFlight: { prompt: string; promise: Promise<vscode.InlineCompletionItem[]> } | null = null;
  private lastSkipLog = 0;

  private skipLogRateLimited(msg: string): void {
    const now = Date.now();
    if (now - this.lastSkipLog > 30_000) {
      this.lastSkipLog = now;
      channel.appendLine(msg);
    }
  }

  async provideInlineCompletionItems(document: vscode.TextDocument, position: vscode.Position): Promise<vscode.InlineCompletionItem[]> {
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
    const { prompt, truncatedLines } = buildPrompt(document, position);
    if (truncatedLines > 0) channel.appendLine(`prompt: truncated ${truncatedLines} lines from the start of the prefix`);

    lastPrompt = prompt;
    if (this.lastItems && prompt === this.lastPrompt) {
      requestLog.begin(prompt, prompt.length, "cached", "identical prompt — reused cached result");
      this.skipLogRateLimited("request: identical prompt — cached result");
      return this.lastItems;
    }
    if (this.inFlight && prompt === this.inFlight.prompt) {
      this.skipLogRateLimited("request: identical prompt already in flight — joining it");
      return this.inFlight.promise;
    }

    const entry = requestLog.begin(prompt, prompt.length, "in-flight");
    channel.appendLine(`request: ${prompt.length} prompt chars`);
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const c = cfg();
    let promise: Promise<vscode.InlineCompletionItem[]>;
    promise = (async () => {
      try {
        const t0 = Date.now();
        const r = await postCompletion(c.port, prompt, 640, [">>>>>>> UPDATED"], controller.signal);
        lastRaw = r.text;
        lastStats = `latency ${Date.now() - t0} ms, completion tokens ${r.completionTokens}`;
        channel.appendLine(`response: ${lastStats}`);
        if (c.debugMode) {
          channel.appendLine(`--- prompt (${prompt.length} chars) ---\n${prompt}\n--- raw completion ---\n${r.text}`);
        }
        renderStatusBar();
        const lines = parsePrediction(r.text);
        requestLog.finish(entry, Date.now() - t0, r.completionTokens, r.text, lines);
        if (lines.length === 0) {
          this.lastPrompt = prompt;
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
        const eol = document.lineAt(position.line).range.end;
        const item = new vscode.InlineCompletionItem(lines.join("\n"), new vscode.Range(position, eol));
        this.lastPrompt = prompt;
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
      }
    })();
    this.inFlight = { prompt, promise };
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
let lastRaw = "(no response yet)";
let debounceTimer: ReturnType<typeof setTimeout> | null = null;
const sidecar = new Sidecar();

function modelName(): string {
  const m = cfg().modelPath;
  return m.slice(m.lastIndexOf("/") + 1); // basename is enough for the status bar
}

function renderStatusBar(): void {
  const s = sidecar.currentState;
  const label = s === "starting" ? "starting…" : s === "ready" ? `ready (${modelName()})` : s === "external" ? "external server" : s;
  statusBarItem.text = `Sepalith: ${label}`;
  statusBarItem.tooltip = `Sepalith sidecar\nstate: ${s}${sidecar.detailText ? `\n${sidecar.detailText}` : ""}\nlast request: ${lastStats}`;
}

function triggerInlineSuggestion(): void {
  void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
}

export function activate(context: vscode.ExtensionContext): void {
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
    vscode.commands.registerCommand("sepalith.startServer", () => void sidecar.start()),
    vscode.commands.registerCommand("sepalith.stopServer", () => sidecar.stop()),
    // manual suggest: bypasses the debounce entirely
    vscode.commands.registerCommand("sepalith.suggest", () => {
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
    vscode.commands.registerCommand("sepalith.clearRequests", () => requestLog.clear()),
    vscode.languages.registerInlineCompletionItemProvider({ language: "r" }, new SepalithProvider()),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId !== "r") return;
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

export function deactivate(): void {
  sidecar.stop(); // kills only a child we spawned; external servers are left alone
}
