# RUN-04-OPUS07: exact-prompt prewarm scheduler (iteration 1)

**Status:** This is source only. I have not executed, typechecked or benchmarked anything. All timing in the tests comes from an injected fake clock, and none of it is a latency claim.

**Precondition:** Save your adapter contract verbatim as `src/exact_prewarm_contract.ts`. Both files import it from that path. If your harness keeps it elsewhere, you only need to change the two import lines.

**Conformance to the contract:**
- `createExactPrewarmScheduler(deps, options?)` is assignable to `SchedulerFactory`.
- `requestForeground(snapshot, signal?)` adds one optional parameter, which keeps it a lossless superset of the contract signature.
- Nothing is weakened to pass tests: foreground priority and the stale guards are unchanged.

## File 1: `src/exact_prewarm_scheduler.ts`

```ts
/**
 * Opt-in exact-prompt prewarm scheduler for ONE native llama-server slot (np=1). Pure module:
 * clock, transport, identity and publication are injected. The caller supplies the already
 * rendered PRM-03 prompt and its no-special-token IDs; this module only prepends the single
 * manual BOS. It never renders, retokenizes, truncates, routes by label, or serves warm tokens.
 *
 * "Useful" warm = all of: not disposed; no foreground running or waiting; valid snapshot that is
 * not a known no-op, has >= minWarmPromptTokens IDs and fits contextSize with the foreground cap;
 * exact prompt is neither in flight nor the last prompt completed in the slot; identity is still
 * current when it fires; and either debounceMs of quiet or maxWaitMs since the first unserved
 * candidate (sustained typing cannot starve warming). A running warm is cancelled when a
 * foreground arrives (unless it is the identical prompt, which the foreground joins), on a
 * file/runtime switch, or when it shares < minSharedPrefixRatio of the newest candidate's IDs;
 * otherwise it completes because its prefix stays reusable by llama-server prompt caching.
 */
import { DEFAULT_REQUEST_TIMEOUT_MS, MAX_REQUEST_TIMEOUT_MS, RequestTimeoutError, validateRequestTimeoutMs } from "./campaign_requests.ts";
import { MANUAL_BOS_ID, PREWARM_MAX_OUTPUT_TOKENS } from "./exact_prewarm_contract.ts";
import type {
  ExactPrewarmScheduler, ExactPrewarmSchedulerDependencies, ExactPromptSnapshot, NativeSlotRequest,
  NativeSlotResponse, SlotRequestKind, SnapshotIdentity,
} from "./exact_prewarm_contract.ts";

export interface ExactPrewarmOptions {
  readonly debounceMs?: number;             // quiet time after the latest candidate (default 300)
  readonly maxWaitMs?: number;              // anti-starvation bound from first unserved candidate (1200)
  readonly minWarmPromptTokens?: number;    // BOS-inclusive; one ubatch by default (256)
  readonly minSharedPrefixRatio?: number;   // keep a running warm if it shares this much (0.5)
  readonly contextSize?: number;            // 4096
  readonly foregroundMaxOutputTokens?: number; // 192
  readonly foregroundTimeoutMs?: number;    // unchanged 5000 default deadline
  readonly warmTimeoutMs?: number;          // owned transport backstop (60000)
}

export interface ExactPrewarmSchedulerWithSignal extends ExactPrewarmScheduler {
  /** A subscriber may leave via signal; the native request aborts when the last one leaves. */
  requestForeground(snapshot: ExactPromptSnapshot, signal?: AbortSignal): Promise<NativeSlotResponse>;
}

interface Waiter { resolve(value: NativeSlotResponse): void; reject(error: unknown): void }
interface Job {
  readonly request: NativeSlotRequest; readonly snapshot: ExactPromptSnapshot;
  readonly controller: AbortController; readonly waiters: Set<Waiter>;
  readonly done: Promise<void>; readonly release: () => void;
  timer: unknown; settled: boolean;
}
interface Pending { request: NativeSlotRequest; snapshot: ExactPromptSnapshot; readonly firstAt: number; timer: unknown; due: boolean }

function namedError(name: string, message: string): Error { const e = new Error(message); e.name = name; return e; }
const cancelled = (why: string): Error => namedError("AbortError", `Suggestion request was cancelled: ${why}`);

function sameIds(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function sharedPrefix(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}
function sameIdentity(a: SnapshotIdentity | null, b: SnapshotIdentity): boolean {
  return a !== null && a.uri === b.uri && a.documentVersion === b.documentVersion && a.contentSha256 === b.contentSha256 &&
    a.cursorLine === b.cursorLine && a.cursorCharacter === b.cursorCharacter && a.runtimeGeneration === b.runtimeGeneration;
}
/** Same native prefill: identical prompt text, identical integer IDs, same server runtime. */
function sameExact(a: NativeSlotRequest, b: NativeSlotRequest): boolean {
  return a.identity.runtimeGeneration === b.identity.runtimeGeneration && a.promptText === b.promptText &&
    sameIds(a.promptTokenIds, b.promptTokenIds);
}
function buildRequest(kind: SlotRequestKind, s: ExactPromptSnapshot, maxOutputTokens: number, budget: number, contextSize: number): NativeSlotRequest {
  const ids = s?.encodedPromptTokenIds;
  if (typeof s?.identity?.uri !== "string" || typeof s.promptText !== "string" || !s.promptText.endsWith("\n") ||
      !Array.isArray(ids) || ids.length === 0 || ids.some((t) => !Number.isSafeInteger(t) || t < 0 || t === MANUAL_BOS_ID)) {
    throw namedError("InvalidSnapshot", "snapshot needs identity, LF-terminated prompt text and nonempty BOS-free integer IDs");
  }
  const promptTokenIds = Object.freeze([MANUAL_BOS_ID, ...ids]);
  if (promptTokenIds.length + budget > contextSize) {
    throw namedError("ContextBudget", `prompt tokens ${promptTokenIds.length} plus output budget ${budget} exceeds context ${contextSize}`);
  }
  return Object.freeze({ kind, identity: s.identity, promptText: s.promptText, promptTokenIds, maxOutputTokens, cachePrompt: true as const });
}
function exactResponse(req: NativeSlotRequest, res: NativeSlotResponse): boolean {
  return typeof res === "object" && res !== null && res.promptText === req.promptText &&
    Array.isArray(res.promptTokenIds) && sameIds(res.promptTokenIds, req.promptTokenIds) &&
    Array.isArray(res.generatedTokenIds) && res.generatedTokenIds.length <= req.maxOutputTokens;
}
function resolveOptions(o: ExactPrewarmOptions): Required<ExactPrewarmOptions> {
  const r = {
    debounceMs: o.debounceMs ?? 300, maxWaitMs: o.maxWaitMs ?? 1200, minWarmPromptTokens: o.minWarmPromptTokens ?? 256,
    minSharedPrefixRatio: o.minSharedPrefixRatio ?? 0.5, contextSize: o.contextSize ?? 4096,
    foregroundMaxOutputTokens: o.foregroundMaxOutputTokens ?? 192,
    foregroundTimeoutMs: validateRequestTimeoutMs(o.foregroundTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS),
    warmTimeoutMs: validateRequestTimeoutMs(o.warmTimeoutMs ?? MAX_REQUEST_TIMEOUT_MS),
  };
  const int = (n: number, lo: number): boolean => Number.isSafeInteger(n) && n >= lo;
  if (!int(r.debounceMs, 0) || !int(r.maxWaitMs, r.debounceMs) || !int(r.minWarmPromptTokens, 2) ||
      !(r.minSharedPrefixRatio >= 0 && r.minSharedPrefixRatio <= 1) || !int(r.foregroundMaxOutputTokens, 1) ||
      !int(r.contextSize, r.foregroundMaxOutputTokens + 2)) throw new RangeError("invalid exact prewarm scheduler options");
  return r;
}

class Scheduler implements ExactPrewarmSchedulerWithSignal {
  private readonly deps: ExactPrewarmSchedulerDependencies;
  private readonly o: Required<ExactPrewarmOptions>;
  private disposed = false;
  private slot: Job | null = null;               // the one native request this client has in flight
  private fg: Job | null = null;                 // current foreground: running, or waiting behind an identical warm
  private pending: Pending | null = null;        // latest warm candidate only (bounded memory)
  private cached: NativeSlotRequest | null = null; // last prompt known to have completed in the slot

  constructor(deps: ExactPrewarmSchedulerDependencies, options: ExactPrewarmOptions) {
    this.deps = deps;
    this.o = resolveOptions(options);
  }

  scheduleWarm(snapshot: ExactPromptSnapshot): void {
    if (this.disposed) return;
    let request: NativeSlotRequest | null = null;
    if (snapshot?.noOp === false) {
      try {
        request = buildRequest("prewarm", snapshot, PREWARM_MAX_OUTPUT_TOKENS, this.o.foregroundMaxOutputTokens, this.o.contextSize);
      } catch { request = null; } // warming is best effort; the foreground path reports invalid snapshots
      if (request !== null && request.promptTokenIds.length < this.o.minWarmPromptTokens) request = null;
    }
    const warm = this.slot !== null && this.slot.request.kind === "prewarm" ? this.slot : null;
    if (warm !== null) {
      const id = snapshot?.identity;
      const sameDoc = id?.uri === warm.request.identity.uri && id?.runtimeGeneration === warm.request.identity.runtimeGeneration;
      const overlap = request === null ? 1 : sharedPrefix(warm.request.promptTokenIds, request.promptTokenIds) / request.promptTokenIds.length;
      if (!sameDoc || overlap < this.o.minSharedPrefixRatio) this.abortJob(warm, cancelled("stale prewarm"));
      else if (request !== null && sameExact(warm.request, request)) request = null; // already warming exactly this
    }
    if (request !== null && this.cached !== null && sameExact(this.cached, request)) request = null;
    if (request === null) { this.clearPending(); return; }
    const prev = this.pending;
    if (prev !== null && sameExact(prev.request, request)) { prev.request = request; prev.snapshot = snapshot; return; }
    const now = this.deps.clock.nowMs();
    const firstAt = prev === null ? now : prev.firstAt;
    this.clearPending();
    const entry: Pending = { request, snapshot, firstAt, timer: null, due: false };
    entry.timer = this.deps.clock.setTimeout(() => { entry.timer = null; entry.due = true; this.pump(); },
      Math.max(0, Math.min(this.o.debounceMs, firstAt + this.o.maxWaitMs - now)));
    this.pending = entry;
  }

  requestForeground(snapshot: ExactPromptSnapshot, signal?: AbortSignal): Promise<NativeSlotResponse> {
    if (this.disposed) return Promise.reject(cancelled("scheduler disposed"));
    if (signal?.aborted) return Promise.reject(cancelled("caller aborted"));
    let request: NativeSlotRequest;
    try {
      const cap = this.o.foregroundMaxOutputTokens;
      request = buildRequest("foreground", snapshot, cap, cap, this.o.contextSize);
    } catch (error) { return Promise.reject(error); }
    let job = this.fg;
    if (job === null || !sameIdentity(job.request.identity, request.identity) || !sameExact(job.request, request)) {
      if (job !== null) this.abortJob(job, cancelled("superseded by newer foreground"));
      const created = this.newJob(request, snapshot, this.o.foregroundTimeoutMs, () => new RequestTimeoutError(this.o.foregroundTimeoutMs));
      this.fg = created;
      const warm = this.slot;
      if (warm !== null && warm.request.kind === "prewarm" && sameExact(warm.request, request)) {
        // Join: the identical prefill is already running. Its cap-1 token is discarded; the full
        // foreground request then reuses the cached prompt. The foreground deadline covers the wait.
        void warm.done.then(() => { if (!created.settled && !this.disposed) this.launch(created); });
      } else {
        if (warm !== null) this.abortJob(warm, cancelled("foreground priority"));
        this.launch(created);
      }
      job = created;
    }
    const owned = job;
    return new Promise<NativeSlotResponse>((resolve, reject) => {
      const leave = (): void => { signal?.removeEventListener("abort", cancel); owned.waiters.delete(waiter); };
      const waiter: Waiter = {
        resolve: (value) => { leave(); resolve(value); },
        reject: (error) => { leave(); reject(error); },
      };
      const cancel = (): void => {
        if (!owned.waiters.has(waiter)) return;
        waiter.reject(cancelled("caller aborted"));
        if (owned.waiters.size === 0) { this.abortJob(owned, cancelled("no remaining subscribers")); this.pump(); }
      };
      owned.waiters.add(waiter);
      signal?.addEventListener("abort", cancel, { once: true });
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.clearPending();
    const reason = cancelled("scheduler disposed");
    if (this.fg !== null) this.abortJob(this.fg, reason);
    if (this.slot !== null) this.abortJob(this.slot, reason);
    this.cached = null;
  }

  private pump(): void {
    const p = this.pending;
    if (this.disposed || this.fg !== null || this.slot !== null || p === null || !p.due) return;
    this.pending = null;
    if (!sameIdentity(this.deps.currentIdentity(), p.request.identity)) return; // went stale while waiting
    if (this.cached !== null && sameExact(this.cached, p.request)) return;
    this.launch(this.newJob(p.request, p.snapshot, this.o.warmTimeoutMs,
      () => namedError("TimeoutError", `prewarm exceeded ${this.o.warmTimeoutMs} ms`)));
  }

  private newJob(request: NativeSlotRequest, snapshot: ExactPromptSnapshot, timeoutMs: number, timeout: () => Error): Job {
    let release = (): void => {};
    const done = new Promise<void>((resolve) => { release = resolve; });
    const job: Job = { request, snapshot, controller: new AbortController(), waiters: new Set(), done, release, timer: null, settled: false };
    job.timer = this.deps.clock.setTimeout(() => { job.timer = null; this.abortJob(job, timeout()); this.pump(); }, timeoutMs);
    return job;
  }

  private launch(job: Job): void {
    this.slot = job;
    let call: Promise<NativeSlotResponse>;
    try { call = Promise.resolve(this.deps.transport.complete(job.request, job.controller.signal)); } catch (error) { call = Promise.reject(error); }
    call.then((value) => this.settle(job, true, value), (error) => this.settle(job, false, error));
  }

  private settle(job: Job, ok: boolean, value: unknown): void {
    if (job.settled) return; // superseded, timed out, unsubscribed or disposed: never published
    this.finish(job);
    const response = value as NativeSlotResponse;
    const exact = ok && exactResponse(job.request, response);
    this.cached = exact ? job.request : null; // a failure leaves server slot state unknown
    const current = sameIdentity(this.deps.currentIdentity(), job.request.identity);
    if (job.request.kind === "prewarm") {
      // The cap-1 warm token is evidence only; it is never handed to a foreground subscriber.
      if (exact && current) { try { this.deps.onWarmPublished(job.snapshot, response); } catch { /* notification only */ } }
    } else {
      const error = !ok ? (value ?? namedError("TransportError", "native transport failed"))
        : !exact ? namedError("PromptMismatch", "native response does not echo the exact prompt within its output cap")
          : !current ? cancelled("stale snapshot") : null;
      for (const waiter of [...job.waiters]) { if (error === null) waiter.resolve(response); else waiter.reject(error); }
      job.waiters.clear();
    }
    this.pump();
  }

  private finish(job: Job): void {
    job.settled = true;
    if (job.timer !== null) { this.deps.clock.clearTimeout(job.timer); job.timer = null; }
    if (this.slot === job) this.slot = null;
    if (this.fg === job) this.fg = null;
    job.release();
  }

  private abortJob(job: Job, reason: Error): void {
    if (job.settled) return;
    const launched = this.slot === job;
    this.finish(job);
    if (launched) this.cached = null;
    job.controller.abort(reason);
    for (const waiter of [...job.waiters]) waiter.reject(reason);
    job.waiters.clear();
  }

  private clearPending(): void {
    if (this.pending !== null && this.pending.timer !== null) this.deps.clock.clearTimeout(this.pending.timer);
    this.pending = null;
  }
}

export function createExactPrewarmScheduler(deps: ExactPrewarmSchedulerDependencies, options: ExactPrewarmOptions = {}): ExactPrewarmSchedulerWithSignal {
  return new Scheduler(deps, options);
}
```

## File 2: `src/exact_prewarm_scheduler.test.ts`

Run with: `node --experimental-strip-types --test src/exact_prewarm_scheduler.test.ts`. On Node ≥ 23.6 the flag is not needed.

```ts
/** Deterministic trace tests. Simulated fake-clock timing only: no latency claim. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createExactPrewarmScheduler } from "./exact_prewarm_scheduler.ts";
import type { ExactPrewarmOptions } from "./exact_prewarm_scheduler.ts";
import type { ExactPromptSnapshot, NativeSlotRequest, NativeSlotResponse, SnapshotIdentity } from "./exact_prewarm_contract.ts";

const flush = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));

class FakeClock {
  now = 0;
  private seq = 0;
  readonly timers = new Map<number, { at: number; cb: () => void }>();
  nowMs(): number { return this.now; }
  setTimeout(cb: () => void, ms: number): unknown { this.timers.set(++this.seq, { at: this.now + ms, cb }); return this.seq; }
  clearTimeout(handle: unknown): void { this.timers.delete(handle as number); }
  async advance(ms: number): Promise<void> {
    const end = this.now + ms;
    for (;;) {
      let id = -1;
      for (const [k, t] of this.timers) if (t.at <= end && (id < 0 || t.at < this.timers.get(id)!.at)) id = k;
      if (id < 0) break;
      const t = this.timers.get(id)!;
      this.timers.delete(id); this.now = t.at; t.cb(); await flush();
    }
    this.now = end; await flush();
  }
}

interface Call { readonly request: NativeSlotRequest; readonly signal: AbortSignal; readonly at: number; resolve(generated?: number[]): void }

function setup(options: ExactPrewarmOptions = {}, rejectOnAbort = true) {
  const clock = new FakeClock();
  const calls: Call[] = [];
  const published: ExactPromptSnapshot[] = [];
  const state: { current: SnapshotIdentity | null } = { current: null };
  const transport = {
    complete(request: NativeSlotRequest, signal: AbortSignal): Promise<NativeSlotResponse> {
      return new Promise((resolve, reject) => {
        calls.push({ request, signal, at: clock.now, resolve: (generated = request.maxOutputTokens === 1 ? [7] : [7, 1]) =>
          resolve({ promptText: request.promptText, promptTokenIds: request.promptTokenIds, generatedTokenIds: generated, operation: "replace" }) });
        if (rejectOnAbort) signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    },
  };
  const scheduler = createExactPrewarmScheduler({
    clock, transport, currentIdentity: () => state.current, onWarmPublished: (snapshot) => { published.push(snapshot); },
  }, { debounceMs: 300, maxWaitMs: 1000, minWarmPromptTokens: 8, ...options });
  const show = (s: ExactPromptSnapshot): ExactPromptSnapshot => { state.current = s.identity; return s; };
  return { clock, calls, published, state, scheduler, show };
}

const ids = (n: number, from = 10): number[] => Array.from({ length: n }, (_, i) => from + i);
function snap(version: number, tokens: number[], uri = "file:///a.R", noOp = false): ExactPromptSnapshot {
  return {
    identity: { uri, documentVersion: version, contentSha256: `sha-${version}`, cursorLine: version, cursorCharacter: 0, runtimeGeneration: 1 },
    promptText: `PRM03 ${uri} v${version}\n`, encodedPromptTokenIds: tokens, noOp,
  };
}

test("rapid typing keeps only the latest candidate; maxWait prevents starvation", async () => {
  const t = setup();
  let last = snap(0, ids(50));
  for (let v = 1; v <= 15; v++) { last = t.show(snap(v, [...ids(50), 100 + v])); t.scheduler.scheduleWarm(last); await t.clock.advance(100); }
  assert.equal(t.calls.length, 1);
  const warm = t.calls[0];
  assert.equal(warm.at, 1000); // bounded by maxWait, not an endlessly reset debounce
  assert.deepEqual(warm.request.promptTokenIds, [0, ...ids(50), 110]); // exact v10 IDs, manual BOS once
  assert.equal(warm.request.kind, "prewarm"); assert.equal(warm.request.maxOutputTokens, 1); assert.equal(warm.request.cachePrompt, true);
  assert.equal(warm.request.promptText, snap(10, []).promptText);
  assert.equal(warm.signal.aborted, false); // newer same-file candidates share its prefix
  warm.resolve(); await flush();
  assert.equal(t.published.length, 0); // stale: editor is at v15
  await t.clock.advance(200);
  assert.equal(t.calls.length, 2); assert.equal(t.calls[1].at, 1700);
  assert.deepEqual(t.calls[1].request.promptTokenIds, [0, ...last.encodedPromptTokenIds]);
  t.calls[1].resolve(); await flush();
  assert.deepEqual(t.published, [last]);
});

test("cursor move: high-overlap warm continues, low-overlap warm is cancelled", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  t.scheduler.scheduleWarm(t.show(snap(2, [...ids(38), 90, 91])));
  assert.equal(t.calls[0].signal.aborted, false);
  const far = t.show(snap(3, [...ids(5), ...ids(35, 500)]));
  t.scheduler.scheduleWarm(far);
  assert.equal(t.calls[0].signal.aborted, true);
  await t.clock.advance(300);
  assert.equal(t.calls.length, 2);
  assert.deepEqual(t.calls[1].request.promptTokenIds, [0, ...far.encodedPromptTokenIds]);
  assert.equal(t.published.length, 0);
});

test("stale warm result is never published", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(20)))); await t.clock.advance(300);
  t.state.current = snap(2, ids(20)).identity; // editor changed without a new candidate
  t.calls[0].resolve(); await flush();
  assert.equal(t.published.length, 0); assert.equal(t.clock.timers.size, 0);
});

test("file switch cancels old warm; foreground joins identical warm, never serves its token", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(30)))); await t.clock.advance(300);
  const b = t.show(snap(1, ids(30, 700), "file:///b.R"));
  t.scheduler.scheduleWarm(b);
  assert.equal(t.calls[0].signal.aborted, true);
  await t.clock.advance(300);
  assert.equal(t.calls.length, 2);
  const fg = t.scheduler.requestForeground(b);
  await flush();
  assert.equal(t.calls.length, 2); assert.equal(t.calls[1].signal.aborted, false);
  t.calls[1].resolve([42]); await flush();
  assert.deepEqual(t.published, [b]);
  assert.equal(t.calls.length, 3);
  assert.equal(t.calls[2].request.kind, "foreground"); assert.equal(t.calls[2].request.maxOutputTokens, 192);
  assert.deepEqual(t.calls[2].request.promptTokenIds, t.calls[1].request.promptTokenIds);
  t.calls[2].resolve([5, 6, 1]);
  assert.deepEqual((await fg).generatedTokenIds, [5, 6, 1]);
  assert.equal(t.clock.timers.size, 0);
});

test("overlapping subscribers share one native request; last leaver aborts; newer foreground supersedes", async () => {
  const t = setup();
  const a = t.show(snap(1, ids(20)));
  const c1 = new AbortController(), c2 = new AbortController();
  const p1 = t.scheduler.requestForeground(a, c1.signal), p2 = t.scheduler.requestForeground(a, c2.signal);
  assert.equal(t.calls.length, 1);
  c1.abort(); await assert.rejects(p1, { name: "AbortError" });
  assert.equal(t.calls[0].signal.aborted, false);
  t.calls[0].resolve([3, 1]);
  assert.deepEqual((await p2).generatedTokenIds, [3, 1]);
  const c3 = new AbortController(), c4 = new AbortController();
  const p3 = t.scheduler.requestForeground(a, c3.signal), p4 = t.scheduler.requestForeground(a, c4.signal);
  c3.abort(); c4.abort();
  await Promise.all([assert.rejects(p3, { name: "AbortError" }), assert.rejects(p4, { name: "AbortError" })]);
  assert.equal(t.calls[1].signal.aborted, true);
  const p5 = t.scheduler.requestForeground(a);
  const p6 = t.scheduler.requestForeground(t.show(snap(2, ids(20, 300))));
  await assert.rejects(p5, { name: "AbortError" });
  assert.equal(t.calls[2].signal.aborted, true); assert.equal(t.calls.length, 4);
  t.calls[3].resolve([9, 1]); assert.deepEqual((await p6).generatedTokenIds, [9, 1]);
  assert.equal(t.clock.timers.size, 0);
});

test("aborted prewarm leaves cache unknown; dispose stops all work and ignores late results", async () => {
  const t = setup({}, false); // transport ignores abort signals and resolves late
  const a = t.show(snap(1, ids(20)));
  t.scheduler.scheduleWarm(a); await t.clock.advance(300);
  t.scheduler.scheduleWarm(t.show(snap(1, ids(20, 400), "file:///b.R")));
  assert.equal(t.calls[0].signal.aborted, true);
  t.scheduler.scheduleWarm(t.show(a)); await t.clock.advance(300);
  assert.equal(t.calls.length, 2); // aborted warm was not treated as cached
  assert.deepEqual(t.calls[1].request.promptTokenIds, t.calls[0].request.promptTokenIds);
  t.calls[0].resolve(); await flush();
  assert.equal(t.published.length, 0);
  t.scheduler.dispose();
  assert.equal(t.calls[1].signal.aborted, true); assert.equal(t.clock.timers.size, 0);
  t.calls[1].resolve(); await flush();
  assert.equal(t.published.length, 0);
  await assert.rejects(t.scheduler.requestForeground(a), { name: "AbortError" });
  t.scheduler.scheduleWarm(a); await t.clock.advance(5000);
  assert.equal(t.calls.length, 2);
});

test("no-op, short, invalid and already-cached candidates start no native work", async () => {
  const t = setup();
  const a = t.show(snap(1, ids(20)));
  t.scheduler.scheduleWarm(t.show(snap(9, ids(20), undefined, true)));
  t.scheduler.scheduleWarm(t.show(snap(10, ids(3)))); // 4 IDs < minWarmPromptTokens
  t.scheduler.scheduleWarm(t.show({ ...snap(11, ids(20)), encodedPromptTokenIds: [0, ...ids(19)] })); // hidden BOS
  t.scheduler.scheduleWarm(t.show({ ...snap(12, ids(20)), promptText: "no final LF" }));
  await t.clock.advance(2000); assert.equal(t.calls.length, 0);
  t.scheduler.scheduleWarm(t.show(a)); t.scheduler.scheduleWarm(t.show(snap(13, ids(20), undefined, true)));
  await t.clock.advance(2000); assert.equal(t.calls.length, 0);
  t.scheduler.scheduleWarm(t.show(a)); await t.clock.advance(300);
  t.calls[0].resolve(); await flush();
  t.scheduler.scheduleWarm(a); await t.clock.advance(2000);
  assert.equal(t.calls.length, 1); assert.equal(t.published.length, 1);
});

test("foreground preempts a different warm immediately and keeps the 5 s deadline", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(20)))); await t.clock.advance(300);
  const fg = t.scheduler.requestForeground(t.show(snap(2, ids(20, 300))));
  assert.equal(t.calls[0].signal.aborted, true); assert.equal(t.calls[1].request.kind, "foreground");
  const rejected = assert.rejects(fg, { name: "TimeoutError" });
  await t.clock.advance(4999); assert.equal(t.calls[1].signal.aborted, false);
  await t.clock.advance(1); await rejected;
  assert.equal(t.calls[1].signal.aborted, true); assert.equal(t.clock.timers.size, 0);
});
```

## Source proof, hypothesis and measurement

**Source proof** (from the supplied files):
- The foreground uses `cache_prompt: true` and integer-ID `/completion`.
- The client's `complete()` rejects `stop_type` `limit`/`length`. So a cap-1 warm cannot go through `complete()`.
- The 5000 ms default deadline and `RequestTimeoutError` are reused unchanged.

**Measured** (from your update, not by me):
- On repeat prompts the processed-prompt count is below the total, which is consistent with prefix reuse.
- Changing prompts miss the 5 s deadline.

**Hypotheses** (not verified; the capsule has no llama-server source):
- **H1:** Aborting a non-streaming `/completion` frees the slot at the next batch boundary and keeps the already-processed prefix in the KV cache.
  - Server-side cancel may lag by the disconnect poll interval plus one 256-token ubatch. At your measured ~130–190 tok/s, that could be over 1 s.
  - This is the main risk of foreground preemption.
- **H2:** A cap-1 warm followed by the identical prompt re-evaluates only about 1 token.
- **H3:** `/tokenize` is not queued behind the slot.
- **Defaults:** 300/1200/256/0.5 are guesses, meant to be tuned.

## Missing integration seam (not implemented)

1. **Transport adapter.** `NativeCampaignClient` has no public call that takes exact IDs, and its `JsonPostTransport` is private. Root needs to add two things:
   - **Warm path:** a raw `/completion` with `{prompt: ids, n_predict: 1, temperature: 0, stream: false, cache_prompt: true, return_tokens: true}`. It should accept a `limit` stop with ≤ 1 token and check `tokens_evaluated === ids.length`.
   - **Foreground path:** factor the post-tokenize half of `complete()` into `completeExact(promptText, promptTokens, nPredict, signal)` so every existing PRM-03 guard stays.
2. **Editor hooks.** Call `scheduleWarm(snapshot)` on active-editor, selection and document changes, using the existing `renderPrompt` plus `tokenize`. Call `dispose()` on editor or extension disposal.
3. **Foreground routing.** Keep `SuggestionRequests` coalescing and its 5 s deadline. Its work function should call `requestForeground(snapshot, signal)` with its shared signal. The scheduler's own 5 s timer is only a backstop.

## Benchmark acceptance and rejection criteria

Run each trace with the switch off and on: same final weights, Q8, same server flags. Use transition traces (file switch and cursor jump) with think-time gaps of 0, 0.5, 1, 2 and 4 s, plus a sustained-typing trace.

**Accept only if all of these hold:**
- Foreground `promptTokenIds` are byte-identical between the off and on runs.
- The protocol-valid rate is unchanged.
- The actual-5 s deadline-hit count on changing prompts rises, with zero cases that pass off but fail on.
- Median foreground client wall time drops at least 30% at gaps ≥ 2 s.
- At a 0 s gap the regression is ≤ 150 ms. This measures H1's abort cost.
- For a foreground after a completed warm, the processed-prompt count is ≤ 1 + the changed tail.
- Sustained typing adds no foreground cancellations.
- Nothing is left in flight on the server after dispose.

**Reject if any of these occur:**
- Any protocol-invalid result.
- Any regression versus the off run.
- An abort-then-foreground run processes as many tokens as a cold run (H1 false).
- Abort-to-foreground-start latency exceeds 500 ms median.
