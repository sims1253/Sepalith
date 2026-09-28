# RUN-04/RUN-06 iteration 2: native slot released only on transport settlement

## Status

Nothing here has been executed. No tests, benchmarks or launches were run. The claims below come from reading the source. Where I say a test "passes by trace", I followed it through by hand; that is not a measurement. Root's measurements are the only real observations.

## Root failure, from the iteration-1 source

In `requestForeground`, a different live warm led to `abortJob(warm)`. That called `finish(job)`, which set `this.slot = null` and called `job.release()`. The code then called `this.launch(created)` in the same synchronous turn. The warm's `transport.complete` promise was still unresolved, so `calls.length` became 2.

The join path had the same flaw. `warm.done` was released by `finish()` on abort, not by the transport settling.

## Revised invariant

- **`slot` means "a `transport.complete()` promise is unsettled".**
  - `slot` is set in `launch()` before calling out.
  - It is cleared in exactly one place: `onSettled()`, the transport promise's own handler.
  - `pump()` is the only caller of `launch()`, and it returns while `slot !== null`.
  - Result: at most one outstanding call, even when abort is ignored.
- **A job's logical outcome is separate from slot ownership.** `closeJob()` handles supersede, timeout, last leaver, stale warm and dispose. It:
  - clears the job's timer;
  - rejects its subscribers immediately;
  - sets `cached = null` if the job was launched;
  - sends `AbortSignal` as a request only.
  
  When the late settlement arrives, it only frees the slot. It is never published, never delivered and never written to `cached`.
- **The foreground deadline starts at `requestForeground()`.** Waiting for an old transport therefore counts against the 5000 ms. When the deadline fires, subscribers reject at once and the slot stays busy until settlement.
- **`pump()` priority:**
  1. A waiting foreground goes first. It is re-checked against `currentIdentity()`; a stale one is rejected without any native work.
  2. Otherwise, the due warm runs if it is still current and not `cached`.

## Files

### `src/exact_prewarm_scheduler.ts` (complete)

```ts
/**
 * Opt-in exact-prompt prewarm scheduler for ONE native llama-server slot (np=1). Pure module:
 * clock, transport, identity and publication are injected. The caller supplies the already
 * rendered PRM-03 prompt and its no-special-token IDs; this module only prepends the single
 * manual BOS. It never renders, retokenizes, truncates, routes by label, or serves warm tokens.
 *
 * Slot ownership: AbortSignal is only a cancellation REQUEST. The slot is busy from the call to
 * transport.complete() until the promise it returned settles, whether or not the job was since
 * aborted, timed out, superseded, unsubscribed or disposed, so at most one transport call is ever
 * outstanding. A job's logical outcome is decoupled from that: closing a job rejects its
 * subscribers at once, and its eventual settlement only frees the slot (it is never published,
 * delivered, or recorded as cached). A foreground that finds the slot busy waits; its deadline
 * starts at requestForeground(), so the wait counts against foregroundTimeoutMs.
 *
 * "Useful" warm = all of: not disposed; slot free and no foreground running or waiting; valid
 * snapshot that is not a known no-op, has >= minWarmPromptTokens IDs and fits contextSize with the
 * foreground cap; exact prompt is neither in flight nor the last prompt completed in the slot;
 * identity is still current when it fires; and either debounceMs of quiet or maxWaitMs since the
 * first unserved candidate (sustained typing cannot starve warming). Cancellation is requested
 * for a running warm when a foreground arrives (unless it is the identical prompt, which the
 * foreground waits behind), on a file/runtime switch, or when it shares < minSharedPrefixRatio of
 * the newest candidate's IDs; otherwise it completes because its prefix stays reusable by
 * llama-server prompt caching.
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
  readonly foregroundTimeoutMs?: number;    // unchanged 5000 default wall deadline, includes waiting
  readonly warmTimeoutMs?: number;          // owned transport backstop (60000)
}

export interface ExactPrewarmSchedulerWithSignal extends ExactPrewarmScheduler {
  /** A subscriber may leave via signal; the job closes (abort requested) when the last one leaves. */
  requestForeground(snapshot: ExactPromptSnapshot, signal?: AbortSignal): Promise<NativeSlotResponse>;
}

interface Waiter { resolve(value: NativeSlotResponse): void; reject(error: unknown): void }
interface Job {
  readonly request: NativeSlotRequest; readonly snapshot: ExactPromptSnapshot;
  readonly controller: AbortController; readonly waiters: Set<Waiter>;
  timer: unknown;
  /** Outcome no longer wanted: completed, superseded, timed out, unsubscribed or disposed. */
  closed: boolean;
  /** transport.complete() was called; the slot stays busy until that promise settles. */
  launched: boolean;
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
  /** Job whose transport promise is unsettled (possibly already closed). Cleared ONLY by settlement. */
  private slot: Job | null = null;
  /** Live foreground: running in the slot, or waiting for the slot to become free. */
  private fg: Job | null = null;
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
    const warm = this.liveWarm();
    if (warm !== null) {
      const id = snapshot?.identity;
      const sameDoc = id?.uri === warm.request.identity.uri && id?.runtimeGeneration === warm.request.identity.runtimeGeneration;
      const overlap = request === null ? 1 : sharedPrefix(warm.request.promptTokenIds, request.promptTokenIds) / request.promptTokenIds.length;
      if (!sameDoc || overlap < this.o.minSharedPrefixRatio) this.closeJob(warm, cancelled("stale prewarm"));
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
    const existing = this.fg;
    const fresh = existing === null || !sameIdentity(existing.request.identity, request.identity) || !sameExact(existing.request, request);
    let job: Job;
    if (fresh) {
      if (existing !== null) this.closeJob(existing, cancelled("superseded by newer foreground"));
      const running = this.slot;
      // An identical live warm keeps running: the foreground waits for it to settle, discards its
      // cap-1 token, then issues the real foreground, which reuses the prompt cache. Anything else
      // gets a cancellation request; either way the foreground waits for actual settlement.
      if (running !== null && !running.closed && !(running.request.kind === "prewarm" && sameExact(running.request, request))) {
        this.closeJob(running, cancelled("foreground priority"));
      }
      const ms = this.o.foregroundTimeoutMs; // wall deadline from now, including time waiting for the slot
      job = this.newJob(request, snapshot, ms, () => new RequestTimeoutError(ms));
      this.fg = job;
    } else {
      job = existing;
    }
    const owned = job;
    const promise = new Promise<NativeSlotResponse>((resolve, reject) => {
      const leave = (): void => { signal?.removeEventListener("abort", cancel); owned.waiters.delete(waiter); };
      const waiter: Waiter = {
        resolve: (value) => { leave(); resolve(value); },
        reject: (error) => { leave(); reject(error); },
      };
      const cancel = (): void => {
        if (!owned.waiters.has(waiter)) return;
        waiter.reject(cancelled("caller aborted"));
        if (owned.waiters.size === 0) { this.closeJob(owned, cancelled("no remaining subscribers")); this.pump(); }
      };
      owned.waiters.add(waiter);
      signal?.addEventListener("abort", cancel, { once: true });
    });
    if (fresh) this.pump(); // dispatches now only if no transport call is outstanding
    return promise;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.clearPending();
    const reason = cancelled("scheduler disposed");
    if (this.fg !== null) this.closeJob(this.fg, reason);
    if (this.slot !== null) this.closeJob(this.slot, reason); // slot stays held until its transport settles
    this.cached = null;
  }

  private liveWarm(): Job | null {
    const s = this.slot;
    return s !== null && !s.closed && s.request.kind === "prewarm" ? s : null;
  }

  private pump(): void {
    if (this.disposed || this.slot !== null) return; // the slot frees only when the transport settles
    const fg = this.fg;
    if (fg !== null) {
      if (sameIdentity(this.deps.currentIdentity(), fg.request.identity)) { this.launch(fg); return; }
      this.closeJob(fg, cancelled("stale snapshot")); // editor moved on while it waited: no native work
    }
    const p = this.pending;
    if (p === null || !p.due) return;
    this.pending = null;
    if (!sameIdentity(this.deps.currentIdentity(), p.request.identity)) return; // went stale while waiting
    if (this.cached !== null && sameExact(this.cached, p.request)) return;
    const ms = this.o.warmTimeoutMs;
    this.launch(this.newJob(p.request, p.snapshot, ms, () => namedError("TimeoutError", `prewarm exceeded ${ms} ms`)));
  }

  private newJob(request: NativeSlotRequest, snapshot: ExactPromptSnapshot, timeoutMs: number, timeout: () => Error): Job {
    const job: Job = { request, snapshot, controller: new AbortController(), waiters: new Set(), timer: null, closed: false, launched: false };
    job.timer = this.deps.clock.setTimeout(() => { job.timer = null; this.closeJob(job, timeout()); this.pump(); }, timeoutMs);
    return job;
  }

  /** Called only from pump(), only while no transport call is outstanding. */
  private launch(job: Job): void {
    this.slot = job; // claimed before calling out, so re-entrant calls cannot dispatch a second request
    job.launched = true;
    let call: Promise<NativeSlotResponse>;
    try { call = Promise.resolve(this.deps.transport.complete(job.request, job.controller.signal)); } catch (error) { call = Promise.reject(error); }
    call.then((value) => this.onSettled(job, true, value), (error) => this.onSettled(job, false, error));
  }

  private onSettled(job: Job, ok: boolean, value: unknown): void {
    if (this.slot === job) this.slot = null; // the ONLY place the native slot is released
    if (!job.closed) {
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
    }
    // A closed job's late settlement only frees the slot: never published, delivered or cached.
    this.pump();
  }

  private finish(job: Job): void {
    job.closed = true;
    if (job.timer !== null) { this.deps.clock.clearTimeout(job.timer); job.timer = null; }
    if (this.fg === job) this.fg = null;
  }

  private closeJob(job: Job, reason: Error): void {
    if (job.closed) return;
    this.finish(job);
    if (job.launched) this.cached = null; // slot contents unknown until the next exact completion
    job.controller.abort(reason);         // a request only: the slot stays busy until settlement
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

### `src/exact_prewarm_scheduler.test.ts` (complete; only 2 of the original 8 tests changed)

Only two tests assumed dispatch happens in the same turn as the abort:

- **Test 6** ("aborted prewarm…"): the transport ignores abort, so the re-warm now dispatches only after `calls[0]` settles. The assertions are reordered accordingly. It still checks that the late result is not published and not cached.
- **Test 8** ("foreground preempts…"): one `await flush()` is added before reading `calls[1]`, because dispatch now follows the warm's rejection microtask.

The other six tests are byte-identical.

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
  // Iteration 2: the abandoned warm's transport has not settled, so it still owns the one native slot.
  assert.equal(t.calls.length, 1);
  t.calls[0].resolve(); await flush();
  assert.equal(t.published.length, 0); // late settlement of an aborted warm is never published...
  assert.equal(t.calls.length, 2); // ...nor treated as cached: the identical prompt is warmed again
  assert.deepEqual(t.calls[1].request.promptTokenIds, t.calls[0].request.promptTokenIds);
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
  assert.equal(t.calls[0].signal.aborted, true);
  await flush(); // Iteration 2: cancellation is requested at once; dispatch waits for the warm transport to settle
  assert.equal(t.calls[1].request.kind, "foreground");
  const rejected = assert.rejects(fg, { name: "TimeoutError" });
  await t.clock.advance(4999); assert.equal(t.calls[1].signal.aborted, false);
  await t.clock.advance(1); await rejected;
  assert.equal(t.calls[1].signal.aborted, true); assert.equal(t.clock.timers.size, 0);
});
```

### `src/exact_prewarm_waiting.test.ts` (new, adversarial; transport always ignores abort)

```ts
/** Adversarial slot-waiting tests. Transport IGNORES AbortSignal and settles only when resolved.
 *  Simulated fake-clock timing only: no latency claim. */
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

function setup(options: ExactPrewarmOptions = {}) {
  const clock = new FakeClock();
  const calls: Call[] = [];
  const published: ExactPromptSnapshot[] = [];
  const state: { current: SnapshotIdentity | null } = { current: null };
  let outstanding = 0, maxOutstanding = 0;
  const transport = {
    complete(request: NativeSlotRequest, signal: AbortSignal): Promise<NativeSlotResponse> {
      outstanding++; maxOutstanding = Math.max(maxOutstanding, outstanding);
      return new Promise((resolve) => {
        let done = false;
        calls.push({ request, signal, at: clock.now, resolve: (generated = request.maxOutputTokens === 1 ? [7] : [7, 1]) => {
          if (done) return;
          done = true; outstanding--;
          resolve({ promptText: request.promptText, promptTokenIds: request.promptTokenIds, generatedTokenIds: generated, operation: "replace" });
        } });
        // Deliberately no abort listener: AbortSignal is a request the native side may ignore.
      });
    },
  };
  const scheduler = createExactPrewarmScheduler({
    clock, transport, currentIdentity: () => state.current, onWarmPublished: (snapshot) => { published.push(snapshot); },
  }, { debounceMs: 300, maxWaitMs: 1000, minWarmPromptTokens: 8, ...options });
  const show = (s: ExactPromptSnapshot): ExactPromptSnapshot => { state.current = s.identity; return s; };
  return { clock, calls, published, state, scheduler, show, maxOutstanding: () => maxOutstanding };
}

const ids = (n: number, from = 10): number[] => Array.from({ length: n }, (_, i) => from + i);
function snap(version: number, tokens: number[], uri = "file:///a.R", noOp = false): ExactPromptSnapshot {
  return {
    identity: { uri, documentVersion: version, contentSha256: `sha-${version}`, cursorLine: version, cursorCharacter: 0, runtimeGeneration: 1 },
    promptText: `PRM03 ${uri} v${version}\n`, encodedPromptTokenIds: tokens, noOp,
  };
}

test("5000 ms deadline includes waiting behind an unsettled warm; slot stays held after the timeout", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  const fgB = t.scheduler.requestForeground(t.show(snap(2, ids(40, 400), "file:///b.R")));
  const timedOut = assert.rejects(fgB, { name: "TimeoutError" });
  assert.equal(t.calls[0].signal.aborted, true);
  await t.clock.advance(4999); assert.equal(t.calls.length, 1);
  await t.clock.advance(1); await timedOut; // 300 + 5000, although nothing was dispatched for it
  assert.equal(t.calls.length, 1);
  const c = t.show(snap(3, ids(40, 800), "file:///c.R"));
  const fgC = t.scheduler.requestForeground(c);
  await flush();
  assert.equal(t.calls.length, 1, "old transport has not settled, so the slot is still busy");
  t.calls[0].resolve(); await flush();
  assert.equal(t.calls.length, 2); assert.equal(t.calls[1].at, 5300);
  assert.equal(t.calls[1].request.kind, "foreground");
  assert.deepEqual(t.calls[1].request.promptTokenIds, [0, ...c.encodedPromptTokenIds]);
  t.calls[1].resolve([4, 1]);
  assert.deepEqual((await fgC).generatedTokenIds, [4, 1]);
  assert.equal(t.published.length, 0); assert.equal(t.clock.timers.size, 0); assert.equal(t.maxOutstanding(), 1);
});

test("timed-out foreground holds the slot until settlement; its late result is not delivered, published or cached", async () => {
  const t = setup();
  const a = t.show(snap(1, ids(40)));
  const fg = t.scheduler.requestForeground(a);
  const timedOut = assert.rejects(fg, { name: "TimeoutError" });
  assert.equal(t.calls.length, 1);
  await t.clock.advance(5000); await timedOut;
  assert.equal(t.calls[0].signal.aborted, true);
  t.scheduler.scheduleWarm(a); await t.clock.advance(300);
  assert.equal(t.calls.length, 1, "due warm must not overlap the unsettled foreground transport");
  t.calls[0].resolve([5, 1]); await flush();
  assert.equal(t.calls.length, 2, "late completion of a closed job must not mark the prompt cached");
  assert.equal(t.calls[1].request.kind, "prewarm");
  assert.deepEqual(t.calls[1].request.promptTokenIds, t.calls[0].request.promptTokenIds);
  assert.equal(t.published.length, 0);
  t.calls[1].resolve(); await flush();
  assert.deepEqual(t.published, [a]);
  assert.equal(t.clock.timers.size, 0); assert.equal(t.maxOutstanding(), 1);
});

test("foreground superseding foreground: one call outstanding; a superseded waiter is never dispatched", async () => {
  const t = setup();
  const fgA = t.scheduler.requestForeground(t.show(snap(1, ids(20))));
  const rejA = assert.rejects(fgA, { name: "AbortError" });
  const fgB = t.scheduler.requestForeground(t.show(snap(2, ids(20, 300))));
  const rejB = assert.rejects(fgB, { name: "AbortError" });
  await rejA;
  assert.equal(t.calls[0].signal.aborted, true); assert.equal(t.calls.length, 1);
  const c = t.show(snap(3, ids(20, 600)));
  const fgC = t.scheduler.requestForeground(c);
  await rejB; assert.equal(t.calls.length, 1);
  t.calls[0].resolve([8, 1]); await flush();
  assert.equal(t.calls.length, 2);
  assert.deepEqual(t.calls[1].request.promptTokenIds, [0, ...c.encodedPromptTokenIds]);
  t.calls[1].resolve([9, 1]);
  assert.deepEqual((await fgC).generatedTokenIds, [9, 1]); // late [8, 1] never reaches C
  assert.equal(t.clock.timers.size, 0); assert.equal(t.maxOutstanding(), 1);
});

test("waiting request survives one subscriber leaving and is dispatched after settlement", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  const b = t.show(snap(2, ids(40, 400), "file:///b.R"));
  const c1 = new AbortController();
  const p1 = t.scheduler.requestForeground(b, c1.signal), p2 = t.scheduler.requestForeground(b);
  c1.abort(); await assert.rejects(p1, { name: "AbortError" });
  assert.equal(t.calls.length, 1);
  t.calls[0].resolve(); await flush();
  assert.equal(t.calls.length, 2);
  t.calls[1].resolve([2, 1]);
  assert.deepEqual((await p2).generatedTokenIds, [2, 1]);
  assert.equal(t.clock.timers.size, 0); assert.equal(t.maxOutstanding(), 1);
});

test("last subscriber leaving while waiting drops the request before dispatch", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  const b = t.show(snap(2, ids(40, 400), "file:///b.R"));
  const c1 = new AbortController(), c2 = new AbortController();
  const p1 = t.scheduler.requestForeground(b, c1.signal), p2 = t.scheduler.requestForeground(b, c2.signal);
  c1.abort(); await assert.rejects(p1, { name: "AbortError" });
  c2.abort(); await assert.rejects(p2, { name: "AbortError" });
  assert.equal(t.clock.timers.size, 0, "deadline timer released with the last subscriber");
  t.calls[0].resolve(); await flush();
  assert.equal(t.calls.length, 1, "cancelled foreground is never dispatched once the slot frees");
  assert.equal(t.published.length, 0);
  const p3 = t.scheduler.requestForeground(b);
  assert.equal(t.calls.length, 2); // free slot: dispatched in the same turn
  t.calls[1].resolve([3, 1]);
  assert.deepEqual((await p3).generatedTokenIds, [3, 1]);
  assert.equal(t.maxOutstanding(), 1);
});

test("foreground behind an identical warm: waits, discards cap-1 token, issues real foreground within the same deadline", async () => {
  const t = setup();
  const a = t.show(snap(1, ids(40)));
  t.scheduler.scheduleWarm(a); await t.clock.advance(300);
  const fg = t.scheduler.requestForeground(a);
  const timedOut = assert.rejects(fg, { name: "TimeoutError" });
  await t.clock.advance(2000);
  assert.equal(t.calls.length, 1); assert.equal(t.calls[0].signal.aborted, false);
  t.calls[0].resolve([42]); await flush();
  assert.deepEqual(t.published, [a]);
  assert.equal(t.calls.length, 2);
  assert.equal(t.calls[1].request.kind, "foreground"); assert.equal(t.calls[1].request.maxOutputTokens, 192);
  assert.equal(t.calls[1].request.promptText, t.calls[0].request.promptText);
  assert.deepEqual(t.calls[1].request.promptTokenIds, t.calls[0].request.promptTokenIds);
  await t.clock.advance(2999); assert.equal(t.calls[1].signal.aborted, false); // t = 5299
  await t.clock.advance(1); await timedOut; // t = 5300 = request time + 5000: the wait was counted
  assert.equal(t.calls[1].signal.aborted, true);
  t.calls[1].resolve([5, 1]); await flush();
  assert.equal(t.clock.timers.size, 0); assert.equal(t.maxOutstanding(), 1);
});

test("editor moves on while a foreground waits: rejected as stale when the slot frees, no dispatch", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  const fg = t.scheduler.requestForeground(t.show(snap(2, ids(40, 400), "file:///b.R")));
  const stale = assert.rejects(fg, { name: "AbortError", message: /stale snapshot/ });
  t.state.current = { ...snap(2, ids(40, 400), "file:///b.R").identity, runtimeGeneration: 2 };
  t.calls[0].resolve(); await flush(); await stale;
  assert.equal(t.calls.length, 1); assert.equal(t.published.length, 0); assert.equal(t.clock.timers.size, 0);
});

test("dispose while a foreground waits rejects it at once; late settlement starts nothing", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  const fg = t.scheduler.requestForeground(t.show(snap(2, ids(40, 400), "file:///b.R")));
  t.scheduler.dispose();
  await assert.rejects(fg, { name: "AbortError", message: /disposed/ });
  assert.equal(t.clock.timers.size, 0);
  t.calls[0].resolve(); await flush();
  assert.equal(t.calls.length, 1); assert.equal(t.published.length, 0);
});

test("a due warm never jumps ahead of a waiting foreground; exact foreground completion makes it redundant", async () => {
  const t = setup();
  t.scheduler.scheduleWarm(t.show(snap(1, ids(40)))); await t.clock.advance(300);
  const a2 = t.show(snap(2, [...ids(38), 90, 91]));
  t.scheduler.scheduleWarm(a2); // high overlap: running warm continues, a2 pending
  assert.equal(t.calls[0].signal.aborted, false);
  const fg = t.scheduler.requestForeground(a2);
  assert.equal(t.calls[0].signal.aborted, true);
  await t.clock.advance(300); // warm a2 is now due, but the slot is still busy
  assert.equal(t.calls.length, 1);
  t.calls[0].resolve(); await flush();
  assert.equal(t.calls.length, 2); assert.equal(t.calls[1].request.kind, "foreground");
  t.calls[1].resolve([6, 1]);
  assert.deepEqual((await fg).generatedTokenIds, [6, 1]);
  await t.clock.advance(2000);
  assert.equal(t.calls.length, 2, "exact foreground completion is reused as cache evidence");
  assert.equal(t.published.length, 0); assert.equal(t.maxOutstanding(), 1);
});

test("seeded adversarial trace: at most one native request outstanding; every foreground settles", async () => {
  const t = setup();
  let seed = 0x5eed;
  const rnd = (n: number): number => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
  const snaps = [snap(1, ids(40)), snap(2, [...ids(38), 90, 91]), snap(3, ids(40, 400), "file:///b.R"), snap(4, ids(40, 800), "file:///c.R")];
  const outcomes: Promise<string>[] = [];
  const controllers: AbortController[] = [];
  for (let step = 0; step < 400; step++) {
    const s = snaps[rnd(snaps.length)];
    switch (rnd(6)) {
      case 0: t.scheduler.scheduleWarm(t.show(s)); break;
      case 1: {
        const c = new AbortController(); controllers.push(c);
        outcomes.push(t.scheduler.requestForeground(t.show(s), c.signal).then(() => "ok", (e: Error) => e.name));
        break;
      }
      case 2: if (controllers.length > 0) controllers[rnd(controllers.length)].abort(); break;
      case 3: t.calls[t.calls.length - 1]?.resolve(); break;
      case 4: await t.clock.advance(rnd(3) === 0 ? 5000 : 150); break;
      default: await flush();
    }
    assert.ok(t.maxOutstanding() <= 1, `two native requests outstanding at step ${step}`);
  }
  t.scheduler.dispose();
  for (const c of t.calls) c.resolve();
  await flush();
  const names = await Promise.all(outcomes);
  assert.ok(names.every((n) => n === "ok" || n === "AbortError" || n === "TimeoutError"), names.join(","));
  for (const c of t.calls) {
    const source = snaps.find((s) => s.promptText === c.request.promptText)!;
    assert.deepEqual(c.request.promptTokenIds, [0, ...source.encodedPromptTokenIds]); // identities preserved, one BOS
  }
  assert.ok(t.maxOutstanding() <= 1); assert.equal(t.clock.timers.size, 0);
});
```

`root_abort_ignoring.test.ts` is left untouched.

## Root test, by trace (not executed)

1. The warm starts at t=300 and becomes `calls[0]`.
2. `requestForeground(b)`:
   - The live warm is a different prompt, so `closeJob` aborts its signal. The first assertion holds.
   - The new fg job is created and `pump()` runs, but `slot` is still `calls[0]`, so it returns. `calls.length === 1`.
3. `resolve()` on `calls[0]`, then flush:
   - `onSettled` clears `slot`.
   - The job is closed, so nothing is published and `cached` stays null.
   - `pump()` finds `b` is current and launches it. `calls.length === 2`.
4. `resolve()` on `calls[1]`, then `await fg`: the result is delivered and `published.length === 0`.

## Behavior changes to review

- **Stale check before dispatch.** A foreground is now checked against `currentIdentity()` before every dispatch, not only at settlement. A non-current snapshot is rejected as `"stale snapshot"` without native work.
  - Before, it was dispatched and then rejected at settle, so the caller sees the same outcome.
  - The integration must still update `currentIdentity` before calling `requestForeground`.
- **Late exact completion is not cached.** An aborted or timed-out job that later completes exactly is still not recorded. The next identical warm is therefore re-issued: a cap-1 request that probably hits the server prompt cache (hypothesis). This follows the "late settlements must not overwrite cached" rule.
- **Due warms wait for the slot.** A due warm waits with no timer while the slot is busy. It dispatches on settlement only if it is still current and no foreground is waiting.

## Remaining risk: transport settlement vs server-side cancellation

What the source proves: at most one unsettled `transport.complete()` promise exists from this client. That is a client-side property only.

- **Server may still be working after the promise settles (hypothesis, unmeasured).** Whether llama-server's slot is actually idle depends on the adapter, which is not yet written. A typical `fetch` rejects as soon as it is aborted, before the server stops working. llama-server may notice the dropped non-streaming connection only between processing steps, so a long cancelled prefill could run to completion. With np=1 the next request would queue server-side behind it. That means no slot overlap, but:
  - foreground latency would include the cancelled task's leftover work;
  - the slot's prompt-cache contents afterwards are unknown. `cached = null` covers this.
  
  The adapter should settle only when the server has actually released the task. Streaming mode or a slot-status check are candidate mechanisms and need measurement.
- **A transport that never settles wedges the scheduler.** It is never freed on its own, by design:
  - every later foreground rejects at its own 5000 ms deadline, so no caller waits without a bound;
  - no warm runs.
  
  The adapter needs its own hard backstop, such as destroying the socket, to guarantee settlement.

Integration is still separate: the native adapter and the editor hooks are not wired.
