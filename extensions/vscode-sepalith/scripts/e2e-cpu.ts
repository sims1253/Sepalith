// End-to-end test against a real llama-server on CPU with the extension's own
// spawn flags, request bodies, renderer, parser and generation flow. Cases are
// TRAIN rows (DAT10 v4); DEV and the sealed final set are never read. Local
// only: needs the llama.cpp CPU build, the arm A GGUF and the TRAIN data.
//
//   npm run e2e-cpu -- [--cases 6] [--out /tmp/e2e.json]
//   env: SEPALITH_SERVER, SEPALITH_MODEL, SEPALITH_TRAIN_DIR, SEPALITH_THREADS (max 8)
//
// The server runs under `nice -n 10` with -ngl 0 and CUDA hidden.
import * as assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createReadStream, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { homedir } from 'node:os';
import { createInterface } from 'node:readline';
import { completionBody, generateSuggestion, interpretCompletion, parseNativeCompletion, parseTokenize, promptIds, SUGGESTION_MAX_TOKENS, tokenizeBody } from '../src/completion.ts';
import type { NativeCompletion } from '../src/completion.ts';
import { BOS_ID, promptContextFromJson } from '../src/prm03.ts';
import type { PromptContext } from '../src/prm03.ts';
import { buildEditorRequest, presentEdit } from '../src/prm03_context.ts';
import { terminateChild } from '../src/process_lifecycle.ts';
import { DEFAULT_CONTEXT_SIZE, llamaServerArgs } from '../src/server_args.ts';

const SERVER = process.env.SEPALITH_SERVER ?? `${homedir()}/.local/opt/llama.cpp-b10453/build-cpu/bin/llama-server`;
const MODEL = process.env.SEPALITH_MODEL ?? '/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-Q8_0.gguf';
const DATA = process.env.SEPALITH_TRAIN_DIR ?? '/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent';
const THREADS = Math.min(8, Number(process.env.SEPALITH_THREADS ?? 8));
const argValue = (name: string) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : undefined; };
const CASES = Number(argValue('--cases') ?? 6);
const OUT = argValue('--out');
for (const forbidden of ['DAT-07-final', 'sealed-final', 'dev250', 'dev75']) assert.ok(!DATA.includes(forbidden), 'TRAIN data only');

interface CaseReport { id: string; check?: string; outcome: string; reason: string | null; tokens: number[]; content: string;
  token_parity?: boolean; exact_vs_train_target?: boolean; [key: string]: string | number | boolean | null | number[] | undefined }
interface Row { id: string; split: string; family: string; prompt_text: string; target_text: string; target_body_text: string; target_operation: string; input_ids: number[]; target_start: number }

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close(() => resolve(port));
    });
    server.on('error', reject);
  });
}

async function post(port: number, endpoint: string, body: string): Promise<Response> {
  const res = await fetch(`http://127.0.0.1:${port}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: AbortSignal.timeout(900_000) });
  if (!res.ok) throw new Error(`${endpoint}: HTTP ${res.status} ${await res.text()}`);
  return res;
}

// timings (prompt tokens actually processed) are logged next to the validated fields
const lastTimings: { prompt_n?: number; prompt_ms?: number; predicted_n?: number; predicted_ms?: number }[] = [];
async function completion(port: number, body: string): Promise<NativeCompletion> {
  // SAFETY: only the optional timings object is read here; parseNativeCompletion validates the rest.
  const json = await (await post(port, '/completion', body)).json() as { timings?: (typeof lastTimings)[number] };
  lastTimings.push(json.timings ?? {});
  return parseNativeCompletion(json);
}

/** Seeded pick: one row per (family, region shape) stratum, history and CRLF included. */
async function pickRows(n: number): Promise<{ row: Row; ctx: PromptContext; stratum: string }[]> {
  const sidecar = createInterface({ input: createReadStream(`${DATA}/context-sidecar.jsonl`) })[Symbol.asyncIterator]();
  const rows = createInterface({ input: createReadStream(`${DATA}/train-token-rows.jsonl`) })[Symbol.asyncIterator]();
  const wanted = ['finish_block|empty', 'rename_propagation|one', 'no_op|one', 'pipe_rewrite|one', 'no_op|empty', 'format_propagation|one',
    'roxygen_drafting|empty', 'na_rm_propagation|one', 'finish_block|one'];
  const picked = new Map<string, { row: Row; ctx: PromptContext; stratum: string }>();
  let crlf: { row: Row; ctx: PromptContext; stratum: string } | null = null;
  for (let index = 0; ; index++) {
    const [s, r] = await Promise.all([sidecar.next(), rows.next()]);
    if (s.done || r.done) break;
    if (index % 7 !== 3) continue; // deterministic thinning
    const side = JSON.parse(s.value);
    const row: Row = JSON.parse(r.value);
    assert.equal(row.split, 'train');
    const ctx = promptContextFromJson(side.context);
    if (ctx.region_old.length > 1 || row.target_start > 1800) continue; // editor regions are one line; keep CPU prefill short
    const stratum = `${row.family}|${ctx.region_old.length ? 'one' : 'empty'}`;
    if (wanted.includes(stratum) && !picked.has(stratum)) picked.set(stratum, { row, ctx, stratum });
    else if (!crlf && ctx.document_eol === 'crlf' && ctx.history.length) crlf = { row, ctx, stratum: `${stratum}|crlf` };
    if (picked.size === wanted.length && crlf) break;
  }
  const out = wanted.filter(w => picked.has(w)).map(w => picked.get(w)!);
  if (crlf) out.splice(Math.min(n - 1, out.length), 0, crlf);
  return out.slice(0, n);
}

async function main() {
  const cases = await pickRows(CASES);
  const port = await freePort();
  const args = llamaServerArgs({ modelPath: MODEL, port, threads: THREADS, contextSize: DEFAULT_CONTEXT_SIZE, gpuLayers: 0, speculative: false });
  console.log(`spawn: nice -n 10 ${SERVER} ${args.join(' ')}`);
  const child = spawn('nice', ['-n', '10', SERVER, ...args], { stdio: ['ignore', 'ignore', 'pipe'], env: { ...process.env, CUDA_VISIBLE_DEVICES: '' } });
  let log = '';
  child.stderr?.on('data', (d: Buffer) => { log = (log + d.toString()).slice(-20000); });
  const report: CaseReport[] = [];
  try {
    const t0 = Date.now();
    for (;;) {
      if (child.exitCode !== null) throw new Error(`server exited: ${log.slice(-2000)}`);
      try { await post(port, '/completion', completionBody([BOS_ID], 1)); break; } catch { await sleep(2000); }
      if (Date.now() - t0 > 300_000) throw new Error('readiness timeout');
    }
    console.log(`ready after ${Math.round((Date.now() - t0) / 1000)} s`);
    for (const { row, ctx, stratum } of cases) {
      // editor state rebuilt from the TRAIN context: document lines + cursor + history
      const region = ctx.region_old.length ? ctx.region_old : [''];
      const request = buildEditorRequest({ lines: [...ctx.prefix, ...region, ...ctx.suffix_lines], eol: ctx.document_eol === 'crlf' ? 'crlf' : 'lf',
        cursorLine: ctx.prefix.length, cursorUtf16: ctx.cursor.utf16_column ?? 0, path: ctx.path, uri: ctx.replacement_range.uri,
        version: ctx.replacement_range.document_version, contentSha256: ctx.replacement_range.content_sha256, history: ctx.history });
      assert.equal(request.prompt, row.prompt_text, `${row.id}: editor prompt equals the TRAIN prompt`);
      const ids = promptIds(parseTokenize(await (await post(port, '/tokenize', tokenizeBody(request.prompt))).json()));
      const trainIds = row.input_ids.slice(0, row.target_start);
      assert.deepEqual(ids, trainIds, `${row.id}: server token IDs equal the TRAIN input_ids prompt`);
      lastTimings.length = 0;
      const w0 = Date.now();
      await completion(port, completionBody(ids, 0)); // prefill warm-up
      const warmupMs = Date.now() - w0;
      const g0 = Date.now();
      const out = await generateSuggestion(body => completion(port, body), ids, true);
      const requestMs = Date.now() - g0;
      const verdict = interpretCompletion(request.context, out.content, out.tokens, SUGGESTION_MAX_TOKENS);
      const shown = presentEdit(request, verdict.result);
      const exact = row.target_operation === 'no_op' ? verdict.outcome === 'no_edit'
        : verdict.outcome === 'edit' && verdict.result.body_text === row.target_body_text;
      const entry = { id: row.id, stratum, family: row.family, prompt_tokens: ids.length, token_parity: true,
        warmup_ms: warmupMs, request_ms: requestMs, request_prompt_tokens_processed: lastTimings[1]?.prompt_n ?? null,
        generated_tokens: out.tokens.length, stop_token: out.tokens[out.tokens.length - 1], outcome: verdict.outcome, reason: verdict.reason,
        operation: verdict.result.operation, presentation: shown.kind, target_operation: row.target_operation, exact_vs_train_target: exact,
        content: out.content, tokens: out.tokens };
      report.push(entry);
      console.log(JSON.stringify({ ...entry, content: undefined, tokens: undefined }));
      assert.ok(verdict.outcome !== 'invalid' || verdict.reason !== null);
    }
    // forced edit (allowNoEdit off) on the first no-op case that answered [NO_EDIT]
    const noop = cases.find((_c, i) => report[i]?.outcome === 'no_edit' && report[i]?.tokens[0] === 80);
    if (noop) {
      const region = noop.ctx.region_old.length ? noop.ctx.region_old : [''];
      const request = buildEditorRequest({ lines: [...noop.ctx.prefix, ...region, ...noop.ctx.suffix_lines], eol: noop.ctx.document_eol === 'crlf' ? 'crlf' : 'lf',
        cursorLine: noop.ctx.prefix.length, cursorUtf16: noop.ctx.cursor.utf16_column ?? 0, path: noop.ctx.path, uri: noop.ctx.replacement_range.uri,
        version: noop.ctx.replacement_range.document_version, contentSha256: noop.ctx.replacement_range.content_sha256, history: noop.ctx.history });
      const ids = promptIds(parseTokenize(await (await post(port, '/tokenize', tokenizeBody(request.prompt))).json()));
      const forced = await generateSuggestion(body => completion(port, body), ids, false);
      assert.notEqual(forced.tokens[0], 80, 'forced edit never starts with the no-op token');
      const verdict = interpretCompletion(request.context, forced.content, forced.tokens, SUGGESTION_MAX_TOKENS);
      const entry = { id: noop.row.id, check: 'forced_edit', generated_tokens: forced.tokens.length, outcome: verdict.outcome,
        reason: verdict.reason, operation: verdict.result.operation, content: forced.content, tokens: forced.tokens };
      report.push(entry);
      console.log(JSON.stringify({ ...entry, content: undefined, tokens: undefined }));
    }
    // cap hit: a generation stopped by the token limit is never a suggestion
    {
      const { row, ctx } = cases[0];
      const capped = await completion(port, completionBody(row.input_ids.slice(0, row.target_start), 4));
      const verdict = interpretCompletion(ctx, capped.content, capped.tokens, 4);
      assert.equal(capped.stopType, 'limit');
      assert.equal(verdict.outcome, 'invalid');
      assert.equal(verdict.reason, 'cap_hit');
      report.push({ id: row.id, check: 'cap_hit', stop_type: capped.stopType, outcome: verdict.outcome, reason: verdict.reason, tokens: capped.tokens, content: capped.content });
      console.log(JSON.stringify({ check: 'cap_hit', outcome: verdict.outcome, reason: verdict.reason }));
    }
  } finally {
    await terminateChild(child, 5000).catch(() => child.kill('SIGKILL'));
  }
  if (OUT) writeFileSync(OUT, JSON.stringify({ server: SERVER, model: MODEL, threads: THREADS, args, cases: report }, null, 1));
  const primary = report.filter(r => r.check === undefined);
  console.log(`e2e: ${primary.length} TRAIN cases, token parity ${primary.filter(r => r.token_parity).length}/${primary.length}, ` +
    `outcomes ${JSON.stringify(primary.reduce((m: Record<string, number>, r) => { const k = String(r.outcome); m[k] = (m[k] ?? 0) + 1; return m; }, {}))}, ` +
    `exact vs TRAIN target ${primary.filter(r => r.exact_vs_train_target).length}/${primary.length}`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
