// Full-TRAIN PRM03 parity (local only; needs the campaign data on disk).
// Streams the DAT10 v4 TRAIN context sidecar and token rows and checks, for
// every row: the TypeScript render equals the row's prompt_text, the editor
// builder rebuilds it from the document state (one-line and empty regions),
// every history event_diff is respelled exactly, and the row's target parses
// to its declared operation and body. With a census parse file from
// build_prm03_fixtures.py --census-all it also compares parse_output and
// valid_generation_tokens on every census sample (arm A outputs on TRAIN).
//
//   node --no-warnings scripts/check-prm03-train.ts [--census /tmp/census-parse.jsonl]
import * as assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { parseOutput, promptContextFromJson, renderPrompt, validGenerationTokens } from '../src/prm03.ts';
import type { PromptContext } from '../src/prm03.ts';
import { buildEditorRequest, eventDiff } from '../src/prm03_context.ts';

const DATA = process.env.SEPALITH_TRAIN_DIR ?? '/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent';
const censusIndex = process.argv.indexOf('--census');
const censusPath = censusIndex > 0 ? process.argv[censusIndex + 1] : null;

async function main() {
  const sidecar = createInterface({ input: createReadStream(`${DATA}/context-sidecar.jsonl`), crlfDelay: Infinity })[Symbol.asyncIterator]();
  const rows = createInterface({ input: createReadStream(`${DATA}/train-token-rows.jsonl`), crlfDelay: Infinity })[Symbol.asyncIterator]();
  const contexts = new Map<string, PromptContext>();
  const counts = { rows: 0, render: 0, rebuild: 0, rebuildSkipped: 0, events: 0, targets: 0 };
  for (;;) {
    const [s, r] = await Promise.all([sidecar.next(), rows.next()]);
    if (s.done || r.done) { assert.equal(s.done, r.done, 'sidecar and rows have the same length'); break; }
    const side = JSON.parse(s.value);
    const row = JSON.parse(r.value);
    assert.equal(side.row_id, row.id);
    assert.equal(row.split, 'train');
    const ctx = promptContextFromJson(side.context);
    if (censusPath) contexts.set(row.id, ctx);
    counts.rows++;
    assert.equal(renderPrompt(ctx), row.prompt_text, `${row.id} render`);
    counts.render++;
    if (ctx.region_old.length <= 1) {
      const lines = [...ctx.prefix, ...(ctx.region_old.length ? ctx.region_old : ['']), ...ctx.suffix_lines];
      const request = buildEditorRequest({ lines, eol: ctx.document_eol === 'crlf' ? 'crlf' : 'lf', cursorLine: ctx.prefix.length,
        cursorUtf16: ctx.cursor.utf16_column ?? 0, path: ctx.path, uri: ctx.replacement_range.uri,
        version: ctx.replacement_range.document_version, contentSha256: ctx.replacement_range.content_sha256, history: ctx.history });
      assert.equal(request.prompt, row.prompt_text, `${row.id} editor rebuild`);
      counts.rebuild++;
    } else counts.rebuildSkipped++;
    for (const event of ctx.history) {
      assert.equal(eventDiff(event.path, event.range_utf16.start.line, event.old_text.split('\n'), event.new_text.split('\n')), event.event_diff, `${row.id} event`);
      counts.events++;
    }
    const parsed = parseOutput(row.target_text, ctx);
    assert.equal(parsed.status, 'accepted', `${row.id} target parse`);
    assert.equal(parsed.operation, row.target_operation, `${row.id} target operation`);
    if (row.target_operation === 'replace') assert.equal(parsed.body_text, row.target_body_text, `${row.id} target body`);
    counts.targets++;
  }
  console.log(`TRAIN rows: ${JSON.stringify(counts)}`);
  if (!censusPath) return;
  let census = 0;
  for await (const line of createInterface({ input: createReadStream(censusPath), crlfDelay: Infinity })) {
    const c = JSON.parse(line);
    assert.deepEqual(parseOutput(c.text, contexts.get(c.context_id)!), c.expected, `${c.id} census parse`);
    assert.equal(validGenerationTokens(c.tokens), c.valid_tokens, `${c.id} census tokens`);
    census++;
  }
  console.log(`census samples: ${census} identical (parse_output and valid_generation_tokens)`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
