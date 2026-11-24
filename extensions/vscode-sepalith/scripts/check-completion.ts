// Output-side unit tests: native response parsing and the gate classification
// (edit, no-op, malformed, cap hit, non-canonical EOS) on a TRAIN-shaped context.
// Plain node, no vscode. Run: npm run check-completion
import * as assert from 'node:assert/strict';
import { completionBody, interpretCompletion, parseNativeCompletion, parseTokenize, termination } from '../src/completion.ts';
import { buildEditorRequest, presentEdit, sha256Text } from '../src/prm03_context.ts';

// response parsing
assert.deepEqual(parseTokenize({ tokens: [5, 6] }), [5, 6]);
for (const bad of [null, {}, { tokens: [1.5] }, { tokens: ['a'] }, { tokens: [-1] }]) assert.throws(() => parseTokenize(bad), /Invalid/);
assert.deepEqual(parseNativeCompletion({ content: 'x', tokens: [7, 1], stop_type: 'eos', tokens_evaluated: 12, truncated: false }),
  { content: 'x', tokens: [7, 1], stopType: 'eos', tokensEvaluated: 12, truncated: false });
for (const bad of [null, { content: 1, stop_type: 'eos' }, { content: '', stop_type: 'stop' }, { content: '', stop_type: 'eos', tokens: [0.5] },
  { content: '', stop_type: 'eos', tokens_evaluated: -1 }, { content: '', stop_type: 'eos', truncated: 'no' }]) {
  assert.throws(() => parseNativeCompletion(bad), /Invalid completion/);
}
const forced = JSON.parse(completionBody([0, 9], 1, true));
assert.deepEqual(forced.logit_bias, [[80, false]], 'forced edit bans the no-op first token');
assert.equal(JSON.parse(completionBody([0, 9], 1)).logit_bias, undefined);
assert.deepEqual(termination([5, 1], 1024), { complete: true, capHit: false, stopTokenId: 1 });
assert.deepEqual(termination([5, 130073], 1024), { complete: true, capHit: false, stopTokenId: 130073 });
assert.deepEqual(termination([5, 6], 2), { complete: false, capHit: true, stopTokenId: null });

// classification on an editor-built context: cursor in the middle of `  xpos <- xlim[1]`
const lines = ['f <- function(x) {', '  xlim2 <- par("usr")[1:2]', '  xpos <- xlim[1]', '}', ''];
const text = lines.join('\n');
const request = buildEditorRequest({ lines, eol: 'lf', cursorLine: 2, cursorUtf16: 9, path: 'R/f.R', uri: 'file:///w/R/f.R',
  version: 3, contentSha256: sha256Text(text), history: [] });
const ctx = request.context;
const edit = interpretCompletion(ctx, '  xpos <- xlim2[1]\n>>>>>>> UPDATED', [300, 301, 1], 1024);
assert.equal(edit.outcome, 'edit');
assert.deepEqual(edit.result.body, ['  xpos <- xlim2[1]']);
assert.deepEqual(presentEdit(request, edit.result),
  { kind: 'replace', line: 2, endCharacter: 17, text: '  xpos <- xlim2[1]', oldText: '  xpos <- xlim[1]' });
const append = interpretCompletion(ctx, '  xpos <- xlim[1] + x\n  ypos <- 1\n>>>>>>> UPDATED\n', [300, 1], 1024);
assert.equal(presentEdit(request, append.result).kind, 'replace', 'cursor mid-line: an append after the cursor text is a preview, not ghost text');
const atEnd = buildEditorRequest({ lines, eol: 'crlf', cursorLine: 2, cursorUtf16: 17, path: 'R/f.R', uri: 'file:///w/R/f.R',
  version: 3, contentSha256: sha256Text(text), history: [] });
assert.deepEqual(presentEdit(atEnd, interpretCompletion(atEnd.context, '  xpos <- xlim[1] + x\n  ypos <- 1\n>>>>>>> UPDATED', [300, 1], 1024).result),
  { kind: 'ghost', line: 2, character: 17, text: ' + x\r\n  ypos <- 1' }, 'insertion at the cursor is ghost text, CRLF mapped');
for (const [raw, tokens] of [['[NO_EDIT]\n>>>>>>> UPDATED', [80, 1]], ['  xpos <- xlim[1]\n>>>>>>> UPDATED', [300, 1]]] as const) {
  const r = interpretCompletion(ctx, raw, tokens, 1024);
  assert.equal(r.outcome, 'no_edit', `${raw} is a no-op`);
  assert.deepEqual(presentEdit(request, r.result), { kind: 'none' });
}
const deletion = interpretCompletion(ctx, '>>>>>>> UPDATED', [300, 1], 1024);
assert.equal(deletion.result.operation, 'delete');
assert.equal(presentEdit(request, deletion.result).kind, 'replace');
const malformed: [string, number[], string][] = [
  ['  xpos <- 1', [300, 1], 'missing_exact_terminal'],
  ['a\n>>>>>>> UPDATED\n>>>>>>> UPDATED', [300, 1], 'duplicate_exact_terminal'],
  ['a\n>>>>>>> UPDATED\nmore', [300, 1], 'content_after_terminal'],
  ['=======\n>>>>>>> UPDATED', [300, 1], 'reserved_full_body_line'],
  ['[NO_EDIT]\nx\n>>>>>>> UPDATED', [300, 1], 'reserved_full_body_line'],
  ['<unused_token_5>\n>>>>>>> UPDATED', [300, 1], 'reserved_slot_literal_has_inconsistent_tokenization'],
  ['  xpos <- xlim2[1]\n>>>>>>> UPDATED', [300, 130073], 'noncanonical_eog'],
  ['  xpos <- xlim2[1]\n>>>>>>> UP', [300, 301], 'cap_hit'],
  ['  xpos <- xlim2[1]\n>>>>>>> UPDATED', [300, 3, 1], 'noncanonical_or_missing_EOS'],
];
for (const [raw, tokens, reason] of malformed) {
  const r = interpretCompletion(ctx, raw, tokens, 2 + (reason === 'cap_hit' ? 0 : 1000));
  assert.equal(r.outcome, 'invalid', raw);
  assert.equal(r.reason, reason, raw);
}
// insertion point (blank cursor line): empty region, empty body is a no-op
const blank = buildEditorRequest({ lines: ['f <- function(x) {', '  ', '}'], eol: 'lf', cursorLine: 1, cursorUtf16: 2, path: 'R/f.R',
  uri: 'file:///w/R/f.R', version: 1, contentSha256: sha256Text('f'), history: [] });
assert.deepEqual(blank.context.region_old, []);
assert.equal(interpretCompletion(blank.context, '\n>>>>>>> UPDATED', [300, 1], 1024).outcome, 'no_edit');
assert.deepEqual(presentEdit(blank, interpretCompletion(blank.context, '  x + 1\n>>>>>>> UPDATED', [300, 1], 1024).result),
  { kind: 'ghost', line: 1, character: 2, text: 'x + 1' }, 'auto-indented blank line: the body replaces its whitespace');
console.log('Completion and output classification checks passed');
