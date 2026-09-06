import * as assert from 'node:assert/strict';
import { parseCompletion } from '../src/completion.ts';

assert.deepEqual(parseCompletion({ choices: [{ text: 'mean(x)' }], usage: { completion_tokens: 4 } }),
  { text: 'mean(x)', completionTokens: 4 });
for (const empty of [{}, { choices: [] }, { choices: null, usage: null }, { choices: [null] }]) {
  assert.deepEqual(parseCompletion(empty), { text: '', completionTokens: 0 });
}
for (const malformed of [null, 1, { choices: {} }, { choices: [1] }, { choices: [{ text: 7 }] },
  { usage: 1 }, { usage: { completion_tokens: '4' } }, { usage: { completion_tokens: -1 } },
  { usage: { completion_tokens: 1.5 } }]) {
  assert.throws(() => parseCompletion(malformed), /Invalid completion/);
}
console.log('Completion boundary checks passed');
