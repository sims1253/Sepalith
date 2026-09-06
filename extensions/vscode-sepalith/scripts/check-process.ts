import * as assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { terminateChild } from '../src/process_lifecycle';

async function main() {
  const unrelated = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });
  const child = spawn(process.execPath, ['-e',
    "process.on('SIGTERM', () => {}); console.log('ready'); setInterval(() => {}, 1000)"],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  try {
    await once(child.stdout!, 'data');
    const first = terminateChild(child, 100);
    assert.equal(terminateChild(child, 100), first, 'concurrent stop shares ownership');
    await first;
    assert.notEqual(child.signalCode ?? child.exitCode, null, 'stop observes exit');
    assert.equal(unrelated.exitCode, null, 'unrelated process remains running');
    assert.equal(unrelated.signalCode, null);
    await terminateChild(child, 100); // Stopping an exited process is harmless.
  } finally {
    child.kill('SIGKILL');
    await terminateChild(unrelated, 100);
  }
  console.log('Owned process termination and escalation checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
