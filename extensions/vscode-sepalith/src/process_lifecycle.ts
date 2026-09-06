import type { ChildProcess } from 'node:child_process';

const pending = new WeakMap<ChildProcess, Promise<void>>();

/** Terminate only this owned child; retain the promise until exit is observed. */
export function terminateChild(child: ChildProcess, graceMs = 5000): Promise<void> {
  const existing = pending.get(child);
  if (existing) return existing;
  if (!child.pid || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  const operation = new Promise<void>((resolve, reject) => {
    let escalation: ReturnType<typeof setTimeout>;
    let deadline: ReturnType<typeof setTimeout>;
    const cleanup = () => {
      clearTimeout(escalation);
      clearTimeout(deadline);
      child.removeListener('exit', exited);
    };
    const exited = () => { cleanup(); resolve(); };
    child.once('exit', exited);
    escalation = setTimeout(() => {
      try { child.kill('SIGKILL'); } catch { /* Deadline retains ownership on failure. */ }
    }, graceMs);
    deadline = setTimeout(() => {
      cleanup();
      reject(new Error(`Owned server ${child.pid} did not exit after termination`));
    }, graceMs * 2);
    try { child.kill('SIGTERM'); } catch { /* Escalation still runs. */ }
  });
  pending.set(child, operation);
  void operation.finally(() => pending.delete(child)).catch(() => {});
  return operation;
}
