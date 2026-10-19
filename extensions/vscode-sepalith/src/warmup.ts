// Prefill-as-you-settle (docs/validation/2026-10-17-prefill-autoresearch.md
// §4): after every change that alters the next completion prompt, send that
// exact prompt with zero generation so llama-server's slot cache already
// holds it when the real request arrives (pooled edit TTFT 690.9 → 74.3 ms
// on the target laptop, §8). Pure — no vscode import — so
// scripts/check-warmup.ts can exercise it under plain node.

/** Idle ms after the last change before a warm-up is sent (§4: ~150–300 ms). */
export const WARMUP_DEBOUNCE_MS = 200;

/**
 * Debounced, single-flight warm-up sender. The prompt is rendered when the
 * warm-up actually starts, so a change that arrives while one is in flight
 * marks the state dirty and the next warm-up uses the newest prompt — queued
 * warm-ups never pile up. In-flight warm-ups are never aborted: b10453 does
 * not cancel an underway prefill, and the real request queued behind it
 * lands on the cache floor.
 */
export class WarmupScheduler {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private dirty = false;
  private disposed = false;
  private lastWarmed = "";
  // null = nothing to warm (no R editor, server not ready, empty document)
  private readonly render: () => Promise<string | null>;
  private readonly send: (prompt: string) => Promise<void>;
  private readonly enabled: () => boolean;
  private readonly delayMs: number;

  // plain fields, not parameter properties: node's type stripping runs the checks
  constructor(render: () => Promise<string | null>, send: (prompt: string) => Promise<void>, enabled: () => boolean, delayMs = WARMUP_DEBOUNCE_MS) {
    this.render = render;
    this.send = send;
    this.enabled = enabled;
    this.delayMs = delayMs;
  }

  /** Note a prompt-changing event; warms after `delayMs` without further events. */
  schedule(): void {
    if (this.disposed || !this.enabled()) return;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.run();
    }, this.delayMs);
  }

  /** Forget the last warmed prompt (the server's cache is gone, e.g. restart). */
  reset(): void {
    this.lastWarmed = "";
  }

  dispose(): void {
    this.disposed = true; // an in-flight warm-up finishes, nothing new starts
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private async run(): Promise<void> {
    if (this.running) {
      this.dirty = true; // the running loop picks up the newest state next
      return;
    }
    this.running = true;
    try {
      do {
        this.dirty = false;
        if (this.disposed || !this.enabled()) break;
        try {
          const prompt = await this.render();
          // an unchanged prompt (e.g. a cursor settle that moved nothing) is already warm
          if (prompt === null || prompt === this.lastWarmed) continue;
          if (this.disposed || !this.enabled()) break; // disabled or disposed while rendering
          await this.send(prompt);
          this.lastWarmed = prompt;
        } catch {
          // best effort: the real request still works cold; `send` logs
        }
        // a change whose debounce is still pending has not settled: its timer starts the next warm-up
      } while (this.dirty && this.timer === null);
    } finally {
      this.running = false;
    }
  }
}
