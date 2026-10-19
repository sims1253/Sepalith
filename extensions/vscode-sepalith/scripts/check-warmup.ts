// Unit tests for the prefill warm-up (src/warmup.ts): debounce, at most one
// warm-up in flight with newest-state supersede, the disable setting, and
// that a warm-up body is the real request body with zero generation.
// No vscode import, plain node. Run: npm run check-warmup
import * as assert from 'node:assert/strict';
import { completionBody, STOPS, SUGGESTION_MAX_TOKENS, WARMUP_MAX_TOKENS } from '../src/completion.ts';
import { buildScopedPrompt } from '../src/context_build.ts';
import { WarmupScheduler } from '../src/warmup.ts';

const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));
const DEBOUNCE = 40;

/** Fake server: records sends, optionally holds them until released. */
class Harness {
  prompt = 'p0';
  enabled = true;
  hold = false;
  fail = false;
  renders = 0;
  sent: string[] = [];
  inFlight = 0;
  maxInFlight = 0;
  private release: (() => void)[] = [];
  readonly scheduler = new WarmupScheduler(
    async () => { this.renders++; return this.prompt; },
    async prompt => {
      this.inFlight++;
      this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
      this.sent.push(prompt);
      try {
        if (this.hold) await new Promise<void>(resolve => this.release.push(resolve));
        if (this.fail) throw new Error('HTTP 503');
      } finally { this.inFlight--; }
    },
    () => this.enabled,
    DEBOUNCE,
  );
  releaseAll(): void { for (const resolve of this.release.splice(0)) resolve(); }
  get held(): number { return this.release.length; }
}

async function main() {
  // debounced: a burst of keystrokes sends one warm-up after the last one
  {
    const h = new Harness();
    for (let i = 1; i <= 5; i++) { h.prompt = `p${i}`; h.scheduler.schedule(); await delay(10); }
    assert.deepEqual(h.sent, [], 'no warm-up before the debounce elapses');
    await delay(DEBOUNCE * 2);
    assert.deepEqual(h.sent, ['p5'], 'one warm-up, newest prompt');
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    assert.deepEqual(h.sent, ['p5'], 'an unchanged prompt is already warm');
    console.log('PASS warm-up debounced, unchanged prompt not resent');
  }
  // single flight: changes during a held warm-up collapse into one follow-up
  {
    const h = new Harness();
    h.hold = true;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    assert.equal(h.held, 1);
    for (let i = 1; i <= 4; i++) { h.prompt = `q${i}`; h.scheduler.schedule(); await delay(DEBOUNCE * 2); }
    assert.equal(h.inFlight, 1, 'still one warm-up in flight');
    assert.deepEqual(h.sent, ['p0'], 'queued changes wait for the in-flight warm-up');
    h.hold = false; h.releaseAll(); await delay(20);
    assert.deepEqual(h.sent, ['p0', 'q4'], 'one follow-up from the newest state');
    assert.equal(h.maxInFlight, 1);
    console.log('PASS at most one warm-up in flight, newest state supersedes');
  }
  // disabled by the setting: no render, no request
  {
    const h = new Harness();
    h.enabled = false;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    assert.equal(h.renders, 0); assert.deepEqual(h.sent, []);
    h.enabled = true; h.hold = true;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    h.prompt = 'later'; h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    h.enabled = false; h.hold = false; h.releaseAll(); await delay(20);
    assert.deepEqual(h.sent, ['p0'], 'disabling drops the queued follow-up');
    console.log('PASS warm-up disabled by the setting');
  }
  // a failed warm-up neither wedges the scheduler nor counts as warm
  {
    const h = new Harness();
    h.fail = true;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    h.fail = false;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    assert.deepEqual(h.sent, ['p0', 'p0'], 'failed prompt is retried');
    h.scheduler.reset(); h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    assert.equal(h.sent.length, 3, 'reset (server restart) re-warms');
    h.scheduler.schedule(); h.scheduler.dispose(); await delay(DEBOUNCE * 2);
    assert.equal(h.sent.length, 3, 'dispose cancels the pending timer');
    console.log('PASS failure, reset and dispose');
  }
  // a follow-up waits for the newest change to settle instead of firing on a stale dirty flag
  {
    const h = new Harness();
    h.hold = true;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    h.prompt = 'q1'; h.scheduler.schedule(); await delay(DEBOUNCE * 2); // settled while in flight: dirty
    h.prompt = 'q2'; h.scheduler.schedule(); // still typing: its debounce is pending
    h.hold = false; h.releaseAll(); await delay(5);
    assert.deepEqual(h.sent, ['p0'], 'no follow-up while the newest change has not settled');
    await delay(DEBOUNCE * 2);
    assert.deepEqual(h.sent, ['p0', 'q2'], 'one follow-up after it settles, newest state');
    console.log('PASS follow-up waits for the latest debounce');
  }
  // disabled or disposed while a warm-up is rendering or in flight: nothing more is sent
  {
    let finishRender: () => void = () => {};
    let on = true;
    const sent: string[] = [];
    const s = new WarmupScheduler(async () => { await new Promise<void>(r => { finishRender = r; }); return 'r'; },
      async prompt => { sent.push(prompt); }, () => on, DEBOUNCE);
    s.schedule(); await delay(DEBOUNCE * 2);
    on = false; finishRender(); await delay(10);
    assert.deepEqual(sent, [], 'disabling during the render drops that warm-up');
    const h = new Harness();
    h.hold = true;
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    h.prompt = 'q1'; h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    h.scheduler.dispose(); h.hold = false; h.releaseAll(); await delay(DEBOUNCE * 2);
    h.scheduler.schedule(); await delay(DEBOUNCE * 2);
    assert.deepEqual(h.sent, ['p0'], 'dispose lets the in-flight warm-up finish and starts nothing new');
    console.log('PASS disable during render, dispose during flight');
  }
  // body: the warm-up is the real request with zero generation
  {
    const lines = ['library(stats)', 'fit_model <- function(df) {', '  model <- lm(y ~ x, data = df)', '  ', '}', 'summary(fit_model(cars))'];
    const { prompt } = buildScopedPrompt(lines, 3, 2, 'R/model.R', null);
    const real = JSON.parse(completionBody(prompt, SUGGESTION_MAX_TOKENS, STOPS));
    const warm = JSON.parse(completionBody(prompt, WARMUP_MAX_TOKENS, STOPS));
    assert.equal(warm.prompt, real.prompt, 'identical prompt text');
    assert.equal(warm.max_tokens, 0, 'zero generation');
    assert.deepEqual({ ...warm, max_tokens: real.max_tokens }, real, 'otherwise the same request');
    console.log('PASS warm-up body equals the real request prompt with max_tokens 0');
  }
  console.log('Warm-up checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
