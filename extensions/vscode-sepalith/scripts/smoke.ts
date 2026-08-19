// Smoke test for the extension plumbing (SPEC.md acceptance item 3).
// Starts the sidecar per the spec's sidecar contract, waits for readiness
// (rule 2), renders one zeta2 prompt for a tiny synthetic R buffer, prints
// the parsed suggestion and latency. Zero-score is fine; crashes are not.
// Run: npx tsx scripts/smoke.ts
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";

// same defaults as the extension's package.json configuration
const SERVER_PATH = "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server";
const MODEL_PATH = "/home/m0hawk/Documents/Sepalith/experiments/models/abl_dropout-Q8_0.gguf";
const PORT = 18099;
const THREADS = 8; // shared machine: never more than 8
const CONTEXT = 8192;

// tiny synthetic R buffer; cursor sits on empty line 3 — v0 empty region (marker alone)
const PREFIX_LINES = ["fib <- function(n) {", "    if (n < 2) return(n)"];
const SUFFIX_LINES = ["}", "", "fib(10)"];

function renderPrompt(): string {
  // must match src/extension.ts renderPrompt (and run_eval.py render_zeta2)
  return [
    "<[fim-suffix]>",
    ...SUFFIX_LINES,
    "<[fim-prefix]><filename>smoke.R",
    ...PREFIX_LINES,
    "<<<<<<< CURRENT",
    "<|user_cursor|>",
    "=======",
    "<[fim-middle]>",
  ].join("\n");
}

function parsePrediction(text: string): string[] {
  if (text.includes(">>>>>>>")) text = text.split(">>>>>>>")[0];
  text = text.split("<|user_cursor|>").join("");
  const lines = text.split("\n").map((l) => l.replace(/\r$/, ""));
  while (lines.length && lines[0].trim() === "") lines.shift();
  while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
  return lines;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function post(body: unknown, timeoutMs: number): Promise<Response> {
  return fetch(`http://127.0.0.1:${PORT}/v1/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

async function ready(): Promise<boolean> {
  try {
    const r = await post({ prompt: "x", max_tokens: 1, temperature: 0, stop: null, stream: false }, 20_000);
    return r.status === 200; // rule 2: HTTP 200 on completions == ready (/health lies during load)
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  let child: ChildProcess | null = null;
  let done = false;
  const rings = { out: [] as string[], err: [] as string[] };
  // rule 3: kill ONLY the tracked child pid — never pkill by name
  const killOurs = () => {
    if (child && child.exitCode === null) child.kill("SIGTERM");
  };
  process.on("SIGINT", () => {
    killOurs();
    process.exit(130);
  });
  process.on("SIGTERM", () => {
    killOurs();
    process.exit(143);
  });

  // rule 4: if the port already answers, use the external server; spawn nothing
  let external = false;
  try {
    await fetch(`http://127.0.0.1:${PORT}/health`, { signal: AbortSignal.timeout(1500) });
    external = true;
  } catch {
    external = false;
  }
  if (external) {
    console.log(`smoke: port ${PORT} already answering — using external server`);
  } else {
    console.log(`smoke: spawning ${SERVER_PATH} (model ${MODEL_PATH}, port ${PORT}, -t ${THREADS}, -c ${CONTEXT}, -ngl 0)`);
    child = spawn(
      SERVER_PATH,
      ["-m", MODEL_PATH, "--port", String(PORT), "--host", "127.0.0.1", "-c", String(CONTEXT), "--parallel", "1", "-t", String(THREADS), "-ngl", "0"],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    console.log(`smoke: sidecar pid ${child.pid}`);
    const ring = (which: "out" | "err", d: Buffer) => {
      const parts = d.toString().split("\n");
      rings[which].push(...parts.slice(0, -1));
      if (rings[which].length > 50) rings[which].splice(0, rings[which].length - 50);
    };
    child.stdout?.on("data", (d: Buffer) => ring("out", d));
    child.stderr?.on("data", (d: Buffer) => ring("err", d));
    child.on("exit", (code, sig) => {
      if (done) return;
      console.error(`smoke: FAIL sidecar exited early (code=${code} sig=${sig})`);
      console.error("--- stdout tail ---\n" + rings.out.join("\n"));
      console.error("--- stderr tail ---\n" + rings.err.join("\n"));
      process.exit(1);
    });
  }

  // rule 2: poll every 3 s, give up after 180 s
  const t0 = Date.now();
  while (!(await ready())) {
    const s = Math.round((Date.now() - t0) / 1000);
    if (s > 180) {
      console.error("smoke: FAIL readiness timeout after 180 s");
      console.error("--- stdout tail ---\n" + rings.out.join("\n"));
      console.error("--- stderr tail ---\n" + rings.err.join("\n"));
      killOurs();
      process.exit(1);
    }
    await sleep(3000);
  }
  console.log(`smoke: server ready after ${Math.round((Date.now() - t0) / 1000)} s`);

  const prompt = renderPrompt();
  console.log(`smoke: prompt (${prompt.length} chars):`);
  console.log(prompt.split("\n").map((l) => "  | " + l).join("\n"));

  const t1 = Date.now();
  const res = await post({ prompt, max_tokens: 640, temperature: 0, stop: [">>>>>>> UPDATED"], stream: false }, 300_000);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { choices?: { text?: string }[]; usage?: { completion_tokens?: number } };
  const latency = Date.now() - t1;
  const text = data.choices?.[0]?.text ?? "";
  console.log(`smoke: latency ${latency} ms, completion tokens ${data.usage?.completion_tokens ?? "?"}`);
  const lines = parsePrediction(text);
  console.log(`smoke: parsed suggestion (${lines.length} lines):`);
  console.log(lines.map((l) => "  > " + l).join("\n") || "  (empty — zero score is fine, crashes are not)");
  done = true;
  killOurs();
  console.log("smoke: OK");
}

main().catch((e) => {
  console.error("smoke: FAIL", e);
  process.exit(1);
});
