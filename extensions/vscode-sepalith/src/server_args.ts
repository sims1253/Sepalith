// llama-server spawn arguments for the sidecar. Pure, shared by the extension
// and scripts/e2e-cpu.ts so the end-to-end test runs the exact product flags.
//
// Sources: the C09 quant gate (docs/training/plan-20260927/c09/gguf_dev_eval.py:
// `-np 1 -t N -tb N --no-webui --seed 3407 --no-context-shift`, default flash
// attention and batch sizes) and the prefill serving confirmation
// (docs/validation/2026-10-17-prefill-autoresearch.md §4: `--alias sepalith
// --temp 0 -c 16384 --parallel 1 --cache-ram 2048`, no `-fa`, `-ub`,
// `--cache-reuse` or KV-type flags). The gate's `--no-cache-prompt` is not
// used: the prefill warm-up depends on the prompt cache. Speculative decoding
// (PR #68) is opt-in because it failed the lossless gate.

export interface ServerOptions {
  modelPath: string;
  port: number;
  threads: number;
  contextSize: number;
  gpuLayers: number;
  speculative: boolean;
}

export const DEFAULT_CONTEXT_SIZE = 16384;
export const CACHE_RAM_MIB = 2048;

export function llamaServerArgs(o: ServerOptions): string[] {
  return [
    "-m", o.modelPath,
    "--alias", "sepalith",
    "--host", "127.0.0.1",
    "--port", String(o.port),
    "-c", String(o.contextSize),
    "--parallel", "1", // one slot: warm-ups and requests share its cache
    "-t", String(o.threads),
    "-tb", String(o.threads),
    "-ngl", String(o.gpuLayers),
    "--cache-ram", String(CACHE_RAM_MIB), // cap the prompt-cache pool for 8-16 GiB laptops
    "--temp", "0",
    "--seed", "3407",
    "--no-context-shift", // a prompt + cap that does not fit fails instead of silently shifting
    "--no-webui",
    ...(o.speculative ? ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "2", "--spec-ngram-simple-size-m", "4"] : []),
  ];
}
