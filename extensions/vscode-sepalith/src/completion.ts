// llama-server native request bodies and response parsing for PRM03 serving.
// Pure (no vscode import) so scripts/check-completion.ts runs it under node.
//
// The request mirrors the quant gate (packages/sepalith/src/sepalith/quant/
// gguf_dev.py `completion_request`): token-ID prompt with one manual BOS 0,
// greedy (temperature 0, top_k 1, only the top_k sampler), no penalties, no
// string stops (end of generation is the native EOS 1 or 130073), seed 3407,
// cap 1,024. One difference: `cache_prompt: true`, so the prefill warm-up
// (n_predict 0, same prompt IDs) leaves the slot cache warm for the request.

import { BOS_ID, NATIVE_EOG_IDS, NO_EDIT_FIRST_TOKEN_ID, parseOutput, validGenerationTokens } from "./prm03.ts";
import type { OutputResult, PromptContext } from "./prm03.ts";

/** Generation cap of a suggestion request (the quant gate's GATE_CAP). */
export const SUGGESTION_MAX_TOKENS = 1024;

/** Generation budget of a prefill warm-up: prompt only, no decode. */
export const WARMUP_MAX_TOKENS = 0;

export const REQUEST_SEED = 3407;

/** /tokenize body: training tokenization is HF split_special_tokens=True, i.e. native parse_special false, no BOS. */
export function tokenizeBody(prompt: string): string {
  return JSON.stringify({ content: prompt, add_special: false, parse_special: false, with_pieces: false });
}

/** Prompt IDs sent to /completion: the manual BOS 0 plus the rendered prompt's tokens. */
export function promptIds(tokens: readonly number[]): number[] {
  return [BOS_ID, ...tokens];
}

/**
 * /completion body. Real requests and prefill warm-ups share it (only
 * n_predict differs) so their prompts match token for token.
 * `suppressNoEdit` bans the no-op's first token: used for a one-token step.
 */
export function completionBody(ids: readonly number[], nPredict: number, suppressNoEdit = false): string {
  const body = {
    prompt: ids, n_predict: nPredict, temperature: 0, top_k: 1, top_p: 1, min_p: 0, typical_p: 1,
    repeat_penalty: 1, presence_penalty: 0, frequency_penalty: 0, dry_multiplier: 0, xtc_probability: 0,
    mirostat: 0, ignore_eos: false, stop: [], seed: REQUEST_SEED, cache_prompt: true, return_tokens: true,
    stream: false, n_probs: 0, samplers: ["top_k"],
  };
  return JSON.stringify(suppressNoEdit ? { ...body, logit_bias: [[NO_EDIT_FIRST_TOKEN_ID, false]] } : body);
}

// The HTTP boundary validates each accessed field below. Unknown values must stay unknown here.
// oxlint-disable-next-line anti-slop/no-unsafe-dictionary-type, anti-slop/no-unknown-parameters
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters
function tokenArray(value: unknown, what: string): number[] {
  if (!Array.isArray(value) || value.some(t => typeof t !== "number" || !Number.isSafeInteger(t) || t < 0))
    throw new Error(`Invalid ${what} token IDs`);
  return value.map(t => Number(t));
}

// HTTP JSON is untrusted until these parsers have checked every consumed field.
// oxlint-disable-next-line anti-slop/no-unknown-parameters
export function parseTokenize(value: unknown): number[] {
  if (!isRecord(value)) throw new Error("Invalid tokenize response");
  return tokenArray(value.tokens, "tokenize");
}

export type StopType = "eos" | "limit" | "word" | "none";

/** Validated subset of a llama-server /completion response. */
export interface NativeCompletion {
  content: string;
  tokens: number[]; // generated IDs, including the final end-of-generation ID when there is one
  stopType: StopType;
  tokensEvaluated: number | null;
  truncated: boolean;
}

// oxlint-disable-next-line anti-slop/no-unknown-parameters
export function parseNativeCompletion(value: unknown): NativeCompletion {
  if (!isRecord(value)) throw new Error("Invalid completion response");
  if (typeof value.content !== "string") throw new Error("Invalid completion content");
  const tokens = tokenArray(value.tokens ?? [], "completion");
  const stop = value.stop_type;
  if (stop !== "eos" && stop !== "limit" && stop !== "word" && stop !== "none") throw new Error("Invalid completion stop_type");
  const evaluated = value.tokens_evaluated;
  if (evaluated !== undefined && (typeof evaluated !== "number" || !Number.isSafeInteger(evaluated) || evaluated < 0))
    throw new Error("Invalid completion tokens_evaluated");
  if (value.truncated !== undefined && typeof value.truncated !== "boolean") throw new Error("Invalid completion truncated flag");
  return { content: value.content, tokens, stopType: stop, tokensEvaluated: evaluated ?? null, truncated: value.truncated === true };
}

/** How a generation ended (campaign_eval.generation_termination). */
export interface Termination {
  complete: boolean; // ended on a native end-of-generation ID (1 or 130073)
  capHit: boolean;
  stopTokenId: number | null;
}

export function termination(tokens: readonly number[], cap: number): Termination {
  const last = tokens.length ? tokens[tokens.length - 1] : null;
  const stopTokenId = last !== null && NATIVE_EOG_IDS.includes(last) ? last : null;
  return { complete: stopTokenId !== null, capHit: tokens.length === cap && stopTokenId === null, stopTokenId };
}

export interface Interpretation {
  outcome: "edit" | "no_edit" | "invalid";
  result: OutputResult;
  reason: string | null;
  termination: Termination;
}

/**
 * The gate's classification (campaign_eval.classify): valid only when the
 * generation ends with the canonical EOS 1, carries no hidden control tokens,
 * and the text parses. `[NO_EDIT]`, a body equal to the region, or an empty
 * body for an insertion point are no-ops; every invalid output (cap hit,
 * EOS 130073, missing or duplicate terminal, reserved marker lines) means no
 * suggestion. `content` is the server's text of the generated IDs without the
 * final end-of-generation ID.
 */
export function interpretCompletion(context: PromptContext, content: string, tokens: readonly number[], cap: number): Interpretation {
  const end = termination(tokens, cap);
  const result = parseOutput(content, context);
  let reason = result.status === "accepted" ? null : result.reason;
  if (!validGenerationTokens(tokens)) {
    reason = end.capHit ? "cap_hit" : end.stopTokenId === 130073 ? "noncanonical_eog" : reason ?? "noncanonical_or_missing_EOS";
  }
  if (reason !== null || result.operation === null) return { outcome: "invalid", result, reason: reason ?? "invalid", termination: end };
  return { outcome: result.operation === "no_op" ? "no_edit" : "edit", result, reason: null, termination: end };
}

/**
 * One greedy suggestion through `post` (a /completion call). With
 * `allowNoEdit` off, a no-op answer (first token `[`, id 80) is replaced by
 * the model's best edit: the first step is re-run with id 80 banned, then
 * greedy generation continues from the chosen token (the "forced edit" of
 * docs/training/plan-20260927/decode-time/noop_threshold.py).
 */
export async function generateSuggestion(post: (body: string) => Promise<NativeCompletion>, ids: readonly number[],
                                         allowNoEdit: boolean, cap = SUGGESTION_MAX_TOKENS): Promise<NativeCompletion> {
  const first = await post(completionBody(ids, cap));
  if (allowNoEdit || first.tokens[0] !== NO_EDIT_FIRST_TOKEN_ID) return first;
  const step = await post(completionBody(ids, 1, true));
  if (step.tokens.length !== 1 || step.stopType === "eos") return step;
  const rest = await post(completionBody([...ids, ...step.tokens], cap - 1));
  return { ...rest, content: step.content + rest.content, tokens: [...step.tokens, ...rest.tokens] };
}
