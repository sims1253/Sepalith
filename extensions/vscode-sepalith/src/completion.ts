// generation stops: the UPDATED terminator plus every prompt marker the
// model has been seen echoing (marker echo = wasted seconds of generation)
export const STOPS = [">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======", "<[fim-middle]>", "<[fim-suffix]>", "<[fim-prefix]>", "<|outline|>"];

/** Generation budget of a real suggestion request. */
export const SUGGESTION_MAX_TOKENS = 320;

/** Generation budget of a prefill warm-up: prompt only, no decode (b10453 `max_tokens: 0`, verified: `eval time = 0.00 ms`). */
export const WARMUP_MAX_TOKENS = 0;

/** Body for POST /v1/completions; real requests and prefill warm-ups share it so their prompts match byte for byte. */
export function completionBody(prompt: string, maxTokens: number, stop: string[] | null): string {
  return JSON.stringify({ prompt, max_tokens: maxTokens, temperature: 0, stop, stream: false });
}

/** Validated subset of the local server's OpenAI-compatible completion response. */
export interface ParsedCompletion {
  text: string;
  completionTokens: number;
}

// HTTP JSON is untrusted until this parser has checked every consumed field.
// oxlint-disable-next-line anti-slop/no-unknown-parameters
export function parseCompletion(value: unknown): ParsedCompletion {
  if (typeof value !== 'object' || value === null) throw new Error('Invalid completion response');
  let text = '';
  let completionTokens = 0;
  if ('choices' in value && value.choices != null) {
    if (!Array.isArray(value.choices)) throw new Error('Invalid completion choices');
    const first = value.choices[0];
    if (first != null) {
      if (typeof first !== 'object') throw new Error('Invalid completion choice');
      if ('text' in first && first.text != null) {
        if (typeof first.text !== 'string') throw new Error('Invalid completion text');
        text = first.text;
      }
    }
  }
  if ('usage' in value && value.usage != null) {
    if (typeof value.usage !== 'object') throw new Error('Invalid completion usage');
    if ('completion_tokens' in value.usage && value.usage.completion_tokens != null) {
      const tokens = value.usage.completion_tokens;
      if (typeof tokens !== 'number' || !Number.isSafeInteger(tokens) || tokens < 0)
        throw new Error('Invalid completion token count');
      completionTokens = tokens;
    }
  }
  return { text, completionTokens };
}
