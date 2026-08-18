# Prompt format: PSM with edit history

How Sepalith builds prompts for edit suggestions. This document explains the
format and the reasons behind each choice. You need it to serve the model or
to build an editor extension.

## The prompt

```
<|history|>
--- a/R/foo.R
+++ b/R/foo.R
@@ -12,1 +12,1 @@
-  mean(x)
+  mean(x, na.rm = TRUE)

<|context|>R/foo.R
... stable file content before the cursor ...
<|cursor|>partial line<|suffix|>
... file content after the cursor ...
<|end|>
```

The model fills in text between `<|cursor|>` and `<|suffix|>`. It stops at
`<|end|>` or when it emits `<|end|>`.

Token budget: history <= 400 tokens, file context <= 1,500 tokens.

## Why this order

Parts of the prompt change at different rates. The slowest parts go first:

| Zone | Changes when | Token cost to re-prefill |
|---|---|---|
| Edit history | Once per debounced edit (~300ms pause) | ~400 |
| File prefix | When the user moves the cursor far | ~1,000 |
| Cursor + suffix | Every keystroke | ~100 |

llama.cpp caches the longest shared prefix between requests. With this
order, a keystroke re-prefills only the cursor zone (~100 tokens, ~5ms).
The stable file prefix and the edit history stay cached. A new edit event
re-prefills the history zone (~400 tokens, ~20ms) plus everything after it.

For comparison: Zeta-2's format puts the suffix first. Every keystroke
invalidates the entire prompt. On a GPU server that re-prefills in 50ms,
this costs nothing. On a laptop CPU where full prefill takes 300ms-2s, it
makes the difference between instant and sluggish.

## How the extension collects edit history

1. Watch `document.onDidChangeTextDocument`.
2. Debounce: wait 300ms after the last change.
3. Diff current content against the last stable state. One unified diff,
   context=0, capped at 20 changed lines.
4. Push the diff into a ring buffer. Keep at most 3 entries. Drop the oldest.
5. Render the oldest entry first (nearest to the file context).
6. Assemble the PSM prompt and send it.

The model sees a sequence of small diffs, not keystrokes. Keystrokes carry
no signal a diff does not already have. Intermediate states that the user
typed through and abandoned do not help the model predict the next edit.

## Training data

Each training example carries zero or more `event_diff` fields. Each field
holds one unified diff of one hunk. The renderer places them in the
`<|history|>` section, oldest first. Examples with no events train the model
to work without history — not every editing session has prior context.

## The Zed problem

Zed has no extension API for custom edit-prediction prompts. Their
`open_ai_compatible_api` provider sends a pre-built prompt that the provider
cannot modify. The model must match Zeta-2's format to work inside Zed.

Two paths:

1. **Contribute a format upstream.** Zed is open source. A PR that adds a
   `prompt_format: "sepalith"` option (or a generic PSM template) to their
   provider would let any PSM-trained model plug in. The edit-prediction
   provider architecture is pluggable by design.
2. **Train a second checkpoint on Zeta-2's format** for Zed users, and serve
   PSM to VS Code and Positron where we control the prompt.

Path 1 helps everyone. It is the right move once the model works.

## VS Code and Positron

The `inlineCompletionProvider` API gives full prompt control. The extension
watches document changes, debounces, diffs, and assembles the PSM prompt.
No API limitation.
