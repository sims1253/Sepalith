# Prompt format: PSM with edit history

How Sepalith builds prompts for edit suggestions. This document explains the
format and the reasons behind each choice. You need it to serve the model or
to build an editor extension.

## The prompt

```
<|context|>R/foo.R
... file content before the cursor ...
<|history|>
--- a/R/foo.R
+++ b/R/foo.R
@@ -12,1 +12,1 @@
-  mean(x)
+  mean(x, na.rm = TRUE)

<|cursor|>partial line<|suffix|>
... file content after the cursor ...
<|end|>
```

The model fills in text between `<|cursor|>` and `<|suffix|>`. It stops at
`<|end|>` or when it emits `<|end|>`.

## Context window and truncation

Capture the entire file before the cursor by default. A longer prefix means
more cache to hit on every subsequent request. The cold-start cost (first
prefill of a large file) is paid once per editing session. Every request
after that reuses the cached prefix.

When the total prompt would exceed the model's context window minus a
generation reserve (512 tokens), truncate from the **start** of the file —
the part furthest from the cursor.

### Truncation anchor with hysteresis

A fixed token budget shifts the truncation point on every cursor move.
Any shift at the start of the prompt kills the entire cache. Fix: keep
the truncation point (the *anchor*) stable until it is forced to move.

Rules for the extension:

1. The anchor starts at the beginning of the file (line 0).
2. On each request, check whether content from the anchor to the cursor
   fits the budget. If yes, keep the anchor. Do not move it closer to
   the cursor.
3. If the content exceeds the budget, advance the anchor in **large
   steps** (e.g., 512 tokens) — not by the minimum amount. Large steps
   mean the next several cursor moves stay within the same anchor.
4. Never move the anchor backward within one editing session, even if
   the cursor moves back. The prompt starting mid-file is fine; the model
   has seen mid-file starts in training.
5. Reset the anchor when the user switches files or the session ends.

Cost: with 512-token steps, the anchor moves once per ~500 tokens of
cursor travel. Each move re-prefills the prompt. Between moves, every
keystroke and cursor jump hits the full cached prefix.

This is an extension-side concern. The model sees whatever prefix it
gets. Training data includes prompts that start mid-file at arbitrary
points, so the model does not expect a file to start at line 1.

This is an argument for a model with a long context window and efficient KV
growth. The GDN hybrid architecture (Qwen3.5) grows its KV cache
near-linearly, so a 16-32K context costs a fraction of what a standard
transformer would use. That lets the extension keep more of the file in
the prompt without hitting memory limits on a laptop.

Token budgets are targets, not enforced limits. They guide training data
construction (what distribution the model sees) and extension context
assembly (what to truncate). The model itself does not check or enforce
them. Default targets: history <= 400 tokens, file context: as much as
fits.

## Why this order

llama.cpp caches the longest shared prefix between consecutive requests.
The prompt order determines how much of that prefix survives each kind of
change. Put the section with the longest shared prefix first.

Consider two requests in one editing session. The user types at line 150.
Then a debounce fires and pushes a new diff into the history.

**File prefix first (this format):** the cache hits for every line before
the edit that triggered the debounce. If that edit was at line 80, lines
1-79 are still identical. Only the history section and everything after
it re-prefills. In the common case the user edits near the same spot, so
the head of the file is long and stable — the cache hits most of the prompt.

**History first:** any new edit event changes the history at position 0.
The cache hits nothing. Every request after an edit re-prefills the entire
prompt, including the file prefix that barely changed.

**Keystroke within one position (both orders):** the file prefix and the
history are unchanged. Only the cursor zone re-prefills. This case does
not distinguish the two orders.

| Change type | File-prefix-first cost | History-first cost |
|---|---|---|
| Keystroke | ~100 tokens (cursor zone) | ~100 tokens (same) |
| Edit event at line L of N | N-L lines + history + cursor | Full prompt |

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

## Scope-aware context

Two additions to the raw file context: the enclosing function and a file
outline. Both serve cache stability first, model quality second.

### The enclosing function is always complete

When the cursor is inside a function definition, the ENTIRE function is in
the prompt. The part above the cursor is ordinary file prefix. The rest of
the function is forced to the head of the suffix section, before the older
file-below content:

```
<|context|>R/foo.R
... file above the cursor (includes the function's opening lines) ...
<|history|>
...
<|cursor|>partial line<|suffix|>
... the REST of the enclosing function (pinned, never truncated) ...
... older file content below the function (truncated from its end) ...
<|end|>
```

Rules:

1. Truncation never eats the pinned function remainder. Prefix truncation
   proceeds from the file start (the anchor rules above); suffix truncation
   cuts the deepest lines below the function, not the function itself.
2. At top level (cursor outside any function), nothing is pinned.
3. The scope comes from tree-sitter (syntax only, always available). LSP or
   `ry` can refine it later, but scope resolution must never require them.

Cache cost: while the cursor stays inside one function, the pinned block is
byte-identical across requests — it lives in the stable suffix head, and
only the cursor zone re-prefills, exactly as before. Crossing into another
function changes the pinned block once; that re-prefill amortizes over all
the keystrokes spent in the new function, like an anchor step.

Why: mid-function suggestions otherwise work blind to the function's end.
Comment and roxygen drafting REQUIRE the full function — you cannot write
the docs for code you cannot see. Training data renders mid-function
examples under the same rule (the function remainder appears below the
cursor), so what the model sees at inference matches what it learned on.

### File outline as the no-LSP conditioning slot

A one-line-per-top-level-signature outline of the current file:

```
<|outline|>
fit_model(data, weights, method = "glm")
predict.fit_model(object, newdata)
drop_unused_levels(x)
```

Placement: between the file prefix and the history, ordered by churn —
prefix (rarely changes) < outline (changes on structural edits) < history
(changes on every debounced edit) < cursor zone. An outline change
re-prefills history plus cursor zone, not the file prefix.

Rules:

1. Deduplicate against the pinned scope: the enclosing function's signature
   is dropped from the outline when the function is already fully present.
2. Signatures only — no types, no bodies, no nested definitions. This is an
   index of the file's vocabulary, not a dump of its AST.
3. The outline occupies the SAME conditioning slot as type information:
   plain prompt (neither), outline (tree-sitter, always available),
   outline plus types (`ry`/LSP). Training uses the conditioning dropout:
   each example carries one of the three levels, seed-fixed per row.

The conditioning ablation (2026-08-19) found no benefit from type
conditioning at the current scale — so the outline is a hypothesis to test
on scenario-level evals, not a settled win. The enclosing-function pin is
different: it changes what of the target is visible, and the drafting
families depend on it structurally.

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
