# RUN-04 transition-panel independent review

The v2 packet is suitable for bounded synthetic renderer/history and transport
preparation. It is not sufficient by itself to admit a live native replay or
an editor application result.

## Verified evidence

The provided CPU command and the driver’s `--fixture ... --check` command both
passed: 14 renderer assertions and 14 mock-transport assertions in each run.
The canonical fixture is nine synthetic events: baseline, two typing changes,
cursor movement, history change, diagnostics refresh, anchor move, file
switch, and an unchanged-repeat control. Its SHA-256 is
`e6aaf98c888cdcc3c3908d98e0e817ae65d8c94c5a17c966a9019a9d8e2bc8ea`.

The renderer checks use the exact pinned execution files and reject a changed
source hash. The seven recorded hashes cover `campaign_protocol.ts`,
`campaign_selection.ts`, `history_provider.ts`, `context_select.ts`,
`campaign_requests.ts`, `campaign_client.ts`, and the PRM-03 Python protocol
mirror. The packet correctly says this is the direct runtime dependency set,
not a complete transitive closure. The current source imports are consistent
with that scope: runtime imports are covered, while `context_build.ts` is a
type-only import.

The fixture carries complete synthetic before/after document URI, version, and
content SHA identities. Each prompt is regenerated through the production
`renderPrompt`, and checks compare the stored prompt and hash to that render.
The observed prompt sizes are 648–966 UTF-16 units for file A and 560 for file
B; the unchanged repeat has an exact prompt hash match and is excluded from
the natural-typing denominator. The history test recovers the exact old value
`2` and new value `3`, and the stale test rejects an older complete-buffer
identity.

`NativeCampaignClient` validates nonnegative in-vocabulary integer token IDs,
prepends exactly BOS 0, requires full prompt `tokens_evaluated`, bounds output
to 192, rejects native control tokens, and requires canonical EOS 1. The
driver pins `/tokenize` with `add_special=false`, `parse_special=false`, and
`with_pieces=false`, and pins greedy integer-ID `/completion` with
`cache_prompt=true`, `return_tokens=true`, and one manual BOS. The mock checks
exercise these request guards.

The mock transport covers delayed fresh completion, complete-document
supersession with an ignored abort, client cancellation after an observable
completion-dispatch barrier, one underlying request for duplicate subscribers,
deadline timeout, and parser rejection. It keeps client abort separate from
the unverified server-cancellation state. The response metric projection
whitelists only known timing/usage/top-level fields and derives a cache-prefix
difference only when both prompt totals and processed counts exist; it does
not invent a cache-hit flag or shared-prefix metric.

Applicability and protocol evidence are separate. A stale, cancelled, or
timed-out response cannot become a fresh quality row. `[NO_EDIT]` is classified
as `no_op`, while malformed output is `rejected`. `makeApplicationPlan` is
called for identity checking, but the driver never mutates an editor, so no
row demonstrates an actual application or acceptance. Native timing, token
IDs, cache observations, and server cancellation remain unmeasured without a
root-owned endpoint.

## Specific verifier gap

I ran a targeted temporary-fixture regression. Replacing a pinned source hash
failed as expected. Replacing `source.renderer` passed. More materially,
changing `nativeProfile.manualBosId` to 7, `canonicalEosId` to 2, and
`tokenize.add_special` to true also passed all 28 checks. `validateFixtureShape`
checks the source hash map, launch flag, parallelism, output cap, and deadlines,
but does not compare all source metadata or the native tokenization/completion
profile with the imported constants. The actual client still sends BOS 0 and
the pinned flags, while a later trace would return the mutated fixture profile;
that can make a custom fixture’s report inconsistent with its request.

The canonical `check-transition-panel.ts` avoids this for the checked-in
golden fixture by requiring byte-identical regeneration. Root should run that
golden check before every live replay, or harden `validateFixtureShape` to
compare execution root, renderer/schema/tokenizer metadata, tokenization
flags, completion options, BOS/EOS, vocabulary, endpoint placeholder,
parallelism, and deadlines. This review does not edit the shared driver.

## Recommendation

**Admit the packet as synthetic preparation and CPU transport verification.**
**Do not admit live native replay from an arbitrary fixture until the golden
fixture gate or the metadata checks above are bound.** Even after that gate,
root must supply a new endpoint trace, retain native timing/cache/cancellation
denominators, and keep editor application, quality, parity, and promotion
claims open.

The review used one CPU thread and no endpoint, model, server, network, SSH,
CUDA, or source mutation. Temporary mutated fixtures used for the regression
were removed.
