# RUN-01 extension lint triage and isolated patch

Observed 2026-09-12. This packet reviews the pinned dirty extension at EXEC HEAD `a7345e35219ceb624957115c39b869101026a817`. The EXEC checkout was read only. The candidate changes live under [isolated-extension](./isolated-extension), which is a copied source tree with its `node_modules` symlinked to the already installed EXEC dependency tree.

## Baseline and rule review

The isolated copy of the EXEC files reproduced the lead report with:

```text
./node_modules/.bin/oxlint --deny-warnings src scripts
```

The baseline exited 1 with 79 errors and 3 warnings. The error rules were:

| Rule | Count | Triage |
| --- | ---: | --- |
| `anti-slop/require-safety-comment-for-type-assertion` | 31 | Resolved where the immediately preceding code performs the required runtime check. |
| `anti-slop/no-known-value-widening` | 14 | Resolved by removing redundant anonymous return annotations and one known-value type-predicate round trip. |
| `anti-slop/no-unknown-returns` | 3 | Resolved for the native transport and the test helper by using named response/function contracts. |
| `anti-slop/no-unsafe-dictionary-type` | 11 | Three protocol parser dictionary findings remain; the native transport/test capture dictionaries were replaced with named contracts. |
| `anti-slop/no-unknown-parameters` | 20 | Four parser/config/history boundary families remain intentionally typed `unknown`; see the deferral section. |

The warnings were one unused test fixture helper, one unsafe optional chain, and one listener spread. The first two were removed. The spread was replaced with `Array.from` so cancellation keeps a listener snapshot even when a callback mutates the `Set`; direct iteration would change that behavior.

The copied `.oxlintrc.json` has no documented per-file exception mechanism for these rules. Its only overrides disable `no-unknown-parameters` for `src/runtime.ts`, `src/context_build.ts`, and `scripts/**`. No rule was disabled in the candidate.

## Candidate patch

The reviewable patch is [patch.diff](./isolated-extension/patch.diff), SHA-256 `68a0788c1fc556a31a8a29cb257d1b858499f117592aba6ad8b5be81ae21cfaa`. It changes only the copied tree:

* Redundant inferred return contracts were removed from context selection, campaign selection, history, protocol target construction, and fixture helpers. Runtime values and exported shapes remain inferred from the same expressions.
* Safety comments were added only beside assertions following object, array-element, integer, EOL, or field-specific checks. The training-row return is covered by exact-key and field validators immediately above it.
* `JsonPostTransport` now uses named `NativeRequestBody` and `NativeServingResponse` contracts. The client still runs `record`, `requiredString`, and `integerIds` at the response boundary, so malformed JSON continues to be rejected at runtime. `response.json()` remains isolated behind the existing error handling and one documented top-level owner assertion.
* `offsetAt` performs the same position checks directly instead of passing an already typed `Position` through the unknown-input type predicate. This preserves the existing error message and surrogate/range checks.
* The PRM-05 fixture now uses a checked local completion body instead of unsafe optional chaining and its request captures use the named native request contract.

The copied lint result is [post-lint.txt](../post-lint.txt), with 19 remaining errors and no warnings. They are 16 `no-unknown-parameters` findings and 3 `no-unsafe-dictionary-type` findings, all in parser boundaries described below. No `no-known-value-widening`, `no-unknown-returns`, or unannotated-assertion findings remain.

## Findings intentionally deferred

The remaining `unknown` parameters are actual runtime decoders:

* `campaign_protocol.ts` accepts decoded prompt/context/range/history/training-row values and validates exact keys, primitive types, arrays, hashes, token ranges, geometry, and identities before returning domain interfaces. Replacing `unknown` with a broad JSON alias would hide malformed input from this parser; replacing it with a separate structural schema is a protocol refactor outside a lint triage patch.
* `campaign_selection.ts:integer` validates caller-provided numeric selection fields before arithmetic and must retain rejection of non-numbers received from JavaScript callers.
* `history_provider.ts:validOptionalOffset` validates optional editor offsets at the event boundary and must retain rejection of malformed host values.
* `campaign_requests.ts:validateRequestTimeoutMs` validates configuration values at the VS Code configuration boundary and must retain rejection of non-finite, out-of-range, or non-number values.

The three remaining unsafe dictionaries are `campaign_protocol.ts:asRecord` and its `exactKeys` consumers. They are the dynamic object view used by those exact-key parsers. The rule classifies `Record<string, unknown>` as unsafe by design; a recursive JSON dictionary alias would be classified as the same union escape hatch. A future schema pass should replace this with a named decoded JSON representation while retaining exact-key and field-level validation. No suppression or unsafe cast was added here.

## Validation

All commands below ran against the copied tree, with no model, server, GPU, network, or editor activity:

| Command | Result |
| --- | --- |
| `npm run compile` | PASS; TypeScript `--noEmit` |
| `npm run build` | PASS; compile plus esbuild bundle, `dist/extension.js` 129047 bytes |
| `npm run check-prm05` | PASS; 38 assertions |
| `node --experimental-strip-types scripts/check-campaign-protocol.ts` | PASS; 14 checks |
| `node --experimental-strip-types scripts/check-campaign-requests.ts` | PASS; 5 lifecycle scenarios and 5 timeout assertions |
| `node --experimental-strip-types scripts/check-campaign-selection.ts` | PASS; 8 fixture rows, fixture SHA `cae8e84e87a5c3f06479b6d49f156a0616fbff166576c9fe81d9a0ee47a14fa3` |
| `node --experimental-strip-types scripts/check-history.ts` | PASS; 51 checks |
| `node scripts/check-provider-requests.cjs` | PASS; 9 provider lifecycle scenarios |
| `npm run check-runtime-profile` | PASS; primary profile checks |
| `npm run check-process` | PASS; owned process termination/escalation checks |
| `npm run test:lint` | PASS; all 15 anti-slop rule tests |
| `npm run check-runtime` | BLOCKED in the isolated layout; its unchanged script imports the shared fixture at `../../../tests/fixtures/release-manifests.json`, which is outside the owned copy. No fixture or EXEC file was copied or changed. |
| `./node_modules/.bin/oxlint --deny-warnings src scripts` | EXPECTED FAIL; 19 deferred parser-boundary errors, zero warnings |

The copied build output and command logs are retained under [isolated-extension](./isolated-extension). The source diff has no whitespace diagnostics; `git diff --no-index --check` returned no diagnostic output (exit 1 reflects the intentional file differences).

## Source identity and mutation boundary

The EXEC package/config hashes were unchanged during this review:

| File | SHA-256 |
| --- | --- |
| `extensions/vscode-sepalith/.oxlintrc.json` | `4e25843b76a26b10b987b99151b48da9adbda25c28b66c757fce6eab32be29ca` |
| `extensions/vscode-sepalith/tsconfig.json` | `3808f18c63ffdbb05117e33967f5d50bf48ee0032b27f965ea0556791a95be90` |
| `extensions/vscode-sepalith/package.json` | `2a9edf2996852a37057008d8255cf61352407d8aecf98a59635e3f0a505b94a2` |
| `extensions/vscode-sepalith/package-lock.json` | `ef1daabf3f81aea2cdc6e7a00248745d898eeed7caca5df1269f4bc846218880` |
| `extensions/vscode-sepalith/src/campaign_client.ts` | `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |
| `extensions/vscode-sepalith/src/campaign_protocol.ts` | `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824` |
| `extensions/vscode-sepalith/src/campaign_requests.ts` | `4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535` |
| `extensions/vscode-sepalith/src/campaign_selection.ts` | `70a03d86c8d1c2cd0e117edcbd69cfbac193382b20b555e2d0a327e7e006e8e3` |
| `extensions/vscode-sepalith/src/context_select.ts` | `a72edaf733395d7b4839f2ca9ed0c2092e4051359279f6839c93e2d5d04b9d59` |
| `extensions/vscode-sepalith/src/history_provider.ts` | `b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251` |
| `extensions/vscode-sepalith/src/extension.ts` | `4fa6a06d3d3cbf69eda51195dc81b8b793739a1ed0f0763ad3d7d63a475a9727` |
| `extensions/vscode-sepalith/src/runtime.ts` | `4de415237088984d53eba6b182dba975c12216a516c52f9fecb3a916dc841616` |
| `extensions/vscode-sepalith/src/next_edit.ts` | `c184e0d26499290ce3a8f339f2172343ab233b0799fac57a86476ff14e0e2214` |

The full dirty helper/source inventory, including scripts and the pinned HEAD, is recorded in [RUN-01-extension-lint-preparation.json](../../receipts/RUN-01-extension-lint-preparation.json). Root can apply or reject the candidate after reviewing the protocol API impact; the EXEC source remains untouched and no VSIX was rebuilt from this candidate.
