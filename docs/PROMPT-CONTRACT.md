# Edit context and prompt contract

`sepalith.protocol` provides a standard-library input contract and the named
`zeta2-v1` renderer. Existing training, evaluation and extension callers keep
their current paths. This module does not change the production prompt format.

## Inputs and snapshots

`EditContext` uses schema version `sepalith.edit-context.v1`. Its required fields
are `path`, `prefix`, `region_old`, `suffix` and `cursor_idx`. The three text
regions are arrays of strings. Their whitespace and empty entries are preserved.
`event_diff` holds the recorded edit history. A missing history becomes empty
only through the explicit `from_legacy` migration method.

Use `EditContext.from_dict` for versioned records and `from_legacy` for existing
eval examples. Unknown top-level fields survive in `extra`; `metadata` holds
named JSON metadata. Both survive serialization. Unsupported schema versions,
invalid field types and non-finite metadata numbers fail validation.

`cursor_idx` selects a line in `region_old`. The legacy convention places the
cursor at that line's end and omits it for an out-of-range index. Optional
`cursor_column` specifies a Unicode code-point offset within that line. This is
an explicit geometry extension: legacy examples omit it. It changes cursor
placement, but uses the same prompt markers. Callers converting editor UTF-16
positions must convert offsets before constructing the record.

Optional `EvidenceRecord` entries capture observed file content, diagnostics,
symbols, retrieval results or other sources. Each has a source kind and content.
Workspace revision, content identity, path, source range, symbol and confidence
are optional. Source ranges use half-open Unicode code-point offsets in the
identified source content, not offsets in the rendered prompt. Confidence, when
supplied, must lie in [0, 1]; its interpretation belongs in source metadata.

The producer should record the observed workspace revision and a content identity
for each source used in a reproducible ablation. Include unsaved-buffer identity
when the evidence differs from disk. The contract preserves supplied identities;
it does not read the workspace, compute hashes or invent missing provenance.
Null identities indicate that the snapshot cannot establish source freshness.
Unknown evidence fields survive round trips, so new sources need no prompt change.

`serialize_snapshot(context)` emits sorted-key, compact JSON. Evidence order and
all text remain significant. Retain this string with the experiment manifest to
freeze the selected input. Dataclasses are frozen at the field level; nested
metadata mappings are not deeply immutable. Serialization captures their values
at the time of the call.

## Renderer behavior

`render_context(context, renderer="zeta2-v1")` preserves the baseline behavior
of `experiments/eval/run_eval.py::render_zeta2`: suffix first, edit history next,
then the file prefix and current region, followed by the middle marker. History
fence removal and leading blank-line handling match that renderer. Literal
regression fixtures cover a cursor on an interior line, an empty edit, history,
Unicode text, suffixes and blank lines.

Evidence, metadata and unknown fields do not enter this prompt. Their presence
cannot silently change the legacy model input. This renderer preserves existing
marker text verbatim; it does not escape marker-like source content or guarantee
how a particular model tokenizer treats special markers.

A token budget requires a supplied tokenizer object with `encode(text)` returning
token IDs. The renderer counts those IDs and rejects an oversized prompt. It
does not truncate or use character counts as token estimates. Supply an adapter
configured with the serving tokenizer's special-token behavior. The budget covers
the rendered input only; the caller reserves output tokens and serving overhead.

## Boundaries for future experiments

The input contract records the edit context and available evidence. Context
selection decides which observed evidence belongs in that context. Rendering
decides which fields become model input. Output parsing and edit application
remain separate contracts.

The requested workspace compressor is a learned model that emits latent vectors
or KV state. Textual summaries are not its intended output. Selection can supply
observed source records to this model; the encoded payload then needs a compatible
decoder injection path, separate from text rendering. The text contract continues
to capture the baseline input and source evidence for ablations.

## Learned latent memory manifest

`sepalith.memory` defines `LatentMemoryManifest` with schema version
`sepalith.latent-memory.v1`. It describes an external local payload without loading
tensors or putting vectors into JSON. `ConsumerCompatibility` records these exact
consumer expectations:

- Encoder, decoder and tokenizer revision identities.
- Representation: `input_embeddings` (the default experimental candidate) or
  `kv_cache`. This default is not a production choice.
- Positive latent count and width, plus `float16`, `bfloat16` or `float32` dtype.
- A positional/layout contract identity and source scope identity.

Use immutable checkpoint/revision identifiers, including the trained adapter
identity when it affects consumption. A layout contract must define how positions,
attention masks and latent order reach the decoder. For KV state it must also
define layers, heads, key/value axes and sequence offsets. The manifest does not
infer these details from count and width. Matching dimensions alone does not make
vectors from another model meaningful memory.

The manifest contains the payload's relative local path, its lowercase SHA256
digest, and source-content SHA256 hashes keyed by normalized workspace-relative
POSIX paths. Source hashes must describe the actual consumed content, including
unsaved editor buffers when relevant. The producer must capture the complete set
of files for the chosen scope. Scope is an identity checked for equality,
not a filesystem enumeration rule. The producer and consumer must use the same
inclusion rules: supply the entire workspace inventory for workspace memory, or
the consistently filtered module inventory for module memory. Dependencies outside
that scope require a separate check or a broader scope.

`validate_for_consumer(expected, current_hashes=...)` requires exact equality for
all compatibility fields and rejects added, changed or missing files by comparing
the complete scoped inventories and their content hashes.
`stale_sources(current_hashes)` reports those paths in sorted order. Neither call
reads the filesystem. Supply the full scoped inventory of current buffer or file hashes;
passing the recorded hashes back without observing the workspace proves nothing
about freshness.

`verify_payload(artifact_directory)` explicitly reads the payload in bounded
chunks and checks its SHA256. The path must resolve inside that directory. Verify
compatibility, freshness and payload integrity before consumption. A future
consumer must also validate the tensor container's actual dimensions and ensure
it loads the same verified bytes; this manifest API does not prevent a file from
changing between verification and a later load. A digest detects changed bytes,
not whether the encoder produced useful latents.

`to_json()` produces deterministic manifest JSON. Unlike edit-context metadata,
unknown latent-manifest and compatibility fields are rejected: new consumption
semantics require an explicit schema revision, not silent omission. The manifest
stores no vectors and allocates no tensors. No learned compressor, decoder
injection, KV loader or llama.cpp integration is implemented here. Future latent
experiments should retain this manifest and the frozen `zeta2-v1` text baseline
against the same observed workspace state.
