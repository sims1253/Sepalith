# Saved-result reanalysis — 2026-09-06

The saved results contain useful, narrow capabilities that the summaries missed.
They do not justify changing the production model. B8b can preserve documentation
that B4 deletes and can supply complementary format outputs. LOC1 fusion improves
some ranks, but its gold construction and child-state candidates prevent a product
localization conclusion. Historical scores and reports remain unchanged.

This is a retrospective analysis, with no training, paid judging or new model
inference. The author inspected the saved evidence and adjudicated documentation;
there was no independent second judge. Numeric outputs, labels and source hashes
are tracked here. Raw prompts, completions, corpora and embeddings remain private.

**Ranked recommendations**

| Rank | Recommendation | Scope and reason |
|---|---|---|
| 1 | Adopt a correction | Report doc_sync preservation, coverage, placement and meaning separately. Its current exact metric tests canonical placement as well as wording. Keep exact scores as historical reconstruction scores. |
| 2 | Adopt a correction | Rename LOC1's historical “recall” to **any-gold hit@k**. Add multi-gold recall and all-gold hit. Qualify its child-state and parser-dependent gold; retract any inference that matching inputs removes absolute product-usefulness bias. |
| 3 | Run a discriminating experiment | For localization, first build a parser-checked, parent-state set with natural pre-edit queries and complete candidate pools. Compare lexical, released neural and a fixed hybrid before paying for another training recipe. This analysis does not authorize that experiment. |
| 4 | Run a discriminating experiment | Test B8b's token-preservation selector on new packages, at an explicit two-output cost and with an R syntax check. Do not tune it further on this battery. |
| 5 | Retain the original decision | Keep B4 over B8b as the default. Keep TU2's original raw-route decision on its measured benchmark: semantic doc_sync rescoring supplies no rescue for its four arms. |
| 6 | Insufficient evidence | General documentation correctness, an SFT mechanism that erases midtraining, parent-state retrieval performance, and production hybrid latency remain unestablished. |

**Source boundary and reproducibility**

Work was isolated in `/home/m0hawk/Documents/Sepalith-reanalysis`, branch
`reanalysis/saved-capabilities-20260906`, from shared HEAD
`83b43b278e72b396cb8f7ef8bf4e31935706e2fc`. The shared checkout was read only.
Repository guidance, PROJECT-MAP, RESEARCH-AUDIT, SHARED-EVALUATION, SYSTEMS and
current comms were read first. No AGENTS.md was found in the repository or its
ancestor directories. Concurrent shared edits and ignored evidence were preserved.

[Input inventory](input-inventory.json) records absolute paths, SHA256, byte size,
mtime, source HEAD, HEAD blob where tracked, and dirty/untracked/ignored status.
Each entry's hash, rather than the worktree HEAD alone, identifies the input used.
[Cache inventory](cache-inventory.json) identifies all 240 embedding files used
(139,561,920 bytes). Snapshots are content-addressed private copies. Supplemental
sources were frozen when their relevance became clear; per-entry versions govern.
[Parent receipts](loc1-provenance.json) record immutable child/parent commits and
hashes of inspected messages and parent source blobs. They do not publish those
blobs. The 80 inventoried non-cache files include the three scenario files needed
to recover the worked examples. [Missing evidence](missing-evidence.json) lists field-level omissions and
scope limits; a fresh Git checkout cannot reproduce the data-dependent results.

Only small, serial, nice-19 offline analyses ran. BLAS used one thread. Inputs were
the saved evaluation sets, about 46 MB of candidate text, and cached embeddings;
no mirror-wide mining, embedding, model loading or benchmark ran. The active quiet
benchmark's processes, affinity, queue and resource claims were not changed.

With Python 3.10+ and NumPy (validated with Python 3.10.12, NumPy 2.2.6), from this
worktree, recover matching private evidence and replay:

```sh
python3 scripts/reanalysis/restore.py --inventory docs/reanalysis/input-inventory.json --snapshot docs/reanalysis/private/inputs
python3 scripts/reanalysis/restore.py --inventory docs/reanalysis/cache-inventory.json --snapshot docs/reanalysis/private/cache
nice -n 19 python3 scripts/reanalysis/run_analysis.py --inventory docs/reanalysis/input-inventory.json --snapshot docs/reanalysis/private/inputs --cache-inventory docs/reanalysis/cache-inventory.json --cache-snapshot docs/reanalysis/private/cache --annotations docs/reanalysis/doc-sync-adjudications.json --out docs/reanalysis/private/replay --private-out docs/reanalysis/private/review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s scripts/reanalysis -p 'test_*.py' -v
```

`restore.py` refuses changed source bytes; it does not silently use newer evidence.
To use archived snapshots, place files under their recorded SHA256 names instead.
The private directory is Git-ignored. Compare the four replay result JSON files to
the tracked copies. Replay recalculates structural dimensions and aggregates the
recorded semantic judgments; it does not pretend to reproduce an independent
semantic judge. To repeat adjudication, inspect the anonymous `packets.json` first,
record separate labels against the rubric, and only then open `unblind.json`.

Parent-object checks are a separate bounded read-only operation:

```sh
nice -n 19 python3 scripts/reanalysis/loc1_provenance.py --inventory docs/reanalysis/input-inventory.json --snapshot docs/reanalysis/private/inputs --mirror /mnt/h/sepalith/git --out docs/reanalysis/private/loc1-provenance.json --private-out docs/reanalysis/private/parent-sources.json
```

**1. doc_sync**

All 15 held-out prompts and targets were reconstructed from the materialized
`sft_v3/eval.jsonl` and the saved scenario records. Prompt SHA1 IDs and target bytes
match. All are the original canonical construction, not missing_param or
version_pair: 14 add `verbose = FALSE`; one adds `call = caller_env()`. They cover
three packages. A function signature establishes an argument and default, but
these prompts do not show the new argument's implementation. Even the canonical
progress-reporting target is not proved by the prompt.

The [rubric](DOC-SYNC-RUBRIC.md) was written before inspecting outputs. Review
packets omitted model labels and historical scores, deduplicated identical
case/raw pairs and ordered by content hash. The author already knew aggregate
claims, so this was partial blinding. All 286 packets were inspected, including
truncated or absent-output packets whose final scoring is unavailable. There are
579 saved arm/case records across 41 files: 38 complete 15-case files and three
3-case packaging probes. Of these, 384 responses can be reconstructed completely;
180 have no raw field and 15 hit their source writer's truncation cap. These 195
records have null retrospective scores, not failures. Short native-Zeta responses
were replayed with their native span parser, not the merge-marker parser. Complete
responses pass saved parsed-prefix and line-count parity checks. Parser and
truncation corrections made during review did not change the recovered successes.

All original validator failures were retained privately and inspected alongside
the full regions. The canonical validator requires exactly one inserted canonical
line immediately before @return/@export. Its broad assertion often hides whether
the actual cause is deletion, wording, placement or another edit. Repeated
canonical sentences in the outputs show why calling every failure a wording
problem is wrong.

| Arm | Fully reconstructable / available | Historical exact | Retrospective target-usable |
|---|---:|---:|---:|
| B4 | 15/15 | 0/15 | 0/15 |
| B8b | 15/15 | 0/15 | 3/15 |
| B2 Qwen 0.8B | 15/15 | 0/15 | 3/15 |
| B5 Granite | 15/15 | 0/15 | 1/15 |
| B12 Spark | 15/15 | 0/15 | 1/15 |
| PFT1 | 15/15 | 0/15 | 1/15 |
| Gemma zero-shot | 12/15 | 0/15 | 2/12 reconstructable |
| GLM three-shot | 13/15 | 0/15 | 2/13 reconstructable |
| GLM zero-shot | 13/15 | 0/15 | 0/13 reconstructable |
| TU2 a624 / b624 / b113 / c113 | 15/15 each | 0/15 each | 0/15 each |
| Earlier SFT v3 | 0/15 raw | **6/15 saved** | unavailable |

These are different training and prompting regimes, not a controlled ranking of
41 models. GLM/Gemma drivers append task instructions, and GLM three-shot appends
worked examples; native Zeta uses a different renderer. The private packets now include
reconstructed request content and the mapping records its SHA256. The GLM
three-shot examples were recovered from their three pinned training-side rows;
the driver's holdout-exclusion checks pass. This additional reconstruction during
review supplied format examples, not missing function behavior, and did not
change the semantic labels. Historical full-request logs and backend chat-template
bytes remain unavailable. These are different full requests across arms. The principal
B4/B8b/TU2 comparisons use the same saved base prompts. All arms and every dimension
are in [doc-sync-results.json](doc-sync-results.json); semantic decisions and
review explanations are in [doc-sync-adjudications.json](doc-sync-adjudications.json).

The 13 target-usable records cover only seven distinct cases. Eight are canonical
wording with a different acceptable parameter position: B8b's three are in this
category. The other five involve paraphrase or additional harmless documentation.
None establishes prompt-supported usable documentation under the strict factual
rubric: unsupported implementation behavior remains unknown, not false. This
zero is an evidence-sufficiency result, not proof that the descriptions are wrong.

B4 covers the introduced parameter and uses roxygen formatting on all 15 cases,
but preserves the existing documentation on none. B8b preserves three complete
blocks. On these three cases the new line occurs first among parameters rather
than immediately before the return/export anchor. That is useful block-preserving
behavior hidden by the original metric. It is not a statistically established
general capability difference: the paired 3 wins/0 losses have exact McNemar
p=0.25, and two cases share a package. TU2's outputs mostly copy the old block or
make unrelated changes; relaxing wording does not create a doc_sync rescue.

Counterexamples constrain the correction. B4 case `28cf376b38b6` emits the exact
canonical progress line but drops the pre-existing call parameter: it still fails
this rubric. Packet `25c88a6275270093` claims a TRUE default against visible FALSE.
Other packets invent result-printing, failure-only messages or error-reporting
behavior; they are not faithful target paraphrases. The sole call-argument case
has no target-usable output in the fully reconstructed evidence. Earlier SFT v3
records six exact/valid passes on these IDs; the broad historical “all arms zero”
claim must therefore be limited to the named later arms. Its raw completions are
missing, so those six passes are preserved reports, not independently rescored.

A per-case readout follows. Counts combine saved arms only to locate examples;
shared outputs and repeated model variants are not independent replications.

| Case ID | Reconstructable outputs | Target-usable outputs | B4 / B8b target-usable |
|---|---:|---:|---|
| `1117e61a40b3` | 22 | 0 | 0 / 0 |
| `23fa463cca45` | 26 | 2 | 0 / 0 |
| `28cf376b38b6` | 25 | 0 | 0 / 0 |
| `2a7fb0a77721` | 26 | 3 | 0 / 1 |
| `353b78ce1c69` | 25 | 0 | 0 / 0 |
| `4feff1af8e2b` | 29 | 0 | 0 / 0 |
| `5c026e31b471` | 25 | 0 | 0 / 0 |
| `6e9b5f4199fe` | 28 | 0 | 0 / 0 |
| `7a6531cb606d` | 23 | 1 | 0 / 1 |
| `99741b3f90dc` | 26 | 4 | 0 / 1 |
| `aacd773af914` | 26 | 0 | 0 / 0 |
| `c9264fc09531` | 26 | 0 | 0 / 0 |
| `ce96d064ade4` | 25 | 1 | 0 / 0 |
| `ee4c6fe3fc42` | 24 | 1 | 0 / 0 |
| `fad0fbd76f1a` | 28 | 1 | 0 / 0 |

**2. B8b paired capabilities and selection**

All 255 scenario IDs join one-to-one with matching family, package and path.
The reconstructed full raw predictions reproduce all saved exact outcomes against
the materialized targets. B8b has 191/255 exact versus B4's 195/255: nine B8b-only
wins and 13 losses, p=0.5235. Validity is 212 versus 217, with 7 wins/12 losses,
p=0.3593. No-op proposal decisions are identical on all 258 IDs, including
120/204 proposals on the scored no-op subset. No-op IDs and recorded metadata
match, but full prompts are not stored in those result files.

Format exactness is **38/67 versus 35/67**, +4.48 percentage points, with 9 wins
and 6 losses (p=0.6072). The shared row bootstrap's 95% interval is
−7.46 to +16.42 points; resampling the four packages gives −5.56 to +12.24 points.
Four package clusters are too few for a dependable population interval. Both
arms have 48/67 format-valid outputs, but on different examples: 7 wins/7 losses.
Failure to establish a winner does not establish equivalence.

All 23 cases discordant on exactness or validity were inspected. The 15 exact
format disagreements are below. The remaining cases include a validity-only
format disagreement (`f0e6ebd982e4`: alternate formatting versus a no-op), four
rename losses, one pipe loss and two na.rm losses. The rename case
`ad5f8937afb1` and na.rm case `2cfb9998b225` pass their historical validators while
changing the wrong symbol/call. Validator-pass is therefore not answer correctness.

| Format case | B8b outcome | Observed distinction | Fixed selector |
|---|---|---|---|
| `1e1241a8d502` | Loss | Adds a body beyond the editable region | Keeps B4 |
| `1fcb1f5a4f52` | Loss | Keeps old indentation; no edit | Keeps B4 |
| `212b4376542c` | Loss | Replaces a function definition with a call | Keeps B4 |
| `2481eb48c1eb` | Win | B4 invents a stop-body; B8b respects region boundary | Missed: target adds a brace |
| `27dfb601907c` | Win | B4 changes plotting text/arguments | Selects B8b (evaluation) |
| `439b0f4fc3e6` | Win | B8b includes canonical leading closing brace | Missed: brace-changing event |
| `4e13cea51b14` | Loss | Adds validation code beyond the region | Keeps B4 |
| `52ebdea332e2` | Win | B4 invents a stop-body | Missed: target adds a brace |
| `6723e09c61bc` | Loss | Changes loop sequence expression | Keeps B4 |
| `8feb5a0869d1` | Win | B4 repeats statements from the prefix | Selects B8b (evaluation) |
| `9262eb2982ef` | Loss | B8b repeats a prefix assignment | Keeps B4 |
| `abbfbb2742d1` | Win | B8b includes canonical leading closing brace | Missed: brace-changing event |
| `d8198911ff15` | Win | B4 invents a stop-body | Missed: target adds a brace |
| `e177d068cc2f` | Win | B4 drops a trailing comma | Selects B8b (development) |
| `f5e962b9dc29` | Win | B4 invents a body and closing braces | Selects B8b (evaluation) |

These disagreements suggest complementary preservation and boundary behavior,
not a universal B8b format advantage. PASWR supplies eleven of the 15 discordant
format cases; only four packages comprise the entire format set. Canonical brace
choices also complicate “formatting”: token-preserving lexical checks deliberately
cannot accept every historical formatting target.

An **answer-aware oracle** could choose the correct arm and get 44/67 format exact
(65.67%) and 204/255 overall (80%). Those are upper bounds using answers, not a
selector result. The actual selector in [b8b.py](../../scripts/reanalysis/b8b.py)
uses only the prompt and two predictions: for a whitespace-only visible event,
keep B4 unless it changes the current region's lexical tokens and B8b preserves
them. It never reads target text, model scores or family labels. It is a lexical
screen, not an R parser; newlines and other layout can affect syntax despite
matching lexical tokens. Its strings/comments and identifier boundaries are
covered by focused fixtures. A tokenization correction during code review did
not change which saved rows it selected.

The [policy and split](OFFLINE-POLICIES.md) were fixed before inspecting the paired
case outputs. SHA256(package) splits the battery into 193 development rows from
seven packages and 62 evaluation rows from six different packages. Format has
24 development rows from two packages and 43 evaluation rows from two others.
No fitting or threshold sweep occurred. Historical aggregate exposure and repeated
use of this battery mean even the evaluation partition is retrospective, not a
fresh confirmatory test.

| Partition | B4 exact | B8b exact | Selector exact | Oracle exact |
|---|---:|---:|---:|---:|
| Development, all 193 | 171 | 163 | 172 | 172 |
| Evaluation, all 62 | 24 | 28 | 27 | 32 |
| Development, format 24 | 15 | 14 | 16 | 16 |
| Evaluation, format 43 | 20 | 24 | 23 | 28 |
| All format 67 (descriptive) | 35 | 38 | 39 | 44 |

The evaluation selector's 3 wins/0 losses have p=0.25. Its format gain is +6.98
points, row-bootstrap interval 0 to +16.28 points. It captures only three of the
eight evaluation B8b-only exact wins. Across all rows it improves B4 from 195 to
199, but the evaluation subset alone scores below always using B8b (27 versus
28). Two model generations, storage and latency are extra costs, not measured
by these offline counts. Retain B4 and treat selection as an experiment candidate.
Neither matching loss curves nor similar aggregates demonstrate that SFT erases
midtraining weights. No weight-identity analysis was performed here.

See [b8b-results.json](b8b-results.json) for all paired outcomes and
[uncertainty.json](uncertainty.json) for row and package-level uncertainty.

**3. LOC1 retrieval reconstruction**

All 60 query IDs join the frozen set and corpus. Cached query/document embeddings
were multiplied with one BLAS thread; no encoder ran. Their corpus hashes and
shapes match. Recomputed BM25 and both neural rankings reproduce all saved hit@5,
hit@10 and reciprocal ranks. The exact historical tokenizer was retained: despite
its comment, camel-case splitting happens after lowercasing and is therefore
ineffective. Fixing it would define a different lexical baseline.

The 60 cases span 33 repositories and 14,280 candidate chunks, with 26 multi-gold
queries and 128 unique gold path/name units. Seven candidate pools have the
post-build 600-chunk cap. The full 144-row set's manifest says 16 pools were capped,
with gold always retained. This gold-dependent capping is matched across arms but
is not an inference-time candidate policy. The samples are therefore neither full
workspace retrieval nor independent rows. Candidate pools contain 98 duplicate
path/name occurrences across 21 rows. Ranking preserves unique **candidate
indices** and their original tie order; recall counts unique gold path/name units.
No selected gold in these 60 rows has duplicate matching candidate chunks, so the
identity fix does not alter their current recall denominators.

Queries are cleaned commit messages: all 60 reproduce from the immutable commit
subject/body using the frozen cleaner. Ten contain a gold function name as a
case-insensitive substring; BM25 hits all ten at k=10. This crude diagnostic is
not a complete leakage detector: other queries name files or partial identifiers.
The “no full target name” subset still uses post-edit commit messages.

All 60 parent commits resolve locally, and 54 source files can be read at the
parent revision. The frozen parser finds only 78 of the 128 gold names in those
parent files. Eighteen queries lack at least one parent gold name; ten have no
parent gold name found. This is a name-availability check, not a proof of semantic
absence: renames, moved files and parser errors can also explain a missing name.
It does establish that treating the child gold as an immediately available
parent-workspace target is unsafe. No saved parent-state ranking or embedding
cache was found in this evidence set; none was synthesized.

Gold means functions whose **child** line spans overlap changed diff-hunk lines
in a single source file, with at least two changed lines across the gold set.
A test co-edit is a weak provenance signal, not proof that every function is a
behavioral target. This set cannot evaluate genuine cross-file gold coverage.
Manual inspection found two concrete parser counterexamples:

- `1a7fab8fc4ae`: the one-line `idx_of` function's saved chunk includes later
  assignments and another function. The frozen parser keeps reading until it
  sees a positive bracket depth, so one-line bodies can absorb subsequent code.
  This is a neural-only hit, and its gold name is absent from the parent source.
- `49944459fc6a`: the purported top-level gold function `error` is a callback
  inside error handling; the chunk includes later unrelated statements and an
  entrypoint. A lexical hit against it is not proof of correct function localization.

These cases are kept in all numeric results. Removing inconvenient rows after
seeing method outcomes would give a misleading cleaned score.

| Saved arm / fixed hybrid | Any-gold hit@10 | Mean multi-gold recall@10 | All-gold hit@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 53/60 = 88.33% | 82.92% | 47/60 = 78.33% | .6947 |
| Muninn | 51/60 = 85.00% | 79.60% | 43/60 = 71.67% | .7461 |
| Muninn-small | 50/60 = 83.33% | 74.46% | 37/60 = 61.67% | .7213 |
| Fixed RRF, small + BM25 (primary) | 53/60 = 88.33% | 81.81% | 44/60 = 73.33% | .7644 |
| Fixed RRF, large + BM25 (sensitivity) | 55/60 = 91.67% | 86.18% | 47/60 = 78.33% | .7670 |

Mean multi-gold recall is the mean fraction of each query's gold units retrieved,
including single-gold queries; all-gold hit requires complete coverage. On just
the 26 multi-gold queries, small-model fusion has the same 25/26 any-gold hits as
BM25 but only 16/26 all-gold hits versus 19/26. A hybrid can improve the first hit
while losing necessary functions.

The eight large-neural/lexical hit@10 disagreements reconstruct as follows:

| Query ID | Winner at k=10 | Counterexample or qualification | Small / large fusion hit |
|---|---|---|---|
| `01198c36250e` | Both neural arms | Partial function identifier in query; not pure semantic retrieval | yes / yes |
| `1a7fab8fc4ae` | Both neural arms | Child-only gold name and overlong parser chunk | yes / yes |
| `548e816f16d7` | Both neural arms | Broad correlation intent; multiple gold functions | yes / yes |
| `49944459fc6a` | BM25 | Gold callback/parser error described above | yes / yes |
| `67bc36e2951c` | BM25 | Generic release query; every BM25 score is zero, hit follows candidate order | no / no |
| `cbc23478b6d0` | BM25 | Error-fix query; fusion also misses the needed helper | no / no |
| `f34fa895a97c` | BM25 and small neural | Removed-code/test co-edit query; large neural misses | yes / yes |
| `fe5cb5f55d01` | BM25 | Follow-on line-wrapping query, file name cue | yes / yes |

Large neural versus lexical gives 3 wins/5 losses (p=.7266); small neural gives
3/6 (p=.5078). Their different hits support investigating combinations, not
assuming an oracle can recognize the right one. Even the all-zero lexical “win”
must be retained when reconstructing the historical score.

The hybrid uses equal-weight reciprocal rank fusion, constant 60, over the same
full saved candidate rankings, with original-index tie-breaking. It returns only
k candidate chunks, not the union of two k-sized lists. This matches output count
and candidate pool, **not** encoder compute, latency, or downstream context tokens.
The cache key hashes corpus text but does not hash query text, model revision or
encoding settings. Current query provenance and exact saved-metric parity support
replay; complete original encoder provenance remains missing.

The fixed repo-disjoint split has 21 development queries from ten repositories
and 39 evaluation queries from 23 others. Primary small fusion loses one hit on
development (18/21 versus BM25 19/21). On evaluation, both fixed hybrids hit 35/39
versus BM25 34/39 and their respective neural baselines 32/39. Against BM25 each
has 2 wins/1 loss, p=1.0. The evaluation repo-bootstrap hit difference interval is
−5.88 to +11.90 points. Small fusion's evaluation mean recall is 82.26% versus
81.41%, but all-gold hit is lower: 28/39 versus 30/39. Its recall-difference
interval is −8.55 to +10.95 points. The large-fusion sensitivity has 84.51% recall
and 30/39 all-gold hits; it was not selected as a winner after evaluation.

The retrospective full-set large hybrid's 55/60 is a useful lead, not a reason to
adopt: it has 4 wins/2 losses versus BM25, p=.6875, and inherits the same gold and
query defects. The released small model may be practical, but neither this replay
nor the historical contended timing establishes a product latency budget. See
[loc1-results.json](loc1-results.json) and [uncertainty.json](uncertainty.json).

**Validation and limits**

The analysis tools fail on duplicate row IDs, unmatched comparisons, changed
snapshot hashes, missing caches, invalid ranking pools, nonfinite scores and
saved-metric disagreement. Focused tests use invented fixtures and cover those
boundaries, distinct hit/recall denominators, fusion output budgets, semantic
unknowns, preservation failures and selector information limits. No raw datasets
or proprietary contexts are included in the commit. JSON outputs contain derived
scores, case IDs and provenance metadata only.

Bootstrap intervals are descriptive. The row-level intervals can be optimistic
with few discordances; package/repository resampling exposes some clustering but
cannot repair a small or biased benchmark. None of the several exploratory
comparisons was a new preregistered confirmatory test. The preserved shared label
TIE-UNDERPOWERED is an old decision label, not evidence of equivalence or a power
calculation. The historical model choices remain reasonable under these limits.

Validated: **21 focused tests pass**, Ruff lint and format checks pass, and the
four numeric result files match an independent replay byte for byte. Review
covered privacy, null handling for missing evidence, native parsing, candidate
identity, selector inputs and grouped uncertainty. The execution hashes are in
[validation.json](validation.json).
