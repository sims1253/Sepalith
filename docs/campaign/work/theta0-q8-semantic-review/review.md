# RUN-09 theta0 Q8 semantic DEV review

This is a diagnostic review of the completed native Q8 DEV artifact. It does
not change the DEV panel, create training labels, load a model, or read final
data.

## Inputs and method

The model artifact is
/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-theta0-q8-cuda-dev-a/quality.json
(SHA-256 3d7bbd56c8f434b4bdc3de784b217feee5da338487aca3090d46fe7566a4a29f).
It records model label theta0-step1000-Q8_0-CUDA-graph0 and provenance
22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559.
The pinned 75-row panel is
docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl
(SHA-256 7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035).

I selected the eight roxygen_drafting and six finish_block records by their
frozen IDs. All eight roxygen rows were protocol-valid; four finish rows were
protocol-valid and two were mechanical failures. The artifact denominators
are 8 roxygen rows, 6 finish rows, 4 finish protocol-valid, and 2 cap/no-EOS
failures. Overall artifact denominators remain 75 attempted, 43 edit cases,
32 strict no-op cases, 26 exact edits, 25 correct no-ops, 5 no-op false
positives, 4 protocol errors, and 4 cap hits.

For finish rows, I used the pinned production protocol
packages/sepalith/src/sepalith/campaign_protocol.py (SHA-256
5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156) to parse
each returned wire response with its panel context. I reconstructed the
declared replacement as context prefix + parsed body + context suffix, then
parsed it with tree-sitter-r from the canonical .venv (tree-sitter 0.26.0).
This removes the protocol terminal marker before R parsing and preserves the
declared geometry. It does not treat the target prose as the only acceptable
documentation.

## Roxygen judgments

Judgments use useful_valid, useful_partial, uncertain, or semantic_invalid;
every row still has edit_exact=false in the model artifact.

| ID | Package | Judgment | Evidence from visible function/context |
| --- | --- | --- | --- |
| dat07-existing-c54ed09efae0ca38cff22981 | bigsimr | useful_valid | The function initializes Julia, conditionally installs Bigsimr, imports functions, and returns the wrapper; the generated setup/parameter/return documentation captures that behavior, although briefly. |
| dat07-existing-152d62f7a57472f8abb30ca9 | corHMM | semantic_invalid | The best=TRUE branch builds and returns best.vector, while the generated return line says the result is a matrix. That is a material type error. |
| dat07-existing-2838b904f1e44b7440e6da9f | fplyr | semantic_invalid | The visible function is fdply, but the title and inherited parameters document flply; the function itself calls .Deprecated("ftply") and then flply. |
| dat07-existing-06847c759fd32bdf16f2e598 | kvh | uncertain | The visible Rcpp wrapper establishes a KVH reader and its six arguments, but cannot establish whether returned values are always strings or nested lists. The generated return claim is therefore not admitted as fully valid from this context alone. |
| dat07-existing-81ddc4c6ddd226b04dcf0910 | peRiodiCS | useful_valid | The visible b_rcs(x, knots, inclx) computes a restricted cubic spline matrix and optionally prepends x; the generated title, parameters, and matrix return are useful and materially accurate. |
| dat07-existing-5b5e844d3a8795c52149f848 | rmarchingcubes | useful_valid | The function calls marching_cubes and returns triangles, vertices, and normals; the generated description and return list match the visible behavior. |
| dat07-existing-06097c12b0328d475be857e1 | standrecon | useful_valid | The visible helper maps status and decay vectors to a conclass vector; the generated parameters and vector return describe that operation accurately. |
| dat07-existing-d38a792699b4cccded445dc0 | TCpRepDesigns | useful_partial | Parameters and the three returned components match the visible output, but the description calls the function a wrapper for TCpRep1 even though it is TCpRep1 itself. |

The useful documentation signal is therefore 4 fully useful rows plus 2
partially useful/uncertain rows; 2 rows contain a material semantic defect.
This is a review judgment, not a replacement score or training label.

## Finish judgments

| ID | Protocol/evaluation | R reconstruction and semantic finding |
| --- | --- | --- |
| e623a61b5a4c066358a477f2 | mechanical invalid | The response hit the 512-token cap without canonical EOS (invalid_generation_tokens); no body was semantically scored or repaired. |
| 4f08633513b5c525240d2540 | protocol-valid, R-invalid | The accepted body is inserted after the open .lec.init context but has no final function-closing brace. Tree-sitter reports an error. It also begins with return() and omits the visible reset/assignment/.Call initialization, so it is not a useful completion even if closure were supplied. |
| d11581e9cfa4e3971aa1466e | protocol-valid, R-invalid | The returned yes.no <- menu(...) fragment has no closing brace for the visible yes.no.menu function. Tree-sitter reports an error; the omitted function body cannot be treated as a valid edit. |
| 157517ba47dbab157f7c361a | protocol-valid, R-invalid | The repeated factor conversions end at x without closing the visible .fac function. Tree-sitter reports an error, so semantic quality is not admitted. |
| f43de3e77f2d92ed7b223464 | protocol-valid, R-invalid | The returned autostrata function has no final closing brace under the declared replacement geometry. Tree-sitter reports an error. Independently, the body uses kmeans/data[[target]] and never assigns the documented autostrata class, which are concrete semantic concerns. |
| 04834fef4fe59742f13677a9 | mechanical invalid | The response hit the 512-token cap without canonical EOS (invalid_generation_tokens) and is visibly truncated/repetitive; it is not repaired or parsed. |

All four protocol-valid finish rows reconstructed with the production parser
had a tree-sitter R parse error. The two cap failures were retained as
mechanical failures. No finish row is admitted as a valid executable edit.

## Limits

This review uses only the named Q8 quality artifact and frozen panel plus the
pinned protocol/parser packages needed to apply the stored wire contract. It
does not infer unique ideal roxygen prose, run package tests, evaluate runtime
behavior, inspect hidden source/corpus rows, or promote the Q8 model.
