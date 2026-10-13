# RUN-09 Opus16 candidate review

Decision: conditional go for a root-controlled build and G0 trace. This is a
default-off experiment, not a backend promotion. The source review found no
obvious new API, ABI, or ordinary-path buffer-sizing defect. A complete native
build, target shader compilation, output comparison, and timing remain open.
The proposed G0 instructions need the correction below.

The proposal hash is
`5a0beac694825511331675ae7eb3bbcfc37c4677faf32295572182bfaa048bb1`.
The original `ggml-vulkan.cpp` hash is
`c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f`.
The complete extracted patch passed a zero-fuzz dry run and applied only to
`private/ggml/src/ggml-vulkan/ggml-vulkan.cpp`. The canonical file still has
its original hash. `artifact-sha256.json` records all returned files.

## Findings that affect the experiment

1. **The proposed G0 medium-only rule rejects legitimate outside-gate rows.**
   The trace covers every Q8 call through `ggml_vk_mul_mat_q_f16`, whereas the
   override requires M > 64 and 64 < N <= 256. For example, the source selector
   correctly uses small when N <= 32. Some decode operations use the separate
   matvec path and produce no such trace; oversized tensors can take the
   M-split path first. Require a medium baseline only on `would_small=1` rows.
   Retain and classify other rows rather than rejecting them for being small.

2. **The split-K argument has a narrower domain than the gate.** The actual
   extracted function returns medium split-K 2 and small split-K 1 for
   M=65, N=85, K=2048, cores=8. That shape passes the candidate gate. For the
   tested target domain M in {2048,6144}, N in 65..256, K in {2048,6144},
   and cores below 96, both return 1. Root must verify the actual eligible
   shapes before claiming unchanged reduction behavior. If smaller M appears,
   either evaluate the changed reduction explicitly or revise the gate in a
   separate reviewed patch. This review preserves the proposal bytes.

3. **Trace rows are host-call evidence, not completed GPU dispatches.** The
   insertion precedes descriptor allocation, lazy compilation, and dispatch.
   One call may also produce several batch/split-K dispatches. `seq` has no
   request identity or timestamp; `compiled=0` can be a normal first-use state.
   `applied` recomputes the selector gate rather than observing GPU execution.
   Correlate rows with sequential request envelopes and successful responses.
   Do not use row count as completed work or latency. Trace output goes to
   stderr, so run all timed comparisons with its environment variable unset.
   Setting `GGML_VK_Q8_0_M2S_TRACE=0` still enables tracing.

4. **The gate is not specific to the notebook.** It excludes coopmat2 and the
   effective Q8_1 operand path, but does not require AMD, forbid coopmat1,
   constrain core count, or require M >= 2048. Keep the environment override
   confined to the reviewed target process. Its safety guard requires all four
   medium/small aligned/unaligned pointers; an optional missing aligned variant
   suppresses the experiment rather than broadening the original selector.

## Source checks

References below use the original pinned file unless marked “patched.”

| Area | Result and evidence |
| --- | --- |
| Ordinary Q8 path | `ggml_vk_get_mul_mat_mat_pipeline` selects the Q8 x F32 accumulator family at 7685–7770. The runtime disables integer dot product without accelerated packed signed support at 6441. With no MMQ family, 9203–9230 selects the ordinary path. The helper therefore receives Q8_0/F32 on the intended route. Device properties still require live G0 confirmation. |
| Availability | AMD starts with l disabled without supported coopmat, m/s enabled at 6952–6954; shared-memory checks can disable m/l at 4310–4320. The actual gate checks the flags and all four m/a_m/s/a_s pointers before selecting small. `make_input` handles a null family without dereferencing its pipelines. Original selector preconditions still apply to the context and family. |
| Alignment | Both the alignment guess and final selection use the modified selector at 8803 and 9225–9228. It has no alignment-dependent gate. Small alignment is 32 and medium is 64 at 4305–4306. Q8 K=2048/6144 remains aligned; other K can change which aligned variant is valid. |
| Buffer sizes | `padded_n`, sizes, and split-K are computed after final selection at 9234–9247. Ordinary contiguous F32 has `qy_needs_dequant=false`, so `padded_n=ne11` and Y storage does not shrink with the tile area. The pad heuristic reduces covered tile area, not allocated tensor bytes. Scratch split-K size uses the actual selected split count. |
| 64-bit wrapper | The selected pipeline passes through the existing wrapper at 9131–9143 and 9230–9232. Initialization gives linked 32/64-bit variants the same tile metadata at 4399–4416. Both macro branches of the exact wrapper passed CPU projection checks; actual Vulkan extension support and compilation remain open. Large batch-one matrices can first take the M-split path at 9994–10015, with split-K disabled. |
| Accumulation and shader | The patch keeps the chosen mmp accumulator family. Q8 medium/small use BK=32 at 4257–4258 and the same scalar shader family at 4887. Local shader inspection confirms paired K iteration and dot-product accumulation in `mul_mm.comp:283–353`. Workgroup/register specialization changes, so equal BK alone does not prove bit-identical GPU output. |
| Trace/API/ABI | Referenced fields exist: pipeline name, wg_denoms, align, atomic compiled, conditional is_64b_indexing, accumulator-family pointers, device flags, and tensor metadata. `sstream` and existing atomic usage are already present. New helpers have internal linkage; no public header, push-constant layout, shader binding, protocol, or exported signature changes. The inserted trace block and complete translation unit were not compiled in this review. |

## CPU evidence and reproduction

Run `python3 run_cpu_checks.py` in this directory. It confines subprocesses to
one CPU and writes `final-cpu-validation.json`. All 17 commands passed.

The supplied standalone test reports 17,211,308 checks and zero failures, both
normally and under UBSan. Its baseline and split-K comparisons are explicitly
handwritten mirrors. Those counts do not demonstrate GPU correctness.

The additional projection compiles exact extracted selector, alignment,
split-K, and 64-bit wrapper functions with reduced context/pipeline carriers.
For each macro branch, separate processes test unset, small, pad, invalid mode,
and trace-variable presence. Each process compares 102,400 selector cases and
runs 294,912 split-K checks. Unset/invalid mode changes 0 selections, small
changes 1,152, and pad changes 768. Missing pointer guards, coopmat2 bypass,
alignment selection, both 64-bit wrapper branches, and the smaller-M split-K
counterexample pass. These tests exercise host wiring, not Vulkan headers,
the full backend ABI, shader arithmetic, or model responses.

## Minimum root-controlled next steps

1. Recheck the two input hashes. Apply `q8m2s.diff` to a new isolated source/build
   tree that matches the accepted compiler flags and Vulkan closure. Build the
   full `llama-server` target there with bounded concurrency. Example build
   shape: `cmake --build ROOT_ISOLATED_BUILD --target llama-server -j1`.
   This review did not run that command. Record source, binary, library,
   compiler, driver, and device identities; preserve the accepted baseline.

2. G0: use the selected theta0 Q8 weights, renderer/tokenizer, integer-token
   protocol, output cap, and accepted batch/thread/context settings. Start a
   fresh candidate server with mode **unset** and trace present. Begin with one
   authorized TRAIN/synthetic request expected to include a 65..256-token
   prefill dispatch. Retain its exact input/output identities and stderr. Stop
   if no eligible row appears. On eligible rows require mode=0, applied=0,
   ordinary Q8/F32, qx_deq=0, quantize_y=0, coopmat/coopmat2=0, mm_lms=011,
   medium pipeline with wg=64,64,1, and the target shape/core domain. Classify
   outside-gate rows separately. A successful response and clean device status
   are required in addition to the trace.

3. Untimed candidate: restart with mode=small and tracing present. Replay the
   same request and then the bounded existing eight-request panel. Compare
   matched eligible rows: small wg=32,32,1, expected alignment, unchanged
   accumulator/64-bit mode, actual split-K and sizes. Check tokens, parser,
   stop reason and response applicability. Report any output difference as a
   numerical/behavior change; identical output is still only fixture evidence.
   Stop on a compile, device, protocol, or applicability failure.

4. Only then benchmark with trace **unset**. Use separate baseline/candidate
   processes because the mode is cached. Pay and exclude first-use pipeline
   compilation for each relevant aligned/unaligned variant. Keep prompt-cache
   and request order identical. Use the existing paired panel with at least
   five ABBA rounds only within root's remaining lease/budget. Keep full-cycle
   p50/p95, failures, and thermal/load state alongside prompt timing. Require
   the lead's improvement threshold and no p95/full-cycle regression. Consider
   pad only if the measured small-mode pattern justifies that follow-up.

A native prefill gain cannot resolve the already observed 0/5 long-source
five-second results by itself. Local versus LAN deployment remains undecided;
any route decision needs its own full-cycle measurement. This experiment does
not authorize promotion or access to sealed/final evaluation data.
