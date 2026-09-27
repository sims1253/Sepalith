# DAT-08 independent assembly review

Scope: metadata and source-code contract review only. No sealed source bytes, final rows, model weights, CUDA, provider, or SSH access was used.

The frozen preparation tests passed 8/8 with:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B docs/campaign/work/lead/r2-final-assembly-preparation-v1/test_assembly.py

The locked metadata summary is 687 sources, 500 selected R sources, 187 groups, and 2,500 planned probes. The five available family ceilings sum to 640; format propagation and roxygen drafting remain explicit gaps. Actual supported yield is correctly recorded as unknown until the post-freeze guarded read.

## Findings

1. **Admission blocker: complete assembly closure is still missing.** assemble_inputs.py:106-115 requires a root-frozen assembly_source_closure_sha256 graph and requires its resolved component path at line 113. The preparation source-manifest.json inventories only the five preparation files; it is not the transitive runtime/builder/guard/entrypoint closure. Root must freeze the new wrapper plus the complete imported closure and verify origins before execution. The existing construct_final.py then needs that same graph and its five generated input files (construct_final.py:25-48).

2. **Admission blocker: coverage and strata are intentionally unresolved.** assemble_inputs.py:13 excludes format and roxygen, and :146-154 emits a shortfall flag rather than asserting the 600-row/30-group target. The maximum currently reachable from declared families is 640, while actual supported/selected counts are unknown. The root run must retain the complete census and treat a shortfall as a measured result; no quality or coverage admission follows from these synthetic tests.

3. **Root preflight gap: selection metadata is not fully structurally bound.** metadata_plan checks registry split, group split/flags, package alias, R suffix, duplicate paths, and source lookup (assemble_inputs.py:30-66), but does not assert planned_requests == len(parents/files) * 5, nonempty/unique parent file lists, source-record parent/version membership, SHA format, or the locked metadata_plan_canonical_sha256 recorded in metadata-check.json. The admission guard later validates request hashes, paths, and IDs (admission_guard_v3.py:406-470), but it cannot establish those selection-to-source metadata relationships. Root should add these checks or record them as explicit post-freeze checks before accepting coverage.

4. **Portability blocker for a copied/provider payload.** The component hard-codes the current worktree in assemble_inputs.py:5-6,109-113, and the entry/runtime modules use the same absolute closure convention. A copied payload will fail the resolved-path membership test unless root stages every path and freezes a graph for that exact staging. The new root wrapper may solve this by running in the reviewed worktree, but its graph/path/origin check remains mandatory.

5. **Operational failure handling needs a root policy.** The component creates the output directory before the guarded read (assemble_inputs.py:123-135) and writes multiple immutable files sequentially (:157-164). A builder/parser failure after directory creation can leave a partial directory and phase-one guard metadata. Root should retain that as an explicitly failed attempt and ensure the orchestrator does not interpret the directory as an assembled result; the assembly-complete marker must only be emitted after all post-write checks.

6. **Phase-one versus selected authorization must be preserved.** The component authorizes all planned requests at :126-139, then emits authorization for selected requests at :157-159. This is compatible with AdmissionGuardV3, which requires exact IDs for the request sequence (admission_guard_v3.py:287-334), but the original component output does not carry the phase-one read.authorization_id. Root should retain both the all-attempt phase-one guard receipt and the selected authorization, and bind the selected IDs as a subset. The current root wrapper's source_read_authorization_id field addresses this requirement.

## Checks that passed

- The 8 synthetic tests exercise all five probe families, exact absence handling, parser/source-hash failure propagation, deterministic caps/diversity, final-group/alias checks, and refusal before checked_json.
- The evaluator adapter accepts a nonempty final split_identity derived from a row ID; it rejects only literal train/dev (evaluator_adapter.py:171-181), so the assembler's dat08-... identity is contract-compatible.
- Integration v3 requires an exact case-spec key set and relative paths (production_integration_v3.py:115-165); the selected case_specs mapping produced by the assembler has one entry per selected request and relative R/... paths.
- The unchanged freeze gate enforces status=frozen, weight/harness/final-access flags, exact hashes, and the 2026-09-14T10:00Z embargo (final_row_gate.py:205-221). The assembler has no clock override.

## Root action order

Freeze the complete relocated/actual closure and origin map; add or record the structural metadata checks above; run the post-embargo guarded probe; preserve all-attempt and selected authorization IDs; then pass the generated inputs unchanged to construct_final.py. Review census, shortfalls, duplicate semantics, protocol/typing/history representation, and final row validation before any admission decision.
