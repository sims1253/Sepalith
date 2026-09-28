# RUN-01 theta0 export wrapper review

This is a read-only launch review. No model bytes were loaded or hashed by this review, no export command was launched, and no SSH, network, CUDA, or server action was performed.

## Reviewed inputs

- docs/campaign/work/lead/export_theta0_runtime.py
- docs/campaign/work/lead/host_memory_guard_v3.py
- docs/campaign/receipts/RUN-01-theta0-export-preparation.json
- docs/campaign/work/lead/host_memory_policy.py
- llama-quantize --help
- convert_hf_to_gguf.py --help

Observed SHA-256 pins:

| input | SHA-256 |
|---|---|
| export_theta0_runtime.py | 670576f0b44bc02d3d28ea00c8a3b212f2464d3d15ac28d56575c0d9cb773d43 |
| host_memory_guard_v3.py | 9518f5d10ea92c424df10f4d49be32f5dd1bf8a42a0deae5d13ea2998f18a845 |
| RUN-01-theta0-export-preparation.json | 71ec95f2373bda9bf2f61f8aae245a3054d942df211472ccbed1b47542175a4b |

## Findings

### CPU admission is explicit at the wrapper boundary

The export wrapper requires os.environ.get("CUDA_VISIBLE_DEVICES") == "". In this review shell the variable was unset, so an unmodified invocation fails with RuntimeError: CPU environment required. Root must set an empty value explicitly, for example env CUDA_VISIBLE_DEVICES=, and should also set the bounded thread variables from the preparation receipt (OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2, MKL_NUM_THREADS=2, PYTHONDONTWRITEBYTECODE=1). The v3 guard does not itself enforce the CUDA variable.

### Guard output must be separate from the export target

host_memory_guard_v3.py creates --output before launching its child. The export wrapper requires the receipt's output_new_only directory to be absent and then creates it. Passing the same path to both tools makes the wrapper reject the run as non-fresh. Use a separate, fresh guard evidence directory and keep the receipt's target path untouched:

    command-json = [".../python3.10", ".../docs/campaign/work/lead/export_theta0_runtime.py"]
    guard --command-json <command-json> \
          --output /home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-01-theta0-export-guard \
          --seconds 900 --minimum-free-mib 20480

The guard starts the wrapper in a new process group. The wrapper's three Popen children inherit that group, so guard killpg cleanup covers them. The wrapper's 240-second per-command wait is followed by up to 10 seconds of graceful wait and 5 seconds of kill wait; the 900-second outer bound covers the three sequential commands. A guard SIGTERM can terminate the wrapper before its Python finally block writes terminal.json; in that case the guard's terminal record and surviving command logs are authoritative, and the target must remain unaccepted.

The current environment contains /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe, host_memory_policy.py, and post_load_cache.py, so the v3 guard's import/query paths are present. The guard's preflight still requires a successful Windows memory query, no new NVIDIA driver event, free memory at least 20,480 MiB, and commit below 90%.

The lead's current host observation reports Windows AvailableMBytes of 17,781 MiB, so the provisional 20,480-MiB admission floor would reject before launch even though WSL MemAvailable is higher. This is an operational gate result, not a fixed task requirement. If the lead explicitly revises startup admission to 16 GiB and keeps the runtime soft guard at 12 GiB with the existing 4 GiB hard floor, 90% commit limit, and driver-event stop rule, the revised values must be recorded in the launch receipt and command. This review does not authorize that revision.

### Hash gates are useful but post-export validation is incomplete in the wrapper

Before mutation the wrapper hashes the parent manifest, the full model.safetensors, tokenizer files, converter, and quantizer against the preparation receipt. It does not hash config.json or generation_config.json, and it does not pin its own wrapper/guard source in the launch record. The full weight hash is deliberately deferred until the live launch; this review did not read model bytes.

After the commands, the wrapper accepts each expected artifact only when it is a regular file larger than 1,000,000,000 bytes and begins with GGUF. It does not inspect GGUF tensor names/types, architecture metadata, tokenizer vocabulary, special-token/EOG metadata, or parent/config identity. The preparation receipt requires those checks, so root must run a separate pinned GGUF closure/metadata validator before accepting exported_pending_root_review. A failed or timed-out conversion leaves partial files in the fresh target and writes a failure terminal record; the wrapper does not delete or mark those files for reuse. Preserve that evidence and choose a new target for any retry.

The wrapper checks free storage on the parent path (100 GiB + 12 GiB) rather than directly on the output path. The two receipt paths are currently under /home, but root should confirm they share the intended filesystem before admission and retain the guard evidence directory separately.

### Quant command review

The converter help accepts the prepared positional model, --outfile, and --outtype f16 command. The b10453 quantizer help accepts both --output-tensor-type q8_0 and --token-embedding-type q8_0, the positional input/output/type/threads form, and Q8_0/Q6_K as valid types. The prepared Q8 and Q6 commands therefore have no apparent CLI syntax error; the trailing 2 is the documented nthreads positional argument. The Q6 command intentionally keeps output and token-embedding tensors at Q8_0 while quantizing the remaining tensors to Q6_K, matching the repository's documented mixed-override pattern. Root must verify the resulting tensor-type metadata because command syntax alone cannot prove the requested artifact semantics.

## Review result

**Partial / launch review only.** The CPU and command contracts are understandable, but the run is blocked until root:

1. supplies CUDA_VISIBLE_DEVICES= and bounded thread variables;
2. confirms the RL terminal is completed and both prior owner PIDs are absent;
3. invokes v3 with a separate guard evidence directory, --minimum-free-mib 20480, and --seconds 900;
4. confirms the target output directory is absent and storage is available on its filesystem;
5. performs post-export GGUF tensor/tokenizer/EOG/config closure validation before acceptance.

No export success, artifact creation, or model-quality result is claimed here.
