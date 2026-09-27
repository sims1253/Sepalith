# Opus16 private native build preparation

Status: deferred after the user accepted a desktop GPU route over LAN.
No build or launch is requested. This directory retains a concrete optional
notebook build plan; no source patch, compiler, linker, shader generator,
model, or native executable ran during this preparation.

The lead supplied actual notebook metadata in
`../lead/opus16-build-metadata/`. The local canonical build is CUDA with Unix
Makefiles and was not used to invent Vulkan flags. The notebook is a Release
Ninja build, C++17, Vulkan enabled, CUDA/LTO/debug/validation disabled.

`build-plan.json` contains exact argv arrays derived from those recorded
commands. It compiles one private copy of the reviewed patched
`ggml-vulkan.cpp` and links one private `libggml-vulkan.so.0.20.0`. It reuses
138 existing embedded-shader objects. The 139th object is the changed host
object. Direct library inputs are the protected `libggml-base.so.0.20.0` and
`/usr/lib/libvulkan.so`. There is no CMake/Ninja build or shader regeneration.

The proposed private output root is:

```text
/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/opus16-private-native-a/
  source/ggml-vulkan.cpp
  objects/ggml-vulkan.cpp.o
  objects/ggml-vulkan.cpp.o.d
  objects/link.d
  bin/libggml-vulkan.so.0.20.0
  tmp/
```

Compiler `-c`, `-o`, `-MF`, and `-MT` point into that root. The linker `-o`
and `-Wl,--dependency-file=` do too. The latter originally targeted the
protected build's `link.d`; leaving it unchanged would violate isolation.
The private library uses literal `$ORIGIN` RPATH. Commands are argv arrays,
never shell text. Original source/header/include and shader-object paths
remain read-only inputs. `TMPDIR` is private and compiler cache is disabled.

`expected-inputs.json` pins the reviewed original/candidate source, patch,
accepted source manifest, notebook settings, and all 11 existing runtime
artifacts. The build plan pins the supplied CMake/Ninja metadata and exact
command text. It does **not** claim that notebook objects or current toolchain
files have already been fingerprinted.

The remaining target preflight is `collect_fingerprints.py build-plan.json`.
This read-only collector emits JSON to stdout. It checks the accepted source,
runtime and metadata hashes; fingerprints all 141 original explicit link
inputs, the generated shader header, Ninja's recorded header dependencies,
compiler programs, CRT objects, compiler specs, and recursively resolved
system library inputs. It invokes only Ninja metadata tools, compiler
identity/path queries, and readelf. Missing/stale dependencies or unresolved
libraries prevent a ready result. No collector was executed by this worker.

If this optional work resumes, root must first review that fingerprint result.
Immediately before **and after** each compile/link command, a guard must
recheck every recorded file's resolved path, size and SHA-256. It must reject
an existing output root, check the candidate source hash, restrict writes to
the new root, use one CPU and a bounded deadline, retain compiler/linker logs,
and stop on memory pressure or any input change. A mismatch is a failed build,
not permission to reuse an uncertain artifact. This preparation contains argv
and preflight tools, not an automatic build launcher.

For a future runtime experiment, copy the pinned unchanged executable/runtime
siblings into a separate private bundle, add the private Vulkan library and
correct SONAME symlinks, and inspect ELF dependencies with readelf. Existing
siblings can retain absolute build RUNPATHs; process-local library resolution
must be checked before any execution. A future launch should use an explicit
private library directory and clear inherited loader overrides. Do not replace
the protected library or silently change the accepted managed manifest.

Rollback is choosing the original protected runtime path with the candidate
mode and trace variables unset. Default-off preserves the selector but is not
a substitute for returning to the original binary. No original file needs
repair or restoration.

The deferred G0 criteria remain those in the candidate review: mode unset,
trace present, and medium required only on `would_small=1` rows. Confirm
ordinary Q8/F32, no dequant/MMQ/coopmat path, flags 011, actual M/N/K/core count,
and the M>=2048/cores<96 split-K domain. Smaller eligible M can change split-K
(65/85/2048/8 gives 2 to 1). Trace rows precede compilation/dispatch, omit
request identity, and do not prove completed GPU work. Trace must be unset for
timing; `TRACE=0` still enables it. A prefill gain does not close the observed
0/5 long-source five-second results.

CPU-only reproduction from this directory:

```sh
python3 make_plan.py ../lead/opus16-build-metadata /home/m0hawk/.local/share/sepalith-campaign-20260915/runs/opus16-private-native-a
PYTHONDONTWRITEBYTECODE=1 python3 test_plan.py
```

These commands only materialize/check the plan. They do not execute its argv.
