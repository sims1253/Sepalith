# RUN-06 Vulkan configure diagnosis

Status: diagnosed, repair pending root review. This note records a read-only inspection of the RUN-06 backend configure failure on `m0hawk@192.168.178.40`. No package install, build, server, model, GPU, cloud, or state operation was performed.

## Finding

The configure stopped in CMake's `FindVulkan` module because the Khronos development header `vulkan/vulkan.h` is absent:

```
Could NOT find Vulkan (missing: Vulkan_INCLUDE_DIR) (found version "")
```

The failing call is `ggml/src/ggml-vulkan/CMakeLists.txt:9 (find_package)`. The installed Vulkan loader/runtime and shader compiler do not provide the header searched by `FindVulkan.cmake` (`find_path` searches for `vulkan/vulkan.h`). The GStreamer and other wrapper headers found under `/usr/include` are unrelated and do not satisfy this check.

## Evidence

All paths below are on the remote host. Hashes are SHA-256 values recorded during inspection.

| Evidence | Observation | SHA-256 |
| --- | --- | --- |
| `~/.local/share/sepalith-campaign-20260915/runs/backend-build-a/vulkan.log` | CMake reached `ggml-vulkan`, then reported missing `Vulkan_INCLUDE_DIR`; configure exited 1 | `6e146f34e186582d3bc17684449d0aafedf598ce1ad85489617cff2144595669` |
| `~/.local/share/sepalith-campaign-20260915/runs/backend-build-a/vulkan-terminal.json` | Recorded Vulkan arm terminal result; exit 1 in 0.7183 s | `4a1b3dda7a182510e1ec354ab80ce0cd486dc37cc9fda3f9f1d733ab10e9de1a` |
| `~/.local/share/sepalith-campaign-20260915/runs/backend-build-a/launch.json` | RUN-01/RUN-06 launch context; no model was launched | `75a8d93f7894948c4e0910ed7f94b2fdecf1580f8cfec5f95c54873e9bdaf900` |
| `/var/cache/pacman/pkg/vulkan-headers-1:1.4.357.0-1-any.pkg.tar.zst` | Cached exact header package; archive contains `usr/include/vulkan/vulkan.h` and `vulkan_core.h` | `2f6c34cc829c4b63c0cf08cf147841c8ded023b746324540fb02322d9c415c07` |

The remote host has no `/usr/include/vulkan`, `/usr/local/include/vulkan`, or `/usr/include/Vulkan` directory. `pacman -Q vulkan-headers` reports the package is not installed. `pkg-config vulkan` is present and reports version `1.4.357` with `-I/usr/include -L/usr/lib -lvulkan`, but the advertised header is missing.

The runtime side is present: `/usr/lib/libvulkan.so` resolves to loader `1.4.357`, `/usr/bin/glslc` reports `2026.3` / Vulkan `1.4.357.0`, and the host has `vulkan-icd-loader 1.4.357.0-1.1`, `shaderc 2026.3-1.1`, `glslang 1:1.4.357.0-1.1`, and `spirv-tools 1:1.4.357.0-1.1`. Therefore the missing item is the development header package, not a loader, ICD, shader compiler, or CMake installation.

## Smallest repair

The exact matching package is already cached and has no dependencies. After root review, the smallest repair command is:

```sh
sudo pacman -U --needed /var/cache/pacman/pkg/vulkan-headers-1:1.4.357.0-1-any.pkg.tar.zst
```

This command was proposed only; it was **not executed** during this diagnosis. The cached package is `extra/vulkan-headers`, version `1:1.4.357.0-1`, architecture `any`, and its package signature and archive contents were inspected. It supplies the header under the exact path required by CMake, without changing the pinned source or CMake files.

Root can validate the repair with the following read-only checks, also not executed here:

```sh
test -f /usr/include/vulkan/vulkan.h
cmake --find-package -DNAME=Vulkan -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST
```

The existing RUN-06 configure invocation can then be retried with its already recorded `GGML_VULKAN=ON` settings. This receipt does not authorize or claim that reconfigure/build/smoke work.

If package installation is unavailable, a rootless diagnostic alternative is to extract the same verified archive into a user directory and pass its include root explicitly to a separately reviewed configure:

```sh
vulkan_header_root="$HOME/.local/vulkan-headers"
mkdir -p "$vulkan_header_root"
tar -xf /var/cache/pacman/pkg/vulkan-headers-1:1.4.357.0-1-any.pkg.tar.zst -C "$vulkan_header_root"
cmake -S <llama.cpp-source> -B <build-dir> -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR="$vulkan_header_root/usr/include"
```

That alternative was also not executed and is less durable than installing the package because it requires an explicit include hint on every configure.

## Scope and next gate

Inspection used one read-only SSH session with a one-thread CPU limit. No model or framework was loaded; no build, install, server, GPU, cloud, or campaign-state mutation occurred. The diagnosis is complete, but RUN-06 remains blocked at the missing-header gate. Root owns the decision to apply the cached package repair and to run any subsequent configure/device/load smoke. No serving readiness or performance claim follows from this note.
