"""Use the frozen CUDA generation backends and the campaign's exact-match scorer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import version
import errno
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import ProxyHandler, build_opener, install_opener

from admission_guard import _required_hash
from protocol import CAP, CAPACITY, MODELS, SEED, STOP_IDS, Generate, Json, PreparedCase, ProtocolError
from sepalith.campaign_protocol import TOKENIZER_CONFIG_SHA256, TOKENIZER_JSON_SHA256, VOCAB_SIZE
from sepalith.quant.gguf_dev import (
    DevTokenizer,
    completion_request,
    free_port,
    generated_ids,
    llama_server,
    post_json,
)
from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
from sepalith.training.eval.campaign_eval import (
    batched_greedy_generate,
    classify,
    generation_termination,
    load_development_model,
)

LLAMA_COMMIT = "3cb7ffb1a"
Q6_K_SHA256 = "6593785330b99076b80ae3ac679bda76c5fcff1ebd1873f46f944c1ec43a3261"
# The shipped small tier is bound by the freeze but not scored (C17 tier decision).
MIXB_E4_SHA256 = "e24fbfbc5ea47c88647445ec682196fc67af9134d4fa36e826beeda045f1c71b"
THREADS = 8  # host threads for torch and llama-server
# 2026-11-24 (user: "Switch to GPU"): one CUDA device, the evaluator's default device and the device of the
# DEV250 references. Every llama.cpp layer is offloaded, as in the C17 quant gate.
DEVICE = "cuda"
GPU_LAYERS = "999"
# Dependencies under these roots are the host's C library, compiler runtime and WSL driver shim. Everything
# else that the dynamic loader resolves for llama-server must be a pinned library.
# cuda_host_guard holds this lock for the job and hands its descriptor to the launcher.
CUDA_LOCK = Path.home() / ".local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock"
SYSTEM_LIBRARY_ROOTS = (Path("/usr/lib"), Path("/usr/lib64"), Path("/lib"), Path("/lib64"))
DISTRIBUTIONS = ("torch", "transformers", "tokenizers", "safetensors", "accelerate")


def verify_file(path: Path, digest: str) -> Path:
    expected = _required_hash(digest, "runtime_artifact")
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ProtocolError("runtime_artifact_must_be_absolute_regular_file")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != expected:
        raise ProtocolError("runtime_artifact_hash_mismatch")
    return path


def verify_tree(record: Json) -> Path:
    root = Path(record["path"])
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ProtocolError("runtime_tree_must_be_absolute_real_directory")
    files = set()
    for directory, dirs, names in os.walk(root):
        base = Path(directory)
        if any((base / name).is_symlink() for name in dirs + names):
            raise ProtocolError("symlink_in_runtime_tree")
        files.update(str((base / name).relative_to(root)) for name in names)
    if not files or files != set(record["files"]):
        raise ProtocolError("runtime_tree_file_inventory_mismatch")
    for relative, digest in record["files"].items():
        verify_file(root / relative, digest)
    return root


def resolved_libraries(ldd_output: str) -> set[Path]:
    """Real paths of the non-system shared libraries in `ldd` output; an unresolved one is an error."""
    found = set()
    for line in ldd_output.splitlines():
        if "=>" not in line:
            continue
        target = line.split("=>", 1)[1].strip()
        if target.startswith("not found"):
            raise ProtocolError("llama_server_library_not_found")
        path = Path(target.split(" (", 1)[0]).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in SYSTEM_LIBRARY_ROOTS):
            found.add(path)
    return found


def verify_server_libraries(server: Path, libraries: Json, *, replay: bool) -> None:
    """Hash every pinned library; outside a replay, the loader must resolve exactly that set."""
    if not libraries:
        raise ProtocolError("llama_server_libraries_unpinned")
    pinned = {verify_file(Path(path), digest) for path, digest in libraries.items()}
    if not replay:
        ldd = subprocess.check_output(["ldd", str(server)], stderr=subprocess.STDOUT, timeout=30, text=True)
        if resolved_libraries(ldd) != pinned:
            raise ProtocolError("llama_server_library_set_mismatch")


def verify_cuda_lease(environ: Mapping[str, str] = os.environ, lock: Path = CUDA_LOCK) -> None:
    """A generating run holds the night job's CUDA lease: the guard's inherited, locked descriptor."""
    try:
        held = os.fstat(int(environ["SEPALITH_CUDA_LOCK_FD"]))
        current = os.stat(lock)
    except (KeyError, ValueError, OSError):
        raise ProtocolError("cuda_lease_required") from None
    if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
        raise ProtocolError("cuda_lease_required")
    probe = os.open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
            raise
        return  # held elsewhere, by the guard
    finally:
        os.close(probe)
    raise ProtocolError("cuda_lease_required")


def verify_server_device(server: Path, device: Json) -> None:
    """The pinned llama-server itself sees exactly the pinned CUDA device."""
    listing = subprocess.check_output([str(server), "--list-devices"], stderr=subprocess.DEVNULL, timeout=60,
                                      text=True)
    devices = re.findall(r"^\s+(\S+): (.+) \(\d+ MiB, \d+ MiB free\)$", listing, re.MULTILINE)
    if devices != [("CUDA0", device["name"])]:
        raise ProtocolError("llama_server_cuda_device_mismatch")


def verify_cuda_device(runtime: Json) -> None:
    """The plan's CUDA build and device: exactly one visible device of the pinned name and capability."""
    import torch

    if torch.version.cuda != runtime["torch_cuda"]:
        raise ProtocolError("torch_cuda_version_mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProtocolError("exactly_one_cuda_device_required")
    properties = torch.cuda.get_device_properties(0)
    if {"name": properties.name, "capability": [properties.major, properties.minor]} != runtime["device"]:
        raise ProtocolError("cuda_device_mismatch")


def score_case(case: PreparedCase, ids: list[int], tokenizer: DevTokenizer) -> dict[str, int]:
    if (not isinstance(ids, list) or not 0 < len(ids) <= CAP
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids)
            or any(token in STOP_IDS for token in ids[:-1])):
        raise ProtocolError("invalid_generated_token_sequence")
    termination = generation_termination(ids, CAP, STOP_IDS)
    complete = termination["response_complete"]
    if not complete and not termination["cap_hit"]:
        raise ProtocolError("generation_ended_before_stop_or_cap")
    raw = tokenizer.decode(ids[:-1] if complete else ids, skip_special_tokens=False,
                           clean_up_tokenization_spaces=False)
    outcome = classify(raw, case.context, case.expected, ids)
    expected_noop = case.expected == list(case.context.region_old)
    edit_exact = not expected_noop and outcome["exact_region"]
    noop_correct = expected_noop and outcome["predicted_noop"]
    return {
        "correct": int(edit_exact or noop_correct),
        "edits": int(not expected_noop),
        "edit_exact": int(edit_exact),
        "noop": int(expected_noop),
        "noop_correct": int(noop_correct),
        "noop_false_suggestions": int(expected_noop and outcome["suggestion"]),
        "protocol_valid": int(outcome["protocol_valid"]),
        "cap_hit": int(termination["cap_hit"]),
        "response_complete": int(complete),
    }


class FrozenRuntime:
    def __init__(self, plan: Json, *, replay: bool) -> None:
        runtime = plan["runtime"]
        verify_file(Path(sys.executable).resolve(strict=True), runtime["python_sha256"])
        if runtime["distributions"] != {name: version(name) for name in DISTRIBUTIONS}:
            raise ProtocolError("python_runtime_version_mismatch")

        tokenizer = plan["tokenizer"]
        for name, digest in (("tokenizer.json", TOKENIZER_JSON_SHA256),
                             ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256)):
            if tokenizer["files"].get(name) != digest:
                raise ProtocolError("tokenizer_differs_from_campaign_pin")
        tokenizer_path = verify_tree(tokenizer)

        self.models = {}
        for label, record in plan["models"].items():
            if label == MODELS[2]:
                if record["sha256"] != Q6_K_SHA256:
                    raise ProtocolError("gguf_differs_from_c17_pin")
                self.models[label] = verify_file(Path(record["path"]), record["sha256"])
            else:
                self.models[label] = verify_tree(record)
        small = plan["release_artifacts"]["Q4_K_M-mixB-e4"]
        if small["sha256"] != MIXB_E4_SHA256 or set(plan["release_artifacts"]) != {"Q4_K_M-mixB-e4"}:
            raise ProtocolError("release_artifact_differs_from_c17_pin")
        verify_file(Path(small["path"]), small["sha256"])
        server = runtime["llama_server"]
        self.server = verify_file(Path(server["path"]), server["sha256"])
        if server["commit"] != LLAMA_COMMIT or server["build"] != DEVICE:
            raise ProtocolError("llama_server_commit_mismatch")
        verify_server_libraries(self.server, server["libraries"], replay=replay)
        if not replay:
            version_output = subprocess.check_output(
                [str(self.server), "--version"], stderr=subprocess.STDOUT, timeout=30,
            )
            if re.search(rb"\b" + LLAMA_COMMIT.encode() + rb"[a-f0-9]*\b", version_output) is None:
                raise ProtocolError("llama_server_version_mismatch")
            verify_cuda_lease()
            verify_cuda_device(runtime)
            verify_server_device(self.server, runtime["device"])
        self.tokenizer = load_pinned_reference_tokenizer(tokenizer_path)

    @contextmanager
    def generator(self, label: str) -> Iterator[Generate]:
        path = self.models[label]
        if label == MODELS[2]:
            # The campaign helper uses urllib's process-wide opener. Force
            # direct loopback traffic even if the host has a proxy configured.
            install_opener(build_opener(ProxyHandler({})))
            port = free_port()
            options = [
                str(self.server), "-m", str(path), "--host", "127.0.0.1", "--port", str(port),
                "-c", str(CAPACITY), "-t", str(THREADS), "-ngl", GPU_LAYERS, "-np", "1",
                "--no-context-shift", "--log-disable",
            ]
            # Prompts and completions stay in memory. Discard backend diagnostics.
            with llama_server(options, port, os.environ, Path(os.devnull)) as url:
                def generate_gguf(prompt: list[int]) -> list[int]:
                    response = post_json(url + "/completion", completion_request(prompt, CAP, SEED), 3600)
                    return generated_ids(response, len(prompt), CAP, STOP_IDS)

                yield generate_gguf
            return

        import torch

        torch.set_num_threads(THREADS)
        torch.manual_seed(SEED)
        model = load_development_model(path, dtype="bf16", device=DEVICE).eval()
        try:
            if model.config.max_position_embeddings < CAPACITY:
                raise ProtocolError("model_context_capacity_below_frozen_capacity")

            def generate_bf16(prompt: list[int]) -> list[int]:
                with torch.inference_mode():
                    return batched_greedy_generate(model, [prompt], cap=CAP, stop_ids=STOP_IDS, device=DEVICE)[0]

            yield generate_bf16
        finally:
            # Free the device before the next model, and before llama-server takes its share.
            model = None
            gc.collect()
            torch.cuda.empty_cache()
