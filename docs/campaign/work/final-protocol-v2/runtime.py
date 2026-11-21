"""Use frozen CPU generation backends and the campaign's exact-match scorer."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
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
THREADS = 8
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
        server = runtime["llama_server"]
        self.server = verify_file(Path(server["path"]), server["sha256"])
        if server["commit"] != LLAMA_COMMIT:
            raise ProtocolError("llama_server_commit_mismatch")
        if not replay:
            version_output = subprocess.check_output(
                [str(self.server), "--version"], stderr=subprocess.STDOUT, timeout=30,
            )
            if re.search(rb"\b" + LLAMA_COMMIT.encode() + rb"[a-f0-9]*\b", version_output) is None:
                raise ProtocolError("llama_server_version_mismatch")
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
                "-c", str(CAPACITY), "-t", str(THREADS), "-ngl", "0", "-np", "1",
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
        model = load_development_model(path, dtype="bf16", device="cpu").eval()
        if model.config.max_position_embeddings < CAPACITY:
            raise ProtocolError("model_context_capacity_below_frozen_capacity")

        def generate_bf16(prompt: list[int]) -> list[int]:
            with torch.inference_mode():
                return batched_greedy_generate(model, [prompt], cap=CAP, stop_ids=STOP_IDS, device="cpu")[0]

        yield generate_bf16
