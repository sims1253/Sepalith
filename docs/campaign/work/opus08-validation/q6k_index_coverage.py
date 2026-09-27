#!/usr/bin/env python3
"""CPU-only q6_K lane/index reference for the OPUS08 Vulkan screen.

This is a small reference model of the pinned b10453 ``mul_mat_vec_q6_k``
lane map.  It does not invoke Vulkan, load a model, or claim that a shader
change is correct.  It checks the storage layout, scalar coverage, q6_K
dequantisation, and the workgroup-64 scheduling shape that a later notebook
run must compile and exercise.

The source contract is deliberately pinned by SHA-256 when ``--source-root``
is supplied.  A changed source tree therefore fails closed instead of making
the CPU result look like evidence for a different shader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


QK_K = 256
QL_BYTES = QK_K // 2
QH_BYTES = QK_K // 4
SCALE_BYTES = QK_K // 16
BLOCK_BYTES = QL_BYTES + QH_BYTES + SCALE_BYTES + 2
TARGET_WIDTHS = (2048, 6144)
SUBGROUP = 64
WORKGROUP = 64
LANES_PER_BLOCK = 16
BLOCKS_PER_WORKGROUP = WORKGROUP // LANES_PER_BLOCK

# These are all public source files used by the mapping or by the canonical
# generator.  The source checkout is the only input; no weights are read.
SOURCE_HASHES = {
    "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp":
        "bc785a2457aa5b04d416e18f5ae2304da267e35cea7b54ac67b5d20987e450af",
    "ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl":
        "3d493e6ee65459a44b2b81645f34f4aa5f3bca4a22ca18cb07706535032d9d8c",
    "ggml/src/ggml-vulkan/vulkan-shaders/types.glsl":
        "8b65027ce3cbd48d7a6a926717cfe78bc27ea8156ba8e2bb3fbd6d07142e710a",
    "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl":
        "fc7bb5cc47bd9761f8a3484466e10b61d7866df41a150cdd47cfa1f389110323",
    "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp":
        "c27978351492b0cdffac906f10662a979e5e7a0810e0b73b59853e7230a231d9",
    "ggml/src/ggml-vulkan/CMakeLists.txt":
        "95b77526cbede60b669abf1bc13657956284d0b20835096e64f24682a5a1a0e2",
    "ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt":
        "0cb2c1fd49730793094cbaca4a46d0e0fc56c8f279aa9712a4e7a53535400d64",
    "ggml/src/ggml-common.h":
        "af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0",
    "ggml/src/ggml-quants.c":
        "07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d",
    "ggml/src/ggml-vulkan/ggml-vulkan.cpp":
        "c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f",
}


def _signed_byte(value: int) -> int:
    """Interpret one q6 scale byte as C/GLSL int8_t."""

    return value if value < 128 else value - 256


def make_block(seed: int) -> bytes:
    """Make one deterministic, valid q6_K block without model data."""

    ql = bytes((37 * i + 13 * seed + 0x21) & 0xFF for i in range(QL_BYTES))
    qh = bytes((53 * i + 7 * seed + 0x49) & 0xFF for i in range(QH_BYTES))
    scales = bytes((((17 * i + 11 * seed) % 127) - 63) & 0xFF for i in range(SCALE_BYTES))
    # 1.0, 1.125, ... are exactly representable in binary16 and keep the
    # float comparison independent of host conversion quirks.
    d = struct.pack("<e", 1.0 + (seed % 4) / 8.0)
    block = ql + qh + scales + d
    if len(block) != BLOCK_BYTES:
        raise AssertionError(f"q6_K block size {len(block)} != {BLOCK_BYTES}")
    return block


def _parts(block: bytes) -> Tuple[bytes, bytes, List[int], float]:
    if len(block) != BLOCK_BYTES:
        raise ValueError(f"q6_K block must be {BLOCK_BYTES} bytes, got {len(block)}")
    ql = block[:QL_BYTES]
    qh = block[QL_BYTES:QL_BYTES + QH_BYTES]
    scales_raw = block[QL_BYTES + QH_BYTES:QL_BYTES + QH_BYTES + SCALE_BYTES]
    scales = [_signed_byte(v) for v in scales_raw]
    d = struct.unpack("<e", block[-2:])[0]
    return ql, qh, scales, d


def canonical_terms(block: bytes) -> List[Tuple[int, int, float]]:
    """Return (q, scale-indexed value, dequantized value) in C row order.

    This follows ``dequantize_row_q6_K`` in ggml-quants.c: each half-block
    emits four groups at offsets 0, 32, 64, and 96, with scale indices
    ``is + 0, +2, +4, +6``.
    """

    ql, qh, scales, d = _parts(block)
    out: List[Optional[Tuple[int, int, float]]] = [None] * QK_K
    for half in range(2):
        ql_base = half * 64
        qh_base = half * 32
        scale_base = half * 8
        scalar_base = half * 128
        for l in range(32):
            is_ = l // 16
            q_values = (
                ((ql[ql_base + l] & 0x0F) | ((qh[qh_base + l] & 0x03) << 4)) - 32,
                ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4)) - 32,
                ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) - 32,
                ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4)) - 32,
            )
            for group, q in enumerate(q_values):
                scalar = scalar_base + 32 * group + l
                scale = scales[scale_base + is_ + 2 * group]
                if out[scalar] is not None:
                    raise AssertionError(f"canonical duplicate scalar {scalar}")
                out[scalar] = (q, scale, d * scale * q)
    if any(value is None for value in out):
        raise AssertionError("canonical q6_K map left an output scalar unset")
    return [value for value in out if value is not None]


def _lane_map(itid: int) -> Tuple[int, int, int, int, int, int]:
    """Return the q6 shader's six per-lane offsets."""

    if not 0 <= itid < LANES_PER_BLOCK:
        raise ValueError(f"itid must be 0..15, got {itid}")
    v_im = itid // 8
    v_in = itid - 8 * v_im
    l0 = 4 * v_in
    is_ = v_in // 4
    ql_offset = 64 * v_im + l0
    qh_offset = 32 * v_im + l0
    s_offset = 8 * v_im + is_
    y_offset = 128 * v_im + l0
    return v_im, v_in, l0, is_, ql_offset, qh_offset, s_offset, y_offset


def shader_lane_terms(
    block: bytes,
    itid: int,
    *,
    inject_fault: bool = False,
) -> Tuple[List[Tuple[int, int, int, float]], Dict[str, set]]:
    """Model one invocation's four vec4 q6 groups.

    The first tuple element is the output scalar, followed by q, scale, and
    dequantized value.  ``accesses`` records byte and packed16 indices so the
    caller can check bounds and complete storage coverage.
    """

    ql, qh, scales, d = _parts(block)
    _, _, l0, _, ql_offset, qh_offset, s_offset, y_offset = _lane_map(itid)
    accesses = {
        "ql": set(),
        "qh": set(),
        "scales": {itid},  # the shared-cache load at shader line 55
        "ql_packed16": set(),
        "qh_packed16": set(),
        "scale_values": set(),
    }
    result: List[Tuple[int, int, int, float]] = []
    # q0..q3 in the shader.  ql is low/high nibble for the first/second half;
    # qh uses bit pairs 0,2,4,6.  The ql+32 source is the second 32-wide row.
    for group, (ql_half_offset, ql_shift, qh_shift, y_group) in enumerate(
        ((ql_offset, 0, 0, 0), (ql_offset + 32, 0, 2, 1),
         (ql_offset, 4, 4, 2), (ql_offset + 32, 4, 6, 3))
    ):
        scale_index = s_offset + 2 * group
        if not 0 <= scale_index < SCALE_BYTES:
            raise AssertionError(f"shader scale index out of bounds: {scale_index}")
        accesses["scale_values"].add(scale_index)
        scale = scales[scale_index]
        for lane_in_vec in range(4):
            ql_index = ql_half_offset + lane_in_vec
            qh_index = qh_offset + lane_in_vec
            if inject_fault and itid == 0 and group == 0 and lane_in_vec == 0:
                # A one-byte source-index slip must be detected by the test;
                # this is never used by production code.
                ql_index += 1
            if not 0 <= ql_index < QL_BYTES:
                raise AssertionError(f"shader ql index out of bounds: {ql_index}")
            if not 0 <= qh_index < QH_BYTES:
                raise AssertionError(f"shader qh index out of bounds: {qh_index}")
            accesses["ql"].add(ql_index)
            accesses["qh"].add(qh_index)
            accesses["ql_packed16"].update((ql_index // 2,))
            accesses["qh_packed16"].update((qh_index // 2,))
            qlow = (ql[ql_index] >> ql_shift) & 0x0F
            qhigh = ((qh[qh_index] >> qh_shift) & 0x03) << 4
            q = qlow | qhigh
            q -= 32
            scalar = y_offset + 32 * y_group + lane_in_vec
            if not 0 <= scalar < QK_K:
                raise AssertionError(f"shader output index out of bounds: {scalar}")
            result.append((scalar, q, scale, d * scale * q))
    return result, accesses


def _merge_accesses(dst: Dict[str, set], src: Dict[str, set]) -> None:
    for key, values in src.items():
        dst.setdefault(key, set()).update(values)


def shader_row_terms(
    blocks: Sequence[bytes],
    width: int,
    *,
    subgroup: int = SUBGROUP,
    workgroup: int = WORKGROUP,
    inject_fault: bool = False,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, set]]:
    """Simulate the q6 shader's workgroup-64 block scheduling and lanes."""

    if width <= 0 or width % QK_K:
        raise ValueError("shader row reference requires a positive QK_K-aligned width")
    if subgroup != SUBGROUP or workgroup != WORKGROUP:
        raise ValueError("this OPUS08 reference is specialized to subgroup/workgroup 64")
    block_count = width // QK_K
    if len(blocks) != block_count:
        raise ValueError(f"expected {block_count} q6_K blocks, got {len(blocks)}")
    outputs: List[Optional[Tuple[int, int, float]]] = [None] * width
    writes = [0] * width
    accesses = {key: set() for key in ("ql", "qh", "scales", "ql_packed16", "qh_packed16", "scale_values")}

    # For each group, tid/16 selects one block and tid%16 selects one q6 lane.
    # This mirrors the full and all_threads=false calls in compute_outputs.
    for first_block in range(0, block_count, BLOCKS_PER_WORKGROUP):
        for tid in range(workgroup):
            ix = tid // LANES_PER_BLOCK
            itid = tid % LANES_PER_BLOCK
            block_index = first_block + ix
            if block_index >= block_count:
                # The source's partial-group path loads a scale then continues
                # before ql/qh reads.  There is no output for this invocation.
                continue
            lane_result, lane_accesses = shader_lane_terms(
                blocks[block_index], itid,
                inject_fault=inject_fault,
            )
            _merge_accesses(accesses, lane_accesses)
            for scalar, q, scale, value in lane_result:
                output_index = block_index * QK_K + scalar
                if not 0 <= output_index < width:
                    raise AssertionError(f"row output index out of bounds: {output_index}")
                writes[output_index] += 1
                if writes[output_index] > 1:
                    raise AssertionError(f"shader duplicate output scalar {output_index}")
                outputs[output_index] = (q, scale, value)

    if any(count != 1 for count in writes):
        missing = [i for i, count in enumerate(writes) if count != 1][:8]
        raise AssertionError(f"shader output coverage is not exactly once; examples={missing}")
    if any(value is None for value in outputs):
        raise AssertionError("shader output coverage left a scalar unset")
    expected_accesses = {
        "ql": set(range(QL_BYTES)),
        "qh": set(range(QH_BYTES)),
        "scales": set(range(SCALE_BYTES)),
        "ql_packed16": set(range(QL_BYTES // 2)),
        "qh_packed16": set(range(QH_BYTES // 2)),
        "scale_values": set(range(SCALE_BYTES)),
    }
    if accesses != expected_accesses:
        raise AssertionError(f"q6_K access coverage mismatch: {accesses} != {expected_accesses}")
    return [value for value in outputs if value is not None], accesses


def validate_width(width: int, *, seed: int = 17) -> Dict[str, object]:
    """Run CPU dequant and lane parity for one aligned width."""

    blocks = [make_block(seed + i) for i in range(width // QK_K)]
    canonical: List[Tuple[int, int, float]] = []
    for block in blocks:
        canonical.extend(canonical_terms(block))
    shader, accesses = shader_row_terms(blocks, width)
    if len(shader) != len(canonical):
        raise AssertionError(f"term count mismatch: {len(shader)} != {len(canonical)}")
    for index, (expected, observed) in enumerate(zip(canonical, shader)):
        eq = expected[:2] == observed[:2] and math.isclose(
            expected[2], observed[2], rel_tol=0.0, abs_tol=1e-12
        )
        if not eq:
            raise AssertionError(f"q6_K mismatch at scalar {index}: expected={expected} observed={observed}")
    expected_accesses = {
        "ql": set(range(QL_BYTES)),
        "qh": set(range(QH_BYTES)),
        "scales": set(range(SCALE_BYTES)),
        "ql_packed16": set(range(QL_BYTES // 2)),
        "qh_packed16": set(range(QH_BYTES // 2)),
        "scale_values": set(range(SCALE_BYTES)),
    }
    if accesses != expected_accesses:
        raise AssertionError(f"q6_K access coverage mismatch: {accesses} != {expected_accesses}")
    return {
        "width": width,
        "blocks": len(blocks),
        "subgroup": SUBGROUP,
        "workgroup": WORKGROUP,
        "terms": len(shader),
        "exact_scalar_coverage": True,
        "exact_storage_coverage": True,
    }


def dispatch_plan(width: int, *, subgroup: int = SUBGROUP, workgroup: int = WORKGROUP) -> Dict[str, object]:
    """Describe the safe route for a candidate shape.

    The current pinned shader is generic q6_K code.  ``candidate`` marks only
    the two OPUS08 model widths; other aligned widths remain a reference-only
    generic route until the runtime selects a known-compatible pipeline.
    """

    if width <= 0:
        return {"mode": "fallback", "reason": "non-positive width"}
    if width % QK_K:
        return {"mode": "fallback", "reason": "partial q6_K block"}
    if subgroup != SUBGROUP or workgroup != WORKGROUP:
        return {"mode": "fallback", "reason": "unsupported subgroup/workgroup shape"}
    blocks = width // QK_K
    if width in TARGET_WIDTHS:
        mode = "candidate"
    elif blocks % BLOCKS_PER_WORKGROUP:
        mode = "generic_q6_partial_workgroup_reference"
    else:
        mode = "generic_q6_reference"
    return {
        "mode": mode,
        "width": width,
        "blocks": blocks,
        "subgroup": subgroup,
        "workgroup": workgroup,
        "blocks_per_workgroup": BLOCKS_PER_WORKGROUP,
    }


def source_contract(source_root: Path) -> Dict[str, object]:
    """Hash and minimally inspect the pinned public shader source."""

    source_root = source_root.resolve()
    observed: Dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[relative] = digest
        if digest != expected:
            raise AssertionError(f"source hash changed for {relative}: {digest} != {expected}")
    shader = (source_root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp").read_text()
    cpu = (source_root / "ggml/src/ggml-quants.c").read_text()
    required_shader_fragments = (
        "layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;",
        "const uint it_size = gl_WorkGroupSize.x/16;",
        "const uint ql_offset = 64*v_im + l0;",
        "const uint qh_offset = 32*v_im + l0;",
        "data_a_packed16[ib0 + i].ql[ql_offset / 2]",
        "vec4(unpack8(q0_u32)) - 32",
    )
    required_cpu_fragments = (
        "void dequantize_row_q6_K",
        "assert(k % QK_K == 0);",
        "ql += 64;",
        "qh += 32;",
        "sc += 8;",
    )
    for fragment in required_shader_fragments:
        if fragment not in shader:
            raise AssertionError(f"missing q6 shader contract fragment: {fragment}")
    for fragment in required_cpu_fragments:
        if fragment not in cpu:
            raise AssertionError(f"missing q6 CPU contract fragment: {fragment}")
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "root": str(source_root),
        "commit": commit,
        "sha256": observed,
        "shader_entrypoint": "main",
        "shader_version": "#version 450",
        "q6_k_block_bytes": BLOCK_BYTES,
        "q6_k_quant_k": QK_K,
        "source_contract_checked": True,
    }


def run_cpu_suite(source_root: Optional[Path] = None) -> Dict[str, object]:
    cases = [validate_width(2048, seed=17), validate_width(6144, seed=71)]
    # Exercise the source's all_threads=false tail group without presenting it
    # as a model-shape acceptance result.
    partial_group = validate_width(3 * QK_K, seed=101)
    plans = [
        dispatch_plan(2048),
        dispatch_plan(6144),
        dispatch_plan(3 * QK_K),
        dispatch_plan(2050),
        dispatch_plan(2048, subgroup=32),
    ]
    if plans[0]["mode"] != "candidate" or plans[1]["mode"] != "candidate":
        raise AssertionError("target q6_K widths did not admit the candidate reference route")
    if plans[2]["mode"] != "generic_q6_partial_workgroup_reference":
        raise AssertionError("partial workgroup case did not remain explicitly reference-only")
    if plans[3]["mode"] != "fallback" or plans[4]["mode"] != "fallback":
        raise AssertionError("unsafe shape did not select fallback")
    blocks = [make_block(17 + i) for i in range(2048 // QK_K)]
    expected = []
    for block in blocks:
        expected.extend(canonical_terms(block))
    faulty, _ = shader_row_terms(blocks, 2048, inject_fault=True)
    # The injected one-byte slip preserves the set of touched bytes (the
    # affected byte is also read for its other nibble), so parity must compare
    # the decoded values rather than only the access set.
    fault_detected = faulty != expected
    if not fault_detected:
        raise AssertionError("intentional ql-index fault was not detected")
    result: Dict[str, object] = {
        "status": "cpu_reference_pass",
        "cases": cases,
        "partial_group_case": partial_group,
        "dispatch_plans": plans,
        "intentional_fault_detected": fault_detected,
        "no_model_or_vulkan": True,
    }
    if source_root is not None:
        result["source"] = source_contract(source_root)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="pinned llamacpp-b10453 checkout; hash it before reporting pass",
    )
    args = parser.parse_args(argv)
    try:
        result = run_cpu_suite(args.source_root)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed with context
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
