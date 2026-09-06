"""Recover landscape prompt additions from frozen constants and pinned examples."""

import ast
import hashlib
import io
import json
from pathlib import Path


def constants(source, names):
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if values.keys() != set(names):
        raise ValueError("Missing prompt constant")
    return values


def additions(e, renderer):
    zero = constants(
        e.read("/landscape_glm_scenarios.py"), {"FMT_INSTRUCTION", "ONESHOT_EXAMPLE"}
    )
    source = e.read("/landscape_glm_3shot.py")
    pins = constants(source, {"FEWSHOT_PINS", "FEWSHOT_LABELS"})
    # Compile only the reviewed, pure construction function, without importing
    # provider backends or any inference driver. Every read uses frozen bytes.
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "build_fewshot"
    )

    def edit_row(row, family, package):
        prompt = renderer.render_zeta2(dict(row, suffix=[]))
        target = "\n".join(row["region_new"]).rstrip() + "\n>>>>>>> UPDATED"
        if len(prompt) + len(target) > 6000:
            return None
        return {"prompt": prompt, "target": target}

    def frozen_open(path):
        return io.StringIO(e.read("/" + str(path)))

    env = pins | {
        "json": json,
        "hashlib": hashlib,
        "edit_row": edit_row,
        "open": frozen_open,
        "HOLDOUT_REF": "sft_v3/eval.jsonl",
        "SCEN_DIR": Path("scenarios_v1"),
        "FAMILIES": tuple(pins["FEWSHOT_PINS"]),
    }
    exec(  # noqa: S102 - reviewed pure function; every read uses frozen bytes
        compile(
            ast.Module(body=[node], type_ignores=[]), "<frozen build_fewshot>", "exec"
        ),
        env,
    )
    block, provenance = env["build_fewshot"]()
    return zero | {
        "three_shot": zero["FMT_INSTRUCTION"] + "\n\n" + block,
        "provenance": provenance,
    }
