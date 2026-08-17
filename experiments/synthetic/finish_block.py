#!/usr/bin/env python3
"""Finish-block data construction from the normalized CRAN corpus (tree-sitter-r).

For every named function definition preceded by a roxygen block, emit:
  kind=signature : roxygen + signature + '{'  ->  target = full body
  kind=mid_body  : signature + first k statements -> target = rest of body
with intent-gate metadata (roxygen richness). Stats summarize gating coverage.
Output: prototype shard + stats JSON (production wiring into NAS datasets/ later).
"""
import json, re, sys, time
from pathlib import Path

import tree_sitter_r
from tree_sitter import Language, Parser

ROOT = Path("/mnt/h/sepalith/normalized")
OUT = Path(__file__).resolve().parent / "finish_block_sample.jsonl"
STATS = Path(__file__).resolve().parent / "finish_block_stats.json"
parser = Parser(Language(tree_sitter_r.language()))


def node_text(src, n):
    return src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def collect_functions(src, tree):
    """Yield (name, fn_node, roxygen_lines) for named function definitions."""
    out = []
    children = tree.root_node.children
    for i, node in enumerate(children):
        if node.type != "binary_operator":
            continue
        op = [c for c in node.children if c.type == "<-" or c.type == "="]
        rhs = [c for c in node.children if c.type == "function_definition"]
        if not op or not rhs:
            continue
        fn = rhs[0]
        name = node.children[0]  # lhs identifier/call
        roxy = []
        j = i - 1
        while j >= 0 and children[j].type == "comment":
            txt = node_text(src, children[j])
            if txt.lstrip().startswith("#'"):
                roxy.insert(0, txt)
            j -= 1
        out.append((node_text(src, name), fn, roxy))
    return out


def main():
    t0 = time.time()
    records, stats = [], dict(files=0, functions=0, with_roxygen=0, rich_roxygen=0,
                              loops=0, emitted_sig=0, emitted_mid=0, pkgs=0)
    for pkg_dir in sorted(ROOT.iterdir()):
        try:
            ver_dir = next(pkg_dir.iterdir())
            src_root = ver_dir / pkg_dir.name / "R"
        except (StopIteration, FileNotFoundError):
            continue
        if not src_root.is_dir():
            continue
        stats["pkgs"] += 1
        for f in src_root.glob("*.R"):
            src = f.read_bytes()
            if len(src) > 400_000:
                continue
            tree = parser.parse(src)
            stats["files"] += 1
            for name, fn, roxy in collect_functions(src, tree):
                stats["functions"] += 1
                if not roxy:
                    continue
                stats["with_roxygen"] += 1
                roxy_text = "\n".join(roxy)
                plain = re.sub(r"#'\s*@", "@", roxy_text)
                words = len(plain.split())
                rich = words >= 12 or ("@param" in roxy_text or "@return" in roxy_text)
                if rich:
                    stats["rich_roxygen"] += 1
                # find body braced_expression and statements
                body = next((c for c in fn.children if c.type == "braced_expression"), None)
                if body is None:
                    continue
                stmts = [c for c in body.children if c.is_named]
                stats["loops"] += sum(1 for s in stmts if s.type == "for_statement"
                                      or s.type == "while_statement")
                if not stmts:
                    continue
                # signature = everything from function start to '{'
                sig_end = body.start_byte
                sig = src[fn.start_byte:sig_end].decode("utf-8", errors="replace")
                prefix = roxy_text + "\n" + name + " <- " + sig + " {"
                body_lines = node_text(src, body)[1:-1]  # strip braces
                if 1 <= body_lines.count("\n") + 1 <= 40:
                    records.append(dict(kind="signature", package=pkg_dir.name,
                                        path=f"R/{f.name}", fn=name, gated=rich,
                                        prefix=prefix, target=body_lines))
                    stats["emitted_sig"] += 1
                # mid-body variant: keep first third, target the rest
                if len(stmts) >= 4:
                    cut = stmts[max(1, len(stmts) // 3) - 1].end_byte
                    head = src[body.start_byte + 1:cut].decode("utf-8", errors="replace")
                    tail = src[cut:body.end_byte - 1].decode("utf-8", errors="replace")
                    if 2 <= tail.count("\n") + 1 <= 30 and head.count("\n") + 1 <= 25:
                        records.append(dict(kind="mid_body", package=pkg_dir.name,
                                            path=f"R/{f.name}", fn=name, gated=rich,
                                            prefix=prefix + "\n" + head, target=tail))
                        stats["emitted_mid"] += 1
    OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    stats["elapsed_s"] = round(time.time() - t0, 1)
    STATS.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"records -> {OUT} ({len(records)})")


if __name__ == "__main__":
    main()
