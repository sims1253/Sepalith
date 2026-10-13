#!/usr/bin/env python3
"""Negative controls for corrected R lexical-scope handling."""
from __future__ import annotations
import hashlib, json, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / 'corrected_loop_scope_audit.R'

def row(row_id: str, target: str, *names: str) -> dict:
    return {
        'row_id': row_id, 'target_definition_name': target,
        'target_definition_span': [1, 1], 'residual_noncall_names': list(names),
        'selected_context_tokens': 1, 'target_body_tokens': 1,
    }

def main() -> None:
    with tempfile.TemporaryDirectory(prefix='sepalith-corrected-loop-') as directory:
        root = Path(directory); source = root / 'fixture.R'
        source.write_text('''
loop_ok <- function(xs) { for (i in xs) { f <- function() i; f() } }
global_bad <- function(xs) { for (i in xs) i + threshold }
default_shadow <- function(xs, y = i) { for (i in xs) i }
default_formal <- function(x, y = x) y
call_head_global <- function(x) { threshold(x) }
member_literal <- function(x) x$group
''')
        targets = [
            row('loop', 'loop_ok', 'i'), row('global', 'global_bad', 'threshold'),
            row('default-shadow', 'default_shadow', 'i'), row('default-formal', 'default_formal', 'x'),
            row('call-head', 'call_head_global', 'threshold'), row('member', 'member_literal', 'group'),
        ]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        inp = root / 'input.json'; out = root / 'output.jsonl'
        inp.write_text(json.dumps({'source_groups': [{'source_path': str(source), 'source_sha256': digest, 'rows': targets}]}))
        completed = subprocess.run(['Rscript', '--vanilla', str(AUDIT), str(inp), str(out)], text=True, capture_output=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        got = {item['row_id']: item for item in map(json.loads, out.read_text().splitlines())}
        assert got['loop']['status'] == 'recoverable_lexically_bound_residuals', got['loop']
        for ident in ('global', 'default-shadow', 'call-head', 'member'):
            assert got[ident]['status'] == 'hold_unbound_mixed_or_outside_scope', got[ident]
        assert got['default-formal']['status'] == 'recoverable_lexically_bound_residuals', got['default-formal']
        assert got['default-shadow']['evidence']['residuals']['i']['occurrence_evidence'][0]['location'] == 'root_formal_default'
        assert got['call-head']['evidence']['residuals']['threshold']['occurrence_evidence'][0]['location'] == 'call_head'
        assert got['member']['evidence']['residuals']['group']['occurrence_evidence'][0]['classification'] == 'member_literal_not_lexical'
    print('6/6 corrected lexical-scope controls passed')

if __name__ == '__main__':
    main()
