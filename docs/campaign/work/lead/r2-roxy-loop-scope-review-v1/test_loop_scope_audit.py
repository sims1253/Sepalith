#!/usr/bin/env python3
import json, subprocess, tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def row(i,t,n): return {'row_id':i,'target_definition_name':t,'target_definition_span':[1,1],'residual_noncall_names':[n],'selected_context_tokens':1,'target_body_tokens':1}
with tempfile.TemporaryDirectory(prefix='sepalith-loop-test-') as d:
 p=Path(d); src=p/'fixture.R'
 src.write_text('loop_ok <- function(xs) { for (i in xs) { f <- function() i + 1; f() } }\ntrue_global <- function(xs) { for (i in xs) i + threshold }\nseq_bad <- function(xs) { for (i in seq_len(i)) i }\n')
 inp=p/'in.json'; out=p/'out.jsonl'
 inp.write_text(json.dumps({'source_groups':[{'source_path':str(src),'source_sha256':'test','rows':[row('ok','loop_ok','i'),row('global','true_global','threshold'),row('sequence','seq_bad','i')]}]}))
 r=subprocess.run(['Rscript','--vanilla',str(HERE/'loop_scope_audit.R'),str(inp),str(out)],text=True,capture_output=True,timeout=30)
 assert r.returncode==0,r.stderr
 got={x['row_id']:x for x in map(json.loads,out.read_text().splitlines())}
 assert got['ok']['status']=='recoverable_lexically_bound_residuals'
 assert got['global']['status']=='hold_unbound_mixed_or_outside_scope'
 assert got['sequence']['status']=='hold_unbound_mixed_or_outside_scope'
 print('PASS: 3 lexical-scope controls')
