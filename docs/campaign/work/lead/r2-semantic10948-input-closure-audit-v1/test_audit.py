#!/usr/bin/env python3
import copy
import importlib.util
import json
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('closure',HERE/'audit_input_closure.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
source=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10952-preparation-v1')
render=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/render-inputs-v1')
terminal=json.loads((source/'terminal.json').read_text());manifest=json.loads((render/'manifest.json').read_text());preps={s:json.loads((source/f'shard-{s:04d}/preparation-manifest.json').read_text()) for s in range(12,27)}
A.validate_metadata(terminal,manifest,preps)

def rejects(mutator):
 t,m,p=copy.deepcopy(terminal),copy.deepcopy(manifest),copy.deepcopy(preps);mutator(t,m,p)
 try:A.validate_metadata(t,m,p)
 except (ValueError,KeyError,AssertionError):return
 raise AssertionError('mutation accepted')

rejects(lambda t,m,p:t.update(status='complete'))
rejects(lambda t,m,p:p[12].update(status='unknown'))
rejects(lambda t,m,p:t['shards'].pop())
rejects(lambda t,m,p:t['shards'].__setitem__(1,copy.deepcopy(t['shards'][0])))
rejects(lambda t,m,p:m['shards'][0].update(rows=m['shards'][0]['rows']+1))
rejects(lambda t,m,p:t.update(hold_rows=3))
try:A.validate_id_join({'a':{}},{'a':{}},{},{'substituted':{}},'mutation')
except ValueError:pass
else:raise AssertionError('substituted ID accepted')
try:A.assert_no_target_keys({'nested':{'gold_completion':'x'}})
except ValueError:pass
else:raise AssertionError('target key accepted')
with tempfile.TemporaryDirectory() as td:
 q=Path(td)/'row';q.write_bytes(b'good');pin={'sha256':A.sha(q),'bytes':4};A.verify_file(q,pin,'ok');q.write_bytes(b'evil')
 try:A.verify_file(q,pin,'mutated')
 except ValueError:pass
 else:raise AssertionError('hash/bytes mutation accepted')
print('PASS 10 controls: actual metadata, terminal status, shard status, missing shard, duplicate shard, row count, hold drift, substituted ID, target key, file mutation')
