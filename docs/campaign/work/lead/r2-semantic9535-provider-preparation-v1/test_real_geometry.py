#!/usr/bin/env python3
"""Use real reviewed rows to test CRLF/non-ASCII source geometry without rendering."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards27to40-v1')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=load('s9535_geometry_test',HERE/'geometry-source/base_materializer.py')
def rows(path):
 out={}
 for x in map(json.loads,Path(path).read_text().splitlines()):
  rid=x.get('row_id') or x.get('row_ref',{}).get('row_id')
  assert rid and rid not in out
  out[rid]=x
 return out
def utf16_units(text):return len(text.encode('utf-16-le'))//2

found={};tokenizer=B.TokenizerAdapter()
for shard in range(27,41):
 m=json.loads((ROOT/f'shard-{shard:04d}/manifest.json').read_text());sem=rows(ROOT/f'shard-{shard:04d}/semantic-ledger.jsonl');binding=m['streaming_binding'];prov=rows(binding['provenance_ledger']['path']);packets=rows(binding['candidate_packet']['path'])
 for rid,s in sem.items():
  if s.get('status')!='semantic_supported_context_closure_root_review_required' or s.get('reasons')!=[]:continue
  raw=Path(s['source_path']).read_bytes();features=[]
  if s.get('target_occurrence_method')=='uniform_crlf_to_lf' and b'\r\n' in raw:features.append('crlf')
  try:text=raw.decode('utf-8')
  except UnicodeDecodeError:continue
  if any(ord(c)>127 for c in text):features.append('nonascii')
  if not any(x not in found for x in features):continue
  try:row,profile=B.materialize_row(s,prov[rid],packets[rid],tokenizer,{})
  except B.Hold:continue
  assert profile['geometry']['application_exact'] is True and profile['protocol']['strict_validator_errors']==[]
  assert row['target_body_text'] not in row['prompt_text']
  assert profile['geometry']['global_replacement_range']['start']['character']==0
  assert profile['geometry']['global_replacement_range']['start']==profile['geometry']['global_replacement_range']['end']
  # Exercise UTF-16 accounting on the real prompt/source rather than treating code points as units.
  assert utf16_units(row['prompt_text'])>=len(row['prompt_text'])
  for feature in features:
   if feature not in found:found[feature]={'row_id':rid,'shard':shard,'source_path':s['source_path'],'document_eol':profile['geometry']['selection']['document_eol'] if 'document_eol' in profile['geometry']['selection'] else None,'prompt_utf16_units':utf16_units(row['prompt_text']),'prompt_codepoints':len(row['prompt_text'])}
  if set(found)=={'crlf','nonascii'}:break
 if set(found)=={'crlf','nonascii'}:break
assert set(found)=={'crlf','nonascii'},found
# Infrastructure source I/O must escape as an error rather than become a semantic hold.
rid=found['nonascii']['row_id'];shard=found['nonascii']['shard'];m=json.loads((ROOT/f'shard-{shard:04d}/manifest.json').read_text());sem=rows(ROOT/f'shard-{shard:04d}/semantic-ledger.jsonl');prov=rows(m['streaming_binding']['provenance_ledger']['path']);packets=rows(m['streaming_binding']['candidate_packet']['path']);broken=dict(sem[rid]);broken['source_path']='/mnt/h/sepalith/normalized/definitely/missing/semantic9535.R';broken_packet=json.loads(json.dumps(packets[rid]));broken_packet['validation']['source_path']=broken['source_path']
try:B.materialize_row(broken,prov[rid],broken_packet,tokenizer,{})
except FileNotFoundError:pass
else:raise AssertionError('missing source did not fail as infrastructure')
print(json.dumps({'status':'PASS','real_geometry':found,'missing_source_failed_command':True},sort_keys=True))
