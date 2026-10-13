#!/usr/bin/env python3
"""Hash-checked binding for an expanded TRAIN reward-buffer sidecar."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from typing import Any,Iterable,Mapping

class BindingError(ValueError): pass

def require(ok:bool,why:str)->None:
 if not ok:raise BindingError(why)

def sha_file(path:Path)->str:
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()

@dataclass(frozen=True)
class RewardBufferIndex:
 root:Path
 records:Mapping[str,Mapping[str,Any]]
 sidecar_sha256:str

 @classmethod
 def load(cls,manifest_path:Path,expected_manifest_sha256:str,ordered_ids:Iterable[str],*,allow_repairs:bool=False):
  require(sha_file(manifest_path)==expected_manifest_sha256,'buffer_manifest_hash_mismatch')
  manifest=json.loads(manifest_path.read_text());require(manifest['status']=='prepared_root_review_required','buffer_status_not_reviewable')
  root=manifest_path.parent;meta=manifest['artifacts']['reward-buffer-sidecar.jsonl'];sidecar=root/'reward-buffer-sidecar.jsonl'
  require(sidecar.stat().st_size==meta['bytes'] and sha_file(sidecar)==meta['sha256'],'buffer_sidecar_identity_mismatch')
  wanted=list(ordered_ids);records={};seen=[]
  with sidecar.open() as f:
   for position,line in enumerate(f):
    x=json.loads(line);rid=x.get('row_id');require(x.get('position')==position,'buffer_position_mismatch')
    require(isinstance(rid,str) and rid not in records,'buffer_id_invalid_or_duplicate')
    if not x.get('supported') and not allow_repairs:raise BindingError(f'buffer_repair_not_admitted:{rid}:{x.get("repair_reason")}')
    if x.get('supported'):
     if x.get('buffer_mode')=='framed_fragment':require(x.get('framed_projection_parse_ok') is True,'framed_gold_parse_not_verified')
     else:require(x.get('gold_applied_parse_ok') is True,'gold_parse_not_verified')
     if x.get('buffer_mode')=='complete_document':require(x.get('baseline_parse_ok') is True,'complete_baseline_parse_not_verified')
     require(x.get('buffer_mode') in {'complete_document','completion_prefix','framed_fragment'},'buffer_mode_invalid')
    records[rid]=x;seen.append(rid)
  require(seen==wanted,'buffer_order_or_coverage_mismatch')
  return cls(root,records,meta['sha256'])

 def envelope_for(self,row_id:str)->dict[str,Any]:
  require(row_id in self.records,'buffer_row_missing');x=self.records[row_id]
  # Explicit metadata only. Baseline bytes remain in the immutable content-addressed blob.
  return {k:x[k] for k in ('row_id','position','supported','repair_reason','buffer_mode','baseline_sha256',
          'baseline_bytes','baseline_blob','baseline_parse_ok','gold_applied_sha256','gold_applied_parse_ok',
          'parse_projection_sha256','diagnostic_suffix','framed_projection_parse_ok','parser_identity') if k in x}

 def baseline_text(self,envelope:Mapping[str,Any])->str:
  require(envelope.get('supported') is True,'unsupported_buffer_load_refused')
  rel=envelope.get('baseline_blob');require(isinstance(rel,str) and not Path(rel).is_absolute() and '..' not in Path(rel).parts,'baseline_blob_path_invalid')
  path=self.root/rel;raw=path.read_bytes();require(len(raw)==envelope['baseline_bytes'],'baseline_blob_size_mismatch')
  require(hashlib.sha256(raw).hexdigest()==envelope['baseline_sha256'],'baseline_blob_hash_mismatch')
  return raw.decode('utf-8')
