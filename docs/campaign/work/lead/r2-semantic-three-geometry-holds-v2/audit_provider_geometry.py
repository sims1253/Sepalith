#!/usr/bin/env python3
"""Test exact raw full-document geometry against the real primary-path EOL gate."""
import argparse,hashlib,json,os
from pathlib import Path

IDS=("4e2a158ea14eb636107c0a1f","4f19a0591c9b56764705cec3","4fc746562d6b6f36c95df72b")
SIDECAR_SHA="6e59fa3ac313fd59f76facfdd0a5e8684f114b18927eff181ba88975f0d64fb6"
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def rows(p):
 out={}
 with Path(p).open() as stream:
  for line in stream:
   x=json.loads(line);out[x['row_id']]=x
 return out
def eols(segment):
 out=[];i=0
 while i<len(segment):
  if segment[i:i+2]==b'\r\n':out.append(b'\r\n');i+=2
  elif segment[i:i+1]==b'\n':out.append(b'\n');i+=1
  elif segment[i:i+1]==b'\r':raise ValueError('lone_cr')
  else:i+=1
 return out
def prepare(sidecar):
 rid=sidecar['row_id'];s=sidecar['source'];path=Path(s['path']);raw=path.read_bytes()
 if sha_bytes(raw)!=s['sha256']:raise ValueError('source_changed')
 start,end=s['target_raw_byte_range'];target=raw[start:end];terminators=eols(target)
 if len(terminators)!=len(sidecar['target_lines']):raise ValueError('target_eol_count')
 anchor=terminators[-1];before=raw[:start]+anchor+raw[end:]
 if before[:start]+target+before[start+len(anchor):]!=raw:raise ValueError('raw_reapply_failed')
 text=before.decode('utf-8');cursor={'line':before[:start].count(b'\n'),'character':0}
 if ('\n'.join(sidecar['target_lines'])).encode() in before.replace(b'\r\n',b'\n'):raise ValueError('target_leak')
 crlf=before.count(b'\r\n');lf=before.count(b'\n');mode='mixed' if crlf and lf>crlf else 'crlf' if crlf else 'lf'
 return {'schema':'sepalith.dat10.semantic_three_geometry.provider_probe.v2','row_id':rid,'path':Path(s['path']).name,'preedit_text':text,'preedit_sha256':sha_bytes(before),'cursor':cursor,'expected_document_eol':mode,'full_raw_reapplication_exact':True,'target_free':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if sha(a.sidecar)!=SIDECAR_SHA:raise ValueError('review_sidecar_pin_mismatch')
 if a.output.exists():raise ValueError('fresh_output_required')
 source=rows(a.sidecar);probes=[prepare(source[r]) for r in IDS];a.output.mkdir(parents=True)
 pp=a.output/'provider-probes.review-only.jsonl';hh=a.output/'preparation-holds.jsonl'
 pp.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in probes))
 holds=[{'row_id':r,'reason':'mixed_document_eol_unsupported_by_primary_path','provider_prediction_emitted':False,'training_admission':False} for r in IDS]
 hh.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in holds))
 result={'schema':'sepalith.dat10.semantic_three_geometry_provider_audit.v2','status':'complete_three_protocol_holds','rows':3,'provider_predictions':0,'protocol_holds':3,'exact_closure':True,'probes':{'path':str(pp),'sha256':sha(pp)},'holds':{'path':str(hh),'sha256':sha(hh)},'training_admission':False}
 (a.output/'manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
