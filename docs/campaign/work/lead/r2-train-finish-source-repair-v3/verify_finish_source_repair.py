#!/usr/bin/env python3
"""Independent output join and raw-applied R parse for repair v3."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p,key):return {x[key]:x for x in map(json.loads,p.open())}
def utf16(line,units):
 used=0
 for i,ch in enumerate(line):
  if used==units:return i
  used+=len(ch.encode('utf-16-le'))//2
  if used>units:raise ValueError('utf16 split')
 if used!=units:raise ValueError('utf16 overflow')
 return len(line)
def apply(doc,rr,text):
 lines=doc.split('\n')
 def off(x):return sum(len(z)+1 for z in lines[:x['line']])+utf16(lines[x['line']],x['character'])
 return doc[:off(rr['start'])]+text+doc[off(rr['end']):]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args()
 manifest=json.loads((a.data/'materialization.json').read_text()); original=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
 rows=load(a.data/'train-token-rows.jsonl','id'); old=load(original,'id'); contexts=load(a.data/'context-sidecar.jsonl','row_id'); ledger=load(a.data/'repair-ledger.jsonl','row_id')
 assert len(rows)==len(old)==len(contexts)==15006 and len(ledger)==3503 and set(rows)==set(old)==set(contexts) and set(ledger)<set(rows)
 assert sha(a.data/'context-sidecar.jsonl')=='36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a'
 paths=[]; applied_hash={}
 with tempfile.TemporaryDirectory() as td:
  td=Path(td)
  for i,(rid,evidence) in enumerate(ledger.items()):
   arow,orow=rows[rid],old[rid]; assert arow['target_body_text']==orow['target_body_text']+'}'
   assert arow['input_ids'][:arow['target_start']]==orow['input_ids'][:orow['target_start']]
   assert arow['prompt_text']==orow['prompt_text'] and arow['target_start']==orow['target_start']
  assert all(rows[rid]==old[rid] for rid in set(rows)-set(ledger))
  # Load provenance once outside the per-row assertions to avoid trusting the ledger's applied hash.
  prov=load(Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl'),'row_id')
  for i,(rid,evidence) in enumerate(ledger.items()):
   row=rows[rid];c=contexts[rid]['context'];doc=prov[rid]['source_provenance']['selection_source']['document_text'];post=apply(doc,c['replacement_range'],row['target_body_text']);
   digest=hashlib.sha256(post.encode()).hexdigest();assert digest==evidence['applied_document_sha256'];applied_hash[rid]=digest
   p=td/f'{i:04d}.R';p.write_text(post);paths.append(p)
  pf=td/'paths.txt';rf=td/'results.txt';pf.write_text(''.join(str(x)+'\n' for x in paths))
  run=subprocess.run(['Rscript','--vanilla',str(Path(__file__).with_name('parse_only.R')),str(pf),str(rf)],capture_output=True,text=True,timeout=900)
  status=rf.read_text().splitlines();assert run.returncode==0 and len(status)==3503 and all(x=='1' for x in status)
 result_path=a.data/'raw-parse-results.jsonl'
 with result_path.open('x') as f:
  for rid in ledger:f.write(json.dumps({'row_id':rid,'applied_document_sha256':applied_hash[rid],'raw_applied_parse_ok':True},sort_keys=True,separators=(',',':'))+'\n')
 version=subprocess.run(['Rscript','--version'],capture_output=True,text=True,timeout=10);rver=(version.stdout or version.stderr).strip()
 report={'schema':'sepalith.dat10.finish-source-repair-independent-verification.v1','status':'pass','rows':15006,'changed_rows':3503,'unchanged_rows':11503,'raw_applied_parse_pass':3503,'raw_applied_parse_fail':0,'context_byte_exact':True,'prompt_ids_unchanged':True,'targets_exact_old_plus_ascii_brace':True,'r_version':rver,'artifacts':{'raw-parse-results.jsonl':{'bytes':result_path.stat().st_size,'sha256':sha(result_path)}}}
 (a.data/'independent-verification.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
