#!/usr/bin/env python3
"""Stream the published 16K rows through the exact frozen CPT validator."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

def load(path):
 spec=importlib.util.spec_from_file_location('frontier_cpt_validator',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('xb') as f:f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--validator',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
 result=json.loads((a.output/'result.json').read_text());expected=result['outputs']['16384'];path=a.output/'cpt_train_ctx16384.jsonl';m=load(a.validator)
 started=time.monotonic();digest=hashlib.sha256();current=None;rows=[];seen=set();totals={'rows':0,'documents':0,'input_tokens':0,'payload_tokens':0,'supervised_tokens':0,'terminal_eos':0};max_len=0;multi=0
 def accept(group):
  nonlocal max_len,multi
  if not group:return
  ident=group[0]['document_id']
  if ident in seen:raise ValueError('duplicate output document')
  seen.add(ident);checked=m.validate_materialized_rows(group,max_sequence_tokens=16384,require_complete_documents=True)
  totals['rows']+=checked['rows'];totals['documents']+=checked['documents'];totals['input_tokens']+=checked['input_tokens'];totals['payload_tokens']+=checked['payload_tokens'];totals['supervised_tokens']+=checked['loss_tokens'];totals['terminal_eos']+=checked['documents']
  max_len=max(max_len,max(len(row['input_ids']) for row in group));multi+=int(len(group)>1)
 with path.open('rb') as f:
  for line_number,line in enumerate(f,1):
   digest.update(line);row=json.loads(line);ident=row.get('document_id')
   if current is None:current=ident
   if ident!=current:accept(rows);rows=[];current=ident
   rows.append(row)
 accept(rows)
 comparable={key:totals[key] for key in expected}
 if comparable!=expected:raise ValueError(f'validator totals differ: {comparable} != {expected}')
 actual={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
 if actual!=result['artifacts'][path.name]:raise ValueError('published JSONL artifact differs')
 provenance=a.output/'document-provenance.jsonl';prov={'bytes':provenance.stat().st_size,'sha256':sha(provenance)}
 if prov!=result['artifacts'][provenance.name]:raise ValueError('provenance artifact differs')
 evidence={'schema':'sepalith.cpt.16k-frontier-output-validation.v1','status':'complete','exact_frozen_validator_sha256':sha(a.validator),'output':actual,'provenance':prov,'totals':totals,'unique_documents':len(seen),'multi_chunk_documents':multi,'maximum_output_row_tokens':max_len,'elapsed_seconds':time.monotonic()-started,'all_rows_schema1_validator_passed':True,'complete_documents_required':True}
 write(a.evidence,evidence);print(json.dumps(evidence,sort_keys=True))
if __name__=='__main__':main()
