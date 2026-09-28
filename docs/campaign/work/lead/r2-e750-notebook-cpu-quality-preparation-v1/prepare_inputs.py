#!/usr/bin/env python3
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import campaign_protocol as protocol
import native_transport as transport
CAPS=(192,384,768);CONTEXT=4096
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rows',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 if sha_file(a.rows)!='7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035':raise ValueError('corrected DEV75 bytes changed')
 tok,audit=transport._load_tokenizer(a.tokenizer);rows=[json.loads(x) for x in a.rows.read_text().splitlines()]
 if len(rows)!=75 or len({r['id'] for r in rows})!=75 or sum(r['operation']=='no_op' for r in rows)!=32 or any(r['split']!='dev' for r in rows):raise ValueError('DEV75 denominator changed')
 out=[];lengths=[]
 for r in rows:
  c=protocol.PromptContext.from_mapping(r['context']);text=protocol.render_prompt(c);ids=protocol.encode_prompt(c,tok,include_bos=True)
  if not ids or ids[0]!=0:raise ValueError('manual BOS mismatch:'+r['id'])
  row={'id':r['id'],'family':r['family'],'package_id':r['package_id'],'operation':r['operation'],'context':r['context'],'region_new':r['region_new'],'prompt_text':text,'prompt_sha256':sha_bytes(text.encode()),'prompt_ids':ids,'prompt_ids_sha256':sha_bytes(canonical(ids)),'prompt_tokens_including_bos':len(ids),'target_sha256':r['target_sha256'],'source_provenance':r['source_provenance']}
  row['cap_fit']={str(cap):len(ids)+cap<=CONTEXT for cap in CAPS};out.append(row);lengths.append(len(ids))
 a.out.mkdir(parents=True,exist_ok=False);panel=a.out/'dev75-tokenized.jsonl'
 with panel.open('x') as f:
  for r in out:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 over={str(cap):[r['id'] for r in out if not r['cap_fit'][str(cap)]] for cap in CAPS}
 man={'schema':'sepalith.run06.e750-notebook-dev75-tokenized.v1','status':'prepared_no_execution','rows':75,'edit_rows':43,'noop_rows':32,'case_ids':[r['id'] for r in out],'rows_source':str(a.rows.resolve()),'rows_sha256':sha_file(a.rows),'panel':str(panel.resolve()),'panel_sha256':sha_file(panel),'tokenizer':audit,'protocol_sha256':sha_file(Path(protocol.__file__)),'prompt_tokens_including_bos':{'min':min(lengths),'max':max(lengths),'sorted':sorted(lengths),'per_id':{r['id']:r['prompt_tokens_including_bos'] for r in out}},'cap_fit':{str(cap):{'fits':75-len(over[str(cap)]),'blocked':len(over[str(cap)]),'blocked_ids':over[str(cap)],'maximum_prompt_tokens':CONTEXT-cap} for cap in CAPS},'same_prompt_ids_across_caps':True,'target_not_used_for_cap_selection':True}
 (a.out/'manifest.json').write_text(json.dumps(man,indent=2)+'\n')
 print(json.dumps({'panel_sha256':man['panel_sha256'],'min':min(lengths),'max':max(lengths),'cap_fit':man['cap_fit']}))
if __name__=='__main__':main()
