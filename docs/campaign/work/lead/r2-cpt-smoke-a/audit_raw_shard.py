from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,sys,datetime
P=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign');D=P/'work/r2-corpus-preparation-v1/profile-shard-v2-2k';W=Path(__file__).resolve().parent
manifest=json.loads((D/'manifest.json').read_text());out={};sets={};all_docs={}
for part in ('cpt_train','cpt_validation'):
 path=D/(part+'.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['artifacts'][path.name]['sha256']
 docs=defaultdict(list);groups=set();packages=set();counts=Counter();ids=set()
 for line in path.open():
  r=json.loads(line);assert r['schema']==1 and r['cpt_partition']==part
  assert r['row_id'] not in ids;ids.add(r['row_id']);groups.add(r['group_id']);packages.add(r['package'])
  a,l,m=r['input_ids'],r['labels'],r['attention_mask'];assert len(a)==len(l)==len(m) and 3<=len(a)<=2048 and set(m)=={1}
  assert all(type(x)is int and 0<=x<130560 for x in a) and a[0]==0 and a[-1]==1 and a.count(0)==a.count(1)==1
  overlap=r['overlap_context_tokens'];assert overlap==(1 if r['chunk_index'] else 0)
  start,end=r['source_token_start'],r['source_token_end'];assert start==r['token_start'] and end==r['token_end']
  assert end-start==len(a)-2-overlap and r['is_document_end']==(end==r['document_token_count'])
  expected=[-100]*(1+overlap)+a[1+overlap:-1]+([1] if r['is_document_end'] else [-100]);assert l==expected
  assert r['supervised_tokens']==sum(x!=-100 for x in l)==sum(x!=-100 for x in l[1:])
  assert r['document_id']==r['source_sha256'];docs[r['document_id']].append(r)
  counts.update(rows=1,input_tokens=len(a),supervised_tokens=r['supervised_tokens'],code_tokens=end-start)
 for doc,rows in docs.items():
  rows.sort(key=lambda r:r['chunk_index']);assert [r['chunk_index'] for r in rows]==list(range(len(rows)))
  cursor=0;tokens=[]
  for r in rows:
   assert r['source_token_start']==cursor
   if cursor:assert r['input_ids'][1]==tokens[-1]
   tokens.extend(r['input_ids'][1+r['overlap_context_tokens']:-1]);cursor=r['source_token_end']
  assert len(tokens)==cursor==rows[-1]['document_token_count'] and sum(r['is_document_end'] for r in rows)==1
  all_docs[(part,doc)]={'tokens':tokens,'first':rows[0]}
 counts.update(documents=len(docs),packages=len(packages));assert all(manifest['counts'][part][k]==v for k,v in counts.items())
 out[part]=dict(counts);sets[part]={'groups':groups,'packages':packages,'docs':set(docs)}
for k in ['groups','packages','docs']:assert not sets['cpt_train'][k]&sets['cpt_validation'][k]
# Independent direct source and pinned tokenizer checks on 3 whole documents.
from tokenizers import Tokenizer
T=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-native/tokenizer.json');assert hashlib.sha256(T.read_bytes()).hexdigest()=='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
tok=Tokenizer.from_file(str(T));tok.encode_special_tokens=True;checks=[]
for key in sorted(all_docs,key=lambda k:hashlib.sha256('|'.join(k).encode()).hexdigest())[:3]:
 d=all_docs[key];raw=Path(d['first']['source_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==key[1]
 text=raw.decode('utf-8');assert tok.encode(text,add_special_tokens=False).ids==d['tokens'];assert tok.decode(d['tokens'],skip_special_tokens=False)==text
 checks.append({'partition':key[0],'document_sha256':key[1],'code_tokens':len(d['tokens'])})
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'root_raw_profile_shard_CPU_audit_pass','counts':out,'CPT_groups_packages_documents_disjoint':True,'all_rows_geometry_labels_boundaries_checked':True,'all_documents_reconstructed_without_loss_or_double_supervision':True,'direct_source_tokenizer_replays':checks,'scope':'TRAIN-only partial inventory profiling shard; not whole corpus or final data; campaign heldout package map independently pinned via DAT02 worker inputs.'}
(W/'raw-shard-root-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
