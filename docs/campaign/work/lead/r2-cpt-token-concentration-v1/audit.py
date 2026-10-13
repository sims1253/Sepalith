import collections,datetime,hashlib,json,os
from pathlib import Path
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.nice(10)
root=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1'); groups=sorted(root.joinpath('groups').iterdir());packages=collections.Counter();docs=[];receipts=[]
for group in groups:
 r=json.loads((group/'receipt.json').read_text());path=group/'documents.jsonl';raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['artifacts']['documents.jsonl']['sha256']
 local=[json.loads(x) for x in raw.splitlines() if x];assert sum(x['source_code_tokens'] for x in local)==r['counts'].get('payload_tokens',0);assert all(x['split']=='train_group' and x['cpt_partition']=='cpt_train' for x in local)
 packages[r['package']]+=r['counts'].get('payload_tokens',0);docs.extend({k:x[k] for k in ('path','package','sha256','source_code_tokens','source_utf8_bytes')} for x in local);receipts.append({'index':r['seeded_index'],'sha256':hashlib.sha256((group/'receipt.json').read_bytes()).hexdigest()})
total=sum(packages.values());top=sorted(docs,key=lambda x:x['source_code_tokens'],reverse=True)[:30]
for row in top:
 with Path(row['path']).open('rb') as stream:prefix=stream.read(512)
 row['first512_sha256']=hashlib.sha256(prefix).hexdigest();row['generated_header_observed']=b'generated' in prefix.lower();row['token_share']=row['source_code_tokens']/total
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'immutable committed TRAIN groups observed at directory capture; growing traversal, not complete corpus','groups':len(groups),'documents':len(docs),'payload_tokens':total,'top_packages':[{'package':name,'tokens':n,'share':n/total} for name,n in packages.most_common(30)],'top_documents':top,'receipt_identity_digest':hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest(),'policy':'No exclusions or weights changed. Concentrated generated source is a data-quality review item, not an automatic exclusion. All eligible materialization continues.'}
Path(__file__).with_name('report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'groups':len(groups),'payload_tokens':total,'largest_package':packages.most_common(1),'largest_document_tokens':top[0]['source_code_tokens']}))
