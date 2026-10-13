import json,glob,hashlib,collections,os,time
BASE='/mnt/e/sepalith/campaign-20260915/data-work'
outdir=f'{BASE}/Noop-pool-coverage-review-v1'; os.makedirs(outdir,exist_ok=True)
curp=f'{BASE}/DAT10-expanded-15006-v1/combined-token-rows.jsonl'; gatep=f'{BASE}/DAT10-novel-v1/candidate-gate-v3/candidate-gate-ledger-v3.jsonl'; supp=f'{BASE}/DAT10-novel-v1/all-data-inventory-v1/support-review-noop-450-or-more.jsonl'
current={json.loads(l)['id'] for l in open(curp) if json.loads(l).get('family')=='no_op'}
gate={json.loads(l)['row_id'] for l in open(gatep) if json.loads(l).get('family')=='no_op'}
support={json.loads(l)['row_id'] for l in open(supp)}
files=sorted(glob.glob(f'{BASE}/DAT10-novel-v1/source-walk-shards-v1/shard-*/structured-materialization-v1/candidate-packets.jsonl'))
rows=[]; C=collections.Counter(); packs=set(); groups=set(); rowids=set(); raws=set(); chashes=set(); geom=set(); sourcefiles=set(); sources=set(); licensefiles=set(); invalid=[]; start=time.time(); dup=collections.Counter()
for p in files:
 shard=p.split('/shard-')[1].split('/')[0]
 with open(p) as f:
  for l in f:
   x=json.loads(l)
   if x.get('family')!='no_op': continue
   rr=x['row_ref']; res=x['result']; pro=res['provenance']; val=x['validation']; c=res['context']; rid=rr['row_id']
   cj=json.dumps(c,sort_keys=True,separators=(',',':')); ch=hashlib.sha256(cj.encode()).hexdigest()
   g=(val.get('source_sha256'),c.get('path'),pro.get('target_start_line'),pro.get('target_end_line'),pro.get('post_edit_snapshot_sha256'))
   checks={
    'status_converted':res.get('status')=='converted', 'operation_no_op':res.get('operation')=='no_op', 'target_matches_region_old':res.get('target_body')==c.get('region_old',[]),
    'full_buffer_application':val.get('full_buffer_application') is True, 'normalized_parent_R_parse':val.get('normalized_parent_R_parse') is True,
    'source_selection_support':val.get('source_selection_support_check') is True, 'before_after_identical':pro.get('before_snapshot_sha256')==pro.get('after_snapshot_sha256'),
    'post_edit_identity':pro.get('post_edit_snapshot_sha256')==pro.get('after_snapshot_sha256'), 'train_split':rr.get('split')=='train_group',
    'license_evidence':bool((val.get('license_evidence') or {}).get('sha256')),
   }
   if not all(checks.values()): invalid.append({'row_id':rid,'failed':[k for k,v in checks.items() if not v]})
   C['rows']+=1; C['suffix_empty' if not c.get('suffix_lines') else 'suffix_nonempty']+=1; C['prefix_empty' if not c.get('prefix') else 'prefix_nonempty']+=1; C[f"region_old_lines_{len(c.get('region_old',[]))}"]+=1; C[f"history_events_{len(c.get('history',[]))}"]+=1; C[f"eol_{c.get('document_eol')}"]+=1
   packs.add(rr.get('package_id')); groups.add(rr.get('group_id')); sourcefiles.add(val.get('source_path')); sources.add(val.get('source_sha256')); licensefiles.add((val.get('license_evidence') or {}).get('sha256'))
   if rid in rowids: dup['row_id']+=1
   if rr.get('raw_line_sha256') in raws: dup['raw_line_sha256']+=1
   if ch in chashes: dup['context_sha256']+=1
   if g in geom: dup['source_geometry']+=1
   rowids.add(rid); raws.add(rr.get('raw_line_sha256')); chashes.add(ch); geom.add(g)
   rows.append({'row_id':rid,'group_id':rr.get('group_id'),'package_id':rr.get('package_id'),'source_line':rr.get('line'),'raw_line_sha256':rr.get('raw_line_sha256'),'context_sha256':ch,'normalized_source_sha256':val.get('source_sha256'),'normalized_source_path':val.get('source_path'),'license_evidence_sha256':(val.get('license_evidence') or {}).get('sha256'),'snapshot_sha256':pro.get('post_edit_snapshot_sha256'),'target_start_line':pro.get('target_start_line'),'target_end_line':pro.get('target_end_line'),'placement':'terminal_empty_suffix' if not c.get('suffix_lines') else 'interior_nonempty_suffix','shard':int(shard)})
# metadata-only list
pout=outdir+'/supported-noop-candidates-metadata.jsonl'; tmp=pout+'.tmp'
with open(tmp,'w') as f:
 for x in sorted(rows,key=lambda z:z['row_id']): f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
os.replace(tmp,pout)
h=hashlib.sha256(open(pout,'rb').read()).hexdigest()
summary={'schema':'sepalith.dat10.noop-supported-candidates.v1','status':'review_only_not_admitted','elapsed_seconds':round(time.time()-start,3),'input_files':len(files),'input_bytes':sum(os.path.getsize(p) for p in files),'counts':dict(C),'unique':{'row_ids':len(rowids),'raw_line_sha256':len(raws),'contexts':len(chashes),'source_geometry':len(geom),'packages':len(packs),'groups':len(groups),'normalized_source_files':len(sourcefiles),'normalized_source_hashes':len(sources),'license_evidence_files':len(licensefiles)},'duplicates_within':dict(dup),'overlap':{'current15006_ids':len(rowids&current),'old_gate_ids':len(rowids&gate),'support_queue_ids':len(rowids&support)},'validation_failure_count':len(invalid),'validation_failures_first20':invalid[:20],'output':{'path':pout,'rows':len(rows),'bytes':os.path.getsize(pout),'sha256':h},'interpretation':{'placement':'terminal_empty_suffix vs interior_nonempty_suffix is derived only from frozen structural context; raw scenario kind/payload was not opened','prompt_dedup':'context hashes are structural evidence, not exact renderer prompt hashes; exact prompt hash/token dedup remains an admission gate'}}
sp=outdir+'/supported-noop-summary.json'; tmp=sp+'.tmp'; open(tmp,'w').write(json.dumps(summary,indent=2,sort_keys=True)+'\n');os.replace(tmp,sp)
print(json.dumps(summary,indent=2,sort_keys=True))
