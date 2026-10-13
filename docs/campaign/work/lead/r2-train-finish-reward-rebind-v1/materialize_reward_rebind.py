#!/usr/bin/env python3
"""Rebind RL reward evidence and signal-pilot identities to repair v3."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from campaign_reward_v2 import apply_region
OLD=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-buffer-v5')
REPAIR=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3')
CONTEXT=REPAIR/'context-sidecar.jsonl';ROWS=REPAIR/'train-token-rows.jsonl';LEDGER=REPAIR/'repair-ledger.jsonl';RAW_PARSE=REPAIR/'raw-parse-results.jsonl'
SELECTED=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/selected-train-ids.json')
OLD_SPEC=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-rl15006-signal-pilot-v1/pilot-spec.json')
OLD_GROUPS=OLD_SPEC.with_name('pilot-groups.jsonl')
PINS={OLD/'reward-buffer-sidecar.jsonl':'126a9654b9d7d788c6cd60a4c90c9e0bf7f88fa718232be3f209473c94a81c4f',OLD/'materialization.json':'a49b1ef464773817d7a18f76cdec68b82eaa4abb960a10a7d919c7535d7add84',ROWS:'3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e',CONTEXT:'36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a',LEDGER:'254d6ef418edec64c79047f18a546dc1af112a84e709b2fe6233d2453bc33cfa',RAW_PARSE:'61d34915138c7e9adcda03f2d23ff4ab287a3efe7e500b4226aa83b0901605c8',REPAIR/'materialization.json':'8b90202ae438b7c048f4a80ed5cf13fd2cd635d719799e13959f8cf92641f47c',SELECTED:'986a1f7910c44423e50988e76abb6671dedf8cd70f5542ff9efd7a03ce504462',OLD_SPEC:'df176d826901f80ccdb2ef485069fb5e27482eb53da549ff8641b0a59850df0b',OLD_GROUPS:'b2bcb6a3eb08d41bdb46b78e75c5772d321692073719187461c8f9c484699a04',HERE/'campaign_reward_v2.py':'b2ce979d8bc9a3112c7ae1810de8c271c180b3d916e98af0998c347bdf7669a7'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def load(p,key):
 out={}
 with p.open() as f:
  for line in f:
   x=json.loads(line);assert x[key] not in out;out[x[key]]=x
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--pilot-output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists() and not a.pilot_output.exists()
 for p,s in PINS.items():assert sha(p)==s,(p,sha(p))
 rows=load(ROWS,'id');contexts=load(CONTEXT,'row_id');ledger=load(LEDGER,'row_id');rawparse=load(RAW_PARSE,'row_id')
 assert len(rows)==len(contexts)==15006 and len(ledger)==len(rawparse)==3503 and set(ledger)==set(rawparse)<set(rows)
 selected=json.loads(SELECTED.read_text())['row_ids'];assert len(selected)==len(set(selected))==15006 and selected==list(rows)
 attempt=a.output.with_name(a.output.name+f'.attempt-{os.getpid()}');attempt.mkdir(parents=True);(attempt/'baselines').mkdir()
 # Preserve every content-addressed baseline byte. Hard links are immutable aliases on this E: volume; fall back to copying.
 link_count=copy_count=0
 for src in sorted((OLD/'baselines').rglob('*.R')):
  dst=attempt/'baselines'/src.relative_to(OLD/'baselines');dst.parent.mkdir(parents=True,exist_ok=True)
  try:os.link(src,dst);link_count+=1
  except OSError:shutil.copy2(src,dst);copy_count+=1
 side=attempt/'reward-buffer-sidecar.jsonl';change=attempt/'rebind-ledger.jsonl';changed=unchanged=held=0
 with (OLD/'reward-buffer-sidecar.jsonl').open() as inf,side.open('w') as out,change.open('w') as led:
  for line in inf:
   old=json.loads(line);rid=old['row_id']
   if rid not in ledger:
    out.write(line);unchanged+=1;held+=int(rid not in rows);continue
   row=rows[rid];ctx=contexts[rid]['context'];base=(attempt/old['baseline_blob']).read_text();post=apply_region(base,ctx['replacement_range'],row['target_body_text'],ctx['document_eol'])
   postsha=hashlib.sha256(post.encode()).hexdigest();assert postsha==rawparse[rid]['applied_document_sha256'] and rawparse[rid]['raw_applied_parse_ok'] is True
   new=dict(old);new.update(buffer_mode='completion_prefix',diagnostic_suffix='',framed_projection_parse_ok=None,parse_projection_sha256=postsha,gold_applied_sha256=postsha,gold_applied_bytes=len(post.encode()),gold_applied_parse_ok=True,target_body_sha256=hashlib.sha256(row['target_body_text'].encode()).hexdigest(),target_body_bytes=len(row['target_body_text'].encode()),target_body_lines=len(row['target_body_text'].split('\n')),target_tokens_including_protocol_eos=len(row['input_ids'])-row['target_start'],target_truncated=False,supported=True,repair_reason=None)
   assert new['replacement_range']==old['replacement_range'] and new['baseline_sha256']==old['baseline_sha256']
   delta={'row_id':rid,'position':old['position'],'old_record_sha256':hashlib.sha256(line.encode()).hexdigest(),'new_record_sha256':hashlib.sha256((canon(new)+'\n').encode()).hexdigest(),'old_target_body_sha256':old['target_body_sha256'],'new_target_body_sha256':new['target_body_sha256'],'old_gold_applied_sha256':old['gold_applied_sha256'],'new_gold_applied_sha256':postsha,'old_buffer_mode':'framed_fragment','new_buffer_mode':'completion_prefix','old_diagnostic_suffix':'\n}','new_diagnostic_suffix':'','raw_applied_parse_ok':True,'parent_historical_provenance_only':True}
   out.write(canon(new)+'\n');led.write(canon(delta)+'\n');changed+=1
 assert changed==3503 and unchanged==11505 and held==2
 artifacts={n:{'path':str(a.output/n),'bytes':(attempt/n).stat().st_size,'sha256':sha(attempt/n)} for n in ['reward-buffer-sidecar.jsonl','rebind-ledger.jsonl']}
 manifest={'schema':'sepalith.rl11.finish-repaired-buffer-materialization.v1','status':'prepared_root_review_required','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':{str(p):s for p,s in PINS.items()},'coverage':{'source_rows':15008,'eligible_rows':15006,'held_contradictory_rows':2,'changed_repaired_finish_rows':3503,'unchanged_records_byte_exact':11505,'framed_fragment_rows_removed':3503,'raw_gold_applied_parse_pass':3503},'buffer_policy':{'repaired_mode':'completion_prefix','reason':'pre-edit prefix may be incomplete; repaired gold application is a complete raw parse','diagnostic_suffix':'','old_framed_evidence_reusable':False},'baselines':{'files':link_count+copy_count,'hardlinked':link_count,'copied':copy_count,'content_addressed_immutable':True},'artifacts':artifacts,'launch_authorized':False}
 (attempt/'materialization.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');os.replace(attempt,a.output)
 # Rebind the fixed TRAIN pilot without changing prompt selection or model placeholder.
 a.pilot_output.mkdir(parents=True);groups=[]
 with OLD_GROUPS.open() as f:
  for line in f:
   g=json.loads(line);g['target_tokens']=len(rows[g['row_id']]['input_ids'])-rows[g['row_id']]['target_start'];groups.append(g)
 gp=a.pilot_output/'pilot-groups.jsonl';gp.write_text(''.join(canon(x)+'\n' for x in groups))
 spec=json.loads(OLD_SPEC.read_text());spec['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();spec['status']='prepared_repaired_pool_model_binding_and_root_admission_required';spec['pool']['inputs']['rows']={'path':str(ROWS),'sha256':PINS[ROWS]};spec['pool']['inputs']['buffer']={'path':str(a.output/'reward-buffer-sidecar.jsonl'),'sha256':sha(a.output/'reward-buffer-sidecar.jsonl')};spec['pool']['inputs']['buffer_manifest']={'path':str(a.output/'materialization.json'),'sha256':sha(a.output/'materialization.json')};spec['selection']['groups_path']=str(gp);spec['selection']['groups_sha256']=sha(gp);spec['estimates']['gold_target_token_presentations_for_reference_only']=sum(x['target_tokens'] for x in groups)*4;spec['repair_binding']={'rows_sha256':PINS[ROWS],'reward_buffer_sha256':sha(a.output/'reward-buffer-sidecar.jsonl'),'changed_finish_rows':3503,'framed_fragment_rows_for_repaired_finish':0,'diagnostic_suffix_for_repaired_finish':''}
 sp=a.pilot_output/'pilot-spec.json';sp.write_text(json.dumps(spec,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'status':'prepared','buffer_manifest_sha256':sha(a.output/'materialization.json'),'buffer_sha256':sha(side if side.exists() else a.output/'reward-buffer-sidecar.jsonl'),'pilot_groups_sha256':sha(gp),'pilot_spec_sha256':sha(sp),'changed':changed},sort_keys=True))
if __name__=='__main__':main()
