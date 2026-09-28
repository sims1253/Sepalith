#!/usr/bin/env python3
"""Correct only the admitted new-3503 finish frame in a fresh derived packet."""
from __future__ import annotations
from collections import Counter
import argparse,datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
from buffer_geometry import apply_region
from materialize_expanded_buffers import sha_file,require
BASE_SIDECAR_SHA='3a7bdab6e84bedb286ac6890d82bcb663aa1d51049d1a5c75d745b194ce2ccb2'
ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-token-rows.jsonl')
ROWS_SHA='7887686e022e520e746c04c6197a9a3c2484fba2668d09a650d0627ff267187b'
CONTEXTS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')
CONTEXTS_SHA='e499c07d6da2ef325f7d9f3156b6c2e70361d3b1bc140198b6c3563b4471e7a0'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 require(not a.output.exists(),'output_must_be_fresh');require(sha_file(a.base/'reward-buffer-sidecar.jsonl')==BASE_SIDECAR_SHA,'base_sidecar_hash')
 require(sha_file(ROWS)==ROWS_SHA and sha_file(CONTEXTS)==CONTEXTS_SHA,'new_finish_input_hash')
 rows={};contexts={}
 with ROWS.open() as f:
  for line in f:
   x=json.loads(line);rows[x['id']]=x
 with CONTEXTS.open() as f:
  for line in f:
   x=json.loads(line);contexts[x['row_id']]=x
 require(len(rows)==len(contexts)==3503 and set(rows)==set(contexts),'new_finish_join')
 attempt=a.output.with_name(a.output.name+f'.attempt-{os.getpid()}');attempt.mkdir();shutil.copytree(a.base/'baselines',attempt/'baselines',copy_function=os.link)
 parse_dir=attempt/'parse-temp';parse_dir.mkdir();changed=[];allrows=[]
 with (a.base/'reward-buffer-sidecar.jsonl').open() as f:
  for line in f:
   x=json.loads(line)
   if x['origin']=='dat10_new_3503':
    require(x['row_id'] in rows and x['family']=='finish_block','new_finish_identity')
    row=rows[x['row_id']];context=contexts[x['row_id']]['context'];baseline=(a.base/x['baseline_blob']).read_text()
    raw_gold=apply_region(baseline,x['replacement_range'],row['target_body_text'],context['document_eol'])
    require(hashlib.sha256(raw_gold.encode()).hexdigest()==x['gold_applied_sha256'],'raw_gold_hash')
    projection=raw_gold+'\n}';psha=hashlib.sha256(projection.encode()).hexdigest();path=parse_dir/f'{x["position"]}-{psha}.R';path.write_text(projection)
    x.update(buffer_mode='framed_fragment',diagnostic_suffix='\n}',parse_projection_sha256=psha,_parse_path=str(path))
    changed.append(x)
   allrows.append(x)
 paths=parse_dir/'paths.txt';results=parse_dir/'results.txt';paths.write_text(''.join(x['_parse_path']+'\n' for x in changed))
 run=subprocess.run(['Rscript','--vanilla',str(Path(__file__).with_name('parse_only.R')),str(paths),str(results)],timeout=600)
 require(run.returncode==0,'parse_harness');statuses=results.read_text().splitlines();require(len(statuses)==3503,'parse_count')
 byid={x['row_id']:ok=='1' for x,ok in zip(changed,statuses)};require(all(byid.values()),'framed_projection_failure')
 counts=Counter();families=Counter();lengths=Counter();supported=0
 side=attempt/'reward-buffer-sidecar.jsonl';repairs=attempt/'repair-ledger.jsonl'
 with side.open('w') as so,repairs.open('w') as ro:
  for x in allrows:
   if x['origin']=='dat10_new_3503':
    x.pop('_parse_path');x.update(supported=True,repair_reason=None,framed_projection_parse_ok=True)
   families[x['family']]+=1;lengths['gt_192_tokens' if x['target_tokens_including_protocol_eos']>192 else 'le_192_tokens']+=1
   if x['supported']:supported+=1;counts['supported']+=1;counts[f'mode_{x["buffer_mode"]}']+=1;counts[f'route_{x["source"]["kind"]}']+=1
   else:
    counts[x['repair_reason'].split(':',1)[0]]+=1;ro.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
   so.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
 shutil.rmtree(parse_dir)
 artifacts={n:{'path':str(a.output/n),'bytes':(attempt/n).stat().st_size,'sha256':sha_file(attempt/n)} for n in ['reward-buffer-sidecar.jsonl','repair-ledger.jsonl']}
 blobs=list((attempt/'baselines').rglob('*.R'))
 summary={'schema':'sepalith.rl11.expanded-buffer-materialization.v2','status':'prepared_root_review_required','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'derivation':{'base':str(a.base),'base_sidecar_sha256':BASE_SIDECAR_SHA,'correction':'all3503 dat10_new finish rows use provenance-authorized framed_fragment; targets unchanged'},
  'coverage':{'rows':len(allrows),'distinct_ids':len({x['row_id'] for x in allrows}),'supported':supported,'repair':len(allrows)-supported,
   'families':dict(sorted(families.items())),'counts':dict(sorted(counts.items())),'target_lengths':dict(sorted(lengths.items())),
   'targets_truncated':0,'empty_buffer_insertions':sum(x.get('baseline_empty') is True for x in allrows)},
  'blobs':{'count':len(blobs),'bytes':sum(x.stat().st_size for x in blobs),'content_addressed':True},'artifacts':artifacts,
  'parser':{'script_sha256':sha_file(Path(__file__).with_name('parse_only.R')),'generated_r_executed':False,'syntax_only':True},'launch_authorized':False}
 (attempt/'materialization.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');os.replace(attempt,a.output);print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
