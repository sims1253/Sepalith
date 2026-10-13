#!/usr/bin/env python3
"""Read-only census of TRAIN finish output/application boundaries."""
from __future__ import annotations
from collections import Counter
import hashlib, json
from pathlib import Path

ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
CONTEXT=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl')
BUFFER=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-buffer-v5/reward-buffer-sidecar.jsonl')
NEW_ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-token-rows.jsonl')
NEW_PROV=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')
EXPECTED={
 ROWS:'65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7',
 CONTEXT:'36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a',
 BUFFER:'126a9654b9d7d788c6cd60a4c90c9e0bf7f88fa718232be3f209473c94a81c4f',
 NEW_ROWS:'7887686e022e520e746c04c6197a9a3c2484fba2668d09a650d0627ff267187b',
 NEW_PROV:'e499c07d6da2ef325f7d9f3156b6c2e70361d3b1bc140198b6c3563b4471e7a0',
}
SOURCE_EVIDENCE={
 Path('docs/campaign/work/lead/r2-expanded-15008-v1/materialize.py'):'4516d89ab385b0d643cfcfe34af261e63cfbafe40ef6bf7a6103bd2ed43fcb7d',
 Path('docs/campaign/work/lead/r2-expanded-rl-buffer-preparation-v1/reframe_new_finish.py'):'311821be603baf8eecc5e83b9a30d9df03703844a03e9a3c4c9434922574774c',
 Path('docs/campaign/work/lead/r2-expanded-rl-reward-coverage-v2/source/experiments/training/campaign_reward_v2.py'):'b2ce979d8bc9a3112c7ae1810de8c271c180b3d916e98af0998c347bdf7669a7',
 Path('docs/campaign/work/lead/r2-expanded-rl-reward-coverage-v2/source/packages/sepalith/src/sepalith/campaign_protocol.py'):'5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156',
 Path('/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/cases/validators.py'):'5bccc747f6be4fc10131246fdf410ee3d2ee3b35cefd418cbda1db1e7f350626',
 Path('/mnt/h/sepalith/datasets/cases_v1/finish_block_compound.jsonl'):'a832e83153c92e41916159183dec3383d794bf993e5833c68faa5dd2a714c5cd',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p,key='id'):
 out={}
 with p.open() as f:
  for line in f:
   x=json.loads(line); k=x[key]
   assert k not in out
   out[k]=x
 return out
def utf16_units(s): return len(s.encode('utf-16-le'))//2
def main():
 for p,s in {**EXPECTED,**SOURCE_EVIDENCE}.items(): assert sha(p)==s,(p,sha(p))
 rows=load(ROWS); contexts=load(CONTEXT,'row_id'); buffers=load(BUFFER,'row_id')
 newrows=load(NEW_ROWS); newprov=load(NEW_PROV,'row_id')
 assert len(rows)==len(contexts)==15006 and set(rows)==set(contexts)
 assert len(newrows)==len(newprov)==3503 and set(newrows)==set(newprov) <= set(rows)
 finish={k:v for k,v in rows.items() if v['family']=='finish_block'}
 assert len(finish)==7785
 assert set(finish)<=set(buffers)
 counts=Counter(); modes=Counter(); endings=Counter(); suffix_lengths=Counter(); origins=Counter(); raw_parse=Counter(); frame_parse=Counter()
 samples=[]
 for rid,row in finish.items():
  c=contexts[rid]['context']; b=buffers[rid]; origin='dat10_new_3503' if rid in newrows else 'accepted_11505'
  origins[origin]+=1; modes[(origin,b['buffer_mode'])]+=1; suffix_lengths[(origin,len(c['suffix_lines']))]+=1
  body=row['target_body_text']; endings[(origin,'rstrip_brace' if body.rstrip().endswith('}') else 'not_rstrip_brace')]+=1
  raw_parse[(origin,str(b.get('gold_applied_parse_ok')))]+=1
  frame_parse[(origin,str(b.get('framed_projection_parse_ok')))]+=1
  if len(c['region_old'])==0: counts[(origin,'empty_region')]+=1
  if c['replacement_range']['start']==c['replacement_range']['end']: counts[(origin,'zero_width_range')]+=1
  if not c['suffix_lines']: counts[(origin,'empty_suffix')]+=1
  if row['input_ids'][-1]==1: counts[(origin,'terminal_eos')]+=1
  if origin=='dat10_new_3503':
   p=newprov[rid]['source_provenance']; sel=newprov[rid]['selection']
   assert p['source_identity']['split']=='train_group'
   assert p['target_convention']=='suffix'
   assert p['finish_splice']=={'literal_source_splice_verified':True,'outer_closing_brace_in_label':False}
   assert p['r_fragment']['body_fragment_boundary']=='raw_prefix_plus_corpus_target_before_outer_brace'
   assert sel['suffix']==[] and c['suffix_lines']==[]
   end=c['replacement_range']['end']
   if c['region_old']:
    assert end=={'line':len(c['prefix'])+len(c['region_old'])-1,'character':utf16_units(c['region_old'][-1])}
   else:
    assert end=={'line':len(c['prefix']),'character':0}
   counts[(origin,'range_ends_at_represented_document_eof')]+=1
   counts[(origin,'provenance_outer_brace_excluded')]+=1
   counts[(origin,'source_fragment_validator_passed')]+=int(p['r_fragment']['passed'] is True)
   counts[(origin,'unsupported_zero_width_primary_route_named')]+=int(p['unsupported_route_candidate']['status']=='unsupported_primary_route')
   if len(samples)<8: samples.append({'row_id':rid,'source_line':p['source_identity']['line'],'source_raw_line_sha256':p['source_identity']['raw_line_sha256'],'target_body_sha256':p['target_body_sha256'],'target_tail':repr(body[-40:]),'prefix_tail':c['prefix'][-2:],'suffix_lines':c['suffix_lines'],'buffer_mode':b['buffer_mode'],'diagnostic_suffix':b['diagnostic_suffix'],'gold_applied_parse_ok':b['gold_applied_parse_ok'],'framed_projection_parse_ok':b['framed_projection_parse_ok']})
 result={
  'schema':'sepalith.dat10.train-finish-boundary-audit.v1','status':'audit_complete_root_decision_required','scope':'TRAIN only; no DEV/final rows or targets read',
  'input_pins':{str(p):s for p,s in EXPECTED.items()},
  'source_evidence_pins':{str(p):s for p,s in SOURCE_EVIDENCE.items()},
  'denominators':{'eligible_train_rows':15006,'finish_rows':len(finish),'accepted_11505_finish_rows':origins['accepted_11505'],'new_finish_rows':origins['dat10_new_3503']},
  'census':{'buffer_modes':{f'{a}:{b}':n for (a,b),n in sorted(modes.items())},'context_suffix_line_counts':{f'{a}:{b}':n for (a,b),n in sorted(suffix_lengths.items())},'target_endings':{f'{a}:{b}':n for (a,b),n in sorted(endings.items())},'existing_gold_applied_parse':{f'{a}:{b}':n for (a,b),n in sorted(raw_parse.items())},'existing_framed_projection_parse':{f'{a}:{b}':n for (a,b),n in sorted(frame_parse.items())},'geometry':{f'{a}:{b}':n for (a,b),n in sorted(counts.items())}},
  'new3503_source_facts':{
   'all_source_split_train_group':True,'all_target_convention_suffix':True,'all_literal_source_splice_verified':True,
   'all_provenance_says_outer_closing_brace_not_in_label':True,'all_fragment_boundary_says_before_outer_brace':True,
   'all_context_suffixes_empty':True,'all_context_ranges_end_at_represented_document_eof':True,
   'zero_width_empty_region_rows':counts[('dat10_new_3503','zero_width_range')],
   'nonempty_region_replacement_rows':3503-counts[('dat10_new_3503','zero_width_range')],
   'inference_prompt_distinguishes_intentionally_partial_from_complete_replacement':False,
  },
  'contract_analysis':{
   'generated_body_application':'Protocol replace body is applied exactly to replacement_range; diagnostic_suffix is not part of generation or edit application.',
   'framed_parse_scope':'PRM03 appends diagnostic_suffix only to syntax-check the applied candidate; exact-region scoring still compares candidate body to the brace-omitting target.',
   'finding':'The 3503 labels are source-authentic corpus fragments but are not complete edits under the represented prompt/context. Empty suffix plus EOF insertion leaves the represented outer function unclosed. The prompt asks for the complete replacement region and exposes no partial-completion marker.',
   'severity':'training/evaluation/serving geometry mismatch for all 3503 new finish rows',
  },
  'source_supported_options':[
   {'option':'append_proven_source_outer_brace_to_target','support':'provenance explicitly locates corpus target before one outer brace; constructor/source replay must independently recover and hash that exact brace','effects':'prompt/range unchanged; target/token hashes change; applied document becomes closed; reward exact target aligns with serving'},
   {'option':'represent_outer_brace_as_immutable_suffix','support':'only if a fresh pre-edit document is source-reconstructed with the proven brace and replacement range moves before it','effects':'target fragment can remain byte-exact; prompt/context/range hashes change; suffix makes partial-body boundary visible to inference'},
   {'option':'exclude_until_rebuilt','support':'strong known contract inconsistency','effects':'preserves original artifacts and avoids training on invalid applied edits; conflicts with all-data policy only until source-supported repair is admitted'},
  ],
  'not_supported':'Do not copy the parser diagnostic suffix into labels without source replay; do not infer arbitrary braces from parse failure; do not claim framed parse success means the original edit is complete.',
  'samples':samples,
 }
 out=Path(__file__).with_name('audit-result.json'); out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'status':'pass','finish_rows':len(finish),'new_finish':origins['dat10_new_3503'],'finding':result['contract_analysis']['severity']},sort_keys=True))
if __name__=='__main__':main()
