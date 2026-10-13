#!/usr/bin/env python3
"""Materialize immutable applied-buffer evidence for ordered expanded TRAIN rows.

The input spec is segment based so later root-admitted increments can be appended
without changing the reconstruction rules. Generated R is only passed to the
fixed base::parse harness; it is never sourced, evaluated, or executed.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
import argparse, datetime, hashlib, json, os, shutil, subprocess
from pathlib import Path
from typing import Any, Mapping
from buffer_geometry import apply_region, sha_text

class PrepError(ValueError): pass

def require(ok: bool, why: str) -> None:
    if not ok: raise PrepError(why)

def canonical(x: Any) -> bytes:
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()

def sha_file(path: Path) -> str:
    d=hashlib.sha256()
    with path.open('rb',buffering=4*1024*1024) as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest()

class JsonlStream:
    def __init__(self,path:Path,expected:str):
        self.path=path; self.expected=expected; self.f=path.open('rb',buffering=4*1024*1024)
        self.digest=hashlib.sha256(); self.rows=0
    def next(self)->tuple[bytes,Mapping[str,Any]]:
        while True:
            raw=self.f.readline()
            if not raw: raise StopIteration
            self.digest.update(raw)
            if raw.strip():
                self.rows+=1; value=json.loads(raw)
                require(isinstance(value,Mapping),f'jsonl_object_required:{self.path}:{self.rows}')
                return raw,value
    def finish(self)->None:
        try: self.next()
        except StopIteration: pass
        else: raise PrepError(f'extra_row:{self.path}')
        self.f.close(); require(self.digest.hexdigest()==self.expected,f'input_hash_mismatch:{self.path}')

class SourceCache:
    def __init__(self,limit:int=32*1024*1024):
        self.limit=limit;self.size=0;self.items:OrderedDict[str,bytes]=OrderedDict();self.identities={}
    def get(self,path:Path,expected:str)->bytes:
        key=str(path)
        prior=self.identities.get(key)
        require(prior in {None,expected},f'conflicting_source_pin:{path}')
        self.identities[key]=expected
        if key in self.items:
            raw=self.items.pop(key);self.items[key]=raw;return raw
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,f'source_hash_mismatch:{path}')
        if len(raw)<=self.limit:
            while self.items and self.size+len(raw)>self.limit:
                _,old=self.items.popitem(last=False);self.size-=len(old)
            self.items[key]=raw;self.size+=len(raw)
        return raw

def source_parts(record:Mapping[str,Any],fmt:str)->tuple[Mapping[str,Any],Mapping[str,Any]|None]:
    if fmt=='rl_context_sidecar':
        ident=record['source_identity'];return ident['source_provenance'],ident.get('source_ref')
    if fmt=='candidate_context_provenance':return record['source_provenance'],record.get('source_ref')
    raise PrepError(f'unsupported_context_format:{fmt}')

def utf16_offset(text:str,pos:Mapping[str,Any])->int:
    # Reuse the production geometry implementation by applying an empty insertion
    # marker and locating it would be ambiguous. This exact helper is for history.
    from buffer_geometry import position_offset
    return position_offset(text,int(pos['line']),int(pos['character']))

def replay_history(text:str,context:Mapping[str,Any])->str:
    eol='\r\n' if context['document_eol']=='crlf' else '\n'
    for i,event in enumerate(context.get('history',[])):
        rr=event['range_utf16'];require(sha_text(text)==rr['content_sha256'],f'history_predecessor_hash:{i}')
        left,right=utf16_offset(text,rr['start']),utf16_offset(text,rr['end'])
        old=str(event['old_text']).replace('\n',eol);new=str(event['new_text']).replace('\n',eol)
        require(text[left:right]==old,f'history_old_text:{i}')
        text=text[:left]+new+text[right:]
    return text

def reconstruct(record:Mapping[str,Any],fmt:str,cache:SourceCache)->tuple[str,str,str,dict[str,Any]]:
    context=record['context'];sp,sref=source_parts(record,fmt)
    selection_geometry=record.get('selection_geometry') or record.get('selection') or {}
    expected_context=context['replacement_range']['content_sha256']
    text=None;route=None
    if selection_geometry.get('availability')=='full_snapshot':
        # Some admitted full snapshots fit wholly in the prompt selection. Use
        # that representation only when its bytes prove the context hash.
        eol='\r\n' if context['document_eol']=='crlf' else '\n'
        candidate=eol.join(list(context['prefix'])+list(context['region_old'])+list(context['suffix_lines']))
        if sha_text(candidate)==expected_context:
            text=candidate;route={'kind':'admitted_context_full_snapshot','source_identity_sha256':hashlib.sha256(canonical(sp)).hexdigest(),
                                  'history_events':len(context.get('history',[]))}
    selection=sp.get('selection_source')
    if text is None and isinstance(selection,Mapping) and isinstance(selection.get('document_text'),str):
        candidate=selection['document_text'];expected=selection.get('content_sha256') or sp.get('pre_edit_document',{}).get('content_sha256')
        require(isinstance(expected,str) and sha_text(candidate)==expected,'builder_document_hash_mismatch')
        text=candidate;route={'kind':'pinned_builder_document','source_path':selection.get('source_jsonl_path') or sp.get('source_path'),
               'source_sha256':selection.get('source_jsonl_sha256') or sp.get('source_sha256'),
               'source_line':selection.get('source_jsonl_line'),'raw_line_sha256':selection.get('source_jsonl_raw_line_sha256'),
               'document_sha256':expected,'history_events':len(context.get('history',[]))}
    if text is None and sp.get('source_snapshot_path') and sp.get('source_snapshot_sha256'):
        path=Path(sp['source_snapshot_path']);expected=sp['source_snapshot_sha256']
        require(path.is_file(),f'pinned_snapshot_unavailable:{path}')
        raw=cache.get(path,expected);text=replay_history(raw.decode('utf-8'),context)
        route={'kind':'pinned_source_snapshot_plus_history','source_path':str(path),'source_sha256':expected,
               'source_bytes':len(raw),'history_events':len(context.get('history',[]))}
    require(text is not None and route is not None,'no_complete_or_prefix_buffer_materialized')
    rr=context['replacement_range'];require(sha_text(text)==expected_context,'reconstructed_context_hash_mismatch')
    if sref:
        route['source_ref_sha256']=hashlib.sha256(canonical(sref)).hexdigest()
        route['source_ref_row_id']=sref.get('row_id');route['source_ref_split']=sref.get('split')
    mode='completion_prefix' if record.get('family')=='finish_block' or (sref and sref.get('family')=='finish_block') else 'complete_document'
    diagnostic_suffix=''
    fragment=sp.get('r_fragment',{})
    if (mode=='completion_prefix' and fmt=='candidate_context_provenance'
            and fragment.get('outer_closing_brace_in_label') is False
            and fragment.get('body_fragment_boundary')=='raw_prefix_plus_corpus_target_before_outer_brace'):
        require(fragment.get('passed') is True,'finish_fragment_not_validated')
        diagnostic_suffix='\n}'
        mode='framed_fragment'
    return text,mode,diagnostic_suffix,route

def write_blob(root:Path,text:str)->tuple[str,str,int]:
    raw=text.encode();sha=hashlib.sha256(raw).hexdigest();rel=f'baselines/{sha[:2]}/{sha}.R';p=root/rel
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): require(p.read_bytes()==raw,f'blob_collision:{sha}')
    else:
        tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);os.replace(tmp,p)
    return sha,rel,len(raw)

def parse_paths(script:Path,paths:list[Path],work:Path)->dict[str,bool]:
    unique=[];seen=set()
    for p in paths:
        s=str(p)
        if s not in seen:seen.add(s);unique.append(p)
    manifest=work/'parse-paths.txt';result=work/'parse-results.txt'
    manifest.write_text(''.join(str(p)+'\n' for p in unique))
    run=subprocess.run(['Rscript','--vanilla',str(script),str(manifest),str(result)],stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,text=True,timeout=1200)
    require(run.returncode==0,f'parse_harness_failed:{run.returncode}:{run.stderr[-500:]}')
    statuses=result.read_text().splitlines();require(len(statuses)==len(unique),'parse_result_count_mismatch')
    require(all(x in {'0','1'} for x in statuses),'parse_result_invalid')
    return {str(p):x=='1' for p,x in zip(unique,statuses)}

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();require(not args.output.exists(),'output_must_be_fresh')
    spec=json.loads(args.spec.read_text());plan=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
    adm=json.loads((plan/spec['admission']['path']).read_text());require(adm['status']==spec['admission']['status'],'admission_status')
    cm=Path(spec['combined_manifest']['path']);require(sha_file(cm)==spec['combined_manifest']['sha256'],'combined_manifest_hash')
    manifest=json.loads(cm.read_text());require(manifest['counts']['combined_rows']==sum(x['rows'] for x in spec['segments']),'segment_total')
    attempt=args.output.with_name(args.output.name+f'.attempt-{os.getpid()}');require(not attempt.exists(),'attempt_exists')
    attempt.mkdir(parents=True);(attempt/'gold-temporary').mkdir();(attempt/'parse-work').mkdir()
    rows=JsonlStream(Path(spec['rows']['path']),spec['rows']['sha256']);origins=JsonlStream(Path(spec['origin']['path']),spec['origin']['sha256'])
    segments={x['origin']:{'spec':x,'stream':JsonlStream(Path(x['path']),x['sha256']),'seen':0} for x in spec['segments']}
    preliminary=attempt/'evidence.preliminary.jsonl';repair_pre=attempt/'repairs.preliminary.jsonl';cache=SourceCache()
    counts=Counter();families=Counter();target_lengths=Counter();baseline_paths=[];gold_paths=[]
    with preliminary.open('w') as out,repair_pre.open('w') as repairs:
      for position in range(manifest['counts']['combined_rows']):
        try:_,row=rows.next();_,origin=origins.next()
        except StopIteration:raise PrepError(f'combined_input_short:{position}')
        require(origin['position']==position and origin['row_id']==row['id'],f'row_origin_mismatch:{position}')
        label=origin['origin'];require(label in segments,f'unknown_origin:{label}')
        segment=segments[label];_,context_record=segment['stream'].next();segment['seen']+=1
        require(context_record['row_id']==row['id'],f'context_order_mismatch:{position}')
        require(row['split']=='train','non_train_row');rid=row['id'];family=row['family'];families[family]+=1
        body=row['target_body_text'];target_bytes=len(body.encode());target_lines=0 if body=='' else body.count('\n')+1
        target_tokens=len(row['input_ids'])-row['target_start'];target_lengths['gt_192_tokens' if target_tokens>192 else 'le_192_tokens']+=1
        common={'schema':'sepalith.rl11.expanded-buffer-evidence-row.v1','position':position,'row_id':rid,'origin':label,
                'family':family,'package_id':row['package_id'],'split':'train','context_record_sha256':hashlib.sha256(canonical(context_record)).hexdigest(),
                'replacement_range':context_record['context']['replacement_range'],'target_operation':row['target_operation'],
                'target_body_sha256':sha_text(body),'target_body_bytes':target_bytes,'target_body_lines':target_lines,
                'target_tokens_including_protocol_eos':target_tokens,'target_truncated':False}
        try:
          baseline,mode,diagnostic_suffix,route=reconstruct(context_record,segment['spec']['format'],cache)
          bsha,brel,bbytes=write_blob(attempt,baseline);bpath=attempt/brel
          operation=row['target_operation'];applied_body=baseline if False else ('' if operation in {'no_op','delete'} else body)
          gold=baseline if operation=='no_op' else apply_region(baseline,context_record['context']['replacement_range'],applied_body,context_record['context']['document_eol'])
          graw=gold.encode();gsha=hashlib.sha256(graw).hexdigest()
          projection=(gold+diagnostic_suffix).encode();projection_sha=hashlib.sha256(projection).hexdigest()
          gpath=attempt/'gold-temporary'/f'{position:05d}-raw-{gsha}.R';gpath.write_bytes(graw)
          ppath=attempt/'gold-temporary'/f'{position:05d}-projection-{projection_sha}.R';ppath.write_bytes(projection)
          common.update({'supported':True,'repair_reason':None,'buffer_mode':mode,'baseline_sha256':bsha,
                         'baseline_bytes':bbytes,'baseline_blob':brel,'gold_applied_sha256':gsha,'gold_applied_bytes':len(graw),
                         'parse_projection_sha256':projection_sha,'diagnostic_suffix':diagnostic_suffix,
                         'source':route,'baseline_empty':baseline==''})
          baseline_paths.append(bpath);gold_paths.extend([gpath,ppath]);counts[f'route_{route["kind"]}']+=1;counts[f'mode_{mode}']+=1
          if baseline=='':counts['empty_baseline']+=1
        except Exception as e:
          common.update({'supported':False,'repair_reason':f'{type(e).__name__}:{e}','buffer_mode':None})
          repairs.write(json.dumps(common,sort_keys=True,separators=(',',':'))+'\n');counts['reconstruction_repair']+=1
        out.write(json.dumps(common,sort_keys=True,separators=(',',':'))+'\n')
    rows.finish();origins.finish()
    for label,x in segments.items():
        require(x['seen']==x['spec']['rows'],f'segment_count:{label}');x['stream'].finish()
    parser=Path(__file__).with_name('parse_only.R');statuses=parse_paths(parser,baseline_paths+gold_paths,attempt/'parse-work')
    final=attempt/'reward-buffer-sidecar.jsonl';repair_final=attempt/'repair-ledger.jsonl';supported=0
    with preliminary.open() as src,final.open('w') as dst,repair_final.open('w') as repair:
      for line in src:
        x=json.loads(line)
        if x['supported']:
          bp=attempt/x['baseline_blob']
          gp=attempt/'gold-temporary'/f'{x["position"]:05d}-raw-{x["gold_applied_sha256"]}.R'
          pp=attempt/'gold-temporary'/f'{x["position"]:05d}-projection-{x["parse_projection_sha256"]}.R'
          x['baseline_parse_ok']=statuses[str(bp)];x['gold_applied_parse_ok']=statuses[str(gp)]
          x['framed_projection_parse_ok']=statuses[str(pp)] if x['buffer_mode']=='framed_fragment' else None
          x['parser_identity']={'command':'Rscript --vanilla parse_only.R','operation':'base::parse(file=...,keep.source=FALSE)',
                                'r_version':'R 4.6.1 (2026-06-24)','generated_r_executed':False}
          reason=None
          if x['buffer_mode']=='complete_document' and not x['baseline_parse_ok']:reason='complete_baseline_parse_failed'
          elif x['buffer_mode']=='framed_fragment' and not x['framed_projection_parse_ok']:reason='framed_gold_projection_parse_failed'
          elif x['buffer_mode']!='framed_fragment' and not x['gold_applied_parse_ok']:reason='gold_applied_parse_failed'
          if reason:
            x['supported']=False;x['repair_reason']=reason;counts[reason]+=1;repair.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
          else:supported+=1;counts['supported']+=1
        else:repair.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
        dst.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
    shutil.rmtree(attempt/'gold-temporary');shutil.rmtree(attempt/'parse-work');preliminary.unlink();repair_pre.unlink()
    artifacts={}
    for name in ['reward-buffer-sidecar.jsonl','repair-ledger.jsonl']:
      p=attempt/name;artifacts[name]={'path':str(args.output/name),'bytes':p.stat().st_size,'sha256':sha_file(p)}
    blob_files=list((attempt/'baselines').rglob('*.R')) if (attempt/'baselines').exists() else []
    blob_total=sum(p.stat().st_size for p in blob_files)
    summary={'schema':'sepalith.rl11.expanded-buffer-materialization.v1','status':'prepared_root_review_required',
      'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_spec_sha256':sha_file(args.spec),
      'coverage':{'rows':manifest['counts']['combined_rows'],'supported':supported,'repair':manifest['counts']['combined_rows']-supported,
                  'distinct_ids':manifest['counts']['combined_rows'],'families':dict(sorted(families.items())),
                  'counts':dict(sorted(counts.items())),'target_lengths':dict(sorted(target_lengths.items())),
                  'targets_truncated':0,'empty_buffer_insertions':counts['empty_baseline']},
      'blobs':{'count':len(blob_files),'bytes':blob_total,'content_addressed':True},'artifacts':artifacts,
      'parser':{'script_sha256':sha_file(parser),'generated_r_executed':False,'syntax_only':True},
      'input_stream_hashes_verified':True,'launch_authorized':False}
    (attempt/'materialization.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    os.replace(attempt,args.output)
    print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
