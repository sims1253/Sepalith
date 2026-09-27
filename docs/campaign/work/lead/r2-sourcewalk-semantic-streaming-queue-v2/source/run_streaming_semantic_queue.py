#!/usr/bin/env python3
"""Analyze committed provenance shards with content-bound child outputs.

Each semantic child is bound to the complete source provenance ledger and to
the exact semantic-ledger bytes it produced.  Row-ID closure remains useful,
but it is never the only identity check used for reuse or publication.
"""
from __future__ import annotations
import argparse,collections,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,uuid
from pathlib import Path
from typing import Any,Iterable,Mapping
HERE=Path(__file__).resolve().parent
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1')
ANALYZER=HERE/'analyze_semantics.py';SCOPE=HERE/'semantic_scope.R';NAMESPACE=HERE/'namespace_scope.R'
DRIVER_SHA='63d165ae1488c1f38174f7e24f68a51e4c80c2d48e5aec44a0446fa062a61f7f'
GLOBAL=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json');GLOBAL_SHA='c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
CPT=PLAN/'docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json';CPT_SHA='6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06'
HOLD_SHA='6a626df3c55b6bc1907eb3b64f8e8fec50095f07db47db0d796f17e658dda751'
STRICT_SHA='7f042596a67ec9983f6f409a8913f4bc8df3361548a1e91fb77c9f914dafcf37'
LICENSE_SHA='e02d588ac77d3a8bf8710217c8d1c678e730681de998932219afe75a2709dd80'
QUEUED='provenance_pass_semantic_analyzer_queued'
QUEUE_SCHEMA='sepalith.dat10.sourcewalk_semantic_streaming_queue.v2'
class QueueError(RuntimeError):pass
def req(x,msg):
 if not x:raise QueueError(msg)
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_bytes(value:bytes)->str:return hashlib.sha256(value).hexdigest()
def canonical(value:Any)->str:return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest_ids(ids:list[str])->str:return sha_bytes(canonical(ids).encode('utf-8'))
def lines(path:Path):
 with path.open() as f:
  for n,line in enumerate(f,1):
   req(bool(line.strip()),f'empty JSON line:{path}:{n}');yield json.loads(line)
def lines_count(path:Path)->int:
 with path.open('rb') as f:return sum(1 for _ in f)
def atomic(path:Path,value:Any):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
 with tmp.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def committed(replay:Path)->list[int]:
 out=[]
 for p in sorted((replay/'shards').glob('shard-[0-9][0-9][0-9][0-9]/receipt.json')):
  try:s=int(p.parent.name.split('-')[1])
  except Exception:continue
  out.append(s)
 req(len(out)==len(set(out)),'duplicate committed shard directory');return out
def parse_shards(value:str,replay:Path)->list[int]:
 if value=='committed':out=committed(replay)
 else:
  try:out=[int(x) for x in value.split(',') if x!='']
  except ValueError as e:raise QueueError('shards must be integers or committed') from e
 req(out and out==sorted(set(out)) and all(x>=0 for x in out),'shards must be nonempty sorted unique nonnegative');return out
def index_map(replay:Path)->tuple[dict,dict[int,dict],str]:
 p=replay/'index/manifest.json';value=json.loads(p.read_text());digest=sha(p)
 req(value.get('schema')=='sepalith.dat10.sourcewalk_raw_index.v3' and value.get('status')=='complete','replay index is not committed v3')
 entries=value.get('index_files');req(isinstance(entries,list),'index entries missing')
 mapping={x.get('shard'):x for x in entries};req(len(mapping)==len(entries),'duplicate index shard')
 return value,mapping,digest
def validate_receipt(replay:Path,shard:int,index:dict,mapping:dict[int,dict],index_sha:str)->dict:
 req(shard in mapping,f'shard absent from committed index:{shard}')
 rp=replay/'shards'/f'shard-{shard:04d}/receipt.json';req(rp.is_file(),f'provenance receipt absent:{shard}')
 receipt=json.loads(rp.read_text());req(receipt.get('schema')=='sepalith.dat10.sourcewalk_provenance_shard.v3' and receipt.get('status')=='complete' and receipt.get('shard')==shard,f'provenance receipt incomplete:{shard}')
 binding=receipt.get('binding');req(isinstance(binding,dict),f'provenance binding absent:{shard}')
 entry=mapping[shard];index_path=Path(entry.get('path',''));req(index_path.is_file(),f'index shard absent:{shard}')
 req(index_path.stat().st_size==entry.get('bytes') and sha(index_path)==entry.get('sha256') and lines_count(index_path)==entry.get('rows'),f'index shard output changed:{shard}')
 token_pin=next((x for x in index['input_inventory']['shard_pins'] if x.get('shard')==shard),None);req(token_pin is not None,f'token pin absent:{shard}')
 packet_manifest=BASE/f'shard-{shard:04d}/structured-materialization-v1/manifest.json';pm=json.loads(packet_manifest.read_text());packet=Path(pm['outputs']['candidate_packets']['path']);packet_decl=pm['outputs']['candidate_packets']
 req(packet.is_file() and sha(packet)==packet_decl.get('sha256') and lines_count(packet)==packet_decl.get('rows'),f'candidate packet output changed:{shard}')
 required={'driver_sha256':DRIVER_SHA,'index_manifest_sha256':index_sha,'index_shard_sha256':entry['sha256'],'token_rows_sha256':token_pin['sha256'],'token_manifest_sha256':token_pin['manifest_sha256'],'candidate_packets_sha256':pm['outputs']['candidate_packets']['sha256'],'candidate_packet_manifest_sha256':sha(packet_manifest),'global_sha256':GLOBAL_SHA,'cpt_sha256':CPT_SHA,'hold_ledger_sha256':HOLD_SHA,'strict_validator_sha256':STRICT_SHA,'license_parser_sha256':LICENSE_SHA}
 req(binding==required,f'provenance binding changed:{shard}')
 outputs=receipt.get('outputs');req(isinstance(outputs,list) and len(outputs)==1,f'provenance output contract invalid:{shard}')
 out=outputs[0];ledger=Path(out.get('path',''));req(ledger.is_file() and ledger.stat().st_size==out.get('bytes') and sha(ledger)==out.get('sha256'),f'provenance ledger changed:{shard}')
 rows=list(lines(ledger));ids=[x.get('row_id') for x in rows];req(receipt.get('rows')==out.get('rows')==len(rows),f'provenance row count differs:{shard}');req(all(isinstance(x,str) and x for x in ids) and len(ids)==len(set(ids)),f'provenance row IDs invalid or duplicated:{shard}')
 req(all(x.get('shard')==shard for x in rows),f'provenance row shard differs:{shard}')
 queued=sorted(x['row_id'] for x in rows if x.get('family')=='roxygen_drafting' and x.get('status')==QUEUED)
 source={'shard':shard,
         'provenance_receipt':{'path':str(rp),'bytes':rp.stat().st_size,'sha256':sha(rp)},
         'provenance_ledger':{'path':str(ledger),'bytes':ledger.stat().st_size,'sha256':out['sha256'],'rows':len(rows),'row_ids_sha256':digest_ids(ids)},
         'queued_ids':queued,'queued_rows':len(queued),'queued_ids_sha256':digest_ids(queued),
         'candidate_packet':{'path':str(packet),'bytes':packet.stat().st_size,'sha256':required['candidate_packets_sha256'],'rows':packet_decl['rows']},
         'index_shard':{'path':str(index_path),'bytes':index_path.stat().st_size,'sha256':entry['sha256'],'rows':entry['rows']},
         'binding':required}
 source['binding_sha256']=sha_bytes(canonical(source).encode('utf-8'))
 source.update({'receipt_path':str(rp),'receipt_sha256':source['provenance_receipt']['sha256'],'receipt_rows':len(rows),'ledger_path':str(ledger),'ledger_sha256':out['sha256'],'packet_path':str(packet),'packet_sha256':required['candidate_packets_sha256']})
 return source
def semantic_output(path:Path,rows:list[dict])->dict:
 ids=[row.get('row_id') for row in rows]
 req(all(isinstance(value,str) and value for value in ids) and len(ids)==len(set(ids)),'semantic output IDs invalid or duplicated')
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'rows':len(rows),'row_ids_sha256':digest_ids(ids)}

def validate_analyzer_manifest(stage:Path,source:dict,code:dict)->tuple[dict,list[dict]]:
 mp=stage/'manifest.json';req(mp.is_file(),'semantic analyzer manifest missing')
 m=json.loads(mp.read_text());req(m.get('schema')=='sepalith.dat10.sourcewalk_roxy_semantic.v6','semantic analyzer schema differs');req(m.get('status')=='complete_review_only','semantic analyzer did not complete review-only')
 req(m.get('code')==code,'semantic analyzer code identity differs')
 req(m.get('rows')==source['queued_rows'] and m.get('exact_id_closure') is True,'semantic analyzer row closure differs')
 inputs=m.get('inputs');req(isinstance(inputs,dict),'semantic analyzer inputs missing')
 provenance=inputs.get('provenance_ledger');req(isinstance(provenance,dict) and provenance.get('path')==source['provenance_ledger']['path'] and provenance.get('sha256')==source['provenance_ledger']['sha256'],'semantic analyzer provenance binding differs')
 packets=inputs.get('candidate_packets');req(isinstance(packets,list) and len(packets)==1 and packets[0].get('path')==source['candidate_packet']['path'] and packets[0].get('sha256')==source['candidate_packet']['sha256'],'semantic analyzer candidate packet binding differs')
 out=m.get('output');ledger=stage/'semantic-ledger.jsonl';req(isinstance(out,dict) and out.get('path')==str(ledger) and out.get('rows')==source['queued_rows'],'semantic analyzer output declaration differs')
 req(ledger.is_file() and out.get('bytes')==ledger.stat().st_size and out.get('sha256')==sha(ledger),'semantic analyzer output bytes/hash differs')
 rows=list(lines(ledger));ids=[row.get('row_id') for row in rows]
 req(sorted(ids)==source['queued_ids'] and digest_ids(ids)==source['queued_ids_sha256'],'semantic analyzer output ID digest differs')
 return m,rows

def reusable(target:Path,source:dict,code:dict)->bool:
 mp=target/'manifest.json'
 if not mp.is_file():return False
 try:m=json.loads(mp.read_text())
 except Exception:return False
 try:
  req(m.get('schema')=='sepalith.dat10.sourcewalk_roxy_semantic.v6','semantic child schema differs')
  req(m.get('status')=='complete_review_only' and m.get('streaming_binding')==source and m.get('code')==code and m.get('exact_id_closure') is True,'semantic child identity differs')
  req(m.get('source_provenance_binding_sha256')==source['binding_sha256'] and m.get('queued_ids_sha256')==source['queued_ids_sha256'],'semantic child source digest differs')
  out=m.get('output');ledger=target/'semantic-ledger.jsonl'
  req(isinstance(out,dict) and out.get('path')==str(ledger) and out.get('bytes')==ledger.stat().st_size and out.get('sha256')==sha(ledger) and out.get('rows')==source['queued_rows'],'semantic child output bytes/hash differs')
  rows=list(lines(ledger));ids=[row.get('row_id') for row in rows]
  req(sorted(ids)==source['queued_ids'] and digest_ids(ids)==source['queued_ids_sha256'] and out.get('row_ids_sha256')==digest_ids(ids),'semantic child output ID digest differs')
  req(m.get('output_binding')==out,'semantic child output binding differs')
  return True
 except (QueueError,OSError,ValueError,KeyError,TypeError):
  return False

def analyze_one(output:Path,source:dict,code:dict)->dict:
 shard=source['shard'];target=output/f'shard-{shard:04d}'
 if reusable(target,source,code):
  m=json.loads((target/'manifest.json').read_text());out=m['output'];return {'shard':shard,'status':'reused','manifest_sha256':sha(target/'manifest.json'),'rows':m['rows'],'queued_ids':source['queued_ids'],'queued_ids_sha256':source['queued_ids_sha256'],'source_binding_sha256':source['binding_sha256'],'provenance_ledger_sha256':source['provenance_ledger']['sha256'],'semantic_output_sha256':out['sha256'],'semantic_output_bytes':out['bytes'],'semantic_output_rows':out['rows'],'semantic_output_row_ids_sha256':out['row_ids_sha256']}
 req(not target.exists(),f'nonreusable semantic target:{shard}')
 stage=output/f'.shard-{shard:04d}-{uuid.uuid4().hex}'
 try:
  done=subprocess.run([sys.executable,str(ANALYZER),'--provenance-ledger',source['provenance_ledger']['path'],'--candidate-packets',source['candidate_packet']['path'],'--expected-candidate-packets-sha256',source['candidate_packet']['sha256'],'--output',str(stage)],check=False)
  req(done.returncode==0,f'semantic analyzer failed:{shard}:{done.returncode}')
  m,rows=validate_analyzer_manifest(stage,source,code)
  m['source_provenance_binding_sha256']=source['binding_sha256'];m['queued_ids_sha256']=source['queued_ids_sha256']
  stage.replace(target)
  out=semantic_output(target/'semantic-ledger.jsonl',rows);m['output']=out;m['output_binding']=out;m['streaming_binding']=source
  atomic(target/'manifest.json',m);fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  return {'shard':shard,'status':'computed','manifest_sha256':sha(target/'manifest.json'),'rows':m['rows'],'queued_ids':source['queued_ids'],'queued_ids_sha256':source['queued_ids_sha256'],'source_binding_sha256':source['binding_sha256'],'provenance_ledger_sha256':source['provenance_ledger']['sha256'],'semantic_output_sha256':out['sha256'],'semantic_output_bytes':out['bytes'],'semantic_output_rows':out['rows'],'semantic_output_row_ids_sha256':out['row_ids_sha256']}
 except BaseException:
  if stage.exists():shutil.rmtree(stage)
  raise
def aggregate_status(results:list[dict],sources:list[dict],terminal:Path|None)->tuple[str,dict]:
 req(len(results)==len(sources),'semantic result/source cardinality differs')
 source_by_shard={source['shard']:source for source in sources};req(len(source_by_shard)==len(sources),'duplicate source shard')
 observed=[]
 for result in results:
  source=source_by_shard.get(result.get('shard'));req(source is not None,'semantic result shard missing from source')
  req(result.get('source_binding_sha256')==source['binding_sha256'] and result.get('provenance_ledger_sha256')==source['provenance_ledger']['sha256'],'semantic source ledger binding missing')
  req(result.get('queued_ids_sha256')==source['queued_ids_sha256'],'semantic queued ID digest differs')
  req(result.get('semantic_output_rows')==result.get('rows') and isinstance(result.get('semantic_output_sha256'),str) and result.get('semantic_output_sha256'),'semantic output hash binding missing')
  req(result.get('semantic_output_row_ids_sha256')==source['queued_ids_sha256'],'semantic output ID digest differs')
  observed.extend(result['queued_ids'])
 req(len(observed)==len(set(observed)),'duplicate semantic ID across shards')
 closure={'global_provenance_terminal':False,'all_terminal_shards_included':False,'global_queued_ids_closed':False}
 if terminal is None:return 'partial_review_only',closure
 req(terminal.is_file(),'declared global provenance terminal is absent');t=json.loads(terminal.read_text());req(t.get('status')=='complete_review_only_no_admission','global provenance manifest is not terminal')
 terminal_receipts=t.get('shard_receipts');req(isinstance(terminal_receipts,list),'terminal shard receipts missing')
 terminal_shards=[x.get('shard') for x in terminal_receipts];selected=[x['shard'] for x in sources];closure['global_provenance_terminal']=True;closure['all_terminal_shards_included']=selected==terminal_shards
 req(all(isinstance(x,int) for x in terminal_shards) and len(terminal_shards)==len(set(terminal_shards)),'terminal shard receipt set invalid')
 expected=[]
 for item in terminal_receipts:
  source=source_by_shard.get(item.get('shard'));req(source is not None,'terminal shard not selected')
  req(item.get('receipt_sha256')==source['receipt_sha256'],'terminal provenance receipt differs from validated source')
  req(item.get('rows')==source['receipt_rows'],'terminal provenance row count differs')
  rp=terminal.parent/'shards'/f"shard-{item['shard']:04d}"/'receipt.json';req(sha(rp)==item['receipt_sha256'],'terminal provenance receipt changed')
  receipt=json.loads(rp.read_text())
  for x in lines(Path(receipt['outputs'][0]['path'])):
   if x.get('family')=='roxygen_drafting' and x.get('status')==QUEUED:expected.append(x['row_id'])
 closure['global_queued_ids_closed']=len(observed)==len(set(observed)) and set(observed)==set(expected)
 if all(closure.values()):return 'complete_review_only',closure
 return 'partial_review_only',closure
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--replay-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--shards',required=True);ap.add_argument('--max-workers',type=int,default=1);ap.add_argument('--terminal-manifest',type=Path);a=ap.parse_args(argv)
 req(1<=a.max_workers<=2,'max-workers must be 1 or 2');req(sha(GLOBAL)==GLOBAL_SHA and sha(CPT)==CPT_SHA,'global/CPT registry bytes changed');a.output.mkdir(parents=True,exist_ok=True)
 index,mapping,index_sha=index_map(a.replay_root);chosen=parse_shards(a.shards,a.replay_root);sources=[validate_receipt(a.replay_root,s,index,mapping,index_sha) for s in chosen]
 code={'analyzer_sha256':sha(ANALYZER),'scope_helper_sha256':sha(SCOPE),'namespace_helper_sha256':sha(NAMESPACE)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.max_workers) as pool:results=list(pool.map(lambda x:analyze_one(a.output,x,code),sources))
 status,closure=aggregate_status(results,sources,a.terminal_manifest)
 manifest={'schema':QUEUE_SCHEMA,'status':status,'training_admission':False,'source_pool_closed':status=='complete_review_only','replay_root':str(a.replay_root),'replay_index_sha256':index_sha,'requested_shards':chosen,'max_workers':a.max_workers,'code':{**code,'queue_runner_sha256':sha(Path(__file__).resolve())},'provenance_rows':sum(x['receipt_rows'] for x in sources),'semantic_rows':sum(x['rows'] for x in results),'shards':[{k:v for k,v in x.items() if k!='queued_ids'} for x in results],'closure':closure,'binding_contract':{'source_provenance_ledger_hash_required':True,'semantic_output_hash_required':True,'semantic_output_id_digest_required':True,'same_ids_alone_sufficient':False}}
 atomic(a.output/'streaming-manifest.json',manifest);print(json.dumps({'status':status,'provenance_rows':manifest['provenance_rows'],'semantic_rows':manifest['semantic_rows']}));return 0
if __name__=='__main__':raise SystemExit(main())
