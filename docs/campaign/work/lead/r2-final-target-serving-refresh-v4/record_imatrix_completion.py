#!/usr/bin/env python3
"""Derive imatrix completion from the pinned log and GGUF artifact, never counters asserted by a launcher."""
import argparse,hashlib,json,re,sys
from pathlib import Path
from prepare_refresh_v4 import artifact,imatrix_argv,require,sha,validate_binding,write_new

def command_sha(argv):return hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()
def parse_log(path,expected_chunks,expected_output):
 text=Path(path).read_text(errors='replace');starts=[int(x) for x in re.findall(r'computing over (\d+) chunks',text)];ends=re.findall(r'stored collected data after (\d+) chunks in ([^\r\n]+)',text)
 require(starts and starts[-1]==expected_chunks,'imatrix log lacks exact compute chunk count');require(ends and int(ends[-1][0])==expected_chunks,'imatrix log lacks exact saved chunk count');require(Path(ends[-1][1].strip()).resolve()==Path(expected_output).resolve(),'imatrix log saved a different artifact');require('has no data - skipping' not in text and 'has partial data' not in text,'imatrix log reports incomplete tensor data')
 return {'compute_chunks':starts[-1],'saved_chunks':int(ends[-1][0]),'save_path':ends[-1][1].strip()}
def parse_artifact(b):
 sys.path.insert(0,str(Path(b['converter_source']['root'])/'gguf-py'));from gguf import GGUFReader
 im=GGUFReader(b['outputs']['imatrix']['path']);f16=GGUFReader(b['outputs']['f16']['path']);chunk=im.get_field('imatrix.chunk_count');size=im.get_field('imatrix.chunk_size');datasets=im.get_field('imatrix.datasets');require(chunk and size and datasets,'imatrix metadata missing');require(chunk.contents()==64 and size.contents()==2048,'imatrix geometry differs');require(str(Path(b['calibration']['text_path'])) in datasets.contents(),'imatrix dataset identity differs')
 names={t.name:t for t in im.tensors};sums={x[:-8] for x in names if x.endswith('.in_sum2')};counts={x[:-7] for x in names if x.endswith('.counts')};require(sums==counts and sums,'imatrix sum/count pairs differ')
 expected={t.name for t in f16.tensors if len(t.shape)>=2 and t.name!='token_embd.weight'};missing=sorted(expected-sums);partial=[]
 for name in sorted(expected&sums):
  values=names[name+'.counts'].data
  if values.size==0 or not bool((values>0).all()):partial.append(name)
 return {'chunk_count':chunk.contents(),'chunk_size':size.contents(),'datasets':datasets.contents(),'entry_count':len(sums),'expected_non_embedding_matrices':len(expected),'missing_non_embedding_matrices':missing,'partial_non_embedding_matrices':partial,'non_embedding_coverage_mismatches':len(missing)+len(partial)}
def record(binding_path,execution_path,out):
 bp=Path(binding_path);b=json.loads(bp.read_text());validate_binding(b,True,True);xp=Path(execution_path);x=json.loads(xp.read_text());expected=imatrix_argv(b)
 require(x.get('schema')=='sepalith.run06.root-command-terminal.v2' and x.get('status')=='completed','imatrix terminal envelope differs');require(x.get('argv')==expected and x.get('command_sha256')==command_sha(expected),'imatrix command identity differs');require(x.get('exit_code')==0,'imatrix exit was not zero')
 log=Path(x.get('log_path',''));require(log.is_file() and not log.is_symlink() and x.get('log_sha256')==sha(log),'imatrix log identity differs');log_evidence=parse_log(log,b['calibration']['chunks'],b['outputs']['imatrix']['path']);im=artifact(b['outputs']['imatrix'],'imatrix');artifact_evidence=parse_artifact(b);require(artifact_evidence['non_embedding_coverage_mismatches']==0,'imatrix artifact non-embedding coverage differs')
 value={'schema':'sepalith.run06.imatrix-completion.v2','status':'pass','binding_sha256':sha(bp),'execution_path':str(xp.resolve()),'execution_sha256':sha(xp),'log_path':str(log.resolve()),'log_sha256':sha(log),'exit_code':0,'processed_chunks':artifact_evidence['chunk_count'],'non_embedding_coverage_mismatches':0,'log_evidence':log_evidence,'artifact_evidence':artifact_evidence,'artifact_sha256':sha(im),'f16_sha256':b['outputs']['f16']['sha256'],'calibration_sha256':b['calibration']['text_sha256'],'tool_sha256':b['tools']['imatrix']['sha256']};write_new(out,value);return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--execution',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(record(a.binding,a.execution,a.out),sort_keys=True))
