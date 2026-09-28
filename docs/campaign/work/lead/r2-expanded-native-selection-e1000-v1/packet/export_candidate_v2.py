"""Root-only sequential F16/Q8 conversion under an external host/process guard."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from selection_contract import check_task_recipe,export_commands,TOK,TOKCFG

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=json.loads(a.spec.read_text())
 assert spec['status']=='root_admitted' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 host=spec['host_guard']
 cmdline=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
 assert os.fsencode(host['path']) in cmdline and sha(Path(host['path']))==host['sha256'],'direct hash-pinned host guard required'
 guard=json.loads(Path(spec['host_guard_preflight']).read_text())
 assert guard['reason'] is None and guard['initial']['AvailableMBytes']>=host['admission_free_mib']
 assert guard['memory_policy']['admission_MiB']==host['admission_free_mib'] and guard['memory_policy']['soft_MiB']==host['operating_free_mib'] and guard['memory_policy']['hard_MiB']==4096
 parent=Path(spec['parent_manifest']['path']);assert sha(parent)==spec['parent_manifest']['sha256'];m=json.loads(parent.read_text())
 assert m['schema_version']=='sepalith.r2-task-sft.parent-manifest.v1' and m['kind']=='merged_task_sft'
 check_task_recipe(m['sft_identity'],m['recipe_sha256'])
 assert m['sft_checkpoint_manifest']['identity']==m['sft_identity'] and m['sft_checkpoint_manifest']['full'] is True
 assert m['sft_checkpoint_manifest']['step']==1000 and m['source_cursor']==16000
 base=Path(m['merged_model_path']);out=Path(spec['output']);assert out.is_absolute() and not out.exists() and out!=base
 expected={'model.safetensors':m['merged_weights_sha256'],'config.json':m['config_sha256'],'generation_config.json':m['generation_config_sha256'],'tokenizer.json':TOK,'tokenizer_config.json':TOKCFG}
 for name,digest in expected.items():assert sha(base/name)==digest,name
 for r in spec['converter_source_files']:assert sha(Path(r['path']))==r['sha256']
 for r in spec['quantizer_files']:assert sha(Path(r['path']))==r['sha256']
 commands=export_commands(str(base),str(out));assert commands==spec['commands'],'command identity differs'
 out.mkdir();started=time.monotonic();result=[];error=None
 try:
  for index,argv in enumerate(commands):
   with (out/f'command-{index}.log').open('xb') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
    write(out/f'command-{index}-launch.json',{'pid':child.pid,'argv':argv})
    try:code=child.wait(timeout=240)
    except BaseException:
     if child.poll() is None:
      child.terminate()
      try:child.wait(timeout=10)
      except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
     raise
   assert code==0,'conversion failed';result.append({'command':index,'exit_code':code})
  files=[]
  for name in ('model-F16.gguf','model-Q8_0.gguf'):
   q=out/name
   with q.open('rb') as f:assert f.read(4)==b'GGUF';os.fsync(f.fileno())
   files.append({'path':str(q),'bytes':q.stat().st_size,'sha256':sha(q)})
  write(out/'artifacts.json',{'status':'exported_pending_independent_integrity','parent_manifest_sha256':spec['parent_manifest']['sha256'],'files':files})
 except BaseException as e:error={'type':type(e).__name__,'message':str(e)};raise
 finally:write(out/'terminal.json',{'seconds':time.monotonic()-started,'commands':result,'error':error,'accepted':False})
if __name__=='__main__':main()
