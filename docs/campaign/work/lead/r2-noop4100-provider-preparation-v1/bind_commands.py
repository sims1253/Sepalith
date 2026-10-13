#!/usr/bin/env python3
"""Emit exact two-lane commands for one immutable provider plan."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());digest=sha(a.plan);req(plan['schema']=='sepalith.dat10.noop4100.provider-run-plan.v1' and plan['status']=='prepared_no_launch' and plan['phase']in ('render16','render32'),'plan differs');req(plan['provider_source_manifest_sha256']=='6e6dfd87116bb8743fa1d593c857c7e2108a4d5e87f3a0814ff36e998b3cd424' and plan['run_shard_sha256']=='9fd8d8ef6730871ffa1a9703e3af4b92dd4e265c07976fa547803e23e6c3e3fb','provider binding differs')
 python='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python';value={'schema':'sepalith.dat10.noop4100.provider-phase-commands.v1','status':'prepared_no_launch','phase':plan['phase'],'plan':str(a.plan.resolve()),'plan_sha256':digest,'output_root':str(a.output_root.resolve()),'lanes':[{'lane':lane,'core':(4,6)[lane],'rows':plan['lane_rows'][lane],'command':[python,'-B',str(HERE/'run_lane.py'),'--plan',str(a.plan.resolve()),'--plan-sha256',digest,'--lane',str(lane),'--output',str(a.output_root.resolve())]}for lane in (0,1)],'execution_authorized':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+a.output.name+'.',dir=a.output.parent)
 try:
  with os.fdopen(fd,'w')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,a.output)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 print(json.dumps(value,sort_keys=True))
if __name__=='__main__':main()
