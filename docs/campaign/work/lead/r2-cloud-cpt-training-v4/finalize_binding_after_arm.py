#!/usr/bin/env python3
"""Bind the already-running root watchdog into the immutable submission file."""
import argparse,hashlib,json,os
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)
def main():
 p=argparse.ArgumentParser();p.add_argument('--prearm',type=Path,required=True);p.add_argument('--armed-state',type=Path,required=True);p.add_argument('--armed-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 require(a.prearm.resolve()==HERE/'binding.prearm.json' and a.armed_receipt.resolve()==HERE/'watchdog-armed.json' and a.output.resolve()==HERE/'binding.root.json','exact launch binding paths required')
 require(a.armed_state.is_file() and not a.armed_state.is_symlink(),'watchdog armed state absent')
 b=json.loads(a.prearm.read_text());w=json.loads(a.armed_state.read_text())
 require(w.get('name')=='sepalith-cpt-'+b['run_id'] and w.get('deadline')==b['absolute_deadline_utc'],'watchdog identity differs')
 require(type(w.get('pid')) is int and w['pid']>1,'watchdog PID invalid')
 require((Path('/proc')/str(w['pid'])).exists(),'watchdog is not live')
 require(not a.armed_receipt.exists() and not a.output.exists(),'final binding already exists')
 a.armed_receipt.write_bytes(a.armed_state.read_bytes());b['watchdog_armed_receipt_sha256']=sha(a.armed_receipt)
 a.output.write_text(json.dumps(b,indent=2,sort_keys=True)+'\n')
 os.sync();print(json.dumps({'status':'final_binding_ready','binding':str(a.output),'binding_sha256':sha(a.output),'watchdog_receipt_sha256':sha(a.armed_receipt),'watchdog_pid':w['pid']}))
if __name__=='__main__':main()
