#!/usr/bin/env python3
"""Bind the already-running root watchdog into the immutable submission file."""
import hashlib,json,os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREARM=HERE/'binding.prearm.json'
STATE=HERE/'watchdog-state-e48ab560a39b42edba21f0b6d2a6fea8'/'armed.json'
ARMED=HERE/'watchdog-armed.json'
FINAL=HERE/'binding.root.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)
def main():
 require(STATE.is_file() and not STATE.is_symlink(),'watchdog armed state absent')
 b=json.loads(PREARM.read_text());w=json.loads(STATE.read_text())
 require(w.get('name')=='sepalith-cpt-'+b['run_id'] and w.get('deadline')==b['absolute_deadline_utc'],'watchdog identity differs')
 require(type(w.get('pid')) is int and w['pid']>1,'watchdog PID invalid')
 require((Path('/proc')/str(w['pid'])).exists(),'watchdog is not live')
 require(not ARMED.exists() and not FINAL.exists(),'final binding already exists')
 ARMED.write_bytes(STATE.read_bytes());b['watchdog_armed_receipt_sha256']=sha(ARMED)
 FINAL.write_text(json.dumps(b,indent=2,sort_keys=True)+'\n')
 os.sync();print(json.dumps({'status':'final_binding_ready','binding':str(FINAL),'binding_sha256':sha(FINAL),'watchdog_receipt_sha256':sha(ARMED),'watchdog_pid':w['pid']}))
if __name__=='__main__':main()
