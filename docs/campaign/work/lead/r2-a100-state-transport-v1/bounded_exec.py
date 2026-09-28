#!/usr/bin/env python3
"""Run a transport command with owned process-group cleanup and redacted logs."""
import argparse,json,os,signal,shutil,subprocess,time
from pathlib import Path
from transport_common import atomic_json,safe_message
def run(argv,timeout,receipt,cleanup_parent=None,cleanup_prefix=None):
 started=time.monotonic();p=subprocess.Popen(argv,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:out,err=p.communicate(timeout=timeout);timed=False
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:out,err=p.communicate(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
 status='timeout' if timed else ('complete' if p.returncode==0 else 'failed');cleaned=[]
 if status!='complete' and cleanup_parent is not None:
  parent=Path(cleanup_parent).resolve();prefix=str(cleanup_prefix or '')
  if not prefix.startswith('.') or '/' in prefix or '\\' in prefix:raise ValueError('cleanup prefix must be a dot-prefixed basename')
  if parent.is_dir():
   for item in parent.iterdir():
    if item.name.startswith(prefix) and not item.is_symlink():
     if item.is_dir():shutil.rmtree(item)
     elif item.is_file():item.unlink()
     cleaned.append(item.name)
 result={'schema':'sepalith.pre04.bounded_transport_exec.v1','status':status,'pid':p.pid,'exit_code':p.returncode,'timeout_seconds':timeout,'elapsed_seconds':time.monotonic()-started,'stdout_tail':safe_message(out),'stderr_tail':safe_message(err),'owned_process_group_terminal':True,'cleaned_owned_temporary':sorted(cleaned),'credential_persisted':False};atomic_json(receipt,result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--timeout',type=int,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--cleanup-parent',type=Path);p.add_argument('--cleanup-prefix');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 if not cmd:raise SystemExit('command required')
 r=run(cmd,a.timeout,a.receipt,a.cleanup_parent,a.cleanup_prefix);print(json.dumps(r,sort_keys=True));raise SystemExit(0 if r['status']=='complete' else 1)
if __name__=='__main__':main()
