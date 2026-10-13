import json,os,subprocess,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKET=HERE.parent/'r2-full-weight-resume-proof-v3'
COMMANDS=json.loads((PACKET/'root-commands.json').read_text())
def write(value):
 value['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 tmp=HERE/'status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(HERE/'status.json')
def main():
 fd=int(os.environ['SEPALITH_CUDA_LOCK_FD']);os.fstat(fd)
 for lane in COMMANDS['execution_order']:
  write({'status':'running','lane':lane})
  result=subprocess.run(COMMANDS[lane],stdin=subprocess.DEVNULL,pass_fds=(fd,))
  if result.returncode:
   write({'status':'failed','lane':lane,'exit_code':result.returncode});return result.returncode
 write({'status':'completed','lanes':COMMANDS['execution_order']});return 0
if __name__=='__main__':raise SystemExit(main())
