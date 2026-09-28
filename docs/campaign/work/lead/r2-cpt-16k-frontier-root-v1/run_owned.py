#!/usr/bin/env python3
import datetime as dt,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
def write(path,value):
 tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('xb') as f:f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 launch={'schema':'sepalith.cpt.16k-frontier-launch.v1','status':'started','started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'supervisor_pid':os.getpid(),'command':[str(HERE/'run_frontier.sh')]}
 write(HERE/'launch.json',launch);code=None
 try: code=subprocess.call([str(HERE/'run_frontier.sh')]);status='complete' if code==0 else 'failed'
 except BaseException as error:
  status='failed';launch['error_type']=type(error).__name__;launch['traceback']=traceback.format_exc(limit=4);code=1
 finally:
  launch.update({'status':status,'exit_code':code,'ended_at':dt.datetime.now(dt.timezone.utc).isoformat()});write(HERE/'terminal.json',launch)
 return code
if __name__=='__main__':sys.exit(main())
