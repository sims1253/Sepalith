import pathlib,subprocess,os,json,time,datetime,hashlib
W=pathlib.Path(__file__).resolve().parent
S=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/materialize_candidates.py')
P=pathlib.Path('/home/m0hawk/Documents/Sepalith/.venv/bin/python')
result=[]
for kind in ['completion','structured']:
 t=time.monotonic()
 with (W/(kind+'.log')).open('xb') as log:
  r=subprocess.run(['timeout','--signal=TERM','--kill-after=15s','1300s',str(P),'-B',str(S),kind,'--output',str(W/kind)],stdout=log,stderr=subprocess.STDOUT)
 item={'kind':kind,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':r.returncode,'seconds':time.monotonic()-t,'admitted':False}
 result.append(item);(W/(kind+'-terminal.json')).write_text(json.dumps(item,indent=2)+'\n')
 if r.returncode:break
(W/'terminal.json').write_text(json.dumps(result,indent=2)+'\n')
