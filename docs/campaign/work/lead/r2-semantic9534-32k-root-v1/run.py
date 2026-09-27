import subprocess,json,datetime,os,pathlib
p=pathlib.Path(__file__).resolve().parent
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
(p/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),start=start)))
with (p/'run.log').open('w') as log:
 r=subprocess.run(['timeout','--signal=TERM','--kill-after=30','5400','bash',str(p/'run.sh'),'/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-32k-semantic9534-postrender-v1/inputs'],stdout=log,stderr=subprocess.STDOUT)
(p/'terminal.json').write_text(json.dumps(dict(start=start,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode)))
raise SystemExit(r.returncode)
