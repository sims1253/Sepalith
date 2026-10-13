import subprocess,pathlib,json,time,signal,urllib.request,os,datetime
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
OUT=HERE/'live-a';OUT.mkdir()
server=None;editor=None;failure=None

def interrupted(*_):raise InterruptedError('controller_stopped')
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
start=time.monotonic()
try:
 server=subprocess.Popen(['python3','-B',str(HERE/'server.py')],stdout=(OUT/'server-supervisor.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(60):
  if server.poll() is not None:raise RuntimeError('server_supervisor_exited')
  try:
   with urllib.request.urlopen('http://127.0.0.1:18403/health',timeout=1) as r:
    if r.status==200:break
  except Exception:pass
  time.sleep(1)
 else:raise TimeoutError('server_readiness')
 cmd=['xvfb-run','-a','--server-args=-screen 0 1280x800x24','node',str(HERE/'run_b4_editor_cycle.mjs'),'--code','/usr/bin/code','--vsix',str(ROOT/'b4-native-v5-capsule/primary.vsix'),'--target-root',str(ROOT),'--port','18403','--run-root',str(HERE/'editor-live-a'),'--timeout-ms','120000','--mode','existing_external','--manual-trigger','0','--close','1']
 editor=subprocess.Popen(cmd,stdout=(OUT/'editor.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
 (OUT/'launch.json').write_text(json.dumps({'controller_pid':os.getpid(),'server_supervisor_pid':server.pid,'editor_pid':editor.pid,'editor_argv':cmd},indent=2))
 code=editor.wait(timeout=160)
 if code:raise RuntimeError('editor_exit:'+str(code))
except Exception as e:failure=type(e).__name__+': '+str(e)
finally:
 for child in [editor,server]:
  if child is None or child.poll() is not None:continue
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();failure=(failure or '')+';forced_cleanup'
 out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'failure':failure,'seconds':time.monotonic()-start,'server_supervisor_exit':None if server is None else server.returncode,'editor_exit':None if editor is None else editor.returncode,'acceptance':'root must inspect actual hostresult, commit/parse and processcleanup'}
 (OUT/'terminal.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
raise SystemExit(1 if failure else 0)
