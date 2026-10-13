import pathlib,json,subprocess,os,time,urllib.request
w=pathlib.Path(__file__).resolve().parent
binary='/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server'
python='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
for arm in json.loads((w/'arms.json').read_text()):
 name=arm['arm'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',GGML_CUDA_GRAPH_OPT='0')
 argv=[binary,'-m',arm['path'],'--host','127.0.0.1','--port','18414','-c','4096','-b','256','-ub','256','--parallel','1','-t','6','-tb','6','--threads-http','2','-ngl','99','-lv','4']
 with (w/(name+'-server.log')).open('xb') as log:
  server=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
  (w/(name+'-launch.json')).write_text(json.dumps({'pid':server.pid,'argv':argv,'start_stat':pathlib.Path(f'/proc/{server.pid}/stat').read_text()})+'\n')
  try:
   deadline=time.monotonic()+90
   while True:
    if server.poll() is not None:raise RuntimeError('server exited during load')
    try:
     with urllib.request.urlopen('http://127.0.0.1:18414/health',timeout=2) as f:health=json.load(f)
     if health.get('status')=='ok':break
    except Exception:pass
    if time.monotonic()>deadline:raise RuntimeError('health deadline')
    time.sleep(1)
   with urllib.request.urlopen('http://127.0.0.1:18414/props',timeout=3) as f:(w/(name+'-props.json')).write_bytes(f.read())
   (w/(name+'-maps.txt')).write_text(pathlib.Path(f'/proc/{server.pid}/maps').read_text())
   subprocess.run([python,'-B',str(w/'client.py'),name],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),check=True,timeout=1000)
  finally:
   if server.poll() is None:
    server.terminate()
    try:server.wait(timeout=10)
    except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
   (w/(name+'-terminal.json')).write_text(json.dumps({'exit_code':server.returncode,'pid_absent':not pathlib.Path(f'/proc/{server.pid}').exists()})+'\n')
