#!/usr/bin/env python3
"""Copy the frozen CPU fixture to a fresh output directory and run one bounded replay."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,signal,subprocess,time
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output-root',required=True);args=ap.parse_args()
out=Path(args.output_root).expanduser().absolute()
if out.exists():raise SystemExit('output-root must not already exist')
manifest=json.loads((P/'runtime-manifest.json').read_text())
for name,expected in manifest['files'].items():
 q=P/name
 if hashlib.sha256(q.read_bytes()).hexdigest()!=expected:raise SystemExit('candidate source pin mismatch: '+name)
out.mkdir(parents=True)
for name in manifest['files']:shutil.copyfile(P/name,out/name)
started=time.monotonic();allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(allowed)})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',GOMAXPROCS='1',UV_THREADPOOL_SIZE='1')
argv=['node',str(out/'trace-cache-events.cjs')]
reason=None
with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
 child=subprocess.Popen(argv,cwd=out,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 try:code=child.wait(timeout=175)
 except subprocess.TimeoutExpired:
  reason='cpu_replay_deadline';os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=2)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=1)
result={'schema':1,'argv':argv,'code':code,'failure':reason,'seconds':time.monotonic()-started,'one_cpu_affinity':min(allowed),'wall_ceiling_seconds':180,'scope':'CPU mock editor only; no real editor, model, native server or network','trace_exists':(out/'cpu-trace.json').exists()}
(out/'runner-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
raise SystemExit(code if code else 1 if reason else 0)
