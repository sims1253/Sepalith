"""Bounded CPU-only import/tokenizer observation, no panel/model/server access."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','LD_','GGML_'))}
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    start=time.monotonic();peak=0;failure=None
    checking=sys.argv[1:]==['--check-policy-only']
    combined=sys.argv[1:]==['--observe-and-check']
    if sys.argv[1:] and not checking and not combined:raise ValueError('unknown CPU observation mode')
    label='policy-check' if checking else 'client-observation'
    with (HERE/(label+'.log')).open('xb') as log:
        mode='--check-policy-only' if checking else '--observe-and-check' if combined else '--observe-only'
        child=subprocess.Popen(['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-I','-S','-B',str(HERE/'guarded_dev_client.py'),mode],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                try:rss=int(next(x for x in Path(f'/proc/{child.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                except (FileNotFoundError,StopIteration):rss=0
                peak=max(peak,rss)
                if rss>1536*1024**2:failure='1.5GiB RSS';break
                if time.monotonic()-start>90:failure='90 seconds';break
                time.sleep(.1)
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    result={'exit':child.returncode,'failure':failure,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':peak,'rss_ceiling_bytes':1536*1024**2,'mode':mode,'model_load':False,'final_access':False,'native_server_launch':False}
    (HERE/('policy-check-terminal.json' if checking else 'observation-terminal.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if failure or child.returncode:raise SystemExit(1)
if __name__=='__main__':main()
